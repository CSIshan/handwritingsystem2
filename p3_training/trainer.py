"""
P3 Trainer
===========
Training loop with:
- MPS/CPU device handling
- Gradient clipping
- Teacher forcing decay
- Cosine LR annealing
- Checkpoint every N epochs
- Early stopping
- Emergency save on Ctrl+C / SIGTERM
- Live training log
"""
import json
import math
import signal
import sys
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.optim as optim

from .config  import TrainConfig
from .model   import HandwritingModel
from .dataset import make_dataloaders
from .prep    import build_vocab

log = logging.getLogger(__name__)


class Trainer:

    def __init__(self, cfg: TrainConfig):
        self.cfg   = cfg
        self.device= cfg.resolve_device()
        log.info(f"Device: {self.device}")

        # Load vocab
        vocab_path = Path(cfg.training_dir) / "vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"vocab.json not found in {cfg.training_dir}. "
                f"Run prep.prepare_dataset() first."
            )
        self.vocab      = json.loads(vocab_path.read_text())
        self.vocab_size = len(self.vocab)

        # Model
        self.model = HandwritingModel(cfg, self.vocab_size).to(self.device)
        log.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Optimizer + scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.epochs, eta_min=1e-5
        )

        # Data
        self.train_loader, self.val_loader = make_dataloaders(
            cfg.training_dir, cfg.batch_size, self.vocab,
            cfg.max_seq_len, cfg.seed,
        )

        # State
        self.epoch       = 0
        self.best_val    = float("inf")
        self.no_improve  = 0
        self.train_log   = []

        # Model dir
        self.model_dir = Path(cfg.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Signal handlers — save on interrupt
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, sig, frame):
        log.warning(f"\nSignal {sig} received — saving emergency checkpoint...")
        self._save_checkpoint("emergency")
        self._save_log()
        sys.exit(0)

    def _teacher_forcing_ratio(self) -> float:
        """Linearly decay from start to end over all epochs."""
        cfg = self.cfg
        progress = self.epoch / max(cfg.epochs - 1, 1)
        return cfg.teacher_forcing_start + progress * (
            cfg.teacher_forcing_end - cfg.teacher_forcing_start
        )

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = total_mdn = total_pen = total_stop = 0.0
        n_batches  = 0
        tf_ratio   = self._teacher_forcing_ratio()

        for batch in self.train_loader:
            char_ids    = batch["char_ids"].to(self.device)
            strokes     = batch["strokes"].to(self.device)
            char_lens   = batch["char_lens"].to(self.device)
            stroke_lens = batch["stroke_lens"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(
                char_ids, char_lens, strokes, stroke_lens, tf_ratio
            )
            losses = self.model.compute_loss(outputs, strokes, stroke_lens)

            losses["total"].backward()

            # Gradient clipping — non-negotiable for LSTM
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip
            )

            self.optimizer.step()

            total_loss += losses["total"].item()
            total_mdn  += losses["mdn"].item()
            total_pen  += losses["pen"].item()
            total_stop += losses["stop"].item()
            n_batches  += 1

        n = max(n_batches, 1)
        return {
            "loss": total_loss / n,
            "mdn":  total_mdn  / n,
            "pen":  total_pen  / n,
            "stop": total_stop / n,
            "tf_ratio": tf_ratio,
        }

    @torch.no_grad()
    def _val_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = total_pen_correct = total_pen_points = 0.0
        n_batches = 0

        for batch in self.val_loader:
            char_ids    = batch["char_ids"].to(self.device)
            strokes     = batch["strokes"].to(self.device)
            char_lens   = batch["char_lens"].to(self.device)
            stroke_lens = batch["stroke_lens"].to(self.device)

            outputs = self.model(char_ids, char_lens, strokes, stroke_lens,
                                 teacher_forcing_ratio=0.0)
            losses  = self.model.compute_loss(outputs, strokes, stroke_lens)

            # Pen accuracy
            mask       = (torch.arange(strokes.size(1), device=self.device)
                          .unsqueeze(0) < stroke_lens.unsqueeze(1))
            pen_pred   = (outputs["pen_up"] > 0.5).float()
            pen_target = strokes[:, :, 2]
            correct    = ((pen_pred == pen_target) * mask).sum().item()
            total_pen_correct += correct
            total_pen_points  += mask.sum().item()

            total_loss += losses["total"].item()
            n_batches  += 1

        n = max(n_batches, 1)
        return {
            "loss":         total_loss / n,
            "pen_accuracy": total_pen_correct / max(total_pen_points, 1),
        }

    def _save_checkpoint(self, tag: str):
        path = self.model_dir / f"checkpoint_{tag}.pt"
        torch.save({
            "epoch":           self.epoch,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val":        self.best_val,
            "config":          self.cfg.__dict__,
            "vocab_size":      self.vocab_size,
        }, path)
        log.info(f"Checkpoint saved: {path}")

    def _save_log(self):
        log_path = self.model_dir / "training_log.json"
        log_path.write_text(json.dumps(self.train_log, indent=2))

    def train(self):
        log.info(
            f"Starting training: {self.cfg.epochs} epochs, "
            f"batch={self.cfg.batch_size}, device={self.device}"
        )
        log.info(
            f"Train: {len(self.train_loader.dataset)} words  "
            f"Val: {len(self.val_loader.dataset)} words"
        )

        for epoch in range(1, self.cfg.epochs + 1):
            self.epoch = epoch

            train_metrics = self._train_epoch()
            val_metrics   = self._val_epoch()
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]

            log_entry = {
                "epoch":        epoch,
                "train_loss":   round(train_metrics["loss"], 4),
                "train_mdn":    round(train_metrics["mdn"],  4),
                "train_pen":    round(train_metrics["pen"],  4),
                "val_loss":     round(val_metrics["loss"],   4),
                "val_pen_acc":  round(val_metrics["pen_accuracy"], 4),
                "tf_ratio":     round(train_metrics["tf_ratio"], 3),
                "lr":           round(lr, 6),
            }
            self.train_log.append(log_entry)
            self._save_log()

            log.info(
                f"Epoch {epoch:3d}/{self.cfg.epochs}  "
                f"train={train_metrics['loss']:.4f}  "
                f"val={val_metrics['loss']:.4f}  "
                f"pen_acc={val_metrics['pen_accuracy']:.3f}  "
                f"tf={train_metrics['tf_ratio']:.2f}  "
                f"lr={lr:.2e}"
            )

            # ── Best model ────────────────────────────────────────────────────
            if val_metrics["loss"] < self.best_val:
                self.best_val   = val_metrics["loss"]
                self.no_improve = 0
                self._save_checkpoint("best")
                log.info(f"  → New best val_loss={self.best_val:.4f}")
            else:
                self.no_improve += 1

            # ── Periodic checkpoint ───────────────────────────────────────────
            if epoch % self.cfg.checkpoint_every == 0:
                self._save_checkpoint(f"epoch_{epoch:04d}")

            # ── Early stopping ────────────────────────────────────────────────
            if self.no_improve >= self.cfg.early_stop_patience:
                log.info(
                    f"Early stopping: no improvement for "
                    f"{self.cfg.early_stop_patience} epochs"
                )
                break

        # Save final model
        self._save_checkpoint("final")

        # Save config used
        config_path = self.model_dir / "train_config.json"
        config_path.write_text(json.dumps(self.cfg.__dict__, indent=2))

        log.info(
            f"Training complete. Best val_loss={self.best_val:.4f} "
            f"at checkpoint_best.pt"
        )
        log.info(f"Load best model with: torch.load('{self.model_dir}/checkpoint_best.pt')")


def run_training(cfg: TrainConfig = None):
    """
    Entry point. Run from terminal:
        python -m handwriting_system.p3_training
    Or:
        from handwriting_system.p3_training import run_training
        run_training()
    """
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s %(levelname)s %(message)s",
        datefmt = "%H:%M:%S",
    )

    cfg = cfg or TrainConfig()
    trainer = Trainer(cfg)
    trainer.train()

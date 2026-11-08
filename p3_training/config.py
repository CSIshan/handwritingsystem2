"""
P3 Training Configuration
==========================
All hyperparameters in one place.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    # ── Model architecture ────────────────────────────────────────────────────
    embed_dim:      int   = 64      # character embedding dimension
    encoder_hidden: int   = 256     # encoder LSTM hidden size (each direction)
    decoder_hidden: int   = 512     # decoder LSTM hidden size
    encoder_layers: int   = 2
    decoder_layers: int   = 2
    mdn_components: int   = 20      # Gaussian mixture components
    dropout:        float = 0.1

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size:     int   = 32
    epochs:         int   = 100
    learning_rate:  float = 1e-3
    grad_clip:      float = 1.0     # CRITICAL — prevents exploding gradients
    early_stop_patience: int = 15   # epochs with no val improvement → stop

    # Teacher forcing: starts at 1.0 (always use ground truth),
    # linearly decays to 0.5 by end of training
    teacher_forcing_start: float = 1.0
    teacher_forcing_end:   float = 0.5

    # ── Data ─────────────────────────────────────────────────────────────────
    train_split:    float = 0.9
    seed:           int   = 42
    max_seq_len:    int   = 500     # max stroke points per word, longer truncated

    # ── Paths ─────────────────────────────────────────────────────────────────
    verified_dir:   str   = "dataset/verified"
    training_dir:   str   = "dataset/training"
    model_dir:      str   = "dataset/model"
    checkpoint_every: int = 5       # save checkpoint every N epochs

    # ── Device ───────────────────────────────────────────────────────────────
    device:         str   = "auto"  # auto: MPS → CPU

    @property
    def encoder_output_dim(self) -> int:
        # Bidirectional: concat fwd + bwd hidden states
        return self.encoder_hidden * 2

    def resolve_device(self):
        """Return torch.device, preferring MPS on M2."""
        import torch
        if self.device == "auto":
            if torch.backends.mps.is_available():
                try:
                    torch.zeros(1).to("mps")
                    return torch.device("mps")
                except Exception:
                    pass
            return torch.device("cpu")
        return torch.device(self.device)

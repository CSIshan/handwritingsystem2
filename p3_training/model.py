"""
P3 Model — Seq2Seq LSTM + MDN
==============================
Encoder: Bidirectional LSTM over character embeddings
Decoder: LSTM unrolled over stroke sequence
Output:  MDN parameters + pen_up + stop probabilities
"""
import torch
import torch.nn as nn
from .config import TrainConfig
from .mdn import mdn_loss, mdn_sample


class Encoder(nn.Module):
    """
    Encodes a character sequence into a context vector.
    Bidirectional so each char sees left + right context.
    """

    def __init__(self, cfg: TrainConfig, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, cfg.embed_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size   = cfg.embed_dim,
            hidden_size  = cfg.encoder_hidden,
            num_layers   = cfg.encoder_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = cfg.dropout if cfg.encoder_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        char_ids: torch.Tensor,   # [B, char_len]
        char_lens: torch.Tensor,  # [B]
    ) -> tuple[torch.Tensor, tuple]:
        """
        Returns:
            outputs:  [B, char_len, encoder_hidden*2]  all hidden states
            (h_n, c_n): final hidden/cell for initialising decoder
        """
        embedded = self.dropout(self.embedding(char_ids))  # [B, L, embed_dim]

        # Pack for efficiency with variable-length sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, char_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs_packed, (h_n, c_n) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs_packed, batch_first=True
        )

        return outputs, (h_n, c_n)


class Decoder(nn.Module):
    """
    Generates stroke sequence conditioned on encoder context.
    Input at each step: previous stroke point [Δx, Δy, pen_up] + context.
    """

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg

        # Project encoder output to decoder hidden size for init
        self.hidden_proj = nn.Linear(cfg.encoder_output_dim, cfg.decoder_hidden)
        self.cell_proj   = nn.Linear(cfg.encoder_output_dim, cfg.decoder_hidden)

        # Input: stroke point (3) + encoder context (encoder_output_dim)
        self.lstm = nn.LSTM(
            input_size  = 3 + cfg.encoder_output_dim,
            hidden_size = cfg.decoder_hidden,
            num_layers  = cfg.decoder_layers,
            batch_first = True,
            dropout     = cfg.dropout if cfg.decoder_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(cfg.dropout)

        # MDN output: K components × (μx, μy, σx, σy, ρ) + π weights
        mdn_out_dim = cfg.mdn_components * 6  # 5 params + 1 for π
        self.mdn_head   = nn.Linear(cfg.decoder_hidden, mdn_out_dim)

        # Pen up and stop are binary
        self.pen_head   = nn.Linear(cfg.decoder_hidden, 1)
        self.stop_head  = nn.Linear(cfg.decoder_hidden, 1)

    def _init_hidden(
        self,
        encoder_outputs: torch.Tensor,  # [B, char_len, enc_dim]
        char_lens:       torch.Tensor,  # [B]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initialise decoder hidden state from encoder final states.
        Uses mean pooling over valid encoder positions.
        """
        B = encoder_outputs.size(0)

        # Mean pool over valid positions
        mask = torch.arange(encoder_outputs.size(1), device=encoder_outputs.device)
        mask = mask.unsqueeze(0) < char_lens.unsqueeze(1)   # [B, L]
        mask = mask.unsqueeze(-1).float()

        context = (encoder_outputs * mask).sum(1) / mask.sum(1).clamp(min=1)
        # context: [B, enc_dim]

        h0 = torch.tanh(self.hidden_proj(context))   # [B, decoder_hidden]
        c0 = torch.tanh(self.cell_proj(context))

        # Expand to num_layers
        h0 = h0.unsqueeze(0).expand(self.cfg.decoder_layers, -1, -1).contiguous()
        c0 = c0.unsqueeze(0).expand(self.cfg.decoder_layers, -1, -1).contiguous()

        return h0, c0

    def _parse_mdn(self, mdn_raw: torch.Tensor) -> tuple:
        """
        Split linear layer output into MDN parameters.
        mdn_raw: [B, T, K*6] or [B, K*6]
        """
        K = self.cfg.mdn_components
        # Split into K groups of 6
        chunks = mdn_raw.split(1, dim=-1)

        # Interleaved: [π0, μx0, μy0, σx0, σy0, ρ0, π1, ...]
        idx = 0
        pis, mu_xs, mu_ys, sig_xs, sig_ys, rhos = [], [], [], [], [], []
        for k in range(K):
            pis.append(mdn_raw[..., idx+0:idx+1])
            mu_xs.append(mdn_raw[..., idx+1:idx+2])
            mu_ys.append(mdn_raw[..., idx+2:idx+3])
            sig_xs.append(mdn_raw[..., idx+3:idx+4])
            sig_ys.append(mdn_raw[..., idx+4:idx+5])
            rhos.append(mdn_raw[..., idx+5:idx+6])
            idx += 6

        pi      = torch.softmax(torch.cat(pis,    dim=-1), dim=-1)
        mu_x    = torch.cat(mu_xs,  dim=-1)
        mu_y    = torch.cat(mu_ys,  dim=-1)
        sigma_x = torch.exp(torch.cat(sig_xs, dim=-1)).clamp(min=1e-4, max=10.0)
        sigma_y = torch.exp(torch.cat(sig_ys, dim=-1)).clamp(min=1e-4, max=10.0)
        rho     = torch.tanh(torch.cat(rhos,   dim=-1))

        return pi, mu_x, mu_y, sigma_x, sigma_y, rho

    def forward(
        self,
        encoder_outputs: torch.Tensor,  # [B, char_len, enc_dim]
        char_lens:       torch.Tensor,  # [B]
        target_strokes:  torch.Tensor,  # [B, T, 3]  teacher forcing targets
        stroke_lens:     torch.Tensor,  # [B]
        teacher_forcing_ratio: float = 1.0,
    ) -> dict:
        """
        Training forward pass with teacher forcing.
        """
        B, T, _ = target_strokes.shape
        device   = encoder_outputs.device

        # Init hidden state from encoder
        h, c = self._init_hidden(encoder_outputs, char_lens)

        # Context: mean pool encoder outputs (same as used for init)
        mask = torch.arange(encoder_outputs.size(1), device=device)
        mask = (mask.unsqueeze(0) < char_lens.unsqueeze(1)).unsqueeze(-1).float()
        context = (encoder_outputs * mask).sum(1) / mask.sum(1).clamp(min=1)
        # Expand context across all timesteps: [B, T, enc_dim]
        context_expanded = context.unsqueeze(1).expand(-1, T, -1)

        # Start token: zero stroke point
        prev_stroke = torch.zeros(B, 1, 3, device=device)

        all_pi, all_mux, all_muy = [], [], []
        all_sigx, all_sigy, all_rho = [], [], []
        all_pen, all_stop = [], []

        import random as _random

        for t in range(T):
            # Decide: use ground truth or previous prediction
            if _random.random() < teacher_forcing_ratio and t > 0:
                inp_stroke = target_strokes[:, t-1:t, :]
            else:
                inp_stroke = prev_stroke

            # Concat stroke + context
            lstm_input = torch.cat([inp_stroke, context_expanded[:, t:t+1, :]], dim=-1)

            out, (h, c) = self.lstm(lstm_input, (h, c))
            out = self.dropout(out)   # [B, 1, decoder_hidden]

            mdn_raw = self.mdn_head(out)  # [B, 1, K*6]
            pi, mu_x, mu_y, sx, sy, rho = self._parse_mdn(mdn_raw.squeeze(1))

            pen  = torch.sigmoid(self.pen_head(out.squeeze(1)))   # [B, 1]
            stop = torch.sigmoid(self.stop_head(out.squeeze(1)))  # [B, 1]

            all_pi.append(pi.unsqueeze(1))
            all_mux.append(mu_x.unsqueeze(1))
            all_muy.append(mu_y.unsqueeze(1))
            all_sigx.append(sx.unsqueeze(1))
            all_sigy.append(sy.unsqueeze(1))
            all_rho.append(rho.unsqueeze(1))
            all_pen.append(pen.unsqueeze(1))
            all_stop.append(stop.unsqueeze(1))

            # For next step (if not teacher forcing)
            # Use pen prediction as pen_up, Δx/Δy from MDN mean of top component
            top_k   = pi.argmax(dim=-1)  # [B]
            pred_dx = mu_x[torch.arange(B), top_k]
            pred_dy = mu_y[torch.arange(B), top_k]
            pred_pu = (pen.squeeze(-1) > 0.5).float()
            prev_stroke = torch.stack([pred_dx, pred_dy, pred_pu], dim=-1).unsqueeze(1)

        return {
            "pi":      torch.cat(all_pi,   dim=1),  # [B, T, K]
            "mu_x":    torch.cat(all_mux,  dim=1),
            "mu_y":    torch.cat(all_muy,  dim=1),
            "sigma_x": torch.cat(all_sigx, dim=1),
            "sigma_y": torch.cat(all_sigy, dim=1),
            "rho":     torch.cat(all_rho,  dim=1),
            "pen_up":  torch.cat(all_pen,  dim=1).squeeze(-1),  # [B, T]
            "stop":    torch.cat(all_stop, dim=1).squeeze(-1),  # [B, T]
        }


class HandwritingModel(nn.Module):
    """Full model: Encoder + Decoder."""

    def __init__(self, cfg: TrainConfig, vocab_size: int):
        super().__init__()
        self.cfg     = cfg
        self.encoder = Encoder(cfg, vocab_size)
        self.decoder = Decoder(cfg)

    def forward(
        self,
        char_ids:    torch.Tensor,
        char_lens:   torch.Tensor,
        strokes:     torch.Tensor,
        stroke_lens: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> dict:
        enc_out, _ = self.encoder(char_ids, char_lens)
        return self.decoder(
            enc_out, char_lens, strokes, stroke_lens, teacher_forcing_ratio
        )

    def compute_loss(
        self,
        outputs:     dict,
        strokes:     torch.Tensor,   # [B, T, 3]
        stroke_lens: torch.Tensor,   # [B]
    ) -> dict:
        """Compute total loss and individual components."""
        B, T, _ = strokes.shape
        device   = strokes.device

        # Build validity mask [B, T]: 1 where position is real, 0 = padding
        mask = torch.arange(T, device=device).unsqueeze(0) < stroke_lens.unsqueeze(1)
        mask = mask.float()

        target_dx = strokes[:, :, 0]   # [B, T]
        target_dy = strokes[:, :, 1]
        # pen_up ground truth must be strictly 0 or 1 for BCE.
        # Real StrokePoint data is always clean 0/1, but we clamp
        # defensively in case of upstream noise (e.g. synthetic/test data).
        target_pu = strokes[:, :, 2].clamp(0.0, 1.0)

        # MDN loss on stroke coordinates
        loss_mdn = mdn_loss(
            outputs["pi"],
            outputs["mu_x"],
            outputs["mu_y"],
            outputs["sigma_x"],
            outputs["sigma_y"],
            outputs["rho"],
            target_dx, target_dy, mask,
        )

        # Pen up loss
        # Weight positives more: pen_up is rare, ~10-20% of points
        pos_weight = torch.tensor([4.0], device=device)
        loss_pen = nn.functional.binary_cross_entropy(
            outputs["pen_up"] * mask,
            target_pu * mask,
            reduction="sum",
        ) / mask.sum().clamp(min=1)

        # Stop loss: 1 only at last valid position of each sequence
        stop_target = torch.zeros(B, T, device=device)
        for b in range(B):
            last = stroke_lens[b].item() - 1
            if 0 <= last < T:
                stop_target[b, int(last)] = 1.0

        loss_stop = nn.functional.binary_cross_entropy(
            outputs["stop"] * mask,
            stop_target * mask,
            reduction="sum",
        ) / mask.sum().clamp(min=1)

        total = loss_mdn + loss_pen + loss_stop

        return {
            "total": total,
            "mdn":   loss_mdn,
            "pen":   loss_pen,
            "stop":  loss_stop,
        }

    @torch.no_grad()
    def generate(
        self,
        word:        str,
        vocab:       dict,
        temperature: float = 1.0,
        max_steps:   int   = 500,
        device:      torch.device = None,
    ) -> list:
        """
        Inference: generate stroke sequence for a word.
        Returns list of [Δx, Δy, pen_up] in training scale.
        Call / 10.0 on Δx, Δy to convert back to mm.
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        unk_id = vocab.get("<UNK>", 1)
        char_ids = torch.tensor(
            [[vocab.get(c, unk_id) for c in word]],
            dtype=torch.long, device=device
        )
        char_lens = torch.tensor([len(word)], device=device)

        enc_out, _ = self.encoder(char_ids, char_lens)

        # Init decoder
        h, c = self.decoder._init_hidden(enc_out, char_lens)

        # Context
        context = enc_out.mean(dim=1, keepdim=True)  # [1, 1, enc_dim]

        prev_stroke = torch.zeros(1, 1, 3, device=device)
        generated   = []

        for step in range(max_steps):
            lstm_input = torch.cat([prev_stroke, context], dim=-1)
            out, (h, c) = self.decoder.lstm(lstm_input, (h, c))

            mdn_raw = self.decoder.mdn_head(out)
            pi, mu_x, mu_y, sx, sy, rho = self.decoder._parse_mdn(
                mdn_raw.squeeze(1).squeeze(0)
            )

            pen_prob  = torch.sigmoid(self.decoder.pen_head(out.squeeze())).item()
            stop_prob = torch.sigmoid(self.decoder.stop_head(out.squeeze())).item()

            # Sample from MDN
            dx, dy = mdn_sample(pi, mu_x, mu_y, sx, sy, rho, temperature)
            pen_up  = 1 if pen_prob > 0.5 else 0

            generated.append([dx, dy, pen_up])

            if stop_prob > 0.5 and step > 5:
                break

            prev_stroke = torch.tensor(
                [[[dx, dy, float(pen_up)]]], dtype=torch.float32, device=device
            )

        return generated

    def get_bbox(self, word: str, vocab: dict, device=None) -> tuple:
        """
        Generate strokes and return (width_mm, ascent_mm, descent_mm).
        Used by ModelBBoxProvider in P6 layout engine.
        """
        strokes = self.generate(word, vocab, temperature=0.8, device=device)
        if not strokes:
            return (10.0, 5.0, 1.0)

        # Convert from training scale back to mm
        xs = [s[0] / 10.0 for s in strokes]
        ys = [s[1] / 10.0 for s in strokes]

        # Accumulate absolute positions from deltas
        abs_x, abs_y = [0.0], [0.0]
        for dx, dy, _ in strokes:
            abs_x.append(abs_x[-1] + dx / 10.0)
            abs_y.append(abs_y[-1] + dy / 10.0)

        width   = max(abs_x) - min(abs_x)
        ascent  = max(0.0,  max(abs_y))
        descent = max(0.0, -min(abs_y))

        return (width, ascent, descent)

import math
from typing import Iterable, Optional, Tuple

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positions enriched with hourly/daily cyclic embeddings."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        # Base absolute position encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Cyclic components (hourly + daily)
        hour = (torch.arange(0, max_len, dtype=torch.float32) % 24) / 24.0
        day = (torch.arange(0, max_len, dtype=torch.float32) % 365) / 365.0
        hour_term = (2 * math.pi * hour).unsqueeze(1) * div_term
        day_term = (2 * math.pi * day).unsqueeze(1) * div_term

        # Blend cyclic information into same dimensions (scaled to keep magnitudes stable)
        pe[:, 0::2] += 0.1 * torch.sin(hour_term) + 0.1 * torch.sin(day_term)
        pe[:, 1::2] += 0.1 * torch.cos(hour_term) + 0.1 * torch.cos(day_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class DilatedConvBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 3, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        layers = []
        for i in range(depth):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.extend(
                [
                    nn.Conv1d(d_model, d_model, kernel_size, padding=padding, dilation=dilation),
                    nn.GELU(),
                    nn.BatchNorm1d(d_model),
                    nn.Dropout(dropout),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        return y


class ChannelAttention(nn.Module):
    def __init__(self, d_model: int, reduction: int = 4):
        super().__init__()
        hidden = max(d_model // reduction, 8)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.net(x.mean(dim=1))
        return x * weights.unsqueeze(1)


class HydraTemporalModel(nn.Module):
    """Hybrid temporal encoder mixing transformer, dilated convolutions, and recent history MLP."""

    def __init__(
        self,
        input_dim: int,
        static_dim: int = 0,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        seq_len: int = 168,
        conv_depth: int = 4,
        dropout: float = 0.1,
        quantiles: Optional[Iterable[float]] = None,
        nwm_index: int = 0,
        patch_size: int = 14,
    ) -> None:
        super().__init__()
        self.quantiles = tuple(quantiles) if quantiles is not None else None
        self.nwm_index = nwm_index
        self.seq_len = seq_len
        self.input_dim = input_dim

        if patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        self.patch_size = patch_size
        if patch_size > 1:
            self.patch_conv = nn.Conv1d(input_dim, d_model, kernel_size=patch_size, stride=patch_size)
            self.input_proj = None
            max_positions = (seq_len + patch_size - 1) // patch_size + 1
        else:
            self.patch_conv = None
            self.input_proj = nn.Linear(input_dim, d_model)
            max_positions = seq_len + 1
        self.positional = PositionalEncoding(d_model, max_len=max_positions)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.conv_branch = DilatedConvBlock(d_model, depth=conv_depth, dropout=dropout)
        self.recent_mlp = nn.Sequential(
            nn.Linear(d_model + input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.channel_attn = ChannelAttention(d_model)

        static_out = 0
        if static_dim > 0:
            self.static_encoder = nn.Sequential(
                nn.Linear(static_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            static_out = d_model
        else:
            self.static_encoder = None

        fusion_in = d_model * 3 + static_out
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, d_model * 2),
            nn.GLU(),
            nn.LayerNorm(d_model),
        )

        self.head_residual = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.residual_bias = nn.Parameter(torch.tensor([-0.15], dtype=torch.float32))

        if self.quantiles:
            self.head_quantiles = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, len(self.quantiles)),
            )
        else:
            self.head_quantiles = None

    def forward(
        self,
        x_seq: torch.Tensor,
        static_feats: Optional[torch.Tensor] = None,
    ) -> dict:
        orig_seq = x_seq
        if self.patch_conv is not None:
            pad_needed = (-orig_seq.size(1)) % self.patch_size
            if pad_needed > 0:
                pad = orig_seq[:, -1:, :].expand(-1, pad_needed, -1)
                work_seq = torch.cat([orig_seq, pad], dim=1)
            else:
                work_seq = orig_seq
            seq = self.patch_conv(work_seq.transpose(1, 2)).transpose(1, 2)
        else:
            seq = self.input_proj(orig_seq)
        seq = self.positional(seq)

        seq_len = seq.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=seq.device, dtype=torch.bool), diagonal=1
        )
        trans_out = self.transformer(seq, mask=causal_mask)
        conv_out = self.conv_branch(seq)
        recent_input = torch.cat([seq[:, -1, :], orig_seq[:, -1, :]], dim=-1)
        recent_out = self.recent_mlp(recent_input)
        trans_out = self.channel_attn(trans_out)[:, -1, :]
        conv_out = conv_out[:, -1, :]

        pieces = [trans_out, conv_out, recent_out]
        if self.static_encoder is not None and static_feats is not None:
            static_vec = self.static_encoder(static_feats)
            pieces.append(static_vec)

        fused = torch.cat(pieces, dim=-1)
        fused = self.fusion(fused)

        residual = self.head_residual(fused).squeeze(-1) + self.residual_bias
        nwm_last = orig_seq[:, -1, self.nwm_index]
        corrected = residual + nwm_last

        outputs = {"residual": residual, "corrected": corrected}
        if self.head_quantiles is not None:
            outputs["quantiles"] = self.head_quantiles(fused)
        return outputs

import math
from typing import Iterable, Optional, Tuple

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
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
    ) -> None:
        super().__init__()
        self.quantiles = tuple(quantiles) if quantiles is not None else None

        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional = PositionalEncoding(d_model, max_len=seq_len + 1)

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
            nn.Linear(d_model, d_model),
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

        self.head_corrected = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

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
        seq = self.input_proj(x_seq)
        seq = self.positional(seq)

        trans_out = self.transformer(seq)
        conv_out = self.conv_branch(seq)
        recent_out = self.recent_mlp(seq[:, -1, :])
        trans_out = self.channel_attn(trans_out)[:, -1, :]
        conv_out = conv_out[:, -1, :]

        pieces = [trans_out, conv_out, recent_out]
        if self.static_encoder is not None and static_feats is not None:
            static_vec = self.static_encoder(static_feats)
            pieces.append(static_vec)

        fused = torch.cat(pieces, dim=-1)
        fused = self.fusion(fused)

        residual = self.head_residual(fused).squeeze(-1)
        corrected = self.head_corrected(fused).squeeze(-1)

        outputs = {"residual": residual, "corrected": corrected}
        if self.head_quantiles is not None:
            outputs["quantiles"] = self.head_quantiles(fused)
        return outputs

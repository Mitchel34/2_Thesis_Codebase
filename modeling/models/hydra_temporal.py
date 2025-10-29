import math
from typing import Iterable, Optional

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
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


class HydraTemporalModel(nn.Module):
    """Minimal transformer baseline for residual post-processing."""

    def __init__(
        self,
        input_dim: int,
        static_dim: int = 0,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        seq_len: int = 168,
        conv_depth: int = 1,
        dropout: float = 0.1,
        quantiles: Optional[Iterable[float]] = None,
        nwm_index: int = 0,
        patch_size: int = 1,
        gain_scale: float = 0.05,
        logvar_min: float = -4.0,
        logvar_max: float = 2.0,
        moe_experts: int = 1,
    ) -> None:
        super().__init__()
        del seq_len, conv_depth, patch_size, gain_scale, moe_experts  # retained for config compatibility
        self.quantiles = tuple(float(q) for q in quantiles) if quantiles else None
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        self.input_norm = nn.LayerNorm(input_dim)
        self.pre_gru = nn.GRU(input_dim, d_model, batch_first=True)
        self.positional = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer_norm = nn.LayerNorm(d_model)
        self.pool_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn_pool = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        dilations = [1, 3, 6]
        self.scale_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=3, dilation=d, padding=d),
                    nn.GELU(),
                    nn.BatchNorm1d(d_model),
                )
                for d in dilations
            ]
        )
        self.scale_dropout = nn.Dropout(dropout)

        if static_dim > 0:
            self.static_encoder = nn.Sequential(
                nn.LayerNorm(static_dim),
                nn.Linear(static_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.static_cross = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            self.static_dropout = nn.Dropout(dropout)
        else:
            self.static_encoder = None
            self.static_cross = None
            self.static_dropout = None

        summary_components = 4 + len(self.scale_convs)  # last, mean, max, attn pooled, dilated summaries
        fusion_dim = d_model * summary_components
        if self.static_encoder is not None:
            fusion_dim += d_model  # static cross context
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.residual_head = nn.Linear(d_model, 1)
        self.residual_logvar_head = nn.Linear(d_model, 1)
        self.corrected_logvar_head = nn.Linear(d_model, 1)
        self.residual_bias = nn.Parameter(torch.tensor([-0.15], dtype=torch.float32))

        if self.quantiles:
            self.quantile_head = nn.Linear(d_model, len(self.quantiles))
        else:
            self.quantile_head = None

    def _summary(self, encoded: torch.Tensor) -> torch.Tensor:
        last = encoded[:, -1, :]
        mean = encoded.mean(dim=1)
        max_pool = encoded.max(dim=1).values
        return torch.cat([last, mean, max_pool], dim=-1)

    def _stabilize_logvar(self, raw: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.log(torch.nn.functional.softplus(raw) + 1e-4), self.logvar_min, self.logvar_max)

    def forward(self, x_seq: torch.Tensor, static_feats: Optional[torch.Tensor] = None) -> dict:
        seq = self.input_norm(x_seq)
        seq, _ = self.pre_gru(seq)
        seq = self.positional(seq)
        seq = self.dropout(seq)

        encoded = self.transformer(seq)
        encoded = self.transformer_norm(encoded)
        summary = self._summary(encoded)
        pool_q = self.pool_token.expand(encoded.size(0), -1, -1)
        attn_pooled, _ = self.attn_pool(pool_q, encoded, encoded)
        attn_summary = attn_pooled.squeeze(1)
        scale_features = []
        transposed = encoded.transpose(1, 2)
        for conv in self.scale_convs:
            conv_out = conv(transposed)
            conv_out = self.scale_dropout(conv_out)
            scale_features.append(conv_out.transpose(1, 2)[:, -1, :])
        multi_scale = torch.cat(scale_features, dim=-1) if scale_features else torch.zeros_like(summary)

        if self.static_encoder is not None and static_feats is not None:
            static_vec = self.static_encoder(static_feats)
            static_query = static_vec.unsqueeze(1)
            static_context, _ = self.static_cross(static_query, encoded, encoded)
            static_context = self.static_dropout(static_context).squeeze(1)
            stacked = torch.cat([summary, attn_summary, multi_scale, static_context], dim=-1)
        else:
            stacked = torch.cat([summary, attn_summary, multi_scale], dim=-1)

        fused = self.fusion(stacked)

        residual_mean = self.residual_head(fused).squeeze(-1) + self.residual_bias
        residual_logvar = self._stabilize_logvar(self.residual_logvar_head(fused).squeeze(-1))
        corrected_logvar = self._stabilize_logvar(self.corrected_logvar_head(fused).squeeze(-1))

        outputs = {
            "residual_mean": residual_mean,
            "residual_logvar": residual_logvar,
            "corrected_logvar": corrected_logvar,
        }

        if self.quantile_head is not None and self.quantiles:
            outputs["quantiles"] = self.quantile_head(fused)

        return outputs

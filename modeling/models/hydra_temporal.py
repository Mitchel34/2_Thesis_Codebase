import math
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positions enriched with hourly/daily cyclic embeddings."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        hour = (torch.arange(0, max_len, dtype=torch.float32) % 24) / 24.0
        day = (torch.arange(0, max_len, dtype=torch.float32) % 365) / 365.0
        hour_term = (2 * math.pi * hour).unsqueeze(1) * div_term
        day_term = (2 * math.pi * day).unsqueeze(1) * div_term

        pe[:, 0::2] += 0.1 * torch.sin(hour_term) + 0.1 * torch.sin(day_term)
        pe[:, 1::2] += 0.1 * torch.cos(hour_term) + 0.1 * torch.cos(day_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.padding = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class ResidualTCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.GELU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.GELU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout),
        )
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.residual(x)


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
        return self.net(x.transpose(1, 2)).transpose(1, 2)


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
    """Simplified Hydra encoder: TCN + transformer + heteroscedastic heads."""

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
        patch_size: int = 1,
        gain_scale: float = 0.05,
        logvar_min: float = -4.0,
        logvar_max: float = 2.0,
        moe_experts: int = 1,  # retained for backwards compatibility; treated as 1
    ) -> None:
        super().__init__()
        self.quantiles = tuple(float(q) for q in quantiles) if quantiles else None
        self.nwm_index = nwm_index
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.d_model = d_model
        self.gain_scale = gain_scale
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.num_experts = moe_experts if moe_experts and moe_experts > 1 else 0

        dilations = [1, 2, 4, 8][:conv_depth]
        blocks = []
        in_ch = input_dim
        for dilation in dilations:
            blocks.append(ResidualTCNBlock(in_ch, d_model, kernel_size=3, dilation=dilation, dropout=dropout))
            in_ch = d_model
        self.tcn = nn.Sequential(*blocks)

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
            static_out = 0

        fusion_in = d_model * 2 + static_out
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.residual_mean_head = nn.Linear(d_model, 1)
        self.corrected_mean_head = nn.Linear(d_model, 1)

        self.residual_logvar_head = nn.Linear(d_model, 1)
        self.corrected_logvar_head = nn.Linear(d_model, 1)
        self.gainbias_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),
        )

        if self.quantiles:
            self.quantile_head = nn.Linear(d_model, len(self.quantiles))
        else:
            self.quantile_head = None

    def forward(
        self,
        x_seq: torch.Tensor,
        static_feats: Optional[torch.Tensor] = None,
    ) -> dict:
        seq = self.tcn(x_seq.transpose(1, 2)).transpose(1, 2)

        seq = self.positional(seq)

        seq_len = seq.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=seq.device, dtype=torch.bool),
            diagonal=1,
        )
        trans_out = self.transformer(seq, mask=causal_mask)

        last_token = trans_out[:, -1, :]
        mean_token = trans_out.mean(dim=1)

        pieces = [last_token, mean_token]
        if self.static_encoder is not None and static_feats is not None:
            pieces.append(self.static_encoder(static_feats))

        fused = torch.cat(pieces, dim=-1)
        fused = self.fusion(fused)

        residual_mean = self.residual_mean_head(fused).squeeze(-1)
        corrected_mean_direct = self.corrected_mean_head(fused).squeeze(-1)

        residual_logvar = self.residual_logvar_head(fused).squeeze(-1).clamp(self.logvar_min, self.logvar_max)
        corrected_logvar = self.corrected_logvar_head(fused).squeeze(-1).clamp(self.logvar_min, self.logvar_max)

        gain_bias = self.gainbias_head(fused)
        gain = 1.0 + self.gain_scale * torch.tanh(gain_bias[:, 0])
        bias = gain_bias[:, 1]

        outputs = {
            "residual_mean": residual_mean,
            "residual_logvar": residual_logvar,
            "corrected_mean": corrected_mean_direct,
            "corrected_logvar": corrected_logvar,
            "gain": gain,
            "bias": bias,
        }

        if self.quantile_head is not None and self.quantiles:
            outputs["quantiles"] = self.quantile_head(fused)
        return outputs

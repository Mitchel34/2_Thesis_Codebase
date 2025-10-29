#!/usr/bin/env python3

"""
LSTM baseline training script for hourly NWM residual correction.
- Mirrors the quick transformer pipeline to enable apples-to-apples comparison.
- Reuses dataset preparation utilities from the transformer script.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
from contextlib import nullcontext
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from torch_optimizer import Ranger
except ImportError:  # pragma: no cover - optional dependency
    Ranger = None

from modeling.train_quick_transformer_torch import (
    SeqDataset,
    add_static_columns,
    compute_hydro_metrics,
    load_data,
    prepare_features,
    ERA5_CANDIDATES,
)


def focal_mse(pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, threshold: float = 1.0) -> torch.Tensor:
    """Focalised mean-squared error that down-weights small residuals."""
    diff = pred - target
    scale = torch.pow(torch.abs(diff) / (threshold + 1e-6) + 1.0, gamma)
    return torch.mean((diff**2) / scale)


class ResidualLSTMModel(nn.Module):
    """Simple LSTM encoder that predicts residual and corrected streamflow."""

    def __init__(
        self,
        input_dim: int,
        static_dim: int = 0,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.static_dim = static_dim
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        if static_dim > 0:
            self.static_proj = nn.Sequential(
                nn.LayerNorm(static_dim),
                nn.Linear(static_dim, lstm_out_dim),
                nn.GELU(),
            )
            fusion_dim = lstm_out_dim * 2
        else:
            self.static_proj = None
            fusion_dim = lstm_out_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res_out = nn.Linear(fusion_dim, 1)
        self.corrected_out = nn.Linear(fusion_dim, 1)

    def forward(self, x: torch.Tensor, static: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        seq_out, _ = self.lstm(x)
        last_hidden = seq_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        if self.static_proj is not None and static is not None:
            static_feat = self.static_proj(static)
            features = torch.cat([last_hidden, static_feat], dim=-1)
        else:
            features = last_hidden
        fused = self.fusion(features)
        residual = self.res_out(fused).squeeze(-1)
        corrected = self.corrected_out(fused).squeeze(-1)
        return {'residual': residual, 'corrected': corrected}


def train_eval(
    data_path: str,
    seq_len: int = 168,
    epochs: int = 40,
    batch_size: int = 64,
    train_days: int = 28,
    val_days: int = 7,
    patience: int = 5,
    hidden_size: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
    bidirectional: bool = False,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    use_amp: bool = True,
    augment: bool = True,
    focal_gamma: float = 2.0,
    focal_std_factor: float = 2.0,
    use_ranger: bool = True,
    output_prefix: str = "quick_lstm",
) -> None:
    torch.set_float32_matmul_precision("medium")
    df = load_data(data_path)
    start = df['timestamp'].min()
    train_cutoff = start + pd.Timedelta(days=train_days)
    val_cutoff = train_cutoff + pd.Timedelta(days=val_days) if val_days > 0 else train_cutoff

    train_mask = df['timestamp'] < train_cutoff
    if train_mask.sum() <= seq_len:
        raise ValueError("Training window shorter than sequence length")

    if val_days > 0:
        val_mask = (df['timestamp'] >= train_cutoff) & (df['timestamp'] < val_cutoff)
    else:
        val_mask = pd.Series(False, index=df.index)

    test_mask = df['timestamp'] >= val_cutoff if val_days > 0 else ~train_mask
    if test_mask.sum() == 0:
        raise ValueError("No evaluation rows available after split")

    dynamic_cols = ['nwm_cms'] + [c for c in ERA5_CANDIDATES if c in df.columns]
    if not dynamic_cols or dynamic_cols[0] != 'nwm_cms':
        raise ValueError("First dynamic feature must be 'nwm_cms'")

    static_cols = add_static_columns(df)

    dyn_scaled, static_scaled, _ = prepare_features(df, df.index[train_mask], dynamic_cols, static_cols)

    residual = df['y_residual_cms'].to_numpy()
    usgs = df['usgs_cms'].to_numpy()
    nwm = df['nwm_cms'].to_numpy()

    zero_static = lambda rows: np.zeros((rows, 0), dtype=np.float32)

    train_idx = train_mask.values
    val_idx = val_mask.values
    test_idx = test_mask.values

    train_dyn = dyn_scaled[train_idx]
    train_static = static_scaled[train_idx] if static_cols else zero_static(train_idx.sum())
    train_resid = residual[train_idx]
    train_usgs = usgs[train_idx]
    train_nwm = nwm[train_idx]

    val_dyn = dyn_scaled[val_idx]
    val_static = static_scaled[val_idx] if static_cols else zero_static(val_idx.sum())
    val_resid = residual[val_idx]
    val_usgs = usgs[val_idx]
    val_nwm = nwm[val_idx]

    test_dyn = dyn_scaled[test_idx]
    test_static = static_scaled[test_idx] if static_cols else zero_static(test_idx.sum())
    test_resid = residual[test_idx]
    test_usgs = usgs[test_idx]
    test_nwm = nwm[test_idx]

    tail_train = min(seq_len, train_dyn.shape[0])
    if tail_train < seq_len:
        raise ValueError("Training window shorter than sequence length")

    train_ds = SeqDataset(
        train_dyn,
        train_static,
        train_resid,
        train_usgs,
        train_nwm,
        seq_len,
        augment=augment,
    )

    val_ds = None
    if val_dyn.shape[0] > 0:
        val_seed_dyn = np.concatenate([train_dyn[-tail_train:], val_dyn], axis=0)
        val_seed_static = (
            np.concatenate([train_static[-tail_train:], val_static], axis=0)
            if static_cols
            else zero_static(val_seed_dyn.shape[0])
        )
        val_seed_resid = np.concatenate([train_resid[-tail_train:], val_resid])
        val_seed_usgs = np.concatenate([train_usgs[-tail_train:], val_usgs])
        val_seed_nwm = np.concatenate([train_nwm[-tail_train:], val_nwm])
        val_ds = SeqDataset(
            val_seed_dyn,
            val_seed_static,
            val_seed_resid,
            val_seed_usgs,
            val_seed_nwm,
            seq_len,
            augment=False,
        )

    history_dyn = train_dyn if val_dyn.shape[0] == 0 else np.concatenate([train_dyn, val_dyn], axis=0)
    history_static = (
        train_static if val_dyn.shape[0] == 0 else np.concatenate([train_static, val_static], axis=0)
    ) if static_cols else zero_static(history_dyn.shape[0])
    history_resid = train_resid if val_dyn.shape[0] == 0 else np.concatenate([train_resid, val_resid])
    history_usgs = train_usgs if val_dyn.shape[0] == 0 else np.concatenate([train_usgs, val_usgs])
    history_nwm = train_nwm if val_dyn.shape[0] == 0 else np.concatenate([train_nwm, val_nwm])

    tail_history = min(seq_len, history_dyn.shape[0])
    if tail_history < seq_len:
        raise ValueError("Insufficient history to seed evaluation sequences")

    test_seed_dyn = np.concatenate([history_dyn[-tail_history:], test_dyn], axis=0)
    test_seed_static = (
        np.concatenate([history_static[-tail_history:], test_static], axis=0)
        if static_cols
        else zero_static(test_seed_dyn.shape[0])
    )
    test_seed_resid = np.concatenate([history_resid[-tail_history:], test_resid])
    test_seed_usgs = np.concatenate([history_usgs[-tail_history:], test_usgs])
    test_seed_nwm = np.concatenate([history_nwm[-tail_history:], test_nwm])

    test_ds = SeqDataset(
        test_seed_dyn,
        test_seed_static,
        test_seed_resid,
        test_seed_usgs,
        test_seed_nwm,
        seq_len,
        augment=False,
    )

    if len(train_ds) == 0 or len(test_ds) == 0:
        raise ValueError("Insufficient data to build sequences")

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
        if val_ds is not None
        else None
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    model = ResidualLSTMModel(
        input_dim=len(dynamic_cols),
        static_dim=len(static_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)

    if use_ranger and Ranger is None:
        print("torch_optimizer.Ranger unavailable; falling back to AdamW")
    use_ranger = use_ranger and Ranger is not None
    if use_ranger:
        optimizer = Ranger(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse = nn.MSELoss()

    amp_enabled = use_amp and device.type == 'cuda'
    amp_dtype = torch.float16
    scaler = torch.cuda.amp.GradScaler() if amp_enabled and device.type == 'cuda' else None

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for xb, sb, y_res, y_usgs, y_nwm in train_loader:
            xb = xb.to(device)
            sb = sb.to(device)
            y_res = y_res.to(device)
            y_usgs = y_usgs.to(device)
            y_nwm = y_nwm.to(device)

            optimizer.zero_grad(set_to_none=True)
            res_threshold = None
            corr_threshold = None
            if focal_std_factor is not None and focal_std_factor > 0:
                res_stats = y_res.detach().float()
                corr_stats = y_usgs.detach().float()
                res_threshold = float(res_stats.abs().mean() + focal_std_factor * res_stats.std())
                corr_threshold = float(corr_stats.abs().mean() + focal_std_factor * corr_stats.std())
            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled)
                if amp_enabled
                else nullcontext()
            )
            with autocast_ctx:
                outputs = model(xb, sb if sb.numel() else None)
                pred_res = outputs['residual']
                pred_corr = outputs['corrected']
                corr_from_res = y_nwm + pred_res

                loss_res = focal_mse(pred_res, y_res, gamma=focal_gamma, threshold=res_threshold)
                loss_corr = focal_mse(pred_corr, y_usgs, gamma=focal_gamma, threshold=corr_threshold)
                loss_consistency = mse(pred_corr, corr_from_res)
                bias_penalty = torch.abs(pred_res.mean())
                loss = loss_res + loss_corr + 0.1 * loss_consistency + 0.01 * bias_penalty

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            running_loss += loss.item() * len(xb)
            sample_count += len(xb)

        train_loss = running_loss / max(sample_count, 1)

        val_loss = float('nan')
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            val_samples = 0
            with torch.no_grad():
                for xb, sb, y_res, y_usgs, y_nwm in val_loader:
                    xb = xb.to(device)
                    sb = sb.to(device)
                    y_res = y_res.to(device)
                    y_usgs = y_usgs.to(device)
                    y_nwm = y_nwm.to(device)
                    res_threshold = None
                    corr_threshold = None
                    if focal_std_factor is not None and focal_std_factor > 0:
                        res_stats = y_res.float()
                        corr_stats = y_usgs.float()
                        res_threshold = float(res_stats.abs().mean() + focal_std_factor * res_stats.std())
                        corr_threshold = float(corr_stats.abs().mean() + focal_std_factor * corr_stats.std())
                    autocast_ctx = (
                        torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled)
                        if amp_enabled
                        else nullcontext()
                    )
                    with autocast_ctx:
                        outputs = model(xb, sb if sb.numel() else None)
                        pred_res = outputs['residual']
                        pred_corr = outputs['corrected']
                        corr_from_res = y_nwm + pred_res

                        loss_res = focal_mse(pred_res, y_res, gamma=focal_gamma, threshold=res_threshold)
                        loss_corr = focal_mse(pred_corr, y_usgs, gamma=focal_gamma, threshold=corr_threshold)
                        loss_consistency = mse(pred_corr, corr_from_res)
                        bias_penalty = torch.abs(pred_res.mean())
                        loss = loss_res + loss_corr + 0.1 * loss_consistency + 0.01 * bias_penalty

                    val_running += loss.item() * len(xb)
                    val_samples += len(xb)

            val_loss = val_running / max(val_samples, 1)
            print(f"Epoch {epoch+1}/{epochs} - train loss: {train_loss:.4f} - val loss: {val_loss:.4f}")

            if val_loss + 1e-6 < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered based on validation loss")
                    break
        else:
            print(f"Epoch {epoch+1}/{epochs} - train loss: {train_loss:.4f}")
            best_state = copy.deepcopy(model.state_dict())

        if use_ranger:
            metric = val_loss if val_loader is not None and not math.isnan(val_loss) else train_loss
            scheduler.step(metric)
        else:
            scheduler.step()

    model.load_state_dict(best_state)
    model.eval()
    preds_res, preds_corr, targets_res, targets_usgs, targets_nwm = [], [], [], [], []
    with torch.no_grad():
        for xb, sb, y_res, y_usgs, y_nwm in test_loader:
            xb = xb.to(device)
            sb = sb.to(device)
            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled)
                if amp_enabled
                else nullcontext()
            )
            with autocast_ctx:
                out = model(xb, sb if sb.numel() else None)
            preds_res.append(out['residual'].to(torch.float32).cpu().numpy())
            preds_corr.append(out['corrected'].to(torch.float32).cpu().numpy())
            targets_res.append(y_res.numpy())
            targets_usgs.append(y_usgs.numpy())
            targets_nwm.append(y_nwm.numpy())

    pred_residual = np.concatenate(preds_res)
    pred_corrected = np.concatenate(preds_corr)
    true_residual = np.concatenate(targets_res)
    true_usgs = np.concatenate(targets_usgs)
    nwm_vals = np.concatenate(targets_nwm)

    corrected_from_res = nwm_vals + pred_residual
    corrected_pred = 0.5 * (pred_corrected + corrected_from_res)

    mse_res = np.mean((pred_residual - true_residual) ** 2)
    rmse_res = float(np.sqrt(mse_res))
    print(f"Residual RMSE: {rmse_res:.3f}")

    baseline_metrics = compute_hydro_metrics(nwm_vals, true_usgs)
    corrected_metrics = compute_hydro_metrics(corrected_pred, true_usgs)

    base_rmse = baseline_metrics['rmse']
    corr_rmse = corrected_metrics['rmse']
    rel_improve = 100.0 * (base_rmse - corr_rmse) / base_rmse if base_rmse and base_rmse > 0 else float('nan')
    print("Baseline Metrics:")
    for k, v in baseline_metrics.items():
        print(f"  {k.upper()}: {v:.4f}")

    print("Corrected Metrics:")
    for k, v in corrected_metrics.items():
        print(f"  {k.upper()}: {v:.4f}")

    print(f"ΔRMSE% (Corrected vs Baseline): {rel_improve:.2f}%")

    out_dir = Path('data/clean/modeling')
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = output_prefix
    np.save(out_dir / f'{base_name}_pred.npy', pred_residual)
    np.save(out_dir / f'{base_name}_true.npy', true_residual)

    timestamps = df.loc[test_mask, 'timestamp'].to_numpy()
    aligned_len = min(len(timestamps), len(corrected_pred))
    result_df = pd.DataFrame({
        'timestamp': timestamps[:aligned_len],
        'y_true_residual_cms': true_residual[:aligned_len],
        'y_pred_residual_cms': pred_residual[:aligned_len],
        'nwm_cms': nwm_vals[:aligned_len],
        'usgs_cms': true_usgs[:aligned_len],
        'corrected_pred_cms': corrected_pred[:aligned_len],
        'corrected_true_cms': true_usgs[:aligned_len],
    })
    result_df.to_csv(out_dir / f'{base_name}_eval.csv', index=False)

    def _clean_dict(d: Dict[str, float]) -> Dict[str, float | None]:
        out: Dict[str, float | None] = {}
        for k, v in d.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out[k] = None
            else:
                out[k] = float(v)
        return out

    metrics_payload = {
        'baseline': _clean_dict(baseline_metrics),
        'corrected': _clean_dict(corrected_metrics),
        'rmse_residual': None if np.isnan(rmse_res) else float(rmse_res),
        'rmse_improvement_pct': None if np.isnan(rel_improve) else float(rel_improve),
    }

    metrics_payload['output_prefix'] = base_name
    with open(out_dir / f'{base_name}_metrics.json', 'w') as fh:
        json.dump(metrics_payload, fh, indent=2)

    with open(out_dir / f'{base_name}_features.txt', 'w') as fh:
        fh.write("Dynamic columns:\n")
        fh.write("\n".join(dynamic_cols))
        if static_cols:
            fh.write("\nStatic columns:\n")
            fh.write("\n".join(static_cols))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to Parquet training data')
    parser.add_argument('--seq-len', type=int, default=168)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--train-days', type=int, default=28)
    parser.add_argument('--val-days', type=int, default=7)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--focal-std-factor', type=float, default=2.0)
    parser.add_argument('--no-ranger', action='store_true')
    parser.add_argument('--output-prefix', default='quick_lstm')
    args = parser.parse_args()

    train_eval(
        data_path=args.data,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_days=args.train_days,
        val_days=args.val_days,
        patience=args.patience,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_amp=not args.no_amp,
        augment=not args.no_augment,
        focal_gamma=args.focal_gamma,
        focal_std_factor=args.focal_std_factor,
        use_ranger=not args.no_ranger,
        output_prefix=args.output_prefix,
    )

#!/usr/bin/env python3
"""
Hydra temporal model training script (PyTorch) for NWM residual correction.
- Dynamic features: NWM + available ERA5 columns (normalized per train split)
- Static features: NLCD percentages, regulation flag, etc. (normalized)
- Targets: residual (USGS - NWM) and corrected flow (USGS)
"""
import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from modeling.models.hydra_temporal import HydraTemporalModel

ERA5_CANDIDATES = [
    'temp_c', 'dewpoint_c', 'pressure_hpa', 'precip_mm', 'radiation_mj_m2',
    'wind_speed', 'vpd_kpa', 'rel_humidity_pct', 'soil_moisture_vwc',
    'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos', 'month_sin', 'month_cos'
]

STATIC_NUMERIC = [
    'urban_percent', 'forest_percent', 'agriculture_percent', 'impervious_percent'
]


class SeqDataset(Dataset):
    def __init__(
        self,
        dyn: np.ndarray,
        static: np.ndarray,
        residual: np.ndarray,
        usgs: np.ndarray,
        nwm: np.ndarray,
        seq_len: int,
    ) -> None:
        self.seq_len = seq_len
        self.dyn = dyn.astype(np.float32)
        self.static = static.astype(np.float32) if static.size else static
        self.residual = residual.astype(np.float32)
        self.usgs = usgs.astype(np.float32)
        self.nwm = nwm.astype(np.float32)
        self.length = max(len(self.dyn) - seq_len, 0)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        j = idx + self.seq_len
        seq = self.dyn[idx:j]
        static_vec = self.static[j] if self.static.size else np.zeros(0, dtype=np.float32)
        return (
            torch.from_numpy(seq),
            torch.from_numpy(static_vec),
            torch.tensor(self.residual[j], dtype=torch.float32),
            torch.tensor(self.usgs[j], dtype=torch.float32),
            torch.tensor(self.nwm[j], dtype=torch.float32),
        )


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {'timestamp', 'nwm_cms', 'usgs_cms', 'y_residual_cms'}
    if not required.issubset(df.columns):
        missing = required.difference(df.columns)
        raise ValueError(f"Missing columns in dataset: {missing}")
    df = df.sort_values('timestamp').dropna(subset=list(required))
    return df.reset_index(drop=True)


def add_static_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in STATIC_NUMERIC:
        if col in df.columns:
            cols.append(col)
    if 'regulation_status' in df.columns:
        df['is_regulated'] = (df['regulation_status'] == 'Regulated').astype(float)
        cols.append('is_regulated')
    return cols


def prepare_features(
    df: pd.DataFrame,
    train_idx: pd.Index,
    dynamic_cols: List[str],
    static_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, dict]:
    dyn_mean = df.loc[train_idx, dynamic_cols].mean()
    dyn_std = df.loc[train_idx, dynamic_cols].std().replace(0, 1)
    dyn_scaled = ((df[dynamic_cols] - dyn_mean) / dyn_std).fillna(0.0)

    static_mean = None
    static_std = None
    if static_cols:
        static_mean = df.loc[train_idx, static_cols].mean()
        static_std = df.loc[train_idx, static_cols].std().replace(0, 1)
        static_scaled = ((df[static_cols] - static_mean) / static_std).fillna(0.0)
    else:
        static_scaled = pd.DataFrame(index=df.index)

    stats = {
        'dyn_mean': dyn_mean,
        'dyn_std': dyn_std,
        'static_mean': static_mean,
        'static_std': static_std,
    }
    return dyn_scaled.to_numpy(), static_scaled.to_numpy(), stats


def train_eval(
    data_path: str,
    seq_len: int = 168,
    epochs: int = 40,
    batch_size: int = 64,
    train_days: int = 28,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    conv_depth: int = 4,
    dropout: float = 0.1,
    lr: float = 1e-3,
) -> None:
    df = load_data(data_path)
    start = df['timestamp'].min()
    split_time = start + pd.Timedelta(days=train_days)
    train_mask = df['timestamp'] < split_time

    dynamic_cols = ['nwm_cms'] + [c for c in ERA5_CANDIDATES if c in df.columns]
    static_cols = add_static_columns(df)

    dyn_scaled, static_scaled, _ = prepare_features(df, df.index[train_mask], dynamic_cols, static_cols)

    residual = df['y_residual_cms'].to_numpy()
    usgs = df['usgs_cms'].to_numpy()
    nwm = df['nwm_cms'].to_numpy()

    train_dyn = dyn_scaled[train_mask.values]
    train_static = static_scaled[train_mask.values] if static_cols else np.zeros((train_mask.sum(), 0), dtype=np.float32)
    train_resid = residual[train_mask.values]
    train_usgs = usgs[train_mask.values]
    train_nwm = nwm[train_mask.values]

    if train_dyn.shape[0] <= seq_len:
        raise ValueError("Training window shorter than sequence length")

    test_mask = ~train_mask
    test_dyn = dyn_scaled[test_mask.values]
    test_static = static_scaled[test_mask.values] if static_cols else np.zeros((test_mask.sum(), 0), dtype=np.float32)
    test_resid = residual[test_mask.values]
    test_usgs = usgs[test_mask.values]
    test_nwm = nwm[test_mask.values]

    if test_dyn.size == 0:
        raise ValueError("No evaluation rows available after split")

    tail = min(seq_len, train_dyn.shape[0])
    seed_dyn = train_dyn[-tail:]
    seed_static = train_static[-tail:] if static_cols else np.zeros((tail, 0), dtype=np.float32)
    seed_resid = train_resid[-tail:]
    seed_usgs = train_usgs[-tail:]
    seed_nwm = train_nwm[-tail:]

    test_dyn_seeded = np.concatenate([seed_dyn, test_dyn], axis=0)
    test_static_seeded = np.concatenate([seed_static, test_static], axis=0) if static_cols else np.zeros((len(test_dyn_seeded), 0), dtype=np.float32)
    test_resid_seeded = np.concatenate([seed_resid, test_resid])
    test_usgs_seeded = np.concatenate([seed_usgs, test_usgs])
    test_nwm_seeded = np.concatenate([seed_nwm, test_nwm])

    train_ds = SeqDataset(train_dyn, train_static, train_resid, train_usgs, train_nwm, seq_len)
    test_ds = SeqDataset(test_dyn_seeded, test_static_seeded, test_resid_seeded, test_usgs_seeded, test_nwm_seeded, seq_len)

    if len(train_ds) == 0 or len(test_ds) == 0:
        raise ValueError("Insufficient data to build sequences")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HydraTemporalModel(
        input_dim=len(dynamic_cols),
        static_dim=len(static_cols),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        seq_len=seq_len,
        conv_depth=conv_depth,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for xb, sb, y_res, y_usgs, y_nwm in train_loader:
            xb = xb.to(device)
            sb = sb.to(device)
            y_res = y_res.to(device)
            y_usgs = y_usgs.to(device)
            y_nwm = y_nwm.to(device)

            optimizer.zero_grad()
            outputs = model(xb, sb if sb.numel() else None)
            pred_res = outputs['residual']
            pred_corr = outputs['corrected']
            corr_from_res = y_nwm + pred_res

            loss_res = mse(pred_res, y_res)
            loss_corr = mse(pred_corr, y_usgs)
            loss_consistency = mse(pred_corr, corr_from_res)
            bias_penalty = torch.abs(pred_res.mean())
            loss = loss_res + loss_corr + 0.1 * loss_consistency + 0.01 * bias_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * len(xb)

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} - train loss: {running_loss / len(train_ds):.4f}")

    model.eval()
    preds_res, preds_corr, targets_res, targets_usgs, targets_nwm = [], [], [], [], []
    with torch.no_grad():
        for xb, sb, y_res, y_usgs, y_nwm in test_loader:
            xb = xb.to(device)
            sb = sb.to(device)
            out = model(xb, sb if sb.numel() else None)
            preds_res.append(out['residual'].cpu().numpy())
            preds_corr.append(out['corrected'].cpu().numpy())
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

    base_rmse = float(np.sqrt(np.mean((nwm_vals - true_usgs) ** 2)))
    corr_rmse = float(np.sqrt(np.mean((corrected_pred - true_usgs) ** 2)))
    rel_improve = 100.0 * (base_rmse - corr_rmse) / base_rmse if base_rmse > 0 else float('nan')
    print(f"Baseline RMSE: {base_rmse:.3f} | Corrected RMSE: {corr_rmse:.3f} | Δ%: {rel_improve:.1f}%")

    out_dir = 'data/clean/modeling'
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'quick_pred.npy'), pred_residual)
    np.save(os.path.join(out_dir, 'quick_true.npy'), true_residual)

    timestamps = df.loc[~train_mask, 'timestamp'].to_numpy()
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
    result_df.to_csv(os.path.join(out_dir, 'quick_eval.csv'), index=False)

    with open(os.path.join(out_dir, 'quick_features.txt'), 'w') as fh:
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
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--conv-depth', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    train_eval(
        data_path=args.data,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_days=args.train_days,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        conv_depth=args.conv_depth,
        dropout=args.dropout,
        lr=args.lr,
    )

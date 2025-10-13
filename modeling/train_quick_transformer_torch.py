#!/usr/bin/env python3

"""
Hydra temporal model training script (PyTorch) for NWM residual correction.
- Dynamic features: NWM + available ERA5 columns (normalized per train split)
- Static features: NLCD percentages, regulation flag, etc. (normalized)
- Targets: residual (USGS - NWM) and corrected flow (USGS)

Version 2 adds:
- TCN + transformer hybrid with FiLM conditioning
- Gain/bias head and heteroscedastic outputs
- Gaussian NLL losses on asinh-transformed targets
"""

import argparse
import copy
import json 
import math
import os
import time
from contextlib import nullcontext
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from torch_optimizer import Ranger
except ImportError:  # pragma: no cover - optional dependency
    Ranger = None

from modeling.models.hydra_temporal import HydraTemporalModel

ERA5_CANDIDATES = [
    "temp_c",
    "dewpoint_c",
    "pressure_hpa",
    "precip_mm",
    "radiation_mj_m2",
    "wind_speed",
    "vpd_kpa",
    "rel_humidity_pct",
    "soil_moisture_vwc",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "month_sin",
    "month_cos",
]

STATIC_NUMERIC = [
    "urban_percent",
    "forest_percent",
    "agriculture_percent",
    "impervious_percent",
]

EPS = 1e-6


def transform_target(x: torch.Tensor) -> torch.Tensor:
    return torch.asinh(x)


def inverse_transform(x: torch.Tensor) -> torch.Tensor:
    return torch.sinh(x)


def transform_np(x: np.ndarray) -> np.ndarray:
    return np.arcsinh(x)


def inverse_transform_np(x: np.ndarray) -> np.ndarray:
    return np.sinh(x)


class SeqDataset(Dataset):
    def __init__(
        self,
        dyn: np.ndarray,
        static: np.ndarray,
        residual: np.ndarray,
        usgs: np.ndarray,
        nwm: np.ndarray,
        seq_len: int,
        augment: bool = False,
    ) -> None:
        self.seq_len = seq_len
        self.dyn = dyn.astype(np.float32)
        self.static = static.astype(np.float32) if static.size else static
        self.residual = residual.astype(np.float32)
        self.usgs = usgs.astype(np.float32)
        self.nwm = nwm.astype(np.float32)
        self.length = max(len(self.dyn) - seq_len, 0)
        self.augment = augment

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        j = idx + self.seq_len
        seq = self.dyn[idx:j]
        if self.augment and np.random.rand() < 0.5:
            seq = seq + np.random.normal(0.0, 0.05, size=seq.shape).astype(np.float32)
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
    required = {"timestamp", "nwm_cms", "usgs_cms", "y_residual_cms"}
    if not required.issubset(df.columns):
        missing = required.difference(df.columns)
        raise ValueError(f"Missing columns in dataset: {missing}")
    df = df.sort_values("timestamp").dropna(subset=list(required))
    return df.reset_index(drop=True)


def add_static_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in STATIC_NUMERIC:
        if col in df.columns:
            cols.append(col)
    if "regulation_status" in df.columns:
        df["is_regulated"] = (df["regulation_status"] == "Regulated").astype(float)
        cols.append("is_regulated")
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
        "dyn_mean": dyn_mean,
        "dyn_std": dyn_std,
        "static_mean": static_mean,
        "static_std": static_std,
    }
    return dyn_scaled.to_numpy(), static_scaled.to_numpy(), stats


def _safe_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if arr.size else float("nan")


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def compute_hydro_metrics(pred: np.ndarray, obs: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if pred.size == 0 or obs.size == 0:
        return {k: float("nan") for k in ["rmse", "mae", "nse", "kge", "pbias", "pearson_r", "spearman_r"]}

    pred = pred.astype(np.float64)
    obs = obs.astype(np.float64)

    diff = pred - obs
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))

    obs_mean = _safe_mean(obs)
    denom = np.sum((obs - obs_mean) ** 2)
    nse = float(1.0 - (np.sum(diff**2) / denom)) if denom > 0 else float("nan")

    pred_mean = _safe_mean(pred)
    pred_std = float(np.std(pred))
    obs_std = float(np.std(obs))
    pearson = float(np.corrcoef(pred, obs)[0, 1]) if pred_std > 0 and obs_std > 0 else float("nan")

    alpha = (pred_std / obs_std) if obs_std > 0 else float("nan")
    beta = (pred_mean / obs_mean) if obs_mean != 0 else float("nan")
    if not np.isnan(alpha) and not np.isnan(beta) and not np.isnan(pearson):
        kge = float(1.0 - math.sqrt((pearson - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    else:
        kge = float("nan")

    pbias = float(100.0 * np.sum(diff) / np.sum(obs)) if np.sum(obs) != 0 else float("nan")
    spearman = _spearman_corr(pred, obs)

    metrics.update(
        {
            "rmse": rmse,
            "mae": mae,
            "nse": nse,
            "kge": kge,
            "pbias": pbias,
            "pearson_r": pearson,
            "spearman_r": spearman,
        }
    )
    return metrics


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inv_var = torch.exp(-logvar)
    return 0.5 * ((target - mean) ** 2 * inv_var + logvar)


def train_eval(
    data_path: str,
    seq_len: int = 168,
    epochs: int = 40,
    batch_size: int = 64,
    train_days: int = 28,
    val_days: int = 7,
    patience: int = 5,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    conv_depth: int = 4,
    dropout: float = 0.1,
    lr: float = 5e-4,
    use_amp: bool = True,
    use_compile: bool = False,
    augment: bool = True,
    use_ranger: bool = True,
    output_prefix: str = "hydra_v2",
) -> None:
    torch.set_float32_matmul_precision("medium")
    df = load_data(data_path)
    start = df["timestamp"].min()
    train_cutoff = start + pd.Timedelta(days=train_days)
    val_cutoff = train_cutoff + pd.Timedelta(days=val_days) if val_days > 0 else train_cutoff

    train_mask = df["timestamp"] < train_cutoff
    if train_mask.sum() <= seq_len:
        raise ValueError("Training window shorter than sequence length")

    if val_days > 0:
        val_mask = (df["timestamp"] >= train_cutoff) & (df["timestamp"] < val_cutoff)
    else:
        val_mask = pd.Series(False, index=df.index)

    test_mask = df["timestamp"] >= val_cutoff if val_days > 0 else ~train_mask
    if test_mask.sum() == 0:
        raise ValueError("No evaluation rows available after split")

    dynamic_cols = ["nwm_cms"] + [c for c in ERA5_CANDIDATES if c in df.columns]
    if not dynamic_cols or dynamic_cols[0] != "nwm_cms":
        raise ValueError("First dynamic feature must be 'nwm_cms'")

    static_cols = add_static_columns(df)

    dyn_scaled, static_scaled, _ = prepare_features(df, df.index[train_mask], dynamic_cols, static_cols)

    residual = df["y_residual_cms"].to_numpy()
    usgs = df["usgs_cms"].to_numpy()
    nwm = df["nwm_cms"].to_numpy()

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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "mps":
        use_amp = False

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
        if val_ds is not None
        else None
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    model = HydraTemporalModel(
        input_dim=len(dynamic_cols),
        static_dim=len(static_cols),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        seq_len=seq_len,
        conv_depth=conv_depth,
        dropout=dropout,
        nwm_index=0,
        patch_size=1,
    )

    if use_compile:
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover
            print(f"torch.compile unavailable ({exc}); continuing without compilation")

    model = model.to(device)

    if use_ranger and Ranger is None:
        print("torch_optimizer.Ranger unavailable; falling back to AdamW")
    use_ranger = use_ranger and Ranger is not None
    if use_ranger:
        optimizer = Ranger(model.parameters(), lr=lr, weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    amp_enabled = use_amp and device.type in ("cuda", "mps")
    amp_dtype = torch.bfloat16 if device.type == "mps" else torch.float16
    scaler = torch.cuda.amp.GradScaler() if amp_enabled and device.type == "cuda" else None

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    patience_counter = 0

    mse = nn.MSELoss()

    overall_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
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
            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled)
                if amp_enabled
                else nullcontext()
            )
            with autocast_ctx:
                outputs = model(xb, sb if sb.numel() else None)

                y_res_t = transform_target(y_res)
                y_usgs_t = transform_target(y_usgs)

                pred_res_t = outputs["residual_mean"]
                logvar_res = outputs["residual_logvar"]
                loss_res = gaussian_nll(pred_res_t, logvar_res, y_res_t).mean()

                pred_corr_t = outputs["corrected_mean"]
                logvar_corr = outputs["corrected_logvar"]
                loss_corr = gaussian_nll(pred_corr_t, logvar_corr, y_usgs_t).mean()

                pred_res_raw = inverse_transform(outputs["residual_mean"])
                pred_corr_raw = inverse_transform(outputs["corrected_mean"])

                corr_from_res = y_nwm + pred_res_raw
                corrected_raw = torch.stack([pred_corr_raw, corr_from_res], dim=0).mean(dim=0)

                loss_consistency = mse(pred_corr_raw, corr_from_res)

                loss = loss_res + loss_corr + 0.1 * loss_consistency

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

        val_loss = float("nan")
        epoch_time = time.time() - epoch_start
        avg_epoch_time = (time.time() - overall_start) / (epoch + 1)
        eta = max(avg_epoch_time * (epochs - (epoch + 1)), 0.0)

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

                    outputs = model(xb, sb if sb.numel() else None)
                    y_res_t = transform_target(y_res)
                    y_usgs_t = transform_target(y_usgs)
                    pred_res_t = outputs["residual_mean"]
                    logvar_res = outputs["residual_logvar"]
                    pred_corr_t = outputs["corrected_mean"]
                    logvar_corr = outputs["corrected_logvar"]

                    loss_res = gaussian_nll(pred_res_t, logvar_res, y_res_t).mean()
                    loss_corr = gaussian_nll(pred_corr_t, logvar_corr, y_usgs_t).mean()
                    pred_res_raw = inverse_transform(outputs["residual_mean"])
                    pred_corr_raw = inverse_transform(outputs["corrected_mean"])
                    corr_from_res = y_nwm + pred_res_raw
                    loss_consistency = mse(pred_corr_raw, corr_from_res)
                    loss = loss_res + loss_corr + 0.1 * loss_consistency

                    val_running += loss.item() * len(xb)
                    val_samples += len(xb)

            val_loss = val_running / max(val_samples, 1)
            print(
                f"Epoch {epoch + 1}/{epochs} | train loss: {train_loss:.4f} | "
                f"val loss: {val_loss:.4f} | epoch time: {epoch_time:.1f}s | "
                f"ETA: {eta/60:.1f} min"
            )

            if use_ranger:
                scheduler.step(val_loss)
            else:
                scheduler.step()

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
            print(
                f"Epoch {epoch + 1}/{epochs} | train loss: {train_loss:.4f} | "
                f"epoch time: {epoch_time:.1f}s | ETA: {eta/60:.1f} min"
            )
            if use_ranger:
                scheduler.step(train_loss)
            else:
                scheduler.step()
            if train_loss + 1e-6 < best_val_loss:
                best_val_loss = train_loss
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()

    preds_res, preds_corr, targets_res, targets_usgs, targets_nwm = [], [], [], [], []
    with torch.no_grad():
        for xb, sb, y_res, y_usgs, y_nwm in test_loader:
            xb = xb.to(device)
            sb = sb.to(device)
            y_res = y_res.to(device)
            y_usgs = y_usgs.to(device)
            y_nwm = y_nwm.to(device)

            outputs = model(xb, sb if sb.numel() else None)
            pred_res_raw = inverse_transform(outputs["residual_mean"])
            pred_corr_raw = inverse_transform(outputs["corrected_mean"])
            corr_from_res = y_nwm + pred_res_raw
            corrected_raw = torch.stack([pred_corr_raw, corr_from_res], dim=0).mean(dim=0)

            preds_res.append(pred_res_raw.cpu().numpy())
            preds_corr.append(corrected_raw.cpu().numpy())
            targets_res.append(y_res.cpu().numpy())
            targets_usgs.append(y_usgs.cpu().numpy())
            targets_nwm.append(y_nwm.cpu().numpy())

    pred_residual = np.concatenate(preds_res)
    pred_corrected = np.concatenate(preds_corr)
    true_residual = np.concatenate(targets_res)
    true_usgs = np.concatenate(targets_usgs)
    nwm_vals = np.concatenate(targets_nwm)

    corrected_from_res = nwm_vals + pred_residual
    corrected_pred = pred_corrected

    baseline_metrics = compute_hydro_metrics(nwm_vals, true_usgs)
    corrected_metrics = compute_hydro_metrics(corrected_pred, true_usgs)

    mse_res = np.mean((pred_residual - true_residual) ** 2)
    rmse_res = float(np.sqrt(mse_res))

    base_rmse = baseline_metrics["rmse"]
    corr_rmse = corrected_metrics["rmse"]
    rel_improve = 100.0 * (base_rmse - corr_rmse) / base_rmse if base_rmse and base_rmse > 0 else float("nan")

    print("Baseline Metrics:")
    for k, v in baseline_metrics.items():
        print(f"  {k.upper()}: {v:.4f}")

    print("Corrected Metrics:")
    for k, v in corrected_metrics.items():
        print(f"  {k.upper()}: {v:.4f}")

    print(f"Residual RMSE: {rmse_res:.3f}")
    print(f"ΔRMSE% (Corrected vs Baseline): {rel_improve:.2f}%")

    out_dir = "data/clean/modeling"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{output_prefix}_pred_residual.npy"), pred_residual)
    np.save(os.path.join(out_dir, f"{output_prefix}_true_residual.npy"), true_residual)

    timestamps = df.loc[test_mask, "timestamp"].to_numpy()
    aligned_len = min(len(timestamps), len(corrected_pred))
    result_df = pd.DataFrame(
        {
            "timestamp": timestamps[:aligned_len],
            "y_true_residual_cms": true_residual[:aligned_len],
            "y_pred_residual_cms": pred_residual[:aligned_len],
            "nwm_cms": nwm_vals[:aligned_len],
            "usgs_cms": true_usgs[:aligned_len],
            "corrected_pred_cms": corrected_pred[:aligned_len],
            "corrected_true_cms": true_usgs[:aligned_len],
        }
    )
    result_df.to_csv(os.path.join(out_dir, f"{output_prefix}_eval.csv"), index=False)

    def _clean_dict(d: Dict[str, float]) -> Dict[str, float | None]:
        out: Dict[str, float | None] = {}
        for k, v in d.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out[k] = None
            else:
                out[k] = float(v)
        return out

    metrics_payload = {
        "baseline": _clean_dict(baseline_metrics),
        "corrected": _clean_dict(corrected_metrics),
        "rmse_residual": None if np.isnan(rmse_res) else float(rmse_res),
        "rmse_improvement_pct": None if np.isnan(rel_improve) else float(rel_improve),
        "output_prefix": output_prefix,
    }

    with open(os.path.join(out_dir, f"{output_prefix}_metrics.json"), "w") as fh:
        json.dump(metrics_payload, fh, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Parquet training data")
    parser.add_argument("--seq-len", type=int, default=168)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-days", type=int, default=28)
    parser.add_argument("--val-days", type=int, default=7)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--conv-depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-ranger", action="store_true")
    parser.add_argument("--output-prefix", default="hydra_v2")
    args = parser.parse_args()

    train_eval(
        data_path=args.data,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_days=args.train_days,
        val_days=args.val_days,
        patience=args.patience,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        conv_depth=args.conv_depth,
        dropout=args.dropout,
        lr=args.lr,
        use_amp=not args.no_amp,
        use_compile=not args.no_compile,
        augment=not args.no_augment,
        use_ranger=not args.no_ranger,
        output_prefix=args.output_prefix,
    )

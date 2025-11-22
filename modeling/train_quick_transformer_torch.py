#!/usr/bin/env python3

"""
Hydra temporal model training script (PyTorch) for NWM residual correction.
- Dynamic features: NWM + available ERA5 columns (normalized per train split)
- Static features: NLCD percentages, regulation flag, etc. (normalized)
- Targets: residual (USGS - NWM) and corrected flow (USGS)

Version 2 adds:
- Transformer encoder with attention pooling and multi-window statistics
- Heteroscedastic heads for residual and corrected flow predictions
- Multi-objective loss (Gaussian NLL, NSE surrogate, quantile pinball) on asinh-transformed targets
"""

import argparse
import copy
import json 
import math
import os
import random
import time
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from torch_optimizer import Ranger
except ImportError:  # pragma: no cover - optional dependency
    Ranger = None

from modeling.models.hydra_temporal import HydraTemporalModel as HydraTemporalV2
from modeling.models.hydra_temporal_v1 import HydraTemporalModel as HydraTemporalV1

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


def _has_gaussian_heads(outputs: Dict[str, torch.Tensor]) -> bool:
    required = {"residual_mean", "residual_logvar", "corrected_logvar"}
    return required.issubset(outputs.keys())


def _unpack_predictions(outputs: Dict[str, torch.Tensor], y_nwm: torch.Tensor) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    bool,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    has_gaussian = _has_gaussian_heads(outputs)
    if has_gaussian:
        pred_res_t = outputs["residual_mean"]
        pred_res_raw = inverse_transform(pred_res_t)
        logvar_res = outputs["residual_logvar"]
        logvar_corr = outputs["corrected_logvar"]
    elif "residual" in outputs:
        pred_res_raw = outputs["residual"]
        pred_res_t = transform_target(pred_res_raw)
        logvar_res = None
        logvar_corr = None
    else:
        raise KeyError("Model outputs must contain residual predictions")

    corr_from_res = outputs.get("corrected", y_nwm + pred_res_raw)
    pred_corr_t = transform_target(corr_from_res)
    return pred_res_raw, pred_res_t, corr_from_res, pred_corr_t, has_gaussian, logvar_res, logvar_corr


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


def flow_weighting(flows: torch.Tensor, emphasis: float) -> torch.Tensor:
    if emphasis <= 0:
        return torch.ones_like(flows)
    baseline = torch.mean(torch.abs(flows)) + EPS
    return 1.0 + emphasis * torch.abs(flows) / baseline


def nse_surrogate(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    denom = torch.sum((target - target.mean()) ** 2) + EPS
    return torch.sum((pred - target) ** 2) / denom


def kge_stabilizer(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    pred_std = torch.sqrt(torch.mean(pred_centered**2) + EPS)
    target_std = torch.sqrt(torch.mean(target_centered**2) + EPS)
    cov = torch.mean(pred_centered * target_centered)
    corr = cov / (pred_std * target_std + EPS)
    alpha = pred_std / (target_std + EPS)
    beta = pred_mean / (target_mean + EPS)
    return (corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2


def quantile_pinball(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    diff = target.unsqueeze(1) - pred
    loss = torch.where(diff >= 0, quantiles * diff, (quantiles - 1.0) * diff)
    return torch.mean(loss * weights)


def train_eval(
    data_path: str,
    seq_len: int = 168,
    epochs: int = 40,
    batch_size: int = 64,
    train_days: int = 28,
    val_days: int = 7,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
    patience: int = 5,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    conv_depth: int = 4,
    dropout: float = 0.1,
    lr: float = 5e-4,
    quantiles: Optional[List[float]] = None,
    quantile_weights: Optional[List[float]] = None,
    weight_nse: float = 0.0,
    weight_quantile: float = 0.0,
    flow_emphasis: float = 0.0,
    consistency_weight: float = 1.0,
    weight_pbias: float = 0.0,
    weight_pbias_final: Optional[float] = None,
    residual_bias_weight: float = 0.0,
    bias_shift_alpha: float = 0.0,
    bias_shift_pbias_target: float = 5.0,
    bias_shift_qmin: float = 0.0,
    bias_shift_qmax: float = 1.0,
    bias_shift_weight_power: float = 0.0,
    bias_shift_strategy: str = "scaled",
    weight_kge: float = 0.0,
    weight_kge_final: Optional[float] = None,
    use_amp: bool = True,
    use_compile: bool = False,
    augment: bool = True,
    use_ranger: bool = True,
    output_prefix: str = "hydra_v2",
    seed: Optional[int] = None,
    model_arch: str = "hydra_v2",
    patch_size: int = 14,
) -> None:
    torch.set_float32_matmul_precision("medium")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    quantiles = list(quantiles) if quantiles else []
    if quantiles and quantile_weights and len(quantile_weights) != len(quantiles):
        raise ValueError("quantile_weights must match the number of quantile levels")
    if quantiles and not quantile_weights:
        quantile_weights = [1.0 for _ in quantiles]
    quantile_weights = list(quantile_weights) if quantile_weights else []
    def _scheduled_weight(start: float, final: Optional[float], progress: float) -> float:
        if final is None:
            return start
        return float(start + (final - start) * progress)
    df = load_data(data_path)
    site_cols = [c for c in ("site_name", "comid") if c in df.columns]
    if site_cols:
        for col in site_cols:
            if df[col].nunique() > 1:
                unique_vals = df[col].nunique()
                raise ValueError(
                    f"Dataset contains {unique_vals} unique values in '{col}'. "
                    "Train each gauge/site separately to avoid sequence leakage."
                )
    if all(ts is not None for ts in (train_start, train_end)):
        train_start_ts = pd.Timestamp(train_start)
        train_end_ts = pd.Timestamp(train_end)
    else:
        start = df["timestamp"].min()
        train_start_ts = start
        train_end_ts = start + pd.Timedelta(days=train_days)

    train_mask = (df["timestamp"] >= train_start_ts) & (df["timestamp"] <= train_end_ts)
    if train_mask.sum() <= seq_len:
        raise ValueError("Training window shorter than sequence length")

    if val_start is not None and val_end is not None:
        val_start_ts = pd.Timestamp(val_start)
        val_end_ts = pd.Timestamp(val_end)
        val_mask = (df["timestamp"] >= val_start_ts) & (df["timestamp"] <= val_end_ts)
        test_mask = df["timestamp"] > val_end_ts
    else:
        if val_days > 0:
            val_start_ts = train_end_ts
            val_end_ts = train_end_ts + pd.Timedelta(days=val_days)
            val_mask = (df["timestamp"] >= val_start_ts) & (df["timestamp"] < val_end_ts)
            test_mask = df["timestamp"] >= val_end_ts
        else:
            val_mask = pd.Series(False, index=df.index)
            test_mask = ~train_mask
    if test_mask.sum() == 0:
        raise ValueError("No evaluation rows available after split")

    dynamic_cols = ["nwm_cms"] + [c for c in ERA5_CANDIDATES if c in df.columns]
    if not dynamic_cols or dynamic_cols[0] != "nwm_cms":
        raise ValueError("First dynamic feature must be 'nwm_cms'")
    if len(dynamic_cols) <= 1:
        raise ValueError(
            "Dynamic feature set contains only 'nwm_cms'. "
            "Ensure ERA5/meteorological columns are present in the parquet."
        )

    static_cols = add_static_columns(df)
    if not static_cols:
        raise ValueError(
            "No static columns detected (e.g., NLCD percentages). "
            "Rebuild the dataset with static land-cover features to proceed."
        )

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

    arch = model_arch.lower()
    if arch == "hydra_v2":
        model = HydraTemporalV2(
            input_dim=len(dynamic_cols),
            static_dim=len(static_cols),
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            seq_len=seq_len,
            conv_depth=conv_depth,
            dropout=dropout,
            quantiles=quantiles if quantiles else None,
            nwm_index=0,
            patch_size=1,
        )
    elif arch == "hydra_v1":
        model = HydraTemporalV1(
            input_dim=len(dynamic_cols),
            static_dim=len(static_cols),
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            seq_len=seq_len,
            conv_depth=conv_depth,
            dropout=dropout,
            quantiles=quantiles if quantiles else None,
            nwm_index=0,
            patch_size=max(1, patch_size),
        )
    else:
        raise ValueError(f"Unsupported model_arch '{model_arch}'. Choose 'hydra_v2' or 'hydra_v1'.")

    if use_compile:
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover
            print(f"torch.compile unavailable ({exc}); continuing without compilation")

    model = model.to(device)
    quantiles_tensor = torch.tensor(quantiles, device=device, dtype=torch.float32) if quantiles else None
    quantile_weight_tensor = (
        torch.tensor(quantile_weights, device=device, dtype=torch.float32) if quantiles else None
    )
    if quantile_weight_tensor is not None:
        quantile_weight_tensor = quantile_weight_tensor / (quantile_weight_tensor.sum() + EPS)

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
    num_train_batches = len(train_loader)
    progress_stride = max(1, num_train_batches // 5)

    mse = nn.MSELoss()

    overall_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        progress = epoch / max(epochs - 1, 1)
        curr_weight_pbias = _scheduled_weight(weight_pbias, weight_pbias_final, progress)
        curr_weight_kge = _scheduled_weight(weight_kge, weight_kge_final, progress)
        model.train()
        running_loss = 0.0
        sample_count = 0
        print(f"Epoch {epoch + 1}/{epochs} - training start ({num_train_batches} batches)")
        for batch_idx, (xb, sb, y_res, y_usgs, y_nwm) in enumerate(train_loader):
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

                (
                    pred_res_raw,
                    pred_res_t,
                    corr_from_res,
                    pred_corr_t,
                    has_gaussian_heads,
                    logvar_res,
                    logvar_corr,
                ) = _unpack_predictions(outputs, y_nwm)

                corr_weights = flow_weighting(y_usgs, flow_emphasis)
                if has_gaussian_heads and logvar_res is not None and logvar_corr is not None:
                    loss_res = gaussian_nll(pred_res_t, logvar_res, y_res_t).mean()
                    corr_nll = gaussian_nll(pred_corr_t, logvar_corr, y_usgs_t)
                    loss_corr = (corr_nll * corr_weights).mean()
                else:
                    loss_res = mse(pred_res_t, y_res_t)
                    corr_mse = (pred_corr_t - y_usgs_t) ** 2
                    loss_corr = (corr_mse * corr_weights).mean()

                loss = loss_res + loss_corr + consistency_weight * mse(pred_corr_t, y_usgs_t)

                if curr_weight_pbias > 0:
                    numer = torch.sum(corr_from_res - y_usgs)
                    denom = torch.sum(y_usgs)
                    pbias_percent = torch.where(
                        torch.abs(denom) > EPS,
                        100.0 * numer / denom,
                        torch.zeros_like(denom),
                    )
                    pbias_penalty = torch.abs(pbias_percent) / 100.0
                    loss = loss + curr_weight_pbias * pbias_penalty

                if weight_nse > 0:
                    loss = loss + weight_nse * nse_surrogate(corr_from_res, y_usgs)

                if curr_weight_kge > 0:
                    loss = loss + curr_weight_kge * kge_stabilizer(corr_from_res, y_usgs)

                if residual_bias_weight > 0:
                    mean_bias = torch.mean(corr_from_res - y_usgs)
                    norm = torch.mean(torch.abs(y_usgs)) + EPS
                    loss = loss + residual_bias_weight * (mean_bias / norm) ** 2

                if (
                    weight_quantile > 0
                    and quantiles_tensor is not None
                    and quantile_weight_tensor is not None
                    and "quantiles" in outputs
                ):
                    q_pred_raw = inverse_transform(outputs["quantiles"])
                    quantile_loss_val = quantile_pinball(q_pred_raw, y_usgs, quantiles_tensor, quantile_weight_tensor)
                    loss = loss + weight_quantile * quantile_loss_val

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

            if (batch_idx + 1) % progress_stride == 0 or batch_idx + 1 == num_train_batches:
                elapsed = time.time() - epoch_start
                frac = (batch_idx + 1) / max(num_train_batches, 1)
                est_epoch_time = elapsed / max(frac, 1e-6)
                remaining_epoch = max(est_epoch_time - elapsed, 0.0)
                print(
                    f"  Batch {batch_idx + 1}/{num_train_batches} ({frac * 100:.0f}%) | "
                    f"elapsed {elapsed:.1f}s | est epoch {est_epoch_time:.1f}s | "
                    f"epoch remaining {remaining_epoch:.1f}s"
                )

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
                    (
                        pred_res_raw,
                        pred_res_t,
                        corr_from_res,
                        pred_corr_t,
                        has_gaussian_heads,
                        logvar_res,
                        logvar_corr,
                    ) = _unpack_predictions(outputs, y_nwm)

                    corr_weights = flow_weighting(y_usgs, flow_emphasis)
                    if has_gaussian_heads and logvar_res is not None and logvar_corr is not None:
                        loss_res = gaussian_nll(pred_res_t, logvar_res, y_res_t).mean()
                        corr_nll = gaussian_nll(pred_corr_t, logvar_corr, y_usgs_t)
                        loss_corr = (corr_nll * corr_weights).mean()
                    else:
                        loss_res = mse(pred_res_t, y_res_t)
                        corr_mse = (pred_corr_t - y_usgs_t) ** 2
                        loss_corr = (corr_mse * corr_weights).mean()

                    loss = loss_res + loss_corr + consistency_weight * mse(pred_corr_t, y_usgs_t)
                    if curr_weight_pbias > 0:
                        numer = torch.sum(corr_from_res - y_usgs)
                        denom = torch.sum(y_usgs)
                        pbias_percent = torch.where(
                            torch.abs(denom) > EPS,
                            100.0 * numer / denom,
                            torch.zeros_like(denom),
                        )
                        pbias_penalty = torch.abs(pbias_percent) / 100.0
                        loss = loss + curr_weight_pbias * pbias_penalty
                    if weight_nse > 0:
                        loss = loss + weight_nse * nse_surrogate(corr_from_res, y_usgs)
                    if curr_weight_kge > 0:
                        loss = loss + curr_weight_kge * kge_stabilizer(corr_from_res, y_usgs)
                    if residual_bias_weight > 0:
                        mean_bias = torch.mean(corr_from_res - y_usgs)
                        norm = torch.mean(torch.abs(y_usgs)) + EPS
                        loss = loss + residual_bias_weight * (mean_bias / norm) ** 2
                    if (
                        weight_quantile > 0
                        and quantiles_tensor is not None
                        and quantile_weight_tensor is not None
                        and "quantiles" in outputs
                    ):
                        q_pred_raw = inverse_transform(outputs["quantiles"])
                        loss = loss + weight_quantile * quantile_pinball(
                            q_pred_raw, y_usgs, quantiles_tensor, quantile_weight_tensor
                        )

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
    total_training_time = time.time() - overall_start
    print(f"Training complete in {total_training_time/60:.2f} minutes")

    bias_shift_info = None
    want_bias_shift = (
        bias_shift_strategy in {"scaled", "full"}
        and (bias_shift_alpha > 0 or bias_shift_strategy == "full")
        and val_loader is not None
    )
    bias_shift_supported = hasattr(model, "residual_bias")
    if want_bias_shift and not bias_shift_supported:
        print("Skipping residual bias shift: current model lacks residual_bias parameter.")
        bias_shift_info = {
            "strategy": bias_shift_strategy,
            "alpha": float(bias_shift_alpha),
            "status": "unsupported",
            "applied": 0.0,
        }
    elif want_bias_shift and bias_shift_supported:
        bias_diffs: List[np.ndarray] = []
        flow_samples: List[np.ndarray] = []
        with torch.no_grad():
            for xb, sb, y_res, y_usgs, y_nwm in val_loader:
                xb = xb.to(device)
                sb = sb.to(device)
                y_res = y_res.to(device)
                y_usgs = y_usgs.to(device)
                y_nwm = y_nwm.to(device)

                outputs = model(xb, sb if sb.numel() else None)
                corr_from_res = _unpack_predictions(outputs, y_nwm)[2]
                diff = (corr_from_res - y_usgs).detach().cpu().numpy()
                flows = y_usgs.detach().cpu().numpy()
                bias_diffs.append(diff)
                flow_samples.append(flows)

        if bias_diffs:
            diff_vec = np.concatenate(bias_diffs)
            flow_vec = np.concatenate(flow_samples)
            qmin = float(np.clip(bias_shift_qmin, 0.0, 1.0))
            qmax = float(np.clip(bias_shift_qmax, 0.0, 1.0))
            if qmax < qmin:
                qmin, qmax = qmax, qmin
            mask = np.ones_like(flow_vec, dtype=bool)
            if qmin > 0.0 or qmax < 1.0:
                low = np.quantile(flow_vec, qmin)
                high = np.quantile(flow_vec, qmax)
                mask = (flow_vec >= low) & (flow_vec <= high)
            sample_count = int(np.count_nonzero(mask))
            bias_shift_info = {
                "strategy": bias_shift_strategy,
                "alpha": float(bias_shift_alpha),
                "qmin": qmin,
                "qmax": qmax,
                "weight_power": float(bias_shift_weight_power),
                "applied": 0.0,
                "mean_bias": None,
                "pbias_percent": None,
                "scale": None,
                "samples": sample_count,
                "status": "skipped",
            }
            if np.any(mask):
                weights = np.ones_like(diff_vec, dtype=np.float64)
                if abs(bias_shift_weight_power) > 1e-8:
                    # Optionally emphasise/discount flows by magnitude before measuring mean bias.
                    weights = (np.abs(flow_vec) + EPS) ** bias_shift_weight_power
                weights = weights[mask]
                bias_slice = diff_vec[mask]
                flow_slice = flow_vec[mask]
                weight_sum = np.sum(weights)
                if weight_sum > 0 and bias_slice.size > 0:
                    mean_bias = float(np.sum(bias_slice * weights) / weight_sum)
                    denom = float(np.sum(flow_slice * weights))
                    if abs(denom) < EPS:
                        pbias_percent = None
                        scale = 1.0
                    else:
                        pbias_percent = 100.0 * float(np.sum(bias_slice * weights) / denom)
                        target = max(bias_shift_pbias_target, 1e-3)
                        scale = float(np.clip(abs(pbias_percent) / target, 0.2, 1.25))
                    if bias_shift_strategy == "full":
                        shift = mean_bias * scale
                    else:
                        shift = bias_shift_alpha * scale * mean_bias
                    model.residual_bias.data -= torch.tensor(shift, device=model.residual_bias.data.device)
                    bias_shift_info.update(
                        {
                            "applied": float(shift),
                            "mean_bias": float(mean_bias),
                            "pbias_percent": None if pbias_percent is None else float(pbias_percent),
                            "scale": float(scale),
                            "status": "applied",
                        }
                    )
                    print(
                        "Applied residual bias shift: "
                        f"{shift:+.4f} (strategy={bias_shift_strategy}, alpha={bias_shift_alpha}, "
                        f"scale={scale:.2f}, q=[{qmin:.2f},{qmax:.2f}], weight_pow={bias_shift_weight_power:.2f})"
                    )
                else:
                    print("Skipped residual bias shift (insufficient weighted samples).")
            else:
                print("Skipped residual bias shift (no validation samples within specified quantiles).")

    preds_res, preds_corr, targets_res, targets_usgs, targets_nwm = [], [], [], [], []
    preds_quantiles = [] if quantiles else None
    with torch.no_grad():
        for xb, sb, y_res, y_usgs, y_nwm in test_loader:
            xb = xb.to(device)
            sb = sb.to(device)
            y_res = y_res.to(device)
            y_usgs = y_usgs.to(device)
            y_nwm = y_nwm.to(device)

            outputs = model(xb, sb if sb.numel() else None)
            pred_res_raw, _, corr_from_res, _, _, _, _ = _unpack_predictions(outputs, y_nwm)
            corrected_raw = corr_from_res

            preds_res.append(pred_res_raw.cpu().numpy())
            preds_corr.append(corrected_raw.cpu().numpy())
            targets_res.append(y_res.cpu().numpy())
            targets_usgs.append(y_usgs.cpu().numpy())
            targets_nwm.append(y_nwm.cpu().numpy())
            if preds_quantiles is not None and "quantiles" in outputs:
                preds_quantiles.append(inverse_transform(outputs["quantiles"]).cpu().numpy())

    pred_residual = np.concatenate(preds_res)
    pred_corrected = np.concatenate(preds_corr)
    true_residual = np.concatenate(targets_res)
    true_usgs = np.concatenate(targets_usgs)
    nwm_vals = np.concatenate(targets_nwm)

    corrected_from_res = nwm_vals + pred_residual
    corrected_pred = pred_corrected

    baseline_metrics = compute_hydro_metrics(nwm_vals, true_usgs)
    corrected_metrics = compute_hydro_metrics(corrected_pred, true_usgs)

    quantile_metrics: Dict[str, Dict[str, float]] = {}
    if preds_quantiles:
        q_pred = np.concatenate(preds_quantiles)
        for idx, q in enumerate(quantiles):
            coverage = float(np.mean(true_usgs <= q_pred[:, idx]))
            bias = float(np.mean(q_pred[:, idx] - true_usgs))
            quantile_metrics[f"{q:.2f}"] = {"coverage": coverage, "bias": bias}

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

    if bias_shift_info is None:
        bias_shift_info = {
            "strategy": bias_shift_strategy,
            "alpha": float(bias_shift_alpha),
            "qmin": float(np.clip(bias_shift_qmin, 0.0, 1.0)),
            "qmax": float(np.clip(bias_shift_qmax, 0.0, 1.0)),
            "weight_power": float(bias_shift_weight_power),
            "applied": 0.0,
            "mean_bias": None,
            "pbias_percent": None,
            "scale": None,
            "samples": 0,
            "status": "no_validation" if val_loader is None else "not_applied",
        }

    metrics_payload = {
        "baseline": _clean_dict(baseline_metrics),
        "corrected": _clean_dict(corrected_metrics),
        "rmse_residual": None if np.isnan(rmse_res) else float(rmse_res),
        "rmse_improvement_pct": None if np.isnan(rel_improve) else float(rel_improve),
        "output_prefix": output_prefix,
    }
    if quantile_metrics:
        metrics_payload["quantiles"] = quantile_metrics
    metrics_payload["bias_shift"] = bias_shift_info

    with open(os.path.join(out_dir, f"{output_prefix}_metrics.json"), "w") as fh:
        json.dump(metrics_payload, fh, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Parquet training data")
    parser.add_argument("--seq-len", type=int, default=168)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-days", type=int, default=4018)
    parser.add_argument("--val-days", type=int, default=365)
    parser.add_argument("--train-start", default=None, help="Explicit train window start (YYYY-MM-DD)")
    parser.add_argument("--train-end", default=None, help="Explicit train window end (YYYY-MM-DD)")
    parser.add_argument("--val-start", default=None, help="Explicit validation window start (YYYY-MM-DD)")
    parser.add_argument("--val-end", default=None, help="Explicit validation window end (YYYY-MM-DD)")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--conv-depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--quantiles", default="0.1,0.5,0.9", help="Comma-separated quantile levels")
    parser.add_argument(
        "--quantile-weights",
        default="0.3,1.0,0.3",
        help="Comma-separated weights for quantile pinball loss (align with quantiles)",
    )
    parser.add_argument("--weight-nse", type=float, default=0.2, help="Weight for NSE surrogate loss")
    parser.add_argument("--weight-quantile", type=float, default=0.1, help="Weight for quantile pinball loss")
    parser.add_argument("--flow-emphasis", type=float, default=0.3, help="Emphasis factor for high-flow weighting")
    parser.add_argument("--consistency-weight", type=float, default=1.0, help="Weight tying corrected predictions to nwm + residual")
    parser.add_argument(
        "--weight-pbias",
        type=float,
        default=0.05,
        help="Weight for absolute percent-bias penalty (matches reported PBIAS metric)",
    )
    parser.add_argument(
        "--weight-pbias-final",
        type=float,
        default=None,
        help="Optional final percent-bias weight; if set, interpolates from --weight-pbias to this value over epochs",
    )
    parser.add_argument("--residual-bias-weight", type=float, default=0.0, help="Weight for squared mean bias penalty")
    parser.add_argument("--bias-shift-alpha", type=float, default=0.0, help="Post-training bias correction factor (0=off)")
    parser.add_argument(
        "--bias-shift-pbias-target",
        type=float,
        default=5.0,
        help="Target percent bias magnitude (validation) used to scale adaptive residual shifts",
    )
    parser.add_argument(
        "--bias-shift-qmin",
        type=float,
        default=0.0,
        help="Lower quantile bound (0-1) of observed flows used to compute bias shift (default uses all samples).",
    )
    parser.add_argument(
        "--bias-shift-qmax",
        type=float,
        default=1.0,
        help="Upper quantile bound (0-1) of observed flows used to compute bias shift (default uses all samples).",
    )
    parser.add_argument(
        "--bias-shift-weight-power",
        type=float,
        default=0.0,
        help="Exponent applied to |flow| when weighting samples for bias shift; negative values emphasize low flows.",
    )
    parser.add_argument(
        "--bias-shift-strategy",
        choices=["scaled", "full"],
        default="scaled",
        help="Selects how the validation bias is converted to a shift: 'scaled' multiplies by alpha; 'full' removes the entire measured bias.",
    )
    parser.add_argument(
        "--weight-kge",
        type=float,
        default=0.05,
        help="Weight for gentle KGE stabilisation (0 disables)",
    )
    parser.add_argument(
        "--weight-kge-final",
        type=float,
        default=None,
        help="Optional final KGE weight; if set, interpolates from --weight-kge to this value over epochs",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-ranger", action="store_true")
    parser.add_argument("--output-prefix", default="hydra_v2")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--model-arch",
        choices=["hydra_v2", "hydra_v1"],
        default="hydra_v2",
        help="Selects the Hydra architecture (legacy v2 vs. hybrid v1 prototype).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=14,
        help="Temporal patch size used for hydra_v1 patch embeddings (ignored for hydra_v2).",
    )
    args = parser.parse_args()
    quantiles = [float(x) for x in args.quantiles.split(",") if x.strip()]
    quantile_weights = [float(x) for x in args.quantile_weights.split(",") if x.strip()]
    if not quantiles:
        quantile_weights = []

    train_eval(
        data_path=args.data,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_days=args.train_days,
        val_days=args.val_days,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        patience=args.patience,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        conv_depth=args.conv_depth,
        dropout=args.dropout,
        lr=args.lr,
        quantiles=quantiles,
        quantile_weights=quantile_weights,
        weight_nse=args.weight_nse,
        weight_quantile=args.weight_quantile,
        flow_emphasis=args.flow_emphasis,
        consistency_weight=args.consistency_weight,
        weight_pbias=args.weight_pbias,
        weight_pbias_final=args.weight_pbias_final,
        residual_bias_weight=args.residual_bias_weight,
        bias_shift_alpha=args.bias_shift_alpha,
        bias_shift_pbias_target=args.bias_shift_pbias_target,
        bias_shift_qmin=args.bias_shift_qmin,
        bias_shift_qmax=args.bias_shift_qmax,
        bias_shift_weight_power=args.bias_shift_weight_power,
        bias_shift_strategy=args.bias_shift_strategy,
        weight_kge=args.weight_kge,
        weight_kge_final=args.weight_kge_final,
        use_amp=not args.no_amp,
        use_compile=not args.no_compile,
        augment=not args.no_augment,
        use_ranger=not args.no_ranger,
        output_prefix=args.output_prefix,
        seed=args.seed,
        model_arch=args.model_arch,
        patch_size=args.patch_size,
    )

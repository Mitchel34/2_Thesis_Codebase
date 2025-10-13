#!/usr/bin/env python3
"""Generate publication-ready plots for hourly residual-correction experiments.

Prototype usage (Watauga quick run)::

    python -m modeling.plot_suite \
        --model hydra=data/clean/modeling/hydra_v2_quick_eval.csv \
        --model lstm=data/clean/modeling/quick_lstm_eval.csv \
        --out-dir results/figures/watauga_quick \
        --station "Watauga River, NC"

The input evaluation CSVs must include ``timestamp``, ``nwm_cms`` and either
``corrected_true_cms`` or ``usgs_cms`` plus ``corrected_pred_cms`` (matching the
existing training scripts). The script emits a hydrograph, skill summary bar
chart, residual density plot, and scatter density comparison. Additional plot
hooks can be added for PIT histograms, flow duration curves, etc.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

cache_root = os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))
os.makedirs(cache_root, exist_ok=True)
fontconfig_dir = os.path.join(cache_root, "fontconfig")
os.makedirs(fontconfig_dir, exist_ok=True)
mpl_cache_dir = os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
os.makedirs(mpl_cache_dir, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.dates import DateFormatter

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")

from modeling.generate_model_improvement_plots import (
    rmse,
    mae,
    nse,
    pbias,
    corr_coef,
)

MetricFunc = Dict[str, callable]
METRIC_FUNCS: MetricFunc = {
    "RMSE": rmse,
    "MAE": mae,
    "NSE": nse,
    "PBIAS": pbias,
    "R": corr_coef,
}
PLOT_METRICS: Iterable[str] = ("RMSE", "MAE", "NSE", "PBIAS", "R")


@dataclass
class ModelEval:
    label: str
    data: pd.DataFrame

    @property
    def truth(self) -> np.ndarray:
        return self.data["truth_cms"].to_numpy(dtype=float)

    @property
    def preds(self) -> np.ndarray:
        return self.data["prediction_cms"].to_numpy(dtype=float)


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "nwm_cms", "corrected_pred_cms"}
    if "corrected_true_cms" not in df.columns:
        if "usgs_cms" in df.columns:
            df = df.rename(columns={"usgs_cms": "corrected_true_cms"})
        else:
            missing = required.difference(df.columns)
            raise ValueError(f"Evaluation CSV missing required columns: {sorted(missing)}")
    required.add("corrected_true_cms")
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Evaluation CSV missing required columns: {sorted(missing)}")
    return df


def load_model_eval(path: str, label: str) -> ModelEval:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Evaluation CSV not found: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = _ensure_columns(df)
    df = df.sort_values("timestamp").reset_index(drop=True)

    renamed = df.rename(
        columns={
            "corrected_pred_cms": "prediction_cms",
            "corrected_true_cms": "truth_cms",
        }
    )
    renamed["timestamp"] = pd.to_datetime(renamed["timestamp"], utc=False)
    return ModelEval(label=label, data=renamed[["timestamp", "truth_cms", "prediction_cms", "nwm_cms"]])


def compute_skill(truth: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {name: func(truth, pred) for name, func in METRIC_FUNCS.items()}


def consolidate_baseline(model_evals: List[ModelEval]) -> pd.DataFrame:
    if not model_evals:
        raise ValueError("At least one model evaluation is required")
    base = model_evals[0].data[["timestamp", "truth_cms", "nwm_cms"]].copy()
    base = base.rename(columns={"nwm_cms": "nwm_baseline"})
    return base


def align_models(model_evals: List[ModelEval]) -> pd.DataFrame:
    base = consolidate_baseline(model_evals)
    frames = []
    for model in model_evals:
        df = model.data.merge(base, on=["timestamp", "truth_cms"], how="inner")
        df["model"] = model.label
        df = df.rename(columns={"prediction_cms": "prediction"})
        frames.append(df[["timestamp", "truth_cms", "nwm_baseline", "prediction", "model"]])
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "model"])


def _spans_multiple_months(merged: pd.DataFrame) -> bool:
    if merged.empty:
        return False
    periods = pd.to_datetime(merged["timestamp"]).dt.to_period("M")
    return periods.nunique() > 1


def plot_hydrograph(merged: pd.DataFrame, out_path: str, station: Optional[str] = None,
                    start: Optional[pd.Timestamp] = None,
                    end: Optional[pd.Timestamp] = None) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    subset = merged
    if start is not None:
        subset = subset[subset["timestamp"] >= start]
    if end is not None:
        subset = subset[subset["timestamp"] <= end]
    subset = subset.sort_values("timestamp")

    truth = subset.drop_duplicates("timestamp")
    ax.plot(truth["timestamp"], truth["truth_cms"], label="USGS (truth)", color="black", linewidth=1.8)
    ax.plot(truth["timestamp"], truth["nwm_baseline"], label="NWM baseline", color="tab:blue", alpha=0.75)

    for model_label, group in subset.groupby("model"):
        ax.plot(group["timestamp"], group["prediction"], label=model_label, linewidth=1.2)

    ax.set_ylabel("Discharge (cms)")
    ax.set_xlabel("Timestamp (UTC)")
    title = "Hydrograph"
    if station:
        title += f" — {station}"
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=2)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_skill_bars(merged: pd.DataFrame, out_path: str) -> None:
    truth = merged.drop_duplicates("timestamp")["truth_cms"].to_numpy(dtype=float)
    baseline = merged.drop_duplicates("timestamp")["nwm_baseline"].to_numpy(dtype=float)
    baseline_metrics = compute_skill(truth, baseline)

    rows = [{"model": "NWM baseline", **baseline_metrics}]
    for model_label, group in merged.groupby("model"):
        metrics = compute_skill(group["truth_cms"].to_numpy(dtype=float), group["prediction"].to_numpy(dtype=float))
        rows.append({"model": model_label, **metrics})
    summary = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(summary["model"]))
    width = 0.15
    offsets = np.linspace(-width * (len(PLOT_METRICS) - 1) / 2, width * (len(PLOT_METRICS) - 1) / 2, len(PLOT_METRICS))

    for metric, offset in zip(PLOT_METRICS, offsets):
        ax.bar(x + offset, summary[metric], width=width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], rotation=15)
    ax.set_ylabel("Metric value")
    ax.set_title("Skill summary")
    ax.legend(ncols=len(PLOT_METRICS), fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_residual_density(merged: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    truth = merged.drop_duplicates("timestamp")
    base_resid = truth["nwm_baseline"].to_numpy(dtype=float) - truth["truth_cms"].to_numpy(dtype=float)
    ax.hist(base_resid, bins=40, alpha=0.4, label="NWM baseline", density=True)
    for model_label, group in merged.groupby("model"):
        resid = group["prediction"].to_numpy(dtype=float) - group["truth_cms"].to_numpy(dtype=float)
        ax.hist(resid, bins=40, alpha=0.4, label=model_label, density=True)
    ax.set_xlabel("Residual (cms)")
    ax.set_ylabel("Density")
    ax.set_title("Residual distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_monthly_timeseries(merged: pd.DataFrame, out_path: str, station: Optional[str] = None) -> None:
    truth = merged.drop_duplicates("timestamp")[["timestamp", "truth_cms", "nwm_baseline"]]
    monthly = truth.set_index("timestamp").resample("MS").mean()

    for model_label, group in merged.groupby("model"):
        series = group.set_index("timestamp")["prediction"].resample("MS").mean()
        monthly[model_label] = series

    monthly = monthly.dropna(how="all")
    if monthly.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    idx = monthly.index
    if isinstance(idx, pd.PeriodIndex):
        idx = idx.to_timestamp()
    ax.plot(idx, monthly["truth_cms"], marker="o", label="USGS (truth)", color="black")
    ax.plot(idx, monthly["nwm_baseline"], marker="o", label="NWM baseline", color="tab:blue")
    for model_label in merged["model"].unique():
        if model_label in monthly.columns:
            ax.plot(idx, monthly[model_label], marker="o", label=model_label)

    ax.set_ylabel("Average discharge (cms)")
    ax.set_xlabel("Month")
    title = "Monthly mean discharge"
    if station:
        title += f" — {station}"
    ax.set_title(title)
    ax.legend(loc="best", ncol=2, fontsize="small")
    ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_scatter_density(merged: pd.DataFrame, out_path: str) -> None:
    models = list(merged["model"].unique())
    fig, axes = plt.subplots(1, len(models) + 1, figsize=(11, 4), sharex=True, sharey=True)

    truth = merged.drop_duplicates("timestamp")
    truth_vals = truth["truth_cms"].to_numpy(dtype=float)
    baseline_vals = truth["nwm_baseline"].to_numpy(dtype=float)
    all_preds = merged["prediction"].to_numpy(dtype=float)
    vmax = np.nanmax(np.concatenate([truth_vals, baseline_vals, all_preds]))
    vmax = max(float(vmax), 1.0)

    axes = np.atleast_1d(axes)
    axes[0].hexbin(truth_vals, baseline_vals, gridsize=40, cmap="Blues", norm=LogNorm())
    axes[0].plot([0, vmax], [0, vmax], "k--", linewidth=1)
    axes[0].set_title("NWM baseline")
    axes[0].set_xlabel("Observed (cms)")
    axes[0].set_ylabel("Predicted (cms)")

    for idx, model_label in enumerate(models, start=1):
        group = merged[merged["model"] == model_label]
        preds = group["prediction"].to_numpy(dtype=float)
        axes[idx].hexbin(group["truth_cms"], preds, gridsize=40, cmap="Oranges", norm=LogNorm())
        axes[idx].plot([0, vmax], [0, vmax], "k--", linewidth=1)
        axes[idx].set_title(model_label)
        axes[idx].set_xlabel("Observed (cms)")

    fig.suptitle("Observed vs. predicted (hexbin density)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_model_args(args: List[str]) -> List[ModelEval]:
    models: List[ModelEval] = []
    for entry in args:
        if "=" not in entry:
            raise ValueError(f"Model specification must be label=path, got: {entry}")
        label, path = entry.split("=", 1)
        models.append(load_model_eval(path, label))
    return models


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate enhanced plots for hourly residual correction")
    parser.add_argument("--model", dest="models", metavar="label=path", action="append", required=True,
                        help="Model label and evaluation CSV path (e.g., hydra=data/clean/modeling/hydra_v2_quick_eval.csv)")
    parser.add_argument("--out-dir", required=True, help="Directory for output figures")
    parser.add_argument("--station", default=None, help="Optional station/site name to annotate plots")
    parser.add_argument("--start", default=None, help="Optional start timestamp (ISO format)")
    parser.add_argument("--end", default=None, help="Optional end timestamp (ISO format)")
    parser.add_argument("--monthly", action="store_true", help="Generate monthly aggregated time-series plot when data span exceeds one month")
    return parser


def maybe_parse_ts(ts: Optional[str]) -> Optional[pd.Timestamp]:
    if ts is None:
        return None
    return pd.to_datetime(ts)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    model_evals = parse_model_args(args.models)
    merged = align_models(model_evals)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    start_ts = maybe_parse_ts(args.start)
    end_ts = maybe_parse_ts(args.end)

    plot_hydrograph(merged, os.path.join(out_dir, "hydrograph.png"), station=args.station,
                    start=start_ts, end=end_ts)
    plot_skill_bars(merged, os.path.join(out_dir, "skill_summary.png"))
    plot_residual_density(merged, os.path.join(out_dir, "residual_density.png"))
    plot_scatter_density(merged, os.path.join(out_dir, "scatter_density.png"))

    if args.monthly or _spans_multiple_months(merged):
        plot_monthly_timeseries(
            merged,
            os.path.join(out_dir, "monthly_timeseries.png"),
            station=args.station,
        )

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()

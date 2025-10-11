#!/usr/bin/env python3
"""Generate full-run diagnostics including quantile calibration plots."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

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

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")


REQUIRED_COLUMNS = {"timestamp", "nwm_cms", "corrected_pred_cms"}
QUANTILE_PREFIX = "corrected_q"


@dataclass
class CoverageResult:
    nominal: float
    empirical: float


@dataclass
class IntervalCoverageResult:
    alpha_low: float
    alpha_high: float
    nominal: float
    empirical: float


def load_evaluation_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Evaluation CSV not found: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    cols = set(df.columns)
    missing = REQUIRED_COLUMNS.difference(cols)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    if "corrected_true_cms" not in df.columns:
        if "usgs_cms" in df.columns:
            df["corrected_true_cms"] = df["usgs_cms"]
        else:
            raise ValueError("CSV requires 'corrected_true_cms' or 'usgs_cms'.")
    return df.sort_values("timestamp").reset_index(drop=True)


def discover_quantiles(df: pd.DataFrame) -> Dict[float, str]:
    mapping: Dict[float, str] = {}
    for col in df.columns:
        if not col.startswith(QUANTILE_PREFIX):
            continue
        suffix = col[len(QUANTILE_PREFIX) :]
        suffix = suffix.strip("_")
        if not suffix:
            continue
        try:
            value = float(suffix)
        except ValueError:
            continue
        alpha = value / 100.0 if value > 1.0 else value
        if 0.0 < alpha < 1.0:
            mapping[alpha] = col
    return dict(sorted(mapping.items()))


def _make_monotonic(arr: np.ndarray) -> np.ndarray:
    result = np.asarray(arr, dtype=float).copy()
    for i in range(1, len(result)):
        if result[i] < result[i - 1]:
            result[i] = result[i - 1]
    return result


def _extend_quantiles(q_values: np.ndarray, alphas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(q_values) != len(alphas):
        raise ValueError("Quantile values and alphas must align")
    if len(q_values) == 0:
        raise ValueError("At least one quantile is required")
    order = np.argsort(alphas)
    alphas_sorted = alphas[order]
    q_sorted = q_values[order]
    alpha_ext = np.concatenate(([0.0], alphas_sorted, [1.0]))
    q_ext = np.empty_like(alpha_ext)
    q_ext[1:-1] = q_sorted
    if len(q_sorted) == 1:
        slope = 1.0
        q_ext[0] = q_sorted[0] - slope * alphas_sorted[0]
        q_ext[-1] = q_sorted[0] + slope * (1.0 - alphas_sorted[0])
    else:
        low_span = max(alphas_sorted[1] - alphas_sorted[0], 1e-6)
        high_span = max(alphas_sorted[-1] - alphas_sorted[-2], 1e-6)
        slope_low = (q_sorted[1] - q_sorted[0]) / low_span
        slope_high = (q_sorted[-1] - q_sorted[-2]) / high_span
        q_ext[0] = q_sorted[0] - slope_low * alphas_sorted[0]
        q_ext[-1] = q_sorted[-1] + slope_high * (1.0 - alphas_sorted[-1])
    q_ext = _make_monotonic(q_ext)
    return alpha_ext, q_ext


def compute_pit(df: pd.DataFrame, quantile_mapping: Dict[float, str]) -> np.ndarray:
    alphas = np.array(list(quantile_mapping.keys()), dtype=float)
    cols = [quantile_mapping[a] for a in alphas]
    quantile_array = df[cols].to_numpy(dtype=float)
    obs = df["corrected_true_cms"].to_numpy(dtype=float)
    pits = np.full(obs.shape, np.nan, dtype=float)
    for idx, (target, q_values) in enumerate(zip(obs, quantile_array)):
        if not np.isfinite(target) or not np.all(np.isfinite(q_values)):
            continue
        alpha_ext, q_ext = _extend_quantiles(q_values, alphas)
        if q_ext[0] == q_ext[-1]:
            pits[idx] = 0.5
            continue
        pit_val = np.interp(target, q_ext, alpha_ext, left=0.0, right=1.0)
        pits[idx] = float(np.clip(pit_val, 0.0, 1.0))
    return pits


def compute_coverage(df: pd.DataFrame, quantile_mapping: Dict[float, str]) -> List[CoverageResult]:
    obs = df["corrected_true_cms"].to_numpy(dtype=float)
    results: List[CoverageResult] = []
    for alpha, col in quantile_mapping.items():
        preds = df[col].to_numpy(dtype=float)
        mask = np.isfinite(obs) & np.isfinite(preds)
        if not np.any(mask):
            empirical = float("nan")
        else:
            empirical = float(np.mean(obs[mask] <= preds[mask]))
        results.append(CoverageResult(alpha, empirical))
    return results


def compute_interval_coverage(df: pd.DataFrame, quantile_mapping: Dict[float, str]) -> List[IntervalCoverageResult]:
    alphas = sorted(quantile_mapping.keys())
    obs = df["corrected_true_cms"].to_numpy(dtype=float)
    results: List[IntervalCoverageResult] = []
    for i, low in enumerate(alphas):
        for high in alphas[i + 1 :]:
            low_col = quantile_mapping[low]
            high_col = quantile_mapping[high]
            low_vals = df[low_col].to_numpy(dtype=float)
            high_vals = df[high_col].to_numpy(dtype=float)
            mask = np.isfinite(obs) & np.isfinite(low_vals) & np.isfinite(high_vals)
            if not np.any(mask):
                empirical = float("nan")
            else:
                inside = (obs[mask] >= low_vals[mask]) & (obs[mask] <= high_vals[mask])
                empirical = float(np.mean(inside))
            nominal = float(high - low)
            results.append(IntervalCoverageResult(low, high, nominal, empirical))
    return results


def summarise_pit(pits: np.ndarray) -> Dict[str, float]:
    valid = pits[np.isfinite(pits)]
    if valid.size == 0:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "ks_statistic": float("nan")}
    sorted_vals = np.sort(valid)
    n = sorted_vals.size
    grid = (np.arange(1, n + 1) - 0.5) / n
    ks_stat = float(np.max(np.abs(sorted_vals - grid)))
    return {
        "count": int(n),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "ks_statistic": ks_stat,
    }


def plot_pit_histogram(pits: np.ndarray, out_path: str) -> None:
    valid = pits[np.isfinite(pits)]
    if valid.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(valid, bins=15, range=(0.0, 1.0), density=True, color="tab:orange", alpha=0.8)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title("Probability Integral Transform")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_quantile_calibration(coverage: Iterable[CoverageResult], out_path: str) -> None:
    coverage_list = list(coverage)
    if not coverage_list:
        return
    alphas = [c.nominal for c in coverage_list]
    empirical = [c.empirical for c in coverage_list]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1, label="Ideal")
    ax.plot(alphas, empirical, marker="o", linewidth=1.5, label="Empirical")
    ax.set_xlabel("Nominal quantile level")
    ax.set_ylabel("Fraction below quantile")
    ax.set_title("Quantile Calibration")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_interval_coverage(intervals: Iterable[IntervalCoverageResult], out_path: str) -> None:
    entries = [i for i in intervals if np.isfinite(i.empirical)]
    if not entries:
        return
    labels = [f"{i.alpha_low:.2f}-{i.alpha_high:.2f}" for i in entries]
    nominal = [i.nominal for i in entries]
    empirical = [i.empirical for i in entries]
    x = np.arange(len(entries))
    fig, ax = plt.subplots(figsize=(max(6, len(entries)), 4))
    ax.bar(x - 0.15, nominal, width=0.3, label="Nominal", color="tab:blue", alpha=0.7)
    ax.bar(x + 0.15, empirical, width=0.3, label="Empirical", color="tab:orange", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Coverage probability")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Central Interval Coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hydrograph(df: pd.DataFrame, quantile_mapping: Dict[float, str], out_path: str, label: str) -> None:
    times = df["timestamp"]
    truth = df["corrected_true_cms"].to_numpy(dtype=float)
    corrected = df["corrected_pred_cms"].to_numpy(dtype=float)
    nwm = df["nwm_cms"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 4))
    band_drawn = False
    if len(quantile_mapping) >= 2:
        alphas = sorted(quantile_mapping.keys())
        low_col = quantile_mapping[alphas[0]]
        high_col = quantile_mapping[alphas[-1]]
        low_vals = df[low_col].to_numpy(dtype=float)
        high_vals = df[high_col].to_numpy(dtype=float)
        ax.fill_between(times, low_vals, high_vals, color="tab:orange", alpha=0.25, label=f"{alphas[0]:.0%}-{alphas[-1]:.0%} band")
        band_drawn = True
    ax.plot(times, truth, color="black", linewidth=1.5, label="USGS (truth)")
    ax.plot(times, nwm, color="tab:blue", alpha=0.8, label="NWM")
    ax.plot(times, corrected, color="tab:red", alpha=0.9, label="Corrected")
    ax.set_xlabel("Time")
    ax.set_ylabel("Flow (cms)")
    title = "Hydrograph"
    if label:
        title = f"{title}: {label}"
    ax.set_title(title)
    ax.legend()
    ax.tick_params(axis="x", rotation=20)
    if band_drawn:
        ax.set_title(f"Hydrograph with Quantile Band: {label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pit_trace(df: pd.DataFrame, pits: np.ndarray, out_path: str, label: str) -> None:
    valid = np.isfinite(pits)
    if not np.any(valid):
        return
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df.loc[valid, "timestamp"], pits[valid], marker="o", linestyle="-", markersize=2, linewidth=0.8)
    ax.axhline(0.5, color="k", linestyle="--", linewidth=1)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("PIT")
    ax.set_xlabel("Time")
    title = "PIT Trace"
    if label:
        title = f"{title}: {label}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_summary(path: str, summary: Dict[str, object]) -> None:
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)


def run(args: argparse.Namespace) -> None:
    df = load_evaluation_csv(args.eval_csv)
    ensure_dir(args.output_dir)
    quantiles = discover_quantiles(df)
    if not quantiles:
        raise ValueError("No quantile columns detected; cannot compute calibration metrics.")

    pits = compute_pit(df, quantiles)
    coverage = compute_coverage(df, quantiles)
    interval_cov = compute_interval_coverage(df, quantiles)
    pit_summary = summarise_pit(pits)

    plot_hydrograph(df, quantiles, os.path.join(args.output_dir, "hydrograph_quantiles.png"), args.label)
    plot_pit_histogram(pits, os.path.join(args.output_dir, "pit_histogram.png"))
    plot_quantile_calibration(coverage, os.path.join(args.output_dir, "quantile_calibration.png"))
    plot_interval_coverage(interval_cov, os.path.join(args.output_dir, "interval_coverage.png"))
    plot_pit_trace(df, pits, os.path.join(args.output_dir, "pit_trace.png"), args.label)

    summary_payload = {
        "pit": pit_summary,
        "quantile_coverage": {f"{c.nominal:.3f}": c.empirical for c in coverage},
        "interval_coverage": {
            f"{i.alpha_low:.3f}-{i.alpha_high:.3f}": {
                "nominal": i.nominal,
                "empirical": i.empirical,
            }
            for i in interval_cov
        },
    }
    write_summary(os.path.join(args.output_dir, "quantile_calibration_summary.json"), summary_payload)

    if args.save_pit_csv:
        pits_path = os.path.join(args.output_dir, "pit_values.csv")
        export_df = df[["timestamp"]].copy()
        export_df["pit"] = pits
        export_df.to_csv(pits_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-run evaluation with quantile calibration diagnostics")
    parser.add_argument("--eval-csv", required=True, help="Path to evaluation CSV containing quantiles")
    parser.add_argument("--output-dir", required=True, help="Directory to write plots and summaries")
    parser.add_argument("--label", default="", help="Label for plot titles")
    parser.add_argument("--save-pit-csv", action="store_true", help="Export PIT values alongside timestamps")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

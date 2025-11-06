#!/usr/bin/env python3
"""Generate CV summary plots for the Watauga site."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")


def load_cv_summary(path: Path) -> list[dict]:
    with path.open() as fh:
        return json.load(fh)


def compute_improvements(cv_data: list[dict], metric: str) -> np.ndarray:
    improvements = []
    for entry in cv_data:
        base = entry["metrics"]["baseline"].get(metric)
        corr = entry["metrics"]["corrected"].get(metric)
        if base is None or corr is None:
            improvements.append(np.nan)
            continue
        if metric in {"rmse", "mae", "pbias"}:
            improvements.append(base - corr)
        else:
            improvements.append(corr - base)
    return np.array(improvements)


def plot_cv_metrics(cv_data: list[dict], out_dir: Path) -> None:
    folds = [entry["fold"] for entry in cv_data]
    baseline_rmse = [entry["metrics"]["baseline"]["rmse"] for entry in cv_data]
    corrected_rmse = [entry["metrics"]["corrected"]["rmse"] for entry in cv_data]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(folds, baseline_rmse, marker="o", label="NWM Baseline", color="red")
    ax.plot(folds, corrected_rmse, marker="o", label="Corrected", color="green")
    ax.set_xlabel("Validation Fold (Year)")
    ax.set_ylabel("RMSE (cms)")
    ax.set_title("Rolling-Origin CV: RMSE per Fold")
    ax.set_xticks(folds)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "watauga_cv_rmse.png", dpi=300)
    plt.close(fig)

    improvements = {
        "RMSE": compute_improvements(cv_data, "rmse"),
        "NSE": compute_improvements(cv_data, "nse"),
        "KGE": compute_improvements(cv_data, "kge"),
        "PBIAS": compute_improvements(cv_data, "pbias"),
    }
    metrics = list(improvements.keys())
    data = np.vstack([improvements[m] for m in metrics]).T

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.2
    x = np.arange(len(folds))
    colors = ["gray", "royalblue", "purple", "orange"]
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width - 1.5 * width, data[:, i], width, label=metric, color=colors[i])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_xlabel("Validation Fold (Year)")
    ax.set_ylabel("Improvement (Corrected − Baseline)")
    ax.set_title("Metric Improvements per Fold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "watauga_cv_improvements.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CV summaries for Watauga.")
    parser.add_argument("--cv-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cv_data = load_cv_summary(args.cv_summary)
    plot_cv_metrics(cv_data, out_dir)


if __name__ == "__main__":
    main()

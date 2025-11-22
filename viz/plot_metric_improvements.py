"""Visualize baseline vs corrected metrics plus improvement percentages."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from . import colors, utils

MetricSpec = Tuple[str, str, bool, bool]

METRICS: List[MetricSpec] = [
    ("rmse", "RMSE (cms)", False, False),
    ("mae", "MAE (cms)", False, False),
    ("nse", "NSE", True, False),
    ("kge", "KGE", True, False),
    ("pearson_r", "Pearson r", True, False),
    ("spearman_r", "Spearman r", True, False),
    ("pbias", "|PBIAS| (%)", False, True),
]


def _prepare_rows(metrics_json: Dict[str, Dict[str, float]]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    baseline = metrics_json["baseline"]
    corrected = metrics_json["corrected"]
    for key, label, higher_is_better, use_abs in METRICS:
        if key not in baseline or key not in corrected:
            continue
        base_raw = baseline.get(key)
        corr_raw = corrected.get(key)
        if base_raw is None or corr_raw is None:
            continue
        if isinstance(base_raw, float) and math.isnan(base_raw):
            continue
        if isinstance(corr_raw, float) and math.isnan(corr_raw):
            continue
        base_val = abs(base_raw) if use_abs else base_raw
        corr_val = abs(corr_raw) if use_abs else corr_raw
        denom = None if math.isclose(base_val, 0.0, abs_tol=1e-9) else base_val
        if higher_is_better:
            delta = corr_val - base_val
            percent = (delta / denom * 100) if denom is not None else None
        else:
            delta = base_val - corr_val
            percent = (delta / denom * 100) if denom is not None else None
        rows.append(
            {
                "key": key,
                "label": label,
                "baseline": base_val,
                "corrected": corr_val,
                "higher_is_better": higher_is_better,
                "delta": delta,
                "percent": percent,
            }
        )
    return rows


def plot_metric_improvements(*, metrics_path: Path, output: Path) -> None:
    utils.configure_style()
    utils.validate_inputs([metrics_path])

    metrics_json = json.loads(metrics_path.read_text())
    rows = _prepare_rows(metrics_json)
    if not rows:
        raise ValueError("No overlapping metrics found between baseline and corrected entries.")

    fig_height = 1.5 * len(rows) + 1.5
    fig, axes = plt.subplots(len(rows), 1, figsize=(8, fig_height), sharex=False)
    if len(rows) == 1:
        axes = [axes]

    baseline_color = colors.COLORS["nwm"]
    corrected_color = colors.COLORS["ml"]

    for ax, row in zip(axes, rows):
        ax.barh(["Baseline"], [row["baseline"]], color=baseline_color, height=0.35, label="Baseline")
        ax.barh(["Corrected"], [row["corrected"]], color=corrected_color, height=0.35, label="Corrected")
        ax.set_title(row["label"], loc="left")
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlabel(row["label"])
        delta = row["delta"]
        percent = row["percent"]
        better = delta >= 0
        color = colors.COLORS["obs"] if better else "#777777"
        annotation = f"Δ={delta:.3f}"
        if percent is not None:
            annotation += f" ({percent:+.1f}%)"
        ax.text(
            0.99,
            0.15,
            annotation,
            va="center",
            ha="right",
            fontsize=9,
            color=color,
            transform=ax.transAxes,
        )

    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    utils.ensure_parent(output)
    fig.savefig(output)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True, help="Metrics JSON path")
    parser.add_argument("--out", type=Path, required=True, help="Output path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_metric_improvements(metrics_path=args.metrics, output=args.out)


if __name__ == "__main__":
    main()

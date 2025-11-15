#!/usr/bin/env python3
"""Generate a swim-lane diagram summarising the hydrologic pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.style.use("seaborn-v0_8-whitegrid")

STAGES = [
    ("Acquisition", "Scripts retrieve USGS, NWM, ERA5, NLCD inputs"),
    ("Preprocessing / Feature Engineering", "Temporal alignment, QA/QC, scaling"),
    ("Sequence Batching & Training", "SeqDataset windows feed hybrid transformer"),
]

SOURCES = [
    ("USGS Discharge", "#1b9e77"),
    ("NWM CHRTOUT", "#d95f02"),
    ("ERA5 Reanalysis", "#7570b3"),
    ("NLCD Land Cover", "#e7298a"),
]


def build_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, len(STAGES))
    ax.set_ylim(0, len(SOURCES))

    # Stage blocks
    for idx, (title, subtitle) in enumerate(STAGES):
        rect = FancyBboxPatch(
            (idx, -0.3),
            width=0.95,
            height=len(SOURCES) + 0.6,
            boxstyle="round,pad=0.02",
            linewidth=1.2,
            edgecolor="#cccccc",
            facecolor="#f6f6f6",
        )
        ax.add_patch(rect)
        ax.text(
            idx + 0.475,
            len(SOURCES) + 0.4,
            title,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            idx + 0.475,
            len(SOURCES) + 0.15,
            subtitle,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#555555",
            wrap=True,
        )

    # Source lanes
    lane_height = 0.8
    for lane_idx, (source, color) in enumerate(SOURCES):
        y = lane_idx + 0.5
        ax.hlines(y, 0.05, len(STAGES) - 0.05, colors=color, linewidth=3)
        for stage_idx in range(len(STAGES) - 1):
            ax.annotate(
                "",
                xy=(stage_idx + 0.98, y),
                xytext=(stage_idx + 0.2, y),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=2),
            )
        ax.text(
            -0.05,
            y,
            source,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=color,
        )

    ax.set_title("Leakage-Safe Pipeline Swim-Lane Overview", fontsize=14, pad=20)
    ax.axis("off")
    fig.tight_layout()
    return fig


def main(output_dir: str | Path = "results/figures/viz") -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    png_path = out_dir / "swim_lane_pipeline_overview.png"
    svg_path = out_dir / "swim_lane_pipeline_overview.svg"
    fig.savefig(png_path, dpi=300, transparent=True)
    fig.savefig(svg_path, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render a normalization strategy table for manuscript figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")

TABLE_CONTENT = {
    "Feature Group": [
        "Dynamic USGS Residuals",
        "NWM Baseline",
        "ERA5 Meteorology",
        "Static NLCD + Regulation",
    ],
    "Description": [
        "Observed minus NWM discharge; targets only",
        "Baseline CHRTOUT streamflow input",
        "Hourly precipitation, radiation, temperature, wind",
        "Land-cover fractions, regulation flag",
    ],
    "Scaling": [
        "Standardised using train-period mean/std",
        "Standardised using train-period mean/std",
        "Standardised per feature (train statistics)",
        "Z-score for continuous, binary for regulation",
    ],
    "Leakage Guardrails": [
        "Residuals computed with causal joins and no future info",
        "Only concurrent NWM flows; no future forecasts",
        "Normalisation stats from 2010-2020 only",
        "Static fields stored separately and merged at runtime",
    ],
}


def build_table() -> plt.Figure:
    df = pd.DataFrame(TABLE_CONTENT)
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#d9edf7")
        else:
            cell.set_facecolor("#fdfdfd")
    fig.tight_layout()
    return fig


def main(output_dir: str | Path = "results/figures/viz") -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_table()
    png = out_dir / "normalization_strategy_table.png"
    svg = out_dir / "normalization_strategy_table.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

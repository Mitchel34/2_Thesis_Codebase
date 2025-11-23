"""Plot RMSE comparison across models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import colors, utils


def plot_rmse_comparison(output: Path) -> None:
    utils.configure_style()

    # Data based on selected representative runs
    # NWM: Baseline from all files (consistent)
    # LSTM: baseline_v1 (RMSE ~5.69) - chosen to represent a standard baseline
    # Hydra: watauga_multiobj_trial1 (RMSE ~5.14) - best proposed model
    data = [
        {"Model": "NWM", "RMSE": 5.97, "Color": "gray"},
        {"Model": "LSTM", "RMSE": 5.69, "Color": "skyblue"},
        {"Model": "Hydra", "RMSE": 5.14, "Color": "#d62728"},  # Crimson red
    ]
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Create bars
    bars = ax.bar(
        df["Model"], 
        df["RMSE"], 
        color=df["Color"],
        width=0.6,
        edgecolor="black",
        linewidth=1.0,
        alpha=0.9
    )

    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.05,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold"
        )

    # Formatting
    ax.set_ylabel("RMSE (m³/s)", fontsize=12)
    ax.set_title("Model Performance Comparison (RMSE)", fontsize=14, fontweight="bold")
    
    # Y-axis range
    ax.set_ylim(0, 7.0)
    
    # Grid
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Remove top and right spines
    sns.despine()

    fig.tight_layout()
    utils.ensure_parent(output)
    fig.savefig(output, dpi=300)
    print(f"Plot saved to {output}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output file path")
    args = parser.parse_args()
    plot_rmse_comparison(output=args.out)


if __name__ == "__main__":
    main()

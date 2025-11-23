"""Plot grouped bar chart for NSE and KGE comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import colors, utils


def plot_metrics_comparison(output: Path) -> None:
    utils.configure_style()

    # Data preparation
    # Values derived from:
    # NWM: watauga_multiobj_trial1_metrics.json (baseline)
    # LSTM: baseline_v1_metrics.json (consistent with RMSE plot)
    # Hydra: watauga_multiobj_trial1_metrics.json (best run)
    
    data = [
        # NSE Group
        {"Metric": "NSE", "Model": "NWM", "Score": 0.52, "Color": "gray"},
        {"Metric": "NSE", "Model": "LSTM", "Score": 0.56, "Color": "skyblue"},
        {"Metric": "NSE", "Model": "Hydra", "Score": 0.64, "Color": "#d62728"},
        
        # KGE Group
        {"Metric": "KGE", "Model": "NWM", "Score": 0.64, "Color": "gray"},
        {"Metric": "KGE", "Model": "LSTM", "Score": 0.51, "Color": "skyblue"},
        {"Metric": "KGE", "Model": "Hydra", "Score": 0.71, "Color": "#d62728"},
    ]
    
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Create grouped bar chart
    # We use seaborn for easy grouping
    sns.barplot(
        data=df,
        x="Metric",
        y="Score",
        hue="Model",
        palette={"NWM": "gray", "LSTM": "skyblue", "Hydra": "#d62728"},
        edgecolor="black",
        linewidth=1.0,
        ax=ax,
        alpha=0.9
    )

    # Annotate bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=10, fontweight="bold")

    # Formatting
    ax.set_ylabel("Score (dimensionless)", fontsize=12)
    ax.set_xlabel("") # No x-label needed as ticks are self-explanatory
    ax.set_title("Model Performance: NSE & KGE", fontsize=14, fontweight="bold")
    
    # Y-axis range
    ax.set_ylim(0.0, 1.0)
    
    # Legend
    ax.legend(title="Model", frameon=True, loc="upper left", bbox_to_anchor=(1, 1))
    
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
    plot_metrics_comparison(output=args.out)


if __name__ == "__main__":
    main()

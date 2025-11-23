"""Plot a high-quality hydrograph with uncertainty bands for 2022."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import colors, utils


def plot_hydrograph_with_uncertainty(
    *,
    csv_path: Path,
    output: Path,
    year: int = 2022,
) -> None:
    utils.configure_style()
    utils.validate_inputs([csv_path])

    # Load data
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    
    # Filter for the specific year
    start_date = pd.Timestamp(f"{year}-01-01")
    end_date = pd.Timestamp(f"{year}-12-31 23:00:00")
    
    # Calculate empirical uncertainty bounds from the residuals of the ENTIRE dataset
    # (assuming the test set represents the error distribution)
    # Residual = True - Pred
    # True = Pred + Residual
    # We want the range where True likely falls.
    # P10 of Residuals and P90 of Residuals give us the offsets.
    residuals = df["y_true_residual_cms"] - df["y_pred_residual_cms"]
    # Alternatively: residuals = df["usgs_cms"] - df["corrected_pred_cms"]
    # These should be identical if constructed correctly.
    
    p10_res = np.percentile(residuals, 10)
    p90_res = np.percentile(residuals, 90)
    
    print(f"Empirical 80% Confidence Interval from Residuals: [{p10_res:.3f}, {p90_res:.3f}] cms")

    # Now filter for the plot window
    mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
    df_plot = df.loc[mask].copy().sort_values("timestamp")

    if df_plot.empty:
        raise ValueError(f"No data found for year {year}")

    # Calculate bounds for the plot
    df_plot["lower_bound"] = df_plot["corrected_pred_cms"] + p10_res
    df_plot["upper_bound"] = df_plot["corrected_pred_cms"] + p90_res
    
    # Clip lower bound to 0 (streamflow can't be negative)
    df_plot["lower_bound"] = df_plot["lower_bound"].clip(lower=0.0)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # 1. Observed (Black Solid)
    ax.plot(
        df_plot["timestamp"], 
        df_plot["usgs_cms"], 
        label="Observed", 
        color="black", 
        linewidth=1.2,
        zorder=2
    )
    
    # 2. NWM (Blue Dashed)
    ax.plot(
        df_plot["timestamp"], 
        df_plot["nwm_cms"], 
        label="NWM", 
        color="blue", 
        linestyle="--", 
        linewidth=1.0,
        alpha=0.8,
        zorder=1
    )
    
    # 3. Corrected (Red Solid)
    ax.plot(
        df_plot["timestamp"], 
        df_plot["corrected_pred_cms"], 
        label="Corrected (Hydra)", 
        color="#d62728",  # Matplotlib 'tab:red'
        linewidth=1.2,
        zorder=3
    )
    
    # 4. Uncertainty Band (Red Shaded)
    ax.fill_between(
        df_plot["timestamp"],
        df_plot["lower_bound"],
        df_plot["upper_bound"],
        color="#d62728",
        alpha=0.2,
        label="80% Prediction Interval",
        zorder=0
    )

    # Formatting
    ax.set_title(f"Watauga River Streamflow Comparison ({year})", fontsize=14, fontweight="bold")
    ax.set_ylabel("Streamflow (m³/s)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    
    # X-axis formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlim(start_date, end_date)
    
    # Y-axis formatting
    ax.set_ylim(bottom=0)
    
    # Legend
    ax.legend(frameon=True, loc="upper right", fontsize=10, framealpha=0.9)
    
    # Grid
    ax.grid(True, which="major", axis="y", linestyle="-", alpha=0.6)
    ax.grid(True, which="major", axis="x", linestyle=":", alpha=0.3)

    fig.tight_layout()
    utils.ensure_parent(output)
    fig.savefig(output, dpi=300)
    print(f"Plot saved to {output}")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, required=True, help="Evaluation CSV path")
    parser.add_argument("--out", type=Path, required=True, help="Output file path")
    parser.add_argument("--year", type=int, default=2022, help="Year to plot")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_hydrograph_with_uncertainty(
        csv_path=args.eval,
        output=args.out,
        year=args.year,
    )


if __name__ == "__main__":
    main()

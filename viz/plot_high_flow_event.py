"""Plot observed vs. NWM vs. Hydra for a 2022 high-flow storm event."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import utils


def _build_event_window(
    df: pd.DataFrame,
    *,
    event_start: Optional[pd.Timestamp],
    event_end: Optional[pd.Timestamp],
    buffer_hours: int,
) -> pd.DataFrame:
    """Slice out the requested event window with a small buffer for context."""
    if event_start is None or event_end is None:
        # Default: February 3–6, 2022 storm (clear multi-day high-flow pulse)
        event_start = pd.Timestamp("2022-02-02 00:00:00")
        event_end = pd.Timestamp("2022-02-07 00:00:00")

    start = pd.Timestamp(event_start) - timedelta(hours=buffer_hours)
    end = pd.Timestamp(event_end) + timedelta(hours=buffer_hours)

    window = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    if window.empty:
        raise ValueError("No data found within the requested event window.")
    return window


def _compute_empirical_bounds(df: pd.DataFrame) -> tuple[float, float]:
    """Use residual quantiles as a simple empirical uncertainty band."""
    residuals = df["usgs_cms"] - df["corrected_pred_cms"]
    lower_offset = np.percentile(residuals, 10)
    upper_offset = np.percentile(residuals, 90)
    return lower_offset, upper_offset


def plot_high_flow_event(
    *,
    csv_path: Path,
    output: Path,
    event_start: Optional[pd.Timestamp],
    event_end: Optional[pd.Timestamp],
    buffer_hours: int = 12,
) -> None:
    utils.configure_style()
    utils.validate_inputs([csv_path])

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")

    lower_offset, upper_offset = _compute_empirical_bounds(df)
    window = _build_event_window(
        df,
        event_start=event_start,
        event_end=event_end,
        buffer_hours=buffer_hours,
    )

    window["hydra_lower"] = (window["corrected_pred_cms"] + lower_offset).clip(lower=0.0)
    window["hydra_upper"] = window["corrected_pred_cms"] + upper_offset

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        window["timestamp"],
        window["usgs_cms"],
        color="black",
        linewidth=1.5,
        label="Observed (USGS)",
        zorder=3,
    )
    ax.plot(
        window["timestamp"],
        window["nwm_cms"],
        color="gray",
        linestyle="--",
        linewidth=1.1,
        label="NWM v3",
        zorder=1,
    )
    ax.plot(
        window["timestamp"],
        window["corrected_pred_cms"],
        color="#d62728",
        linewidth=1.5,
        label="Hydra-corrected",
        zorder=4,
    )
    ax.fill_between(
        window["timestamp"],
        window["hydra_lower"],
        window["hydra_upper"],
        color="#d62728",
        alpha=0.18,
        label="Hydra P10–P90 (residual-derived)",
        zorder=0,
    )

    # Highlight the peak for quick visual comparison
    peak_idx = window["usgs_cms"].idxmax()
    peak_row = window.loc[peak_idx]
    ax.axvline(peak_row["timestamp"], color="k", linestyle=":", linewidth=0.9, alpha=0.8)
    ax.text(
        peak_row["timestamp"],
        peak_row["usgs_cms"] * 0.98,
        "Peak",
        rotation=90,
        ha="right",
        va="top",
        fontsize=9,
        color="black",
    )

    mae_h = (window["corrected_pred_cms"] - window["usgs_cms"]).abs().mean()
    mae_n = (window["nwm_cms"] - window["usgs_cms"]).abs().mean()
    ax.text(
        0.01,
        0.95,
        f"Window MAE – Hydra: {mae_h:.1f} vs NWM: {mae_n:.1f}",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
    )

    ax.set_title("2022 High-Flow Storm: Observed vs. NWM vs. Hydra", fontsize=14, fontweight="bold")
    ax.set_ylabel("Streamflow (m³/s)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylim(bottom=0)

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()

    ax.legend(frameon=True, loc="upper left", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)

    utils.ensure_parent(output)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, required=True, help="Evaluation CSV path")
    parser.add_argument("--out", type=Path, required=True, help="Output file path")
    parser.add_argument("--event-start", type=str, help="Event start timestamp (inclusive)")
    parser.add_argument("--event-end", type=str, help="Event end timestamp (inclusive)")
    parser.add_argument(
        "--buffer-hours",
        type=int,
        default=12,
        help="Hours to add before/after the event window for context",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_high_flow_event(
        csv_path=args.eval,
        output=args.out,
        event_start=pd.Timestamp(args.event_start) if args.event_start else None,
        event_end=pd.Timestamp(args.event_end) if args.event_end else None,
        buffer_hours=args.buffer_hours,
    )


if __name__ == "__main__":
    main()

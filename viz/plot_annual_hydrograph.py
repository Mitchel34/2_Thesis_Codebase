"""Plot a full-year hydrograph comparing observed, NWM, and corrected flows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from . import colors, utils


def _maybe_resample(df: pd.DataFrame, freq: Optional[str]) -> pd.DataFrame:
    if not freq:
        return df
    rule = freq.upper()
    agg = {
        "usgs_cms": "mean",
        "nwm_cms": "mean",
        "corrected_pred_cms": "mean",
    }
    resampled = df.set_index("timestamp").resample(rule).agg(agg).dropna().reset_index()
    return resampled


def plot_annual_hydrograph(
    *,
    csv_path: Path,
    output: Path,
    start: Optional[str],
    end: Optional[str],
    resample: Optional[str],
) -> None:
    utils.configure_style()
    utils.validate_inputs([csv_path])

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    if start:
        df = df[df["timestamp"] >= pd.to_datetime(start)]
    if end:
        df = df[df["timestamp"] <= pd.to_datetime(end)]
    df = df.sort_values("timestamp")

    if resample:
        df = _maybe_resample(df, resample)

    if df.empty:
        raise ValueError("No data left after filtering; adjust start/end filters.")

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(df["timestamp"], df["usgs_cms"], label="USGS Observed", color=colors.COLORS["obs"], linewidth=1.4)
    ax.plot(df["timestamp"], df["nwm_cms"], label="NWM Baseline", color=colors.COLORS["nwm"], linewidth=1.0)
    ax.plot(
        df["timestamp"],
        df["corrected_pred_cms"],
        label="Corrected (Transformer)",
        color=colors.COLORS["ml"],
        linewidth=1.0,
    )

    ax.set_title("Watauga River: Full-Year Hydrograph")
    ax.set_ylabel("Discharge (cms)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend(frameon=False, ncol=3, loc="upper right")

    fig.autofmt_xdate()
    fig.tight_layout()
    utils.ensure_parent(output)
    fig.savefig(output)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, required=True, help="Evaluation CSV path")
    parser.add_argument("--out", type=Path, required=True, help="Output file path")
    parser.add_argument("--start", help="Optional ISO start timestamp")
    parser.add_argument("--end", help="Optional ISO end timestamp")
    parser.add_argument("--resample", help="Optional pandas offset alias (e.g., 'D' for daily mean)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_annual_hydrograph(
        csv_path=args.eval,
        output=args.out,
        start=args.start,
        end=args.end,
        resample=args.resample,
    )


if __name__ == "__main__":
    main()

"""Create WRR-ready multi-panel results figure."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import colors, utils


DEFAULT_COLUMNS = {
    "timestamp": "timestamp",
    "obs": "usgs_cms",
    "nwm": "nwm_cms",
    "ml": "corrected_pred_cms",
}


def _select_window(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start and end:
        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        return df.loc[mask]
    return df.head(200)


def _panel_hydro(ax, window: pd.DataFrame) -> None:
    ax.plot(window["timestamp"], window["obs"], color=colors.COLORS["obs"], label="Observed")
    ax.plot(window["timestamp"], window["nwm"], color=colors.COLORS["nwm"], label="NWM")
    ax.plot(window["timestamp"], window["ml"], color=colors.COLORS["ml"], label="Corrected")
    ax.set_ylabel("Discharge (cms)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.legend(frameon=False, loc="upper left")


def _panel_scatter(ax, df: pd.DataFrame) -> None:
    ax.hexbin(df["obs"], df["ml"], gridsize=50, cmap="viridis", mincnt=1)
    lims = [0, max(df["obs"].max(), df["ml"].max())]
    ax.plot(lims, lims, color="black", linewidth=1)
    ax.set_xlabel("Observed (cms)")
    ax.set_ylabel("Corrected (cms)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)


def _panel_hist(ax, df: pd.DataFrame) -> None:
    res_ml = df["ml"] - df["obs"]
    res_nwm = df["nwm"] - df["obs"]
    ax.hist(res_nwm, bins=40, alpha=0.4, label="NWM", color=colors.COLORS["nwm"])
    ax.hist(res_ml, bins=40, alpha=0.5, label="Corrected", color=colors.COLORS["ml"])
    ax.set_xlabel("Residual (cms)")
    ax.legend(frameon=False)


def _panel_monthly(ax, df: pd.DataFrame) -> None:
    monthly = df.copy()
    monthly["month"] = monthly["timestamp"].dt.month
    residual = monthly["ml"] - monthly["obs"]
    data = [residual[monthly["month"] == m] for m in range(1, 13)]
    ax.boxplot(data, showfliers=False)
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel("Residual (cms)")


def plot_panels(*, csv_path: Path, output: Path, start: str | None, end: str | None, columns: Tuple[str, str, str, str]) -> None:
    utils.configure_style()
    utils.validate_inputs([csv_path])

    df = pd.read_csv(csv_path, parse_dates=[columns[0]])
    df = df.rename(columns={columns[0]: "timestamp", columns[1]: "obs", columns[2]: "nwm", columns[3]: "ml"})
    df.sort_values("timestamp", inplace=True)

    window = _select_window(df, start, end)

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    _panel_hydro(axs[0, 0], window)
    utils.add_panel_label(axs[0, 0], "A")

    _panel_scatter(axs[0, 1], df)
    utils.add_panel_label(axs[0, 1], "B")

    _panel_hist(axs[1, 0], df)
    utils.add_panel_label(axs[1, 0], "C")

    _panel_monthly(axs[1, 1], df)
    utils.add_panel_label(axs[1, 1], "D")

    fig.tight_layout()
    utils.ensure_parent(output)
    fig.savefig(output)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, required=True, help="Evaluation CSV path")
    parser.add_argument("--out", type=Path, required=True, help="Output file path")
    parser.add_argument("--start", help="Optional start timestamp (ISO)")
    parser.add_argument("--end", help="Optional end timestamp (ISO)")
    parser.add_argument("--timestamp-col", default=DEFAULT_COLUMNS["timestamp"])
    parser.add_argument("--obs-col", default=DEFAULT_COLUMNS["obs"])
    parser.add_argument("--nwm-col", default=DEFAULT_COLUMNS["nwm"])
    parser.add_argument("--ml-col", default=DEFAULT_COLUMNS["ml"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    columns = (args.timestamp_col, args.obs_col, args.nwm_col, args.ml_col)
    plot_panels(csv_path=args.eval, output=args.out, start=args.start, end=args.end, columns=columns)


if __name__ == "__main__":
    main()

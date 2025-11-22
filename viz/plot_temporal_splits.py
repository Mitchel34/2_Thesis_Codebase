"""Visualize train/validation/test and rolling-origin splits."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from . import colors, utils


Date = datetime


def _parse_date(value: str) -> Date:
    return datetime.fromisoformat(value)


def _load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    return data


def _bar(ax, y: float, start: Date, end: Date, color: str, label: str, height: float = 0.8):
    ax.barh(y, (end - start).days, left=start, height=height, color=color, edgecolor="black", label=label)


def plot_splits(*, splits_file: Path, rolling_file: Path | None, output: Path) -> None:
    utils.configure_style()
    splits = _load_config(splits_file)
    rolling = _load_config(rolling_file) if rolling_file else {"windows": []}

    fig, ax = plt.subplots(figsize=(8, 3))

    levels = {"train": 0, "val": 1, "test": 2}
    seen_labels = set()
    for key, color_key in [("train", "train"), ("val", "val"), ("test", "test")]:
        start = _parse_date(splits[key]["start"])
        end = _parse_date(splits[key]["end"])
        label = key.capitalize() if key not in seen_labels else ""
        _bar(ax, levels[key], start, end, colors.COLORS[color_key], label)
        seen_labels.add(key)

    for idx, window in enumerate(rolling.get("windows", [])):
        start = _parse_date(window["train_start"])
        end = _parse_date(window["test_end"])
        ax.barh(3 + idx, (end - start).days, left=start, height=0.6, color="#bbbbbb", alpha=0.4)

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Train", "Validation", "Test"])
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlabel("Date")
    fig.autofmt_xdate()

    utils.ensure_parent(output)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, required=True, help="JSON file with train/val/test ranges")
    parser.add_argument("--rolling", type=Path, help="JSON file with rolling-origin windows")
    parser.add_argument("--out", type=Path, required=True, help="Output path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_splits(splits_file=args.splits, rolling_file=args.rolling, output=args.out)


if __name__ == "__main__":
    main()

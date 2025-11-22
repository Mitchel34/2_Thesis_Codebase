"""WRR-friendly matplotlib styling helpers."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator

import matplotlib as mpl

_DEFAULT_FONT = "DejaVu Sans"

@dataclass
class StyleConfig:
    font_family: str = _DEFAULT_FONT
    figure_dpi: int = 300
    line_width: float = 1.5
    axis_width: float = 1.1
    tick_width: float = 1.1
    grid_color: str = "#dddddd"


WRR_STYLE = StyleConfig()


def apply_wrr_style(config: StyleConfig | None = None) -> None:
    cfg = config or WRR_STYLE
    mpl.rcParams.update(
        {
            "figure.dpi": cfg.figure_dpi,
            "font.family": cfg.font_family,
            "axes.linewidth": cfg.axis_width,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": cfg.grid_color,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "lines.linewidth": cfg.line_width,
            "xtick.major.width": cfg.tick_width,
            "ytick.major.width": cfg.tick_width,
            "xtick.minor.width": cfg.tick_width * 0.8,
            "ytick.minor.width": cfg.tick_width * 0.8,
        }
    )


@contextlib.contextmanager
def temporary_style(config: StyleConfig | None = None) -> Iterator[None]:
    prev = mpl.rcParams.copy()
    apply_wrr_style(config)
    try:
        yield
    finally:
        mpl.rcParams.update(prev)

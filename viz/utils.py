"""Utility helpers shared across visualization scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .style import apply_wrr_style


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def add_panel_label(ax, label: str, *, x: float = -0.1, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight="bold")


def configure_style() -> None:
    apply_wrr_style()


def validate_inputs(paths: Iterable[str | Path]) -> None:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"Missing required input files:\n{joined}")

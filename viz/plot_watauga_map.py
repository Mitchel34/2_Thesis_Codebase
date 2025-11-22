"""Generate WRR-style study area map for the Watauga basin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from . import colors, utils

try:  # optional deps
    import geopandas as gpd  # type: ignore
    _HAS_GPD = True
except Exception:  # pragma: no cover
    _HAS_GPD = False


def _load_site_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = path.read_text()
    if path.suffix in {".json", ".geojson"}:
        return json.loads(data)
    try:
        import yaml  # type: ignore

        return yaml.safe_load(data)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Unsupported metadata format: {path}") from exc


def _prepare_axes(fig):
    ax = fig.add_subplot(111)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return ax


def plot_map(*, lat: float, lon: float, output: Path, shapefile: Path | None, metadata: dict[str, Any]) -> None:
    utils.configure_style()
    fig = plt.figure(figsize=(5, 4))
    ax = _prepare_axes(fig)

    if shapefile and _HAS_GPD:
        basin = gpd.read_file(shapefile)
        basin.boundary.plot(ax=ax, color="black", linewidth=0.8)
        basin.plot(ax=ax, color="#f0f4ff", alpha=0.6)
        ax.set_xlim(basin.total_bounds[0] - 0.2, basin.total_bounds[2] + 0.2)
        ax.set_ylim(basin.total_bounds[1] - 0.2, basin.total_bounds[3] + 0.2)
    else:
        ax.set_xlim(lon - 0.5, lon + 0.5)
        ax.set_ylim(lat - 0.5, lat + 0.5)

    ax.scatter(lon, lat, s=80, marker="*", color=colors.COLORS["obs"], edgecolor="black", label="USGS Gauge")

    name = metadata.get("name", "Watauga River")
    ax.set_title(f"{name} Study Area")
    ax.legend(frameon=False)

    utils.ensure_parent(output)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", type=float, required=True, help="Gauge latitude")
    parser.add_argument("--lon", type=float, required=True, help="Gauge longitude")
    parser.add_argument("--metadata", type=Path, help="Optional JSON/YAML metadata file")
    parser.add_argument("--shapefile", type=Path, help="Optional watershed shapefile")
    parser.add_argument("--out", type=Path, required=True, help="Output figure path")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    metadata = _load_site_metadata(args.metadata) if args.metadata else {}
    shapefile = args.shapefile if args.shapefile and args.shapefile.exists() else None
    plot_map(lat=args.lat, lon=args.lon, output=args.out, shapefile=shapefile, metadata=metadata)


if __name__ == "__main__":
    main()

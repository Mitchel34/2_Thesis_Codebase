#!/usr/bin/env python3
"""Plot a geographic context map for the Watauga River gauge."""

from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely.geometry import Point

plt.style.use("seaborn-v0_8-whitegrid")

NATURAL_EARTH_URL = (
    "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
)
NATURAL_EARTH_ZIP = Path("data/external/naturalearth/ne_110m_admin_0_countries.zip")

NLCD_LEGEND = {
    "Forest": ("#006400", "62%"),
    "Developed": ("#d7191c", "9%"),
    "Agriculture": ("#ffd700", "14%"),
    "Water/Wetlands": ("#2c7bb6", "3%"),
}


def load_world() -> gpd.GeoDataFrame:
    if not NATURAL_EARTH_ZIP.exists():
        NATURAL_EARTH_ZIP.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(NATURAL_EARTH_URL, NATURAL_EARTH_ZIP)
    return gpd.read_file(NATURAL_EARTH_ZIP)


def build_map(lon: float, lat: float, out_dir: Path, site_name: str) -> None:
    world = load_world()
    usa = world[world["ADMIN"] == "United States of America"]

    site = gpd.GeoDataFrame({"name": [site_name]}, geometry=[Point(lon, lat)], crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(8, 5))
    usa.plot(ax=ax, color="#f0f0f0", edgecolor="#cccccc")
    site.plot(ax=ax, color="black", markersize=40)

    ax.annotate(
        site_name,
        xy=(lon, lat),
        xytext=(lon - 4, lat + 3),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontsize=11,
        fontweight="bold",
    )

    ax.set_xlim(lon - 6, lon + 6)
    ax.set_ylim(lat - 4, lat + 4)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Watauga River Study Area")
    ax.set_aspect("equal", adjustable="box")

    legend_x = lon + 3.5
    legend_y = lat - 3.2
    for idx, (label, (color, frac)) in enumerate(NLCD_LEGEND.items()):
        rect = Rectangle((legend_x, legend_y + idx * 0.6), 0.5, 0.4, facecolor=color, edgecolor="none")
        ax.add_patch(rect)
        ax.text(legend_x + 0.6, legend_y + idx * 0.6 + 0.2, f"{label} ({frac})", va="center", fontsize=9)

    ax.text(
        legend_x,
        legend_y + len(NLCD_LEGEND) * 0.6 + 0.3,
        "NLCD 2021 Composition",
        fontsize=10,
        fontweight="bold",
    )

    fig.tight_layout()
    png_path = out_dir / "watauga_context_map.png"
    svg_path = out_dir / "watauga_context_map.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Watauga geographic context map")
    parser.add_argument("--lon", type=float, default=-81.687)
    parser.add_argument("--lat", type=float, default=36.215)
    parser.add_argument("--site-name", default="Watauga River, NC")
    parser.add_argument("--output-dir", default="results/figures/viz")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_map(args.lon, args.lat, out_dir, args.site_name)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a study-area map and metadata summary for a given site."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import urllib.request
from shapely.geometry import Point, box

plt.style.use("seaborn-v0_8-whitegrid")

NATURAL_EARTH_URL = (
    "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
)
NATURAL_EARTH_ZIP = Path("data/external/naturalearth/ne_110m_admin_0_countries.zip")


def load_world_boundaries() -> gpd.GeoDataFrame:
    """Load Natural Earth country polygons with simple on-disk caching."""
    if not NATURAL_EARTH_ZIP.exists():
        NATURAL_EARTH_ZIP.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(NATURAL_EARTH_URL, NATURAL_EARTH_ZIP)
    return gpd.read_file(NATURAL_EARTH_ZIP)


def load_site_info(site_id: str) -> dict:
    from config.master_study_sites import MASTER_STUDY_SITES

    if site_id not in MASTER_STUDY_SITES:
        raise KeyError(f"Site {site_id} not found in MASTER_STUDY_SITES.")
    return MASTER_STUDY_SITES[site_id]


def plot_map(site_info: dict, out_path: Path) -> None:
    world = load_world_boundaries()
    name_col = "ADMIN" if "ADMIN" in world.columns else "name"
    usa = world[world[name_col] == "United States of America"]

    lat = site_info["lat"]
    lon = site_info["lon"]
    site_name = site_info["name"]

    point = gpd.GeoDataFrame(
        {"site": [site_name]}, geometry=[Point(lon, lat)], crs="EPSG:4326"
    )

    buffer_deg = 5.0
    bbox = box(lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    usa.plot(ax=axes[0], color="#f0f0f0", edgecolor="white")
    bbox_gdf.boundary.plot(ax=axes[0], color="darkgray", linewidth=1.5)
    point.plot(ax=axes[0], color="red", markersize=50)
    axes[0].set_xlim(lon - 12, lon + 12)
    axes[0].set_ylim(lat - 8, lat + 8)
    axes[0].set_title(f"{site_name} Gauge Location")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].annotate(
        site_name,
        xy=(lon, lat),
        xytext=(lon + 1.0, lat + 1.5),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=9,
        color="red",
    )
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].axis("off")
    metadata = [
        ("USGS ID", site_info.get("usgs_id", "N/A")),
        ("NWM COMID", site_info.get("nwm_comid", "N/A")),
        ("State", site_info.get("state", "N/A")),
        ("Region", site_info.get("region", "N/A")),
        ("Biome", site_info.get("biome", "N/A")),
        ("Regulation", site_info.get("regulation_status", "Unknown")),
    ]
    table = axes[1].table(
        cellText=[[key, value] for key, value in metadata],
        colLabels=["Field", "Value"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    axes[1].set_title("Site Metadata")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot study area map for a site.")
    parser.add_argument("--site-id", default="03479000", help="USGS site ID")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/watauga/watauga_study_area.png"),
    )
    args = parser.parse_args()

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    site_info = load_site_info(args.site_id)
    plot_map(site_info, out_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a study-area map and static feature summary for a given site."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
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


def load_land_use(site_id: str, csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    key_col = "site_key" if "site_key" in df.columns else df.columns[0]
    # Ensure site identifiers retain leading zeros regardless of CSV dtype.
    df_ids = df[key_col].astype(str).str.zfill(len(site_id))
    row = df.loc[df_ids == site_id]
    if row.empty:
        raise ValueError(f"No NLCD row found for {site_id} in {csv_path}")
    row = row.iloc[0]
    columns = [
        "urban_percent",
        "forest_percent",
        "agriculture_percent",
        "impervious_percent",
        "water_wetlands_percent",
    ]
    data = {col: float(row[col]) for col in columns if col in row}
    return pd.Series(data)


def plot_map(site_info: dict, land_use: pd.Series, out_path: Path) -> None:
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
    axes[0].set_title("Watauga River Gauge Location")
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

    categories = land_use.index
    values = land_use.values
    axes[1].bar(categories, values, color="#4c72b0")
    axes[1].set_ylim(0, max(values) * 1.2)
    axes[1].set_ylabel("Percent of Watershed (%)")
    axes[1].set_title("NLCD 2021 Land-Cover Composition")
    for idx, val in enumerate(values):
        axes[1].text(idx, val + max(values) * 0.05, f"{val:.1f}%", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot study area map for a site.")
    parser.add_argument("--site-id", default="03479000", help="USGS site ID")
    parser.add_argument(
        "--land-use-csv",
        type=Path,
        default=Path("data/raw/land_use/nlcd_2021_land_use_metrics.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/watauga/watauga_study_area.png"),
    )
    args = parser.parse_args()

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    site_info = load_site_info(args.site_id)
    land_use = load_land_use(args.site_id, args.land_use_csv)
    plot_map(site_info, land_use, out_path)


if __name__ == "__main__":
    main()

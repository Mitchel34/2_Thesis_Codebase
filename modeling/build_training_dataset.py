#!/usr/bin/env python3
"""
Build an hourly, multi-year training dataset by aligning:
- NWM hourly analysis (CHRTOUT) from retrospective datasets (v2.1: 1979–2020, v3.0: 2021–2023)
- USGS observed streamflow (hourly, UTC)
- ERA5/ERA5-Land environmental features (hourly preferred; 6-hourly acceptable with safe alignment)
- NLCD 2021 static land cover metrics

Targets:
- y_residual_cms = usgs_obs_cms - nwm_cms  (per valid_time)
- y_corrected_cms = usgs_obs_cms  (optional direct corrected runoff target)

Outputs:
- Parquet dataset under data/clean/modeling/hourly_training_{start}_{end}.parquet
- Small CSV sample for inspection

Assumptions:
- NWM CSVs exist under data/raw/nwm_v3/retrospective/*.csv with columns:
    timestamp (valid_time), streamflow_cms, site_name, comid
- USGS hourly CSVs exist per site under data/raw/usgs/*{usgs_id}*.csv with columns timestamp, flow_cms or flow_cfs
- ERA5 CSVs exist per site/month under data/raw/era5/** with at least timestamp and environmental columns
- NLCD metrics exist at data/raw/land_use/nlcd_2021_land_use_metrics.csv with site-level or bbox-level features

Notes:
- If ERA5 is 6-hourly, we align to hourly via reindex with nearest tolerance=3H (no forward-looking leakage)
- We filter to timestamps where both NWM and USGS obs exist to form targets
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Iterable

from datetime import datetime

# Local config (study sites)
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from config.master_study_sites import MASTER_STUDY_SITES


def _read_many_csv(patterns: List[str], usecols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        return None
    frames = []
    for f in sorted(files):
        try:
            frames.append(pd.read_csv(f, usecols=usecols))
        except Exception:
            continue
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_nwm_hourly(raw_dir: str) -> pd.DataFrame:
    """Load hourly NWM analysis (CHRTOUT) from retrospective sources only.
    Returns columns: [timestamp, site_name, comid, nwm_cms]
    """
    pats = [
        os.path.join(raw_dir, 'nwm_v3', 'retrospective', '*.csv'),
    ]
    df = _read_many_csv(pats)
    if df is None or df.empty:
        raise FileNotFoundError("No NWM hourly CSVs found under data/raw/nwm_v3/retrospective")
    # Normalize columns
    if 'timestamp' not in df.columns:
        raise ValueError("NWM retrospective CSVs must include 'timestamp'")
    # Robust timestamp parse: handle mixed date-only (YYYY-MM-DD) and date-time strings
    # Strip whitespace first to avoid residuals
    df['timestamp'] = df['timestamp'].astype(str).str.strip()
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    except TypeError:
        # pandas < 2.2 fallback (no format='mixed')
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=False)
    # Drop rows that failed to parse
    df = df.dropna(subset=['timestamp'])
    flow_col = 'streamflow_cms' if 'streamflow_cms' in df.columns else (
        'nwm_cms' if 'nwm_cms' in df.columns else None
    )
    if flow_col is None:
        if 'value' in df.columns:
            flow_col = 'value'
        else:
            raise ValueError("NWM retrospective CSVs missing streamflow column (expected streamflow_cms or nwm_cms)")
    df = df.rename(columns={flow_col: 'nwm_cms'})
    keep = ['timestamp', 'site_name', 'comid', 'nwm_cms']
    return df[keep]


def load_usgs_hourly(raw_dir: str, usgs_id: str) -> pd.DataFrame:
    pats = [os.path.join(raw_dir, 'usgs', f"*{usgs_id}*.csv")]
    df = _read_many_csv(pats)
    if df is None or df.empty:
        raise FileNotFoundError(f"No USGS CSVs found for {usgs_id}")
    # Expect timestamp, flow_cms (or similar). Try common variants.
    if 'timestamp' not in df.columns:
        # Try typical column names
        for c in ['datetime', 'time', 'date_time', 'time_utc']:
            if c in df.columns:
                df['timestamp'] = df[c]
                break
    # Normalize to naive UTC (no tz) for consistent merges
    ts = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df['timestamp'] = ts.dt.tz_localize(None)

    # Identify flow column; support both cms and cfs (convert to cms)
    flow_col = None
    cms_candidates = ['flow_cms', 'discharge_cms', 'streamflow_cms', 'value_cms']
    for c in cms_candidates:
        if c in df.columns:
            flow_col = c
            unit = 'cms'
            break
    if flow_col is None:
        if 'flow_cfs' in df.columns:
            flow_col = 'flow_cfs'
            unit = 'cfs'
        elif 'value' in df.columns:
            flow_col = 'value'
            unit = 'unknown'
        else:
            raise ValueError(
                "USGS CSV missing flow column (expected one of flow_cms/discharge_cms/streamflow_cms/value_cms/flow_cfs/value)"
            )

    out = df[['timestamp', flow_col]].copy()
    if unit == 'cfs':
        # Convert cubic feet per second to cubic meters per second
        out['usgs_cms'] = pd.to_numeric(out[flow_col], errors='coerce') * 0.028316846592
    else:
        out['usgs_cms'] = pd.to_numeric(out[flow_col], errors='coerce')
    out = out[['timestamp', 'usgs_cms']]
    # Hourly expectation: if finer than hourly, resample; if coarser, upsample with no forward look
    out = (
        out.set_index('timestamp')
           .sort_index()
           .resample('1H').mean()
           .reset_index()
    )
    return out


def load_era5_features(raw_dir: str, site_name: str) -> Optional[pd.DataFrame]:
    # ERA5 files likely live under data/raw/era5/<site>/*.csv, or per-site folders like data/raw/era5/<comid>/*.csv.
    # Try nested globs and a general fallback.
    possible = [
        os.path.join(raw_dir, 'era5', site_name.replace(' ', '_').lower(), '*.csv'),
        os.path.join(raw_dir, 'era5', '**', '*.csv'),
        os.path.join(raw_dir, 'era5', '*.csv'),
    ]
    df = _read_many_csv(possible)
    if df is None or df.empty:
        return None
    # Require timestamp column
    if 'timestamp' not in df.columns:
        for c in ['time', 'datetime']:
            if c in df.columns:
                df['timestamp'] = df[c]
                break
    if 'timestamp' not in df.columns:
        return None
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Drop obvious non-feature columns
    drop_like = {'lat', 'lon', 'latitude', 'longitude', 'x', 'y', 'site_name', 'comid'}
    feature_cols = [c for c in df.columns if c not in drop_like and c != 'timestamp']
    # Ensure numeric
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # Coerce to hourly: if 6-hourly, reindex to hourly via nearest within 3H (no leakage across >3H gaps)
    df = df[['timestamp'] + feature_cols].drop_duplicates('timestamp').sort_values('timestamp')
    # Build an hourly timeline over the data span
    full_index = pd.date_range(df['timestamp'].min(), df['timestamp'].max(), freq='1H')
    hourly = (
        df.set_index('timestamp')
          .reindex(full_index, method=None)
          .reset_index()
          .rename(columns={'index': 'timestamp'})
    )
    # Fill by nearest within 3H window
    for c in feature_cols:
        hourly[c] = hourly[c].fillna(method='ffill', limit=3)  # up to 3 hours forward from last known
        # do not backfill to avoid using future info; leave NaNs if gap is at start
    return hourly


def load_nlcd_metrics(raw_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(raw_dir, 'land_use', 'nlcd_2021_land_use_metrics.csv')
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    # Expect site_name or some key; we’ll attempt a permissive join later
    return df


def build_dataset(
    raw_dir: str = 'data/raw',
    out_dir: str = 'data/clean/modeling',
    start: Optional[str] = None,
    end: Optional[str] = None,
    sites: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)

    # Load NWM (multi-site, hourly only)
    nwm = load_nwm_hourly(raw_dir)
    if start:
        nwm = nwm[nwm['timestamp'] >= pd.to_datetime(start)]
    if end:
        nwm = nwm[nwm['timestamp'] <= pd.to_datetime(end)]

    frames = []
    target_sites = (
        {sid: MASTER_STUDY_SITES[sid] for sid in sites if sid in MASTER_STUDY_SITES}
        if sites
        else MASTER_STUDY_SITES
    )

    if sites and not target_sites:
        raise ValueError(f"Requested sites {sites} not found in MASTER_STUDY_SITES")

    for site_id, info in target_sites.items():
        site_name = info['name']
        usgs_id = info.get('usgs_id') or info.get('usgs_site') or info.get('usgs')
        comid = info.get('nwm_comid')
        if not (usgs_id and comid):
            continue
        # Subset NWM for this site/COMID
        nwm_site = nwm[nwm['comid'] == comid].copy()
        if nwm_site.empty:
            continue
        # Load USGS hourly
        usgs = load_usgs_hourly(raw_dir, usgs_id)
        # Load ERA5 features (optional)
        era5 = load_era5_features(raw_dir, site_name)
        # Merge
        df = nwm_site.merge(usgs, on='timestamp', how='inner')  # require obs to build targets
        if era5 is not None:
            df = df.merge(era5, on='timestamp', how='left')
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        # Targets
        df['y_residual_cms'] = df['usgs_cms'] - df['nwm_cms']
        df['y_corrected_cms'] = df['usgs_cms']
        df['site_name'] = site_name  # ensure present consistently
        frames.append(df)

    if not frames:
        raise RuntimeError("No aligned records found across NWM and USGS. Ensure inputs exist and overlap in time.")

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(['site_name', 'timestamp'])
    out = out.drop_duplicates(subset=['site_name', 'timestamp'])

    # Attach NLCD (static). Perform a permissive left-join by site_name if available.
    nlcd = load_nlcd_metrics(raw_dir)
    if nlcd is not None:
        key = 'site_name'
        if key in nlcd.columns:
            # avoid column collisions
            dup_cols = [c for c in nlcd.columns if c in out.columns and c != key]
            nlcd_ren = nlcd.drop(columns=dup_cols)
            out = out.merge(nlcd_ren, on=key, how='left')

    # Save outputs
    if sites:
        suffix = "_".join(sorted(sites))
        tag_base = suffix or "subset"
    else:
        tag_base = "all_sites"
    date_span = f"{(start or out['timestamp'].min().strftime('%Y%m%d'))}_{(end or out['timestamp'].max().strftime('%Y%m%d'))}"
    tag = f"{tag_base}_{date_span}"
    parquet_path = os.path.join(out_dir, f"hourly_training_{tag}.parquet")
    csv_sample_path = os.path.join(out_dir, f"hourly_training_{tag}_sample.csv")
    out.to_parquet(parquet_path, index=False)
    out.head(2000).to_csv(csv_sample_path, index=False)
    print(f"Saved: {parquet_path} (rows={len(out)})")
    print(f"Sample: {csv_sample_path}")
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Build hourly, multi-year residual training dataset')
    ap.add_argument('--raw-dir', default='data/raw', help='Root folder for raw source CSVs')
    ap.add_argument('--out-dir', default='data/clean/modeling', help='Output folder for modeling dataset')
    ap.add_argument('--start', default=None, help='Start date (YYYY-MM-DD) to filter NWM/USGS')
    ap.add_argument('--end', default=None, help='End date (YYYY-MM-DD) to filter NWM/USGS')
    ap.add_argument('--sites', nargs='*', default=None, help='Optional list of site IDs to process')
    args = ap.parse_args()
    build_dataset(raw_dir=args.raw_dir, out_dir=args.out_dir, start=args.start, end=args.end, sites=args.sites)

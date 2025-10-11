#!/usr/bin/env python3
"""
🔄 NWM v3.0+ HOURLY Data Collector
=================================

This module provides multiple ways to assemble hourly NWM streamflow (channel_rt/CHRTOUT)
for model training/testing, with clear source selection by date range:

- Retrospective v3.0 (2021–early 2023):
    Bucket: s3://noaa-nwm-retrospective-3-0-pds
    Path:   CONUS/netcdf/CHRTOUT/{YYYY}/{YYYYMMDDHHMM}.CHRTOUT_DOMAIN1
    Notes:  Hourly analysis NetCDF; one file per valid time. Coverage in 2023 commonly
                    ends on 2023-01-31 23:00Z in the public archive.

- Operational Analysis Assimilation (tm00) for 2023-02 onward:
    Bucket: s3://noaa-nwm-pds
    Path:   nwm.{YYYYMMDD}/analysis_assim/CHRTOUT/nwm.t{HH}z.analysis_assim.channel_rt.tm00.conus.nc
    Notes:  One file per analysis cycle hour (valid time = {YYYY-MM-DD}T{HH}:00Z).

Auto mode
- For a requested window that straddles 2023-02-01T00:00Z, we pull:
    - Retrospective v3.0 up to 2023-01-31T23:00Z, and
    - Analysis Assimilation tm00 from 2023-02-01T00:00Z onward,
    and stitch them into one unified hourly CSV.

This ensures end-to-end hourly coverage across the January→February boundary.
"""

import boto3
from botocore.config import Config
from botocore import UNSIGNED
import xarray as xr
import pandas as pd
import numpy as np
import os
import logging
import sys
import argparse
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')
import tempfile
import requests
import time
import concurrent.futures as cf
try:  # optional dependency for full_physics .comp files (zstd compression)
    import zstandard as zstd
except ImportError:  # pragma: no cover
    zstd = None
try:
    import s3fs  # for fast, anonymous S3 access
except Exception:  # pragma: no cover
    s3fs = None


def _finalize_hourly_frame(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Clean and normalize an hourly streamflow dataframe prior to persistence."""
    if df is None or df.empty:
        return df
    df = df.copy()
    ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df['timestamp'] = ts.dt.tz_convert(None)
    df = df.dropna(subset=['timestamp'])
    subset_cols = [col for col in ['timestamp', 'comid'] if col in df.columns]
    if len(subset_cols) < 2 and 'site_name' in df.columns:
        subset_cols.append('site_name')
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols)
    sort_cols = ['timestamp'] + [col for col in ['site_name', 'comid'] if col in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    if 'hour' in df.columns:
        df['hour'] = df['timestamp'].dt.hour.astype(int)
    return df

# Retrospective storage switch point (compressed full_physics from Feb 2023 onward)
RETRO_FULL_PHYSICS_START = pd.Timestamp('2023-02-01 00:00:00')


def _chrout_key_candidates(ts: pd.Timestamp) -> List[Tuple[str, bool]]:
    """Return list of (key, is_comp) candidates for a given timestamp."""
    base = f"{ts:%Y%m%d%H%M}.CHRTOUT_DOMAIN1"
    if ts >= RETRO_FULL_PHYSICS_START:
        return [
            (f"full_physics/{base}.comp", True),
            (f"CONUS/netcdf/CHRTOUT/{ts:%Y}/{base}", False),
        ]
    return [(f"CONUS/netcdf/CHRTOUT/{ts:%Y}/{base}", False)]


def _decompress_comp(src_path: str, dst_path: str) -> str:
    """Decompress a .comp file (zstd) to NetCDF; returns output path."""
    if zstd is None:
        raise RuntimeError(
            "zstandard package required to read NWM full_physics .comp files. "
            "Install via `pip install zstandard`."
        )
    dctx = zstd.ZstdDecompressor()
    with open(src_path, 'rb') as src, open(dst_path, 'wb') as dst:
        dctx.copy_stream(src, dst)
    return dst_path


# ---- Process-safe worker for CHRTOUT hour fetch ----
def _fetch_chrout_hour_worker(args):
    """
    Process-safe worker to fetch a single CHRTOUT hour and extract flows for provided study sites.
    Falls back to boto3 local download to avoid s3fs/h5py thread-safety issues.
    args: (bucket: str, ts_iso: str, study_sites: List[dict])
    returns: List[dict]
    """
    bucket, ts_iso, study_sites = args
    ts = pd.Timestamp(ts_iso)
    candidates = _chrout_key_candidates(ts)
    # Try boto3 head + download to local temp, then open with h5netcdf (reads minimal data per file)
    from botocore.config import Config as _Cfg
    from botocore import UNSIGNED as _UNS
    import boto3 as _boto3
    import xarray as _xr
    import numpy as _np
    import os as _os
    import tempfile as _tempfile
    try:
        s3c = _boto3.client('s3', region_name='us-east-1', config=_Cfg(signature_version=_UNS))
        key = None
        is_comp = False
        for cand, is_comp_cand in candidates:
            try:
                s3c.head_object(Bucket=bucket, Key=cand)
                key = cand
                is_comp = is_comp_cand
                break
            except Exception:
                continue
        if key is None:
            return []
        tmp_dir = _tempfile.mkdtemp(prefix="nwm_v3_proc_")
        tmp_file = _os.path.join(tmp_dir, _os.path.basename(key))
        nc_path = None
        try:
            s3c.download_file(bucket, key, tmp_file)
            if is_comp:
                nc_path = tmp_file + '.nc'
                _decompress_comp(tmp_file, nc_path)
            else:
                nc_path = tmp_file
            # Use default engine (netCDF4) as in the previously working code path
            with _xr.open_dataset(nc_path) as ds:
                if 'feature_id' not in ds or 'streamflow' not in ds:
                    return []
                feature_ids = _np.array(ds['feature_id'].values)
                values = _np.array(ds['streamflow'].values)
                out = []
                for site in study_sites:
                    comid = site['comid']
                    site_name = site['name']
                    match = _np.where(feature_ids == comid)[0]
                    if match.size:
                        out.append({
                            'timestamp': pd.Timestamp(ts),
                            'site_name': site_name,
                            'comid': comid,
                            'streamflow_cms': float(values[int(match[0])]),
                            'data_source': 'retrospective_v3p0_hourly',
                            'hour': int(ts.hour),
                            'file': key,
                        })
                return out
        finally:
            try:
                if nc_path and _os.path.exists(nc_path) and nc_path != tmp_file:
                    _os.remove(nc_path)
                if _os.path.exists(tmp_file):
                    _os.remove(tmp_file)
                _os.rmdir(tmp_dir)
            except Exception:
                pass
    except Exception:
        return []

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import study sites
from config.master_study_sites import MASTER_STUDY_SITES

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NWMHourlyCollector:
    """Collects hourly NWM operational data using short-range forecasts"""
    
    def __init__(self, data_dir="data/raw/nwm_v3"):
        self.data_dir = data_dir
        self.operational_bucket = 'noaa-nwm-pds'
        # Anonymous access to public NOAA bucket
        self.s3_client = boto3.client(
            's3',
            region_name='us-east-1',
            config=Config(signature_version=UNSIGNED)
        )
        
        # Setup study sites
        self.study_sites = [
            {'name': site_info['name'], 'comid': site_info['nwm_comid']} 
            for site_id, site_info in MASTER_STUDY_SITES.items() if site_info.get('nwm_comid')
        ]
        self.target_comids = [site['comid'] for site in self.study_sites]
        
        logger.info(f"✅ Initialized NWM hourly collector for {len(self.study_sites)} sites")
        logger.info(f"🎯 Target COMIDs: {self.target_comids}")

    # ------------------------------
    # Analysis Assimilation (tm00)
    # ------------------------------
    def collect_analysis_assim_hourly_streamflow(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Collect hourly NWM analysis_assim (tm00) CHRTOUT for target COMIDs.

        Source (per hour):
          s3://noaa-nwm-pds/nwm.{YYYYMMDD}/analysis_assim/CHRTOUT/
            nwm.t{HH}z.analysis_assim.channel_rt.tm00.conus.nc

        Valid time is {YYYYMMDD}T{HH}:00Z.
        """
        bucket = 'noaa-nwm-pds'
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        if end_dt < start_dt:
            raise ValueError("end_date must be after start_date")

        logger.info("🚀 COLLECTING OPERATIONAL analysis_assim (tm00) CHRTOUT - HOURLY")
        logger.info("=" * 76)
        logger.info(f"📅 Period: {start_dt} → {end_dt}")
        logger.info("🎯 Product: analysis_assim tm00 channel_rt (CHRTOUT)")

        hours = pd.date_range(start=start_dt, end=end_dt, freq='H')
        rows: List[dict] = []
        for t in hours:
            ymd = t.strftime('%Y%m%d')
            hh = t.strftime('%H')
            key = f"nwm.{ymd}/analysis_assim/CHRTOUT/nwm.t{hh}z.analysis_assim.channel_rt.tm00.conus.nc"
            try:
                # Object existence check
                self.s3_client.head_object(Bucket=bucket, Key=key)
            except Exception:
                logger.debug(f"Missing analysis_assim file for {t}: s3://{bucket}/{key}")
                continue
            # Download and parse
            tmp_file = f"/tmp/nwm_analysis_assim_{ymd}_t{hh}.nc"
            try:
                self.s3_client.download_file(bucket, key, tmp_file)
                with xr.open_dataset(tmp_file) as ds:
                    if 'feature_id' not in ds or 'streamflow' not in ds:
                        logger.debug(f"Unexpected variables in {key}; skipping")
                        continue
                    feature_ids = np.array(ds['feature_id'].values)
                    values = np.array(ds['streamflow'].values)
                    for site in self.study_sites:
                        comid = site['comid']
                        site_name = site['name']
                        match = np.where(feature_ids == comid)[0]
                        if match.size:
                            rows.append({
                                'timestamp': pd.Timestamp(t),
                                'site_name': site_name,
                                'comid': comid,
                                'streamflow_cms': float(values[int(match[0])]),
                                'data_source': 'analysis_assim_tm00_hourly',
                                'hour': int(t.hour),
                                'file': key,
                            })
            except Exception as e:
                logger.debug(f"Parse failed {key}: {e}")
            finally:
                try:
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)
                except Exception:
                    pass

        if not rows:
            logger.error("❌ No analysis_assim hourly data collected for the requested range")
            return None

        df = _finalize_hourly_frame(pd.DataFrame(rows))
        if df is None or df.empty:
            logger.error("❌ No analysis_assim hourly data collected for the requested range")
            return None
        out_dir = os.path.join(self.data_dir, 'operational')
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(
            out_dir,
            f"nwm_v3_hourly_analysis_assim_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv"
        )
        df.to_csv(out_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
        logger.info(f"💾 Saved analysis_assim hourly CSV: {out_file} (rows={len(df)})")
        hrs = sorted(pd.to_datetime(df['timestamp']).dt.hour.unique().tolist())
        logger.info(f"⏰ Hourly coverage (unique hours): {hrs}")
        return df

    def collect_v3_hourly_auto(
        self,
        start_date: str,
        end_date: str,
        archive_base_url: Optional[str] = None,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """Auto-stitch hourly streamflow across Jan→Feb 2023 boundary.

        - Up to 2023-01-31T23:00Z: retrospective v3.0 CHRTOUT
        - From 2023-02-01T00:00Z onward: analysis_assim tm00
        - Optional HTTP archive fallback when analysis_assim objects are no longer on S3
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        cutoff = pd.Timestamp('2023-02-01T00:00:00Z').tz_convert(None)

        frames = []
        if start_dt < cutoff:
            sub_end = min(end_dt, cutoff - pd.Timedelta(hours=1))
            retro = self.collect_retrospective_v3_streamflow(
                start_date=start_dt, end_date=sub_end,
                max_workers=kwargs.get('max_workers', 6),
                checkpoint_every=kwargs.get('checkpoint_every', 200),
                resume=kwargs.get('resume', True),
                concurrency=kwargs.get('concurrency', 'process'),
            )
            if retro is not None:
                frames.append(retro)
        if end_dt >= cutoff:
            sub_start = max(start_dt, cutoff)
            assim = self.collect_analysis_assim_hourly_streamflow(sub_start, end_dt)
            if assim is not None:
                frames.append(assim)
            elif archive_base_url:
                logger.info(
                    "⚠️  analysis_assim dataset unavailable; attempting archive fallback via %s",
                    archive_base_url,
                )
                archive_df = self.collect_hourly_archive_data(
                    start_date=sub_start,
                    end_date=end_dt,
                    base_url=archive_base_url,
                )
                if archive_df is not None:
                    frames.append(archive_df)
                else:
                    logger.warning(
                        "⚠️  Archive fallback also returned no data for %s → %s",
                        sub_start,
                        end_dt,
                    )
            else:
                logger.warning(
                    "⚠️  analysis_assim dataset unavailable for %s → %s; provide --archive-base-url to try HTTP archive fallback",
                    sub_start,
                    end_dt,
                )

        if not frames:
            logger.error("❌ Auto v3 hourly collector produced no data for the requested window")
            return None

        all_df = pd.concat(frames, ignore_index=True)
        all_df = _finalize_hourly_frame(all_df)
        out_dir = os.path.join(self.data_dir, 'retrospective')
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(
            out_dir,
            f"nwm_v3_hourly_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv"
        )
        all_df.to_csv(out_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
        logger.info(f"💾 Saved unified v3 hourly (auto) CSV: {out_file} (rows={len(all_df)})")
        hrs = sorted(pd.to_datetime(all_df['timestamp']).dt.hour.unique().tolist())
        logger.info(f"⏰ Hourly coverage (unique hours): {hrs}")
        return all_df

    def _resample_to_6h(self, df: pd.DataFrame, how: str = "sample") -> pd.DataFrame:
        """Resample hourly streamflow to 6-hour timesteps at 00/06/12/18Z.
        how="sample" keeps exact timesteps; how="mean" averages preceding 6 hours.
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['site_name', 'comid', 'timestamp'])

        if how == "mean":
            # Mean of previous 6 hours aligned to 00/06/12/18
            def agg(group: pd.DataFrame) -> pd.DataFrame:
                g = group.set_index('timestamp').resample('6H').mean(numeric_only=True)
                g = g.reset_index()
                # Reattach constant columns
                g['site_name'] = group['site_name'].iloc[0]
                g['comid'] = group['comid'].iloc[0]
                g['data_source'] = group['data_source'].iloc[0] + "_6h_mean"
                return g
            out = df.groupby(['site_name', 'comid'], group_keys=False).apply(agg)
            out['hour'] = out['timestamp'].dt.hour
            return out
        else:
            # Keep only 6-hour boundaries
            mask = df['timestamp'].dt.hour.isin([0, 6, 12, 18])
            out = df.loc[mask].copy()
            out['data_source'] = out['data_source'] + "_6h_sample"
            return out
    
    def collect_hourly_operational_data(self, start_date="2025-01-01", end_date="2025-08-27"):
        """Collect hourly operational data using short-range forecasts"""
        
        logger.info("🚀 COLLECTING HOURLY OPERATIONAL DATA (Short-Range Forecasts)")
        logger.info("=" * 65)
        logger.info(f"📅 Period: {start_date} to {end_date}")
        logger.info("🎯 Method: Short-range forecasts f001-f024 from 00Z runs")
        logger.info("⏰ Resolution: HOURLY (24 timesteps per day)")
        
        all_data = []
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # The operational S3 bucket (noaa-nwm-pds) retains only a rolling window (weeks to a few months).
        # Guard against queries far in the past (e.g., 2024) which will reliably return no data.
        now_utc = pd.Timestamp.utcnow()
        rolling_window_days = 90  # conservative estimate; public retention varies
        if end_dt < (now_utc - pd.Timedelta(days=rolling_window_days)):
            logger.warning("⚠️  Requested dates appear older than the operational bucket's rolling retention window.")
            logger.warning("   The noaa-nwm-pds bucket generally retains only recent weeks/months of data.")
            logger.warning("   For older periods (e.g., 2024), use --mode retrospective to read the v2.1 Zarr archive (through 2020),")
            logger.warning("   or adjust dates to a recent period (e.g., 2025).")
            return None
        
        current_date = start_dt
        while current_date <= end_dt:
            date_str = current_date.strftime("%Y%m%d")
            logger.info(f"📅 Processing {date_str}...")
            
            daily_data = self._process_hourly_operational_date(date_str)
            
            if daily_data:
                all_data.extend(daily_data)
                logger.info(f"   ✅ Found {len(daily_data)} hourly timesteps")
            else:
                logger.warning(f"   ⚠️  No data found for {date_str}")
            
            current_date += timedelta(days=1)
        
        if all_data:
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            df = _finalize_hourly_frame(df)
            
            # Save hourly data
            output_dir = os.path.join(self.data_dir, 'operational')
            os.makedirs(output_dir, exist_ok=True)

            # Name file by requested date range
            output_file = os.path.join(
                output_dir,
                f"nwm_v3_hourly_operational_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv"
            )
            df.to_csv(output_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
            
            logger.info(f"💾 Saved hourly testing data: {output_file}")
            logger.info(f"📊 Hourly dataset: {len(df)} timesteps, {df.columns.tolist()} features")
            
            # Analyze temporal coverage
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            hourly_coverage = sorted(df['hour'].unique())
            
            logger.info(f"⏰ Hourly coverage: {hourly_coverage}")
            logger.info(f"📈 Hours per day: {len(hourly_coverage)}")
            
            if len(hourly_coverage) == 24:
                logger.info("✅ PERFECT: Full 24-hour coverage achieved!")
            else:
                logger.warning(f"⚠️  Partial coverage: {len(hourly_coverage)}/24 hours")
            
            return df
        else:
            logger.error("❌ No hourly data collected!")
            return None

    def _http_head(self, url: str) -> bool:
        """Lightweight HEAD request to check object existence on HTTP server."""
        try:
            r = requests.head(url, timeout=15)
            return r.ok
        except Exception:
            return False

    def _http_download(self, url: str, dst_path: str) -> bool:
        """Stream download to a local file; returns True on success."""
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dst_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception as e:
            logger.debug(f"HTTP download failed: {url} :: {e}")
            return False

    def collect_hourly_archive_data(self, start_date: str, end_date: str, base_url: str) -> Optional[pd.DataFrame]:
        """Collect hourly NWM short-range forecasts (channel_rt) from an HTTP archive for multi-year ranges.

        Expects directory structure like:
          {base_url.rstrip('/')}/nwm.YYYYMMDD/short_range/nwm.t{HH}z.short_range.channel_rt.f{FFF}.conus.nc

        Example candidates (confirm which lists short_range/channel_rt):
          - https://www.ncei.noaa.gov/thredds/fileServer/model-nwm
          - https://www.ncei.noaa.gov/thredds/fileServer/nwm
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        rows: List[dict] = []

        logger.info("🚀 COLLECTING ARCHIVE HOURLY (Short-Range) FORECASTS")
        logger.info("=" * 65)
        logger.info(f"📅 Period: {start_dt.date()} → {end_dt.date()}")
        logger.info(f"🌐 Base URL: {base_url.rstrip('/')}")
        logger.info("🎯 Product: short_range channel_rt f001–f024 (00Z primary, 12Z fallback)")

        cur = start_dt
        while cur <= end_dt:
            ymd = cur.strftime('%Y%m%d')
            logger.info(f"📅 {ymd}")
            found_for_day = 0
            for base_hour in (0, 12):
                for fh in range(1, 25):
                    f3 = f"{fh:03d}"
                    key = f"nwm.{ymd}/short_range/nwm.t{base_hour:02d}z.short_range.channel_rt.f{f3}.conus.nc"
                    url = f"{base_url.rstrip('/')}/{key}"
                    tmp_path = None
                    try:
                        if not self._http_head(url):
                            continue
                        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
                            tmp_path = tmp.name
                        if not self._http_download(url, tmp_path):
                            if tmp_path and os.path.exists(tmp_path):
                                os.remove(tmp_path)
                            continue
                        with xr.open_dataset(tmp_path) as ds:
                            init_time = pd.Timestamp(year=cur.year, month=cur.month, day=cur.day, hour=base_hour)
                            valid_time = init_time + pd.Timedelta(hours=fh)
                            if 'feature_id' in ds and 'streamflow' in ds:
                                feature_ids = np.array(ds['feature_id'].values)
                                values = np.array(ds['streamflow'].values)
                                for site in self.study_sites:
                                    comid = site['comid']
                                    site_name = site['name']
                                    match = np.where(feature_ids == comid)[0]
                                    if match.size:
                                        idx = int(match[0])
                                        rows.append({
                                            'init_time': init_time,
                                            'timestamp': valid_time,
                                            'site_name': site_name,
                                            'comid': comid,
                                            'streamflow_cms': float(values[idx]),
                                            'data_source': 'short_range_forecast_archive',
                                            'hour': valid_time.hour,
                                            'forecast_hour': f"f{f3}",
                                            'lead_hour': fh,
                                            'file': key,
                                        })
                                        found_for_day += 1
                    except Exception as e:
                        logger.debug(f"Skip URL {url}: {e}")
                    finally:
                        try:
                            if tmp_path and os.path.exists(tmp_path):
                                os.remove(tmp_path)
                        except Exception:
                            pass
            if found_for_day == 0:
                logger.warning(f"   ⚠️  No files found for {ymd} under {base_url}")
            else:
                logger.info(f"   ✅ Collected {found_for_day} records for {ymd}")
            cur += timedelta(days=1)

        if not rows:
            logger.error("❌ No archive hourly data collected for the requested range")
            return None

        df = pd.DataFrame(rows).sort_values(['timestamp', 'site_name', 'lead_hour'])
        out_dir = os.path.join(self.data_dir, 'archive')
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(
            out_dir,
            f"nwm_v3_hourly_archive_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv"
        )
        df.to_csv(out_file, index=False)
        logger.info(f"💾 Saved archive hourly CSV: {out_file} (rows={len(df)})")

        hrs = sorted(pd.to_datetime(df['timestamp']).dt.hour.unique().tolist())
        logger.info(f"⏰ Hourly coverage (unique hours): {hrs}")
        return df

    def collect_retrospective_streamflow(self, start_date: str = "2020-01-01", end_date: str = "2020-01-10",
                                         resample_6h: bool = False, resample_method: str = "sample") -> Optional[pd.DataFrame]:
        """Collect hourly retrospective (v2.1) streamflow from Zarr for the selected COMIDs.
        Data source: s3://noaa-nwm-retrospective-2-1-zarr-pds/chrtout.zarr (hourly)
        """
        try:
            import fsspec  # noqa: F401 (ensures s3fs path handling)
        except ImportError:
            logger.error("Missing fsspec/s3fs. Please install 's3fs' to read S3 Zarr stores.")
            raise

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        logger.info("🚀 COLLECTING RETROSPECTIVE (v2.1) STREAMFLOW - HOURLY via Zarr")
        logger.info("=" * 70)
        logger.info(f"📅 Period: {start_date} to {end_date}")
        logger.info("🎯 Product: chrtout (streamflow) — hourly resolution")

        zarr_url = "s3://noaa-nwm-retrospective-2-1-zarr-pds/chrtout.zarr"
        logger.info(f"📦 Opening Zarr store: {zarr_url} (anonymous)")

        # Open dataset lazily
        ds = xr.open_zarr(
            store="s3://noaa-nwm-retrospective-2-1-zarr-pds/chrtout.zarr",
            storage_options={"anon": True},
            consolidated=True
        )
        # Subset by time
        ds_sub = ds.sel(time=slice(np.datetime64(start_dt), np.datetime64(end_dt)))

        # Build DataFrame for each site
        frames: List[pd.DataFrame] = []
        for site in self.study_sites:
            comid = site['comid']
            site_name = site['name']
            try:
                # Use xarray selection by coordinate value
                ts = ds_sub['streamflow'].sel(feature_id=comid)
                df = ts.to_dataframe(name='streamflow_cms').reset_index()
                df.rename(columns={"time": "timestamp"}, inplace=True)
                df['site_name'] = site_name
                df['comid'] = comid
                df['data_source'] = 'retrospective_v2p1_hourly'
                df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                frames.append(df[['timestamp', 'site_name', 'comid', 'streamflow_cms', 'data_source', 'hour']])
            except KeyError:
                logger.warning(f"COMID {comid} not found in retrospective dataset; skipping")
                continue

        if not frames:
            logger.error("❌ No retrospective data collected for requested sites/range")
            return None

        out = pd.concat(frames, ignore_index=True).sort_values(['timestamp', 'site_name'])

        # Report temporal coverage
        hours = sorted(pd.to_datetime(out['timestamp']).dt.hour.unique())
        logger.info(f"⏰ Hourly coverage in sample: {hours} (count={len(hours)})")
        if len(hours) == 24:
            logger.info("✅ HOURLY: 24 hours per day confirmed for retrospective chrtout")
        else:
            logger.warning("⚠️ Unexpected hour coverage; dataset may have gaps in this period")

        # Save hourly
        output_dir = os.path.join(self.data_dir, 'retrospective')
        os.makedirs(output_dir, exist_ok=True)
        hourly_file = os.path.join(output_dir, f"nwm_v2p1_hourly_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv")
        out.to_csv(hourly_file, index=False)
        logger.info(f"💾 Saved retrospective hourly CSV: {hourly_file} (rows={len(out)})")

        if resample_6h:
            out6 = self._resample_to_6h(out, how=resample_method)
            six_file = os.path.join(output_dir, f"nwm_v2p1_6h_{resample_method}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.csv")
            out6.to_csv(six_file, index=False)
            logger.info(f"💾 Saved retrospective 6-hour CSV: {six_file} (rows={len(out6)})")
            return out6

        return out

    def collect_retrospective_v3_streamflow(self, start_date: str, end_date: str, max_workers: int = 6, checkpoint_every: int = 200, resume: bool = False, concurrency: str = "process") -> Optional[pd.DataFrame]:
        """Collect hourly NWM v3.0 retrospective CHRTOUT (2021–2023) for target COMIDs.

        Source: s3://noaa-nwm-retrospective-3-0-pds/CONUS/netcdf/CHRTOUT/{YYYY}/{YYYYMMDDHHMM}.CHRTOUT_DOMAIN1

        Each hourly NetCDF contains streamflow for all feature_id; we extract only our COMIDs.
        """
        bucket = "noaa-nwm-retrospective-3-0-pds"
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        if end_dt < start_dt:
            raise ValueError("end_date must be after start_date")
        if end_dt >= RETRO_FULL_PHYSICS_START and zstd is None:
            raise RuntimeError(
                "zstandard package required to read NWM retrospective full_physics (.comp) files. "
                "Install via `pip install zstandard` and retry."
            )

        logger.info("🚀 COLLECTING RETROSPECTIVE v3.0 (2021–2023) CHRTOUT - HOURLY")
        logger.info("=" * 70)
        logger.info(f"📅 Period: {start_dt} → {end_dt}")
        logger.info("🎯 Product: CHRTOUT hourly NetCDF (CONUS DOMAIN1)")

        # Output path and resume
        out_dir = os.path.join(self.data_dir, 'retrospective')
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(
            out_dir,
            f"nwm_v3_hourly_{pd.to_datetime(start_dt).strftime('%Y%m%d')}_{pd.to_datetime(end_dt).strftime('%Y%m%d')}.csv"
        )
        already = set()
        if resume and os.path.exists(out_file):
            try:
                prev = pd.read_csv(out_file, usecols=["timestamp"])  # small read
                already = set(pd.to_datetime(prev["timestamp"]).astype("datetime64[ns]").tolist())
                logger.info(f"↩️  Resume enabled: {len(already)} timestamps already written")
            except Exception:
                already = set()

        # Build hours to fetch
        hours = pd.date_range(start=start_dt, end=end_dt, freq="H")
        hours = [pd.Timestamp(t) for t in hours if t.to_datetime64() not in already]
        if not hours:
            logger.info(f"Nothing to fetch; file exists with requested period: {out_file}")
            return pd.read_csv(out_file, parse_dates=["timestamp"]) if os.path.exists(out_file) else None

        # Prepare s3fs if available
        fs = None
        if s3fs is not None:
            try:
                fs = s3fs.S3FileSystem(anon=True)
            except Exception:
                fs = None

        def fetch_one(ts: pd.Timestamp):
            candidates = _chrout_key_candidates(ts)
            # First, if s3fs available, try netcdf variant directly (only works for non-comp)
            if fs is not None:
                for key, is_comp in candidates:
                    if is_comp:
                        continue
                    try:
                        if not fs.exists(f"{bucket}/{key}"):
                            continue
                        s3url = f"s3://{bucket}/{key}"
                        ds = xr.open_dataset(
                            s3url,
                            engine="h5netcdf",
                            backend_kwargs={"storage_options": {"anon": True}},
                            chunks={},
                        )
                        if 'feature_id' not in ds or 'streamflow' not in ds:
                            ds.close()
                            continue
                        feature_ids = np.asarray(ds['feature_id'].values)
                        values = np.asarray(ds['streamflow'].values)
                        out = []
                        for site in self.study_sites:
                            comid = site['comid']
                            site_name = site['name']
                            idx = np.where(feature_ids == comid)[0]
                            if idx.size:
                                out.append({
                                    'timestamp': pd.Timestamp(ts),
                                    'site_name': site_name,
                                    'comid': comid,
                                    'streamflow_cms': float(values[int(idx[0])]),
                                    'data_source': 'retrospective_v3p0_hourly',
                                    'hour': int(ts.hour),
                                    'file': key,
                                })
                        ds.close()
                        if out:
                            return out
                    except Exception:
                        continue

            # Fallback: download candidate, optionally decompress, then open
            for key, is_comp in candidates:
                try:
                    self.s3_client.head_object(Bucket=bucket, Key=key)
                except Exception:
                    continue
                tmp_file = None
                nc_path = None
                try:
                    tmp_dir = tempfile.mkdtemp(prefix="nwm_v3_dl_")
                    tmp_file = os.path.join(tmp_dir, os.path.basename(key))
                    self.s3_client.download_file(bucket, key, tmp_file)
                    if is_comp:
                        nc_path = tmp_file + '.nc'
                        _decompress_comp(tmp_file, nc_path)
                        open_path = nc_path
                    else:
                        open_path = tmp_file
                    with xr.open_dataset(open_path) as ds:
                        if 'feature_id' not in ds or 'streamflow' not in ds:
                            continue
                        feature_ids = np.array(ds['feature_id'].values)
                        values = np.array(ds['streamflow'].values)
                        out = []
                        for site in self.study_sites:
                            comid = site['comid']
                            site_name = site['name']
                            match = np.where(feature_ids == comid)[0]
                            if match.size:
                                out.append({
                                    'timestamp': pd.Timestamp(ts),
                                    'site_name': site_name,
                                    'comid': comid,
                                    'streamflow_cms': float(values[int(match[0])]),
                                    'data_source': 'retrospective_v3p0_hourly',
                                    'hour': int(ts.hour),
                                    'file': key,
                                })
                        if out:
                            return out
                except Exception:
                    continue
                finally:
                    try:
                        if nc_path and os.path.exists(nc_path):
                            os.remove(nc_path)
                        if tmp_file and os.path.exists(tmp_file):
                            os.remove(tmp_file)
                        if 'tmp_dir' in locals():
                            os.rmdir(tmp_dir)
                    except Exception:
                        pass
            return []

        # Concurrent fetch and periodic checkpoint writes
        rows: List[dict] = []
        last_flush = time.time()
        processed = 0
        if concurrency == "thread":
            executor_cls = cf.ThreadPoolExecutor
            iterator = hours
            mapper = fetch_one
        else:
            # Use processes with a process-safe worker to avoid HDF5 thread issues
            executor_cls = cf.ProcessPoolExecutor
            # Pack minimal args to avoid pickling large objects
            packed = [(bucket, str(ts), self.study_sites) for ts in hours]
            iterator = packed
            mapper = _fetch_chrout_hour_worker

        with executor_cls(max_workers=max_workers) as ex:
            for recs in ex.map(mapper, iterator):
                processed += 1
                if recs:
                    # process mode returns list of dicts; thread mode returns list too
                    rows.extend(recs)
                if rows and (processed % checkpoint_every == 0 or (time.time() - last_flush) > 30):
                    df_ck = pd.DataFrame(rows).sort_values(['timestamp', 'site_name'])
                    mode = 'a' if os.path.exists(out_file) else 'w'
                    df_ck.to_csv(out_file, index=False, mode=mode, header=not os.path.exists(out_file))
                    logger.info(f"💾 Checkpoint flush: +{len(df_ck)} rows → {out_file}")
                    rows.clear()
                    last_flush = time.time()

        # Final flush
        if rows:
            df_ck = pd.DataFrame(rows).sort_values(['timestamp', 'site_name'])
            mode = 'a' if os.path.exists(out_file) else 'w'
            df_ck.to_csv(out_file, index=False, mode=mode, header=not os.path.exists(out_file))
            rows.clear()

        if not os.path.exists(out_file):
            logger.error("❌ No v3.0 retrospective data collected for requested range")
            return None

        df = pd.read_csv(out_file)
        df = _finalize_hourly_frame(df)
        df.to_csv(out_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
        hrs = sorted(df['timestamp'].dt.hour.dropna().unique().tolist())
        logger.info(f"⏰ Hourly coverage (unique hours): {hrs}")
        logger.info(f"💾 Saved v3.0 retrospective hourly CSV: {out_file} (rows={len(df)})")
        return df
    
    def _process_hourly_operational_date(self, date_str):
        """Process a single date to get 24 hours of forecast data"""
        
        daily_data = []
        successful_hours = 0
        
        # Use 00Z short-range forecasts for f001-f024 (24 hourly forecasts)
        # Try 00Z first, then 12Z fallback
        for base_hour in (0, 12):
            for forecast_hour in range(1, 25):  # f001 to f024
                file_pattern = f"nwm.{date_str}/short_range/nwm.t{base_hour:02d}z.short_range.channel_rt.f{forecast_hour:03d}.conus.nc"
                try:
                    # Head to confirm existence
                    self.s3_client.head_object(Bucket=self.operational_bucket, Key=file_pattern)
                    # Download and process
                    temp_file = f"/tmp/nwm_hourly_{date_str}_t{base_hour:02d}_f{forecast_hour:03d}.nc"
                    self.s3_client.download_file(self.operational_bucket, file_pattern, temp_file)
                    with xr.open_dataset(temp_file) as ds:
                        init_time = pd.to_datetime(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {base_hour:02d}:00:00")
                        valid_time = init_time + pd.Timedelta(hours=forecast_hour)
                        if 'feature_id' in ds and 'streamflow' in ds:
                            feature_ids = ds.feature_id.values
                            streamflow_values = ds.streamflow.values
                            for site in self.study_sites:
                                comid = site['comid']
                                site_name = site['name']
                                if comid in feature_ids:
                                    idx = np.where(feature_ids == comid)[0][0]
                                    flow_value = float(streamflow_values[idx])
                                    daily_data.append({
                                        'init_time': init_time,
                                        'timestamp': valid_time,
                                        'site_name': site_name,
                                        'comid': comid,
                                        'streamflow_cms': flow_value,
                                        'data_source': 'short_range_forecast',
                                        'hour': valid_time.hour,
                                        'forecast_hour': f"f{forecast_hour:03d}",
                                        'lead_hour': forecast_hour,
                                        'file': file_pattern
                                    })
                    successful_hours += 1
                    os.remove(temp_file)
                except Exception:
                    # Missing file; try next hour or fallback base_hour
                    continue
        
        return daily_data
    
    def validate_temporal_consistency(self):
        """Validate that training and testing data now have consistent temporal resolution"""
        
        logger.info("🔍 VALIDATING TEMPORAL CONSISTENCY")
        logger.info("=" * 40)
        
        training_hours = None
        # Load training data
        training_file = os.path.join(self.data_dir, 'retrospective', 'nwm_v3_training_2020_2023.csv')
        if os.path.exists(training_file):
            training_df = pd.read_csv(training_file)
            training_df['timestamp'] = pd.to_datetime(training_df['timestamp'])
            training_df['hour'] = training_df['timestamp'].dt.hour
            training_hours = sorted(training_df['hour'].unique())
            
            logger.info(f"📚 Training data hours: {training_hours}")
            logger.info(f"📚 Training resolution: {len(training_hours)} hours/day")
        
        # Load new hourly testing data
        testing_file = os.path.join(self.data_dir, 'operational', 'nwm_v3_hourly_testing_2025.csv')
        if os.path.exists(testing_file):
            testing_df = pd.read_csv(testing_file)
            testing_df['timestamp'] = pd.to_datetime(testing_df['timestamp'])
            testing_df['hour'] = testing_df['timestamp'].dt.hour
            testing_hours = sorted(testing_df['hour'].unique())
            
            logger.info(f"🧪 Testing data hours: {testing_hours}")
            logger.info(f"🧪 Testing resolution: {len(testing_hours)} hours/day")
            
            # Check consistency
            if training_hours is not None and set(training_hours) == set(testing_hours):
                logger.info("✅ PERFECT: Temporal resolutions now match!")
                logger.info("🎯 Ready for consistent model training")
                return True
            else:
                if training_hours is None:
                    logger.info("ℹ️ Training file not found; skipping cross-dataset comparison")
                    return True
                logger.warning("⚠️  Temporal mismatch still exists")
                missing_in_testing = set(training_hours) - set(testing_hours)
                missing_in_training = set(testing_hours) - set(training_hours)
                if missing_in_testing:
                    logger.warning(f"   Missing in testing: {sorted(missing_in_testing)}")
                if missing_in_training:
                    logger.warning(f"   Missing in training: {sorted(missing_in_training)}")
                return False
        
        return False

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description="Collect hourly NWM data (operational, retrospective, or archive).")
    parser.add_argument("--out-dir", default="data/raw/nwm_v3", help="Output directory for CSV files.")
    parser.add_argument("--start-date", default="2025-01-01", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d"), help="End date in YYYY-MM-DD format.")
    parser.add_argument("--mode", choices=["operational", "retrospective", "retrospective_v3", "archive", "v3_auto"], default="operational", help="Data source mode: operational short-range, retrospective v2.1 (Zarr), retrospective v3.0 CHRTOUT, analysis_assim+retrospective auto-stitch (v3_auto), or HTTP archive.")
    parser.add_argument("--resample-6h", action="store_true", help="For retrospective mode, also write a 6-hour dataset aligned at 00/06/12/18Z.")
    parser.add_argument("--resample-method", choices=["sample", "mean"], default="sample", help="6-hour resampling method: sample exact hours or mean of prior 6 hours.")
    parser.add_argument("--archive-base-url", default=None, help="HTTP base URL for archived NWM, e.g., https://www.ncei.noaa.gov/thredds/fileServer/model-nwm")
    # Performance knobs for retrospective_v3
    parser.add_argument("--max-workers", type=int, default=6, help="Parallel workers for CHRTOUT fetch (retrospective_v3).")
    parser.add_argument("--checkpoint-every", type=int, default=200, help="Flush rows to CSV every N records (retrospective_v3).")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume and skip timestamps already present in output CSV (retrospective_v3). Default: True")
    parser.add_argument("--concurrency", choices=["thread", "process"], default="process", help="Concurrency model for retrospective_v3: threads (fast but may segfault with HDF5) or processes (safer). Default: process")
    args = parser.parse_args()

    logger.info("🔄 STARTING HOURLY NWM DATA COLLECTION")
    logger.info("=" * 45)
    logger.info("🎯 Goal: Collect hourly operational data using short-range forecasts")
    logger.info("📋 Strategy: Use f001-f024 from 00Z runs for 24-hour coverage")
    
    try:
        # Initialize collector
        collector = NWMHourlyCollector(data_dir=args.out_dir)
        
        if args.mode == "operational":
            # Collect hourly operational data
            hourly_df = collector.collect_hourly_operational_data(start_date=args.start_date, end_date=args.end_date)
            if hourly_df is not None:
                success = collector.validate_temporal_consistency()
                if success:
                    logger.info("✅ HOURLY COLLECTION SUCCESSFUL!")
                    logger.info("🎯 Training and testing now have matching hourly resolution")
                    logger.info("📊 Ready for enhanced model training with consistent timesteps")
                else:
                    logger.warning("⚠️  Temporal validation incomplete")
        elif args.mode == "retrospective":
            # Retrospective v2.1
            retro_df = collector.collect_retrospective_streamflow(
                start_date=args.start_date,
                end_date=args.end_date,
                resample_6h=args.resample_6h,
                resample_method=args.resample_method
            )
            if retro_df is not None:
                logger.info("✅ RETROSPECTIVE COLLECTION COMPLETE")
                if args.resample_6h:
                    logger.info("🎯 6-hour dataset ready for alignment with 6-hour sources (e.g., ERA5)")
        elif args.mode == "retrospective_v3":
            v3 = collector.collect_retrospective_v3_streamflow(
                start_date=args.start_date,
                end_date=args.end_date,
                max_workers=args.max_workers,
                checkpoint_every=args.checkpoint_every,
                resume=args.resume,
                concurrency=args.concurrency,
            )
            if v3 is not None:
                logger.info("✅ RETROSPECTIVE v3.0 COLLECTION COMPLETE")
        elif args.mode == "v3_auto":
            v3a = collector.collect_v3_hourly_auto(
                start_date=args.start_date,
                end_date=args.end_date,
                archive_base_url=args.archive_base_url,
                max_workers=args.max_workers,
                checkpoint_every=args.checkpoint_every,
                resume=args.resume,
                concurrency=args.concurrency,
            )
            if v3a is not None:
                logger.info("✅ AUTO (retrospective + analysis_assim) HOURLY COLLECTION COMPLETE")
        else:
            # Archive HTTP mode
            if not args.archive_base_url:
                logger.error("❌ archive mode requires --archive-base-url (e.g., https://www.ncei.noaa.gov/thredds/fileServer/model-nwm)")
                return
            arc = collector.collect_hourly_archive_data(
                start_date=args.start_date,
                end_date=args.end_date,
                base_url=args.archive_base_url,
            )
            if arc is not None:
                logger.info("✅ ARCHIVE COLLECTION COMPLETE")
        
    except Exception as e:
        logger.error(f"❌ Hourly collection failed: {str(e)}")
        raise e

if __name__ == "__main__":
    main()

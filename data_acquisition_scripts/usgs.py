#!/usr/bin/env python3
"""
Optimized USGS Streamflow Data Fetching Script
Enhanced with robust 503 error handling, circuit breaker pattern, and intelligent retry logic.
"""
import os
import sys
import requests
import pandas as pd
import argparse
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import random
import io
import pytz
from datetime import date as dt_date

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config.master_study_sites import MASTER_STUDY_SITES

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class USGSCollector:
    """Enhanced USGS data collector with circuit breaker pattern and intelligent retry logic."""
    
    def __init__(self, max_retries=6, base_delay=1.0, max_delay=30.0, 
                 circuit_breaker_threshold=5, circuit_breaker_timeout=300):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        
        # Circuit breaker state
        self.consecutive_failures = 0
        self.circuit_open_time = None
        self.circuit_open = False
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # Minimum time between requests
        
    def _wait_for_rate_limit(self):
        """Ensure minimum time between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
    
    def _check_circuit_breaker(self):
        """Check if circuit breaker should be opened or closed."""
        if not self.circuit_open:
            return True
            
        # Check if timeout has elapsed
        if time.time() - self.circuit_open_time > self.circuit_breaker_timeout:
            logger.info("🔄 Circuit breaker cooling period ended, attempting recovery")
            self.circuit_open = False
            self.consecutive_failures = 0
            return True
            
        return False
    
    def _record_failure(self):
        """Record a failure for circuit breaker logic."""
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.circuit_breaker_threshold:
            self.circuit_open = True
            self.circuit_open_time = time.time()
            logger.warning(f"⚡ Circuit breaker OPENED after {self.consecutive_failures} failures. "
                         f"Cooling down for {self.circuit_breaker_timeout/60:.1f} minutes")
    
    def _record_success(self):
        """Record a success for circuit breaker logic."""
        if self.consecutive_failures > 0:
            logger.info(f"✅ Recovery successful after {self.consecutive_failures} failures")
        self.consecutive_failures = 0
        self.circuit_open = False
    
    def fetch_with_retry(self, url: str, max_retries: Optional[int] = None) -> Optional[requests.Response]:
        """
        Enhanced fetch with exponential backoff, circuit breaker, and 503-specific handling.
        """
        if not self._check_circuit_breaker():
            logger.warning("⚡ Circuit breaker is OPEN - skipping request")
            return None
            
        max_retries = max_retries or self.max_retries
        
        for attempt in range(max_retries + 1):
            try:
                # Rate limiting
                self._wait_for_rate_limit()
                
                # Make request with timeout
                self.last_request_time = time.time()
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    self._record_success()
                    return response
                    
                elif response.status_code == 503:
                    # 503 Service Unavailable - USGS server overload
                    if attempt < max_retries:
                        # Exponential backoff with jitter for 503 errors
                        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                        jitter = random.uniform(0.1, 0.5)  # Add small random jitter
                        total_delay = delay + jitter
                        
                        logger.warning(f"🔄 USGS HTTP 503 (Server Overload). Attempt {attempt + 1}/{max_retries + 1}. "
                                     f"Sleeping {total_delay:.1f}s ...")
                        time.sleep(total_delay)
                        continue
                    else:
                        logger.error(f"❌ Final 503 error after {max_retries + 1} attempts")
                        self._record_failure()
                        return None
                        
                else:
                    logger.error(f"❌ HTTP {response.status_code}: {response.reason}")
                    self._record_failure()
                    return None
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    logger.warning(f"⏱️ Timeout. Attempt {attempt + 1}/{max_retries + 1}. Sleeping {delay}s ...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"❌ Final timeout after {max_retries + 1} attempts")
                    self._record_failure()
                    return None
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    logger.warning(f"🌐 Network error: {e}. Attempt {attempt + 1}/{max_retries + 1}. Sleeping {delay}s ...")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"❌ Final network error after {max_retries + 1} attempts: {e}")
                    self._record_failure()
                    return None
        
        return None

NWIS_CSV = (
    "https://waterservices.usgs.gov/nwis/iv/"
    "?format=rdb&sites={site}&startDT={start}&endDT={end}&parameterCd=00060&siteStatus=all"
)

# Helpers
UTC = pytz.UTC


def fetch_range(site_id: str, start: str, end: str, collector: USGSCollector) -> pd.DataFrame:
    url = NWIS_CSV.format(site=site_id, start=start, end=end)
    r = collector.fetch_with_retry(url)
    if r is None:
        return pd.DataFrame()
    # RDB format: skip comment lines (#) and metadata header (5-10 lines)
    lines = [ln for ln in r.text.splitlines() if not ln.startswith("#")]
    # Find header line (starts with 'agency_cd')
    header_idx = next(i for i, ln in enumerate(lines) if ln.startswith("agency_cd"))
    data = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(data), sep="\t")
    if df.empty:
        return df
    # Standardize columns
    # USGS RDB has columns: agency_cd, site_no, datetime, tz_cd, 00060, 00060_cd
    col_flow = next(c for c in df.columns if c.endswith("00060"))
    df = df.rename(columns={"site_no": "usgs_id", col_flow: "flow_cfs", "tz_cd": "tz"})
    df = df[["usgs_id", "datetime", "tz", "flow_cfs"]]
    # Parse times with timezone (USGS returns local TZ)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    # Convert to UTC
    # Note: when tz is provided per-row, we cannot reliably localize; USGS returns ISO
    df["timestamp_utc"] = pd.to_datetime(df["datetime"], utc=True)
    # Numeric
    df["flow_cfs"] = pd.to_numeric(df["flow_cfs"], errors="coerce")
    df = df.dropna(subset=["timestamp_utc", "flow_cfs"]).copy()
    df = df.set_index("timestamp_utc").sort_index()
    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = df.index
    if getattr(idx, 'tz', None) is not None:
        df = df.copy()
        df.index = idx.tz_localize(None) if idx.tz is None else idx.tz_convert(None)
    df = df[~df.index.duplicated(keep='last')]
    hourly = df.resample("1h").mean(numeric_only=True)
    # Add site_name and usgs_id back in from the first row of the original df
    if not df.empty:
        hourly['site_name'] = df['site_name'].iloc[0]
        hourly['usgs_id'] = df['usgs_id'].iloc[0]
    hourly["quality"] = "avg"
    hourly = hourly.dropna(subset=["flow_cfs"]).copy()
    hourly.reset_index(inplace=True)
    rename_col = 'timestamp_utc' if 'timestamp_utc' in hourly.columns else hourly.columns[0]
    hourly.rename(columns={rename_col: "timestamp"}, inplace=True)
    return hourly


def fetch_site(usgs_id: str, site_info: dict, start_date: str, end_date: str, collector: USGSCollector) -> pd.DataFrame:
    """Fetches data for a single site and date range."""
    logger.info(f"Fetching USGS {usgs_id} {site_info['name']} {start_date}..{end_date}")
    
    df = fetch_range(usgs_id, start_date, end_date, collector)
    if not df.empty:
        df["usgs_id"] = usgs_id
        df["site_name"] = site_info["name"]
        try:
            df = df[~df.index.duplicated(keep='last')]
            df.index = df.index.tz_convert(None)
        except AttributeError:
            pass
        df = resample_hourly(df)
    return df


def save_partition(df: pd.DataFrame, out_dir: str, site_id: str, year: int):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{site_id}_{year}.csv")
    df.to_csv(path, index=False)
    logger.info(f"Saved {path} rows={len(df)}")


def consolidate(directory: str, prefix: str, output_filename: str):
    """Consolidates yearly or monthly files into a single CSV."""
    files = sorted([f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.csv')])
    frames = []
    for f in files:
        frames.append(pd.read_csv(os.path.join(directory, f)))
    if frames:
        all_df = pd.concat(frames, ignore_index=True).sort_values(["usgs_id", "timestamp"])  
        all_df.to_csv(os.path.join(directory, output_filename), index=False)
        logger.info(f"Saved consolidated {output_filename} rows={len(all_df)}")


def _month_ranges(start_date: dt_date, end_date: dt_date):
    cur = dt_date(start_date.year, start_date.month, 1)
    while cur <= end_date:
        # Compute end of month
        if cur.month == 12:
            next_month = dt_date(cur.year + 1, 1, 1)
        else:
            next_month = dt_date(cur.year, cur.month + 1, 1)
        month_end = min(end_date, next_month - timedelta(days=1))
        yield cur, month_end
        cur = next_month


def main():
    parser = argparse.ArgumentParser(description="Fetch USGS streamflow data.")
    parser.add_argument("--out-dir", default="data/raw/usgs", help="Output directory for CSV files.")
    parser.add_argument("--sites", nargs="+", help="Optional list of site IDs to fetch.")
    parser.add_argument("--start-date", help="Optional ISO start date YYYY-MM-DD to limit range.")
    parser.add_argument("--end-date", help="Optional ISO end date YYYY-MM-DD to limit range.")
    parser.add_argument("--skip-consolidate-site", action="store_true", help="Skip writing per-site consolidated CSVs.")
    parser.add_argument("--consolidate-global", action="store_true", help="Write all-sites consolidated CSV across sites (off by default).")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Determine which sites to fetch
    if args.sites:
        sites_to_fetch = {site_id: MASTER_STUDY_SITES[site_id] for site_id in args.sites if site_id in MASTER_STUDY_SITES}
    else:
        sites_to_fetch = MASTER_STUDY_SITES

    # Define date ranges for acquisition
    if args.start_date and args.end_date:
        ranges = [(args.start_date, args.end_date)]
        logger.info(f"Using custom range: {args.start_date}..{args.end_date}")
    else:
        ranges = [
            ("2020-01-01", "2020-12-31"),
            ("2021-01-01", "2021-12-31"),
            ("2022-01-01", "2022-12-31"),
            ("2023-01-01", "2023-12-31"),
            ("2025-01-01", datetime.utcnow().strftime('%Y-%m-%d')),  # Up to today
        ]
    
    collector = USGSCollector()

    # Correctly iterate over items (site_id, site_info)
    for usgs_id, site_info in sites_to_fetch.items():
        for start, end in ranges:
            year = int(start[:4])
            start_date = datetime.strptime(start, "%Y-%m-%d").date()
            end_date = datetime.strptime(end, "%Y-%m-%d").date()
            year_frames = []
            for m_start, m_end in _month_ranges(start_date, end_date):
                m_s = m_start.isoformat()
                m_e = m_end.isoformat()
                try:
                    # Pass the full site_info dictionary
                    df = fetch_site(usgs_id, site_info, m_s, m_e, collector)
                    if not df.empty:
                        # Load data into memory and validate structure
                        df = df.copy()  # Ensure data is in memory
                        logger.debug(f"   ✅ {usgs_id} {m_s}..{m_e}: {len(df)} records")
                    else:
                        logger.warning(f"   ⚠️  {usgs_id} {m_s}..{m_e}: No data returned")
                        continue
                except requests.RequestException as req_error:
                    logger.error(f"   ❌ Network error for {usgs_id} {m_s}..{m_e}: {req_error}")
                    continue
                except pd.errors.ParserError as parse_error:
                    logger.error(f"   ❌ Data parsing error for {usgs_id} {m_s}..{m_e}: {parse_error}")
                    continue
                except Exception as e:
                    logger.error(f"   ❌ Processing error for {usgs_id} {m_s}..{m_e}: {e}")
                    continue
                    
                year_frames.append(df)
                # Gentle throttle between month requests
                time.sleep(0.5)
            if year_frames:
                try:
                    year_df = pd.concat(year_frames, ignore_index=True)
                    # De-duplicate in case of overlaps and validate data integrity
                    year_df = year_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                    
                    # Validate data quality
                    valid_records = len(year_df.dropna(subset=['flow_cfs']))
                    total_records = len(year_df)
                    logger.info(f"   📊 {usgs_id} {year}: {valid_records}/{total_records} valid records")
                    
                    save_partition(year_df, out_dir, usgs_id, year)
                except pd.errors.ParserError as parse_error:
                    logger.error(f"   ❌ Failed to consolidate {usgs_id} {year}: {parse_error}")
                except Exception as consolidation_error:
                    logger.error(f"   ❌ Consolidation error for {usgs_id} {year}: {consolidation_error}")
            else:
                logger.warning(f"   ⚠️  No data collected for {usgs_id} {year}")

    # Consolidated per-site files (optional)
    if not args.skip_consolidate_site:
        for usgs_id in sites_to_fetch.keys():
            consolidate(out_dir, f"{usgs_id}_", f"{usgs_id}_consolidated.csv")
    else:
        logger.info("Skipping per-site consolidated files as requested")

    # Global consolidated (off by default)
    if args.consolidate_global:
        all_files = [f for f in os.listdir(out_dir) if f.endswith('_consolidated.csv')]
        frames = []
        for f in all_files:
            try:
                frames.append(pd.read_csv(os.path.join(out_dir, f)))
            except pd.errors.EmptyDataError:
                logger.warning(f"Skipping empty file: {f}")
                continue
        if frames:
            all_df = pd.concat(frames, ignore_index=True).sort_values(["usgs_id","timestamp"]).reset_index(drop=True)
            all_df.to_csv(os.path.join(out_dir, 'usgs_streamflow_all_sites_2020_2023_2025.csv'), index=False)
            logger.info(f"Saved all-sites consolidated CSV rows={len(all_df)}")
        else:
            logger.info("No per-site consolidated files found; skipping global consolidation")
    else:
        logger.info("Global all-sites consolidation disabled (default)")

    logger.info("USGS streamflow acquisition complete")


if __name__ == "__main__":
    main()

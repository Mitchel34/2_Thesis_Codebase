#!/usr/bin/env python3
"""
ERA5 Chunked Fetcher (hourly or 6-hourly, per-site, per-month)
==============================================================
Downloads ERA5 single-level variables for each study site using small per-month
requests to avoid CDS cost/queue limits. Outputs CSVs with derived features and
timestamps at either hourly cadence (preferred) or 6-hourly (00/06/12/18) to match NWM.

Defaults:
- Years: 2020–2023 (training) and 2025 (testing)
- Cadence: hourly (00..23) by default; switch to 6h with --cadence 6h
- Variables: t2m, d2m, tp, sp, u10, v10, ssrd, e, swvl1

Requirements:
- cdsapi, xarray, pandas, numpy
- ~/.cdsapirc configured
"""
import os
import sys
import argparse
import zipfile
import tarfile
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import cdsapi
    import xarray as xr
except ImportError:
    print("Missing dependencies (cdsapi, xarray). Install them via pip and retry.")
    sys.exit(1)

# Add project root to Python path (one level up from data_acquisition_scripts)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config.master_study_sites import MASTER_STUDY_SITES

# Prepare site list from master configuration
STUDY_SITES = []
for usgs_id, site_info in MASTER_STUDY_SITES.items():
    site = site_info.copy()
    site['usgs_id'] = usgs_id
    STUDY_SITES.append(site)

OUTPUT_DIR = 'data/clean/raw/era5'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SINGLE_DATASET = 'reanalysis-era5-single-levels'
SINGLE_VARIABLES = [
    '2m_temperature',
    '2m_dewpoint_temperature',
    'total_precipitation',
    'surface_pressure',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'surface_solar_radiation_downwards',
    'evaporation',
]
LAND_DATASET = 'reanalysis-era5-land'
LAND_VARIABLES = [
    'volumetric_soil_water_layer_1',
]
HOURS_6H = ['00:00','06:00','12:00','18:00']
HOURS_HOURLY = [f"{h:02d}:00" for h in range(24)]
MONTHS = [f"{m:02d}" for m in range(1,13)]
DAYS = [f"{d:02d}" for d in range(1,32)]

c = cdsapi.Client()


def bbox_around(lat: float, lon: float, delta: float = 0.35):
    """
    Create an ERA5 area bbox around a point and snap to the 0.25° grid so that
    at least one grid cell is guaranteed to be included. Using a slightly
    larger default delta avoids empty selections that can cause FileNotFound.
    """
    # Expand and snap to 0.25° grid
    north = min(90.0, round((lat + delta) * 4) / 4)
    south = max(-90.0, round((lat - delta) * 4) / 4)
    east = min(180.0, round((lon + delta) * 4) / 4)
    west = max(-180.0, round((lon - delta) * 4) / 4)
    # Ensure proper ordering
    if south > north:
        south, north = north, south
    if west > east:
        west, east = east, west
    # ERA5 area ordering: north, west, south, east
    return [north, west, south, east]


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    # Helper for safe column access
    def s(name):
        return df[name]

    # Units and simple derived metrics (add columns only when inputs exist)
    if '2m_temperature' in df.columns:
        df['temp_c'] = s('2m_temperature') - 273.15
    if '2m_dewpoint_temperature' in df.columns:
        df['dewpoint_c'] = s('2m_dewpoint_temperature') - 273.15
    if 'surface_pressure' in df.columns:
        df['pressure_hpa'] = s('surface_pressure') / 100.0
    if 'total_precipitation' in df.columns:
        df['precip_mm'] = s('total_precipitation') * 1000.0
    if 'surface_solar_radiation_downwards' in df.columns:
        df['radiation_mj_m2'] = s('surface_solar_radiation_downwards') / 1e6
    # Handle either 'evaporation' or 'total_evaporation'
    if 'evaporation' in df.columns:
        df['evap_mm'] = df['evaporation'] * 1000.0
    elif 'total_evaporation' in df.columns:
        df['evap_mm'] = df['total_evaporation'] * 1000.0
    # Wind
    if '10m_u_component_of_wind' in df.columns and '10m_v_component_of_wind' in df.columns:
        df['wind_speed'] = np.sqrt(s('10m_u_component_of_wind')**2 + s('10m_v_component_of_wind')**2)
        wind_dir = np.degrees(np.arctan2(s('10m_v_component_of_wind'), s('10m_u_component_of_wind')))
        df['wind_dir_deg'] = (wind_dir + 360) % 360
    # VPD/Relative humidity via Magnus (only if temp and dewpoint exist)
    if 'temp_c' in df.columns and 'dewpoint_c' in df.columns:
        es_t = 0.6108 * np.exp(17.27 * s('temp_c') / (s('temp_c') + 237.3))
        es_d = 0.6108 * np.exp(17.27 * s('dewpoint_c') / (s('dewpoint_c') + 237.3))
        df['vpd_kpa'] = es_t - es_d
        with np.errstate(invalid='ignore', divide='ignore'):
            df['rel_humidity_pct'] = np.clip((es_d / es_t) * 100.0, 0, 100)
    # Soil moisture helpers
    if 'volumetric_soil_water_layer_1' in df.columns:
        df['soil_moisture_vwc'] = df['volumetric_soil_water_layer_1']  # m3/m3
        df['soil_moisture_pct'] = df['volumetric_soil_water_layer_1'] * 100.0
    # Time features
    ts = pd.to_datetime(df['timestamp'])
    df['hour'] = ts.dt.hour
    df['doy'] = ts.dt.dayofyear
    df['month'] = ts.dt.month
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    df['doy_sin'] = np.sin(2*np.pi*df['doy']/365.25)
    df['doy_cos'] = np.cos(2*np.pi*df['doy']/365.25)
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    return df


def download_month_single(site: dict, year: int, month: str, output_dir: str, hours: list[str], verbose: bool = True) -> str | None:
    lat = site['lat']
    lon = site['lon']
    area = bbox_around(lat, lon)
    
    # Create site-specific directory
    site_dir = os.path.join(output_dir, str(site['nwm_comid']))
    os.makedirs(site_dir, exist_ok=True)
    
    nc_path = os.path.join(site_dir, f"era5_single_{site['nwm_comid']}_{year}_{month}.nc")
    if os.path.exists(nc_path):
        if verbose:
            print(f"✓ ERA5 single-levels already exists: {nc_path}")
        return nc_path
    try:
        if verbose:
            print(f"→ Requesting ERA5 single-levels for {site['nwm_comid']} {year}-{month} at bbox {area}…")
        c.retrieve(
            SINGLE_DATASET,
            {
                'product_type': 'reanalysis',
                'variable': SINGLE_VARIABLES,
                'year': str(year),
                'month': month,
                'day': DAYS,
                'time': hours,
                'area': area,
                'format': 'netcdf',
                'grid': '0.25/0.25',
            },
            nc_path,
        )
        if verbose:
            print(f"✓ Downloaded single-levels: {nc_path}")
        return nc_path
    except Exception as e:
        print(f"✗ ERA5 single-levels failed for {site['nwm_comid']} {year}-{month}: {e}")
        return None


def download_month_land(site: dict, year: int, month: str, output_dir: str, hours: list[str], verbose: bool = True) -> str | None:
    lat = site['lat']
    lon = site['lon']
    area = bbox_around(lat, lon)
    
    # Create site-specific directory
    site_dir = os.path.join(output_dir, str(site['nwm_comid']))
    os.makedirs(site_dir, exist_ok=True)
    
    nc_path = os.path.join(site_dir, f"era5_land_{site['nwm_comid']}_{year}_{month}.nc")
    if os.path.exists(nc_path):
        if verbose:
            print(f"✓ ERA5-Land already exists: {nc_path}")
        return nc_path
    try:
        if verbose:
            print(f"→ Requesting ERA5-Land for {site['nwm_comid']} {year}-{month} at bbox {area}…")
        c.retrieve(
            LAND_DATASET,
            {
                'product_type': 'reanalysis',
                'variable': LAND_VARIABLES,
                'year': str(year),
                'month': month,
                'day': DAYS,
                'time': hours,
                'area': area,
                'format': 'netcdf',
            },
            nc_path,
        )
        if verbose:
            print(f"✓ Downloaded ERA5-Land: {nc_path}")
        return nc_path
    except Exception as e:
        print(f"✗ ERA5-Land soil moisture failed for {site['nwm_comid']} {year}-{month}: {e}")
        return None


def reduce_nc_to_df(nc_path: str, filter_hours: list[int] | None = None, verbose: bool = True) -> pd.DataFrame:
    try:
        # Resolve if the downloaded file is an archive containing .nc files
        resolved_path = nc_path
        if zipfile.is_zipfile(nc_path):
            with zipfile.ZipFile(nc_path, 'r') as zf, tempfile.TemporaryDirectory() as td:
                zf.extractall(td)
                nc_files = sorted([str(p) for p in Path(td).rglob('*.nc')])
                if not nc_files:
                    raise FileNotFoundError('No .nc file inside ZIP archive')
                frames = []
                for fp in nc_files:
                    if verbose:
                        print(f"↳ Open ZIP member: {fp}")
                    try:
                        ds = xr.open_dataset(fp, engine='netcdf4')
                    except Exception:
                        ds = xr.open_dataset(fp, engine='scipy')
                    reduce_dims = []
                    if 'latitude' in ds.dims:
                        reduce_dims.append('latitude')
                    if 'longitude' in ds.dims:
                        reduce_dims.append('longitude')
                    if reduce_dims:
                        ds = ds.mean(dim=reduce_dims, skipna=True)
                    if verbose:
                        print(f"↳ Reduced dims: {reduce_dims if reduce_dims else 'none'}")
                    dfp = ds.to_dataframe().reset_index()
                    # time column detection
                    time_col = None
                    for cand in ['time', 'valid_time', 'forecast_time', 'time0', 'time1']:
                        if cand in dfp.columns:
                            time_col = cand
                            break
                    if time_col is None:
                        for col in dfp.columns:
                            if 'time' in col.lower():
                                time_col = col
                                break
                    if time_col is None:
                        raise ValueError('time coordinate missing in ERA5 dataset')
                    dfp['timestamp'] = pd.to_datetime(dfp[time_col], errors='coerce')
                    dfp = dfp[dfp['timestamp'].notna()]
                    dfp['hour'] = pd.to_datetime(dfp['timestamp']).dt.hour
                    if filter_hours is not None:
                        dfp = dfp[dfp['hour'].isin(filter_hours)].copy()
                    drop_cols = [c for c in dfp.columns if c in {time_col, 'latitude', 'longitude'}]
                    dfp = dfp.drop(columns=drop_cols, errors='ignore')
                    frames.append(dfp)
                    ds.close()
                # Outer-merge all frames on timestamp/hour
                df = frames[0]
                for dfp in frames[1:]:
                    df = pd.merge(df, dfp, on=['timestamp','hour'], how='outer')
                return df
        elif tarfile.is_tarfile(nc_path):
            with tarfile.open(nc_path, 'r:*') as tf, tempfile.TemporaryDirectory() as td:
                tf.extractall(td)
                nc_files = sorted([str(p) for p in Path(td).rglob('*.nc')])
                if not nc_files:
                    raise FileNotFoundError('No .nc file inside TAR archive')
                frames = []
                for fp in nc_files:
                    if verbose:
                        print(f"↳ Open TAR member: {fp}")
                    try:
                        ds = xr.open_dataset(fp, engine='netcdf4')
                    except Exception:
                        ds = xr.open_dataset(fp, engine='scipy')
                    reduce_dims = []
                    if 'latitude' in ds.dims:
                        reduce_dims.append('latitude')
                    if 'longitude' in ds.dims:
                        reduce_dims.append('longitude')
                    if reduce_dims:
                        ds = ds.mean(dim=reduce_dims, skipna=True)
                    if verbose:
                        print(f"↳ Reduced dims: {reduce_dims if reduce_dims else 'none'}")
                    dfp = ds.to_dataframe().reset_index()
                    time_col = None
                    for cand in ['time', 'valid_time', 'forecast_time', 'time0', 'time1']:
                        if cand in dfp.columns:
                            time_col = cand
                            break
                    if time_col is None:
                        for col in dfp.columns:
                            if 'time' in col.lower():
                                time_col = col
                                break
                    if time_col is None:
                        raise ValueError('time coordinate missing in ERA5 dataset')
                    dfp['timestamp'] = pd.to_datetime(dfp[time_col], errors='coerce')
                    dfp = dfp[dfp['timestamp'].notna()]
                    dfp['hour'] = pd.to_datetime(dfp['timestamp']).dt.hour
                    if filter_hours is not None:
                        dfp = dfp[dfp['hour'].isin(filter_hours)].copy()
                    drop_cols = [c for c in dfp.columns if c in {time_col, 'latitude', 'longitude'}]
                    dfp = dfp.drop(columns=drop_cols, errors='ignore')
                    frames.append(dfp)
                    ds.close()
                df = frames[0]
                for dfp in frames[1:]:
                    df = pd.merge(df, dfp, on=['timestamp','hour'], how='outer')
                return df

        # Try to open the resolved path as NetCDF
        try:
            ds = xr.open_dataset(resolved_path, engine='netcdf4')
        except Exception:
            ds = xr.open_dataset(resolved_path, engine='scipy')

        # Average within tiny bbox to a single time series
        reduce_dims = []
        if 'latitude' in ds.dims:
            reduce_dims.append('latitude')
        if 'longitude' in ds.dims:
            reduce_dims.append('longitude')
        if reduce_dims:
            ds = ds.mean(dim=reduce_dims, skipna=True)
        if verbose:
            print(f"↳ Reduced dims: {reduce_dims if reduce_dims else 'none'} from {resolved_path}")

        # Move to DataFrame and resolve time column name
        df = ds.to_dataframe().reset_index()
        time_col = None
        for cand in ['time', 'valid_time', 'forecast_time', 'time0', 'time1']:
            if cand in df.columns:
                time_col = cand
                break
        if time_col is None:
            # Fallback: any column containing 'time'
            for col in df.columns:
                if 'time' in col.lower():
                    time_col = col
                    break
        if time_col is None:
            raise ValueError('time coordinate missing in ERA5 dataset')

        df['timestamp'] = pd.to_datetime(df[time_col], errors='coerce')
        df = df[df['timestamp'].notna()]
        # Optional filter to 6-hourly hours
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        if filter_hours is not None:
            df = df[df['hour'].isin(filter_hours)].copy()
        # Drop non-needed columns
        drop_cols = [c for c in df.columns if c in {time_col, 'latitude', 'longitude'}]
        df = df.drop(columns=drop_cols, errors='ignore')
        ds.close()
        return df
    except Exception as e:
        print(f"✗ ERA5 processing failed for {nc_path}: {e}")
        return None


def process_to_csv(single_nc: str | None, land_nc: str | None, site: dict, year: int, month: str, output_dir: str, filter_hours: list[int] | None = None, keep_nc: bool = False, verbose: bool = True) -> str | None:
    # Load and rename variables
    single_df = reduce_nc_to_df(single_nc, filter_hours=filter_hours, verbose=verbose) if single_nc else None
    land_df = reduce_nc_to_df(land_nc, filter_hours=filter_hours, verbose=verbose) if land_nc else None
    if single_df is None and land_df is None:
        return None
    rename = {
        't2m': '2m_temperature',
        'd2m': '2m_dewpoint_temperature',
        'tp': 'total_precipitation',
        'sp': 'surface_pressure',
        'u10': '10m_u_component_of_wind',
        'v10': '10m_v_component_of_wind',
        'ssrd': 'surface_solar_radiation_downwards',
        'e': 'evaporation',
        'swvl1': 'soil_moisture_vwc',
    }
    frames = []
    for df in [single_df, land_df]:
        if df is not None:
            frames.append(df.rename(columns=rename))
    merged = frames[0]
    for df in frames[1:]:
        merged = pd.merge(merged, df, on=['timestamp','hour'], how='outer')
    merged = merged.sort_values('timestamp')
    merged = merged.drop_duplicates(subset=['timestamp'], keep='last')
    if verbose:
        print(f"↳ Rows after merge: {len(merged)}  | Columns: {len(merged.columns)}")
    # Normalize soil moisture column name if provider used verbose name
    if 'volumetric_soil_water_layer_1' in merged.columns and 'soil_moisture_vwc' not in merged.columns:
        merged = merged.rename(columns={'volumetric_soil_water_layer_1': 'soil_moisture_vwc'})
    # Derived features from combined columns
    merged = derive_features(merged)
    # Metadata
    merged['site_name'] = site['name']
    merged['comid'] = site['nwm_comid']
    merged['lat'] = site['lat']
    merged['lon'] = site['lon']
    
    # Create site-specific directory for CSV output
    site_dir = os.path.join(output_dir, str(site['nwm_comid']))
    os.makedirs(site_dir, exist_ok=True)
    
    # Drop auxiliary ERA5 columns that are not needed downstream (with possible suffixes)
    drop_aux = [c for c in merged.columns if c.startswith('number') or c.startswith('expver')]
    merged = merged.drop(columns=drop_aux, errors='ignore')
    # Drop any lingering verbose soil column if duplicate exists
    if 'soil_moisture_vwc' in merged.columns and 'volumetric_soil_water_layer_1' in merged.columns:
        merged = merged.drop(columns=['volumetric_soil_water_layer_1'], errors='ignore')
    # Drop columns that are completely empty to keep outputs clean
    merged = merged.dropna(axis=1, how='all')
    csv_path = os.path.join(site_dir, f"era5_enh_{site['nwm_comid']}_{year}_{month}.csv")
    merged.to_csv(csv_path, index=False)
    if verbose:
        print(f"✓ Wrote CSV: {csv_path}")
    # Cleanup
    if not keep_nc:
        for p in [single_nc, land_nc]:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
    return csv_path


def run(output_dir: str, years_train=(2020,2021,2022,2023), years_test=(2025,), site_filter: str | None = None, months: list[str] | None = None, cadence: str = 'hourly', keep_nc: bool = False, verbose: bool = True):
    successes, failures = [], []
    years = [*years_train, *years_test]
    # Determine request and filter hours based on cadence
    if cadence == 'hourly':
        request_hours = HOURS_HOURLY
        filter_hours = None  # keep all hours
    else:
        request_hours = HOURS_6H
        filter_hours = [0, 6, 12, 18]
    for site in STUDY_SITES:
        if site_filter and site_filter.lower() not in (str(site.get('name','')) + str(site.get('nwm_comid','')) + str(site.get('usgs_id',''))).lower():
            continue
        if not site.get('nwm_comid'):
            print(f"Skipping site {site.get('name')} due to missing nwm_comid.")
            continue
        if verbose:
            print(f"\n=== ERA5 site {site['name']} COMID {site['nwm_comid']} ===")
        for year in years:
            use_months = months if months else MONTHS
            for month in use_months:
                single_nc = download_month_single(site, year, month, output_dir, hours=request_hours, verbose=verbose)
                land_nc = download_month_land(site, year, month, output_dir, hours=request_hours, verbose=verbose)
                if not single_nc and not land_nc:
                    failures.append((site['nwm_comid'], year, month, 'download'))
                    continue
                csv_path = process_to_csv(single_nc, land_nc, site, year, month, output_dir, filter_hours=filter_hours, keep_nc=keep_nc, verbose=verbose)
                if csv_path:
                    successes.append(csv_path)
                else:
                    failures.append((site['nwm_comid'], year, month, 'process'))
    print("\nERA5 chunked fetch complete")
    print(f"Success files: {len(successes)} | Failures: {len(failures)}")
    if failures:
        print("Examples:", failures[:5])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ERA5 chunked downloader (hourly or 6-hourly)')
    parser.add_argument('--out-dir', default='data/raw/era5', help='Output directory for CSV files.')
    parser.add_argument('--site', help='Filter by site name or COMID substring', default=None)
    parser.add_argument('--years-train', nargs='*', type=int, default=[2020,2021,2022,2023], help='Training years')
    parser.add_argument('--years-test', nargs='*', type=int, default=[2025], help='Testing years')
    parser.add_argument('--months', nargs='*', default=None, help='Subset months as 01..12')
    parser.add_argument('--cadence', choices=['hourly','6h'], default='hourly', help='Temporal cadence for ERA5 requests and outputs')
    parser.add_argument('--keep-nc', action='store_true', help='Keep downloaded .nc files for inspection')
    parser.add_argument('--quiet', action='store_true', help='Reduce logging verbosity')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("ERA5 chunked downloader starting…")
    run(
        args.out_dir,
        tuple(args.years_train),
        tuple(args.years_test),
        site_filter=args.site,
        months=args.months,
        cadence=args.cadence,
        keep_nc=args.keep_nc,
        verbose=not args.quiet,
    )

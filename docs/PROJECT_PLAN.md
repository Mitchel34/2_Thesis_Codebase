# Project Plan (Updated): Hourly NWM Post-processing with USGS Ground Truth

Date: 2025-09-22
Owner: Mitchel Carson

## Objectives
- Build a robust, site-centric pipeline to collect ERA5, USGS, NWM retrospective (v2.1/v3.0), and NLCD features at hourly cadence.
- Train models to predict residuals (USGS − NWM) and produce corrected streamflow at hourly valid times.
- Prioritize hourly-only analysis data (CHRTOUT) across the full period; defer lead-aware forecast analysis to optional, recent windows.
- Evaluate with NSE, KGE, RMSE, Correlation Coefficient, and Percent Bias. Produce publication-quality plots.
- Scale to multiple study sites and prepare a lightweight web app for exploration/download.

## Current Status (✅)
- NWM
  - v3.0 retrospective CHRTOUT (2021–2023) pull implemented and validated for sample windows (hourly only, no leads). Files saved under `data/raw/nwm_v3/retrospective/`.
  - v2.1 retrospective hourly (1979–2020) reader via S3 Zarr already implemented (optional for longer history).
  - Operational short_range (with leads) implemented but used only for recent windows; not required for baseline.
  - NCEI THREDDS archive for 2021–2023 short_range could not be confirmed (probes returned 404/0-coverage). Decision: do not depend on archived short_range.
- ERA5
  - Hourly/6h downloader functional with derived features; per-site monthly CSV outputs.
- USGS
  - Hourly resampling and unit conversion to cms working; timestamps normalized to UTC.
- Modeling dataset builder
  - Updated to hourly-only: loads NWM CHRTOUT, aligns with USGS and ERA5, computes residuals, outputs Parquet and CSV sample.
  - Verified on 2023-01-01 00–05Z window, wrote Parquet and sample CSV in `data/clean/modeling/`.
- Documentation
  - This plan updated to reflect hourly-only decision and concrete next steps.

## Key Decisions
- Use hourly NWM analysis (CHRTOUT) for 2021–2023 and earlier (v2.1) to ensure full coverage without lead dependencies.
- Remove lead_time dependency from core pipeline and dataset. Keep optional lead-aware path for future extensions with operational data.
- Store aligned datasets in Parquet; sample CSVs for quick inspection.

## Data Sources & Layout
- USGS NWIS instantaneous discharge → hourly CMS
  - Input: `data/raw/usgs/<usgs_id>_<year>.csv`
- ERA5 Single-level + ERA5-Land → hourly/6h features
  - Input: `data/raw/era5/<site>/<site>_<yyyy-mm>.csv` (merged per site-month)
- NWM retrospective
  - v3.0 CHRTOUT (2021–2023, hourly): `s3://noaa-nwm-retrospective-3-0-pds/CONUS/netcdf/CHRTOUT/YYYY/YYYYMMDDHHMM.CHRTOUT_DOMAIN1`
  - v2.1 CHRTOUT (1979–2020, hourly via Zarr): `s3://noaa-nwm-retrospective-2-1-zarr-pds/chrtout.zarr`
  - Local outputs: `data/raw/nwm_v3/retrospective/nwm_v3_hourly_YYYYMMDD_YYYYMMDD.csv`
- NLCD static features: `data/raw/land_use/nlcd_2021_land_use_metrics.csv`

## Current Scripts
- `data_acquisition_scripts/usgs.py` – USGS hourly, UTC normalization, cms conversion.
- `data_acquisition_scripts/era5.py` – ERA5 downloader with hourly/6h cadence and derived features.
- `data_acquisition_scripts/nwm.py` – NWM collectors:
  - `retrospective_v3` (v3.0 CHRTOUT hourly, 2021–2023)
  - `retrospective` (v2.1 CHRTOUT hourly via Zarr, 1979–2020)
  - `operational` (short_range with leads, optional/recent)
  - `archive` (short_range via HTTP; not used—no working endpoint confirmed)
- `modeling/build_training_dataset.py` – hourly-only alignment; computes residuals; outputs Parquet + sample CSV.
- `data_acquisition_scripts/tools/check_nwm_archive_leads.py` – probe utility confirming lack of NCEI archived short_range coverage for our test windows.

## Minimal Contract (hourly-only)
Inputs
- Study sites: `config/master_study_sites.py` (requires `usgs_id` and `nwm_comid`)
- Date window: start, end (UTC)

Outputs
- Aligned hourly dataset with columns (core):
  - `timestamp`, `site_name`, `comid`, `nwm_cms`, `usgs_cms`, [ERA5 features...], [NLCD features...]
  - Targets: `y_residual_cms`, `y_corrected_cms`
- Files: `data/clean/modeling/hourly_training_{start}_{end}.parquet` (+ sample CSV)

Error modes
- Missing overlaps between sources; network access issues; partial hours

Success criteria
- End-to-end build completes on day/week windows with high coverage and sensible residuals.

## Edge Cases
- Sparse ERA5 (6h) → safe hourly alignment via nearest within 3h (no future leakage)
- NWM v3.0 occasional gaps → leave NaN; no silent fills
- Units and timezones → cms conversion and UTC normalization already implemented

## Implementation Steps (near-term)
1) Data slices
- Pull CHRTOUT hours for a 7-day window in 2023 and 2022 (v3.0 and v2.1 as available).
- Ensure corresponding USGS and ERA5 windows exist.

2) Build aligned datasets
- Use `modeling/build_training_dataset.py` to create Parquet outputs for each window.
- Quick QA: row counts, NaN rates, basic stats.

3) Baseline metrics and plots
- Implement `modeling/metrics.py` to compute NSE, KGE, RMSE, CC, PBias on NWM vs USGS (pre-correction baseline).
- Implement `modeling/plots.py` to produce:
  - Hydrographs (USGS vs NWM),
  - Residual histograms/density,
  - Scatter USGS vs NWM with 1:1 line.

4) First model (hourly residual)
- Start with a compact transformer or temporal CNN that ingests: [NWM, ERA5 features, optional static NLCD].
- Predict next-step residual; corrected = NWM + residual.
- Train/val/test split by time; early stopping; simple HPO sweep.

5) Reporting
- Save metrics tables and plots per site/date window under `data/clean/modeling/` or `reports/`.
- Document findings and update this plan with learned insights.

## Optional (later)
- Lead-aware path using operational short_range for recent periods (evaluate per lead)
- Multi-site model with site embeddings
- Lightweight web app (FastAPI + simple FE) for browsing sites/windows and downloading corrected series

## How to run (quick)
- NWM v3.0 CHRTOUT sample (already validated):
  - `python data_acquisition_scripts/nwm.py --mode retrospective_v3 --start-date 2023-01-01T00:00 --end-date 2023-01-01T05:00 --out-dir data/raw/nwm_v3`
- Build dataset for that window:
  - `python modeling/build_training_dataset.py --start 2023-01-01T00:00 --end 2023-01-01T05:00`
- Next: expand to a 7-day window and run metrics/plots.

- Optimizing NWM V.30 API Calls (Bottleneck in data retrival)
process-based concurrency
validate
build aligned dataset
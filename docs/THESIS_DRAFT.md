# Thesis Draft Notes

## Data Acquisition Pipeline
### Overview
- Objective: assemble hourly-aligned inputs that enable correcting National Water Model (NWM) streamflow forecasts with site-specific observations.
- Core sources: USGS observed discharge (ground truth), NWM CHRTOUT retrospective/operational analysis (model baseline), ERA5 reanalysis (meteorological context), NLCD 2021 static land-cover metrics (physiographic context).
- Design principle: favor authoritative, well-maintained archives that cover 2010–2022 without gaps while exposing recent 2023+ windows for future extension.

### NWM Streamflow Collection (`data_acquisition_scripts/nwm.py`)
- Supports retrospective v3.0 (2021–2023) and operational analysis assimilation (Feb 2023 onward) buckets with automatic stitching across the archive boundary.
- Uses process-based concurrency and boto3 downloads to pull per-hour CHRTOUT NetCDFs, extracting only the study sites' `streamflow` by COMID to limit I/O.
- Handles `.comp` full_physics compression (zstd) when the archive switched formats in 2023, ensuring continuity in hourly coverage.
- Cleans timestamps to naive UTC, deduplicates by `timestamp`/`comid`, and persists hourly CSVs under `data/raw/nwm_v3/retrospective/` for downstream alignment.
- Rationale: NWM provides physically-based baseline forecasts; correcting its residuals targets practical improvements for operational hydrology.

### USGS Ground-Truth Collection (`data_acquisition_scripts/usgs.py`)
- Implements resilient download logic against NWIS CSV endpoints with throttling, exponential backoff, and a circuit breaker to survive sustained 503 outages.
- Normalizes timestamps to UTC hourly cadence and converts flow units (cfs→cms when necessary) to match NWM conventions.
- Provides per-site CSVs in `data/raw/usgs/` that act as the ground-truth corrections target.
- Rationale: USGS gauges represent the trusted observed discharge required to quantify NWM residuals and corrected flows.

### ERA5 Atmospheric Features (`data_acquisition_scripts/era5.py`)
- Uses the CDS API to pull monthly, site-specific subsets of ERA5 single-level and ERA5-Land variables at hourly cadence (optional 6-hour fallback).
- Derives physically interpretable features (temperature, dewpoint, VPD, wind speed, pressure, precipitation, radiation, soil moisture) and cyclical time encodings.
- Stores outputs per site under `data/raw/era5/<comid>/`, enabling consistent merges with the hydrologic data.
- Rationale: meteorological forcings help explain departures between NWM simulations and observed flows, capturing weather-driven dynamics absent from pure baselines.

### Dataset Assembly (`modeling/build_training_dataset.py`)
- Aligns NWM, USGS, ERA5, and NLCD static metrics into an hourly panel; computes residual (`y_residual_cms`) and corrected (`y_corrected_cms`) targets.
- Enforces safe temporal joins: ERA5 6-hour data are matched within ±3 hours without forward leakage; rows require both NWM and USGS coverage.
- Outputs Parquet shards (e.g., 2010–2022 train/val/test splits) plus small CSV samples for inspection under `data/clean/modeling/`.
- Rationale: creating a single, normalized dataset reduces feature-engineering drift and guarantees reproducible experiments across transformer and recurrent baselines.

## Model Architectures
### Hybrid Transformer v2 (`modeling/train_quick_transformer_torch.py` + `modeling/models/hydra_temporal.py`)
- Causal TCN stem with dilated residual blocks feeds a transformer encoder, with channel attention pooling the final sequence state.
- Static features FiLM-condition the temporal channels while gain+bias heads provide physics-guided corrections on top of the residual path.
- Residual and corrected outputs include heteroscedastic log-variances and are trained with Gaussian NLL on asinh-transformed targets plus raw-scale consistency losses.
- Ensemble of direct, residual, and gain+bias corrected streams reduces systematic error and stabilizes high-flow extremes.
- Motivation: increase capacity and bias control without reworking the data pipeline or expanding to external pretrained backbones.

### Residual LSTM Baseline (`modeling/train_quick_lstm_torch.py`)
- Stacked LSTM encoder (configurable depth, bidirectionality) processes normalized dynamic sequences; dropout mitigates overfitting on smaller windows.
- Optional static-feature projection mirrors the transformer fusion so both models ingest identical information.
- Dual linear heads output residual and corrected predictions with the same focal/consistency/bias-penalty loss composition for parity with the transformer.
- Shares data preparation, augmentation, and metric computation paths with the transformer script for reproducible comparisons.
- Motivation: provides a strong recurrent baseline rooted in hydrologic literature, quantifying the incremental value of transformer mechanisms.

## Preliminary Results and Discussion
### Experiment Setup
- Dataset: `data/clean/modeling/hourly_training_2023-01-01_2023-01-31.parquet` (hourly sequences for the study site cohort).
- Splits: 21 days for training, 7 days for validation, remaining hours for evaluation; both models trained for up to 20 epochs with early stopping (patience 4).
- Optimization: AdamW (no Ranger available), focal + consistency loss stack, gradient clipping at 1.0, data augmentation enabled for training batches.
### Metric Snapshot (Evaluation on 2023-01-28 onward)
- Raw NWM baseline: RMSE 0.708 cms, MAE 0.588 cms, NSE -7.73, KGE -0.68, PBIAS +8.17%, r = 0.51.
- Hybrid transformer v2: RMSE 0.440 cms (37.9% improvement vs. NWM), MAE 0.332 cms, NSE -2.37, KGE -0.01, PBIAS -3.50%, r = 0.48. Residual RMSE 0.57 cms.
- Residual LSTM: RMSE 1.351 cms (90.8% worse than NWM), MAE 1.31 cms, NSE -30.81, KGE 0.23, PBIAS +23.57%, r = 0.41. Residual RMSE 0.95 cms.

### Full-run Sweep #1 (2010–2022 Train / 2022 Evaluation)
- Configuration: Hydra v2 with gain_scale 0.05, log-variance clamp [-4, 2], quantile weight 0.1, quantiles {0.1, 0.5, 0.9}, and four MoE experts (`hydra_v2_full_q_sweep`).
- Baseline NWM on the 2022 hold-out window posts RMSE 1.03 cms, MAE 0.92 cms, NSE -175.8, KGE -4.05, and PBIAS +30.7%.
- Corrected outputs cut RMSE to 0.36 cms (64.8% reduction), MAE to 0.34 cms, and tighten residual RMSE to 0.45 cms while nudging PBIAS to -7.5%.
- Calibration diagnostics (n=12 evaluation hours) show q10 under-coverage (≈0% of observations), q90 over-coverage (≈100%), and the 80% central band covering every case; PIT mean skews high (0.58) with KS ≈0.41, signalling heavy upper-tail dispersion.
- New plots and PIT traces exported to `results/plots/full_run_sweep1` via `modeling/generate_model_improvement_plots.py` and `modeling/evaluate_full_run.py`.
### Interpretation
- Upgraded Hydra v2 cuts RMSE by ~38% versus raw NWM and >25% versus the earlier transformer, with gain/bias fusion shaving systematic error while heteroscedastic NLL dampens noisy high-flow spikes.
- Residual predictions tighten substantially (RMSE 0.57 cms vs. 3.36 cms previously), feeding more stable corrected flows; the LSTM baseline still overfits to mean bias and needs curriculum/regularisation sweeps.
- Quantile coverage from the 2022 evaluation suggests boosting lower-tail weight (raise quantile_weight or relax logvar_min) while moderating gain_scale/logvar_max to rein in the inflated upper band.
- Next steps: scale Hydra v2 to 2010–2022, iterate on gain/variance clamps and quantile weighting, and continue exploring MoE heads for calibrated uncertainty.

## Narrative Development
### Research Focus
- **Problem framing:** National Water Model (NWM) residuals remain site-specific and regime dependent; correcting them with machine learning requires respecting local hydrology, regulation, and meteorological drivers.
- **Guiding questions:**
  1. How do hybrid transformer residual corrections compare to LSTM baselines across distinct hydro-climatic regimes (humid Appalachian, semi-arid Southwest, agricultural Midwest, boreal Northeast, Gulf coastal)?
  2. Does regulation status (regulated vs. unregulated) influence attainable error reductions or calibration behaviour?
  3. Can a consistent pipeline—data alignment, training, calibration—generalize across stations without bespoke feature engineering?

### Site Cohorts
- **Core validation site:** `03479000` Watauga River (Appalachian mixed forest, unregulated) anchors methodology and ablation analysis.
- **Expansion set:**
  - `08082500` Clear Fork Brazos River, TX (semi-arid grassland, unregulated) — tests performance under flashy flow and hydrologic scarcity.
  - `09504000` Oak Creek, AZ (desert shrubland, regulated) — characterizes regulated desert watershed.
  - `04137500` Au Sable River, MI (Great Lakes mixed forest, unregulated) — cold-season snowmelt and groundwater influence.
  - `05464500` Cedar River, IA (agricultural, regulated) — nutrient-laden, managed Midwest river.
  - `01034500` Penobscot River, ME (boreal transition, regulated) — northern snowmelt with dam operations.
  - `09234500` Green River, WY (intermountain, regulated) — high-elevation snowpack regime.
  - `08030500` Sabine River, TX/LA (Gulf coastal, regulated) — hurricane-prone, tidal influence.
  - `08279500` Rio Grande at Embudo, NM (arid/semi-arid, regulated) — monsoon vs. snowmelt interplay.
  - `06807000` Missouri River at Nebraska City, NE (Great Plains, regulated) — major stem with flood-control operations.
  - `05420500` Mississippi River at Clinton, IA (agricultural, regulated) — large basin with levee management.
  - `01031500` Piscataquis River, ME (boreal transition, unregulated) — contrasting unregulated northeastern catchment.

### Comparative Experiment Plan
- **Per-site training:**
  - Reuse `modeling/train_quick_transformer_torch.py` and `modeling/train_quick_lstm_torch.py` with station-specific filters and identical hyperparameters for baseline comparison.
  - Store artifacts under `data/clean/modeling/<usgs_id>/` with shared metrics schema to simplify aggregation.
- **Metrics & diagnostics:**
  - Compute RMSE/MAE/NSE/KGE/PBIAS/ρ for raw NWM, Hydra, LSTM at validation and holdout periods.
  - Generate calibration outputs (coverage, PIT, reliability) for Hydra quantile runs.
- **Aggregation:**
  - Summarize improvements by biome and regulation class to test hypotheses about regime sensitivity.
  - Highlight exemplar hydrographs for contrasting behaviours (e.g., snowmelt-driven vs. flash flood regimes).

### Narrative Arc
1. **Motivation:** NWM residual biases impede decision-making; site-aware ML corrections aim to bridge physics and data-driven models.
2. **Methodology:** Consistent, leakage-safe pipeline with transformer vs. LSTM baselines ensures fair comparison; per-site training underscores operational viability.
3. **Results:** Present comparative metrics and plots across stations, emphasizing common gains (RMSE reductions) and regime-specific challenges (e.g., regulated flood-control sites).
4. **Discussion:** Interpret why Hydra generalizes better (attention, FiLM conditioning) and where it struggles (regulation changes, tidal influence); connect to hydrologic theory.
5. **Implications:** Outline deployment strategy (web app, reproducible scripts) and future research (multi-task models, foundation-model revisit once data contract stabilized).

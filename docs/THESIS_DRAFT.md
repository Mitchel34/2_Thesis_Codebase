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
- Rationale: creating a single, normalized dataset reduces feature-engineering drift and guarantees reproducible experiments across baseline and foundation models.

## Model Architectures
### Hybrid Transformer v2 (`modeling/train_quick_transformer_torch.py` + `modeling/models/hydra_temporal.py`)
- Causal TCN stem with dilated residual blocks feeds a transformer encoder, with channel attention pooling the final sequence state.
- Static features FiLM-condition the temporal channels while gain+bias heads provide physics-guided corrections on top of the residual path.
- Residual and corrected outputs include heteroscedastic log-variances and are trained with Gaussian NLL on asinh-transformed targets plus raw-scale consistency losses.
- Ensemble of direct, residual, and gain+bias corrected streams reduces systematic error and stabilizes high-flow extremes.
- Motivation: increase capacity and bias control without reworking the data pipeline or resorting to external foundation weights.

### Residual LSTM Baseline (`modeling/train_quick_lstm_torch.py`)
### Residual LSTM Baseline (`modeling/train_quick_lstm_torch.py`)
- Stacked LSTM encoder (configurable depth, bidirectionality) processes normalized dynamic sequences; dropout mitigates overfitting on smaller windows.
- Optional static-feature projection mirrors the transformer fusion so both models ingest identical information.
- Dual linear heads output residual and corrected predictions with the same focal/consistency/bias-penalty loss composition for parity with the transformer.
- Shares data preparation, augmentation, and metric computation paths with the transformer script for reproducible comparisons.
- Motivation: provides a strong recurrent baseline rooted in hydrologic literature, quantifying the incremental value of transformer mechanisms.
### Hugging Face Foundation Transformer (`modeling/train_hf_foundation.py`)
- Loads a pre-trained `TimeSeriesTransformerForPrediction` checkpoint (default `kashif/timeseries-transformer-tourism-hourly`) via the Transformers library and fine-tunes it on residual targets.
- Uses residual-only context windows (default 168 hours) with teacher forcing and autoregressive generation to produce next-step corrections.
- Generation outputs residual forecasts that are recombined with NWM baselines to compute corrected flows using the shared hydro metric suite.
- Provides a reproducible CLI aligned with other training scripts, with outputs saved under `data/clean/modeling/` for direct comparison.
- Motivation: evaluate whether a general-purpose foundation model can transfer to hydrologic residual correction with limited fine-tuning.
### Hydra + Foundation Hybrid (`modeling/train_hydra_foundation_torch.py`)
- Reuses Hydra's convolutional branch, recent-history MLP, channel attention, and dual residual/corrected heads while swapping the internal encoder for a pre-trained `TimeSeriesTransformerModel`.
- Optional freezing of the backbone allows rapid benchmarking of foundation representations versus fully fine-tuned encoders.
- Maintains the same focal/consistency training objective, static-feature handling, and artifact outputs to keep comparisons consistent across models.
- Motivation: test whether combining Hydra's inductive biases with foundation pretraining yields additive gains over either approach alone.

## Preliminary Results and Discussion
### Experiment Setup
- Dataset: `data/clean/modeling/hourly_training_2023-01-01_2023-01-31.parquet` (hourly sequences for the study site cohort).
- Splits: 21 days for training, 7 days for validation, remaining hours for evaluation; both models trained for up to 20 epochs with early stopping (patience 4).
- Optimization: AdamW (no Ranger available), focal + consistency loss stack, gradient clipping at 1.0, data augmentation enabled for training batches.
### Metric Snapshot (Evaluation on 2023-01-28 onward)
- Raw NWM baseline: RMSE 0.708 cms, MAE 0.588 cms, NSE -7.73, KGE -0.68, PBIAS +8.17%, r = 0.51.
- Hybrid transformer v2: RMSE 0.440 cms (37.9% improvement vs. NWM), MAE 0.332 cms, NSE -2.37, KGE -0.01, PBIAS -3.50%, r = 0.48. Residual RMSE 0.57 cms.
- Residual LSTM: RMSE 1.351 cms (90.8% worse than NWM), MAE 1.31 cms, NSE -30.81, KGE 0.23, PBIAS +23.57%, r = 0.41. Residual RMSE 0.95 cms.
### Interpretation
- Upgraded Hydra v2 cuts RMSE by ~38% versus raw NWM and >25% versus the earlier transformer, with gain/bias fusion shaving systematic error while heteroscedastic NLL dampens noisy high-flow spikes.
- Residual predictions tighten substantially (RMSE 0.57 cms vs. 3.36 cms previously), feeding more stable corrected flows; the LSTM baseline still overfits to mean bias and needs curriculum/regularisation sweeps.
- Next steps: scale Hydra v2 to 2010–2022, tune gain scaling and variance clamping, and explore quantile/MoE heads for calibrated uncertainty.

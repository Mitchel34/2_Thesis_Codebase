# Evaluation & Multi-Site Workflow Upgrade

## Plotting Pipeline Goals
- Replace `modeling/plot_quick_eval.py` with a parameterized module that operates on Parquet/CSV artifacts and supports station-level + aggregated plots.
- Standard outputs:
  - Hydrographs faceted by station (USGS vs. NWM vs. corrected).
  - Residual density & QQ plots per station.
  - Reliability diagrams and PIT histograms for quantile runs.
  - Skill score bar charts (ΔRMSE/MAE/NSE) across stations and hydrologic regimes.
  - Scatter w/ density shading for peak-flow vs. low-flow behaviour.
- Implement consistent styling (AGU-compliant fonts/palette), auto-save to `results/figures/` (ignored by git), optional PDF/SVG for manuscript.
- Script skeleton:
  ```bash
  python -m modeling.plot_suite \
    --eval-csv data/clean/modeling/hydra_v2_full_q_sweep3_eval.csv \
    --station-meta config/master_study_sites.py \
    --out-dir results/figures/hydra_v2_sweep3 \
    --metrics-json data/clean/modeling/hydra_v2_full_q_sweep3_metrics.json
  ```
- Refactor plotting code into composable functions (load_eval, compute_skill, make_hydrograph, make_reliability, etc.) for reuse in notebooks or app.

## Multi-Site Modeling Workflow
- Extend dataset builder to emit site-aware shards (groupby `comid`) with metadata columns (region, biome, regulation).
- Training loop strategy:
  1. Baseline Hydra v2 + LSTM for each site independently using identical hyperparameters for comparability.
  2. Optional multi-task experiment: shared model with site embedding to assess transfer.
- Automation:
  - Create `scripts/train_all_sites.sh` to iterate over `MASTER_STUDY_SITES` entries with available NWM COMIDs, logging to `logs/train_<site>.log` and storing metrics under `data/clean/modeling/<site>/`.
  - Use consistent filename schema: `hydra_v2_<site>_metrics.json`, `hydra_v2_<site>_eval.csv`, etc.
- Aggregation:
  - Build notebook or script to compile metrics across stations into a table (RMSE, MAE, NSE, PBIAS, correlation) and compute improvements vs. NWM.
  - Tag results by biome/region/regulation to support narrative about regime dependence.

## Immediate Action Items
1. Draft new plotting module (`modeling/plot_suite.py`) with hydrograph + error distribution prototypes for Watauga.
2. Update `.gitignore` as needed to keep `results/figures/` out of version control.
3. Modify `modeling/build_training_dataset.py` to allow filtering by `usgs_id` and to emit per-site metadata in the clean dataset.
4. Create CLI wrappers for site-wise Hydra and LSTM training (use config file or CLI arguments for station selection).
5. Generate first pass of multi-site metrics once additional sites have data pulled; feed into thesis narrative work.

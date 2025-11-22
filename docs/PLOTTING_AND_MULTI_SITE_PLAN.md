# Evaluation Plan (Watauga Focus)

## Plotting Pipeline Goals
- Replace `modeling/plot_quick_eval.py` with a parameterized module that operates on Parquet/CSV artifacts and supports Watauga-focused station plots now, while keeping hooks for future aggregation.
- Standard outputs (initially just for Watauga, driven by Trial 1 Hydra metrics and matching LSTM/NWM baselines):
  - Hydrographs faceted by station (USGS vs. NWM vs. corrected).
  - Residual density & QQ plots per station.
  - Reliability diagrams and PIT histograms for quantile runs.
  - Skill score bar charts (ΔRMSE/MAE/NSE) across stations and hydrologic regimes.
  - Scatter w/ density shading for peak-flow vs. low-flow behaviour.
- Implement consistent styling (AGU-compliant fonts/palette), auto-save to `results/figures/` (ignored by git), optional PDF/SVG for manuscript.
- Script skeleton (single-site, Watauga-only for now):
  ```bash
  python -m modeling.plot_suite \
    --eval-csv data/clean/modeling/hydra_v2_full_q_sweep3_eval.csv \
    --station-meta config/master_study_sites.py \
    --out-dir results/figures/hydra_v2_sweep3 \
    --metrics-json data/clean/modeling/hydra_v2_full_q_sweep3_metrics.json
  ```
- Refactor plotting code into composable functions (load_eval, compute_skill, make_hydrograph, make_reliability, etc.) for reuse in notebooks or app, with clear extension points once additional sites are onboarded.
  - `load_eval()` should accept the Trial 1 evaluation CSV (`watauga_multiobj_trial1_eval.csv`) plus companion NWM/LSTM CSVs.
  - `compute_skill()` should emit a tidy table summarizing RMSE/MAE/NSE/KGE/PBIAS deltas for manuscript tables.

## Future Multi-Site Workflow (Deferred)
- Document the desired extensions now, but treat them explicitly as future work after Watauga analysis locks.
- Extend dataset builder to emit site-aware shards (groupby `comid`) with metadata columns (region, biome, regulation).
- Training loop strategy to revisit later:
  1. Baseline Hydra v2 + LSTM for each site independently using identical hyperparameters for comparability.
  2. Optional multi-task experiment: shared model with site embedding to assess transfer.
- Automation ideas to pick up once additional sites are ready:
  - Create `scripts/train_all_sites.sh` to iterate over `MASTER_STUDY_SITES` entries with available NWM COMIDs, logging to `logs/train_<site>.log` and storing metrics under `data/clean/modeling/<site>/`.
  - Use consistent filename schema: `hydra_v2_<site>_metrics.json`, `hydra_v2_<site>_eval.csv`, etc.
- Aggregation goals for later:
  - Build notebook or script to compile metrics across stations into a table (RMSE, MAE, NSE, PBIAS, correlation) and compute improvements vs. NWM.
  - Tag results by biome/region/regulation to support narrative about regime dependence once multi-site data exists.

## Immediate Action Items
1. Draft new plotting module (`modeling/plot_suite.py`) with hydrograph + error distribution prototypes for Watauga using the Trial 1 evaluation artifacts.
2. Extract the required CSV/JSON bundles (Hydra Trial 1, LSTM baseline, raw NWM) into `data/clean/modeling/` and document their paths for plotting scripts.
3. Update `.gitignore` as needed to keep `results/figures/` out of version control.
4. Ensure `modeling/build_training_dataset.py` cleanly filters to the Watauga `usgs_id` and tags any site metadata needed for plots.
5. Create CLI wrappers for Watauga Hydra and LSTM training to streamline reproducibility.
6. Capture a short "future work" note outlining how the plotting + training stack generalizes once new sites are prioritized.

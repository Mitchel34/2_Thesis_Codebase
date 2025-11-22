# Visualization Toolkit

Utilities for generating Water Resources Research (WRR) quality figures for the Watauga study.

## Installation

```bash
pip install matplotlib pandas numpy seaborn geopandas shapely contextily graphviz
```

Graphviz binaries are required for diagram rendering. On macOS:

```bash
brew install graphviz
```

## Commands

```bash
python -m viz.plot_watauga_map --lat 36.215 --lon -81.687 --out figs/02_watauga_map.pdf
python -m viz.plot_temporal_splits --splits configs/train_val_test.json --rolling configs/rolling_windows.json --out figs/03_temporal_splits.pdf
python -m viz.plot_annual_hydrograph --eval artifacts/watauga_best_eval/watauga_batchsweep_bs1024_lr8e4_shift01_eval.csv --out figs/04_watauga_full_year.pdf
python -m viz.render_model_architecture --out figs/05_architecture.pdf
python -m viz.render_pipeline_diagram --out figs/pipeline.pdf
python -m viz.plot_results_panels --eval artifacts/watauga_best_eval/watauga_batchsweep_bs1024_lr8e4_shift01_eval.csv --out figs/06_results_panels.pdf
python -m viz.plot_metric_improvements --metrics artifacts/watauga_best_eval/watauga_batchsweep_bs1024_lr8e4_shift01_metrics.json --out figs/07_metric_improvements.pdf
```

Use `--start`/`--end` on `plot_results_panels` to zoom into a specific storm period.

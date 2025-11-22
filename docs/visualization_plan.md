# WRR Visualization Generation Plan

## 1. Inputs and Data Sources

### 1.1 Geographic and Hydrologic Inputs
- `config/master_study_sites.py` provides lat/lon, gauge IDs, watershed descriptors.
- Optional watershed shapefiles under `data/shapefiles/watauga/`.

### 1.2 Prediction and Evaluation Inputs
- Primary evaluation table: `outputs/eval/batchsweep_bs1024_lr8e4_shift01_eval.csv` (NWM, ML corrections, metrics).
- Recommended supplemental exports: `outputs/eval/per_event_metrics.csv`, `outputs/eval/per_month_metrics.csv`, `outputs/eval/hydrograph_samples/*.csv`.

### 1.3 Metadata for Diagrams
- Model configs: `configs/model_config.yaml`, `configs/data_config.yaml`.
- Temporal splits: `configs/train_val_test.json`, `configs/rolling_windows.json`.

## 2. `viz/` Package Structure
```
viz/
    __init__.py
    style.py
    colors.py
    utils.py
    plot_watauga_map.py
    plot_temporal_splits.py
    render_model_architecture.py
    render_pipeline_diagram.py
    plot_results_panels.py
    plot_skill_maps.py            # optional
    plot_event_skill.py           # optional
    plot_scatter_density.py       # optional
    README.md
```

### 2.1 WRR Style Defaults (`viz/style.py`)
- Font: DejaVu Sans or STIXGeneral.
- Axis width 1.1 pt, line width 1.5 pt, tick width 1.1 pt.
- Panel labels (A, B, C) upper-left; y-grid only; DPI 300.
- Palettes: cividis/viridis for continuous, RdBu_r for differences.

### 2.2 Color Palette (`viz/colors.py`)
```python
COLORS = {
    "obs": "#1b9e77",
    "nwm": "#d95f02",
    "ml":  "#7570b3",
    "event": "#aaaaaa",
}
```

## 3. Figure Specifications

### Figure 2 – Watauga Basin Map (`viz/plot_watauga_map.py`)
- Purpose: show study area, highlight gauge, watershed.
- Inputs: lat/lon, optional shapefile.
- Implementation: geopandas + contextily if available; fallback to Matplotlib outlines.
- Style: thin outlines, star marker for USGS site, north arrow, scale bar, inset map.

### Figure 3 – Temporal Split Diagram (`viz/plot_temporal_splits.py`)
- Purpose: visualize Train/Val/Test and rolling-origin windows with horizontal bars.
- Inputs: JSON configs.
- Style: color-coded bars, month ticks, labels inside bars, subtle vertical grid.

### Figure 4 – Full-Year Hydrograph (`viz/plot_annual_hydrograph.py`)
- Purpose: show the entire evaluation year so reviewers can see seasonal dynamics and multiple storm regimes in one glance.
- Inputs: evaluation CSV (default `artifacts/watauga_best_eval/watauga_batchsweep_bs1024_lr8e4_shift01_eval.csv`). Optional `--start/--end` filters plus `--resample` (e.g., `D`) for daily means.
- Style: monthly tick marks, WRR palette lines (obs/NWM/corrected), legend across the top, optional annotations for peaks.

### Figure 5 – Architecture Diagram (`viz/render_model_architecture.py`)
- Purpose: depict GRU + transformer hybrid.
- Implementation: Graphviz DOT (preferred) with nodes for inputs, GRU, transformer, pooling, outputs; export PDF + PNG.

### Figure 6 – Results Panels (`viz/plot_results_panels.py`)
- Panels: (A) Hydrograph zoom, (B) Scatter density (hexbin) with 1:1 line, (C) Residual histogram, (D) Monthly metric boxplots.
- Inputs: evaluation CSV + grouped metrics.
- Style: consistent colors, panel labels, axis titles with units.

### Figure 7 – Metric Improvements (`viz/plot_metric_improvements.py`)
- Purpose: highlight how each core metric changes from baseline to corrected, with text annotations for absolute and percent improvements.
- Inputs: metrics JSON (`artifacts/watauga_best_eval/watauga_batchsweep_bs1024_lr8e4_shift01_metrics.json`).
- Style: stacked horizontal mini-plots, baseline vs corrected bars per metric, improvement text inside subplot, WRR colors for baseline/corrected.

### Additional Figures (optional but recommended)
- `viz/plot_skill_maps.py`: ΔNSE maps or event skill summary.
- `viz/plot_event_skill.py`: peak event timing/magnitude errors.
- `viz/plot_scatter_density.py`: standalone scatter density figure.

## 4. Workflow Diagrams

### Pipeline Diagram (`viz/render_pipeline_diagram.py`)
- Use Graphviz to show: data ingest → preprocessing → model → correction → evaluation → visualization.
- Shapes: rectangles for processes, parallelograms for data; muted WRR colors; export PDF/PNG.

## 5. CLI and Automation

### Example Commands
```
python -m viz.plot_watauga_map --lat 36.215 --lon -81.687 --out figs/02_watauga_map.pdf
python -m viz.plot_temporal_splits --splits configs/train_val_test.json --rolling configs/rolling_windows.json --out figs/03_temporal_splits.pdf
python -m viz.plot_annual_hydrograph --eval artifacts/watauga_best_eval/watauga_batchsweep_bs1024_lr8e4_shift01_eval.csv --out figs/04_watauga_full_year.pdf
python -m viz.render_model_architecture --out figs/05_architecture.pdf
python -m viz.plot_results_panels --eval artifacts/watauga_best_eval/watauga_batchsweep_bs1024_lr8e4_shift01_eval.csv --out figs/06_results_panels.pdf
python -m viz.plot_metric_improvements --metrics artifacts/watauga_best_eval/watauga_batchsweep_bs1024_lr8e4_shift01_metrics.json --out figs/07_metric_improvements.pdf
python -m viz.render_pipeline_diagram --out figs/pipeline.pdf
```

### README Guidance
- Installation requirements: matplotlib, seaborn, numpy, pandas, scipy, geopandas, shapely, contextily, graphviz.
- Usage instructions for each script.
- Single regeneration script: `scripts/build_all_figs.sh`.

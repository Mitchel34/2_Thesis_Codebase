#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="figs"
mkdir -p "$OUT_DIR"

python -m viz.plot_watauga_map --lat 36.215 --lon -81.687 --out "$OUT_DIR/02_watauga_map.pdf"
python -m viz.plot_temporal_splits --splits configs/train_val_test.json --rolling configs/rolling_windows.json --out "$OUT_DIR/03_temporal_splits.pdf"
python -m viz.plot_annual_hydrograph --eval artifacts/watauga_best_eval/watauga_batchsweep_bs1024_lr8e4_shift01_eval.csv --out "$OUT_DIR/04_watauga_full_year.pdf"
python -m viz.render_model_architecture --out "$OUT_DIR/05_architecture.pdf"
python -m viz.render_pipeline_diagram --out "$OUT_DIR/pipeline.pdf"
python -m viz.plot_results_panels --eval artifacts/watauga_best_eval/watauga_batchsweep_bs1024_lr8e4_shift01_eval.csv --out "$OUT_DIR/06_results_panels.pdf"
python -m viz.plot_metric_improvements --metrics artifacts/watauga_best_eval/watauga_batchsweep_bs1024_lr8e4_shift01_metrics.json --out "$OUT_DIR/07_metric_improvements.pdf"

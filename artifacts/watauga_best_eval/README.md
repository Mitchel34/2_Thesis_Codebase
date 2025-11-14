## Watauga Best-Run Evaluation Snapshot

These files capture the evaluation artefacts for the Watauga River configuration that beats the raw NWM baseline on every headline metric (`batchsweep_bs1024_lr8e4_shift01`, archived in `archive/baselines/batchsweep_bs1024_lr8e4_shift01_20251020.json`).

### Contents

| File | Description |
| --- | --- |
| `watauga_batchsweep_bs1024_lr8e4_shift01_eval.csv` | Hourly evaluation dataframe for the 2022 deployment window. Columns include `timestamp`, `usgs_cms`, `nwm_cms`, `corrected_pred_cms`, per-target quantiles, residual diagnostics, and metadata flags. |
| `watauga_batchsweep_bs1024_lr8e4_shift01_metrics.json` | Aggregate metrics (RMSE, NSE, KGE, PBIAS, correlations, quantile coverage, bias-shift information). Mirrors the entry in `archive/baselines/…` but is colocated here for quick access by plotting scripts. |

### Usage

Pass these paths to the plotting helpers, for example:

```bash
PYTHONPATH=. .venv/bin/python3 modeling/plot_watauga_results.py \
  --metrics-json artifacts/watauga_best_eval/watauga_batchsweep_bs1024_lr8e4_shift01_metrics.json \
  --eval-csv artifacts/watauga_best_eval/watauga_batchsweep_bs1024_lr8e4_shift01_eval.csv \
  --output-dir results/figures/watauga \
  --site-name "Watauga River, NC"
```

These artefacts are lightweight (∼8 k rows) and safe to keep under version control so that external visualization agents (e.g., Overleaf or Atlas integrations) can generate the manuscript figures without needing the full raw datasets.

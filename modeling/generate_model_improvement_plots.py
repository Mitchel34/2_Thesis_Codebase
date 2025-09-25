#!/usr/bin/env python3
"""Generate box plots that highlight corrected-model improvements over NWM.

The script expects an evaluation CSV produced by ``train_quick_transformer_torch.py``
or a compatible workflow. The file must include the following columns:
``timestamp``, ``nwm_cms``, ``usgs_cms`` (or ``corrected_true_cms``), and
``corrected_pred_cms``.

Outputs:
- Box plot of hourly absolute errors (baseline vs corrected)
- Daily metric box plots (RMSE, MAE, NSE) comparing baseline vs corrected
- Daily improvement box plots for the same metrics
- Optional monthly versions when at least two months are available
- CSV exports of the computed daily/monthly metrics

Figures are written to the requested output directory (default ``results/plots``).
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable

import numpy as np
import pandas as pd

cache_root = os.environ.setdefault('XDG_CACHE_HOME', os.path.join(os.getcwd(), '.cache'))
os.makedirs(cache_root, exist_ok=True)
fontconfig_dir = os.path.join(cache_root, 'fontconfig')
os.makedirs(fontconfig_dir, exist_ok=True)
mpl_cache_dir = os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), '.mplconfig'))
os.makedirs(mpl_cache_dir, exist_ok=True)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('ggplot')


def _filter_valid(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays with NaN pairs removed."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _filter_valid(y_true, y_pred)
    if y_true.size == 0:
        return float('nan')
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _filter_valid(y_true, y_pred)
    if y_true.size == 0:
        return float('nan')
    return float(np.mean(np.abs(y_pred - y_true)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _filter_valid(y_true, y_pred)
    if y_true.size == 0:
        return float('nan')
    return float(np.mean(y_pred - y_true))


def pbias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _filter_valid(y_true, y_pred)
    if y_true.size == 0:
        return float('nan')
    denom = np.sum(y_true)
    if np.isclose(denom, 0.0):
        return float('nan')
    return float(100.0 * np.sum(y_pred - y_true) / denom)


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _filter_valid(y_true, y_pred)
    if y_true.size < 2:
        return float('nan')
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if np.isclose(denom, 0.0):
        return float('nan')
    return float(1.0 - np.sum((y_pred - y_true) ** 2) / denom)


def corr_coef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _filter_valid(y_true, y_pred)
    if y_true.size < 2:
        return float('nan')
    if np.isclose(np.std(y_true), 0.0) or np.isclose(np.std(y_pred), 0.0):
        return float('nan')
    return float(np.corrcoef(y_true, y_pred)[0, 1])


METRIC_FUNCS: Dict[str, callable] = {
    'RMSE': rmse,
    'MAE': mae,
    'Bias': bias,
    'PBIAS': pbias,
    'NSE': nse,
    'R': corr_coef,
}

METRICS_FOR_PLOTS: Iterable[str] = ('RMSE', 'MAE', 'NSE')

METRIC_DIRECTION = {
    'RMSE': 'lower',
    'MAE': 'lower',
    'NSE': 'higher',
}


def load_evaluation_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Evaluation CSV not found: {path}")
    df = pd.read_csv(path, parse_dates=['timestamp'])

    required = {'timestamp', 'nwm_cms', 'corrected_pred_cms'}
    if 'corrected_true_cms' not in df.columns:
        if 'usgs_cms' in df.columns:
            df['corrected_true_cms'] = df['usgs_cms']
        else:
            raise ValueError("CSV must include 'corrected_true_cms' or 'usgs_cms'.")
    required.add('corrected_true_cms')
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    df = df.sort_values('timestamp').reset_index(drop=True)
    df['truth_cms'] = df['corrected_true_cms'].astype(float)
    df['nwm_cms'] = df['nwm_cms'].astype(float)
    df['corrected_pred_cms'] = df['corrected_pred_cms'].astype(float)
    if 'site_name' in df.columns:
        df['site_name'] = df['site_name'].fillna('unknown')
    return df


def compute_group_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group_value, group in df.groupby(group_col):
        y_true = group['truth_cms'].values
        baseline = group['nwm_cms'].values
        corrected = group['corrected_pred_cms'].values
        valid_pairs = np.isfinite(y_true) & np.isfinite(baseline) & np.isfinite(corrected)
        entry = {'group_value': group_value, 'count': int(np.sum(valid_pairs))}
        for label, series in [('NWM', baseline), ('Corrected', corrected)]:
            for metric_name, func in METRIC_FUNCS.items():
                col = f'{metric_name}_{label}'
                entry[col] = func(y_true, series)
        rows.append(entry)
    return pd.DataFrame(rows)


def save_metrics_table(df: pd.DataFrame, path: str) -> None:
    if df.empty:
        return
    df_sorted = df.sort_values('group_value')
    df_sorted.to_csv(path, index=False)


def plot_absolute_error_boxplot(df: pd.DataFrame, out_path: str, label: str) -> None:
    errors = [
        df['nwm_cms'] - df['truth_cms'],
        df['corrected_pred_cms'] - df['truth_cms'],
    ]
    abs_errors = [np.abs(e[np.isfinite(e)]) for e in errors]
    if not any(arr.size for arr in abs_errors):
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(abs_errors, tick_labels=['NWM', 'Corrected'])
    ax.set_ylabel('Absolute error (cms)')
    ax.set_title(f'Hourly Absolute Error Distribution ({label})')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_metric_boxplot(metric_df: pd.DataFrame, metric: str, out_path: str, label: str, group_desc: str) -> None:
    col_base = f'{metric}_NWM'
    col_corr = f'{metric}_Corrected'
    if col_base not in metric_df or col_corr not in metric_df:
        return
    values = [
        metric_df[col_base].values.astype(float),
        metric_df[col_corr].values.astype(float),
    ]
    values = [v[np.isfinite(v)] for v in values]
    if not any(arr.size for arr in values):
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(values, tick_labels=['NWM', 'Corrected'])
    ax.set_ylabel(metric)
    ax.set_title(f'{group_desc} {metric} Comparison ({label})')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_improvement_boxplot(metric_df: pd.DataFrame, metric: str, out_path: str, label: str, group_desc: str) -> None:
    col_base = f'{metric}_NWM'
    col_corr = f'{metric}_Corrected'
    if col_base not in metric_df or col_corr not in metric_df:
        return
    base_vals = metric_df[col_base].values.astype(float)
    corr_vals = metric_df[col_corr].values.astype(float)
    mask = np.isfinite(base_vals) & np.isfinite(corr_vals)
    if not np.any(mask):
        return
    direction = METRIC_DIRECTION.get(metric, 'lower')
    if direction == 'lower':
        improvement = base_vals[mask] - corr_vals[mask]
    else:
        improvement = corr_vals[mask] - base_vals[mask]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([improvement], tick_labels=['Improvement'])
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ylabel = 'Positive = Improved'
    ax.set_ylabel(ylabel)
    ax.set_title(f'{group_desc} {metric} Improvement ({label})')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate box plots showing corrected-model improvements over NWM.')
    parser.add_argument('--eval-csv', default=os.path.join('data', 'clean', 'modeling', 'quick_eval.csv'), help='Path to evaluation CSV (default: %(default)s)')
    parser.add_argument('--output-dir', default=os.path.join('results', 'plots'), help='Directory to write plots and metrics (default: %(default)s)')
    parser.add_argument('--label', default='All Sites', help='Label used in plot titles (e.g., station or basin name)')
    args = parser.parse_args()

    df = load_evaluation_data(args.eval_csv)
    ensure_directory(args.output_dir)

    abs_plot_path = os.path.join(args.output_dir, 'hourly_abs_error_boxplot.png')
    plot_absolute_error_boxplot(df, abs_plot_path, args.label)

    # Daily metrics
    df['date'] = df['timestamp'].dt.date
    daily_metrics = compute_group_metrics(df, 'date')
    if not daily_metrics.empty:
        save_metrics_table(daily_metrics, os.path.join(args.output_dir, 'daily_metrics.csv'))
        for metric in METRICS_FOR_PLOTS:
            out_metric = os.path.join(args.output_dir, f'daily_{metric.lower()}_boxplot.png')
            _plot_metric_boxplot(daily_metrics, metric, out_metric, args.label, 'Daily')
            out_improvement = os.path.join(args.output_dir, f'daily_{metric.lower()}_improvement_boxplot.png')
            _plot_improvement_boxplot(daily_metrics, metric, out_improvement, args.label, 'Daily')

    # Monthly metrics (plots render even when only a single month is available)
    df['month'] = df['timestamp'].dt.to_period('M').astype(str)
    monthly_metrics = compute_group_metrics(df, 'month')
    if not monthly_metrics.empty:
        save_metrics_table(monthly_metrics, os.path.join(args.output_dir, 'monthly_metrics.csv'))
        for metric in METRICS_FOR_PLOTS:
            out_metric = os.path.join(args.output_dir, f'monthly_{metric.lower()}_boxplot.png')
            _plot_metric_boxplot(monthly_metrics, metric, out_metric, args.label, 'Monthly')
            out_improvement = os.path.join(args.output_dir, f'monthly_{metric.lower()}_improvement_boxplot.png')
            _plot_improvement_boxplot(monthly_metrics, metric, out_improvement, args.label, 'Monthly')

    print(f'Plots and metrics saved to {args.output_dir}')


if __name__ == '__main__':
    main()

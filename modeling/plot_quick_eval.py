#!/usr/bin/env python3
"""
Read quick_eval.csv and produce simple evaluation plots:
- Box plot of absolute errors: baseline (|NWM-USGS|) vs corrected (|CorrectedPred-USGS|)
- Hydrograph for the test window: NWM vs USGS vs Corrected
- Scatter of NWM vs USGS and Corrected vs USGS with 1:1 line
Outputs saved under data/clean/modeling/ as PNG files.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN_CSV = os.path.join('data', 'clean', 'modeling', 'quick_eval.csv')
OUT_DIR = os.path.join('data', 'clean', 'modeling')

def main():
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"Missing {IN_CSV}. Run the trainer to generate it.")
    df = pd.read_csv(IN_CSV, parse_dates=['timestamp'])
    # Compute absolute errors
    df['abs_err_baseline'] = (df['nwm_cms'] - df['corrected_true_cms']).abs()
    df['abs_err_corrected'] = (df['corrected_pred_cms'] - df['corrected_true_cms']).abs()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Box plot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.boxplot([df['abs_err_baseline'].dropna(), df['abs_err_corrected'].dropna()], labels=['Baseline', 'Corrected'])
    ax.set_ylabel('Absolute error (cms)')
    ax.set_title('Absolute Error Distribution: Baseline vs Corrected')
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, 'quick_boxplot_abs_error.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Hydrograph
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df['timestamp'], df['corrected_true_cms'], label='USGS (true)', color='black', linewidth=1.5)
    ax.plot(df['timestamp'], df['nwm_cms'], label='NWM', color='tab:blue', alpha=0.8)
    ax.plot(df['timestamp'], df['corrected_pred_cms'], label='Corrected', color='tab:orange', alpha=0.8)
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Flow (cms)')
    ax.legend()
    ax.set_title('Hydrograph: Test Window')
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, 'quick_hydrograph.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Scatter plots
    lim = float(np.nanmax([df['nwm_cms'].max(), df['corrected_true_cms'].max(), df['corrected_pred_cms'].max()]))
    lim = max(lim, 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(8,4), sharex=True, sharey=True)
    axes[0].scatter(df['corrected_true_cms'], df['nwm_cms'], s=8, alpha=0.6)
    axes[0].plot([0, lim], [0, lim], 'k--', linewidth=1)
    axes[0].set_title('NWM vs USGS')
    axes[0].set_xlabel('USGS (cms)')
    axes[0].set_ylabel('NWM (cms)')

    axes[1].scatter(df['corrected_true_cms'], df['corrected_pred_cms'], s=8, alpha=0.6, color='tab:orange')
    axes[1].plot([0, lim], [0, lim], 'k--', linewidth=1)
    axes[1].set_title('Corrected vs USGS')
    axes[1].set_xlabel('USGS (cms)')

    fig.suptitle('Scatter vs USGS with 1:1 line')
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, 'quick_scatter.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Print quick stats
    base_rmse = float(np.sqrt(np.nanmean((df['nwm_cms'] - df['corrected_true_cms'])**2)))
    corr_rmse = float(np.sqrt(np.nanmean((df['corrected_pred_cms'] - df['corrected_true_cms'])**2)))
    rel_improve = 100.0 * (base_rmse - corr_rmse) / base_rmse if base_rmse > 0 else np.nan
    print(f"Saved plots to {OUT_DIR}")
    print(f"Baseline RMSE: {base_rmse:.3f} | Corrected RMSE: {corr_rmse:.3f} | Δ%: {rel_improve:.1f}%")

if __name__ == '__main__':
    main()

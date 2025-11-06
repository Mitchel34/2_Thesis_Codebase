#!/usr/bin/env python3
"""Utility script to generate key figures for the Watauga River analysis."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")


def load_metrics(metrics_path: Path) -> dict:
    with metrics_path.open() as fh:
        return json.load(fh)


def load_eval(eval_path: Path) -> pd.DataFrame:
    df = pd.read_csv(eval_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["residual_nwm"] = df["usgs_cms"] - df["nwm_cms"]
    df["residual_corrected"] = df["usgs_cms"] - df["corrected_pred_cms"]
    return df


def plot_hydrograph(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    site_name: str,
    out_path: Path,
) -> None:
    window = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    if window.empty:
        raise ValueError("Selected hydrograph window yielded no data.")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        window["timestamp"],
        window["usgs_cms"],
        label="USGS Observed",
        linewidth=1.5,
        color="black",
    )
    ax.plot(
        window["timestamp"],
        window["nwm_cms"],
        label="NWM Baseline",
        linewidth=1,
        color="red",
    )
    ax.plot(
        window["timestamp"],
        window["corrected_pred_cms"],
        label="Corrected (Transformer)",
        linewidth=1,
        color="green",
    )
    ax.set_ylabel("Discharge (cms)")
    ax.set_title(f"{site_name}: Observed vs. Baseline vs. Corrected")
    ax.legend(loc="upper right")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_monthly_timeseries(df: pd.DataFrame, out_path: Path) -> None:
    monthly = (
        df.set_index("timestamp")[["usgs_cms", "nwm_cms", "corrected_pred_cms"]]
        .resample("M")
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly["timestamp"], monthly["usgs_cms"], label="USGS Observed", color="black", linewidth=1.5)
    ax.plot(monthly["timestamp"], monthly["nwm_cms"], label="NWM Baseline", color="red", linewidth=1)
    ax.plot(
        monthly["timestamp"],
        monthly["corrected_pred_cms"],
        label="Corrected (Transformer)",
        color="green",
        linewidth=1,
    )
    ax.set_ylabel("Discharge (cms)")
    ax.set_title("Monthly Mean Discharge (Test Window)")
    ax.legend(loc="upper right")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_residual_histograms(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(
        np.percentile(df["residual_nwm"], 1),
        np.percentile(df["residual_nwm"], 99),
        40,
    )
    ax.hist(
        df["residual_nwm"],
        bins=bins,
        alpha=0.6,
        label="Residual (USGS - NWM)",
        density=True,
    )
    ax.hist(
        df["residual_corrected"],
        bins=bins,
        alpha=0.6,
        label="Residual (USGS - Corrected)",
        density=True,
    )
    ax.set_xlabel("Residual (cms)")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_quantile_reliability(metrics: dict, out_path: Path) -> None:
    quantile_info = metrics.get("quantiles", {})
    if not quantile_info:
        return

    levels = np.array(sorted(float(q) for q in quantile_info.keys()))
    coverages = np.array([quantile_info[f"{q:.2f}"]["coverage"] for q in levels])
    biases = np.array([quantile_info[f"{q:.2f}"]["bias"] for q in levels])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Reliability curve
    axes[0].plot(levels, levels, linestyle="--", color="gray", label="1:1 reference")
    axes[0].scatter(levels, coverages, color="tab:blue", s=40)
    axes[0].set_xlabel("Nominal Quantile")
    axes[0].set_ylabel("Observed Coverage")
    axes[0].set_title("Quantile Coverage")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # Bias per quantile
    axes[1].bar(levels, biases, width=0.04, color="tab:orange")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Quantile")
    axes[1].set_ylabel("Bias (cms)")
    axes[1].set_title("Quantile Bias")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    quantile = pd.qcut(df["usgs_cms"], q=10, labels=False, duplicates="drop")
    cmap = plt.cm.viridis
    scatter = ax.scatter(
        df["usgs_cms"],
        df["corrected_pred_cms"],
        c=quantile,
        s=8,
        alpha=0.5,
        cmap=cmap,
        label="Corrected",
    )
    ax.scatter(
        df["usgs_cms"],
        df["nwm_cms"],
        s=6,
        alpha=0.15,
        label="NWM",
        color="red",
    )
    lims = [
        0,
        max(np.percentile(df["usgs_cms"], 99), np.percentile(df["corrected_pred_cms"], 99)),
    ]
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Observed (cms)")
    ax.set_ylabel("Predicted (cms)")
    ax.set_title("Observed vs Predicted Discharge")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Observed Flow Decile")
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_cdf(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for resid, label, color in [
        (df["residual_nwm"], "USGS - NWM", "red"),
        (df["residual_corrected"], "USGS - Corrected", "green"),
    ]:
        sorted_vals = np.sort(resid.values)
        cdf = np.linspace(0, 1, len(sorted_vals), endpoint=False)
        ax.plot(sorted_vals, cdf, label=label, color=color)
    ax.set_xlabel("Residual (cms)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Residual CDF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_monthly_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    df_month = df.copy()
    df_month["month"] = df_month["timestamp"].dt.strftime("%b")
    order = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    data = [
        df_month.loc[df_month["month"] == m, "residual_corrected"].values
        for m in order
        if m in df_month["month"].unique()
    ]
    ax.boxplot(data, labels=[m for m in order if m in df_month["month"].unique()])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Residual (cms)")
    ax.set_title("Corrected Residuals by Month")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_bias_vs_percentile(df: pd.DataFrame, out_path: Path) -> None:
    percentiles = np.linspace(0, 100, 11)
    bins = np.percentile(df["usgs_cms"], percentiles)
    df["flow_bin"] = pd.cut(df["usgs_cms"], bins=bins, include_lowest=True, labels=False)
    groups = df.groupby("flow_bin")
    centers = []
    bias_nwm = []
    bias_corr = []
    for idx, group in groups:
        if group.empty:
            continue
        centers.append(group["usgs_cms"].median())
        bias_nwm.append(np.mean(group["residual_nwm"]))
        bias_corr.append(np.mean(group["residual_corrected"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(centers, bias_nwm, marker="o", label="USGS - NWM", color="red")
    ax.plot(centers, bias_corr, marker="o", label="USGS - Corrected", color="green")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Observed Flow (cms)")
    ax.set_ylabel("Mean Residual (cms)")
    ax.set_title("Bias vs Flow Quantile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def export_metric_table(metrics: dict, out_dir: Path) -> None:
    baseline = metrics["baseline"]
    corrected = metrics["corrected"]
    rows = []
    for metric in ["rmse", "mae", "nse", "kge", "pbias", "pearson_r", "spearman_r"]:
        rows.append(
            {
                "metric": metric.upper(),
                "baseline": baseline[metric],
                "corrected": corrected[metric],
                "delta": corrected[metric] - baseline[metric],
            }
        )
    df = pd.DataFrame(rows)
    csv_path = out_dir / "watauga_metrics_summary.csv"
    df.to_csv(csv_path, index=False)

    tex_path = out_dir / "watauga_metrics_summary.tex"
    with tex_path.open("w") as fh:
        fh.write(df.to_latex(index=False, float_format="%.3f"))

    # Skill bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics_to_plot = ["rmse", "nse", "kge", "pbias"]
    baseline_vals = [baseline[m] for m in metrics_to_plot]
    corrected_vals = [corrected[m] for m in metrics_to_plot]
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    ax.bar(x - width / 2, baseline_vals, width, label="NWM Baseline", color="red", alpha=0.7)
    ax.bar(x + width / 2, corrected_vals, width, label="Corrected", color="green", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics_to_plot])
    ax.set_title("Skill Metrics (Baseline vs Corrected)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "watauga_skill_bars.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Watauga result visualizations.")
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--site-name", default="Watauga River, NC")
    parser.add_argument("--hydro-start", default="2022-02-01")
    parser.add_argument("--hydro-end", default="2022-03-15")
    args = parser.parse_args()

    metrics = load_metrics(args.metrics_json)
    df = load_eval(args.eval_csv)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    hydro_start = pd.to_datetime(args.hydro_start)
    hydro_end = pd.to_datetime(args.hydro_end)

    plot_hydrograph(
        df,
        start=hydro_start,
        end=hydro_end,
        site_name=args.site_name,
        out_path=out_dir / "watauga_hydrograph.png",
    )
    plot_monthly_timeseries(df, out_dir / "watauga_monthly_means.png")
    plot_residual_histograms(df, out_dir / "watauga_residual_hist.png")
    plot_cdf(df, out_dir / "watauga_residual_cdf.png")
    plot_monthly_boxplot(df, out_dir / "watauga_monthly_bias_boxplot.png")
    plot_bias_vs_percentile(df, out_dir / "watauga_bias_vs_flow.png")
    plot_quantile_reliability(metrics, out_dir / "watauga_quantile_reliability.png")
    plot_scatter(df, out_dir / "watauga_observed_vs_predicted.png")
    export_metric_table(metrics, out_dir)


if __name__ == "__main__":
    main()

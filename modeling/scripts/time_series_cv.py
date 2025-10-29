#!/usr/bin/env python3
"""Rolling-origin cross-validation driver."""

import argparse
import json
import subprocess
from pathlib import Path
import sys
import os

import pandas as pd


FOLDS = [
    ("2010-01-01", "2012-12-31", "2013-01-01", "2013-12-31"),
    ("2010-01-01", "2013-12-31", "2014-01-01", "2014-12-31"),
    ("2010-01-01", "2014-12-31", "2015-01-01", "2015-12-31"),
    ("2010-01-01", "2015-12-31", "2016-01-01", "2016-12-31"),
    ("2010-01-01", "2016-12-31", "2017-01-01", "2017-12-31"),
    ("2010-01-01", "2017-12-31", "2018-01-01", "2018-12-31"),
    ("2010-01-01", "2018-12-31", "2019-01-01", "2019-12-31"),
    ("2010-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
    ("2010-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
]


def run_fold(args, dataset_start, fold_id, train_start, train_end, val_start, val_end):
    output_prefix = f"{args.output_prefix}_fold{fold_id}"
    metrics_path = Path("data/clean/modeling") / f"{output_prefix}_metrics.json"
    if args.skip_existing and metrics_path.exists():
        with open(metrics_path) as fh:
            return json.load(fh)
    if args.trainer == "transformer":
        cmd = [
            sys.executable,
            "-m",
            "modeling.train_quick_transformer_torch",
            "--data",
            args.data,
            "--seq-len",
            str(args.seq_len),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--patience",
            str(args.patience),
            "--d-model",
            str(args.d_model),
            "--num-heads",
            str(args.num_heads),
            "--num-layers",
            str(args.num_layers),
            "--conv-depth",
            str(args.conv_depth),
            "--dropout",
            str(args.dropout),
            "--lr",
            str(args.lr),
            "--weight-nse",
            str(args.weight_nse),
            "--weight-quantile",
            str(args.weight_quantile),
            "--flow-emphasis",
            str(args.flow_emphasis),
            "--consistency-weight",
            str(args.consistency_weight),
            "--weight-pbias",
            str(args.weight_pbias),
            "--residual-bias-weight",
            str(args.residual_bias_weight),
            "--bias-shift-alpha",
            str(args.bias_shift_alpha),
            "--bias-shift-pbias-target",
            str(args.bias_shift_pbias_target),
            "--bias-shift-qmin",
            str(args.bias_shift_qmin),
            "--bias-shift-qmax",
            str(args.bias_shift_qmax),
            "--bias-shift-weight-power",
            str(args.bias_shift_weight_power),
            "--bias-shift-strategy",
            args.bias_shift_strategy,
            "--weight-kge",
            str(args.weight_kge),
            "--quantiles",
            args.quantiles,
            "--quantile-weights",
            args.quantile_weights,
            "--train-start",
            train_start,
            "--train-end",
            train_end,
            "--val-start",
            val_start,
            "--val-end",
            val_end,
            "--output-prefix",
            output_prefix,
        ]
        if args.no_augment:
            cmd.append("--no-augment")
        if args.no_ranger:
            cmd.append("--no-ranger")
        if args.no_compile:
            cmd.append("--no-compile")
        if args.no_amp:
            cmd.append("--no-amp")
        if args.weight_pbias_final is not None:
            cmd.extend(["--weight-pbias-final", str(args.weight_pbias_final)])
        if args.weight_kge_final is not None:
            cmd.extend(["--weight-kge-final", str(args.weight_kge_final)])
    else:
        if dataset_start is None:
            raise ValueError("dataset_start must be provided for GRU trainer")
        train_start_ts = pd.Timestamp(train_start)
        val_start_ts = pd.Timestamp(val_start)
        val_end_ts = pd.Timestamp(val_end)
        dataset_start_ts = pd.Timestamp(dataset_start)
        train_days = int((val_start_ts - dataset_start_ts).days)
        val_days = int((val_end_ts - val_start_ts).days) + 1
        cmd = [
            sys.executable,
            "-m",
            "modeling.train_quick_lstm_torch",
            "--data",
            args.data,
            "--seq-len",
            str(args.seq_len),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--train-days",
            str(train_days),
            "--val-days",
            str(val_days),
            "--patience",
            str(args.patience),
            "--hidden-size",
            str(args.lstm_hidden_size),
            "--num-layers",
            str(args.lstm_num_layers),
            "--dropout",
            str(args.lstm_dropout),
            "--lr",
            str(args.lstm_lr),
            "--weight-decay",
            str(args.lstm_weight_decay),
            "--focal-gamma",
            str(args.lstm_focal_gamma),
            "--focal-std-factor",
            str(args.lstm_focal_std_factor),
            "--output-prefix",
            output_prefix,
        ]
        if args.lstm_bidirectional:
            cmd.append("--bidirectional")
        if args.no_augment:
            cmd.append("--no-augment")
        if args.no_ranger:
            cmd.append("--no-ranger")
        if args.no_amp:
            cmd.append("--no-amp")
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    subprocess.run(cmd, check=True, env=env)
    with open(metrics_path) as fh:
        metrics = json.load(fh)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--trainer", choices=["transformer", "gru"], default="transformer")
    parser.add_argument("--seq-len", type=int, default=168)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--conv-depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-nse", type=float, default=0.2)
    parser.add_argument("--weight-quantile", type=float, default=0.05)
    parser.add_argument("--flow-emphasis", type=float, default=0.2)
    parser.add_argument("--consistency-weight", type=float, default=2.5)
    parser.add_argument("--weight-pbias", type=float, default=0.0)
    parser.add_argument("--weight-pbias-final", type=float, default=None)
    parser.add_argument("--residual-bias-weight", type=float, default=0.0)
    parser.add_argument("--bias-shift-alpha", type=float, default=0.0)
    parser.add_argument("--bias-shift-pbias-target", type=float, default=5.0)
    parser.add_argument("--bias-shift-qmin", type=float, default=0.0)
    parser.add_argument("--bias-shift-qmax", type=float, default=1.0)
    parser.add_argument("--bias-shift-weight-power", type=float, default=0.0)
    parser.add_argument("--bias-shift-strategy", choices=["scaled", "full"], default="scaled")
    parser.add_argument("--weight-kge", type=float, default=0.05)
    parser.add_argument("--weight-kge-final", type=float, default=None)
    parser.add_argument("--quantiles", default="0.1,0.5,0.9")
    parser.add_argument("--quantile-weights", default="0.2,1.2,0.2")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-ranger", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--lstm-hidden-size", type=int, default=256)
    parser.add_argument("--lstm-num-layers", type=int, default=2)
    parser.add_argument("--lstm-dropout", type=float, default=0.1)
    parser.add_argument("--lstm-lr", type=float, default=1e-3)
    parser.add_argument("--lstm-weight-decay", type=float, default=1e-4)
    parser.add_argument("--lstm-focal-gamma", type=float, default=2.0)
    parser.add_argument("--lstm-focal-std-factor", type=float, default=2.0)
    parser.add_argument("--lstm-bidirectional", action="store_true")
    args = parser.parse_args()

    dataset_start = None
    if args.trainer == "gru":
        df_ts = pd.read_parquet(args.data, columns=["timestamp"])
        if df_ts.empty:
            raise ValueError("Dataset has no rows; cannot perform CV.")
        dataset_start = pd.Timestamp(df_ts["timestamp"].min())

    results = []
    for i, (ts, te, vs, ve) in enumerate(FOLDS, start=1):
        metrics = run_fold(args, dataset_start, i, ts, te, vs, ve)
        results.append({
            "fold": i,
            "train_start": ts,
            "train_end": te,
            "val_start": vs,
            "val_end": ve,
            "metrics": metrics,
        })

    summary_path = Path("data/clean/modeling") / f"{args.output_prefix}_cv_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved CV summary to {summary_path}")


if __name__ == "__main__":
    main()

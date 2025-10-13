#!/usr/bin/env python3
"""Hyperparameter optimisation for Hydra v2 using Optuna.

Example usage (M1/M2 Mac, MPS backend):

    PYTHONPATH=. python3.11 modeling/hpo_optuna.py \
        --data data/clean/modeling/hourly_training_2010-01-01_2022-12-31.parquet \
        --study-name hydra_full_opt \
        --n-trials 20 \
        --epochs 60 \
        --batch-size 64

This script runs full training loops, so expect long runtimes. By default it
uses early-stopping disabled and saves no artifacts during trials. The best
hyperparameters are printed at the end; you can rerun the main training script
with those settings to produce evaluation files/plots.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict

try:
    import optuna
    from optuna import pruners
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Optuna is required for hyperparameter optimisation. Install via `pip install optuna`.") from exc

from modeling.train_quick_transformer_torch import train_eval


@dataclass
class TrialResult:
    corrected_rmse: float
    metrics: Dict[str, Any]
    params: Dict[str, Any]


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Single Optuna trial."""

    # Sample hyperparameters.
    d_model = trial.suggest_categorical("d_model", [96, 128])
    num_layers = trial.suggest_int("num_layers", 3, 4)
    num_heads = trial.suggest_categorical("num_heads", [4, 8])
    conv_depth = trial.suggest_int("conv_depth", 3, 4)
    dropout = trial.suggest_float("dropout", 0.08, 0.16)
    lr = trial.suggest_float("lr", 2e-4, 5e-4, log=True)
    seq_len = trial.suggest_categorical("seq_len", [168, 240])

    quantiles = [0.1, 0.5, 0.9]

    # Train the model with sampled hyperparameters.
    trial_prefix = f"{args.output_prefix}_trial{trial.number}"
    metrics = train_eval(
        data_path=args.data,
        seq_len=seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_days=args.train_days,
        val_days=args.val_days,
        patience=args.patience,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        conv_depth=conv_depth,
        dropout=dropout,
        lr=lr,
        use_amp=not args.no_amp,
        use_compile=False,
        augment=not args.no_augment,
        use_ranger=not args.no_ranger,
        output_prefix=trial_prefix,
    )
    metrics_path = Path("data/clean/modeling") / f"{trial_prefix}_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with open(metrics_path) as fh:
        metrics = json.load(fh)

    corrected_rmse = metrics["corrected"]["rmse"]
    trial.set_user_attr("metrics", metrics)

    return float(corrected_rmse)


def create_study(args: argparse.Namespace) -> optuna.Study:
    storage = args.storage
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = pruners.MedianPruner(n_warmup_steps=2)
    if storage:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )
    return study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hydra v2 Optuna hyperparameter search")
    parser.add_argument("--data", required=True, help="Path to cleaned modeling parquet")
    parser.add_argument("--study-name", default="hydra_optuna")
    parser.add_argument("--storage", default=None, help="Optuna storage URL (e.g., sqlite:///hydra_opt.db)")
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-days", type=int, default=4018)
    parser.add_argument("--val-days", type=int, default=365)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--moe-experts", type=int, default=3)
    parser.add_argument("--output-prefix", default="hydra_v2_optuna")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-ranger", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    study = create_study(args)
    study.optimize(lambda t: objective(t, args), n_trials=args.n_trials)

    print("Best value (corrected RMSE):", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Dump trial metrics for record keeping.
    summary = []
    for trial in study.trials:
        entry = {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "metrics": trial.user_attrs.get("metrics"),
        }
        summary.append(entry)

    out_path = f"{args.output_prefix}_optuna_trials.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved trial history to {out_path}")


if __name__ == "__main__":
    main()

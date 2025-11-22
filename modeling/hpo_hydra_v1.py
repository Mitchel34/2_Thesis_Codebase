#!/usr/bin/env python3
"""Focused Optuna sweep for Hydra v1 (Hybrid Patch/Conv) architecture.

This runner tunes the new v1 architecture to outperform the LSTM baseline.
Targeting patch_size, lr, and d_model as primary hyperparameters.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence

import optuna
from optuna import pruners
from optuna.samplers import TPESampler

from modeling.train_quick_transformer_torch import train_eval

# Focused search space for Hydra v1
PARAM_DISTRIBUTIONS = {
    "patch_size": optuna.distributions.CategoricalDistribution([1, 2, 4, 6, 12, 24]),
    "d_model": optuna.distributions.CategoricalDistribution([64, 96, 128, 192]),
    "num_layers": optuna.distributions.IntDistribution(2, 4),
    "num_heads": optuna.distributions.CategoricalDistribution([4, 8]),
    "conv_depth": optuna.distributions.IntDistribution(2, 4),
    "dropout": optuna.distributions.FloatDistribution(0.1, 0.3),
    "lr": optuna.distributions.FloatDistribution(1e-4, 2e-3, log=True),
    # Keep bias shift params active as we just fixed the model to support it
    "bias_shift_alpha": optuna.distributions.FloatDistribution(0.05, 0.2),
}

BASELINE_METRICS = {
    "rmse": 5.9715,
    "nse": 0.5189,
    "kge": 0.6386,
    "pbias_abs": 5.9948,
}


def _read_metrics(prefix: str) -> Dict[str, float]:
    metrics_path = Path("data/clean/modeling") / f"{prefix}_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    with open(metrics_path) as fh:
        payload = json.load(fh)
    return payload


def _compute_objectives(metrics: Dict[str, Dict[str, float]], enforce: bool) -> Sequence[float]:
    corrected = metrics["corrected"]
    rmse = float(corrected["rmse"])
    nse = float(corrected["nse"])
    kge = float(corrected["kge"])
    pbias = float(corrected["pbias"])
    
    # Primary objective: Minimize RMSE
    # Secondary objectives: Maximize NSE/KGE, Minimize Abs PBIAS
    values = [rmse, -nse, -kge, abs(pbias)]
    
    if enforce:
        penalties = 0.0
        if nse <= BASELINE_METRICS["nse"]:
            penalties += 1.0
        if kge <= BASELINE_METRICS["kge"]:
            penalties += 1.0
        if abs(pbias) >= BASELINE_METRICS["pbias_abs"]:
            penalties += 1.0
        values = [values[0] + 10 * penalties, values[1], values[2], values[3] + 10 * penalties]
    return values


def objective(trial: optuna.Trial, args: argparse.Namespace) -> Sequence[float]:
    # Suggest hyperparameters
    patch_size = trial.suggest_categorical("patch_size", [1, 2, 4, 6, 12, 24])
    d_model = trial.suggest_categorical("d_model", [64, 96, 128, 192])
    num_heads = trial.suggest_categorical("num_heads", [4, 8])
    
    # Ensure d_model is divisible by num_heads
    if d_model % num_heads != 0:
        raise optuna.TrialPruned("num_heads must divide d_model")
        
    num_layers = trial.suggest_int("num_layers", 2, 4)
    conv_depth = trial.suggest_int("conv_depth", 2, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
    bias_shift_alpha = trial.suggest_float("bias_shift_alpha", 0.05, 0.2)

    trial_prefix = f"{args.output_prefix}_trial{trial.number}"
    
    try:
        train_eval(
            data_path=args.data,
            seq_len=168, # Fixed for now
            epochs=args.epochs,
            batch_size=1024, # Fixed for speed
            train_days=args.train_days,
            val_days=args.val_days,
            patience=args.patience,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            conv_depth=conv_depth,
            dropout=dropout,
            lr=lr,
            quantiles=args.quantiles,
            quantile_weights=args.quantile_weights,
            # Fixed weights for this focused sweep
            weight_nse=0.25,
            weight_quantile=0.1,
            flow_emphasis=0.25,
            consistency_weight=2.5,
            weight_pbias=0.04,
            residual_bias_weight=0.0,
            bias_shift_alpha=bias_shift_alpha,
            bias_shift_pbias_target=5.0,
            bias_shift_qmin=0.05,
            bias_shift_qmax=0.90,
            bias_shift_weight_power=-0.3,
            bias_shift_strategy="scaled",
            weight_kge=0.0,
            weight_kge_final=None,
            use_amp=not args.no_amp,
            use_compile=False,
            augment=not args.no_augment,
            use_ranger=not args.no_ranger,
            output_prefix=trial_prefix,
            seed=args.seed,
            model_arch="hydra_v1", # Explicitly use v1
            patch_size=patch_size,
        )
        metrics = _read_metrics(trial_prefix)
        trial.set_user_attr("metrics", metrics)
        return _compute_objectives(metrics, args.enforce_baseline)
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned(f"Trial failed: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hydra v1 HPO Sweep")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--study-name", type=str, required=True)
    parser.add_argument("--storage", type=str, required=True)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-days", type=int, default=4018)
    parser.add_argument("--val-days", type=int, default=365)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--output-prefix", type=str, default="hydra_v1_sweep")
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--quantile-weights", type=float, nargs="+", default=[0.25, 0.5, 0.25])
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-ranger", action="store_true")
    parser.add_argument("--enforce-baseline", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Ensure storage directory exists
    if args.storage.startswith("sqlite:///"):
        db_path = args.storage.replace("sqlite:///", "")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        directions=["minimize", "maximize", "maximize", "minimize"], # RMSE, NSE, KGE, PBIAS
        sampler=TPESampler(seed=args.seed),
        pruner=pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    print(f"Starting HPO sweep for Hydra v1 with {args.n_trials} trials...")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    
    print("Best trials (Pareto front):")
    for t in study.best_trials:
        print(f"  Trial {t.number}: {t.values} - Params: {t.params}")


if __name__ == "__main__":
    main()

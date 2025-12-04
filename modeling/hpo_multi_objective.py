#!/usr/bin/env python3
"""Multi-objective Optuna sweep for the Hydra transformer (Watauga site).

This runner minimises corrected RMSE and absolute percent bias while
simultaneously maximising NSE and KGE. Trials are pruned with ASHA to keep the
runtime manageable on a single M1/M2 laptop. Example call:

    PYTHONPATH=. python modeling/hpo_multi_objective.py \
        --data data/clean/modeling/hourly_training_03479000_20100101_20201231.parquet \
        --study-name watauga_multiobj \
        --storage sqlite:///local_only/hpo/watauga_multiobj.db \
        --n-trials 18 \
        --epochs 25 \
        --batch-size 1024 \
        --prior-trials local_only/archive/baselines/batchsweep_bs1024_lr8e4_shift01_trials.json

The best (Pareto-dominant) trial can be retrieved with ``optuna-dashboard`` or
``optuna.study.Study.pareto_front_trials``.
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

PARAM_DISTRIBUTIONS = {
    "seq_len": optuna.distributions.CategoricalDistribution([168, 192]),
    "d_model": optuna.distributions.CategoricalDistribution([112, 128, 160]),
    "num_layers": optuna.distributions.IntDistribution(3, 5),
    "num_heads": optuna.distributions.CategoricalDistribution([4, 5]),
    "conv_depth": optuna.distributions.IntDistribution(3, 4),
    "dropout": optuna.distributions.FloatDistribution(0.05, 0.15),
    "batch_size": optuna.distributions.CategoricalDistribution([768, 1024, 1280]),
    "lr": optuna.distributions.FloatDistribution(6e-4, 1e-3, log=True),
    "weight_nse": optuna.distributions.FloatDistribution(0.15, 0.35),
    "weight_quantile": optuna.distributions.FloatDistribution(0.05, 0.2),
    "flow_emphasis": optuna.distributions.FloatDistribution(0.15, 0.35),
    "consistency_weight": optuna.distributions.FloatDistribution(1.5, 3.5),
    "weight_pbias": optuna.distributions.FloatDistribution(0.02, 0.06),
    "bias_shift_alpha": optuna.distributions.FloatDistribution(0.05, 0.15),
    "bias_shift_qmin": optuna.distributions.FloatDistribution(0.02, 0.08),
    "bias_shift_qmax": optuna.distributions.FloatDistribution(0.85, 0.95),
    "bias_shift_weight_power": optuna.distributions.FloatDistribution(-0.5, -0.1),
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


def _load_prior_trials(path: str) -> List[optuna.trial.FrozenTrial]:
    prior_path = Path(path)
    if not prior_path.exists():
        raise FileNotFoundError(prior_path)
    with open(prior_path) as fh:
        raw = json.load(fh)
    trials = []
    for entry in raw:
        metrics = entry.get("metrics") or {}
        corrected = metrics.get("corrected", {})
        if not corrected:
            continue
        rmse = float(corrected.get("rmse", math.inf))
        nse = float(corrected.get("nse", float("nan")))
        kge = float(corrected.get("kge", float("nan")))
        pbias = float(corrected.get("pbias", float("nan")))
        values = [rmse, -nse, -kge, abs(pbias)]
        params = {}
        distributions = {}
        for key, value in (entry.get("params") or {}).items():
            if key not in PARAM_DISTRIBUTIONS:
                continue
            params[key] = value
            distributions[key] = copy.deepcopy(PARAM_DISTRIBUTIONS[key])
        trial = optuna.trial.create_trial(
            params=params,
            distributions=distributions,
            values=values,
            user_attrs={"metrics": metrics},
            state=optuna.trial.TrialState.COMPLETE,
        )
        trials.append(trial)
    return trials


def _suggest_hyperparams(trial: optuna.Trial) -> Dict[str, float | int]:
    params: Dict[str, float | int] = {}
    params["seq_len"] = trial.suggest_categorical("seq_len", [168, 192])
    params["d_model"] = trial.suggest_categorical("d_model", [112, 128, 160])
    params["num_layers"] = trial.suggest_int("num_layers", 3, 5)
    params["num_heads"] = trial.suggest_categorical("num_heads", [4, 5])
    if params["d_model"] % params["num_heads"] != 0:
        raise optuna.TrialPruned("num_heads must divide d_model")
    params["conv_depth"] = trial.suggest_int("conv_depth", 3, 4)
    params["dropout"] = trial.suggest_float("dropout", 0.05, 0.15)
    params["batch_size"] = trial.suggest_categorical("batch_size", [768, 1024, 1280])
    params["lr"] = trial.suggest_float("lr", 6e-4, 1e-3, log=True)
    params["weight_nse"] = trial.suggest_float("weight_nse", 0.15, 0.35)
    params["weight_quantile"] = trial.suggest_float("weight_quantile", 0.05, 0.2)
    params["flow_emphasis"] = trial.suggest_float("flow_emphasis", 0.15, 0.35)
    params["consistency_weight"] = trial.suggest_float("consistency_weight", 1.5, 3.5)
    params["weight_pbias"] = trial.suggest_float("weight_pbias", 0.02, 0.06)
    params["bias_shift_alpha"] = trial.suggest_float("bias_shift_alpha", 0.05, 0.15)
    params["bias_shift_qmin"] = trial.suggest_float("bias_shift_qmin", 0.02, 0.08)
    params["bias_shift_qmax"] = trial.suggest_float("bias_shift_qmax", 0.85, 0.95)
    params["bias_shift_weight_power"] = trial.suggest_float("bias_shift_weight_power", -0.5, -0.1)
    return params


def _compute_objectives(metrics: Dict[str, Dict[str, float]], enforce: bool) -> Sequence[float]:
    corrected = metrics["corrected"]
    rmse = float(corrected["rmse"])
    nse = float(corrected["nse"])
    kge = float(corrected["kge"])
    pbias = float(corrected["pbias"])
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
    sampled = _suggest_hyperparams(trial)
    trial_prefix = f"{args.output_prefix}_trial{trial.number}"
    train_eval(
        data_path=args.data,
        seq_len=sampled["seq_len"],
        epochs=args.epochs,
        batch_size=sampled["batch_size"],
        train_days=args.train_days,
        val_days=args.val_days,
        patience=args.patience,
        d_model=sampled["d_model"],
        num_heads=sampled["num_heads"],
        num_layers=sampled["num_layers"],
        conv_depth=sampled["conv_depth"],
        dropout=sampled["dropout"],
        lr=sampled["lr"],
        quantiles=args.quantiles,
        quantile_weights=args.quantile_weights,
        weight_nse=sampled["weight_nse"],
        weight_quantile=sampled["weight_quantile"],
        flow_emphasis=sampled["flow_emphasis"],
        consistency_weight=sampled["consistency_weight"],
        weight_pbias=sampled["weight_pbias"],
        weight_pbias_final=None,
        residual_bias_weight=0.0,
        bias_shift_alpha=sampled["bias_shift_alpha"],
        bias_shift_pbias_target=5.0,
        bias_shift_qmin=sampled["bias_shift_qmin"],
        bias_shift_qmax=sampled["bias_shift_qmax"],
        bias_shift_weight_power=sampled["bias_shift_weight_power"],
        bias_shift_strategy="scaled",
        weight_kge=args.weight_kge,
        weight_kge_final=args.weight_kge_final,
        use_amp=not args.no_amp,
        use_compile=False,
        augment=not args.no_augment,
        use_ranger=not args.no_ranger,
        output_prefix=trial_prefix,
        seed=args.seed,
    )
    metrics = _read_metrics(trial_prefix)
    trial.set_user_attr("metrics", metrics)
    for key, value in sampled.items():
        trial.set_user_attr(key, value)
    return _compute_objectives(metrics, args.enforce_baseline)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-objective Optuna sweep for Hydra transformer")
    parser.add_argument("--data", required=True)
    parser.add_argument("--study-name", default="watauga_multiobj")
    parser.add_argument("--storage", default=None, help="Optuna storage URI (e.g., sqlite:///path/to.db)")
    parser.add_argument("--output-prefix", default="watauga_multiobj")
    parser.add_argument("--prior-trials", default=None, help="Optional JSON containing previous trial summaries to warm start")
    parser.add_argument("--n-trials", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--train-days", type=int, default=3287)
    parser.add_argument("--val-days", type=int, default=365)
    parser.add_argument("--weight-kge", type=float, default=0.05)
    parser.add_argument("--weight-kge-final", type=float, default=None)
    parser.add_argument("--quantiles", default="0.05,0.5,0.95")
    parser.add_argument("--quantile-weights", default="0.5,1.0,0.5")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-ranger", action="store_true")
    parser.add_argument("--enforce-baseline", action="store_true", help="Penalise trials that fail to beat baseline NSE/KGE/PBIAS")
    return parser.parse_args()


def _make_study(args: argparse.Namespace) -> optuna.Study:
    sampler = TPESampler(seed=args.seed, multivariate=True, group=True)
    pruner = pruners.SuccessiveHalvingPruner(min_resource=5, reduction_factor=3)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        directions=["minimize", "minimize", "minimize", "minimize"],
        sampler=sampler,
        pruner=pruner,
    )
    if args.prior_trials:
        for trial in _load_prior_trials(args.prior_trials):
            study.add_trial(trial)
    return study


def main() -> None:
    args = parse_args()
    args.quantiles = [float(x) for x in args.quantiles.split(",") if x.strip()]
    args.quantile_weights = [float(x) for x in args.quantile_weights.split(",") if x.strip()]
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    study = _make_study(args)
    study.optimize(lambda t: objective(t, args), n_trials=args.n_trials)

    pareto = study.best_trials
    out_payload = []
    for trial in pareto:
        out_payload.append(
            {
                "number": trial.number,
                "values": trial.values,
                "params": trial.params,
                "metrics": trial.user_attrs.get("metrics"),
            }
        )
    out_path = Path(f"{args.output_prefix}_pareto.json")
    with open(out_path, "w") as fh:
        json.dump(out_payload, fh, indent=2)
    print(f"Saved Pareto front ({len(pareto)} trials) to {out_path}")


if __name__ == "__main__":
    main()

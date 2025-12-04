#!/usr/bin/env python3
"""End-to-end pipeline for a single site (data → train → plots)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
MAKE = "make"
MIN_IMPROVEMENT = 5.0  # percentage improvement threshold
DEFAULT_HPO_TRIALS = 8


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT, env=full_env)


def main() -> None:
    parser = argparse.ArgumentParser("Site pipeline runner")
    parser.add_argument("site_id", help="USGS site identifier")
    parser.add_argument("name", help="Readable station name")
    parser.add_argument("output_prefix", default="hydra_site", nargs="?")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--hpo-trials", type=int, default=DEFAULT_HPO_TRIALS)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-hpo", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--start", default="2010-01-01", help="Start date for dataset build (YYYY-MM-DD)")
    parser.add_argument("--end", default="2020-12-31", help="End date for dataset build (YYYY-MM-DD)")
    parser.add_argument("--train-start", default="2010-01-01", help="Training window start")
    parser.add_argument("--train-end", default="2018-12-31", help="Training window end")
    parser.add_argument("--val-start", default="2019-01-01", help="Validation window start")
    parser.add_argument("--val-end", default="2019-12-31", help="Validation window end")
    parser.add_argument("--nwm-version", default="v2", choices=["v2", "v3"], help="Retrospective archive to use")
    args = parser.parse_args()

    start_tag = args.start.replace("-", "")
    end_tag = args.end.replace("-", "")
    default_tag = f"{args.site_id}_{start_tag}_{end_tag}"
    data_path = Path(args.data_path) if args.data_path else Path("data/clean/modeling") / f"hourly_training_{default_tag}.parquet"

    base_env = {"PYTHONPATH": str(ROOT)}

    if not args.skip_build:
        run_cmd([
            PYTHON,
            "modeling/build_training_dataset.py",
            "--raw-dir",
            "data/raw",
            "--out-dir",
            "data/clean/modeling",
            "--start",
            args.start,
            "--end",
            args.end,
            "--nwm-version",
            args.nwm_version,
            "--sites",
            args.site_id,
        ], env=base_env)

    metrics_path = Path("data/clean/modeling") / f"{args.output_prefix}_metrics.json"
    needs_hpo = not args.skip_hpo

    if not args.skip_train:
        train_cmd = [
            MAKE,
            "train_full",
            f"OUTPUT_PREFIX={args.output_prefix}",
            f"DATA_PATH={data_path}",
            f"TRAIN_START={args.train_start}",
            f"TRAIN_END={args.train_end}",
            f"VAL_START={args.val_start}",
            f"VAL_END={args.val_end}",
        ]
        run_cmd(train_cmd, env=base_env)

        if metrics_path.exists():
            with open(metrics_path) as fh:
                metrics = json.load(fh)
            improvement = metrics.get("rmse_improvement_pct") or 0.0
            needs_hpo = needs_hpo and (improvement < MIN_IMPROVEMENT)
            print(f"Initial improvement: {improvement:.2f}% -> {'HPO required' if needs_hpo else 'acceptable'}")
        else:
            print("Warning: metrics file missing after training; forcing HPO")
            needs_hpo = not args.skip_hpo

    if needs_hpo:
        hpo_prefix = f"hydra_hpo_{args.site_id}"
        run_cmd(
            [
                MAKE,
                "hpo",
                f"HPO_OUTPUT_PREFIX={hpo_prefix}",
                f"DATA_PATH={data_path}",
                f"HPO_TRIALS={args.hpo_trials}",
            ],
            env=base_env,
        )

        hpo_trials_path = Path(f"{hpo_prefix}_optuna_trials.json")
        if not hpo_trials_path.exists():
            raise FileNotFoundError(f"HPO trials file not found: {hpo_trials_path}")

        with open(hpo_trials_path) as fh:
            trials = json.load(fh)
        best_trial = min(trials, key=lambda t: t["value"])
        params = best_trial["params"]
        overrides = {
            "D_MODEL": params["d_model"],
            "NUM_LAYERS": params["num_layers"],
            "NUM_HEADS": params["num_heads"],
            "CONV_DEPTH": params["conv_depth"],
            "DROPOUT": params["dropout"],
            "LR": params["lr"],
            "SEQ_LEN": params["seq_len"],
        }
        retrain_cmd = [
            MAKE,
            "train_full",
            f"OUTPUT_PREFIX={args.output_prefix}",
            f"DATA_PATH={data_path}",
            f"TRAIN_START={args.train_start}",
            f"TRAIN_END={args.train_end}",
            f"VAL_START={args.val_start}",
            f"VAL_END={args.val_end}",
        ] + [f"{k}={v}" for k, v in overrides.items()]
        run_cmd(retrain_cmd, env=base_env)

    if not args.skip_plots:
        run_cmd(
            [
                MAKE,
                "plots_full",
                f"OUTPUT_PREFIX={args.output_prefix}",
                f"EVAL_CSV=data/clean/modeling/{args.output_prefix}_eval.csv",
                f"FIGURES_DIR=results/figures/{args.output_prefix}",
                f"STATION_NAME={args.name}",
            ],
            env=base_env,
        )


if __name__ == "__main__":
    main()

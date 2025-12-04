#!/usr/bin/env python3
"""Batch runner for the per-site pipeline (data → train → plots) across multiple gauges."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

sys.path.insert(0, str(ROOT))
from config.master_study_sites import MASTER_STUDY_SITES  # noqa: E402


DEFAULT_SITES = ["03479000", "04137500", "09234500"]
DEFAULT_HPO_TRIALS = 8


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", default=DEFAULT_SITES, help="USGS site IDs to process")
    parser.add_argument("--start", default="2010-01-01", help="Dataset build start")
    parser.add_argument("--end", default="2020-12-31", help="Dataset build end")
    parser.add_argument("--train-start", default="2010-01-01", help="Training window start")
    parser.add_argument("--train-end", default="2018-12-31", help="Training window end")
    parser.add_argument("--val-start", default="2019-01-01", help="Validation window start")
    parser.add_argument("--val-end", default="2019-12-31", help="Validation window end")
    parser.add_argument("--nwm-version", default="v2", choices=["v2", "v3"], help="Retrospective archive to use")
    parser.add_argument("--prefix-suffix", default="hydra_v2", help="Appended to each site ID for OUTPUT_PREFIX")
    parser.add_argument("--hpo-trials", type=int, default=DEFAULT_HPO_TRIALS, help="Optuna trials per site")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-hpo", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    for site_id in args.sites:
        if site_id not in MASTER_STUDY_SITES:
            raise KeyError(f"Site {site_id} missing from MASTER_STUDY_SITES.")
        site_info = MASTER_STUDY_SITES[site_id]
        name = site_info["name"]
        output_prefix = f"{site_id}_{args.prefix_suffix}"
        cmd = [
            PYTHON,
            "scripts/run_site_pipeline.py",
            site_id,
            name,
            output_prefix,
            "--start",
            args.start,
            "--end",
            args.end,
            "--train-start",
            args.train_start,
            "--train-end",
            args.train_end,
            "--val-start",
            args.val_start,
            "--val-end",
            args.val_end,
            "--nwm-version",
            args.nwm_version,
            "--hpo-trials",
            str(args.hpo_trials),
        ]
        if args.skip_build:
            cmd.append("--skip-build")
        if args.skip_train:
            cmd.append("--skip-train")
        if args.skip_hpo:
            cmd.append("--skip-hpo")
        if args.skip_plots:
            cmd.append("--skip-plots")
        print(f"\n=== Processing site {site_id} ({name}) ===")
        run_cmd(cmd)


if __name__ == "__main__":
    main()

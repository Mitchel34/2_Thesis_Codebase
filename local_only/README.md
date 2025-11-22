# Local-Only Storage

This directory is intentionally ignored by Git. Move any large or sensitive artefacts here:

- `archive/`: sweep specs, serialized checkpoints, evaluation logs
- `artifacts/`: generated figures, tables, and post-processed metric bundles
- `results/`: intermediate evaluation outputs and notebooks
- `figs/`: rendered publication figures
- `logs/`: Hydra runs, tensorboard traces, and training logs
- `experiments/`: hyperparameter sweeps (Optuna / Hydra exports)
- data snapshots under `local_only/data/` if you prefer keeping the canonical `data/` path empty

Recreate the expected structure by running `scripts/setup_local_storage.sh` (added in this overhaul) or by manually creating the folders above. All scripts should treat `local_only` as the canonical sink for heavy products that should remain on your workstation.

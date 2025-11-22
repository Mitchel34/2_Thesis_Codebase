# Watauga Streamflow Residual Corrections

This repository holds the data acquisition scripts, modeling code, and thesis documentation for correcting National Water Model (NWM) streamflow guidance with a hybrid GRU–transformer residual learner. The codebase is trimmed to the pieces you need to regenerate the dataset and models while keeping large artefacts on your own machine.

## Repository Layout

- `config/` – site definitions used by every acquisition script (`master_study_sites.py`).
- `configs/` – JSON split specifications (train/val/test, rolling windows).
- `data_acquisition_scripts/` – standalone collectors for USGS, NWM v3, ERA5, and NLCD (+ helper tools).
- `modeling/` – dataset builder, transformer training loop, plotting utilities, and HPO entry-points.
- `scripts/` – orchestration helpers (`run_acquisition_pipeline.sh`, `run_site_pipeline.py`, `setup_local_storage.sh`).
- `docs/` – thesis draft plus run logs such as `MODEL_RUN_2010_2022.md`.
- `local_only/` – ignored directory that now houses archives, logs, figures, and other large artefacts.
- `.gitignore`, `Makefile`, `requirements.txt` – repo hygiene, training shortcuts, and dependency lock.

## Setup

1. **Python environment (3.11 recommended).**

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Initialize the local-only storage tree.** This keeps raw data, logs, checkpoints, and plots off of version control.

   ```bash
   bash scripts/setup_local_storage.sh
   ```

   The script creates `local_only/{archive,artifacts,results,figs,logs,experiments,data/...}` and you can symlink or copy any long-lived artefacts there.

3. **Set `PYTHONPATH`** to the repo root (many scripts do this automatically, but exporting it avoids surprises):

   ```bash
   export PYTHONPATH="$(pwd)"
   ```

4. **(Optional but recommended) Install pre-commit hooks.** They format code with Black/Ruff and
  catch trailing whitespace before you commit:

  ```bash
  pre-commit install
  ```

## Credentials & External Access

| Source | Requirements |
| --- | --- |
| **ERA5 / ERA5-Land** | Create `~/.cdsapirc` with a Copernicus Climate Data Store API key. Example:<br>`url: https://cds.climate.copernicus.eu/api/v2`<br>`key: <uid>:<api-key>`<br>`verify: 1` |
| **NWM Retrospective / Operational** | Public NOAA S3 buckets (`noaa-nwm-retrospective-3-0-pds`, `noaa-nwm-pds`) can be accessed anonymously. If you rely on a mirrored archive, set `NWM_ARCHIVE_BASE_URL` before running `scripts/run_acquisition_pipeline.sh`. Optional AWS credentials (`AWS_PROFILE`, `AWS_ACCESS_KEY_ID`, etc.) are only needed for private mirrors. |
| **USGS NWIS** | HTTPS requests only; be mindful of rate limits. No API key required. |
| **NLCD 2021** | MRLC and USGS APIs are open. If you switch to Google Earth Engine outputs, set `GOOGLE_APPLICATION_CREDENTIALS` to your service account JSON. |

## Data Acquisition

You can run each collector individually or use the orchestration script that sequences USGS → NWM → ERA5:

```bash
bash scripts/run_acquisition_pipeline.sh
```

Each script exposes CLI arguments so you can narrow the time range or site list. Common examples:

```bash
# USGS hourly discharge
python data_acquisition_scripts/usgs.py \
  --sites 03479000 \
  --start-date 2010-01-01 \
  --end-date 2023-12-31 \
  --out-dir data/raw/usgs

# NWM v3 auto mode (retrospective + operational hand-off)
python data_acquisition_scripts/nwm.py \
  --mode v3_auto \
  --start-date 2010-01-01 \
  --end-date 2023-12-31 \
  --out-dir data/raw/nwm_v3 \
  --max-workers 6 \
  --resume

# ERA5 + ERA5-Land atmospheric forcings
python data_acquisition_scripts/era5.py \
  --sites 03479000 \
  --years 2010 2023 \
  --cadence hourly \
  --out-dir data/raw/era5

# NLCD 2021 static descriptors
python data_acquisition_scripts/land_use.py \
  --sites 03479000 \
  --out-dir data/raw/land_use
```

Logs will land in `logs/` (ignored by Git); move any long-lived CSV/Parquet outputs into `local_only/` once you finish a run.

## Building Datasets & Training Models

1. **Assemble the modeling parquet** (creates residual targets and splits):

   ```bash
   python modeling/build_training_dataset.py \
     --raw-dir data/raw \
     --out-dir data/clean/modeling \
     --start 2010-01-01 \
     --end 2022-12-31 \
     --sites 03479000
   ```

2. **Train the hybrid transformer via `make`:**

   ```bash
   make train_full OUTPUT_PREFIX=watauga_hydra DATA_PATH=data/clean/modeling/hourly_training_2010-01-01_2022-12-31.parquet
   ```

   Results are written to `data/clean/modeling/watauga_hydra_eval.csv` plus accompanying NumPy arrays. Copy the evaluation bundle to `local_only/results/` after inspection.

3. **Full site pipeline (data → train → plots → optional HPO):**

   ```bash
   python scripts/run_site_pipeline.py 03479000 "Watauga River, NC" watauga_hydra \
     --hpo-trials 10
   ```

   This command reuses the acquisition outputs already stored under `data/` and kicks off Optuna-based hyperparameter search when the baseline improvement is <5 %.

## Local-Only Storage Policy

- Keep any of the following inside `local_only/`: `archive/` sweeps, `artifacts/` (figures, tables), `results/` evaluation CSVs, `figs/`, `logs/`, and Hydra/Optuna exports. Everything in that directory is ignored except `README.md` and `.gitkeep`.
- If you need to share artefacts, publish them through an external channel (S3, figshare, etc.) and link to them from the docs instead of pushing binaries to Git.

## Supporting Documents

- `docs/MODEL_RUN_2010_2022.md` – full provenance for the baseline run (2010–2022) with metrics.
- `docs/PROJECT_PLAN.md` – scope of thesis milestones and upcoming experiments.
- `docs/wrr_draft.tex` – manuscript submitted to *Water Resources Research*.

Use these notes as the single source of truth for regenerating data and models; any future experiments should add their own local-only bundle plus a short write-up under `docs/`.

## Contributing

Contributions are welcome! Please read `CONTRIBUTING.md` for environment setup, testing, and
documentation expectations before opening a pull request. Running `pre-commit install` keeps your
commits aligned with the automated formatting and linting rules.

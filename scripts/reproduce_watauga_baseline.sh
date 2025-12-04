#!/usr/bin/env bash
#
# Recreate the baseline Watauga workflow end-to-end:
#   1. Ensure a Python virtual environment exists and install requirements.
#   2. Prepare the local storage tree (keeps artefacts out of Git).
#   3. Re-run USGS, NWM (v2 + v3), and ERA5 acquisitions for 2010-01-01..2022-12-31.
#   4. Build the hourly training parquet and train the Hydra model via `make train_full`.
# The script is intentionally idempotent: if an artefact already exists it is reused.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SITE_ID="03479000"
SITE_NAME="Watauga River, NC"
START_DATE="2010-01-01"
END_DATE="2022-12-31"
VENV_DIR=".venv"
PYTHON_BIN="${PYTHON:-python3}"
OUTPUT_PREFIX="watauga_baseline"
RAW_DIR="data/raw"
CLEAN_DIR="data/clean/modeling"
START_COMPACT="${START_DATE//-/}"
END_COMPACT="${END_DATE//-/}"
PARQUET_PATH="${CLEAN_DIR}/hourly_training_${SITE_ID}_${START_COMPACT}_${END_COMPACT}.parquet"
USGS_SENTINEL="${RAW_DIR}/usgs/${SITE_ID}/${SITE_ID}_consolidated.csv"
NWM_V2_SENTINEL="${RAW_DIR}/nwm_v3/retrospective/nwm_v2p1_hourly_20100101_20201231.csv"
NWM_V3_SENTINEL="${RAW_DIR}/nwm_v3/retrospective/nwm_v3_hourly_20210101_20221231.csv"
ERA5_SENTINEL="${RAW_DIR}/era5/${SITE_ID}"

TRAIN_YEARS=(2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020)
TEST_YEARS=(2021 2022)

export PYTHONPATH="${REPO_ROOT}"

log() {
  printf "\n[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

ensure_virtualenv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    log "Creating virtual environment in ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  else
    log "Virtual environment already exists – reusing ${VENV_DIR}"
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip >/dev/null
  log "Installing Python requirements"
  python -m pip install -r requirements.txt
}

run_setup() {
  log "Ensuring local storage tree exists"
  bash scripts/setup_local_storage.sh
}

acquire_usgs() {
  if [[ -f "${USGS_SENTINEL}" ]]; then
    log "USGS data already present (${USGS_SENTINEL}); skipping download"
    return
  fi
  log "Downloading USGS discharge for ${SITE_NAME}"
  python data_acquisition_scripts/usgs.py \
    --sites "${SITE_ID}" \
    --start-date "${START_DATE}" \
    --end-date "${END_DATE}" \
    --out-dir "${RAW_DIR}/usgs/${SITE_ID}"
}

acquire_nwm() {
  if [[ -f "${NWM_V2_SENTINEL}" ]]; then
    log "NWM v2.1 retrospective CSV found; skipping v2 fetch"
  else
    log "Fetching NWM v2.1 retrospective data (2010-2020)"
    python data_acquisition_scripts/nwm.py \
      --mode retrospective \
      --sites "${SITE_ID}" \
      --start-date "2010-01-01" \
      --end-date "2020-12-31" \
      --out-dir "${RAW_DIR}/nwm_v3"
  fi

  if [[ -f "${NWM_V3_SENTINEL}" ]]; then
    log "NWM v3 retrospective CSV found; skipping v3 fetch"
  else
    log "Fetching NWM v3 retrospective data (2021-2022)"
    python data_acquisition_scripts/nwm.py \
      --mode retrospective_v3 \
      --sites "${SITE_ID}" \
      --start-date "2021-01-01" \
      --end-date "2022-12-31" \
      --out-dir "${RAW_DIR}/nwm_v3" \
      --resume
  fi
}

acquire_era5() {
  if [[ -d "${ERA5_SENTINEL}" ]]; then
    log "ERA5 directory ${ERA5_SENTINEL} already exists; skipping fetch"
    return
  fi
  log "Requesting ERA5 + ERA5-Land forcings (hourly cadence)"
  python data_acquisition_scripts/era5.py \
    --site "${SITE_ID}" \
    --years-train "${TRAIN_YEARS[@]}" \
    --years-test "${TEST_YEARS[@]}" \
    --cadence hourly \
    --out-dir "${RAW_DIR}/era5/${SITE_ID}"
}

build_dataset() {
  mkdir -p "${CLEAN_DIR}"
  if [[ -f "${PARQUET_PATH}" ]]; then
    log "Modeling parquet already exists (${PARQUET_PATH}); skipping rebuild"
    return
  fi
  log "Building hourly training dataset (${START_DATE}..${END_DATE})"
  python modeling/build_training_dataset.py \
    --raw-dir "${RAW_DIR}" \
    --out-dir "${CLEAN_DIR}" \
    --start "${START_DATE}" \
    --end "${END_DATE}" \
    --sites "${SITE_ID}"
}

train_model() {
  log "Training Hydra model via make (OUTPUT_PREFIX=${OUTPUT_PREFIX})"
  make train_full \
    OUTPUT_PREFIX="${OUTPUT_PREFIX}" \
    DATA_PATH="${PARQUET_PATH}"
}

main() {
  ensure_virtualenv
  run_setup
  acquire_usgs
  acquire_nwm
  acquire_era5
  build_dataset
  train_model
  log "All steps complete – review results under data/clean/modeling and local_only/"
}

main "$@"

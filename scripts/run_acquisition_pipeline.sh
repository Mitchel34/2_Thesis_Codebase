#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
ERA5_PID_FILE="$LOG_DIR/era5.pid"
USGS_LOG="$LOG_DIR/usgs.log"
NWM_LOG="$LOG_DIR/nwm.log"
PIPELINE_LOG="$LOG_DIR/pipeline.log"
ARCHIVE_BASE_URL="${NWM_ARCHIVE_BASE_URL:-}"
ARCHIVE_FLAG=()
if [[ -n "$ARCHIVE_BASE_URL" ]]; then
  ARCHIVE_FLAG=(--archive-base-url "$ARCHIVE_BASE_URL")
  echo "$(date '+%Y-%m-%d %H:%M:%S') using archive fallback: $ARCHIVE_BASE_URL" | tee -a "$PIPELINE_LOG"
fi

# Wait for prior ERA5 job if pid file exists and process alive
if [[ -f "$ERA5_PID_FILE" ]]; then
  if kill -0 "$(cat "$ERA5_PID_FILE")" 2>/dev/null; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') waiting for ERA5 PID $(cat "$ERA5_PID_FILE")" | tee -a "$PIPELINE_LOG"
    while kill -0 "$(cat "$ERA5_PID_FILE")" 2>/dev/null; do
      sleep 60
    done
    echo "$(date '+%Y-%m-%d %H:%M:%S') ERA5 job finished" | tee -a "$PIPELINE_LOG"
  fi
fi

# Run USGS acquisition
{
  echo "$(date '+%Y-%m-%d %H:%M:%S') starting USGS acquisition" >> "$PIPELINE_LOG"
  python3.11 data_acquisition_scripts/usgs.py \
    --out-dir data/raw/usgs \
    --sites 03479000 \
    --start-date 2020-01-01 \
    --end-date 2023-12-31
} >> "$USGS_LOG" 2>&1

echo "$(date '+%Y-%m-%d %H:%M:%S') finished USGS acquisition" | tee -a "$PIPELINE_LOG"

# Run NWM acquisition (v3 auto)
{
  echo "$(date '+%Y-%m-%d %H:%M:%S') starting NWM acquisition" >> "$PIPELINE_LOG"
  python3.11 data_acquisition_scripts/nwm.py \
    --mode v3_auto \
    --start-date 2020-01-01 \
    --end-date 2023-12-31 \
    --out-dir data/raw/nwm_v3 \
    --max-workers 4 \
    --checkpoint-every 200 \
    --resume \
    --concurrency process \
    "${ARCHIVE_FLAG[@]}"
} >> "$NWM_LOG" 2>&1

echo "$(date '+%Y-%m-%d %H:%M:%S') finished NWM acquisition" | tee -a "$PIPELINE_LOG"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_ONLY_DIR="$ROOT_DIR/local_only"

mkdir -p "$LOCAL_ONLY_DIR/archive" \
         "$LOCAL_ONLY_DIR/artifacts" \
         "$LOCAL_ONLY_DIR/results" \
         "$LOCAL_ONLY_DIR/figs" \
         "$LOCAL_ONLY_DIR/logs" \
         "$LOCAL_ONLY_DIR/experiments" \
         "$LOCAL_ONLY_DIR/data/raw" \
         "$LOCAL_ONLY_DIR/data/clean" \
         "$LOCAL_ONLY_DIR/data/external"

printf "Local-only storage initialized under %s\n" "$LOCAL_ONLY_DIR"

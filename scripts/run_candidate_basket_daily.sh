#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
LOCK_FILE="$LOG_DIR/run_candidate_basket_daily.lock"

mkdir -p "$LOG_DIR"

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "[$(date --iso-8601=seconds)] candidate basket job is already running"
    exit 0
  fi
fi

cd "$ROOT_DIR"

run_step() {
  echo "[$(date --iso-8601=seconds)] RUN $*"
  "$@"
}

run_step .venv/bin/python scripts/run_tick_symbol_generalization.py --config configs/tick/multi_asset_365d/generalization_daily.yaml
run_step .venv/bin/python scripts/run_tick_symbol_generalization.py --config configs/tick/multi_asset_365d/singles/doge_daily.yaml
run_step .venv/bin/python scripts/run_tick_symbol_generalization.py --config configs/tick/multi_asset_365d/singles/avax_daily.yaml
run_step .venv/bin/python scripts/run_tick_candidate_basket_tracker.py --config configs/tick/multi_asset_365d/candidate_basket_tracker.yaml

echo "[$(date --iso-8601=seconds)] candidate basket job completed"

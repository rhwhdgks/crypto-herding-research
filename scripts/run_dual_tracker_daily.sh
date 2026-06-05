#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
LOCK_FILE="$LOG_DIR/run_dual_tracker_daily.lock"

mkdir -p "$LOG_DIR"

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "[$(date --iso-8601=seconds)] daily dual tracker job is already running"
    exit 0
  fi
fi

cd "$ROOT_DIR"

run_step() {
  echo "[$(date --iso-8601=seconds)] RUN $*"
  "$@"
}

run_step .venv/bin/python scripts/run_tick_archive_backfill.py --config configs/tick/xrp_5y/backfill.yaml
run_step .venv/bin/python scripts/run_tick_short_horizon_study.py --config configs/tick/xrp_5y/short_horizon.yaml
run_step .venv/bin/python scripts/run_tick_cost_sanity.py --config configs/tick/xrp_5y/cost_sanity.yaml
run_step .venv/bin/python scripts/run_tick_subset_candidates.py --config configs/tick/xrp_5y/subset_candidates.yaml
run_step .venv/bin/python scripts/run_tick_candidate_paper_sim.py --config configs/tick/xrp_5y/candidate_paper_sim.yaml
run_step .venv/bin/python scripts/run_tick_candidate_refinement.py --config configs/tick/xrp_5y/candidate_refinement.yaml
run_step .venv/bin/python scripts/run_tick_overlap_core_analysis.py --config configs/tick/xrp_5y/overlap_core.yaml
run_step .venv/bin/python scripts/run_tick_overlap_core_regime.py --config configs/tick/xrp_5y/overlap_core_regime.yaml
run_step .venv/bin/python scripts/run_tick_overlap_core_variants.py --config configs/tick/xrp_5y/overlap_core_variants.yaml
run_step .venv/bin/python scripts/run_tick_dual_tracker.py --config configs/tick/xrp_5y/dual_tracker.yaml

echo "[$(date --iso-8601=seconds)] daily dual tracker job completed"

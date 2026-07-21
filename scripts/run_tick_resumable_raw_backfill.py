from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from tick_resumable_backfill import load_backfill_state, run_resumable_raw_backfill
from utils import load_config, setup_logging


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="월별 checkpoint를 사용하는 raw tick 백필")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "tick" / "semantic_validation" / "raw_2y_build.yaml"),
    )
    parser.add_argument("--status", action="store_true", help="실행하지 않고 현재 진행 상태만 출력")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    output_dir = PROJECT_ROOT / config["output"]["base_dir"]
    if args.status:
        print(json.dumps(load_backfill_state(output_dir), indent=2, ensure_ascii=False))
        return

    def log_progress(state: dict) -> None:
        LOGGER.info(
            "raw backfill progress: %d/%d (%.2f%%), last=%s %s",
            state["completed_jobs"],
            state["total_jobs"],
            state["progress_share"] * 100.0,
            state["last_completed_symbol"],
            state["last_completed_month"],
        )

    final_state = run_resumable_raw_backfill(config, PROJECT_ROOT, log_progress)
    LOGGER.info("raw backfill complete: %s", json.dumps(final_state, ensure_ascii=False))


if __name__ == "__main__":
    main()

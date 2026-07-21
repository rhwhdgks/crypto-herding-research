"""Compatibility entry point for corrected complete-basket execution validation."""
from __future__ import annotations

from pathlib import Path

from corrected_candidate import run_corrected_candidate_validation
from utils import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs/tick/corrected_candidate/validation.yaml"


def main() -> int:
    config = load_config(CONFIG)
    output = PROJECT_ROOT / config["output"]["base_dir"]
    result = run_corrected_candidate_validation(config, output)
    execution = result["execution"]
    if result["selected_candidate_id"] is None:
        print("no evaluable OOS candidate; execution was not run")
    else:
        print(
            "complete-basket execution validation complete: "
            f"trades={execution['trade_count']}, terminal={execution['terminal_return']:.6f}, "
            f"max_drawdown={execution['max_drawdown']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

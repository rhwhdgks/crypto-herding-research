"""Compatibility entry point for the corrected selection-aware permutation pipeline."""
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
    print(
        "selection-aware permutation complete: "
        f"p={result['selection_aware_p_value']:.6f}, candidate={result['selected_candidate_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

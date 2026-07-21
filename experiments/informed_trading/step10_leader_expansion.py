"""Compatibility wrapper for the fixed schema-v2 candidate-family validation."""
from __future__ import annotations

from pathlib import Path

from corrected_candidate import run_corrected_candidate_validation
from utils import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs/tick/corrected_candidate/validation.yaml"


def main() -> int:
    config = load_config(CONFIG)
    run_corrected_candidate_validation(config, PROJECT_ROOT / config["output"]["base_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

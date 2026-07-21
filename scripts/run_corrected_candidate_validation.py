from __future__ import annotations

import argparse
from pathlib import Path

from corrected_candidate import run_corrected_candidate_validation
from utils import load_config, save_config_snapshot, save_input_manifest, save_provenance_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run schema-v2 selection-aware candidate validation.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output = PROJECT_ROOT / config["output"]["base_dir"]
    result = run_corrected_candidate_validation(config, output)
    save_config_snapshot(config, output / "config_snapshot.yaml")
    input_manifest = save_input_manifest(
        [PROJECT_ROOT / config["input"]["micro_frame_path"], PROJECT_ROOT / config["input"]["futures_state_path"]],
        output / "input_manifest.json",
    )
    save_provenance_manifest(
        config,
        output / "provenance.json",
        schema_version=2,
        pipeline_version="tick-semantics-v2",
        statistical_method="train-fitted-threshold; selection-aware-max-stat; complete-basket-compound-execution",
        input_manifest_path=input_manifest,
        random_seed=int(config["permutation"]["seed"]),
    )
    print(f"corrected candidate validation complete: {result['selected_candidate_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

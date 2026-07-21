from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from tick_short_horizon import prepare_micro_herding_frame, rebuild_bucket_schema_v2_from_run_counts
from utils import load_config, save_config_snapshot, save_dataframe, save_provenance_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explicitly rebuild schema-v2 tick features from archived tick/run counts."
    )
    parser.add_argument("--legacy-bucket-features", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source = args.legacy_bucket_features
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    output = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    config = load_config(args.config)
    legacy = pd.read_csv(source)
    bucket_v2 = rebuild_bucket_schema_v2_from_run_counts(legacy)
    micro_v2 = prepare_micro_herding_frame(bucket_v2, config)
    interval = int(bucket_v2["interval_minutes"].iloc[0])
    intermediate = output / "intermediate"
    save_dataframe(bucket_v2, intermediate / f"tick_bucket_features_{interval}m.csv", index=False)
    save_dataframe(micro_v2, intermediate / f"tick_micro_frame_{interval}m.csv", index=False)
    save_config_snapshot(config, output / "config_snapshot.yaml")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / "migration_record.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "migration": "recompute_conditional_run_z_from_raw_tick_and_run_counts",
                "legacy_fixed_p_scores_ignored": True,
                "aggressor_fields": "unavailable_in_archived_bucket_aggregates",
                "source_path": str(source.relative_to(PROJECT_ROOT)),
                "source_sha256": source_hash,
                "row_count": int(len(micro_v2)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_provenance_manifest(
        config,
        output / "provenance.json",
        schema_version=2,
        pipeline_version="tick-semantics-v2",
        statistical_method="conditional-run-z rebuilt from archived raw tick/run counts; exact-clock horizon",
        input_manifest_path=output / "migration_record.json",
    )
    print(f"wrote {len(micro_v2):,} schema-v2 rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

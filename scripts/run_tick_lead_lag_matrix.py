from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from tick_archive_backfill import execute_backfill_plan
from tick_data import resolve_tick_date_window
from tick_lead_lag import (
    build_lead_lag_matrix_report,
    plot_lead_lag_matrix,
    run_lead_lag_matrix,
)
from tick_short_horizon import build_tick_short_horizon_dataset, prepare_micro_herding_frame
from utils import (
    load_config,
    prepare_output_dirs,
    save_config_snapshot,
    save_dataframe,
    save_input_manifest,
    save_provenance_manifest,
    save_text,
    setup_logging,
)


def _validate_multiple_testing_family(
    analysis: dict,
    symbols: list[str],
    forward_horizon: int,
    event_filters: list[dict],
) -> tuple[str, int]:
    family = analysis.get("multiple_testing_family")
    if not isinstance(family, dict):
        raise ValueError("analysis.multiple_testing_family is required for the primary lead-lag matrix")
    if set(family.get("leaders", [])) != set(symbols) or set(family.get("targets", [])) != set(symbols):
        raise ValueError("Multiple-testing leader/target family must match analysis.symbols")
    if {int(value) for value in family.get("horizons_minutes", [])} != {int(forward_horizon)}:
        raise ValueError("Multiple-testing horizon family does not match the requested horizon")
    if {int(value) for value in family.get("sessions", [])} != {
        int(value) for value in analysis.get("session_hours_utc", [])
    }:
        raise ValueError("Multiple-testing session family does not match analysis.session_hours_utc")

    run_sides = {
        str(value)
        for event_filter in event_filters
        for value in event_filter.get("run_clustering_side", ["any"])
    }
    price_directions = {
        str(value)
        for event_filter in event_filters
        for value in event_filter.get("price_direction", ["any"])
    }
    if set(map(str, family.get("run_clustering_sides", []))) != run_sides:
        raise ValueError("Multiple-testing run-clustering family does not match analysis.event_filter")
    if set(map(str, family.get("price_directions", []))) != price_directions:
        raise ValueError("Multiple-testing price-direction family does not match analysis.event_filter")

    family_json = json.dumps(family, sort_keys=True, separators=(",", ":"))
    family_id = hashlib.sha256(family_json.encode("utf-8")).hexdigest()
    planned_size = len(symbols) * (len(symbols) - 1) * len(event_filters)
    return family_id, planned_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tick lead-lag 매트릭스 연구를 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "tick" / "multi_asset_365d" / "lead_lag_matrix.yaml"),
        help="설정 파일 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["data"] = resolve_tick_date_window(config["data"])
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    output_dirs = prepare_output_dirs(PROJECT_ROOT, config)
    save_config_snapshot(config, output_dirs["base"] / "config_snapshot.yaml")

    interval = int(config["analysis"]["interval_minutes"][0])
    forward_horizon = int(config["analysis"]["forward_horizons_minutes"][0])
    symbols = list(config["analysis"]["symbols"])
    event_filter = config["analysis"].get("event_filter", {})
    filter_keys = list(event_filter)
    filter_values = [
        value if isinstance(value, list) else [value]
        for value in event_filter.values()
    ]
    event_filters = [
        {key: [value] for key, value in zip(filter_keys, combination)}
        for combination in itertools.product(*filter_values)
    ] if filter_keys else [{}]
    family_id, planned_family_size = _validate_multiple_testing_family(
        config["analysis"], symbols, forward_horizon, event_filters
    )

    cached_micro_frame_path = str(config.get("research", {}).get("cached_micro_frame_path", "")).strip()

    if cached_micro_frame_path:
        backfill_summary = pd.DataFrame()
        micro_frame = pd.read_csv(cached_micro_frame_path)
        micro_frame["bucket_start"] = pd.to_datetime(micro_frame["bucket_start"], utc=True)
        for timestamp_column in ["bucket_end", "signal_timestamp"]:
            if timestamp_column in micro_frame.columns:
                micro_frame[timestamp_column] = pd.to_datetime(micro_frame[timestamp_column], utc=True)
        if "first_timestamp" in micro_frame.columns:
            micro_frame["first_timestamp"] = pd.to_datetime(micro_frame["first_timestamp"], utc=True, errors="coerce")
        if "last_timestamp" in micro_frame.columns:
            micro_frame["last_timestamp"] = pd.to_datetime(micro_frame["last_timestamp"], utc=True, errors="coerce")
        start_ts = pd.Timestamp(config["data"]["start"], tz="UTC")
        end_ts = pd.Timestamp(config["data"]["end"], tz="UTC") + pd.Timedelta(days=1)
        micro_frame = micro_frame.loc[
            (micro_frame["bucket_start"] >= start_ts) & (micro_frame["bucket_start"] < end_ts)
        ].copy()
        save_dataframe(micro_frame, output_dirs["intermediate"] / f"tick_micro_frame_{interval}m.csv", index=False)
        input_manifest = save_input_manifest(
            [PROJECT_ROOT / cached_micro_frame_path],
            output_dirs["base"] / "input_manifest.json",
        )
    else:
        backfill_summary = execute_backfill_plan(
            data_cfg=config["data"],
            max_workers=int(config.get("backfill", {}).get("max_workers", 4)),
        )
        save_dataframe(backfill_summary, output_dirs["base"] / "backfill_summary.csv", index=False)

        bucket_frames_by_interval, load_summary = build_tick_short_horizon_dataset(config)
        save_dataframe(load_summary, output_dirs["base"] / "tick_data_load_summary.csv", index=False)

        bucket_frame = bucket_frames_by_interval.get(interval, pd.DataFrame())
        if bucket_frame.empty:
            save_text(
                "# Tick Lead-Lag Matrix 연구\n\n- 사용할 버킷 데이터가 없습니다.\n",
                output_dirs["base"] / "tick_lead_lag_matrix_report.md",
            )
            return

        save_dataframe(bucket_frame, output_dirs["intermediate"] / f"tick_bucket_features_{interval}m.csv", index=False)
        micro_frame = prepare_micro_herding_frame(bucket_frame, config)
        save_dataframe(micro_frame, output_dirs["intermediate"] / f"tick_micro_frame_{interval}m.csv", index=False)
        input_manifest = output_dirs["base"] / "tick_data_load_summary.csv"

    matrix_summary = run_lead_lag_matrix(
        micro_frame=micro_frame,
        symbols=symbols,
        interval_minutes=interval,
        forward_horizon_minutes=forward_horizon,
        event_filters=event_filters,
        min_inference_events=int(config["analysis"].get("min_inference_events", 30)),
        min_inference_unique_days=int(config["analysis"].get("min_inference_unique_days", 20)),
    )
    if len(matrix_summary) != planned_family_size:
        raise RuntimeError(
            f"Multiple-testing family is incomplete: expected {planned_family_size}, got {len(matrix_summary)}"
        )
    matrix_summary["multiple_testing_family_sha256"] = family_id
    matrix_summary["multiple_testing_family_size"] = planned_family_size
    save_dataframe(matrix_summary, output_dirs["base"] / "lead_lag_matrix_summary.csv", index=False)
    save_provenance_manifest(
        config,
        output_dirs["base"] / "provenance.json",
        schema_version=2,
        pipeline_version="tick-semantics-v2",
        statistical_method="conditional-run-z; exact-clock-forward-return; UTC-day-cluster-bootstrap",
        input_manifest_path=input_manifest if input_manifest.exists() else None,
        random_seed=20260715,
    )

    plot_paths: list[str] = []
    for filter_name in matrix_summary.get("event_filter", pd.Series(dtype=str)).unique():
        filtered = matrix_summary.loc[matrix_summary["event_filter"] == filter_name]
        if filtered.empty:
            continue
        pivot_delta = filtered.pivot(index="target", columns="leader", values="delta_mean_return")
        pivot_t = filtered.pivot(index="target", columns="leader", values="delta_t_stat")
        pivot_count = filtered.pivot(index="target", columns="leader", values="event_count")
        slug = "filter_" + hashlib.sha256(filter_name.encode("utf-8")).hexdigest()[:12]
        save_dataframe(pivot_delta.reset_index(), output_dirs["base"] / f"lead_lag_matrix_{slug}_delta.csv", index=False)
        save_dataframe(pivot_t.reset_index(), output_dirs["base"] / f"lead_lag_matrix_{slug}_tstat.csv", index=False)
        save_dataframe(pivot_count.reset_index(), output_dirs["base"] / f"lead_lag_matrix_{slug}_count.csv", index=False)
        plot_path = output_dirs["plots"] / f"lead_lag_matrix_{slug}.png"
        plot_lead_lag_matrix(matrix_summary, event_filter=filter_name, path=plot_path)
        plot_paths.append(plot_path.relative_to(PROJECT_ROOT).as_posix())

    report = build_lead_lag_matrix_report(
        config=config,
        backfill_summary=backfill_summary,
        matrix_summary=matrix_summary,
        plot_paths=plot_paths,
    )
    save_text(report, output_dirs["base"] / "tick_lead_lag_matrix_report.md")


if __name__ == "__main__":
    raise SystemExit(main())

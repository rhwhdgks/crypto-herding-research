from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tick_archive_backfill import execute_backfill_plan
from tick_data import resolve_tick_date_window
from tick_lead_lag import (
    build_lead_lag_matrix_report,
    plot_lead_lag_matrix,
    run_lead_lag_matrix,
)
from tick_short_horizon import build_tick_short_horizon_dataset, prepare_micro_herding_frame
from utils import load_config, prepare_output_dirs, save_config_snapshot, save_dataframe, save_text, setup_logging


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
    directions = list(config["analysis"].get("directions", ["up", "down"]))

    cached_micro_frame_path = str(config.get("research", {}).get("cached_micro_frame_path", "")).strip()

    if cached_micro_frame_path:
        backfill_summary = pd.DataFrame()
        micro_frame = pd.read_csv(cached_micro_frame_path)
        micro_frame["bucket_start"] = pd.to_datetime(micro_frame["bucket_start"], utc=True)
        if "first_timestamp" in micro_frame.columns:
            micro_frame["first_timestamp"] = pd.to_datetime(micro_frame["first_timestamp"], utc=True, errors="coerce")
        if "last_timestamp" in micro_frame.columns:
            micro_frame["last_timestamp"] = pd.to_datetime(micro_frame["last_timestamp"], utc=True, errors="coerce")
        save_dataframe(micro_frame, output_dirs["intermediate"] / f"tick_micro_frame_{interval}m.csv", index=False)
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

    matrix_summary = run_lead_lag_matrix(
        micro_frame=micro_frame,
        symbols=symbols,
        interval_minutes=interval,
        forward_horizon_minutes=forward_horizon,
        directions=directions,
    )
    save_dataframe(matrix_summary, output_dirs["base"] / "lead_lag_matrix_summary.csv", index=False)

    plot_paths: list[str] = []
    for direction in directions:
        direction_frame = matrix_summary.loc[matrix_summary["direction"] == direction]
        if direction_frame.empty:
            continue
        pivot_delta = direction_frame.pivot(index="target", columns="leader", values="delta_mean_return")
        pivot_t = direction_frame.pivot(index="target", columns="leader", values="delta_t_stat")
        pivot_count = direction_frame.pivot(index="target", columns="leader", values="event_count")
        save_dataframe(pivot_delta.reset_index(), output_dirs["base"] / f"lead_lag_matrix_{direction}_delta.csv", index=False)
        save_dataframe(pivot_t.reset_index(), output_dirs["base"] / f"lead_lag_matrix_{direction}_tstat.csv", index=False)
        save_dataframe(pivot_count.reset_index(), output_dirs["base"] / f"lead_lag_matrix_{direction}_count.csv", index=False)
        plot_path = output_dirs["plots"] / f"lead_lag_matrix_{direction}.png"
        plot_lead_lag_matrix(matrix_summary, direction=direction, path=plot_path)
        plot_paths.append(str(plot_path))

    report = build_lead_lag_matrix_report(
        config=config,
        backfill_summary=backfill_summary,
        matrix_summary=matrix_summary,
        plot_paths=plot_paths,
    )
    save_text(report, output_dirs["base"] / "tick_lead_lag_matrix_report.md")


if __name__ == "__main__":
    raise SystemExit(main())

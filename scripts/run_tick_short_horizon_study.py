from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tick_short_horizon import (
    build_tick_short_horizon_dataset,
    build_tick_short_horizon_report,
    plot_tick_short_horizon_bars,
    prepare_micro_herding_frame,
    summarize_micro_herding,
)
from utils import load_config, prepare_output_dirs, save_config_snapshot, save_dataframe, save_text, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tick 기반 short-horizon micro-herding 연구를 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "tick" / "xrp_365d" / "short_horizon.yaml"),
        help="tick short-horizon 설정 파일 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    output_dirs = prepare_output_dirs(PROJECT_ROOT, config)
    save_config_snapshot(config, output_dirs["base"] / "config_snapshot.yaml")

    bucket_frames_by_interval, load_summary = build_tick_short_horizon_dataset(config)
    save_dataframe(load_summary, output_dirs["base"] / "tick_data_load_summary.csv", index=False)

    pooled_summary_by_interval: dict[int, object] = {}
    symbol_summary_by_interval: dict[int, object] = {}
    plot_paths: list[str] = []

    for interval, bucket_frame in bucket_frames_by_interval.items():
        if bucket_frame.empty:
            continue

        save_dataframe(bucket_frame, output_dirs["intermediate"] / f"tick_bucket_features_{interval}m.csv", index=False)
        micro_frame = prepare_micro_herding_frame(bucket_frame, config)
        pooled_summary, symbol_summary = summarize_micro_herding(
            micro_frame=micro_frame,
            forward_horizons=config["analysis"]["forward_horizons_minutes"],
        )

        pooled_summary_by_interval[int(interval)] = pooled_summary
        symbol_summary_by_interval[int(interval)] = symbol_summary

        save_dataframe(micro_frame, output_dirs["intermediate"] / f"tick_micro_frame_{interval}m.csv", index=False)
        save_dataframe(pooled_summary, output_dirs["base"] / f"tick_micro_pooled_summary_{interval}m.csv", index=False)
        save_dataframe(symbol_summary, output_dirs["base"] / f"tick_micro_symbol_summary_{interval}m.csv", index=False)

        plot_path = output_dirs["plots"] / f"tick_micro_delta_{interval}m.png"
        plot_tick_short_horizon_bars(pooled_summary, int(interval), plot_path)
        plot_paths.append(str(plot_path))

    report = build_tick_short_horizon_report(
        config=config,
        pooled_summary_by_interval=pooled_summary_by_interval,
        symbol_summary_by_interval=symbol_summary_by_interval,
        plot_paths=plot_paths,
    )
    save_text(report, output_dirs["base"] / "tick_short_horizon_report.md")


if __name__ == "__main__":
    raise SystemExit(main())

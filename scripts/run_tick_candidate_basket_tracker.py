from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tick_candidate_basket import (
    build_candidate_basket_report,
    build_candidate_variant_trade_log,
    load_candidate_micro_frames,
    summarize_candidate_basket,
)
from tick_variant_tracker import (
    build_variant_daily_summary,
    build_variant_recent_views,
    build_variant_signal_log,
    plot_variant_recent_signals,
    plot_variant_tracker_cumulative,
    summarize_variant_tracker_periods,
)
from utils import load_config, prepare_output_dirs, save_config_snapshot, save_dataframe, save_text, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tick candidate basket tracker를 생성합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "tick" / "multi_asset_365d" / "candidate_basket_tracker.yaml"),
        help="candidate basket tracker 설정 파일 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    output_dirs = prepare_output_dirs(PROJECT_ROOT, config)
    save_config_snapshot(config, output_dirs["base"] / "config_snapshot.yaml")

    micro_frame_paths = {
        symbol: PROJECT_ROOT / raw_path
        for symbol, raw_path in config["input"]["micro_frame_paths"].items()
    }
    analysis_cfg = config["analysis"]
    micro_frame = load_candidate_micro_frames(micro_frame_paths)
    trade_log = build_candidate_variant_trade_log(
        micro_frame=micro_frame,
        candidates=[dict(item) for item in analysis_cfg["candidates"]],
        focus_horizon_minutes=int(analysis_cfg["focus_horizon_minutes"]),
        round_trip_cost_bps=float(analysis_cfg.get("round_trip_cost_bps", 4.0)),
    )

    save_dataframe(trade_log, output_dirs["base"] / "variant_trade_log.csv", index=False)

    if str(analysis_cfg.get("as_of_mode", "now_utc")) == "now_utc":
        as_of_utc = pd.Timestamp.now(tz="UTC")
    else:
        as_of_utc = pd.Timestamp(analysis_cfg["as_of_utc"])
        as_of_utc = as_of_utc.tz_localize("UTC") if as_of_utc.tzinfo is None else as_of_utc.tz_convert("UTC")

    signal_log = build_variant_signal_log(trade_log, as_of_utc=as_of_utc)
    tracker_summary = summarize_variant_tracker_periods(
        signal_log=signal_log,
        recent_days_list=[int(value) for value in analysis_cfg.get("recent_days_list", [30, 60, 90])],
        as_of_utc=as_of_utc,
    )
    candidate_summary = summarize_candidate_basket(
        trade_log=trade_log,
        recent_days_list=[int(value) for value in analysis_cfg.get("recent_days_list", [30, 60, 90])],
    )
    active_signals, recent_signals = build_variant_recent_views(
        signal_log=signal_log,
        recent_count=int(analysis_cfg.get("recent_signal_count", 8)),
    )
    daily_summary = build_variant_daily_summary(signal_log)

    save_dataframe(signal_log, output_dirs["base"] / "signal_log.csv", index=False)
    save_dataframe(tracker_summary, output_dirs["base"] / "tracker_summary.csv", index=False)
    save_dataframe(candidate_summary, output_dirs["base"] / "candidate_summary.csv", index=False)
    save_dataframe(active_signals, output_dirs["base"] / "active_signals.csv", index=False)
    save_dataframe(recent_signals, output_dirs["base"] / "recent_closed_signals.csv", index=False)
    save_dataframe(daily_summary, output_dirs["base"] / "daily_tracker_summary.csv", index=False)

    cumulative_plot = output_dirs["plots"] / "tracker_cumulative.png"
    recent_plot = output_dirs["plots"] / "recent_signals.png"
    plot_variant_tracker_cumulative(daily_summary, cumulative_plot)
    plot_variant_recent_signals(recent_signals, recent_plot)

    report = build_candidate_basket_report(
        summary=candidate_summary,
        recent_signals=recent_signals,
        as_of_utc=as_of_utc,
        plot_paths=[str(cumulative_plot), str(recent_plot)],
    )
    save_text(report, output_dirs["base"] / "tick_candidate_basket_tracker_report.md")


if __name__ == "__main__":
    raise SystemExit(main())

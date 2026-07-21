from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from tick_forward_tracker import (
    build_daily_tracker_summary,
    build_recent_signal_views,
    build_signal_log,
    build_tracker_report,
    load_paper_trade_log,
    plot_recent_signals,
    plot_tracker_cumulative,
    summarize_tracker_periods,
)
from utils import load_config, prepare_output_dirs, save_config_snapshot, save_dataframe, save_text, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlap core forward paper tracker를 생성합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "tick" / "xrp_365d" / "overlap_core_tracker.yaml"),
        help="overlap core tracker 설정 파일 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    output_dirs = prepare_output_dirs(PROJECT_ROOT, config)
    save_config_snapshot(config, output_dirs["base"] / "config_snapshot.yaml")

    input_dir = PROJECT_ROOT / config["input"]["base_dir"]
    analysis_cfg = config["analysis"]
    if str(analysis_cfg.get("as_of_mode", "now_utc")) == "now_utc":
        as_of_utc = pd.Timestamp.now(tz="UTC")
    else:
        as_of_utc = pd.Timestamp(analysis_cfg["as_of_utc"])
        as_of_utc = as_of_utc.tz_localize("UTC") if as_of_utc.tzinfo is None else as_of_utc.tz_convert("UTC")

    trades = load_paper_trade_log(input_dir)
    signal_log = build_signal_log(trades, as_of_utc=as_of_utc)
    tracker_summary = summarize_tracker_periods(
        signal_log=signal_log,
        recent_days_list=[int(value) for value in analysis_cfg.get("recent_days_list", [7, 30, 60])],
        as_of_utc=as_of_utc,
    )
    active_signals, recent_signals = build_recent_signal_views(
        signal_log=signal_log,
        recent_count=int(analysis_cfg.get("recent_signal_count", 12)),
    )
    daily_summary = build_daily_tracker_summary(signal_log)

    save_dataframe(signal_log, output_dirs["base"] / "signal_log.csv", index=False)
    save_dataframe(tracker_summary, output_dirs["base"] / "tracker_summary.csv", index=False)
    save_dataframe(active_signals, output_dirs["base"] / "active_signals.csv", index=False)
    save_dataframe(recent_signals, output_dirs["base"] / "recent_closed_signals.csv", index=False)
    save_dataframe(daily_summary, output_dirs["base"] / "daily_tracker_summary.csv", index=False)

    cumulative_plot = output_dirs["plots"] / "tracker_cumulative.png"
    recent_plot = output_dirs["plots"] / "recent_signals.png"
    plot_tracker_cumulative(daily_summary, cumulative_plot)
    plot_recent_signals(recent_signals, recent_plot)

    report = build_tracker_report(
        signal_log=signal_log,
        tracker_summary=tracker_summary,
        active_signals=active_signals,
        recent_signals=recent_signals,
        daily_summary=daily_summary,
        as_of_utc=as_of_utc,
        plot_paths=[str(cumulative_plot), str(recent_plot)],
    )
    save_text(report, output_dirs["base"] / "tick_overlap_core_tracker_report.md")


if __name__ == "__main__":
    raise SystemExit(main())

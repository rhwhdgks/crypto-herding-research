from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tick_overlap_core_regime import (
    build_overlap_core_regime_report,
    enrich_overlap_core_trades,
    load_overlap_core_trade_log,
    load_trade_sample_with_prior_features,
    plot_bucket_summary,
    plot_hour_summary,
    plot_rolling_summary,
    plot_yearly_summary,
    summarize_group,
    summarize_rolling_windows,
    summarize_strength_and_prior_buckets,
)
from utils import load_config, prepare_output_dirs, save_config_snapshot, save_dataframe, save_text, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tick overlap_core regime 분석을 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "tick" / "xrp_5y" / "overlap_core_regime.yaml"),
        help="overlap_core regime 설정 파일 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    output_dirs = prepare_output_dirs(PROJECT_ROOT, config)
    save_config_snapshot(config, output_dirs["base"] / "config_snapshot.yaml")

    overlap_dir = PROJECT_ROOT / config["input"]["overlap_base_dir"]
    trade_sample_dir = PROJECT_ROOT / config["input"]["trade_sample_base_dir"]
    analysis_cfg = config.get("analysis", {})

    overlap_trades = load_overlap_core_trade_log(overlap_dir)
    trade_sample = load_trade_sample_with_prior_features(trade_sample_dir)
    sample_frame = enrich_overlap_core_trades(overlap_trades, trade_sample)

    yearly_summary = summarize_group(sample_frame, ["calendar_year"])
    month_summary = summarize_group(sample_frame, ["calendar_month", "calendar_month_label"]).sort_values("calendar_month")
    quarter_summary = summarize_group(sample_frame, ["calendar_quarter"])
    hour_summary = summarize_group(sample_frame, ["hour_utc"]).sort_values("hour_utc")
    strength_summary, prior_summary = summarize_strength_and_prior_buckets(
        sample_frame,
        strength_bucket_count=int(analysis_cfg.get("strength_bucket_count", 4)),
        prior_drop_bucket_count=int(analysis_cfg.get("prior_drop_bucket_count", 4)),
    )
    rolling_summary = summarize_rolling_windows(
        sample_frame,
        window_days_list=[int(value) for value in analysis_cfg.get("rolling_window_days", [90, 180])],
    )

    save_dataframe(sample_frame, output_dirs["base"] / "overlap_core_regime_sample.csv", index=False)
    save_dataframe(yearly_summary, output_dirs["base"] / "yearly_summary.csv", index=False)
    save_dataframe(month_summary, output_dirs["base"] / "month_of_year_summary.csv", index=False)
    save_dataframe(quarter_summary, output_dirs["base"] / "quarter_summary.csv", index=False)
    save_dataframe(hour_summary, output_dirs["base"] / "hour_summary.csv", index=False)
    save_dataframe(strength_summary, output_dirs["base"] / "strength_bucket_summary.csv", index=False)
    save_dataframe(prior_summary, output_dirs["base"] / "prior_drop_bucket_summary.csv", index=False)
    save_dataframe(rolling_summary, output_dirs["base"] / "rolling_summary.csv", index=False)

    yearly_plot = output_dirs["plots"] / "yearly_summary.png"
    hour_plot = output_dirs["plots"] / "hour_summary.png"
    strength_plot = output_dirs["plots"] / "strength_bucket_summary.png"
    prior_plot = output_dirs["plots"] / "prior_drop_bucket_summary.png"
    rolling_plot = output_dirs["plots"] / "rolling_summary.png"
    plot_yearly_summary(yearly_summary, yearly_plot)
    plot_hour_summary(hour_summary, hour_plot)
    plot_bucket_summary(strength_summary, "strength_bucket", "강도 bucket별 평균 순수익", strength_plot)
    plot_bucket_summary(prior_summary, "prior_drop_bucket", "직전 하락폭 bucket별 평균 순수익", prior_plot)
    plot_rolling_summary(rolling_summary, rolling_plot)

    report = build_overlap_core_regime_report(
        sample_frame=sample_frame,
        yearly_summary=yearly_summary,
        month_summary=month_summary,
        quarter_summary=quarter_summary,
        hour_summary=hour_summary,
        strength_summary=strength_summary,
        prior_summary=prior_summary,
        rolling_summary=rolling_summary,
        plot_paths=[str(yearly_plot), str(hour_plot), str(strength_plot), str(prior_plot), str(rolling_plot)],
    )
    save_text(report, output_dirs["base"] / "tick_overlap_core_regime_report.md")


if __name__ == "__main__":
    raise SystemExit(main())

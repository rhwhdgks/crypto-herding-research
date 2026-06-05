from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tick_sol_xrp_filter import (
    build_filter_candidate_log,
    build_filtered_trade_log,
    build_sol_context,
    build_sol_filter_report,
    load_trade_and_micro_frames,
    plot_filter_break_even,
    plot_filter_oos,
    summarize_filter_costs,
    summarize_filter_months,
    summarize_filter_oos,
)
from utils import load_config, prepare_output_dirs, save_config_snapshot, save_dataframe, save_text, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SOL -> XRP 조건부 필터 연구를 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "tick" / "multi_asset_365d" / "sol_xrp_filter.yaml"),
        help="설정 파일 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    output_dirs = prepare_output_dirs(PROJECT_ROOT, config)
    save_config_snapshot(config, output_dirs["base"] / "config_snapshot.yaml")

    trades, micro = load_trade_and_micro_frames(
        trade_log_path=config["input"]["trade_log_path"],
        micro_frame_path=config["input"]["micro_frame_path"],
    )
    start_ts = micro["bucket_start"].min()
    end_ts = micro["bucket_start"].max()
    sol_context = build_sol_context(micro)
    filtered_trades = build_filtered_trade_log(trades, sol_context=sol_context, start_ts=start_ts, end_ts=end_ts)
    save_dataframe(filtered_trades, output_dirs["intermediate"] / "filtered_trade_frame.csv", index=False)

    candidate_log = build_filter_candidate_log(filtered_trades)
    save_dataframe(candidate_log, output_dirs["base"] / "filter_candidate_trade_log.csv", index=False)

    holdout_days = [int(v) for v in config["analysis"]["holdout_days"]]
    cost_grid = [float(v) for v in config["analysis"]["cost_bps_grid"]]
    oos_summary = summarize_filter_oos(candidate_log, holdout_days_list=holdout_days)
    cost_summary, break_even_summary = summarize_filter_costs(
        candidate_log,
        holdout_days_list=holdout_days,
        cost_bps_grid=cost_grid,
    )
    monthly_summary = summarize_filter_months(candidate_log)

    save_dataframe(oos_summary, output_dirs["base"] / "filter_oos_summary.csv", index=False)
    save_dataframe(cost_summary, output_dirs["base"] / "filter_cost_summary.csv", index=False)
    save_dataframe(break_even_summary, output_dirs["base"] / "filter_break_even_summary.csv", index=False)
    save_dataframe(monthly_summary, output_dirs["base"] / "filter_monthly_summary.csv", index=False)

    plot_oos_path = output_dirs["plots"] / "filter_oos.png"
    plot_break_even_path = output_dirs["plots"] / "filter_break_even.png"
    plot_filter_oos(oos_summary, plot_oos_path)
    plot_filter_break_even(break_even_summary, plot_break_even_path)

    report = build_sol_filter_report(
        oos_summary=oos_summary,
        break_even_summary=break_even_summary,
        monthly_summary=monthly_summary,
        plot_paths=[str(plot_oos_path), str(plot_break_even_path)],
    )
    save_text(report, output_dirs["base"] / "tick_sol_xrp_filter_report.md")


if __name__ == "__main__":
    raise SystemExit(main())

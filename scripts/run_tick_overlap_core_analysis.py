from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from tick_candidate_paper_sim import plot_equity_curve, plot_monthly_bars
from tick_cost_sanity import plot_cost_curves
from tick_overlap_core_analysis import (
    build_overlap_core_report,
    load_overlap_core_trades,
    plot_overlap_blocks,
    plot_overlap_oos,
    prepare_overlap_paper_sim_trade_log,
    summarize_cost_grid,
    summarize_overlap_blocks,
    summarize_overlap_oos,
    build_equity_curves,
    build_monthly_summary,
)
from utils import load_config, prepare_output_dirs, save_config_snapshot, save_dataframe, save_text, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tick 겹침 핵심 후보 검증을 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "tick" / "xrp_365d" / "overlap_core.yaml"),
        help="tick overlap core 설정 파일 경로",
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
    round_trip_cost_bps = float(analysis_cfg.get("round_trip_cost_bps", 4))
    holdout_days_list = [int(value) for value in analysis_cfg.get("holdout_days_list", [30, 60])]
    cost_bps_grid = [float(value) for value in analysis_cfg.get("cost_bps_grid", [2, 4, 6, 8, 10])]
    block_days = int(analysis_cfg.get("block_days", 30))
    block_step_days = int(analysis_cfg.get("block_step_days", 10))

    trades = load_overlap_core_trades(input_dir)
    save_dataframe(trades, output_dirs["base"] / "overlap_core_trades.csv", index=False)

    unique_event_days = int(trades["entry_timestamp"].dt.normalize().nunique()) if not trades.empty else 0
    valid_holdout_days = [days for days in holdout_days_list if days < unique_event_days]

    oos_summary = summarize_overlap_oos(
        trades=trades,
        round_trip_cost_bps=round_trip_cost_bps,
        holdout_days_list=valid_holdout_days,
    )
    block_detail, block_summary = summarize_overlap_blocks(
        trades=trades,
        round_trip_cost_bps=round_trip_cost_bps,
        block_days=block_days,
        step_days=block_step_days,
    )
    cost_summary, break_even_summary = summarize_cost_grid(
        trades=trades,
        holdout_days_list=valid_holdout_days,
        cost_bps_grid=cost_bps_grid,
    )

    paper_trade_log = prepare_overlap_paper_sim_trade_log(trades, round_trip_cost_bps=round_trip_cost_bps)
    monthly_summary = build_monthly_summary(paper_trade_log)
    equity_curves = build_equity_curves(paper_trade_log)

    save_dataframe(oos_summary, output_dirs["base"] / "overlap_oos_summary.csv", index=False)
    save_dataframe(block_detail, output_dirs["base"] / "overlap_block_detail.csv", index=False)
    save_dataframe(block_summary, output_dirs["base"] / "overlap_block_summary.csv", index=False)
    save_dataframe(cost_summary, output_dirs["base"] / "cost_summary.csv", index=False)
    save_dataframe(break_even_summary, output_dirs["base"] / "break_even_summary.csv", index=False)
    save_dataframe(paper_trade_log, output_dirs["base"] / "paper_trade_log.csv", index=False)
    save_dataframe(monthly_summary, output_dirs["base"] / "monthly_summary.csv", index=False)
    save_dataframe(equity_curves, output_dirs["base"] / "equity_curves.csv", index=False)

    oos_plot = output_dirs["plots"] / "overlap_oos.png"
    block_plot = output_dirs["plots"] / "overlap_blocks.png"
    cost_plot = output_dirs["plots"] / "cost_curve.png"
    equity_plot = output_dirs["plots"] / "equity_curve.png"
    equity_recent_plot = output_dirs["plots"] / "equity_curve_recent_90d.png"
    monthly_plot = output_dirs["plots"] / "monthly_bars.png"

    plot_overlap_oos(oos_summary, oos_plot)
    plot_overlap_blocks(block_detail, block_plot)
    plot_cost_curves(cost_summary, cost_plot)
    plot_equity_curve(equity_curves, equity_plot)
    plot_equity_curve(equity_curves, equity_recent_plot, recent_days=90)
    plot_monthly_bars(monthly_summary, monthly_plot)

    report = build_overlap_core_report(
        oos_summary=oos_summary,
        break_even_summary=break_even_summary,
        cost_summary=cost_summary,
        monthly_summary=monthly_summary,
        block_summary=block_summary,
        distinct_event_days=unique_event_days,
        round_trip_cost_bps=round_trip_cost_bps,
        plot_paths=[
            str(oos_plot),
            str(block_plot),
            str(cost_plot),
            str(equity_plot),
            str(equity_recent_plot),
            str(monthly_plot),
        ],
    )
    save_text(report, output_dirs["base"] / "tick_overlap_core_report.md")


if __name__ == "__main__":
    raise SystemExit(main())

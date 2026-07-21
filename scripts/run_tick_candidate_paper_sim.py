from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from tick_candidate_paper_sim import (
    build_candidate_paper_report,
    build_candidate_trade_log,
    build_equity_curves,
    build_monthly_summary,
    plot_equity_curve,
    plot_monthly_bars,
    summarize_candidate_performance,
)
from utils import load_config, prepare_output_dirs, save_config_snapshot, save_dataframe, save_text, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tick 후보 규칙 paper simulation을 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "tick" / "xrp_365d" / "candidate_paper_sim.yaml"),
        help="tick candidate paper simulation 설정 파일 경로",
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
    trade_log = build_candidate_trade_log(
        base_dir=input_dir,
        selected_candidates=[str(value) for value in analysis_cfg.get("candidates", [])],
        round_trip_cost_bps=float(analysis_cfg.get("round_trip_cost_bps", 4)),
    )
    save_dataframe(trade_log, output_dirs["base"] / "candidate_trade_log.csv", index=False)

    summary = summarize_candidate_performance(
        trade_log=trade_log,
        recent_days_list=[int(value) for value in analysis_cfg.get("recent_days_list", [30, 60, 90])],
    )
    monthly_summary = build_monthly_summary(trade_log)
    curves = build_equity_curves(trade_log)

    save_dataframe(summary, output_dirs["base"] / "candidate_summary.csv", index=False)
    save_dataframe(monthly_summary, output_dirs["base"] / "candidate_monthly_summary.csv", index=False)
    save_dataframe(curves, output_dirs["base"] / "candidate_equity_curves.csv", index=False)

    full_plot = output_dirs["plots"] / "candidate_equity_full.png"
    recent_plot = output_dirs["plots"] / "candidate_equity_recent_90d.png"
    monthly_plot = output_dirs["plots"] / "candidate_monthly_bars.png"
    plot_equity_curve(curves, full_plot)
    plot_equity_curve(curves, recent_plot, recent_days=90)
    plot_monthly_bars(monthly_summary, monthly_plot)

    report = build_candidate_paper_report(
        summary=summary,
        monthly_summary=monthly_summary,
        round_trip_cost_bps=float(analysis_cfg.get("round_trip_cost_bps", 4)),
        plot_paths=[str(full_plot), str(recent_plot), str(monthly_plot)],
    )
    save_text(report, output_dirs["base"] / "tick_candidate_paper_sim_report.md")


if __name__ == "__main__":
    raise SystemExit(main())

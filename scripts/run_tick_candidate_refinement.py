from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tick_candidate_refinement import (
    build_candidate_refinement_report,
    build_partition_frame,
    load_candidate_trade_log,
    plot_partition_cost_summary,
    plot_partition_period_summary,
    summarize_partitions_by_cost,
    summarize_partitions_by_period,
)
from utils import load_config, prepare_output_dirs, save_config_snapshot, save_dataframe, save_text, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tick 후보 겹침 정제 연구를 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "tick" / "xrp_365d" / "candidate_refinement.yaml"),
        help="tick candidate refinement 설정 파일 경로",
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
    trade_log = load_candidate_trade_log(input_dir)

    partition_frame, overlap_summary = build_partition_frame(
        trade_log=trade_log,
        prev_candidate=str(analysis_cfg.get("prev_candidate", "prev_neg")),
        ratio_candidate=str(analysis_cfg.get("ratio_candidate", "ratio_1_40_16_18")),
    )
    period_summary = summarize_partitions_by_period(
        partition_frame=partition_frame,
        round_trip_cost_bps=float(analysis_cfg.get("round_trip_cost_bps", 4)),
        recent_days_list=[int(value) for value in analysis_cfg.get("recent_days_list", [30, 60, 90])],
    )
    cost_summary = summarize_partitions_by_cost(
        partition_frame=partition_frame,
        cost_bps_grid=[float(value) for value in analysis_cfg.get("cost_bps_grid", [2, 4, 6, 8, 10])],
    )

    save_dataframe(partition_frame, output_dirs["base"] / "partition_trade_log.csv", index=False)
    save_dataframe(overlap_summary, output_dirs["base"] / "overlap_summary.csv", index=False)
    save_dataframe(period_summary, output_dirs["base"] / "period_summary.csv", index=False)
    save_dataframe(cost_summary, output_dirs["base"] / "cost_summary.csv", index=False)

    period_plot = output_dirs["plots"] / "partition_period_summary.png"
    cost_plot = output_dirs["plots"] / "partition_cost_summary.png"
    plot_partition_period_summary(period_summary, period_plot)
    plot_partition_cost_summary(cost_summary, cost_plot)

    report = build_candidate_refinement_report(
        overlap_summary=overlap_summary,
        period_summary=period_summary,
        cost_summary=cost_summary,
        round_trip_cost_bps=float(analysis_cfg.get("round_trip_cost_bps", 4)),
        plot_paths=[str(period_plot), str(cost_plot)],
    )
    save_text(report, output_dirs["base"] / "tick_candidate_refinement_report.md")


if __name__ == "__main__":
    raise SystemExit(main())

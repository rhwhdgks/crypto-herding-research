from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tick_subset_candidates import (
    build_candidate_masks,
    build_subset_candidate_report,
    enrich_trade_features,
    load_trade_sample,
    plot_candidate_block_share,
    plot_candidate_net_returns,
    summarize_candidates_by_blocks,
    summarize_candidates_by_period,
)
from utils import load_config, prepare_output_dirs, save_config_snapshot, save_dataframe, save_text, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tick subset 후보 비교 연구를 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "tick" / "xrp_365d" / "subset_candidates.yaml"),
        help="tick subset 후보 설정 파일 경로",
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
    trades = load_trade_sample(input_dir)
    trades = enrich_trade_features(trades)
    candidate_masks = build_candidate_masks(trades)

    period_summary = summarize_candidates_by_period(
        trades=trades,
        candidate_masks=candidate_masks,
        holdout_days_list=[int(value) for value in analysis_cfg.get("holdout_days_list", [30, 60])],
        round_trip_cost_bps=float(analysis_cfg.get("round_trip_cost_bps", 4)),
    )
    block_detail, block_summary = summarize_candidates_by_blocks(
        trades=trades,
        candidate_masks=candidate_masks,
        block_days=int(analysis_cfg.get("block_days", 30)),
        round_trip_cost_bps=float(analysis_cfg.get("round_trip_cost_bps", 4)),
    )

    save_dataframe(period_summary, output_dirs["base"] / "candidate_period_summary.csv", index=False)
    save_dataframe(block_detail, output_dirs["base"] / "candidate_block_detail.csv", index=False)
    save_dataframe(block_summary, output_dirs["base"] / "candidate_block_summary.csv", index=False)

    net_plot = output_dirs["plots"] / "candidate_net_return.png"
    block_plot = output_dirs["plots"] / "candidate_block_share.png"
    plot_candidate_net_returns(period_summary, net_plot)
    plot_candidate_block_share(block_summary, block_plot)

    report = build_subset_candidate_report(
        period_summary=period_summary,
        block_summary=block_summary,
        round_trip_cost_bps=float(analysis_cfg.get("round_trip_cost_bps", 4)),
        plot_paths=[str(net_plot), str(block_plot)],
    )
    save_text(report, output_dirs["base"] / "tick_subset_candidates_report.md")


if __name__ == "__main__":
    raise SystemExit(main())

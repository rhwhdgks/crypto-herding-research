from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tick_overlap_core_variants import (
    build_variant_masks,
    build_variant_report,
    build_variant_trade_log,
    enrich_variant_buckets,
    load_overlap_core_regime_sample,
    plot_variant_break_even,
    plot_variant_oos,
    summarize_variant_block_stability,
    summarize_variant_costs,
    summarize_variant_months,
    summarize_variant_oos,
)
from utils import load_config, prepare_output_dirs, save_config_snapshot, save_dataframe, save_text, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5년 overlap_core 변형 비교를 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "tick" / "xrp_5y" / "overlap_core_variants.yaml"),
        help="overlap_core 변형 비교 설정 파일 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    output_dirs = prepare_output_dirs(PROJECT_ROOT, config)
    save_config_snapshot(config, output_dirs["base"] / "config_snapshot.yaml")

    input_dir = PROJECT_ROOT / config["input"]["base_dir"]
    analysis_cfg = config.get("analysis", {})

    sample = load_overlap_core_regime_sample(input_dir)
    sample, thresholds = enrich_variant_buckets(sample)
    masks = build_variant_masks(sample)
    trade_log = build_variant_trade_log(sample, masks)

    oos_summary = summarize_variant_oos(
        trade_log,
        holdout_days_list=[int(value) for value in analysis_cfg.get("holdout_days_list", [30, 60])],
    )
    block_summary = summarize_variant_block_stability(
        trade_log,
        block_days=int(analysis_cfg.get("block_days", 30)),
    )
    cost_summary, break_even_summary = summarize_variant_costs(
        trade_log,
        holdout_days_list=[int(value) for value in analysis_cfg.get("holdout_days_list", [30, 60])],
        cost_bps_grid=[float(value) for value in analysis_cfg.get("cost_bps_grid", [2, 4, 6, 8, 10])],
    )
    monthly_summary = summarize_variant_months(trade_log)

    save_dataframe(sample, output_dirs["base"] / "variant_input_sample.csv", index=False)
    save_dataframe(trade_log, output_dirs["base"] / "variant_trade_log.csv", index=False)
    save_dataframe(oos_summary, output_dirs["base"] / "variant_oos_summary.csv", index=False)
    save_dataframe(block_summary, output_dirs["base"] / "variant_block_summary.csv", index=False)
    save_dataframe(cost_summary, output_dirs["base"] / "variant_cost_summary.csv", index=False)
    save_dataframe(break_even_summary, output_dirs["base"] / "variant_break_even_summary.csv", index=False)
    save_dataframe(monthly_summary, output_dirs["base"] / "variant_monthly_summary.csv", index=False)

    oos_plot = output_dirs["plots"] / "variant_oos.png"
    break_even_plot = output_dirs["plots"] / "variant_break_even.png"
    plot_variant_oos(oos_summary, oos_plot)
    plot_variant_break_even(break_even_summary, break_even_plot)

    report = build_variant_report(
        thresholds=thresholds,
        oos_summary=oos_summary,
        block_summary=block_summary,
        break_even_summary=break_even_summary,
        monthly_summary=monthly_summary,
        plot_paths=[str(oos_plot), str(break_even_plot)],
    )
    save_text(report, output_dirs["base"] / "tick_overlap_core_variants_report.md")


if __name__ == "__main__":
    raise SystemExit(main())

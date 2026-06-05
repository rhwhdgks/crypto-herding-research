from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tick_fixed_rule_oos import (
    build_fixed_rule_report,
    filter_micro_frame_by_symbols,
    load_micro_frame,
    plot_holdout_symbol_delta,
    plot_period_delta,
    summarize_fixed_rule,
)
from utils import load_config, prepare_output_dirs, save_config_snapshot, save_dataframe, save_text, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tick 고정 룰 short-horizon OOS 검증을 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "tick" / "xrp_365d" / "fixed_rule_oos.yaml"),
        help="tick 고정 룰 OOS 설정 파일 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    output_dirs = prepare_output_dirs(PROJECT_ROOT, config)
    save_config_snapshot(config, output_dirs["base"] / "config_snapshot.yaml")

    input_dir = PROJECT_ROOT / config["input"]["base_dir"]
    rule_cfg = config["rule"]

    micro_frame = load_micro_frame(
        base_dir=input_dir,
        interval_minutes=int(rule_cfg["interval_minutes"]),
    )
    micro_frame = filter_micro_frame_by_symbols(micro_frame, rule_cfg.get("symbols"))
    summary, symbol_summary = summarize_fixed_rule(
        frame=micro_frame,
        event_label=str(rule_cfg["event_label"]),
        horizon_minutes=int(rule_cfg["horizon_minutes"]),
        holdout_days=int(rule_cfg["holdout_days"]),
    )

    save_dataframe(summary, output_dirs["base"] / "fixed_rule_summary.csv", index=False)
    save_dataframe(symbol_summary, output_dirs["base"] / "fixed_rule_symbol_summary.csv", index=False)

    period_plot = output_dirs["plots"] / "fixed_rule_period_delta.png"
    symbol_plot = output_dirs["plots"] / "fixed_rule_holdout_symbol_delta.png"
    plot_period_delta(summary, period_plot)
    plot_holdout_symbol_delta(symbol_summary, symbol_plot)

    report = build_fixed_rule_report(
        interval_minutes=int(rule_cfg["interval_minutes"]),
        event_label=str(rule_cfg["event_label"]),
        horizon_minutes=int(rule_cfg["horizon_minutes"]),
        holdout_days=int(rule_cfg["holdout_days"]),
        symbols=rule_cfg.get("symbols"),
        summary=summary,
        symbol_summary=symbol_summary,
        plot_paths=[str(period_plot), str(symbol_plot)],
    )
    save_text(report, output_dirs["base"] / "tick_fixed_rule_oos_report.md")


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tick_cost_sanity import (
    build_cost_sanity_report,
    build_tick_trade_frame,
    plot_cost_curves,
    summarize_cost_grid,
)
from tick_fixed_rule_oos import filter_micro_frame_by_symbols, load_micro_frame
from utils import load_config, prepare_output_dirs, save_config_snapshot, save_dataframe, save_text, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tick 고정 룰 비용 sanity check를 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "tick" / "xrp_365d" / "cost_sanity.yaml"),
        help="tick cost sanity 설정 파일 경로",
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
    analysis_cfg = config["analysis"]

    frame = load_micro_frame(input_dir, interval_minutes=int(rule_cfg["interval_minutes"]))
    frame = filter_micro_frame_by_symbols(frame, rule_cfg.get("symbols"))

    trades = build_tick_trade_frame(
        frame=frame,
        event_label=str(rule_cfg["event_label"]),
        horizon_minutes=int(rule_cfg["horizon_minutes"]),
        enforce_non_overlap=bool(analysis_cfg.get("enforce_non_overlap", True)),
    )
    save_dataframe(trades, output_dirs["base"] / "trade_sample.csv", index=False)

    cost_summary, break_even_summary = summarize_cost_grid(
        trades=trades,
        holdout_days_list=[int(value) for value in analysis_cfg.get("holdout_days_list", [30, 60])],
        cost_bps_grid=[float(value) for value in analysis_cfg.get("cost_bps_grid", [2, 4, 6, 8, 10, 12])],
    )
    save_dataframe(cost_summary, output_dirs["base"] / "cost_summary.csv", index=False)
    save_dataframe(break_even_summary, output_dirs["base"] / "break_even_summary.csv", index=False)

    plot_path = output_dirs["plots"] / "cost_curve.png"
    plot_cost_curves(cost_summary, plot_path)

    report = build_cost_sanity_report(
        rule_label=f"{rule_cfg['event_label']} micro-herding",
        symbol_label=", ".join(rule_cfg.get("symbols", [])) or "전체",
        horizon_minutes=int(rule_cfg["horizon_minutes"]),
        cost_summary=cost_summary,
        break_even_summary=break_even_summary,
        plot_paths=[str(plot_path)],
    )
    save_text(report, output_dirs["base"] / "tick_cost_sanity_report.md")


if __name__ == "__main__":
    raise SystemExit(main())

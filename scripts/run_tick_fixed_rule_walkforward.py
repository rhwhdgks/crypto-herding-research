from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from tick_fixed_rule_oos import filter_micro_frame_by_symbols, load_micro_frame
from tick_fixed_rule_walkforward import build_walkforward_report, compute_walkforward_summary, plot_walkforward_delta
from utils import load_config, prepare_output_dirs, save_config_snapshot, save_dataframe, save_text, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tick 고정 룰 순차 블록 검증을 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "tick" / "xrp_365d" / "fixed_rule_walkforward.yaml"),
        help="tick fixed-rule walk-forward 설정 파일 경로",
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

    summary = compute_walkforward_summary(
        frame=frame,
        event_label=str(rule_cfg["event_label"]),
        horizon_minutes=int(rule_cfg["horizon_minutes"]),
        block_days=int(analysis_cfg["block_days"]),
        step_days=int(analysis_cfg["step_days"]),
    )
    save_dataframe(summary, output_dirs["base"] / "walkforward_summary.csv", index=False)

    plot_path = output_dirs["plots"] / "walkforward_delta.png"
    plot_walkforward_delta(summary, plot_path)

    report = build_walkforward_report(
        summary=summary,
        interval_minutes=int(rule_cfg["interval_minutes"]),
        event_label=str(rule_cfg["event_label"]),
        horizon_minutes=int(rule_cfg["horizon_minutes"]),
        symbols=rule_cfg.get("symbols"),
        block_days=int(analysis_cfg["block_days"]),
        step_days=int(analysis_cfg["step_days"]),
        plot_paths=[str(plot_path)],
    )
    save_text(report, output_dirs["base"] / "tick_fixed_rule_walkforward_report.md")


if __name__ == "__main__":
    raise SystemExit(main())

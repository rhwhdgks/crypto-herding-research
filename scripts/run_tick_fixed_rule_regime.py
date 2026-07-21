from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from tick_fixed_rule_oos import filter_micro_frame_by_symbols, load_micro_frame
from tick_fixed_rule_regime import build_regime_report, compute_rolling_regime_summary, plot_regime_lines
from utils import load_config, prepare_output_dirs, save_config_snapshot, save_dataframe, save_text, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tick 고정 룰 regime 점검을 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "tick" / "xrp_365d" / "fixed_rule_regime.yaml"),
        help="tick fixed-rule regime 설정 파일 경로",
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

    frame = load_micro_frame(input_dir, interval_minutes=int(rule_cfg["interval_minutes"]))
    frame = filter_micro_frame_by_symbols(frame, rule_cfg.get("symbols"))
    regime_summary = compute_rolling_regime_summary(
        frame=frame,
        event_label=str(rule_cfg["event_label"]),
        horizon_minutes=int(rule_cfg["horizon_minutes"]),
        window_days_list=[int(value) for value in config["analysis"]["window_days_list"]],
    )
    save_dataframe(regime_summary, output_dirs["base"] / "rolling_regime_summary.csv", index=False)

    plot_path = output_dirs["plots"] / "rolling_regime_lines.png"
    plot_regime_lines(regime_summary, plot_path)

    report = build_regime_report(regime_summary, [str(plot_path)])
    save_text(report, output_dirs["base"] / "tick_fixed_rule_regime_report.md")
    print(output_dirs["base"] / "tick_fixed_rule_regime_report.md")


if __name__ == "__main__":
    raise SystemExit(main())

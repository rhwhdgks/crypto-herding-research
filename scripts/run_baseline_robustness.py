from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baseline_robustness import (
    build_baseline_robustness_report,
    build_chronological_segment_specs,
    build_day_type_segment_specs,
    build_session_segment_specs,
    build_volatility_segment_specs,
    compute_rolling_beta2,
    load_baseline_intermediate_outputs,
    plot_focus_returns,
    plot_group_beta2,
    plot_rolling_beta2,
    summarize_segments,
)
from utils import load_config, save_dataframe, save_text, setup_logging


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="baseline 2년 1분봉 robustness 분석을 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "baseline" / "config.yaml"),
        help="baseline 설정 파일 경로",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="robustness 결과를 저장할 디렉터리",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))

    baseline_output_dir = PROJECT_ROOT / config.get("output", {}).get("base_dir", "outputs/baseline")
    output_dir = Path(args.output_dir) if args.output_dir else baseline_output_dir / "robustness"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    analysis_frame, market_return = load_baseline_intermediate_outputs(baseline_output_dir)
    holding_periods = [int(period) for period in config["event_study"]["holding_periods"]]

    LOGGER.info("baseline robustness 분석을 시작합니다. 관측치=%d.", len(analysis_frame))

    segment_specs = []
    segment_specs.extend(build_chronological_segment_specs(analysis_frame.index))
    segment_specs.extend(build_session_segment_specs(analysis_frame.index))
    segment_specs.extend(build_day_type_segment_specs(analysis_frame.index))
    segment_specs.extend(build_volatility_segment_specs(analysis_frame))

    regression_summary, best_summary, focus_summary = summarize_segments(
        analysis_frame=analysis_frame,
        market_return=market_return,
        segment_specs=segment_specs,
        holding_periods=holding_periods,
        focus_horizons=[15, 120, 1440],
        min_observations=10_000,
    )
    rolling_summary = compute_rolling_beta2(
        analysis_frame=analysis_frame,
        window_minutes=30 * 1440,
        step_minutes=1440,
        min_observations=30 * 1440,
    )

    save_dataframe(regression_summary, output_dir / "regression_summary.csv", index=False)
    save_dataframe(best_summary, output_dir / "best_horizon_summary.csv", index=False)
    save_dataframe(focus_summary, output_dir / "focus_horizon_summary.csv", index=False)
    save_dataframe(rolling_summary, output_dir / "rolling_beta2_30d.csv", index=False)

    plot_group_beta2(
        regression_summary=regression_summary,
        segment_group="chronological",
        order=["full_sample", "first_half", "second_half", "q1", "q2", "q3", "q4"],
        path=plots_dir / "chronological_beta2.png",
        title="반기·분기별 beta2",
    )
    plot_group_beta2(
        regression_summary=regression_summary,
        segment_group="session",
        order=["utc_00_07", "utc_08_15", "utc_16_23"],
        path=plots_dir / "session_beta2.png",
        title="시간대별 beta2",
    )
    plot_focus_returns(
        focus_summary=focus_summary,
        segment_group="session",
        horizon_minutes=1440,
        order=["utc_00_07", "utc_08_15", "utc_16_23"],
        path=plots_dir / "session_1d_returns.png",
        title="시간대별 1일 event-study 평균 수익률",
    )
    plot_focus_returns(
        focus_summary=focus_summary,
        segment_group="volatility_regime",
        horizon_minutes=1440,
        order=["low_vol", "mid_vol", "high_vol"],
        path=plots_dir / "volatility_1d_returns.png",
        title="변동성 상태별 1일 event-study 평균 수익률",
    )
    plot_rolling_beta2(
        rolling_summary=rolling_summary,
        path=plots_dir / "rolling_beta2_30d.png",
        title="30일 rolling beta2 (1일 step)",
    )

    plot_paths = [
        str(plots_dir / "chronological_beta2.png"),
        str(plots_dir / "session_beta2.png"),
        str(plots_dir / "session_1d_returns.png"),
        str(plots_dir / "volatility_1d_returns.png"),
        str(plots_dir / "rolling_beta2_30d.png"),
    ]
    report = build_baseline_robustness_report(
        regression_summary=regression_summary,
        best_summary=best_summary,
        focus_summary=focus_summary,
        rolling_summary=rolling_summary,
        plot_paths=plot_paths,
    )
    save_text(report, output_dir / "baseline_robustness_report.md")

    LOGGER.info(
        "baseline robustness 분석이 완료됐습니다. segment=%d, rolling_windows=%d.",
        len(regression_summary),
        len(rolling_summary),
    )


if __name__ == "__main__":
    main()

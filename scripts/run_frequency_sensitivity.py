from __future__ import annotations

import argparse
import logging
from pathlib import Path

from frequency_sensitivity import (
    build_frequency_sensitivity_report,
    load_aligned_return_panel,
    plot_standardized_beta2,
    run_frequency_sensitivity,
)
from utils import (
    load_config,
    save_config_snapshot,
    save_dataframe,
    save_input_manifest,
    save_provenance_manifest,
    save_text,
    setup_logging,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CSAD 빈도·시장 구성 민감도 분석을 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "research" / "frequency_sensitivity.yaml"),
        help="고정 분석 설정 파일 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))

    input_path = PROJECT_ROOT / config["data"]["input_path"]
    output_dir = PROJECT_ROOT / config["output"]["base_dir"]
    plot_path = output_dir / "plots" / "standardized_beta2_by_frequency.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("완전 정렬된 baseline 수익률 패널을 읽습니다: %s", input_path)
    return_panel = load_aligned_return_panel(
        input_path,
        expected_symbols=config["data"]["expected_symbols"],
        start=config["data"]["start"],
        end=config["data"]["end"],
    )
    LOGGER.info("빈도·시장 구성 12-cell 분석을 시작합니다. 1분 관측치=%d", len(return_panel))
    summary, comparison = run_frequency_sensitivity(
        return_panel,
        analysis_cfg=config["analysis"],
        regression_cfg=config["regression"],
        multiple_testing_cfg=config["multiple_testing"],
        decision_cfg=config["decision_rule"],
    )

    save_dataframe(summary, output_dir / "frequency_sensitivity_summary.csv", index=False)
    save_dataframe(comparison, output_dir / "frequency_universe_comparison.csv", index=False)
    plot_standardized_beta2(summary, plot_path)
    report = build_frequency_sensitivity_report(
        summary,
        comparison,
        config,
        input_path=config["data"]["input_path"],
        plot_path=plot_path.relative_to(PROJECT_ROOT).as_posix(),
    )
    save_text(report, output_dir / "frequency_sensitivity_report.md")
    save_config_snapshot(config, output_dir / "config_snapshot.yaml")
    input_manifest = save_input_manifest([input_path], output_dir / "input_manifest.json")
    save_provenance_manifest(
        config,
        output_dir / "provenance.json",
        schema_version=1,
        pipeline_version="csad-frequency-sensitivity-v1",
        train_start=config["data"]["start"],
        train_end=config["data"]["end"],
        statistical_method="12-cell HAC CSAD regressions with family-wide BH-FDR",
        input_manifest_path=input_manifest,
    )
    supported = summary.loc[
        summary["broad_herding_supported_at_frequency"], "frequency"
    ].drop_duplicates()
    LOGGER.info(
        "분석이 완료됐습니다. cells=%d, broad_herding_frequencies=%s",
        len(summary),
        ",".join(supported) if not supported.empty else "none",
    )


if __name__ == "__main__":
    main()

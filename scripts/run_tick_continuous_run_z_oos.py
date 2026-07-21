from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tick_continuous_run_z import (
    build_continuous_run_z_report,
    load_continuous_run_z_frame,
    plot_feature_correlations,
    plot_oos_coefficients,
    plot_split_comparison,
    prepare_continuous_run_z_frame,
    run_continuous_run_z_models,
    save_scaling_artifact,
)
from tick_raw_confirmatory import validate_confirmatory_raw_frame
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
    parser = argparse.ArgumentParser(
        description="연속형 run-z feature의 사전등록 개발·OOS 공동 회귀"
    )
    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT
            / "configs"
            / "tick"
            / "semantic_validation"
            / "continuous_run_z_oos.yaml"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))

    input_path = PROJECT_ROOT / config["data"]["input_path"]
    protocol_path = PROJECT_ROOT / config["protocol"]["path"]
    output_dir = PROJECT_ROOT / config["output"]["base_dir"]
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("사전등록 continuous run-z frame을 읽습니다: %s", input_path)
    frame = load_continuous_run_z_frame(input_path, config["data"]["symbols"])
    validate_confirmatory_raw_frame(frame, config["data"])
    prepared = prepare_continuous_run_z_frame(frame, config)
    (
        coefficients,
        diagnostics,
        symbol_coefficients,
        artifact,
        correlations,
        coverage,
    ) = run_continuous_run_z_models(prepared, config)

    save_dataframe(coverage, output_dir / "coverage.csv", index=False)
    save_dataframe(coefficients, output_dir / "pooled_coefficients.csv", index=False)
    save_dataframe(diagnostics, output_dir / "pooled_diagnostics.csv", index=False)
    save_dataframe(
        symbol_coefficients,
        output_dir / "symbol_descriptive_coefficients.csv",
        index=False,
    )
    save_dataframe(
        correlations,
        output_dir / "run_intensity_correlations.csv",
        index=False,
    )
    save_scaling_artifact(artifact, output_dir / "scaling_artifact.json")

    oos_plot = plots_dir / "oos_coefficients.png"
    split_plot = plots_dir / "development_oos_comparison.png"
    correlation_plot = plots_dir / "run_intensity_correlations.png"
    plot_oos_coefficients(coefficients, oos_plot)
    plot_split_comparison(coefficients, split_plot)
    plot_feature_correlations(correlations, correlation_plot)
    plot_paths = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (oos_plot, split_plot, correlation_plot)
    ]

    report = build_continuous_run_z_report(
        coverage,
        coefficients,
        diagnostics,
        correlations,
        artifact,
        config,
        plot_paths,
    )
    save_text(report, output_dir / "continuous_run_z_oos_report.md")
    save_config_snapshot(config, output_dir / "config_snapshot.yaml")
    save_text(protocol_path.read_text(encoding="utf-8"), output_dir / "protocol_snapshot.md")
    manifest = save_input_manifest(
        [input_path, protocol_path],
        output_dir / "input_manifest.json",
    )
    save_provenance_manifest(
        config,
        output_dir / "provenance.json",
        schema_version=2,
        pipeline_version="tick-continuous-run-z-oos-v1",
        statistical_method=(
            "joint run-intensity OLS with UTC-day clustered covariance; "
            "separate three-hypothesis BH-FDR families by split and outcome"
        ),
        input_manifest_path=manifest,
        train_start=config["split"]["development_start"],
        train_end=config["split"]["oos_start"],
        oos_start=config["split"]["oos_start"],
        oos_end=config["split"]["oos_end_exclusive"],
    )

    oos_passes = coefficients.loc[coefficients["is_primary_oos_pass"]]
    predictive_passes = oos_passes.loc[
        oos_passes["family"].isin(
            ["future_excess_return", "future_excess_abs_return"]
        )
    ]
    LOGGER.info(
        "continuous run-z OOS 완료: rows=%s, oos_passes=%d, predictive_passes=%d",
        f"{len(frame):,}",
        len(oos_passes),
        len(predictive_passes),
    )


if __name__ == "__main__":
    main()

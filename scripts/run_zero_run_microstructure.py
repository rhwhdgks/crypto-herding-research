from __future__ import annotations

import argparse
import logging
from pathlib import Path

from utils import (
    load_config,
    save_config_snapshot,
    save_dataframe,
    save_input_manifest,
    save_json,
    save_provenance_manifest,
    save_text,
    setup_logging,
)
from zero_run_microstructure import (
    FIXED_EFFECTS,
    FUTURE_CONTROLS,
    MECHANISM_CONTROLS,
    MECHANISM_OUTCOMES,
    apply_scaling_artifact,
    build_analysis_frame,
    build_artifact_manifest,
    build_decision_tables,
    build_sample_coverage,
    fit_scaling_artifact,
    load_tick_frame,
    run_future_lead_placebo,
    run_leave_one_asset_out,
    run_oos_prediction_metrics,
    run_permutation_tests,
    run_primary_models,
    save_scaling_artifact,
    validate_frozen_config,
    validate_inference_gates,
    validate_tick_frame,
)
from zero_run_reporting import (
    build_korean_report,
    plot_future_coefficients,
    plot_mechanism_coefficients,
    plot_permutation_comparison,
    plot_prediction_improvement,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-run intensity의 사전등록 시장 미시구조·미래 변동성 연구"
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/research/zero_run_microstructure_v1.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    validate_frozen_config(config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))

    tick_path = PROJECT_ROOT / config["data"]["tick_input_path"]
    protocol_path = PROJECT_ROOT / config["protocol"]["path"]
    output_dir = PROJECT_ROOT / config["output"]["base_dir"]
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    ohlcv_paths = {
        symbol: PROJECT_ROOT
        / config["data"]["ohlcv_path_template"].format(symbol=symbol)
        for symbol in config["data"]["symbols"]
    }

    LOGGER.info("동결된 2년 7자산 tick frame을 검증합니다: %s", tick_path)
    tick_frame = load_tick_frame(tick_path, config["data"]["symbols"])
    tick_coverage = validate_tick_frame(tick_frame, config["data"])

    LOGGER.info("1분봉에서 5·15·30분 비방향성 outcome을 구성합니다")
    analysis_frame, ohlcv_quality = build_analysis_frame(
        tick_frame, ohlcv_paths, config
    )
    scaling_artifact = fit_scaling_artifact(analysis_frame, config)
    analysis_frame = apply_scaling_artifact(analysis_frame, scaling_artifact)
    sample_coverage = build_sample_coverage(analysis_frame, config)

    LOGGER.info("개발·OOS UTC-day clustered fixed-effect 모형을 적합합니다")
    coefficients, diagnostics = run_primary_models(analysis_frame, config)
    validate_inference_gates(sample_coverage, diagnostics, config)

    LOGGER.info("개발→OOS 고정계수 예측 성능을 계산합니다")
    prediction_metrics = run_oos_prediction_metrics(analysis_frame, config)
    LOGGER.info("자산별 leave-one-asset-out 42개 모형을 적합합니다")
    loao_detail, loao_summary = run_leave_one_asset_out(
        analysis_frame, config, coefficients
    )
    LOGGER.info("OOS circular-shift permutation %d회를 실행합니다", config["analysis"]["permutation_repetitions"])
    permutation_detail, permutation_summary = run_permutation_tests(
        analysis_frame, config
    )
    LOGGER.info("7일 미래-lead placebo를 실행합니다")
    placebo = run_future_lead_placebo(analysis_frame, config)
    mechanism_decisions, future_decisions, family_decisions = build_decision_tables(
        coefficients,
        prediction_metrics,
        loao_summary,
        permutation_summary,
        placebo,
        config,
    )

    LOGGER.info("중간 산출물과 판정표를 저장합니다")
    analysis_frame.to_parquet(output_dir / "analysis_frame.parquet", index=False)
    save_dataframe(tick_coverage, output_dir / "tick_input_coverage.csv", index=False)
    save_dataframe(ohlcv_quality, output_dir / "ohlcv_input_quality.csv", index=False)
    save_dataframe(sample_coverage, output_dir / "sample_coverage.csv", index=False)
    save_dataframe(coefficients, output_dir / "clustered_coefficients.csv", index=False)
    save_dataframe(diagnostics, output_dir / "model_diagnostics.csv", index=False)
    save_dataframe(
        prediction_metrics, output_dir / "oos_prediction_metrics.csv", index=False
    )
    save_dataframe(loao_detail, output_dir / "loao_coefficients.csv", index=False)
    save_dataframe(loao_summary, output_dir / "loao_summary.csv", index=False)
    save_dataframe(
        permutation_detail, output_dir / "permutation_draws.csv", index=False
    )
    save_dataframe(
        permutation_summary, output_dir / "permutation_summary.csv", index=False
    )
    save_dataframe(placebo, output_dir / "future_lead_placebo.csv", index=False)
    save_dataframe(
        mechanism_decisions, output_dir / "mechanism_decisions.csv", index=False
    )
    save_dataframe(future_decisions, output_dir / "future_decisions.csv", index=False)
    save_dataframe(family_decisions, output_dir / "family_decisions.csv", index=False)
    save_scaling_artifact(scaling_artifact, output_dir / "scaling_artifact.json")

    mechanism_plot = plots_dir / "mechanism_coefficients.png"
    future_plot = plots_dir / "future_magnitude_coefficients.png"
    prediction_plot = plots_dir / "oos_prediction_improvement.png"
    permutation_plot = plots_dir / "permutation_null_comparison.png"
    plot_mechanism_coefficients(mechanism_decisions, mechanism_plot)
    plot_future_coefficients(future_decisions, future_plot)
    plot_prediction_improvement(prediction_metrics, prediction_plot)
    plot_permutation_comparison(permutation_summary, permutation_plot)
    plot_paths = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (
            mechanism_plot,
            future_plot,
            prediction_plot,
            permutation_plot,
        )
    ]

    report = build_korean_report(
        tick_coverage,
        ohlcv_quality,
        sample_coverage,
        coefficients,
        mechanism_decisions,
        future_decisions,
        family_decisions,
        prediction_metrics,
        loao_summary,
        permutation_summary,
        placebo,
        diagnostics,
        config,
        plot_paths,
    )
    save_text(report, output_dir / "zero_run_microstructure_report.md")
    save_config_snapshot(config, output_dir / "config_snapshot.yaml")
    save_text(
        protocol_path.read_text(encoding="utf-8"), output_dir / "protocol_snapshot.md"
    )
    input_paths = [config_path, protocol_path, tick_path, *ohlcv_paths.values()]
    input_manifest = save_input_manifest(
        input_paths, output_dir / "input_manifest.json"
    )
    save_provenance_manifest(
        config,
        output_dir / "provenance.json",
        schema_version=1,
        pipeline_version="zero-run-microstructure-v1",
        statistical_method=(
            "asset/hour/weekday fixed-effect OLS with UTC-day clustered covariance; "
            "split-family BH-FDR; LOAO; circular-shift permutation; future-lead placebo"
        ),
        input_manifest_path=input_manifest,
        random_seed=int(config["analysis"]["random_seed"]),
        train_start=config["split"]["development_start"],
        train_end=config["split"]["oos_start"],
        oos_start=config["split"]["oos_start"],
        oos_end=config["split"]["oos_end_exclusive"],
    )
    save_json(
        {
            "rows": int(len(analysis_frame)),
            "symbols": int(analysis_frame["symbol"].nunique()),
            "mechanism_family_success": bool(
                family_decisions.loc[
                    family_decisions["family"].eq("mechanism"), "family_success"
                ].iloc[0]
            ),
            "absolute_return_family_success": bool(
                family_decisions.loc[
                    family_decisions["family"].eq("future_absolute_return"),
                    "family_success",
                ].iloc[0]
            ),
            "realized_volatility_family_success": bool(
                family_decisions.loc[
                    family_decisions["family"].eq("future_realized_volatility"),
                    "family_success",
                ].iloc[0]
            ),
            "directional_alpha_tested": False,
            "tracker_activation_allowed": False,
        },
        output_dir / "run_summary.json",
    )
    save_json(
        {
            "focal_feature": "zero_run_intensity = -run_z_zero",
            "fixed_effects": FIXED_EFFECTS,
            "covariance": "UTC-day clustered with finite-sample correction",
            "mechanism_outcomes": MECHANISM_OUTCOMES,
            "mechanism_controls": MECHANISM_CONTROLS,
            "future_controls": FUTURE_CONTROLS,
            "future_outcome_families": {
                "absolute_return_bps": [
                    f"abs_forward_return_{horizon}m_bps"
                    for horizon in config["analysis"]["horizons_minutes"]
                ],
                "realized_volatility_bps": [
                    f"realized_volatility_{horizon}m_bps"
                    for horizon in config["analysis"]["horizons_minutes"]
                ],
            },
            "signed_future_return_outcome": None,
            "multiple_testing": "BH-FDR within split and preregistered family",
        },
        output_dir / "model_specification.json",
    )
    artifact_manifest = build_artifact_manifest(output_dir)
    save_dataframe(
        artifact_manifest, output_dir / "artifact_manifest.csv", index=False
    )

    LOGGER.info(
        "Zero-run 연구 완료: rows=%s mechanism=%s abs=%s rv=%s",
        f"{len(analysis_frame):,}",
        bool(family_decisions.loc[family_decisions["family"].eq("mechanism"), "family_success"].iloc[0]),
        bool(family_decisions.loc[family_decisions["family"].eq("future_absolute_return"), "family_success"].iloc[0]),
        bool(family_decisions.loc[family_decisions["family"].eq("future_realized_volatility"), "family_success"].iloc[0]),
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from tick_semantic_validation import (
    analyze_run_price_semantics,
    build_tick_semantic_report,
    load_tick_micro_frame,
    plot_contingency_heatmap,
    plot_predictive_coefficients,
    run_market_neutral_predictive_regression,
    summarize_raw_aggressor_pilot,
    summarize_schema_coverage,
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
    parser = argparse.ArgumentParser(description="Tick run-clustering의 의미 검증을 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "tick" / "semantic_validation" / "analysis.yaml"),
        help="고정 의미 검증 설정 파일",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    analysis_cfg = config["analysis"]
    symbols = list(config["data"]["symbols"])
    migrated_path = PROJECT_ROOT / config["data"]["migrated_cache_path"]
    raw_pilot_path = PROJECT_ROOT / config["data"]["raw_pilot_path"]
    output_dir = PROJECT_ROOT / config["output"]["base_dir"]
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("5년 migrated v2 cache를 읽습니다: %s", migrated_path)
    migrated = load_tick_micro_frame(migrated_path, symbols)
    price_summary, price_contingency = analyze_run_price_semantics(
        migrated,
        expected_symbols=symbols,
        minimum_events=int(analysis_cfg["minimum_directional_events"]),
        family_size=int(analysis_cfg["price_direction_family_size"]),
        fdr_alpha=float(analysis_cfg["fdr_alpha"]),
        proxy_minimum_concordance=float(analysis_cfg["directional_proxy_minimum_concordance"]),
    )
    predictive, predictive_diagnostics = run_market_neutral_predictive_regression(
        migrated,
        expected_symbols=symbols,
        trailing_volatility_buckets=int(analysis_cfg["trailing_volatility_buckets"]),
        family_size=int(analysis_cfg["predictive_family_size"]),
        fdr_alpha=float(analysis_cfg["fdr_alpha"]),
        economic_effect_threshold_bps=float(analysis_cfg["economic_effect_threshold_bps"]),
    )

    input_paths = [migrated_path]
    coverage_frames = [summarize_schema_coverage(migrated, "5년 migrated v2 cache")]
    raw_summary = pd.DataFrame()
    raw_contingency = pd.DataFrame()
    if raw_pilot_path.is_file():
        LOGGER.info("raw aggressor pilot을 읽습니다: %s", raw_pilot_path)
        raw_pilot = load_tick_micro_frame(raw_pilot_path, symbols)
        raw_summary, raw_contingency = summarize_raw_aggressor_pilot(raw_pilot, symbols)
        coverage_frames.append(summarize_schema_coverage(raw_pilot, "2024-04 raw pilot"))
        input_paths.append(raw_pilot_path)
    else:
        LOGGER.warning("raw pilot 파일이 없어 aggressor 결과를 N/A로 둡니다: %s", raw_pilot_path)

    schema_coverage = pd.concat(coverage_frames, ignore_index=True)
    save_dataframe(schema_coverage, output_dir / "schema_coverage.csv", index=False)
    save_dataframe(price_summary, output_dir / "run_price_association.csv", index=False)
    save_dataframe(price_contingency, output_dir / "run_price_contingency.csv", index=False)
    save_dataframe(predictive, output_dir / "market_neutral_predictive_coefficients.csv", index=False)
    save_dataframe(predictive_diagnostics, output_dir / "market_neutral_predictive_diagnostics.csv", index=False)
    save_dataframe(raw_summary, output_dir / "raw_aggressor_pilot_summary.csv", index=False)
    save_dataframe(raw_contingency, output_dir / "raw_aggressor_pilot_contingency.csv", index=False)

    price_plot = plots_dir / "run_side_vs_price_direction.png"
    predictive_plot = plots_dir / "market_neutral_predictive_coefficients.png"
    plot_contingency_heatmap(
        price_contingency,
        column_name="price_direction",
        path=price_plot,
        title="5년 event: run side별 가격 방향 분포",
    )
    plot_predictive_coefficients(predictive, predictive_plot)
    plot_paths = [
        price_plot.relative_to(PROJECT_ROOT).as_posix(),
        predictive_plot.relative_to(PROJECT_ROOT).as_posix(),
    ]
    if not raw_contingency.empty:
        raw_plot = plots_dir / "run_side_vs_aggressor_direction_raw_pilot.png"
        plot_contingency_heatmap(
            raw_contingency,
            column_name="aggressor_direction",
            path=raw_plot,
            title="2024-04 raw pilot: run side별 aggressor 방향",
        )
        plot_paths.append(raw_plot.relative_to(PROJECT_ROOT).as_posix())

    report = build_tick_semantic_report(
        schema_coverage,
        price_summary,
        predictive,
        predictive_diagnostics,
        raw_summary,
        config,
        plot_paths,
    )
    save_text(report, output_dir / "tick_semantic_validation_report.md")
    save_config_snapshot(config, output_dir / "config_snapshot.yaml")
    input_manifest = save_input_manifest(input_paths, output_dir / "input_manifest.json")
    save_provenance_manifest(
        config,
        output_dir / "provenance.json",
        schema_version=2,
        pipeline_version="tick-semantic-validation-v1",
        statistical_method="8-test directional BH-FDR; 3-test UTC-day clustered market-neutral OLS BH-FDR",
        input_manifest_path=input_manifest,
    )
    LOGGER.info(
        "Tick 의미 검증이 완료됐습니다. price_proxy=%s, predictive_survivors=%d",
        bool(price_summary.loc[price_summary["scope"] == "pooled", "supports_price_direction_proxy"].iloc[0]),
        int(predictive["supports_economic_predictive_association"].sum()),
    )


if __name__ == "__main__":
    main()

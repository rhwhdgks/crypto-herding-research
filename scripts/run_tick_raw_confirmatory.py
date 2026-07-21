from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tick_raw_confirmatory import build_raw_confirmatory_report, validate_confirmatory_raw_frame
from tick_semantic_validation import (
    analyze_run_aggressor_semantics,
    analyze_run_price_semantics,
    load_tick_micro_frame,
    plot_contingency_heatmap,
    plot_predictive_coefficients,
    run_market_neutral_predictive_regression,
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
    parser = argparse.ArgumentParser(description="동일 2년 raw tick confirmatory 검정")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "tick" / "semantic_validation" / "confirmatory_2y.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    symbols = list(config["data"]["symbols"])
    analysis = config["analysis"]
    input_path = PROJECT_ROOT / config["data"]["input_path"]
    output_dir = PROJECT_ROOT / config["output"]["base_dir"]
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("동일 2년 raw micro frame을 읽습니다: %s", input_path)
    frame = load_tick_micro_frame(input_path, symbols)
    coverage = validate_confirmatory_raw_frame(frame, config["data"])
    price_summary, price_table = analyze_run_price_semantics(
        frame,
        symbols,
        int(analysis["minimum_directional_events"]),
        int(analysis["price_direction_family_size"]),
        float(analysis["fdr_alpha"]),
        float(analysis["price_proxy_minimum_concordance"]),
    )
    aggressor_summary, aggressor_table = analyze_run_aggressor_semantics(
        frame,
        symbols,
        int(analysis["minimum_directional_events"]),
        int(analysis["aggressor_direction_family_size"]),
        float(analysis["fdr_alpha"]),
        float(analysis["aggressor_proxy_minimum_concordance"]),
    )
    predictive, diagnostics = run_market_neutral_predictive_regression(
        frame,
        symbols,
        int(analysis["trailing_volatility_buckets"]),
        int(analysis["predictive_family_size"]),
        float(analysis["fdr_alpha"]),
        float(analysis["economic_effect_threshold_bps"]),
    )
    save_dataframe(coverage, output_dir / "coverage.csv", index=False)
    save_dataframe(price_summary, output_dir / "run_price_association.csv", index=False)
    save_dataframe(price_table, output_dir / "run_price_contingency.csv", index=False)
    save_dataframe(aggressor_summary, output_dir / "run_aggressor_association.csv", index=False)
    save_dataframe(aggressor_table, output_dir / "run_aggressor_contingency.csv", index=False)
    save_dataframe(predictive, output_dir / "market_neutral_predictive_coefficients.csv", index=False)
    save_dataframe(diagnostics, output_dir / "market_neutral_predictive_diagnostics.csv", index=False)

    price_plot = plots_dir / "run_side_vs_price_direction.png"
    aggressor_plot = plots_dir / "run_side_vs_aggressor_direction.png"
    predictive_plot = plots_dir / "market_neutral_predictive_coefficients.png"
    plot_contingency_heatmap(price_table, "price_direction", price_plot, "동일 2년 raw: run side별 가격 방향")
    plot_contingency_heatmap(
        aggressor_table,
        "aggressor_direction",
        aggressor_plot,
        "동일 2년 raw: run side별 aggressor 방향",
    )
    plot_predictive_coefficients(predictive, predictive_plot)
    plot_paths = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in [price_plot, aggressor_plot, predictive_plot]
    ]
    report = build_raw_confirmatory_report(
        coverage,
        price_summary,
        aggressor_summary,
        predictive,
        diagnostics,
        config,
        plot_paths,
    )
    save_text(report, output_dir / "tick_raw_confirmatory_report.md")
    save_config_snapshot(config, output_dir / "config_snapshot.yaml")
    manifest = save_input_manifest([input_path], output_dir / "input_manifest.json")
    save_provenance_manifest(
        config,
        output_dir / "provenance.json",
        schema_version=2,
        pipeline_version="tick-raw-confirmatory-v1",
        statistical_method="separate 8-price, 8-aggressor, 3-predictive BH-FDR families",
        input_manifest_path=manifest,
        train_start=config["data"]["expected_start"],
        train_end=config["data"]["expected_end"],
    )
    LOGGER.info(
        "confirmatory 완료: price_proxy=%s aggressor_proxy=%s predictive=%d",
        bool(price_summary.loc[price_summary["scope"] == "pooled", "supports_price_direction_proxy"].iloc[0]),
        bool(aggressor_summary.loc[aggressor_summary["scope"] == "pooled", "supports_aggressor_direction_proxy"].iloc[0]),
        int(predictive["supports_economic_predictive_association"].sum()),
    )


if __name__ == "__main__":
    main()

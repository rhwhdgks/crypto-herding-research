from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

from final_research_reporting import (
    build_mechanical_null_report,
    plot_empirical_vs_null,
    plot_mechanism_null,
    plot_null_fpr,
)
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
from zero_run_mechanical_null import (
    add_frozen_strata,
    analyze_group_null_calibration,
    analyze_mechanism_null,
    build_audit_decisions,
    draw_stratified_audit_sample,
    load_audit_frame,
    sample_exact_conditional_run_z,
    validate_frozen_config,
    verify_preregistration_seal,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-run 조건부 배열 null 기계성 감사")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs/research/zero_run_mechanical_null_audit_v1.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    validate_frozen_config(config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))

    protocol_path = PROJECT_ROOT / config["protocol"]["path"]
    seal_path = protocol_path.with_suffix(".seal.json")
    seal = verify_preregistration_seal(protocol_path, config_path, seal_path)
    output_dir = PROJECT_ROOT / config["output"]["base_dir"]
    if (output_dir / "mechanical_null_decisions.csv").exists():
        raise FileExistsError(
            "Frozen mechanical-null results already exist; refusing to overwrite an observed audit"
        )
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    tick_path = PROJECT_ROOT / config["data"]["tick_input_path"]
    analysis_path = PROJECT_ROOT / config["data"]["zero_run_analysis_path"]
    scaler_path = PROJECT_ROOT / config["data"]["zero_run_scaler_path"]
    prior_mechanism_path = PROJECT_ROOT / config["data"]["zero_run_mechanism_decisions_path"]

    LOGGER.info("봉인된 protocol/config SHA-256을 확인했습니다")
    frame, integrity = load_audit_frame(tick_path, analysis_path, config)
    frame, boundaries = add_frozen_strata(
        frame, int(config["analysis"]["stratification_quantiles"])
    )
    sample, strata_summary = draw_stratified_audit_sample(
        frame,
        int(config["analysis"]["maximum_rows_per_asset_tx_zero_stratum"]),
        int(config["analysis"]["random_seed"]),
    )
    LOGGER.info(
        "OOS %d개 중 층화표본 %d개에서 exact conditional null %d회를 생성합니다",
        len(frame),
        len(sample),
        int(config["analysis"]["monte_carlo_repetitions"]),
    )
    null_run_z, null_diagnostics = sample_exact_conditional_run_z(
        sample["transaction_count"].to_numpy(),
        sample["zero_ticks"].to_numpy(),
        int(config["analysis"]["monte_carlo_repetitions"]),
        int(config["analysis"]["random_seed"]),
    )
    group_summary, group_replicates = analyze_group_null_calibration(
        sample, null_run_z, config
    )
    mechanism_summary, mechanism_draws, model_diagnostics = analyze_mechanism_null(
        sample, null_run_z, scaler_path, prior_mechanism_path, config
    )
    decisions = build_audit_decisions(group_summary, mechanism_summary, config)

    save_dataframe(integrity, output_dir / "input_integrity.csv", index=False)
    save_dataframe(boundaries, output_dir / "strata_boundaries.csv", index=False)
    save_dataframe(strata_summary, output_dir / "sampling_strata_summary.csv", index=False)
    sample_keys = sample[
        [
            "symbol",
            "bucket_start",
            "transaction_count_quintile",
            "zero_tick_share_quintile",
            "liquidity_quintile",
            "transaction_count",
            "zero_ticks",
            "zero_runs",
            "sampling_weight",
        ]
    ]
    save_dataframe(sample_keys, output_dir / "stratified_sample_keys.csv", index=False)
    save_json(asdict(null_diagnostics), output_dir / "null_sampling_diagnostics.json")
    save_dataframe(group_summary, output_dir / "empirical_null_group_comparison.csv", index=False)
    group_replicates.to_parquet(output_dir / "null_group_replicates.parquet", index=False)
    save_dataframe(mechanism_summary, output_dir / "mechanism_null_comparison.csv", index=False)
    mechanism_draws.to_parquet(output_dir / "mechanism_null_coefficients.parquet", index=False)
    save_dataframe(model_diagnostics, output_dir / "mechanism_model_diagnostics.csv", index=False)
    save_dataframe(decisions, output_dir / "mechanical_null_decisions.csv", index=False)

    fpr_plot = plots_dir / "conditional_null_fpr.png"
    empirical_plot = plots_dir / "empirical_vs_conditional_null.png"
    mechanism_plot = plots_dir / "mechanism_vs_conditional_null.png"
    plot_null_fpr(group_summary, fpr_plot)
    plot_empirical_vs_null(group_summary, empirical_plot)
    plot_mechanism_null(mechanism_summary, mechanism_plot)
    plot_paths = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in [fpr_plot, empirical_plot, mechanism_plot]
    ]
    report = build_mechanical_null_report(
        group_summary,
        mechanism_summary,
        decisions,
        integrity,
        strata_summary,
        asdict(null_diagnostics),
        config,
        plot_paths,
    )
    save_text(report, output_dir / "mechanical_null_audit_report.md")
    save_config_snapshot(config, output_dir / "mechanical_null_config_snapshot.yaml")
    save_text(protocol_path.read_text(encoding="utf-8"), output_dir / "mechanical_null_protocol_snapshot.md")
    save_json(seal, output_dir / "preregistration_seal_verification.json")
    input_manifest = save_input_manifest(
        [config_path, protocol_path, seal_path, tick_path, analysis_path, scaler_path, prior_mechanism_path],
        output_dir / "mechanical_null_input_manifest.json",
    )
    save_provenance_manifest(
        config,
        output_dir / "mechanical_null_provenance.json",
        schema_version=1,
        pipeline_version="zero-run-mechanical-null-audit-v1",
        statistical_method=(
            "exact conditional binary-arrangement run distribution; stratified WLS; "
            "999 Monte Carlo draws; 23-group and 5-outcome BH-FDR"
        ),
        input_manifest_path=input_manifest,
        random_seed=int(config["analysis"]["random_seed"]),
        train_start=None,
        train_end=None,
        oos_start=config["data"]["expected_oos_start"],
        oos_end=config["data"]["expected_oos_end_exclusive"],
    )
    final = decisions.loc[decisions["decision"].eq("final_mechanical_null_audit")].iloc[0]
    save_json(
        {
            "population_rows": int(len(frame)),
            "audit_sample_rows": int(len(sample)),
            "strata": int(len(strata_summary)),
            "null_repetitions": int(null_diagnostics.repetitions),
            "group_tests_per_family": int(len(group_summary)),
            "mechanism_tests": int(len(mechanism_summary)),
            "final_passed": bool(final["passed"]),
            "final_classification": str(final["classification"]),
            "directional_alpha_tested": False,
            "tracker_activation_allowed": False,
        },
        output_dir / "mechanical_null_run_summary.json",
    )
    LOGGER.info(
        "기계성 감사 완료: sample=%d, classification=%s",
        len(sample),
        final["classification"],
    )


if __name__ == "__main__":
    main()

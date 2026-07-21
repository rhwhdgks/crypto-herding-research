from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd

from cmc_fixed_universe import (
    build_asset_coverage,
    build_fixed_history_manifest,
    build_fixed_panels,
    collect_fixed_universe_history,
    collection_status,
    run_fixed_regressions,
    validate_fixed_quality,
)
from cmc_temporal_validation import (
    build_temporal_validation_report,
    compare_historical_and_holdout,
    evaluate_temporal_persistence,
    plot_temporal_comparison,
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
    parser = argparse.ArgumentParser(description="CMC fixed-62 시간 외부표본 검증을 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "cmc_fixed_62" / "temporal_validation_v1.yaml"),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--collect-only", action="store_true")
    mode.add_argument("--analysis-only", action="store_true")
    mode.add_argument("--status", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, universe = _load_runtime_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    if args.status:
        print(json.dumps(collection_status(config["source"], universe), ensure_ascii=False, indent=2))
        return
    if not args.analysis_only:
        manifest, history = collect_fixed_universe_history(config["source"], universe)
        LOGGER.info("Temporal fixed-62 수집 완료: checkpoints=%d rows=%d", len(manifest), len(history))
        if args.collect_only:
            return
    run_analysis(config, universe)


def run_analysis(config: dict, universe: list[dict]) -> None:
    source_cfg = config["source"]
    analysis_cfg = config["analysis"]
    output_dir = Path(config["output"]["base_dir"])
    intermediate_dir = output_dir / "intermediate"
    plots_dir = output_dir / "plots"
    for path in (output_dir, intermediate_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest, history = build_fixed_history_manifest(source_cfg, universe)
    asset_coverage = build_asset_coverage(history, universe, source_cfg)
    panels = {
        variant["name"]: build_fixed_panels(history, variant, analysis_cfg)
        for variant in analysis_cfg["variants"]
    }
    quality = validate_fixed_quality(
        manifest, history, asset_coverage, panels, source_cfg, analysis_cfg, universe
    )
    targets, coefficients, diagnostics = run_fixed_regressions(panels, analysis_cfg)
    decision_detail, decision_summary = evaluate_temporal_persistence(
        targets, config["decision"]
    )
    historical_targets = pd.read_csv(config["historical_comparison"]["targets_path"])
    comparison = compare_historical_and_holdout(
        historical_targets, targets, config["historical_comparison"]
    )

    save_dataframe(manifest, output_dir / "checkpoint_manifest.csv", index=False)
    save_dataframe(asset_coverage, output_dir / "asset_coverage.csv", index=False)
    save_dataframe(quality, output_dir / "data_quality_checks.csv", index=False)
    save_dataframe(targets, output_dir / "regression_targets.csv", index=False)
    save_dataframe(coefficients, output_dir / "regression_coefficients.csv", index=False)
    save_dataframe(diagnostics, output_dir / "regression_diagnostics.csv", index=False)
    save_dataframe(decision_detail, output_dir / "persistence_decision_detail.csv", index=False)
    save_dataframe(decision_summary, output_dir / "persistence_decision_summary.csv", index=False)
    save_dataframe(comparison, output_dir / "historical_holdout_comparison.csv", index=False)

    for variant_name, result in panels.items():
        variant_dir = intermediate_dir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        for frequency in ("daily", "weekly"):
            save_dataframe(result[f"{frequency}_market"], output_dir / f"{frequency}_market_return_{variant_name}.csv")
            save_dataframe(result[f"{frequency}_csad"], output_dir / f"{frequency}_csad_{variant_name}.csv")
            save_dataframe(result[f"{frequency}_coverage"], output_dir / f"{frequency}_coverage_{variant_name}.csv", index=False)
            result[f"{frequency}_rows"].to_parquet(variant_dir / f"{frequency}_member_rows.parquet", index=False)
            result[f"{frequency}_panel"].to_parquet(variant_dir / f"{frequency}_return_panel.parquet")

    plot_path = plots_dir / "historical_vs_holdout_coefficients.png"
    plot_temporal_comparison(comparison, plot_path)
    relative_plots = [plot_path.relative_to(PROJECT_ROOT).as_posix()]
    report = build_temporal_validation_report(
        config, asset_coverage, quality, panels, targets, decision_summary, comparison, relative_plots
    )
    save_text(report, output_dir / "cmc_fixed_62_temporal_validation_report.md")
    save_config_snapshot(config, output_dir / "config_snapshot.yaml")
    shutil.copy2(config["protocol"]["path"], output_dir / "protocol_snapshot.md")
    input_manifest = save_input_manifest(
        [
            source_cfg["manifest_path"],
            source_cfg["normalized_path"],
            config["protocol"]["path"],
            config["universe"]["source_config_path"],
            config["historical_comparison"]["targets_path"],
        ],
        output_dir / "input_manifest.json",
    )
    save_provenance_manifest(
        config,
        output_dir / "provenance.json",
        schema_version=2,
        pipeline_version="cmc-fixed-62-temporal-validation-v1",
        train_start=analysis_cfg["start"],
        train_end=analysis_cfg["end"],
        statistical_method="Temporal holdout daily-weekly HAC corrected CSAD with variant-period BH-FDR",
        input_manifest_path=input_manifest,
    )
    LOGGER.info(
        "CMC fixed-62 시간 외부검증 완료: output=%s primary_pass=%s",
        output_dir,
        bool(decision_summary.set_index("variant").loc["replication_primary", "all_required_cells_pass"]),
    )


def _load_runtime_config(path: str | Path) -> tuple[dict, list[dict]]:
    config = load_config(path)
    universe_path = _project_path(config["universe"]["source_config_path"])
    universe_config = load_config(universe_path)
    universe = list(universe_config["universe"])
    if len(universe) != int(config["universe"]["expected_assets"]):
        raise ValueError("Temporal validation universe count does not match its frozen config")
    resolved = dict(config)
    resolved["source"] = dict(config["source"])
    for key in ("cache_dir", "state_path", "manifest_path", "normalized_path"):
        resolved["source"][key] = str(_project_path(config["source"][key]))
    resolved["protocol"] = dict(config["protocol"])
    resolved["protocol"]["path"] = str(_project_path(config["protocol"]["path"]))
    resolved["universe"] = dict(config["universe"])
    resolved["universe"]["source_config_path"] = str(universe_path)
    resolved["historical_comparison"] = dict(config["historical_comparison"])
    resolved["historical_comparison"]["targets_path"] = str(
        _project_path(config["historical_comparison"]["targets_path"])
    )
    resolved["output"] = dict(config["output"])
    resolved["output"]["base_dir"] = str(_project_path(config["output"]["base_dir"]))
    return resolved, universe


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()

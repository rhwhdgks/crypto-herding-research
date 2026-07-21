from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd

from cmc_fixed_universe import (
    build_asset_coverage,
    build_benchmark_comparison,
    build_fixed_history_manifest,
    build_method_audit_summary,
    build_fixed_panels,
    build_fixed_report,
    collect_fixed_universe_history,
    collection_status,
    plot_fixed_results,
    run_fixed_regressions,
    validate_fixed_quality,
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
    parser = argparse.ArgumentParser(description="CMC 고정 62종목 선행논문 재현을 실행합니다.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "cmc_fixed_62" / "replication_v1.yaml"),
        help="동결된 fixed-62 연구 설정 YAML",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--collect-only", action="store_true", help="raw checkpoint만 수집")
    mode.add_argument("--analysis-only", action="store_true", help="기존 checkpoint로만 분석")
    mode.add_argument("--status", action="store_true", help="수집 진행률만 표시")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _resolve_runtime_paths(load_config(args.config))
    setup_logging(config.get("logging", {}).get("level", "INFO"))

    if args.status:
        print(json.dumps(collection_status(config["source"], config["universe"]), ensure_ascii=False, indent=2))
        return

    if not args.analysis_only:
        manifest, history = collect_fixed_universe_history(config["source"], config["universe"])
        LOGGER.info("CMC fixed-62 수집 완료: checkpoints=%d rows=%d", len(manifest), len(history))
        if args.collect_only:
            return
    run_analysis(config)


def run_analysis(config: dict) -> None:
    source_cfg = config["source"]
    analysis_cfg = config["analysis"]
    output_dir = Path(config["output"]["base_dir"])
    intermediate_dir = output_dir / "intermediate"
    plots_dir = output_dir / "plots"
    for path in (output_dir, intermediate_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest, history = build_fixed_history_manifest(source_cfg, config["universe"])
    asset_coverage = build_asset_coverage(history, config["universe"], source_cfg)
    panels_by_variant = {
        variant["name"]: build_fixed_panels(history, variant, analysis_cfg)
        for variant in analysis_cfg["variants"]
    }
    method_panels = {
        variant["name"]: build_fixed_panels(history, variant, analysis_cfg)
        for variant in analysis_cfg["method_audit_variants"]
    }
    quality = validate_fixed_quality(
        manifest,
        history,
        asset_coverage,
        panels_by_variant,
        source_cfg,
        analysis_cfg,
        config["universe"],
    )
    targets, coefficients, diagnostics = run_fixed_regressions(panels_by_variant, analysis_cfg)
    comparison = build_benchmark_comparison(targets, config["benchmark"])
    method_targets, _, _ = run_fixed_regressions(method_panels, analysis_cfg)
    method_audit_summary = build_method_audit_summary(method_targets, config["benchmark"])

    save_dataframe(manifest, output_dir / "checkpoint_manifest.csv", index=False)
    save_dataframe(asset_coverage, output_dir / "asset_coverage.csv", index=False)
    save_dataframe(quality, output_dir / "data_quality_checks.csv", index=False)
    save_dataframe(targets, output_dir / "regression_targets.csv", index=False)
    save_dataframe(coefficients, output_dir / "regression_coefficients.csv", index=False)
    save_dataframe(diagnostics, output_dir / "regression_diagnostics.csv", index=False)
    save_dataframe(comparison, output_dir / "paper_benchmark_comparison.csv", index=False)
    save_dataframe(method_targets, output_dir / "method_audit_regression_targets.csv", index=False)
    save_dataframe(method_audit_summary, output_dir / "method_audit_summary.csv", index=False)

    for variant_name, panels in panels_by_variant.items():
        variant_dir = intermediate_dir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        save_dataframe(panels["daily_market"], output_dir / f"daily_market_return_{variant_name}.csv")
        save_dataframe(panels["daily_csad"], output_dir / f"daily_csad_{variant_name}.csv")
        save_dataframe(panels["weekly_market"], output_dir / f"weekly_market_return_{variant_name}.csv")
        save_dataframe(panels["weekly_csad"], output_dir / f"weekly_csad_{variant_name}.csv")
        save_dataframe(panels["daily_coverage"], output_dir / f"daily_coverage_{variant_name}.csv", index=False)
        save_dataframe(panels["weekly_coverage"], output_dir / f"weekly_coverage_{variant_name}.csv", index=False)
        panels["daily_rows"].to_parquet(variant_dir / "daily_member_rows.parquet", index=False)
        panels["daily_panel"].to_parquet(variant_dir / "daily_return_panel.parquet")
        panels["weekly_rows"].to_parquet(variant_dir / "weekly_member_rows.parquet", index=False)
        panels["weekly_panel"].to_parquet(variant_dir / "weekly_return_panel.parquet")

    plot_paths = [plots_dir / "daily_active_assets.png", plots_dir / "target_coefficients.png"]
    plot_fixed_results(panels_by_variant, targets, *plot_paths)
    relative_plots = [path.relative_to(PROJECT_ROOT).as_posix() for path in plot_paths]
    report = build_fixed_report(
        config,
        asset_coverage,
        quality,
        panels_by_variant,
        targets,
        comparison,
        method_audit_summary,
        relative_plots,
    )
    save_text(report, output_dir / "cmc_fixed_62_replication_report.md")
    save_config_snapshot(config, output_dir / "config_snapshot.yaml")
    shutil.copy2(config["protocol"]["path"], output_dir / "protocol_snapshot.md")
    input_manifest = save_input_manifest(
        [source_cfg["manifest_path"], source_cfg["normalized_path"], config["protocol"]["path"]],
        output_dir / "input_manifest.json",
    )
    save_provenance_manifest(
        config,
        output_dir / "provenance.json",
        schema_version=2,
        pipeline_version="cmc-fixed-62-replication-v1",
        train_start=analysis_cfg["start"],
        train_end=analysis_cfg["end"],
        statistical_method="Fixed-62 daily and weekly HAC CSAD with variant-period six-test BH-FDR",
        input_manifest_path=input_manifest,
    )
    LOGGER.info(
        "CMC fixed-62 재현 완료: daily=%d weekly=%d output=%s",
        len(panels_by_variant["replication_primary"]["daily_market"]),
        len(panels_by_variant["replication_primary"]["weekly_market"]),
        output_dir,
    )


def _resolve_runtime_paths(config: dict) -> dict:
    resolved = dict(config)
    resolved["source"] = dict(config["source"])
    for key in ("cache_dir", "state_path", "manifest_path", "normalized_path"):
        resolved["source"][key] = str(_project_path(config["source"][key]))
    resolved["protocol"] = dict(config["protocol"])
    resolved["protocol"]["path"] = str(_project_path(config["protocol"]["path"]))
    resolved["output"] = dict(config["output"])
    resolved["output"]["base_dir"] = str(_project_path(config["output"]["base_dir"]))
    return resolved


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()

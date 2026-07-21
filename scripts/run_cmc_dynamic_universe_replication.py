from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd

from cmc_dynamic_universe import (
    build_cmc_dynamic_report,
    build_daily_research_panel,
    build_monthly_dynamic_universe,
    build_paper_benchmark_comparison,
    build_snapshot_manifest,
    build_weekly_research_panel,
    collect_cmc_snapshots,
    collect_current_metadata,
    collection_status,
    load_snapshot_history,
    plot_csad_series,
    plot_target_coefficients,
    plot_universe_diagnostics,
    run_dynamic_csad_regressions,
    validate_analysis_quality,
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
    parser = argparse.ArgumentParser(
        description="CMC point-in-time 동적 universe CSAD 재현을 실행합니다."
    )
    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT / "configs" / "cmc_dynamic_universe" / "replication_v1.yaml"
        ),
        help="동결된 연구 설정 YAML 경로",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--collect-only", action="store_true", help="checkpoint만 수집")
    mode.add_argument("--analysis-only", action="store_true", help="기존 checkpoint로만 분석")
    mode.add_argument("--status", action="store_true", help="수집 진행률만 표시")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    runtime_config = _resolve_runtime_paths(config)
    source_cfg = runtime_config["source"]
    excluded_tags = runtime_config["universe"]["excluded_metadata_tags"]

    if args.status:
        print(json.dumps(collection_status(source_cfg), ensure_ascii=False, indent=2))
        return

    if not args.analysis_only:
        manifest, metadata = collect_cmc_snapshots(
            source_cfg,
            excluded_metadata_tags=excluded_tags,
        )
        LOGGER.info(
            "CMC 수집이 완료됐습니다. snapshots=%d, metadata=%d",
            len(manifest),
            len(metadata),
        )
        if args.collect_only:
            return

    run_analysis(runtime_config)


def run_analysis(config: dict) -> None:
    source_cfg = config["source"]
    universe_cfg = config["universe"]
    analysis_cfg = config["analysis"]
    output_dir = Path(config["output"]["base_dir"])
    intermediate_dir = output_dir / "intermediate"
    plots_dir = output_dir / "plots"
    for path in (output_dir, intermediate_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("CMC checkpoint를 검증하고 전체 history를 로드합니다.")
    manifest = build_snapshot_manifest(source_cfg)
    metadata = collect_current_metadata(
        source_cfg,
        excluded_metadata_tags=universe_cfg["excluded_metadata_tags"],
    )
    snapshots = load_snapshot_history(source_cfg)

    LOGGER.info("전월 말 Top-%d universe를 구성합니다.", universe_cfg["top_n"])
    membership, formation_audit, turnover = build_monthly_dynamic_universe(
        snapshots,
        metadata,
        universe_cfg,
        analysis_cfg,
    )
    daily_rows, daily_panel, daily_market, daily_csad, daily_coverage = (
        build_daily_research_panel(snapshots, membership, analysis_cfg)
    )
    weekly_rows, weekly_panel, weekly_market, weekly_csad, weekly_coverage = (
        build_weekly_research_panel(daily_rows, analysis_cfg)
    )
    _validate_unique_daily_ids(daily_rows)
    quality_checks = validate_analysis_quality(
        manifest,
        membership,
        daily_coverage,
        source_cfg,
        analysis_cfg,
    )

    LOGGER.info(
        "품질 gate를 통과했습니다. daily=%d, weekly=%d",
        int(daily_coverage["eligible_day"].sum()),
        int(weekly_coverage["eligible_week"].sum()),
    )
    targets, coefficients, diagnostics = run_dynamic_csad_regressions(
        {
            "daily": (daily_csad, daily_market),
            "weekly": (weekly_csad, weekly_market),
        },
        analysis_cfg,
    )
    benchmark_comparison = build_paper_benchmark_comparison(
        targets,
        config["benchmark"],
    )

    output_manifest = output_dir / "snapshot_manifest.csv"
    save_dataframe(manifest, output_manifest, index=False)
    save_dataframe(metadata, output_dir / "current_metadata.csv", index=False)
    save_dataframe(membership, output_dir / "monthly_membership.csv", index=False)
    save_dataframe(formation_audit, output_dir / "monthly_formation_audit.csv", index=False)
    save_dataframe(turnover, output_dir / "universe_turnover.csv", index=False)
    save_dataframe(daily_coverage, output_dir / "daily_coverage.csv", index=False)
    save_dataframe(weekly_coverage, output_dir / "weekly_coverage.csv", index=False)
    save_dataframe(quality_checks, output_dir / "data_quality_checks.csv", index=False)
    save_dataframe(daily_market, output_dir / "daily_market_return.csv")
    save_dataframe(daily_csad, output_dir / "daily_csad.csv")
    save_dataframe(weekly_market, output_dir / "weekly_market_return.csv")
    save_dataframe(weekly_csad, output_dir / "weekly_csad.csv")
    save_dataframe(targets, output_dir / "regression_targets.csv", index=False)
    save_dataframe(coefficients, output_dir / "regression_coefficients.csv", index=False)
    save_dataframe(diagnostics, output_dir / "regression_diagnostics.csv", index=False)
    save_dataframe(
        benchmark_comparison,
        output_dir / "paper_benchmark_comparison.csv",
        index=False,
    )

    daily_rows.to_parquet(intermediate_dir / "daily_member_rows.parquet", index=False)
    daily_panel.to_parquet(intermediate_dir / "daily_return_panel.parquet")
    weekly_rows.to_parquet(intermediate_dir / "weekly_member_rows.parquet", index=False)
    weekly_panel.to_parquet(intermediate_dir / "weekly_return_panel.parquet")

    plot_paths = [
        plots_dir / "universe_turnover.png",
        plots_dir / "daily_csad_vs_market.png",
        plots_dir / "target_coefficients.png",
    ]
    plot_universe_diagnostics(turnover, plot_paths[0])
    plot_csad_series(daily_csad, daily_market, plot_paths[1])
    plot_target_coefficients(targets, plot_paths[2])

    report = build_cmc_dynamic_report(
        config,
        manifest,
        metadata,
        membership,
        turnover,
        daily_coverage,
        weekly_coverage,
        quality_checks,
        targets,
        benchmark_comparison,
        daily_csad,
        weekly_csad,
        [path.relative_to(PROJECT_ROOT).as_posix() for path in plot_paths],
    )
    save_text(report, output_dir / "cmc_dynamic_universe_report.md")
    save_config_snapshot(config, output_dir / "config_snapshot.yaml")
    shutil.copy2(config["protocol"]["path"], output_dir / "protocol_snapshot.md")
    input_manifest = save_input_manifest(
        [source_cfg["manifest_path"], source_cfg["metadata_path"], config["protocol"]["path"]],
        output_dir / "input_manifest.json",
    )
    save_provenance_manifest(
        config,
        output_dir / "provenance.json",
        schema_version=2,
        pipeline_version="cmc-dynamic-universe-replication-v1",
        train_start=analysis_cfg["start"],
        train_end=analysis_cfg["end"],
        statistical_method="Daily and weekly HAC corrected CSAD with period-wise six-test BH-FDR",
        input_manifest_path=input_manifest,
    )
    LOGGER.info(
        "CMC 동적 universe 재현이 완료됐습니다. output=%s",
        output_dir,
    )


def _resolve_runtime_paths(config: dict) -> dict:
    resolved = dict(config)
    resolved["source"] = dict(config["source"])
    for key in ("cache_dir", "metadata_path", "state_path", "manifest_path"):
        resolved["source"][key] = str(_project_path(config["source"][key]))
    resolved["protocol"] = dict(config["protocol"])
    resolved["protocol"]["path"] = str(_project_path(config["protocol"]["path"]))
    resolved["output"] = dict(config["output"])
    resolved["output"]["base_dir"] = str(_project_path(config["output"]["base_dir"]))
    return resolved


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _validate_unique_daily_ids(daily_rows: pd.DataFrame) -> None:
    duplicate_count = int(daily_rows.duplicated(["snapshot_date", "cmc_id"]).sum())
    if duplicate_count:
        raise ValueError(f"Daily member panel contains {duplicate_count} duplicate date/CMC-ID rows")


if __name__ == "__main__":
    main()

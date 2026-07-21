from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from binance_archive_point_in_time import (  # noqa: E402
    build_archive_report,
    build_episode_coverage,
    build_external_estimate_comparison,
    build_membership_transitions,
    build_moderator_summary,
    build_monthly_membership,
    build_posthoc_taxonomy_audit,
    build_point_in_time_panels,
    build_weight_concentration_summary,
    collect_archive_inventory,
    collection_status,
    download_archives,
    load_archive_inventory,
    load_file_manifest,
    load_normalized_history,
    mark_analysis_complete,
    membership_used_archive_keys,
    normalize_archive_history,
    plot_meta_forest,
    run_descriptive_meta_analysis,
    validate_archive_quality,
    verify_membership_checksums,
)
from binance_external_validation import evaluate_external_robustness  # noqa: E402
from cmc_fixed_universe import run_fixed_regressions  # noqa: E402
from utils import (  # noqa: E402
    load_config,
    save_config_snapshot,
    save_dataframe,
    save_input_manifest,
    save_provenance_manifest,
    save_text,
    setup_logging,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binance 공식 archive 기반 point-in-time universe 외부검증을 실행합니다."
    )
    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT
            / "configs"
            / "research"
            / "binance_archive_point_in_time_universe_v1.yaml"
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inventory-only", action="store_true")
    mode.add_argument("--collect-only", action="store_true")
    mode.add_argument("--analysis-only", action="store_true")
    mode.add_argument("--status", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_runtime_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    if args.status:
        print(json.dumps(collection_status(config["source"]), ensure_ascii=False, indent=2))
        return

    if args.analysis_only:
        candidates, files, inventory_manifest = load_archive_inventory(config["source"])
        file_manifest = load_file_manifest(config["source"])
        history = load_normalized_history(config["source"])
        membership, monthly_stats = build_monthly_membership(
            history, config["universe"], config["analysis"]
        )
    else:
        candidates, files, inventory_manifest = collect_archive_inventory(
            config["source"], config["universe"]
        )
        LOGGER.info(
            "Archive inventory complete: prefixes=%d candidates=%d files=%d",
            len(candidates),
            int(candidates["included"].sum()),
            len(files),
        )
        if args.inventory_only:
            return
        file_manifest = download_archives(files, config["source"])
        history = normalize_archive_history(
            file_manifest, config["source"], config["universe"]
        )
        membership, monthly_stats = build_monthly_membership(
            history, config["universe"], config["analysis"]
        )
        required_keys = membership_used_archive_keys(history, membership)
        file_manifest = verify_membership_checksums(
            file_manifest, required_keys, config["source"]
        )
        LOGGER.info(
            "Archive collection complete: rows=%d episodes=%d checksums=%d",
            len(history),
            history["asset_key"].nunique(),
            int(file_manifest["checksum_verified"].fillna(False).sum()),
        )
        if args.collect_only:
            return

    run_analysis(
        config,
        candidates,
        files,
        inventory_manifest,
        file_manifest,
        history,
        membership,
        monthly_stats,
    )


def run_analysis(
    config: dict,
    candidates: pd.DataFrame,
    files: pd.DataFrame,
    inventory_manifest: pd.DataFrame,
    file_manifest: pd.DataFrame,
    history: pd.DataFrame,
    membership: pd.DataFrame,
    monthly_stats: pd.DataFrame,
) -> None:
    output_dir = Path(config["output"]["base_dir"])
    intermediate_dir = output_dir / "intermediate"
    plots_dir = output_dir / "plots"
    for path in (output_dir, intermediate_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)

    panels = {
        variant["name"]: build_point_in_time_panels(
            history, membership, variant, config["analysis"]
        )
        for variant in config["analysis"]["variants"]
    }
    quality = validate_archive_quality(
        candidates,
        files,
        inventory_manifest,
        file_manifest,
        history,
        membership,
        panels,
        config["source"],
        config["analysis"],
    )
    targets, coefficients, diagnostics = run_fixed_regressions(panels, config["analysis"])
    decision_detail, decision_summary = evaluate_external_robustness(
        targets, config["decision"]
    )
    comparison = build_external_estimate_comparison(
        targets, config["comparison"], config["decision"]
    )
    meta = run_descriptive_meta_analysis(comparison, float(config["decision"]["alpha"]))
    moderators = build_moderator_summary(comparison)
    taxonomy_audit = build_posthoc_taxonomy_audit(membership, config["universe"])
    weight_concentration = build_weight_concentration_summary(panels)
    episodes = build_episode_coverage(history, membership, config["source"])
    transitions = build_membership_transitions(membership)

    save_dataframe(candidates, output_dir / "candidate_symbol_inventory.csv", index=False)
    save_dataframe(inventory_manifest, output_dir / "archive_inventory_manifest.csv", index=False)
    save_dataframe(file_manifest, output_dir / "archive_file_manifest.csv", index=False)
    save_dataframe(
        pd.read_csv(config["source"]["source_quality_path"]),
        output_dir / "source_quality_audit.csv",
        index=False,
    )
    save_dataframe(episodes, output_dir / "listing_episode_coverage.csv", index=False)
    save_dataframe(monthly_stats, output_dir / "prior_month_liquidity_stats.csv", index=False)
    save_dataframe(membership, output_dir / "point_in_time_membership.csv", index=False)
    save_dataframe(transitions, output_dir / "membership_transitions.csv", index=False)
    save_dataframe(quality, output_dir / "data_quality_checks.csv", index=False)
    save_dataframe(targets, output_dir / "regression_targets.csv", index=False)
    save_dataframe(coefficients, output_dir / "regression_coefficients.csv", index=False)
    save_dataframe(diagnostics, output_dir / "regression_diagnostics.csv", index=False)
    save_dataframe(decision_detail, output_dir / "external_validation_decision_detail.csv", index=False)
    save_dataframe(decision_summary, output_dir / "external_validation_decision_summary.csv", index=False)
    save_dataframe(comparison, output_dir / "standardized_external_comparison.csv", index=False)
    save_dataframe(meta, output_dir / "descriptive_meta_analysis.csv", index=False)
    save_dataframe(moderators, output_dir / "moderator_summary.csv", index=False)
    save_dataframe(taxonomy_audit, output_dir / "posthoc_taxonomy_audit.csv", index=False)
    save_dataframe(
        weight_concentration, output_dir / "weight_concentration_summary.csv", index=False
    )

    for variant_name, result in panels.items():
        variant_dir = intermediate_dir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        for frequency in ("daily", "weekly"):
            save_dataframe(
                result[f"{frequency}_market"],
                output_dir / f"{frequency}_market_return_{variant_name}.csv",
            )
            save_dataframe(
                result[f"{frequency}_csad"],
                output_dir / f"{frequency}_csad_{variant_name}.csv",
            )
            save_dataframe(
                result[f"{frequency}_coverage"],
                output_dir / f"{frequency}_coverage_{variant_name}.csv",
                index=False,
            )
            result[f"{frequency}_rows"].to_parquet(
                variant_dir / f"{frequency}_member_rows.parquet", index=False
            )
            result[f"{frequency}_panel"].to_parquet(
                variant_dir / f"{frequency}_return_panel.parquet"
            )

    plot_path = plots_dir / "corrected_csad_meta_forest.png"
    plot_meta_forest(comparison, meta, plot_path)
    report = build_archive_report(
        config,
        candidates,
        file_manifest,
        episodes,
        membership,
        quality,
        panels,
        targets,
        decision_summary,
        meta,
        taxonomy_audit,
        weight_concentration,
        [plot_path.relative_to(PROJECT_ROOT).as_posix()],
    )
    save_text(report, output_dir / "binance_archive_point_in_time_report.md")
    save_config_snapshot(config, output_dir / "config_snapshot.yaml")
    shutil.copy2(config["protocol"]["path"], output_dir / "protocol_snapshot.md")

    input_paths = [
        config["source"]["candidate_inventory_path"],
        config["source"]["archive_index_path"],
        config["source"]["inventory_manifest_path"],
        config["source"]["file_manifest_path"],
        config["source"]["source_quality_path"],
        config["source"]["normalized_path"],
        config["protocol"]["path"],
        *config["comparison"].values(),
    ]
    input_manifest = save_input_manifest(input_paths, output_dir / "input_manifest.json")
    save_provenance_manifest(
        config,
        output_dir / "provenance.json",
        schema_version=2,
        pipeline_version="binance-archive-point-in-time-universe-v1",
        train_start=config["analysis"]["start"],
        train_end=config["analysis"]["end"],
        statistical_method="Daily-weekly HAC CSAD with variant-period BH-FDR and descriptive DL meta-analysis",
        input_manifest_path=input_manifest,
    )
    primary = decision_summary.set_index("variant").loc[
        config["decision"]["primary_variant"], "all_required_cells_pass"
    ]
    sensitivity = decision_summary.set_index("variant").loc[
        config["decision"]["sensitivity_variant"], "all_required_cells_pass"
    ]
    mark_analysis_complete(
        config["source"], output_dir, bool(primary), bool(sensitivity)
    )
    LOGGER.info(
        "Binance archive point-in-time validation complete: output=%s primary_pass=%s",
        output_dir,
        bool(primary),
    )


def _load_runtime_config(path: str | Path) -> dict:
    config = load_config(path)
    resolved = dict(config)
    resolved["source"] = dict(config["source"])
    for key in (
        "cache_dir",
        "inventory_dir",
        "zip_dir",
        "checksum_dir",
        "state_path",
        "candidate_inventory_path",
        "archive_index_path",
        "inventory_manifest_path",
        "file_manifest_path",
        "source_quality_path",
        "normalized_path",
    ):
        resolved["source"][key] = str(_project_path(config["source"][key]))
    resolved["protocol"] = dict(config["protocol"])
    resolved["protocol"]["path"] = str(_project_path(config["protocol"]["path"]))
    resolved["comparison"] = {
        key: str(_project_path(value)) for key, value in config["comparison"].items()
    }
    resolved["output"] = dict(config["output"])
    resolved["output"]["base_dir"] = str(_project_path(config["output"]["base_dir"]))
    return resolved


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()

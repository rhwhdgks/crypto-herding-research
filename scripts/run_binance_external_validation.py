from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from binance_external_validation import (  # noqa: E402
    build_binance_panels,
    build_external_validation_report,
    compare_cmc_and_binance,
    evaluate_external_robustness,
    load_binance_daily_history,
    plot_cross_provider_comparison,
    validate_binance_quality,
)
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
        description="Binance 14종목 공급자·universe 외부 강건성 검증을 실행합니다."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "research" / "binance_14_external_validation_v1.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_runtime_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    run_analysis(config)


def run_analysis(config: dict) -> None:
    output_dir = Path(config["output"]["base_dir"])
    intermediate_dir = output_dir / "intermediate"
    plots_dir = output_dir / "plots"
    for path in (output_dir, intermediate_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)

    history, manifest = load_binance_daily_history(config["source"])
    panels = {
        variant["name"]: build_binance_panels(history, variant, config["analysis"])
        for variant in config["analysis"]["variants"]
    }
    quality = validate_binance_quality(
        history, manifest, panels, config["source"], config["analysis"]
    )
    targets, coefficients, diagnostics = run_fixed_regressions(panels, config["analysis"])
    decision_detail, decision_summary = evaluate_external_robustness(targets, config["decision"])
    historical = pd.read_csv(config["comparison"]["cmc_historical_targets"])
    holdout = pd.read_csv(config["comparison"]["cmc_holdout_targets"])
    comparison = compare_cmc_and_binance(
        targets, historical, holdout, config["decision"]
    )

    save_dataframe(manifest, output_dir / "input_file_manifest.csv", index=False)
    save_dataframe(quality, output_dir / "data_quality_checks.csv", index=False)
    save_dataframe(targets, output_dir / "regression_targets.csv", index=False)
    save_dataframe(coefficients, output_dir / "regression_coefficients.csv", index=False)
    save_dataframe(diagnostics, output_dir / "regression_diagnostics.csv", index=False)
    save_dataframe(decision_detail, output_dir / "external_robustness_decision_detail.csv", index=False)
    save_dataframe(decision_summary, output_dir / "external_robustness_decision_summary.csv", index=False)
    save_dataframe(comparison, output_dir / "cross_provider_comparison.csv", index=False)

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

    plot_path = plots_dir / "cross_provider_standardized_coefficients.png"
    plot_cross_provider_comparison(comparison, plot_path)
    report = build_external_validation_report(
        config,
        history,
        quality,
        panels,
        targets,
        decision_summary,
        comparison,
        [plot_path.relative_to(PROJECT_ROOT).as_posix()],
    )
    save_text(report, output_dir / "binance_14_external_validation_report.md")
    save_config_snapshot(config, output_dir / "config_snapshot.yaml")
    shutil.copy2(config["protocol"]["path"], output_dir / "protocol_snapshot.md")
    input_paths = [
        row.path for row in manifest.itertuples()
    ] + [
        config["protocol"]["path"],
        config["comparison"]["cmc_historical_targets"],
        config["comparison"]["cmc_holdout_targets"],
    ]
    input_manifest = save_input_manifest(
        input_paths, output_dir / "input_manifest.json"
    )
    save_provenance_manifest(
        config,
        output_dir / "provenance.json",
        schema_version=2,
        pipeline_version="binance-14-external-validation-v1",
        train_start=config["analysis"]["start"],
        train_end=config["analysis"]["end"],
        statistical_method="Daily-weekly HAC corrected CSAD with variant-period BH-FDR",
        input_manifest_path=input_manifest,
    )
    primary = decision_summary.set_index("variant").loc[
        config["decision"]["primary_variant"], "all_required_cells_pass"
    ]
    LOGGER.info(
        "Binance 14종목 외부 강건성 검증 완료: output=%s primary_pass=%s",
        output_dir,
        bool(primary),
    )


def _load_runtime_config(path: str | Path) -> dict:
    config = load_config(path)
    resolved = dict(config)
    resolved["source"] = dict(config["source"])
    resolved["source"]["data_dir"] = str(_project_path(config["source"]["data_dir"]))
    resolved["protocol"] = dict(config["protocol"])
    resolved["protocol"]["path"] = str(_project_path(config["protocol"]["path"]))
    resolved["comparison"] = dict(config["comparison"])
    for key in ("cmc_historical_targets", "cmc_holdout_targets"):
        resolved["comparison"][key] = str(_project_path(config["comparison"][key]))
    resolved["output"] = dict(config["output"])
    resolved["output"]["base_dir"] = str(_project_path(config["output"]["base_dir"]))
    return resolved


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()

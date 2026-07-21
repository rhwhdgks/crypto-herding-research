from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd

from cmc_structural_breaks import (
    build_break_date_table,
    build_paper_regime_comparison,
    build_structural_break_report,
    fit_scsad_regimes,
    load_scsad_break_frame,
    plot_regime_gamma3,
    plot_scsad_regimes,
    run_no_break_stability_diagnostics,
    search_structural_breaks,
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
        description="CMC daily SCSAD의 다중 구조 변화를 분석합니다."
    )
    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT
            / "configs"
            / "cmc_dynamic_universe"
            / "structural_break_v1.yaml"
        ),
        help="동결된 structural-break 설정 YAML",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_config = load_config(args.config)
    setup_logging(raw_config.get("logging", {}).get("level", "INFO"))
    config = _resolve_paths(raw_config)
    input_cfg = config["input"]
    output_dir = Path(config["output"]["base_dir"])
    plots_dir = output_dir / "plots"
    intermediate_dir = output_dir / "intermediate"
    for path in (output_dir, plots_dir, intermediate_dir):
        path.mkdir(parents=True, exist_ok=True)

    _validate_replication_provenance(input_cfg["replication_provenance_path"])
    frame = load_scsad_break_frame(
        input_cfg["csad_path"],
        input_cfg["market_return_path"],
        start=input_cfg["start"],
        end=input_cfg["end"],
    )
    LOGGER.info("CMC daily SCSAD break search를 시작합니다. observations=%d", len(frame))
    search = search_structural_breaks(frame, config["break_search"])
    paper_dates = config["paper_benchmark"]["regime_starts"]
    selected_break_dates = build_break_date_table(
        frame.index,
        search.selected_break_indices,
        paper_dates,
        solution_name="bic_selected",
    )
    selected_targets, selected_coefficients, selected_diagnostics, selected_fitted = (
        fit_scsad_regimes(
            frame,
            search.selected_break_indices,
            config["regression"],
            solution_name="bic_selected",
        )
    )

    fixed_break_count = int(config["break_search"]["paper_fixed_break_count"])
    if fixed_break_count not in search.break_indices_by_count:
        raise ValueError(
            f"Configured paper_fixed_break_count={fixed_break_count} is infeasible"
        )
    fixed_indices = search.break_indices_by_count[fixed_break_count]
    fixed_break_dates = build_break_date_table(
        frame.index,
        fixed_indices,
        paper_dates,
        solution_name="fixed_four_global_rss",
    )
    fixed_targets, fixed_coefficients, fixed_diagnostics, fixed_fitted = (
        fit_scsad_regimes(
            frame,
            fixed_indices,
            config["regression"],
            solution_name="fixed_four_global_rss",
        )
    )

    paper_indices = _paper_dates_to_indices(frame.index, paper_dates)
    paper_targets, paper_coefficients, paper_diagnostics, paper_fitted = (
        fit_scsad_regimes(
            frame,
            paper_indices,
            config["regression"],
            solution_name="paper_published_dates",
        )
    )
    paper_break_dates = build_break_date_table(
        frame.index,
        paper_indices,
        paper_dates,
        solution_name="paper_published_dates",
    )
    paper_regime_comparison = build_paper_regime_comparison(
        paper_targets,
        config["paper_benchmark"],
    )
    stability = run_no_break_stability_diagnostics(frame)

    save_dataframe(
        search.information_criteria,
        output_dir / "break_count_information_criteria.csv",
        index=False,
    )
    save_dataframe(selected_break_dates, output_dir / "selected_break_dates.csv", index=False)
    save_dataframe(fixed_break_dates, output_dir / "fixed_four_break_dates.csv", index=False)
    save_dataframe(paper_break_dates, output_dir / "paper_published_break_dates.csv", index=False)
    save_dataframe(selected_targets, output_dir / "selected_regime_targets.csv", index=False)
    save_dataframe(selected_coefficients, output_dir / "selected_regime_coefficients.csv", index=False)
    save_dataframe(selected_diagnostics, output_dir / "selected_regime_diagnostics.csv", index=False)
    save_dataframe(fixed_targets, output_dir / "fixed_four_regime_targets.csv", index=False)
    save_dataframe(fixed_coefficients, output_dir / "fixed_four_regime_coefficients.csv", index=False)
    save_dataframe(fixed_diagnostics, output_dir / "fixed_four_regime_diagnostics.csv", index=False)
    save_dataframe(paper_targets, output_dir / "paper_date_regime_targets.csv", index=False)
    save_dataframe(paper_coefficients, output_dir / "paper_date_regime_coefficients.csv", index=False)
    save_dataframe(paper_diagnostics, output_dir / "paper_date_regime_diagnostics.csv", index=False)
    save_dataframe(
        paper_regime_comparison,
        output_dir / "paper_regime_coefficient_comparison.csv",
        index=False,
    )
    save_dataframe(stability, output_dir / "no_break_stability_diagnostics.csv", index=False)
    frame.to_parquet(intermediate_dir / "scsad_break_input.parquet")
    pd.concat(
        [
            selected_fitted.rename("bic_selected"),
            fixed_fitted.rename("fixed_four_global_rss"),
            paper_fitted.rename("paper_published_dates"),
        ],
        axis=1,
    ).to_parquet(intermediate_dir / "regime_fitted_values.parquet")

    plot_paths = [
        plots_dir / "scsad_structural_regimes.png",
        plots_dir / "regime_gamma3_coefficients.png",
    ]
    plot_scsad_regimes(frame, selected_fitted, selected_break_dates, plot_paths[0])
    plot_regime_gamma3(selected_targets, plot_paths[1])
    report = build_structural_break_report(
        config,
        frame,
        search,
        selected_break_dates,
        selected_targets,
        fixed_break_dates,
        paper_regime_comparison,
        stability,
        [path.relative_to(PROJECT_ROOT).as_posix() for path in plot_paths],
    )
    save_text(report, output_dir / "cmc_scsad_structural_break_report.md")
    save_config_snapshot(config, output_dir / "config_snapshot.yaml")
    shutil.copy2(config["protocol"]["path"], output_dir / "protocol_snapshot.md")
    input_manifest = save_input_manifest(
        [
            input_cfg["csad_path"],
            input_cfg["market_return_path"],
            input_cfg["replication_provenance_path"],
            config["protocol"]["path"],
        ],
        output_dir / "input_manifest.json",
    )
    save_provenance_manifest(
        config,
        output_dir / "provenance.json",
        schema_version=2,
        pipeline_version="cmc-scsad-structural-break-v1",
        train_start=input_cfg["start"],
        train_end=input_cfg["end"],
        statistical_method=(
            "Exact global multiple-break RSS dynamic programming with BIC selection; "
            "regime Newey-West HAC and BH-FDR"
        ),
        input_manifest_path=input_manifest,
    )
    LOGGER.info(
        "CMC SCSAD structural-break 분석이 완료됐습니다. breaks=%d, supported=%s, output=%s",
        search.selected_break_count,
        search.structural_change_supported,
        output_dir,
    )


def _paper_dates_to_indices(
    index: pd.DatetimeIndex,
    paper_dates: list[str],
) -> list[int]:
    positions = []
    for value in paper_dates:
        timestamp = pd.Timestamp(value)
        timestamp = (
            timestamp.tz_localize("UTC")
            if timestamp.tzinfo is None
            else timestamp.tz_convert("UTC")
        )
        position = int(index.searchsorted(timestamp, side="left"))
        if position <= 0 or position >= len(index):
            raise ValueError(f"Paper break date lies outside the input sample: {value}")
        positions.append(position)
    if positions != sorted(set(positions)):
        raise ValueError("Paper break dates do not map to unique increasing observations")
    return positions


def _validate_replication_provenance(path: str | Path) -> None:
    import json

    provenance = json.loads(Path(path).read_text(encoding="utf-8"))
    if provenance.get("pipeline_version") != "cmc-dynamic-universe-replication-v1":
        raise ValueError("Structural-break input is not from the corrected CMC replication")
    if int(provenance.get("schema_version", 0)) < 2:
        raise ValueError("Structural-break input provenance schema is too old")


def _resolve_paths(config: dict) -> dict:
    resolved = dict(config)
    resolved["protocol"] = dict(config["protocol"])
    resolved["protocol"]["path"] = str(_project_path(config["protocol"]["path"]))
    resolved["input"] = dict(config["input"])
    for key in (
        "replication_dir",
        "csad_path",
        "market_return_path",
        "replication_provenance_path",
    ):
        resolved["input"][key] = str(_project_path(config["input"][key]))
    resolved["output"] = dict(config["output"])
    resolved["output"]["base_dir"] = str(_project_path(config["output"]["base_dir"]))
    return resolved


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()

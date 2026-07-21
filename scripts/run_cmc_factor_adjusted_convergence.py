from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd

from cmc_factor_convergence import (
    aggregate_daily_convergence,
    build_factor_convergence_report,
    build_fixed_regimes,
    build_quality_summary,
    estimate_point_in_time_factor_model,
    load_factor_member_rows,
    load_point_in_time_estimation_history,
    plot_convergence_timeseries,
    plot_regime_delta2,
    run_convergence_regressions,
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
        description="CMC 동적 universe의 시장요인 조정 초과 수렴을 분석합니다."
    )
    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT
            / "configs"
            / "cmc_dynamic_universe"
            / "factor_adjusted_convergence_v1.yaml"
        ),
        help="동결된 factor-adjusted convergence 설정 YAML",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_config = load_config(args.config)
    setup_logging(raw_config.get("logging", {}).get("level", "INFO"))
    config = _resolve_paths(raw_config)
    input_cfg = config["input"]
    factor_cfg = config["factor_model"]
    output_dir = Path(config["output"]["base_dir"])
    intermediate_dir = output_dir / "intermediate"
    plots_dir = output_dir / "plots"
    for path in (output_dir, intermediate_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)

    _validate_input_provenance(
        input_cfg["replication_provenance_path"],
        "cmc-dynamic-universe-replication-v1",
    )
    _validate_input_provenance(
        input_cfg["structural_provenance_path"],
        "cmc-scsad-structural-break-v1",
    )
    member_rows = load_factor_member_rows(
        input_cfg["member_rows_path"],
        input_cfg["start"],
        input_cfg["end"],
    )
    estimation_history = load_point_in_time_estimation_history(
        member_rows,
        input_cfg["snapshot_dir"],
        input_cfg["start"],
        input_cfg["end"],
    )
    break_dates = pd.read_csv(
        input_cfg["structural_break_dates_path"],
        parse_dates=["previous_regime_end", "next_regime_start"],
    )
    regimes = build_fixed_regimes(
        input_cfg["start"],
        input_cfg["end"],
        break_dates,
    )

    daily_by_window: dict[str, pd.DataFrame] = {}
    primary_observations = int(factor_cfg["primary_window_observations"])
    primary_name = f"window_{primary_observations}"
    primary_members = None
    for window_cfg in factor_cfg["windows"]:
        name = str(window_cfg["name"])
        observations = int(window_cfg["observations"])
        expected_name = f"window_{observations}"
        if name != expected_name:
            raise ValueError(
                f"Window name {name!r} must match observations as {expected_name!r}"
            )
        LOGGER.info(
            "과거정보 전용 시장모형을 계산합니다. window=%d, min_obs=%d",
            observations,
            int(window_cfg["minimum_observations"]),
        )
        member_models = estimate_point_in_time_factor_model(
            estimation_history,
            window_observations=observations,
            minimum_observations=int(window_cfg["minimum_observations"]),
            minimum_regressor_variance=float(
                factor_cfg["minimum_regressor_variance"]
            ),
            minimum_residual_sigma=float(factor_cfg["minimum_residual_sigma"]),
        )
        daily = aggregate_daily_convergence(
            member_models,
            minimum_daily_model_coverage=float(
                factor_cfg["minimum_daily_model_coverage"]
            ),
            minimum_cross_section_assets=int(
                factor_cfg["minimum_cross_section_assets"]
            ),
        )
        daily_by_window[name] = daily
        save_dataframe(
            daily,
            output_dir / f"factor_convergence_daily_{name}.csv",
            index=True,
        )
        if observations == primary_observations:
            primary_members = member_models

    if primary_name not in daily_by_window or primary_members is None:
        raise ValueError("Configured primary factor window was not calculated")
    primary_members.to_parquet(
        intermediate_dir / "primary_window_member_factor_models.parquet",
        index=False,
    )

    targets, coefficients, mean_tests = run_convergence_regressions(
        daily_by_window,
        regimes,
        config["regression"],
        primary_window_name=primary_name,
    )
    quality = build_quality_summary(daily_by_window, primary_name)
    save_dataframe(targets, output_dir / "convergence_regression_targets.csv", index=False)
    save_dataframe(
        coefficients,
        output_dir / "convergence_regression_coefficients.csv",
        index=False,
    )
    save_dataframe(mean_tests, output_dir / "mean_convergence_hac_tests.csv", index=False)
    save_dataframe(quality, output_dir / "factor_model_quality_summary.csv", index=False)
    save_dataframe(pd.DataFrame(regimes), output_dir / "fixed_regime_definitions.csv", index=False)

    plot_paths = [
        plots_dir / "factor_adjusted_convergence_timeseries.png",
        plots_dir / "regime_extreme_convergence_coefficients.png",
    ]
    plot_convergence_timeseries(
        daily_by_window[primary_name],
        break_dates,
        plot_paths[0],
    )
    plot_regime_delta2(targets, plot_paths[1])
    relative_plot_paths = [
        path.relative_to(PROJECT_ROOT).as_posix() for path in plot_paths
    ]
    report = build_factor_convergence_report(
        config,
        quality,
        targets,
        mean_tests,
        relative_plot_paths,
    )
    save_text(report, output_dir / "cmc_factor_adjusted_convergence_report.md")
    save_config_snapshot(config, output_dir / "config_snapshot.yaml")
    shutil.copy2(config["protocol"]["path"], output_dir / "protocol_snapshot.md")
    input_manifest = save_input_manifest(
        [
            input_cfg["member_rows_path"],
            input_cfg["snapshot_manifest_path"],
            input_cfg["replication_provenance_path"],
            input_cfg["structural_break_dates_path"],
            input_cfg["structural_provenance_path"],
            config["protocol"]["path"],
        ],
        output_dir / "input_manifest.json",
    )
    save_provenance_manifest(
        config,
        output_dir / "provenance.json",
        schema_version=2,
        pipeline_version="cmc-factor-adjusted-convergence-v1",
        train_start=input_cfg["start"],
        train_end=input_cfg["end"],
        statistical_method=(
            "Point-in-time rolling leave-one-out market model; normal-residual "
            "counterfactual CSAD; Newey-West HAC and six-period BH-FDR"
        ),
        input_manifest_path=input_manifest,
    )
    supported = int(
        targets.loc[
            targets["window"].eq(primary_name),
            "supports_extreme_excess_convergence",
        ].sum()
    )
    LOGGER.info(
        "시장요인 조정 수렴 분석이 완료됐습니다. primary_pass=%d/6, output=%s",
        supported,
        output_dir,
    )


def _validate_input_provenance(path: str | Path, expected_pipeline: str) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("pipeline_version") != expected_pipeline:
        raise ValueError(
            f"Expected provenance {expected_pipeline!r}, got {payload.get('pipeline_version')!r}"
        )
    if int(payload.get("schema_version", 0)) < 2:
        raise ValueError("Input provenance schema is too old")


def _resolve_paths(config: dict) -> dict:
    resolved = dict(config)
    resolved["protocol"] = dict(config["protocol"])
    resolved["protocol"]["path"] = str(_project_path(config["protocol"]["path"]))
    resolved["input"] = dict(config["input"])
    for key in (
        "member_rows_path",
        "snapshot_dir",
        "snapshot_manifest_path",
        "replication_provenance_path",
        "structural_break_dates_path",
        "structural_provenance_path",
    ):
        resolved["input"][key] = str(_project_path(config["input"][key]))
    resolved["output"] = dict(config["output"])
    resolved["output"]["base_dir"] = str(
        _project_path(config["output"]["base_dir"])
    )
    return resolved


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()

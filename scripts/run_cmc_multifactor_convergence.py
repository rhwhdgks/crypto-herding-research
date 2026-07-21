from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd

from cmc_factor_convergence import (
    aggregate_daily_convergence,
    build_fixed_regimes,
    build_quality_summary,
    load_factor_member_rows,
    run_convergence_regressions,
)
from cmc_multifactor_convergence import (
    build_factor_correlation,
    build_multifactor_report,
    build_point_in_time_factors,
    build_single_multi_comparison,
    estimate_multifactor_models,
    load_multifactor_history,
    plot_multifactor_timeseries,
    plot_primary_model_comparison,
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
        description="CMC point-in-time 다요인 초과 수렴 강건성을 분석합니다."
    )
    parser.add_argument(
        "--config",
        default=str(
            PROJECT_ROOT
            / "configs"
            / "cmc_dynamic_universe"
            / "multifactor_convergence_v1.yaml"
        ),
        help="동결된 multi-factor convergence 설정 YAML",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_config = load_config(args.config)
    setup_logging(raw_config.get("logging", {}).get("level", "INFO"))
    config = _resolve_paths(raw_config)
    input_cfg = config["input"]
    factor_cfg = config["factor_construction"]
    model_cfg = config["factor_model"]
    output_dir = Path(config["output"]["base_dir"])
    intermediate_dir = output_dir / "intermediate"
    plots_dir = output_dir / "plots"
    for path in (output_dir, intermediate_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)

    _validate_provenance(
        input_cfg["replication_provenance_path"],
        "cmc-dynamic-universe-replication-v1",
    )
    _validate_provenance(
        input_cfg["structural_provenance_path"],
        "cmc-scsad-structural-break-v1",
    )
    _validate_provenance(
        input_cfg["single_factor_provenance_path"],
        "cmc-factor-adjusted-convergence-v1",
    )
    member_rows = load_factor_member_rows(
        input_cfg["member_rows_path"], input_cfg["start"], input_cfg["end"]
    )
    history = load_multifactor_history(
        member_rows,
        input_cfg["snapshot_dir"],
        input_cfg["start"],
        input_cfg["end"],
        momentum_calendar_days=int(factor_cfg["momentum_calendar_days"]),
    )
    history, factor_returns, factor_diagnostics = build_point_in_time_factors(
        history,
        tail_fraction=float(factor_cfg["tail_fraction"]),
        minimum_leg_assets=int(factor_cfg["minimum_leg_assets"]),
    )
    factor_correlation = build_factor_correlation(factor_returns)
    save_dataframe(factor_returns, output_dir / "point_in_time_factor_returns.csv", index=True)
    save_dataframe(factor_diagnostics, output_dir / "factor_construction_diagnostics.csv", index=False)
    save_dataframe(factor_correlation, output_dir / "factor_return_correlation.csv", index=False)

    break_dates = pd.read_csv(
        input_cfg["structural_break_dates_path"],
        parse_dates=["previous_regime_end", "next_regime_start"],
    )
    regimes = build_fixed_regimes(
        input_cfg["start"], input_cfg["end"], break_dates
    )
    factor_columns = list(factor_cfg["factor_columns"])
    primary_observations = int(model_cfg["primary_window_observations"])
    primary_name = f"window_{primary_observations}"
    daily_by_window: dict[str, pd.DataFrame] = {}
    primary_members = None
    empirical_daily = None
    for window_cfg in model_cfg["windows"]:
        name = str(window_cfg["name"])
        observations = int(window_cfg["observations"])
        if name != f"window_{observations}":
            raise ValueError(f"Window name {name!r} does not match observations")
        is_primary = observations == primary_observations
        LOGGER.info(
            "다요인 rolling 모형을 계산합니다. window=%d, min_obs=%d, empirical=%s",
            observations,
            int(window_cfg["minimum_observations"]),
            is_primary,
        )
        members = estimate_multifactor_models(
            history,
            factor_columns=factor_columns,
            window_observations=observations,
            minimum_observations=int(window_cfg["minimum_observations"]),
            maximum_condition_number=float(model_cfg["maximum_condition_number"]),
            minimum_residual_sigma=float(model_cfg["minimum_residual_sigma"]),
            empirical_minimum_residuals=(
                int(model_cfg["empirical_minimum_residuals"])
                if is_primary
                else None
            ),
        )
        normal_daily = aggregate_daily_convergence(
            members,
            minimum_daily_model_coverage=float(model_cfg["minimum_daily_model_coverage"]),
            minimum_cross_section_assets=int(model_cfg["minimum_cross_section_assets"]),
        )
        daily_by_window[name] = normal_daily
        save_dataframe(
            normal_daily,
            output_dir / f"multifactor_convergence_daily_{name}.csv",
            index=True,
        )
        if is_primary:
            primary_members = members
            empirical_members = members.copy()
            empirical_members["expected_abs_deviation"] = empirical_members[
                "expected_abs_deviation_empirical"
            ]
            empirical_daily = aggregate_daily_convergence(
                empirical_members,
                minimum_daily_model_coverage=float(model_cfg["minimum_daily_model_coverage"]),
                minimum_cross_section_assets=int(model_cfg["minimum_cross_section_assets"]),
            )
            save_dataframe(
                empirical_daily,
                output_dir / "multifactor_convergence_daily_window_365_empirical.csv",
                index=True,
            )
    if primary_members is None or empirical_daily is None:
        raise ValueError("Primary multi-factor window was not calculated")
    primary_members.to_parquet(
        intermediate_dir / "primary_window_multifactor_models.parquet", index=False
    )

    normal_targets, normal_coefficients, normal_means = run_convergence_regressions(
        daily_by_window,
        regimes,
        config["regression"],
        primary_window_name=primary_name,
    )
    empirical_targets, empirical_coefficients, empirical_means = (
        run_convergence_regressions(
            {"window_365_empirical": empirical_daily},
            regimes,
            config["regression"],
            primary_window_name="window_365_empirical",
        )
    )
    quality = build_quality_summary(daily_by_window, primary_name)
    empirical_quality = build_quality_summary(
        {"window_365_empirical": empirical_daily}, "window_365_empirical"
    )
    quality = pd.concat([quality, empirical_quality], ignore_index=True)

    single_dir = Path(input_cfg["single_factor_dir"])
    single_daily = _load_daily(
        single_dir / "factor_convergence_daily_window_365.csv"
    )
    single_targets = pd.read_csv(single_dir / "convergence_regression_targets.csv")
    single_targets = single_targets.loc[single_targets["window"].eq("window_365")]
    normal_primary_targets = normal_targets.loc[
        normal_targets["window"].eq(primary_name)
    ]
    comparison, comparison_correlation = build_single_multi_comparison(
        single_daily,
        daily_by_window[primary_name],
        empirical_daily,
        single_targets,
        normal_primary_targets,
        empirical_targets,
    )

    save_dataframe(normal_targets, output_dir / "normal_convergence_regression_targets.csv", index=False)
    save_dataframe(normal_coefficients, output_dir / "normal_convergence_regression_coefficients.csv", index=False)
    save_dataframe(normal_means, output_dir / "normal_mean_convergence_hac_tests.csv", index=False)
    save_dataframe(empirical_targets, output_dir / "empirical_convergence_regression_targets.csv", index=False)
    save_dataframe(empirical_coefficients, output_dir / "empirical_convergence_regression_coefficients.csv", index=False)
    save_dataframe(empirical_means, output_dir / "empirical_mean_convergence_hac_tests.csv", index=False)
    save_dataframe(quality, output_dir / "multifactor_model_quality_summary.csv", index=False)
    save_dataframe(comparison, output_dir / "single_vs_multifactor_comparison.csv", index=False)
    save_dataframe(comparison_correlation, output_dir / "single_vs_multifactor_correlation.csv", index=False)
    save_dataframe(pd.DataFrame(regimes), output_dir / "fixed_regime_definitions.csv", index=False)

    plot_paths = [
        plots_dir / "multifactor_convergence_timeseries.png",
        plots_dir / "single_vs_multifactor_regime_coefficients.png",
    ]
    plot_multifactor_timeseries(
        daily_by_window[primary_name], empirical_daily, plot_paths[0]
    )
    plot_primary_model_comparison(
        single_targets,
        normal_primary_targets,
        empirical_targets,
        plot_paths[1],
    )
    relative_plots = [path.relative_to(PROJECT_ROOT).as_posix() for path in plot_paths]
    report = build_multifactor_report(
        quality,
        normal_targets,
        empirical_targets,
        comparison,
        factor_diagnostics,
        relative_plots,
    )
    save_text(report, output_dir / "cmc_multifactor_convergence_report.md")
    save_config_snapshot(config, output_dir / "config_snapshot.yaml")
    shutil.copy2(config["protocol"]["path"], output_dir / "protocol_snapshot.md")
    input_manifest = save_input_manifest(
        [
            input_cfg["member_rows_path"],
            input_cfg["snapshot_manifest_path"],
            input_cfg["replication_provenance_path"],
            input_cfg["structural_break_dates_path"],
            input_cfg["structural_provenance_path"],
            input_cfg["single_factor_provenance_path"],
            single_dir / "factor_convergence_daily_window_365.csv",
            single_dir / "convergence_regression_targets.csv",
            config["protocol"]["path"],
        ],
        output_dir / "input_manifest.json",
    )
    save_provenance_manifest(
        config,
        output_dir / "provenance.json",
        schema_version=2,
        pipeline_version="cmc-multifactor-convergence-v1",
        train_start=input_cfg["start"],
        train_end=input_cfg["end"],
        statistical_method=(
            "Point-in-time rolling MKT/SIZE/LIQ/MOM model; normal and empirical "
            "residual counterfactual CSAD; Newey-West HAC and six-period BH-FDR"
        ),
        input_manifest_path=input_manifest,
    )
    normal_full = normal_primary_targets.loc[
        normal_primary_targets["period"].eq("full_sample")
    ].iloc[0]
    empirical_full = empirical_targets.loc[
        empirical_targets["period"].eq("full_sample")
    ].iloc[0]
    LOGGER.info(
        "다요인 초과 수렴 분석이 완료됐습니다. dual_primary_pass=%s, output=%s",
        bool(
            normal_full["supports_extreme_excess_convergence"]
            and empirical_full["supports_extreme_excess_convergence"]
        ),
        output_dir,
    )


def _load_daily(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame.sort_index()


def _validate_provenance(path: str | Path, expected_pipeline: str) -> None:
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
        "single_factor_dir",
        "single_factor_provenance_path",
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

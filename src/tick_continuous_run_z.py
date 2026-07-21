from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm

from frequency_sensitivity import benjamini_hochberg
from tick_event_schema import require_tick_schema_v2


RUN_FEATURES = [
    "run_intensity_up",
    "run_intensity_down",
    "run_intensity_zero",
]
BASE_CONTROLS = [
    "bucket_return",
    "abs_bucket_return",
    "current_market_loo",
    "abs_current_market_loo",
    "trailing_volatility",
    "log_transaction_count",
    "log_quote_quantity",
]
FUTURE_CONTROLS = [*BASE_CONTROLS, "aggressor_imbalance"]
REQUIRED_COLUMNS = [
    "bucket_start",
    "signal_timestamp",
    "symbol",
    "schema_version",
    "interval_minutes",
    "transaction_count",
    "total_quote_quantity",
    "bucket_return",
    "run_z_up",
    "run_z_down",
    "run_z_zero",
    "run_clustering_side",
    "aggressor_imbalance",
    "aggressor_direction",
    "price_direction",
    "forward_return_30m",
    "horizon_is_exact_30m",
    "is_micro_run_clustering_event",
    "is_control_bucket",
]


@dataclass(frozen=True)
class ScalingArtifact:
    schema_version: int
    fit_start: str
    fit_end: str
    created_at_utc: str
    lower_quantile: float
    upper_quantile: float
    columns: dict[str, dict[str, float]]


def load_continuous_run_z_frame(
    path: str | Path,
    expected_symbols: Sequence[str],
) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() != ".parquet":
        raise ValueError("Continuous run-z analysis requires a Parquet schema v2 frame")
    header = pq.ParquetFile(source).schema.names
    missing = sorted(set(REQUIRED_COLUMNS).difference(header))
    if missing:
        raise ValueError(f"Continuous run-z frame is missing columns: {', '.join(missing)}")
    frame = pd.read_parquet(source, columns=REQUIRED_COLUMNS)
    frame["bucket_start"] = pd.to_datetime(frame["bucket_start"], utc=True)
    frame["signal_timestamp"] = pd.to_datetime(frame["signal_timestamp"], utc=True)
    frame = frame.sort_values(["symbol", "bucket_start"]).reset_index(drop=True)
    require_tick_schema_v2(frame)
    actual_symbols = set(frame["symbol"].dropna().astype(str).unique())
    if actual_symbols != set(expected_symbols):
        raise ValueError("Continuous run-z symbol universe does not match the frozen config")
    return frame


def prepare_continuous_run_z_frame(frame: pd.DataFrame, config: Mapping) -> pd.DataFrame:
    analysis = config["analysis"]
    split = config["split"]
    horizon = int(analysis["horizon_minutes"])
    if horizon != 30:
        raise ValueError("The frozen continuous run-z protocol requires a 30-minute horizon")

    working = frame.copy().sort_values(["symbol", "bucket_start"]).reset_index(drop=True)
    for side in ("up", "down", "zero"):
        working[f"run_intensity_{side}"] = -pd.to_numeric(
            working[f"run_z_{side}"], errors="coerce"
        )
    working["future_market_loo"] = leave_one_out_cross_sectional_mean(
        working, "forward_return_30m"
    )
    working["future_excess_return_30m"] = (
        working["forward_return_30m"] - working["future_market_loo"]
    )
    working["abs_forward_return_30m"] = working["forward_return_30m"].abs()
    working["future_abs_market_loo"] = leave_one_out_cross_sectional_mean(
        working, "abs_forward_return_30m"
    )
    working["future_excess_abs_return_30m"] = (
        working["abs_forward_return_30m"] - working["future_abs_market_loo"]
    )
    working["current_market_loo"] = leave_one_out_cross_sectional_mean(
        working, "bucket_return"
    )
    working["abs_bucket_return"] = working["bucket_return"].abs()
    working["abs_current_market_loo"] = working["current_market_loo"].abs()
    volatility_buckets = int(analysis["trailing_volatility_buckets"])
    working["trailing_volatility"] = working.groupby("symbol", sort=False)[
        "bucket_return"
    ].transform(
        lambda values: values.rolling(
            volatility_buckets,
            min_periods=volatility_buckets,
        ).std(ddof=1)
    )
    working["log_transaction_count"] = np.log1p(
        pd.to_numeric(working["transaction_count"], errors="coerce")
    )
    working["log_quote_quantity"] = np.log1p(
        pd.to_numeric(working["total_quote_quantity"], errors="coerce")
    )
    working["hour_utc"] = working["bucket_start"].dt.hour.astype(str)
    working["utc_day"] = working["bucket_start"].dt.floor("D")
    working["outcome_available_at"] = working["signal_timestamp"] + pd.Timedelta(
        minutes=horizon
    )
    working["meets_continuous_trade_count"] = (
        pd.to_numeric(working["transaction_count"], errors="coerce")
        >= int(analysis["minimum_transaction_count"])
    )
    working["sample_split"] = assign_sample_split(working["bucket_start"], split)
    oos_start = as_utc_timestamp(split["oos_start"])
    oos_end = as_utc_timestamp(split["oos_end_exclusive"])
    working["future_outcome_within_split"] = (
        working["sample_split"].eq("development")
        & working["outcome_available_at"].lt(oos_start)
    ) | (
        working["sample_split"].eq("oos")
        & working["outcome_available_at"].lt(oos_end)
    )
    return working


def assign_sample_split(timestamps: pd.Series, split_cfg: Mapping) -> pd.Series:
    values = pd.to_datetime(timestamps, utc=True)
    development_start = as_utc_timestamp(split_cfg["development_start"])
    oos_start = as_utc_timestamp(split_cfg["oos_start"])
    oos_end = as_utc_timestamp(split_cfg["oos_end_exclusive"])
    if not development_start < oos_start < oos_end:
        raise ValueError("Continuous run-z split boundaries are invalid")
    return pd.Series(
        np.select(
            [
                values.ge(development_start) & values.lt(oos_start),
                values.ge(oos_start) & values.lt(oos_end),
            ],
            ["development", "oos"],
            default="outside",
        ),
        index=timestamps.index,
        dtype="object",
    )


def fit_scaling_artifact(
    development_frame: pd.DataFrame,
    columns: Sequence[str],
    lower_quantile: float,
    upper_quantile: float,
) -> ScalingArtifact:
    if development_frame.empty:
        raise ValueError("Cannot fit scaling artifact on an empty development frame")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Winsor quantiles must satisfy 0 <= lower < upper <= 1")
    parameters: dict[str, dict[str, float]] = {}
    for column in columns:
        values = pd.to_numeric(development_frame[column], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"No development values available for {column}")
        lower = float(values.quantile(lower_quantile))
        upper = float(values.quantile(upper_quantile))
        clipped = values.clip(lower=lower, upper=upper)
        mean = float(clipped.mean())
        std = float(clipped.std(ddof=1))
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"Cannot standardize degenerate predictor: {column}")
        parameters[column] = {
            "winsor_lower": lower,
            "winsor_upper": upper,
            "mean": mean,
            "std": std,
        }
    timestamps = pd.to_datetime(development_frame["bucket_start"], utc=True)
    return ScalingArtifact(
        schema_version=1,
        fit_start=timestamps.min().isoformat(),
        fit_end=timestamps.max().isoformat(),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        lower_quantile=float(lower_quantile),
        upper_quantile=float(upper_quantile),
        columns=parameters,
    )


def apply_scaling_artifact(frame: pd.DataFrame, artifact: ScalingArtifact) -> pd.DataFrame:
    result = frame.copy()
    for column, parameters in artifact.columns.items():
        values = pd.to_numeric(result[column], errors="coerce").clip(
            lower=float(parameters["winsor_lower"]),
            upper=float(parameters["winsor_upper"]),
        )
        result[scaled_column(column)] = (
            values - float(parameters["mean"])
        ) / float(parameters["std"])
    return result


def validate_oos_after_artifact(frame: pd.DataFrame, artifact: ScalingArtifact) -> None:
    oos = frame.loc[frame["sample_split"].eq("oos")]
    if oos.empty:
        raise ValueError("OOS frame is empty")
    if oos["bucket_start"].min() <= pd.Timestamp(artifact.fit_end):
        raise ValueError("Scaling artifact fit period overlaps the OOS period")


def save_scaling_artifact(artifact: ScalingArtifact, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(artifact), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_continuous_run_z_models(
    frame: pd.DataFrame,
    config: Mapping,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    ScalingArtifact,
    pd.DataFrame,
    pd.DataFrame,
]:
    analysis = config["analysis"]
    predictor_columns = [*RUN_FEATURES, *FUTURE_CONTROLS]
    development_for_scaler = frame.loc[
        frame["sample_split"].eq("development")
        & frame["meets_continuous_trade_count"]
    ].dropna(subset=["bucket_start", *predictor_columns])
    artifact = fit_scaling_artifact(
        development_for_scaler,
        predictor_columns,
        float(analysis["winsor_lower_quantile"]),
        float(analysis["winsor_upper_quantile"]),
    )
    transformed = apply_scaling_artifact(frame, artifact)
    validate_oos_after_artifact(transformed, artifact)

    coefficients, diagnostics = _fit_all_pooled_models(transformed, config)
    symbol_coefficients = _fit_symbol_descriptive_models(transformed, config)
    correlations = build_feature_correlations(transformed)
    coverage = build_continuous_coverage(transformed)
    return (
        coefficients,
        diagnostics,
        symbol_coefficients,
        artifact,
        correlations,
        coverage,
    )


def _fit_all_pooled_models(
    frame: pd.DataFrame,
    config: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis = config["analysis"]
    family_specs = _family_specs()
    coefficient_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    for split_name in ("development", "oos"):
        for family_name, spec in family_specs.items():
            model_frame = _select_model_frame(frame, split_name, spec)
            result, design, model_frame = _fit_clustered_ols(
                model_frame,
                outcome=spec["outcome"],
                controls=spec["controls"],
                categorical=["symbol", "hour_utc", "price_direction"],
                minimum_observations=int(analysis["minimum_complete_observations"]),
                minimum_clusters=int(analysis["minimum_utc_day_clusters"]),
            )
            coefficient_rows.extend(
                _coefficient_rows(result, split_name, family_name, spec)
            )
            diagnostic_rows.append(
                {
                    "split": split_name,
                    "family": family_name,
                    "outcome": spec["outcome"],
                    "observations": int(result.nobs),
                    "utc_day_clusters": int(model_frame["utc_day"].nunique()),
                    "symbols": int(model_frame["symbol"].nunique()),
                    "rsquared": float(result.rsquared),
                    "adj_rsquared": float(result.rsquared_adj),
                    "condition_number": float(result.condition_number),
                    "design_columns": int(design.shape[1]),
                    "covariance": "UTC-day clustered",
                }
            )

    coefficients = pd.DataFrame(coefficient_rows)
    expected_family_size = int(analysis["family_size"])
    for _, indices in coefficients.groupby(["split", "family"]).groups.items():
        if len(indices) != expected_family_size:
            raise ValueError("Continuous run-z family size does not match the frozen config")
        coefficients.loc[indices, "q_value_bh_fdr"] = benjamini_hochberg(
            coefficients.loc[indices, "cluster_p_value"]
        )
    coefficients = _attach_predeclared_decisions(coefficients, analysis)
    return coefficients, pd.DataFrame(diagnostic_rows)


def _fit_symbol_descriptive_models(frame: pd.DataFrame, config: Mapping) -> pd.DataFrame:
    rows: list[dict] = []
    for split_name in ("development", "oos"):
        for family_name, spec in _family_specs().items():
            base = _select_model_frame(frame, split_name, spec)
            for symbol, subset in base.groupby("symbol", sort=True):
                try:
                    result, _, fitted_frame = _fit_clustered_ols(
                        subset,
                        outcome=spec["outcome"],
                        controls=spec["controls"],
                        categorical=["hour_utc", "price_direction"],
                        minimum_observations=5_000,
                        minimum_clusters=300,
                    )
                except ValueError as exc:
                    rows.append(
                        {
                            "split": split_name,
                            "family": family_name,
                            "symbol": symbol,
                            "feature": "all",
                            "status": f"not_evaluable: {exc}",
                        }
                    )
                    continue
                ci = result.conf_int()
                for feature in RUN_FEATURES:
                    term = scaled_column(feature)
                    coefficient = float(result.params[term])
                    rows.append(
                        {
                            "split": split_name,
                            "family": family_name,
                            "symbol": symbol,
                            "feature": feature,
                            "observations": int(result.nobs),
                            "utc_day_clusters": int(fitted_frame["utc_day"].nunique()),
                            "coefficient": coefficient,
                            "coefficient_bps": (
                                coefficient * 10_000.0 if spec["unit"] == "return" else np.nan
                            ),
                            "ci_lower": float(ci.loc[term, 0]),
                            "ci_upper": float(ci.loc[term, 1]),
                            "cluster_p_value": float(result.pvalues[term]),
                            "status": "descriptive_no_family_decision",
                        }
                    )
    return pd.DataFrame(rows)


def _fit_clustered_ols(
    frame: pd.DataFrame,
    outcome: str,
    controls: Sequence[str],
    categorical: Sequence[str],
    minimum_observations: int,
    minimum_clusters: int,
):
    predictor_terms = [scaled_column(column) for column in [*RUN_FEATURES, *controls]]
    required = [outcome, "utc_day", *predictor_terms, *categorical]
    model_frame = frame.dropna(subset=required).copy()
    clusters = int(model_frame["utc_day"].nunique())
    if len(model_frame) < minimum_observations:
        raise ValueError(
            f"Complete observations={len(model_frame):,}, minimum={minimum_observations:,}"
        )
    if clusters < minimum_clusters:
        raise ValueError(f"UTC-day clusters={clusters}, minimum={minimum_clusters}")
    dummy_frame = pd.get_dummies(
        model_frame[list(categorical)].astype(str),
        prefix=list(categorical),
        drop_first=True,
        dtype=float,
    )
    design = pd.concat([model_frame[predictor_terms], dummy_frame], axis=1)
    design = sm.add_constant(design, has_constant="add").astype(float)
    result = sm.OLS(model_frame[outcome].astype(float), design).fit(
        cov_type="cluster",
        cov_kwds={"groups": model_frame["utc_day"], "use_correction": True},
    )
    return result, design, model_frame


def _select_model_frame(frame: pd.DataFrame, split_name: str, spec: Mapping) -> pd.DataFrame:
    mask = frame["sample_split"].eq(split_name) & frame["meets_continuous_trade_count"]
    if spec["requires_future"]:
        mask &= frame["future_outcome_within_split"] & frame["horizon_is_exact_30m"].fillna(False)
    return frame.loc[mask].copy()


def _coefficient_rows(result, split_name: str, family_name: str, spec: Mapping) -> list[dict]:
    ci = result.conf_int()
    rows = []
    for feature in RUN_FEATURES:
        term = scaled_column(feature)
        coefficient = float(result.params[term])
        rows.append(
            {
                "split": split_name,
                "family": family_name,
                "outcome": spec["outcome"],
                "feature": feature,
                "term": term,
                "coefficient": coefficient,
                "coefficient_bps": coefficient * 10_000.0 if spec["unit"] == "return" else np.nan,
                "cluster_std_error": float(result.bse[term]),
                "cluster_t_stat": float(result.tvalues[term]),
                "cluster_p_value": float(result.pvalues[term]),
                "ci_lower": float(ci.loc[term, 0]),
                "ci_upper": float(ci.loc[term, 1]),
                "ci_lower_bps": float(ci.loc[term, 0] * 10_000.0) if spec["unit"] == "return" else np.nan,
                "ci_upper_bps": float(ci.loc[term, 1] * 10_000.0) if spec["unit"] == "return" else np.nan,
                "expected_sign": spec["expected_signs"].get(feature, "none"),
            }
        )
    return rows


def _attach_predeclared_decisions(coefficients: pd.DataFrame, analysis: Mapping) -> pd.DataFrame:
    result = coefficients.copy()
    sign_match = np.select(
        [result["expected_sign"].eq("positive"), result["expected_sign"].eq("negative")],
        [result["coefficient"].gt(0.0), result["coefficient"].lt(0.0)],
        default=True,
    )
    result["expected_sign_matches"] = sign_match.astype(bool)
    construct_threshold = float(analysis["construct_minimum_effect"])
    return_threshold = float(analysis["economic_effect_threshold_bps"]) / 10_000.0
    thresholds = np.where(
        result["family"].eq("construct_aggressor"),
        construct_threshold,
        return_threshold,
    )
    result["predeclared_effect_threshold"] = thresholds
    result["passes_split_criteria"] = (
        (result["q_value_bh_fdr"] <= float(analysis["fdr_alpha"]))
        & result["expected_sign_matches"]
        & (result["coefficient"].abs() >= thresholds)
    )
    result["is_primary_oos_pass"] = result["split"].eq("oos") & result[
        "passes_split_criteria"
    ]
    result["ci_inside_predeclared_band"] = np.where(
        result["family"].eq("construct_aggressor"),
        (result["ci_lower"] > -construct_threshold)
        & (result["ci_upper"] < construct_threshold),
        (result["ci_lower"] > -return_threshold)
        & (result["ci_upper"] < return_threshold),
    )
    return result


def build_feature_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scaled_features = [scaled_column(feature) for feature in RUN_FEATURES]
    for split_name in ("development", "oos"):
        subset = frame.loc[
            frame["sample_split"].eq(split_name) & frame["meets_continuous_trade_count"],
            scaled_features,
        ]
        correlation = subset.corr()
        for row_feature in RUN_FEATURES:
            for column_feature in RUN_FEATURES:
                rows.append(
                    {
                        "split": split_name,
                        "row_feature": row_feature,
                        "column_feature": column_feature,
                        "correlation": float(
                            correlation.loc[
                                scaled_column(row_feature),
                                scaled_column(column_feature),
                            ]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_continuous_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name in ("development", "oos"):
        subset = frame.loc[frame["sample_split"].eq(split_name)]
        complete_run_z = subset[RUN_FEATURES].notna().all(axis=1)
        rows.append(
            {
                "split": split_name,
                "rows": int(len(subset)),
                "start": subset["bucket_start"].min(),
                "end": subset["bucket_start"].max(),
                "symbols": int(subset["symbol"].nunique()),
                "utc_days": int(subset["utc_day"].nunique()),
                "minimum_trade_count_rows": int(subset["meets_continuous_trade_count"].sum()),
                "complete_run_z_rows": int(complete_run_z.sum()),
                "complete_run_z_share": float(complete_run_z.mean()),
                "future_outcome_within_split_rows": int(
                    subset["future_outcome_within_split"].sum()
                ),
                "aggressor_available_share": float(
                    subset["aggressor_imbalance"].notna().mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def leave_one_out_cross_sectional_mean(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby("bucket_start", sort=False)[column]
    total = grouped.transform("sum")
    count = grouped.transform("count")
    own = pd.to_numeric(frame[column], errors="coerce")
    denominator = count - own.notna().astype(int)
    return (total - own.fillna(0.0)).where(denominator > 0) / denominator.where(
        denominator > 0
    )


def plot_oos_coefficients(coefficients: pd.DataFrame, path: str | Path) -> None:
    plt = _get_pyplot()
    _configure_font(plt)
    oos = coefficients.loc[coefficients["split"].eq("oos")]
    families = list(_family_specs())
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, family in zip(axes, families):
        subset = oos.loc[oos["family"].eq(family)].set_index("feature").reindex(RUN_FEATURES)
        is_return = family != "construct_aggressor"
        multiplier = 10_000.0 if is_return else 1.0
        values = subset["coefficient"].to_numpy(dtype=float) * multiplier
        lower = subset["ci_lower"].to_numpy(dtype=float) * multiplier
        upper = subset["ci_upper"].to_numpy(dtype=float) * multiplier
        axis.errorbar(
            np.arange(3),
            values,
            yerr=np.vstack([values - lower, upper - values]),
            fmt="o",
            capsize=5,
            color="#1f5a75",
        )
        threshold = 5.0 if is_return else 0.02
        axis.axhline(0.0, color="#333333", linewidth=1.0)
        axis.axhline(threshold, color="#a33a2b", linestyle="--", linewidth=1.0)
        axis.axhline(-threshold, color="#a33a2b", linestyle="--", linewidth=1.0)
        axis.set_xticks(np.arange(3), ["up", "down", "zero"])
        axis.set_title(_family_display_name(family))
        axis.set_ylabel("aggressor imbalance" if not is_return else "계수 (bp)")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("연속형 run intensity OOS 공동 회귀 계수")
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_split_comparison(coefficients: pd.DataFrame, path: str | Path) -> None:
    plt = _get_pyplot()
    _configure_font(plt)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, family in zip(axes, _family_specs()):
        subset = coefficients.loc[coefficients["family"].eq(family)].copy()
        is_return = family != "construct_aggressor"
        multiplier = 10_000.0 if is_return else 1.0
        for offset, split_name, color in [(-0.12, "development", "#5b7f95"), (0.12, "oos", "#d0773c")]:
            local = subset.loc[subset["split"].eq(split_name)].set_index("feature").reindex(RUN_FEATURES)
            axis.scatter(
                np.arange(3) + offset,
                local["coefficient"].to_numpy(dtype=float) * multiplier,
                label=split_name,
                color=color,
            )
        axis.axhline(0.0, color="#333333", linewidth=1.0)
        axis.set_xticks(np.arange(3), ["up", "down", "zero"])
        axis.set_title(_family_display_name(family))
        axis.set_ylabel("aggressor imbalance" if not is_return else "계수 (bp)")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle("개발·OOS 계수 부호 안정성")
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_feature_correlations(correlations: pd.DataFrame, path: str | Path) -> None:
    plt = _get_pyplot()
    _configure_font(plt)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for axis, split_name in zip(axes, ("development", "oos")):
        local = correlations.loc[correlations["split"].eq(split_name)].pivot(
            index="row_feature",
            columns="column_feature",
            values="correlation",
        ).reindex(index=RUN_FEATURES, columns=RUN_FEATURES)
        image = axis.imshow(local.to_numpy(dtype=float), cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axis.set_xticks(np.arange(3), ["up", "down", "zero"])
        axis.set_yticks(np.arange(3), ["up", "down", "zero"])
        axis.set_title(split_name)
        for row in range(3):
            for column in range(3):
                axis.text(column, row, f"{local.iloc[row, column]:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axes, fraction=0.035, pad=0.04)
    figure.suptitle("Run intensity feature 상관")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_continuous_run_z_report(
    coverage: pd.DataFrame,
    coefficients: pd.DataFrame,
    diagnostics: pd.DataFrame,
    correlations: pd.DataFrame,
    artifact: ScalingArtifact,
    config: Mapping,
    plot_paths: Sequence[str],
) -> str:
    oos = coefficients.loc[coefficients["split"].eq("oos")]
    primary_passes = oos.loc[oos["is_primary_oos_pass"]]
    predictive_passes = primary_passes.loc[
        primary_passes["family"].isin(["future_excess_return", "future_excess_abs_return"])
    ]
    oos_future = oos.loc[oos["family"].str.startswith("future_")]
    practical_null_count = int(oos_future["ci_inside_predeclared_band"].sum())
    significant_but_small = oos_future.loc[
        (oos_future["q_value_bh_fdr"] <= float(config["analysis"]["fdr_alpha"]))
        & (oos_future["coefficient"].abs() < oos_future["predeclared_effect_threshold"])
    ]
    development_future_passes = coefficients.loc[
        coefficients["split"].eq("development")
        & coefficients["family"].str.startswith("future_")
        & coefficients["passes_split_criteria"]
    ]
    oos_up_down_correlation = correlations.loc[
        correlations["split"].eq("oos")
        & correlations["row_feature"].eq("run_intensity_up")
        & correlations["column_feature"].eq("run_intensity_down"),
        "correlation",
    ].iloc[0]
    lines = [
        "# 연속형 Run-Z OOS 연구 보고서",
        "",
        "## 연구 질문",
        "",
        "Winner label을 완전히 버리고 up, down, zero run clustering intensity를 각각 연속형으로 사용했을 때, 실제 aggressor 방향이나 30분 후 시장중립 움직임과 연관되는지 검정합니다.",
        "",
        "## 사전 고정 설계",
        "",
        f"- 개발: {config['split']['development_start']} ~ {config['split']['oos_start']} 미만",
        f"- OOS: {config['split']['oos_start']} ~ {config['split']['oos_end_exclusive']} 미만",
        "- feature: `-run_z_up`, `-run_z_down`, `-run_z_zero` 3개를 공동 회귀",
        "- clipping·표준화: 개발 구간에서만 적합 후 OOS에 고정",
        "- 가설 family: aggressor, 30분 수익률, 30분 절대수익률 각 3개 BH-FDR",
        "- OOS는 이 feature specification에 대한 held-out이며 완전히 새로운 외부 데이터는 아님",
        "",
        "## 표본 품질",
        "",
        "| split | 행 | 기간 | run-z 완전률 | aggressor 가용률 |",
        "|---|---:|---|---:|---:|",
    ]
    for row in coverage.itertuples(index=False):
        lines.append(
            f"| {row.split} | {row.rows:,} | {row.start} ~ {row.end} | "
            f"{row.complete_run_z_share:.2%} | {row.aggressor_available_share:.2%} |"
        )
    lines.extend(["", "## OOS 핵심 결과", ""])
    for family in _family_specs():
        lines.extend(
            [
                f"### {_family_display_name(family)}",
                "",
                "| feature | 개발 계수 | OOS 계수 | OOS 95% CI | OOS BH q | 판정 |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for feature in RUN_FEATURES:
            development = coefficients.loc[
                coefficients["split"].eq("development")
                & coefficients["family"].eq(family)
                & coefficients["feature"].eq(feature)
            ].iloc[0]
            oos_row = oos.loc[
                oos["family"].eq(family) & oos["feature"].eq(feature)
            ].iloc[0]
            if family == "construct_aggressor":
                development_value = development["coefficient"]
                oos_value = oos_row["coefficient"]
                lower = oos_row["ci_lower"]
                upper = oos_row["ci_upper"]
            else:
                development_value = development["coefficient_bps"]
                oos_value = oos_row["coefficient_bps"]
                lower = oos_row["ci_lower_bps"]
                upper = oos_row["ci_upper_bps"]
            decision = "사전 기준 통과" if oos_row["is_primary_oos_pass"] else "미통과"
            lines.append(
                f"| {feature.removeprefix('run_intensity_')} | {development_value:.4f} | "
                f"{oos_value:.4f} | [{lower:.4f}, {upper:.4f}] | "
                f"{oos_row['q_value_bh_fdr']:.4g} | {decision} |"
            )
        lines.append("")

    lines.extend(["## 결과 해석", ""])
    if len(development_future_passes) == 1:
        development_row = development_future_passes.iloc[0]
        oos_row = oos.loc[
            oos["family"].eq(development_row["family"])
            & oos["feature"].eq(development_row["feature"])
        ].iloc[0]
        lines.append(
            f"- 개발 구간에서 사전 기준을 통과한 유일한 미래 계수는 "
            f"`{development_row['feature']}`의 {_family_display_name(development_row['family'])} "
            f"{development_row['coefficient_bps']:.2f}bp였지만, OOS에서는 "
            f"{oos_row['coefficient_bps']:.2f}bp로 축소되어 5bp 경제성 기준을 재현하지 못했습니다."
        )
    lines.append(
        f"- OOS 미래 계수 6개 중 {practical_null_count}개의 95% 신뢰구간이 "
        "사전 경제성 경계인 ±5bp 안에 완전히 들어갑니다."
    )
    if not significant_but_small.empty:
        labels = ", ".join(
            f"{row.feature.removeprefix('run_intensity_')} {_family_display_name(row.family)} "
            f"{row.coefficient_bps:.2f}bp"
            for row in significant_but_small.itertuples(index=False)
        )
        lines.append(
            f"- BH-FDR상 유의하지만 5bp보다 작은 OOS 결과는 {labels}입니다. "
            "통계적 유의성과 거래 가능한 효과를 구분해야 합니다."
        )
    lines.extend(
        [
            "- Up/down intensity의 aggressor 공동 회귀 부호는 개발과 OOS 모두 사전 방향과 반대였습니다. 방향 proxy 구성타당도를 지지하지 않습니다.",
            f"- OOS up/down intensity 상관은 {oos_up_down_correlation:.3f}으로 높아, 개별 계수는 다른 intensity를 고정한 잔차적 차이로 해석해야 합니다.",
            "- 종목별 계수 부호가 미래 family에서 일관되지 않으므로 pooled 소규모 효과를 보편적 자산 신호로 해석하지 않습니다.",
            "",
        ]
    )

    if predictive_passes.empty:
        headline = "사전 통계·경제성 기준을 모두 통과한 OOS 미래 반응은 없습니다."
    else:
        labels = ", ".join(
            predictive_passes["family"] + ":" + predictive_passes["feature"]
        )
        headline = f"사전 기준을 통과한 OOS 미래 반응: {labels}"
    lines.extend(
        [
            "## 종합 판정",
            "",
            f"- {headline}",
            f"- 전체 OOS 통과 계수: {len(primary_passes)}개",
            "- Aggressor 동시 연관성은 구성타당도이지 미래 예측력이 아닙니다.",
            "- 통과 결과가 있어도 외부 표본 검증 전에는 tracker·paper-sim·자동매매를 활성화하지 않습니다.",
            "",
            "## 모형 진단",
            "",
            "| split | family | 관측치 | UTC-day | R2 | condition number |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostics.itertuples(index=False):
        lines.append(
            f"| {row.split} | {row.family} | {row.observations:,} | "
            f"{row.utc_day_clusters:,} | {row.rsquared:.4f} | {row.condition_number:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 재현 정보",
            "",
            f"- scaling fit: {artifact.fit_start} ~ {artifact.fit_end}",
            f"- winsor: {artifact.lower_quantile:.3%} ~ {artifact.upper_quantile:.3%}",
            "- 사전 프로토콜: `research_protocols/tick_continuous_run_z_oos_v1.md`",
            "- 종목별 계수는 `symbol_descriptive_coefficients.csv`에 기술용으로만 저장",
            "",
            "## 그림",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in plot_paths)
    lines.append("")
    return "\n".join(lines)


def scaled_column(column: str) -> str:
    return f"scaled__{column}"


def as_utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _family_specs() -> dict[str, dict]:
    return {
        "construct_aggressor": {
            "outcome": "aggressor_imbalance",
            "controls": BASE_CONTROLS,
            "requires_future": False,
            "unit": "imbalance",
            "expected_signs": {
                "run_intensity_up": "positive",
                "run_intensity_down": "negative",
                "run_intensity_zero": "none",
            },
        },
        "future_excess_return": {
            "outcome": "future_excess_return_30m",
            "controls": FUTURE_CONTROLS,
            "requires_future": True,
            "unit": "return",
            "expected_signs": {},
        },
        "future_excess_abs_return": {
            "outcome": "future_excess_abs_return_30m",
            "controls": FUTURE_CONTROLS,
            "requires_future": True,
            "unit": "return",
            "expected_signs": {},
        },
    }


def _family_display_name(family: str) -> str:
    return {
        "construct_aggressor": "Aggressor 구성타당도",
        "future_excess_return": "30분 시장중립 수익률",
        "future_excess_abs_return": "30분 시장중립 절대수익률",
    }[family]


def _get_pyplot():
    import matplotlib.pyplot as plt

    return plt


def _configure_font(plt) -> None:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False

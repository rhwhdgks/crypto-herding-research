from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from frequency_sensitivity import benjamini_hochberg
from regression import (
    _fit_ols,
    run_csad_regression,
    run_no_intercept_csad_regression,
    run_scsad_regression,
)


ORIGINAL_MODELS = ("standard_csad", "no_intercept_csad", "scsad")
ALL_AUDIT_MODELS = (*ORIGINAL_MODELS, "intercept_restored")
MODEL_FUNCTIONS = {
    "standard_csad": run_csad_regression,
    "no_intercept_csad": run_no_intercept_csad_regression,
    "scsad": run_scsad_regression,
}
TARGET_TERMS = {
    "standard_csad": "market_return_sq",
    "no_intercept_csad": "market_return_sq",
    "intercept_restored": "market_return_sq",
    "scsad": "market_return_cu",
}


@dataclass
class EmpiricalPanel:
    dataset_id: str
    provider: str
    sample: str
    universe_type: str
    weighting_class: str
    frequency: str
    rows: pd.DataFrame
    market: pd.Series
    csad: pd.Series
    loo_market: pd.Series
    loo_csad: pd.Series
    metrics: pd.DataFrame
    integrity: dict[str, object]


def validate_audit_config(config: Mapping, project_root: str | Path = ".") -> None:
    root = Path(project_root)
    protocol = root / str(config["protocol"]["path"])
    if not protocol.is_file():
        raise FileNotFoundError(f"Frozen protocol not found: {protocol}")
    datasets = list(config["empirical"]["datasets"])
    ids = [str(item["id"]) for item in datasets]
    if len(ids) != len(set(ids)):
        raise ValueError("Empirical dataset ids must be unique")
    for item in datasets:
        base = root / str(item["intermediate_dir"])
        for frequency in config["empirical"]["frequencies"]:
            path = base / f"{frequency}_member_rows.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"Empirical member rows not found: {path}")
        target_path = root / str(item["targets_path"])
        if not target_path.is_file():
            raise FileNotFoundError(f"Existing target result not found: {target_path}")
    repetitions = int(config["simulation"]["repetitions"])
    if repetitions < 100:
        raise ValueError("Monte Carlo repetitions must be at least 100")
    if set(config["empirical"]["models"]) != set(ORIGINAL_MODELS):
        raise ValueError("Audit must preserve all three preregistered CSAD models")


def load_empirical_panels(
    config: Mapping,
    project_root: str | Path = ".",
) -> dict[tuple[str, str], EmpiricalPanel]:
    root = Path(project_root)
    minimum_loo_assets = int(config["empirical"]["minimum_loo_assets"])
    panels: dict[tuple[str, str], EmpiricalPanel] = {}
    for dataset in config["empirical"]["datasets"]:
        for frequency in config["empirical"]["frequencies"]:
            path = root / str(dataset["intermediate_dir"]) / f"{frequency}_member_rows.parquet"
            rows = pd.read_parquet(path)
            panel = build_empirical_panel(
                rows,
                dataset,
                frequency=str(frequency),
                minimum_loo_assets=minimum_loo_assets,
                source_path=path,
            )
            panels[(panel.dataset_id, panel.frequency)] = panel
    return panels


def build_empirical_panel(
    member_rows: pd.DataFrame,
    metadata: Mapping,
    frequency: str,
    minimum_loo_assets: int = 3,
    source_path: str | Path | None = None,
) -> EmpiricalPanel:
    required = {
        "period",
        "asset_return",
        "return_weight",
        "valid_return_weight",
        "eligible",
    }
    missing = sorted(required.difference(member_rows.columns))
    if missing:
        raise ValueError(f"Member rows missing required columns: {missing}")

    rows = member_rows.copy()
    rows["period"] = pd.to_datetime(rows["period"], utc=True)
    asset_column = _asset_id_column(rows)
    valid = rows.loc[
        rows["eligible"].fillna(False).astype(bool)
        & rows["valid_return_weight"].fillna(False).astype(bool)
        & pd.to_numeric(rows["asset_return"], errors="coerce").notna()
        & pd.to_numeric(rows["return_weight"], errors="coerce").gt(0)
    ].copy()
    valid["asset_return"] = pd.to_numeric(valid["asset_return"], errors="raise")
    valid["return_weight"] = pd.to_numeric(valid["return_weight"], errors="raise")
    if valid.empty:
        raise ValueError(f"No valid rows in {source_path or metadata['id']}")
    if valid.duplicated(["period", asset_column]).any():
        raise ValueError(f"Duplicate period/asset rows in {source_path or metadata['id']}")

    valid["weighted_return"] = valid["asset_return"] * valid["return_weight"]
    aggregates = valid.groupby("period", sort=True).agg(
        weighted_sum=("weighted_return", "sum"),
        total_weight=("return_weight", "sum"),
        active_assets=(asset_column, "nunique"),
    )
    market = (aggregates["weighted_sum"] / aggregates["total_weight"]).rename("market_return")
    valid["market_return_rebuilt"] = valid["period"].map(market)
    valid["absolute_deviation"] = (
        valid["asset_return"] - valid["market_return_rebuilt"]
    ).abs()
    csad = valid.groupby("period")["absolute_deviation"].mean().rename("csad")

    valid["weight_share"] = valid["return_weight"] / valid["period"].map(
        aggregates["total_weight"]
    )
    valid["weight_share_sq"] = valid["weight_share"].pow(2)
    valid["is_btc"] = _btc_mask(valid)
    valid["btc_weight_component"] = valid["weight_share"].where(valid["is_btc"], 0.0)
    metrics = valid.groupby("period", sort=True).agg(
        active_assets=(asset_column, "nunique"),
        weight_hhi=("weight_share_sq", "sum"),
        max_weight=("weight_share", "max"),
        btc_weight=("btc_weight_component", "sum"),
    )
    metrics["market_return"] = market
    metrics["csad"] = csad

    denominator = valid["period"].map(aggregates["total_weight"]) - valid["return_weight"]
    numerator = valid["period"].map(aggregates["weighted_sum"]) - valid["weighted_return"]
    valid["loo_market_asset"] = (numerator / denominator).where(denominator.gt(0))
    enough_assets = valid["period"].map(aggregates["active_assets"]).ge(minimum_loo_assets)
    loo_valid = valid.loc[enough_assets & valid["loo_market_asset"].notna()].copy()
    loo_valid["loo_absolute_deviation"] = (
        loo_valid["asset_return"] - loo_valid["loo_market_asset"]
    ).abs()
    loo_market = loo_valid.groupby("period")["loo_market_asset"].mean().rename("market_return")
    loo_csad = loo_valid.groupby("period")["loo_absolute_deviation"].mean().rename("csad")

    integrity = _integrity_record(rows, market, csad, metadata, frequency, source_path)
    return EmpiricalPanel(
        dataset_id=str(metadata["id"]),
        provider=str(metadata["provider"]),
        sample=str(metadata["sample"]),
        universe_type=str(metadata["universe_type"]),
        weighting_class=str(metadata["weighting_class"]),
        frequency=str(frequency),
        rows=valid,
        market=market,
        csad=csad,
        loo_market=loo_market,
        loo_csad=loo_csad,
        metrics=metrics,
        integrity=integrity,
    )


def run_empirical_model_audit(
    panels: Mapping[tuple[str, str], EmpiricalPanel],
    config: Mapping,
) -> dict[str, pd.DataFrame]:
    diagnostic_rows: list[dict] = []
    coefficient_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    integrity_rows: list[dict] = []
    for panel in panels.values():
        integrity_rows.append(panel.integrity)
        metrics = panel.metrics.reset_index().copy()
        metrics.insert(0, "frequency", panel.frequency)
        metrics.insert(0, "dataset_id", panel.dataset_id)
        metric_frames.append(metrics)
        benchmarks = {
            "baseline": (panel.csad, panel.market),
            "leave_one_out": (panel.loo_csad, panel.loo_market),
        }
        for benchmark, (csad, market) in benchmarks.items():
            for model_name in ALL_AUDIT_MODELS:
                row, coefficients = fit_audit_model(
                    csad,
                    market,
                    model_name=model_name,
                    cov_type=str(config["empirical"]["cov_type"]),
                    hac_maxlags=config["empirical"]["hac_maxlags"],
                )
                row.update(_panel_labels(panel, benchmark))
                diagnostic_rows.append(row)
                coefficients = coefficients.assign(**_panel_labels(panel, benchmark))
                coefficient_frames.append(coefficients)

    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics["q_value_bh_fdr"] = np.nan
    original = diagnostics["model"].isin(ORIGINAL_MODELS)
    for _, indices in diagnostics.loc[original].groupby(["dataset_id", "benchmark"]).groups.items():
        diagnostics.loc[indices, "q_value_bh_fdr"] = benjamini_hochberg(
            diagnostics.loc[indices, "target_p_value"]
        )
    alpha = float(config["empirical"]["alpha"])
    diagnostics["supports_negative_relation"] = (
        diagnostics["target_coefficient"].lt(0)
        & diagnostics["q_value_bh_fdr"].le(alpha)
        & original
    )
    mechanical = build_intercept_mechanical_comparison(
        diagnostics,
        reduction_fraction=float(config["decision"]["intercept_effect_reduction_fraction"]),
    )
    return {
        "model_diagnostics": diagnostics,
        "coefficient_details": pd.concat(coefficient_frames, ignore_index=True),
        "intercept_mechanical_comparison": mechanical,
        "panel_metrics": pd.concat(metric_frames, ignore_index=True),
        "input_integrity": pd.DataFrame(integrity_rows),
    }


def fit_audit_model(
    csad: pd.Series,
    market_return: pd.Series,
    model_name: str,
    cov_type: str = "HAC",
    hac_maxlags: int | str | None = "auto",
) -> tuple[dict[str, object], pd.DataFrame]:
    if model_name in MODEL_FUNCTIONS:
        coefficients, _, frame, model, _ = MODEL_FUNCTIONS[model_name](
            csad,
            market_return,
            cov_type=cov_type,
            hac_maxlags=hac_maxlags,
        )
    elif model_name == "intercept_restored":
        frame = _base_regression_frame(csad, market_return)
        design = sm.add_constant(
            frame[["market_return", "abs_market_return", "market_return_sq"]],
            has_constant="add",
        )
        model = _fit_ols(
            frame["csad"], design, cov_type=cov_type, hac_maxlags=hac_maxlags
        )
        coefficients = _coefficient_table(model)
        frame["fitted_csad"] = model.fittedvalues
        frame["residual"] = model.resid
    else:
        raise ValueError(f"Unsupported audit model: {model_name}")

    target_term = TARGET_TERMS[model_name]
    target_scale, dependent_scale = _target_dependent_scales(frame, model_name)
    scale_ratio = target_scale / dependent_scale if dependent_scale > 0 else np.nan
    residual = np.asarray(model.resid, dtype=float)
    exog = np.asarray(model.model.exog, dtype=float)
    exog_names = list(model.model.exog_names)
    target_index = exog_names.index(target_term)
    vif_values = _vif_values(exog, exog_names)
    lag = _automatic_hac_lag(len(residual))
    jb = stats.jarque_bera(residual)
    lb_p = _ljung_box_p_value(residual, lag)
    bp_p = _breusch_pagan_p_value(residual, exog, exog_names)
    confidence = model.conf_int().loc[target_term]
    intercept = float(model.params["const"]) if "const" in model.params.index else np.nan
    intercept_p = float(model.pvalues["const"]) if "const" in model.pvalues.index else np.nan
    row = {
        "model": model_name,
        "target_term": target_term,
        "target_coefficient": float(model.params[target_term]),
        "target_standardized_coefficient": float(model.params[target_term]) * scale_ratio,
        "target_std_error": float(model.bse[target_term]),
        "target_standardized_std_error": float(model.bse[target_term]) * scale_ratio,
        "target_t_stat": float(model.tvalues[target_term]),
        "target_p_value": float(model.pvalues[target_term]),
        "target_ci_lower": float(confidence.iloc[0]),
        "target_ci_upper": float(confidence.iloc[1]),
        "intercept": intercept,
        "intercept_p_value": intercept_p,
        "nobs": int(model.nobs),
        "rsquared": float(model.rsquared),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "sse": float(np.dot(residual, residual)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "residual_mean": float(np.mean(residual)),
        "residual_skew": float(stats.skew(residual, bias=False)),
        "residual_excess_kurtosis": float(stats.kurtosis(residual, fisher=True, bias=False)),
        "jarque_bera_p_value": float(jb.pvalue),
        "durbin_watson": float(sm.stats.stattools.durbin_watson(residual)),
        "ljung_box_lag": lag,
        "ljung_box_p_value": lb_p,
        "breusch_pagan_p_value": bp_p,
        "target_vif": float(vif_values.get(target_term, np.nan)),
        "max_vif": float(max(vif_values.values())) if vif_values else np.nan,
        "condition_number_standardized": _standardized_condition_number(exog, exog_names),
        "design_rank": int(np.linalg.matrix_rank(exog)),
        "design_columns": len(exog_names),
        "cov_type": str(model.cov_type),
    }
    term_frame = coefficients.reset_index().rename(columns={"index": "term"})
    term_frame["model"] = model_name
    return row, term_frame


def build_intercept_mechanical_comparison(
    diagnostics: pd.DataFrame,
    reduction_fraction: float = 0.5,
) -> pd.DataFrame:
    keys = [
        "dataset_id",
        "provider",
        "sample",
        "universe_type",
        "weighting_class",
        "frequency",
        "benchmark",
    ]
    no_intercept = diagnostics.loc[diagnostics["model"].eq("no_intercept_csad")].copy()
    restored = diagnostics.loc[diagnostics["model"].eq("intercept_restored")].copy()
    keep = keys + [
        "target_coefficient",
        "target_standardized_coefficient",
        "target_p_value",
        "q_value_bh_fdr",
        "supports_negative_relation",
        "sse",
        "residual_mean",
    ]
    no_intercept = no_intercept[keep].rename(
        columns={column: f"no_intercept_{column}" for column in keep if column not in keys}
    )
    restored_keep = keys + [
        "target_coefficient",
        "target_standardized_coefficient",
        "target_p_value",
        "intercept",
        "intercept_p_value",
        "sse",
        "residual_mean",
    ]
    restored = restored[restored_keep].rename(
        columns={column: f"restored_{column}" for column in restored_keep if column not in keys}
    )
    result = no_intercept.merge(restored, on=keys, how="inner", validate="one_to_one")
    initial = result["no_intercept_target_standardized_coefficient"].abs()
    restored_abs = result["restored_target_standardized_coefficient"].abs()
    result["absolute_standardized_reduction_fraction"] = np.where(
        initial.gt(0), 1.0 - restored_abs / initial, np.nan
    )
    result["restored_raw_support"] = (
        result["restored_target_coefficient"].lt(0)
        & result["restored_target_p_value"].le(0.05)
    )
    result["support_disappears_after_restoring_intercept"] = (
        result["no_intercept_supports_negative_relation"]
        & ~result["restored_raw_support"]
    )
    result["material_magnitude_reduction"] = result[
        "absolute_standardized_reduction_fraction"
    ].ge(reduction_fraction)
    result["mechanical_change"] = (
        result["support_disappears_after_restoring_intercept"]
        | result["material_magnitude_reduction"]
    )
    result["sse_improvement_fraction"] = np.where(
        result["no_intercept_sse"].gt(0),
        1.0 - result["restored_sse"] / result["no_intercept_sse"],
        np.nan,
    )
    return result


def run_conditional_concentration_audit(
    panels: Mapping[tuple[str, str], EmpiricalPanel],
    config: Mapping,
) -> pd.DataFrame:
    rows: list[dict] = []
    metrics = ("active_assets", "weight_hhi", "btc_weight")
    minimum = int(config["empirical"]["minimum_conditional_observations"])
    tolerance = float(config["empirical"]["metric_variance_tolerance"])
    for panel in panels.values():
        for metric in metrics:
            for model_name in ORIGINAL_MODELS:
                row = _fit_conditional_model(
                    panel.csad,
                    panel.market,
                    panel.metrics[metric],
                    model_name,
                    minimum,
                    tolerance,
                    str(config["empirical"]["cov_type"]),
                    config["empirical"]["hac_maxlags"],
                )
                row.update(_panel_labels(panel, "baseline"))
                row["metric"] = metric
                rows.append(row)
    result = pd.DataFrame(rows)
    result["interaction_q_value_bh_fdr"] = np.nan
    available = result["status"].eq("estimated")
    for _, indices in result.loc[available].groupby(["dataset_id", "frequency"]).groups.items():
        result.loc[indices, "interaction_q_value_bh_fdr"] = benjamini_hochberg(
            result.loc[indices, "interaction_p_value"]
        )
    result["interaction_supported"] = (
        result["interaction_q_value_bh_fdr"].le(float(config["empirical"]["alpha"]))
        & available
    )
    return result


def run_volatility_regime_audit(
    panels: Mapping[tuple[str, str], EmpiricalPanel],
    config: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    regression_rows: list[dict] = []
    assignment_frames: list[pd.DataFrame] = []
    regime_cfg = config["volatility_regimes"]
    minimum = int(regime_cfg["minimum_regression_observations"])
    for panel in panels.values():
        assignment = assign_historical_volatility_regime(
            panel.market,
            frequency=panel.frequency,
            config=regime_cfg,
        )
        assignment_frame = assignment.reset_index()
        assignment_frame.insert(0, "frequency", panel.frequency)
        assignment_frame.insert(0, "dataset_id", panel.dataset_id)
        assignment_frames.append(assignment_frame)
        for regime in ("low", "mid", "high"):
            timestamps = assignment.index[assignment["volatility_regime"].eq(regime)]
            for model_name in ORIGINAL_MODELS:
                if len(timestamps) < minimum:
                    row = {
                        "model": model_name,
                        "regime": regime,
                        "status": "insufficient_observations",
                        "nobs": len(timestamps),
                        "target_coefficient": np.nan,
                        "target_standardized_coefficient": np.nan,
                        "target_p_value": np.nan,
                    }
                else:
                    fitted, _ = fit_audit_model(
                        panel.csad.reindex(timestamps),
                        panel.market.reindex(timestamps),
                        model_name,
                        cov_type=str(config["empirical"]["cov_type"]),
                        hac_maxlags=config["empirical"]["hac_maxlags"],
                    )
                    row = {
                        **fitted,
                        "regime": regime,
                        "status": "estimated",
                    }
                row.update(_panel_labels(panel, "baseline"))
                regression_rows.append(row)
    regressions = pd.DataFrame(regression_rows)
    regressions["q_value_bh_fdr"] = np.nan
    estimated = regressions["status"].eq("estimated")
    for _, indices in regressions.loc[estimated].groupby(["dataset_id", "frequency"]).groups.items():
        regressions.loc[indices, "q_value_bh_fdr"] = benjamini_hochberg(
            regressions.loc[indices, "target_p_value"]
        )
    regressions["supports_negative_relation"] = (
        regressions["target_coefficient"].lt(0)
        & regressions["q_value_bh_fdr"].le(float(config["empirical"]["alpha"]))
        & estimated
    )
    regressions["significant_opposite_positive"] = (
        regressions["target_coefficient"].gt(0)
        & regressions["q_value_bh_fdr"].le(float(config["empirical"]["alpha"]))
        & estimated
    )
    return regressions, pd.concat(assignment_frames, ignore_index=True)


def assign_historical_volatility_regime(
    market_return: pd.Series,
    frequency: str,
    config: Mapping,
) -> pd.DataFrame:
    cfg = config[str(frequency)]
    market = market_return.sort_index().dropna()
    realized = market.rolling(
        int(cfg["rolling_window"]), min_periods=int(cfg["rolling_minimum"])
    ).std(ddof=1).shift(1)
    prior_realized = realized.shift(1)
    low_quantile, high_quantile = [float(value) for value in config["quantiles"]]
    low = prior_realized.expanding(min_periods=int(cfg["expanding_minimum"])).quantile(
        low_quantile
    )
    high = prior_realized.expanding(min_periods=int(cfg["expanding_minimum"])).quantile(
        high_quantile
    )
    regime = pd.Series(pd.NA, index=market.index, dtype="string", name="volatility_regime")
    available = realized.notna() & low.notna() & high.notna()
    regime.loc[available & realized.le(low)] = "low"
    regime.loc[available & realized.gt(low) & realized.lt(high)] = "mid"
    regime.loc[available & realized.ge(high)] = "high"
    return pd.DataFrame(
        {
            "market_return": market,
            "lagged_realized_volatility": realized,
            "historical_low_threshold": low,
            "historical_high_threshold": high,
            "volatility_regime": regime,
        }
    )


def build_empirical_heterogeneity(
    model_diagnostics: pd.DataFrame,
    panel_metrics: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    empirical = model_diagnostics.loc[
        model_diagnostics["benchmark"].eq("baseline")
        & model_diagnostics["model"].isin(ORIGINAL_MODELS)
    ].copy()
    metric_summary = panel_metrics.groupby(["dataset_id", "frequency"], as_index=False).agg(
        mean_active_assets=("active_assets", "mean"),
        min_active_assets=("active_assets", "min"),
        max_active_assets=("active_assets", "max"),
        mean_weight_hhi=("weight_hhi", "mean"),
        mean_max_weight=("max_weight", "mean"),
        max_observed_weight=("max_weight", "max"),
        mean_btc_weight=("btc_weight", "mean"),
    )
    empirical = empirical.merge(
        metric_summary, on=["dataset_id", "frequency"], how="left", validate="many_to_one"
    )
    group_rows: list[dict] = []
    for moderator in ("provider", "sample", "universe_type", "weighting_class", "frequency"):
        grouped = empirical.groupby(["model", moderator], dropna=False)
        for keys, group in grouped:
            model_name, level = keys
            group_rows.append(
                {
                    "moderator": moderator,
                    "level": level,
                    "model": model_name,
                    "estimate_count": len(group),
                    "mean_standardized_coefficient": group[
                        "target_standardized_coefficient"
                    ].mean(),
                    "median_standardized_coefficient": group[
                        "target_standardized_coefficient"
                    ].median(),
                    "negative_share": group["target_standardized_coefficient"].lt(0).mean(),
                    "bh_support_share": group["supports_negative_relation"].mean(),
                }
            )
    random_effects = _random_effects_summary(empirical)
    meta_coefficients, meta_diagnostics = _descriptive_meta_regressions(empirical)
    univariate_coefficients, univariate_diagnostics = (
        _descriptive_univariate_meta_regressions(empirical)
    )
    return {
        "empirical_estimates_with_moderators": empirical,
        "heterogeneity_group_summary": pd.DataFrame(group_rows),
        "descriptive_random_effects": random_effects,
        "descriptive_meta_regression_coefficients": meta_coefficients,
        "descriptive_meta_regression_diagnostics": meta_diagnostics,
        "descriptive_univariate_meta_coefficients": univariate_coefficients,
        "descriptive_univariate_meta_diagnostics": univariate_diagnostics,
    }


def _fit_conditional_model(
    csad: pd.Series,
    market: pd.Series,
    metric: pd.Series,
    model_name: str,
    minimum_observations: int,
    variance_tolerance: float,
    cov_type: str,
    hac_maxlags: int | str | None,
) -> dict[str, object]:
    frame, dependent, design, target_term = _model_design(csad, market, model_name)
    frame = frame.join(metric.rename("metric"), how="inner").dropna(subset=["metric"])
    if len(frame) < minimum_observations or float(frame["metric"].std(ddof=1)) <= variance_tolerance:
        return {
            "model": model_name,
            "status": "not_identified",
            "nobs": len(frame),
            "interaction_coefficient": np.nan,
            "interaction_standardized_coefficient": np.nan,
            "interaction_t_stat": np.nan,
            "interaction_p_value": np.nan,
        }
    z_metric = (frame["metric"] - frame["metric"].mean()) / frame["metric"].std(ddof=1)
    base_columns = list(design.columns)
    design = design.reindex(frame.index).copy()
    design["metric_z"] = z_metric
    design["target_x_metric"] = frame[target_term] * z_metric
    dependent = dependent.reindex(frame.index)
    fitted = _fit_ols(dependent, design, cov_type=cov_type, hac_maxlags=hac_maxlags)
    interaction = design["target_x_metric"]
    scale = float(interaction.std(ddof=1) / dependent.std(ddof=1))
    return {
        "model": model_name,
        "status": "estimated",
        "nobs": int(fitted.nobs),
        "base_design_columns": ",".join(base_columns),
        "interaction_coefficient": float(fitted.params["target_x_metric"]),
        "interaction_standardized_coefficient": float(fitted.params["target_x_metric"]) * scale,
        "interaction_t_stat": float(fitted.tvalues["target_x_metric"]),
        "interaction_p_value": float(fitted.pvalues["target_x_metric"]),
        "condition_number": float(fitted.condition_number),
    }


def _model_design(
    csad: pd.Series,
    market: pd.Series,
    model_name: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, str]:
    frame = _base_regression_frame(csad, market)
    if model_name == "standard_csad":
        dependent = frame["csad"]
        design = sm.add_constant(frame[["abs_market_return", "market_return_sq"]])
        target = "market_return_sq"
    elif model_name == "no_intercept_csad":
        dependent = frame["csad"]
        design = frame[["market_return", "abs_market_return", "market_return_sq"]]
        target = "market_return_sq"
    elif model_name == "scsad":
        frame["scsad"] = frame["csad"].where(
            frame["market_return"].ge(0), -frame["csad"]
        )
        frame["market_return_cu"] = frame["market_return"].pow(3)
        dependent = frame["scsad"]
        design = sm.add_constant(
            frame[["market_return", "market_return_sq", "market_return_cu"]]
        )
        target = "market_return_cu"
    else:
        raise ValueError(f"Unsupported conditional model: {model_name}")
    return frame, dependent, design, target


def _random_effects_summary(empirical: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (model_name, frequency), group in empirical.groupby(["model", "frequency"]):
        theta = group["target_standardized_coefficient"].to_numpy(dtype=float)
        se = group["target_standardized_std_error"].to_numpy(dtype=float)
        valid = np.isfinite(theta) & np.isfinite(se) & (se > 0)
        theta = theta[valid]
        se = se[valid]
        k = len(theta)
        if k < 2:
            continue
        weights = 1.0 / se**2
        fixed = float(np.sum(weights * theta) / np.sum(weights))
        q_stat = float(np.sum(weights * (theta - fixed) ** 2))
        c_value = float(np.sum(weights) - np.sum(weights**2) / np.sum(weights))
        tau_sq = max((q_stat - (k - 1)) / c_value, 0.0) if c_value > 0 else 0.0
        random_weights = 1.0 / (se**2 + tau_sq)
        estimate = float(np.sum(random_weights * theta) / np.sum(random_weights))
        estimate_se = float(np.sqrt(1.0 / np.sum(random_weights)))
        i_squared = max((q_stat - (k - 1)) / q_stat, 0.0) * 100.0 if q_stat > 0 else 0.0
        rows.append(
            {
                "model": model_name,
                "frequency": frequency,
                "estimate_count": k,
                "random_effect_standardized_coefficient": estimate,
                "random_effect_std_error": estimate_se,
                "ci_lower": estimate - 1.96 * estimate_se,
                "ci_upper": estimate + 1.96 * estimate_se,
                "q_stat": q_stat,
                "tau_squared": tau_sq,
                "i_squared_percent": i_squared,
            }
        )
    return pd.DataFrame(rows)


def _descriptive_meta_regressions(
    empirical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficient_frames: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    continuous = ["mean_active_assets", "mean_weight_hhi", "mean_btc_weight"]
    categorical = ["provider", "frequency", "universe_type", "weighting_class"]
    for model_name, group in empirical.groupby("model"):
        working = group.copy()
        for column in continuous:
            scale = float(working[column].std(ddof=1))
            working[f"z_{column}"] = (
                (working[column] - working[column].mean()) / scale if scale > 0 else 0.0
            )
        design = pd.get_dummies(
            working[categorical + [f"z_{column}" for column in continuous]],
            columns=categorical,
            drop_first=True,
            dtype=float,
        )
        design = sm.add_constant(design, has_constant="add").astype(float)
        dependent = working["target_standardized_coefficient"].astype(float)
        fitted = sm.OLS(dependent, design).fit(cov_type="HC3")
        confidence = fitted.conf_int()
        coefficient_frames.append(
            pd.DataFrame(
                {
                    "term": fitted.params.index,
                    "coefficient": fitted.params.values,
                    "std_error_hc3": fitted.bse.values,
                    "t_stat": fitted.tvalues.values,
                    "p_value": fitted.pvalues.values,
                    "ci_lower": confidence.iloc[:, 0].values,
                    "ci_upper": confidence.iloc[:, 1].values,
                    "model": model_name,
                }
            )
        )
        diagnostics.append(
            {
                "model": model_name,
                "nobs": int(fitted.nobs),
                "design_columns": design.shape[1],
                "design_rank": int(np.linalg.matrix_rank(design.to_numpy())),
                "rank_deficient": bool(np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]),
                "rsquared": float(fitted.rsquared),
                "condition_number": float(fitted.condition_number),
                "interpretation": "descriptive_only_overlapping_and_confounding_samples",
            }
        )
    return pd.concat(coefficient_frames, ignore_index=True), pd.DataFrame(diagnostics)


def _descriptive_univariate_meta_regressions(
    empirical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    moderators = [
        "provider",
        "frequency",
        "universe_type",
        "weighting_class",
        "mean_active_assets",
        "mean_weight_hhi",
        "mean_btc_weight",
    ]
    coefficient_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict] = []
    categorical = {"provider", "frequency", "universe_type", "weighting_class"}
    for model_name, model_group in empirical.groupby("model"):
        for moderator in moderators:
            working = model_group[["target_standardized_coefficient", moderator]].dropna()
            if moderator in categorical:
                design = pd.get_dummies(
                    working[[moderator]],
                    columns=[moderator],
                    drop_first=True,
                    dtype=float,
                )
            else:
                scale = float(working[moderator].std(ddof=1))
                if scale <= 0:
                    diagnostic_rows.append(
                        {
                            "model": model_name,
                            "moderator": moderator,
                            "status": "not_identified",
                            "nobs": len(working),
                            "design_columns": 0,
                            "design_rank": 0,
                            "rank_deficient": False,
                            "rsquared": np.nan,
                        }
                    )
                    continue
                design = pd.DataFrame(
                    {
                        f"z_{moderator}": (
                            working[moderator] - working[moderator].mean()
                        )
                        / scale
                    },
                    index=working.index,
                )
            design = sm.add_constant(design, has_constant="add").astype(float)
            dependent = working["target_standardized_coefficient"].astype(float)
            fitted = sm.OLS(dependent, design).fit(cov_type="HC3")
            confidence = fitted.conf_int()
            coefficient_frames.append(
                pd.DataFrame(
                    {
                        "term": fitted.params.index,
                        "coefficient": fitted.params.values,
                        "std_error_hc3": fitted.bse.values,
                        "t_stat": fitted.tvalues.values,
                        "p_value": fitted.pvalues.values,
                        "ci_lower": confidence.iloc[:, 0].values,
                        "ci_upper": confidence.iloc[:, 1].values,
                        "model": model_name,
                        "moderator": moderator,
                    }
                )
            )
            rank = int(np.linalg.matrix_rank(design.to_numpy()))
            diagnostic_rows.append(
                {
                    "model": model_name,
                    "moderator": moderator,
                    "status": "estimated",
                    "nobs": int(fitted.nobs),
                    "design_columns": design.shape[1],
                    "design_rank": rank,
                    "rank_deficient": rank < design.shape[1],
                    "rsquared": float(fitted.rsquared),
                }
            )
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    coefficients["q_value_bh_fdr"] = np.nan
    tested = coefficients["term"].ne("const")
    for _, indices in coefficients.loc[tested].groupby("model").groups.items():
        coefficients.loc[indices, "q_value_bh_fdr"] = benjamini_hochberg(
            coefficients.loc[indices, "p_value"]
        )
    coefficients["bh_supported"] = (
        coefficients["q_value_bh_fdr"].le(0.05) & tested
    )
    return coefficients, pd.DataFrame(diagnostic_rows)


def _panel_labels(panel: EmpiricalPanel, benchmark: str) -> dict[str, str]:
    return {
        "dataset_id": panel.dataset_id,
        "provider": panel.provider,
        "sample": panel.sample,
        "universe_type": panel.universe_type,
        "weighting_class": panel.weighting_class,
        "frequency": panel.frequency,
        "benchmark": benchmark,
    }


def _base_regression_frame(csad: pd.Series, market: pd.Series) -> pd.DataFrame:
    frame = pd.concat(
        [csad.rename("csad"), market.rename("market_return")], axis=1, join="inner"
    ).dropna()
    frame["abs_market_return"] = frame["market_return"].abs()
    frame["market_return_sq"] = frame["market_return"].pow(2)
    return frame


def _target_dependent_scales(frame: pd.DataFrame, model_name: str) -> tuple[float, float]:
    if model_name == "scsad":
        target = frame["market_return"].pow(3)
        dependent = frame["csad"].where(frame["market_return"].ge(0), -frame["csad"])
    else:
        target = frame["market_return"].pow(2)
        dependent = frame["csad"]
    return float(target.std(ddof=1)), float(dependent.std(ddof=1))


def _coefficient_table(model) -> pd.DataFrame:
    confidence = model.conf_int()
    frame = pd.DataFrame(
        {
            "coefficient": model.params,
            "std_error": model.bse,
            "t_stat": model.tvalues,
            "p_value": model.pvalues,
            "ci_lower": confidence.iloc[:, 0],
            "ci_upper": confidence.iloc[:, 1],
        }
    )
    frame.index.name = "term"
    return frame


def _vif_values(exog: np.ndarray, names: Sequence[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for index, name in enumerate(names):
        if name == "const":
            continue
        try:
            value = float(variance_inflation_factor(exog, index))
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            value = np.nan
        values[name] = value
    return values


def _standardized_condition_number(exog: np.ndarray, names: Sequence[str]) -> float:
    standardized = np.asarray(exog, dtype=float).copy()
    for index, name in enumerate(names):
        if name == "const":
            continue
        scale = float(np.std(standardized[:, index], ddof=1))
        if scale > 0:
            standardized[:, index] = (
                standardized[:, index] - np.mean(standardized[:, index])
            ) / scale
    return float(np.linalg.cond(standardized))


def _automatic_hac_lag(nobs: int) -> int:
    return max(int(np.floor(4.0 * ((nobs / 100.0) ** (2.0 / 9.0)))), 1)


def _ljung_box_p_value(residual: np.ndarray, lag: int) -> float:
    if len(residual) <= lag + 1:
        return np.nan
    try:
        result = acorr_ljungbox(residual, lags=[lag], return_df=True)
        return float(result["lb_pvalue"].iloc[-1])
    except (ValueError, np.linalg.LinAlgError):
        return np.nan


def _breusch_pagan_p_value(
    residual: np.ndarray,
    exog: np.ndarray,
    names: Sequence[str],
) -> float:
    diagnostic_exog = exog
    if "const" not in names:
        diagnostic_exog = sm.add_constant(exog, has_constant="add")
    try:
        return float(het_breuschpagan(residual, diagnostic_exog)[1])
    except (ValueError, np.linalg.LinAlgError):
        return np.nan


def _asset_id_column(frame: pd.DataFrame) -> str:
    for candidate in ("asset_key", "cmc_id", "instrument_id", "symbol", "research_symbol"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("Could not identify asset id column")


def _btc_mask(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    if "base_asset" in frame.columns:
        mask |= frame["base_asset"].astype(str).str.upper().eq("BTC")
    if "research_symbol" in frame.columns:
        mask |= frame["research_symbol"].astype(str).str.upper().eq("BTC")
    if "symbol" in frame.columns:
        symbol = frame["symbol"].astype(str).str.upper().str.replace("-", "", regex=False)
        mask |= symbol.eq("BTC") | symbol.str.startswith("BTCUSDT")
    if "instrument_id" in frame.columns:
        mask |= frame["instrument_id"].astype(str).str.upper().str.startswith("BTC-")
    return mask


def _integrity_record(
    original_rows: pd.DataFrame,
    market: pd.Series,
    csad: pd.Series,
    metadata: Mapping,
    frequency: str,
    source_path: str | Path | None,
) -> dict[str, object]:
    stored = original_rows.copy()
    stored["period"] = pd.to_datetime(stored["period"], utc=True)
    market_difference = np.nan
    csad_difference = np.nan
    if "market_return" in stored.columns:
        stored_market = stored.groupby("period")["market_return"].first().dropna()
        common = market.index.intersection(stored_market.index)
        if len(common):
            market_difference = float((market.loc[common] - stored_market.loc[common]).abs().max())
    if "csad" in stored.columns:
        stored_csad = stored.groupby("period")["csad"].first().dropna()
        common = csad.index.intersection(stored_csad.index)
        if len(common):
            csad_difference = float((csad.loc[common] - stored_csad.loc[common]).abs().max())
    tolerance = 1e-12
    return {
        "dataset_id": str(metadata["id"]),
        "frequency": str(frequency),
        "source_path": str(source_path or "in_memory"),
        "source_rows": len(original_rows),
        "analysis_periods": len(market),
        "start": market.index.min(),
        "end": market.index.max(),
        "max_abs_market_rebuild_difference": market_difference,
        "max_abs_csad_rebuild_difference": csad_difference,
        "rebuild_matches_stored": bool(
            (np.isnan(market_difference) or market_difference <= tolerance)
            and (np.isnan(csad_difference) or csad_difference <= tolerance)
        ),
    }

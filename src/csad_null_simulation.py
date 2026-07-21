from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats

from csad_specification_audit import ALL_AUDIT_MODELS, ORIGINAL_MODELS
from frequency_sensitivity import benjamini_hochberg


LOGGER = logging.getLogger(__name__)


def run_null_monte_carlo(config: Mapping) -> dict[str, pd.DataFrame]:
    simulation_cfg = config["simulation"]
    seed = int(simulation_cfg["seed"])
    repetitions = int(simulation_cfg["repetitions"])
    alpha = float(simulation_cfg["alpha"])
    result_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    dgp_items = list(simulation_cfg["dgp"].items())
    scenarios = list(simulation_cfg["scenarios"])
    for dgp_index, (dgp_name, dgp_cfg) in enumerate(dgp_items):
        for scenario_index, scenario in enumerate(scenarios):
            for repetition in range(repetitions):
                rng = np.random.default_rng(
                    np.random.SeedSequence([seed, dgp_index, scenario_index, repetition])
                )
                returns = simulate_null_returns(
                    dgp_name,
                    dgp_cfg,
                    observations=int(scenario["observations"]),
                    assets=int(scenario["assets"]),
                    rng=rng,
                )
                weights = simulate_weights(
                    returns,
                    weighting=str(scenario["weighting"]),
                    config=simulation_cfg["weight_process"],
                    rng=rng,
                )
                market, csad = construct_market_and_csad(returns, weights)
                replicate_rows = []
                for model_name in ALL_AUDIT_MODELS:
                    fitted = fit_fast_hac_model(csad, market, model_name)
                    fitted.update(
                        {
                            "dgp": dgp_name,
                            "scenario": str(scenario["id"]),
                            "frequency": str(scenario["frequency"]),
                            "observations": int(scenario["observations"]),
                            "assets": int(scenario["assets"]),
                            "weighting": str(scenario["weighting"]),
                            "repetition": repetition,
                        }
                    )
                    replicate_rows.append(fitted)
                p_values = pd.Series(
                    [
                        row["target_p_value"]
                        for row in replicate_rows
                        if row["model"] in ORIGINAL_MODELS
                    ],
                    index=[
                        row["model"]
                        for row in replicate_rows
                        if row["model"] in ORIGINAL_MODELS
                    ],
                    dtype=float,
                )
                q_values = benjamini_hochberg(p_values)
                for row in replicate_rows:
                    row["q_value_bh_fdr"] = (
                        float(q_values.loc[row["model"]])
                        if row["model"] in ORIGINAL_MODELS
                        else np.nan
                    )
                    row["raw_false_positive"] = bool(
                        row["target_coefficient"] < 0
                        and row["target_p_value"] <= alpha
                    )
                    row["bh_false_positive"] = bool(
                        row["model"] in ORIGINAL_MODELS
                        and row["target_coefficient"] < 0
                        and row["q_value_bh_fdr"] <= alpha
                    )
                    result_rows.append(row)
                diagnostic_rows.append(
                    {
                        "dgp": dgp_name,
                        "scenario": str(scenario["id"]),
                        "frequency": str(scenario["frequency"]),
                        "observations": int(scenario["observations"]),
                        "assets": int(scenario["assets"]),
                        "weighting": str(scenario["weighting"]),
                        "repetition": repetition,
                        **simulation_diagnostics(returns, weights, market),
                    }
                )
            LOGGER.info(
                "Null Monte Carlo complete: dgp=%s scenario=%s repetitions=%d",
                dgp_name,
                scenario["id"],
                repetitions,
            )
    simulations = pd.DataFrame(result_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    return {
        "simulation_replicates": simulations,
        "simulation_diagnostics": diagnostics,
        "false_positive_summary": summarize_false_positive_rates(simulations, simulation_cfg),
        "simulation_diagnostic_summary": summarize_simulation_diagnostics(diagnostics),
        "null_intercept_mechanical_summary": summarize_null_intercept_mechanics(
            simulations, simulation_cfg
        ),
    }


def simulate_null_returns(
    dgp_name: str,
    dgp_config: Mapping,
    observations: int,
    assets: int,
    rng: np.random.Generator,
) -> np.ndarray:
    idiosyncratic_scale = float(dgp_config["idiosyncratic_scale"])
    asset_scale = idiosyncratic_scale * np.exp(rng.normal(0.0, 0.25, size=assets))
    if dgp_name == "independent_gaussian":
        return rng.normal(size=(observations, assets)) * asset_scale

    loading = rng.normal(
        float(dgp_config["factor_loading_mean"]),
        float(dgp_config["factor_loading_std"]),
        size=assets,
    )
    factor_scale = float(dgp_config["factor_scale"])
    if dgp_name == "common_factor":
        factor = rng.normal(size=observations) * factor_scale
        idiosyncratic = rng.normal(size=(observations, assets)) * asset_scale
        return factor[:, None] * loading[None, :] + idiosyncratic
    if dgp_name == "heteroskedastic_factor":
        phi = float(dgp_config["log_volatility_phi"])
        innovation = float(dgp_config["log_volatility_innovation"])
        log_volatility = np.zeros(observations)
        for index in range(1, observations):
            log_volatility[index] = (
                phi * log_volatility[index - 1] + rng.normal(0.0, innovation)
            )
        volatility = np.exp(log_volatility - np.mean(log_volatility))
        factor = rng.normal(size=observations) * factor_scale
        idiosyncratic = rng.normal(size=(observations, assets)) * asset_scale
        return volatility[:, None] * (
            factor[:, None] * loading[None, :] + idiosyncratic
        )
    if dgp_name == "fat_tail_correlated":
        degrees = float(dgp_config["degrees_of_freedom"])
        variance_scale = np.sqrt(degrees / (degrees - 2.0))
        factor = rng.standard_t(degrees, size=observations) / variance_scale * factor_scale
        idiosyncratic = (
            rng.standard_t(degrees, size=(observations, assets))
            / variance_scale
            * asset_scale
        )
        return factor[:, None] * loading[None, :] + idiosyncratic
    raise ValueError(f"Unsupported null DGP: {dgp_name}")


def simulate_weights(
    returns: np.ndarray,
    weighting: str,
    config: Mapping,
    rng: np.random.Generator,
) -> np.ndarray:
    observations, assets = returns.shape
    if weighting == "equal":
        return np.ones((observations, assets), dtype=float)

    initial_sigma = float(config["lognormal_sigma"])
    minimum = float(config["minimum_weight"])
    state = rng.normal(0.0, initial_sigma, size=assets)
    weights = np.empty((observations, assets), dtype=float)
    if weighting == "lagged_lognormal_liquidity":
        phi = float(config["liquidity_ar1"])
        innovation = float(config["liquidity_innovation"])
        for index in range(observations):
            weights[index] = _softmax_weights(state, minimum)
            state = phi * state + rng.normal(0.0, innovation, size=assets)
            state -= np.mean(state)
        return weights

    if weighting not in {"contemporaneous_evolving_size", "lagged_evolving_size"}:
        raise ValueError(f"Unsupported simulation weighting: {weighting}")
    persistence = 0.995
    for index in range(observations):
        if weighting == "lagged_evolving_size":
            weights[index] = _softmax_weights(state, minimum)
        state = persistence * state + np.clip(returns[index], -0.5, 0.5)
        state -= np.mean(state)
        if weighting == "contemporaneous_evolving_size":
            weights[index] = _softmax_weights(state, minimum)
    return weights


def construct_market_and_csad(
    returns: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total = weights.sum(axis=1)
    market = np.sum(weights * returns, axis=1) / total
    csad = np.mean(np.abs(returns - market[:, None]), axis=1)
    return market, csad


def fit_fast_hac_model(
    csad: np.ndarray,
    market: np.ndarray,
    model_name: str,
) -> dict[str, object]:
    market = np.asarray(market, dtype=float)
    csad = np.asarray(csad, dtype=float)
    if model_name == "standard_csad":
        dependent = csad
        design = np.column_stack([np.ones(len(market)), np.abs(market), market**2])
        target_index = 2
        intercept_index: int | None = 0
        target_term = "market_return_sq"
    elif model_name == "no_intercept_csad":
        dependent = csad
        design = np.column_stack([market, np.abs(market), market**2])
        target_index = 2
        intercept_index = None
        target_term = "market_return_sq"
    elif model_name == "intercept_restored":
        dependent = csad
        design = np.column_stack(
            [np.ones(len(market)), market, np.abs(market), market**2]
        )
        target_index = 3
        intercept_index = 0
        target_term = "market_return_sq"
    elif model_name == "scsad":
        dependent = np.where(market >= 0.0, csad, -csad)
        design = np.column_stack(
            [np.ones(len(market)), market, market**2, market**3]
        )
        target_index = 3
        intercept_index = 0
        target_term = "market_return_cu"
    else:
        raise ValueError(f"Unsupported simulation model: {model_name}")

    beta, standard_error = _ols_hac(dependent, design)
    target_se = float(standard_error[target_index])
    target_t = float(beta[target_index] / target_se) if target_se > 0 else np.nan
    target_p = float(2.0 * stats.norm.sf(abs(target_t))) if np.isfinite(target_t) else np.nan
    target = market**3 if target_term == "market_return_cu" else market**2
    scale_ratio = float(np.std(target, ddof=1) / np.std(dependent, ddof=1))
    residual = dependent - design @ beta
    intercept = float(beta[intercept_index]) if intercept_index is not None else np.nan
    return {
        "model": model_name,
        "target_term": target_term,
        "target_coefficient": float(beta[target_index]),
        "target_standardized_coefficient": float(beta[target_index]) * scale_ratio,
        "target_std_error": target_se,
        "target_standardized_std_error": target_se * scale_ratio,
        "target_t_stat": target_t,
        "target_p_value": target_p,
        "intercept": intercept,
        "sse": float(np.dot(residual, residual)),
        "residual_mean": float(np.mean(residual)),
        "hac_maxlags": _automatic_hac_lag(len(dependent)),
    }


def summarize_false_positive_rates(
    simulations: pd.DataFrame,
    config: Mapping,
) -> pd.DataFrame:
    rows: list[dict] = []
    grouping = [
        "dgp",
        "scenario",
        "frequency",
        "observations",
        "assets",
        "weighting",
        "model",
    ]
    for keys, group in simulations.groupby(grouping, sort=False):
        row = dict(zip(grouping, keys, strict=True))
        count = len(group)
        raw_rate = float(group["raw_false_positive"].mean())
        bh_rate = (
            float(group["bh_false_positive"].mean())
            if row["model"] in ORIGINAL_MODELS
            else np.nan
        )
        raw_low, raw_high = _wilson_interval(int(group["raw_false_positive"].sum()), count)
        if row["model"] in ORIGINAL_MODELS:
            bh_low, bh_high = _wilson_interval(int(group["bh_false_positive"].sum()), count)
        else:
            bh_low, bh_high = np.nan, np.nan
        row.update(
            {
                "repetitions": count,
                "raw_false_positive_rate": raw_rate,
                "raw_wilson_ci_lower": raw_low,
                "raw_wilson_ci_upper": raw_high,
                "bh_false_positive_rate": bh_rate,
                "bh_wilson_ci_lower": bh_low,
                "bh_wilson_ci_upper": bh_high,
                "fpr_classification": _fpr_classification(bh_rate, config),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_null_intercept_mechanics(
    simulations: pd.DataFrame,
    config: Mapping,
) -> pd.DataFrame:
    keys = ["dgp", "scenario", "frequency", "observations", "assets", "weighting", "repetition"]
    no_intercept = simulations.loc[simulations["model"].eq("no_intercept_csad")].copy()
    restored = simulations.loc[simulations["model"].eq("intercept_restored")].copy()
    merged = no_intercept.merge(restored, on=keys, suffixes=("_no_intercept", "_restored"))
    initial = merged["target_standardized_coefficient_no_intercept"].abs()
    restored_abs = merged["target_standardized_coefficient_restored"].abs()
    merged["magnitude_reduction"] = np.where(
        initial.gt(0), 1.0 - restored_abs / initial, np.nan
    )
    merged["restored_raw_false_positive"] = merged["raw_false_positive_restored"]
    merged["support_disappears"] = (
        merged["raw_false_positive_no_intercept"]
        & ~merged["restored_raw_false_positive"]
    )
    reduction = float(config.get("intercept_effect_reduction_fraction", 0.5))
    merged["mechanical_change"] = merged["support_disappears"] | merged[
        "magnitude_reduction"
    ].ge(reduction)
    grouping = ["dgp", "scenario", "frequency", "observations", "assets", "weighting"]
    return merged.groupby(grouping, as_index=False).agg(
        repetitions=("repetition", "size"),
        no_intercept_raw_false_positive_rate=("raw_false_positive_no_intercept", "mean"),
        restored_raw_false_positive_rate=("restored_raw_false_positive", "mean"),
        support_disappearance_rate=("support_disappears", "mean"),
        median_magnitude_reduction=("magnitude_reduction", "median"),
        mechanical_change_rate=("mechanical_change", "mean"),
        mean_no_intercept_residual=("residual_mean_no_intercept", "mean"),
        mean_restored_residual=("residual_mean_restored", "mean"),
    )


def summarize_simulation_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    return diagnostics.groupby(
        ["dgp", "scenario", "frequency", "observations", "assets", "weighting"],
        as_index=False,
    ).agg(
        repetitions=("repetition", "size"),
        mean_asset_volatility=("mean_asset_volatility", "mean"),
        mean_pairwise_correlation=("mean_pairwise_correlation", "mean"),
        mean_excess_kurtosis=("cross_asset_excess_kurtosis", "mean"),
        mean_squared_market_return_acf1=("squared_market_return_acf1", "mean"),
        mean_weight_hhi=("mean_weight_hhi", "mean"),
        mean_max_weight=("mean_max_weight", "mean"),
    )


def simulation_diagnostics(
    returns: np.ndarray,
    weights: np.ndarray,
    market: np.ndarray,
) -> dict[str, float]:
    standard_deviation = np.std(returns, axis=0, ddof=1)
    correlation = np.corrcoef(returns, rowvar=False)
    upper = correlation[np.triu_indices_from(correlation, k=1)]
    shares = weights / weights.sum(axis=1, keepdims=True)
    squared_market = market**2
    squared_acf = (
        float(np.corrcoef(squared_market[:-1], squared_market[1:])[0, 1])
        if len(squared_market) > 2
        else np.nan
    )
    return {
        "mean_asset_volatility": float(np.mean(standard_deviation)),
        "mean_pairwise_correlation": float(np.nanmean(upper)),
        "cross_asset_excess_kurtosis": float(
            stats.kurtosis(returns.ravel(), fisher=True, bias=False)
        ),
        "squared_market_return_acf1": squared_acf,
        "mean_weight_hhi": float(np.mean(np.sum(shares**2, axis=1))),
        "mean_max_weight": float(np.mean(np.max(shares, axis=1))),
    }


def compare_empirical_to_null(
    empirical_diagnostics: pd.DataFrame,
    simulations: pd.DataFrame,
    config: Mapping,
) -> pd.DataFrame:
    empirical = empirical_diagnostics.loc[
        empirical_diagnostics["benchmark"].eq("baseline")
        & empirical_diagnostics["model"].isin(ORIGINAL_MODELS)
    ].copy()
    rows: list[dict] = []
    mapping = config["null_mapping"]
    for empirical_row in empirical.itertuples(index=False):
        scenario = str(mapping[empirical_row.dataset_id][empirical_row.frequency])
        matching = simulations.loc[
            simulations["scenario"].eq(scenario)
            & simulations["model"].eq(empirical_row.model)
        ]
        for dgp, group in matching.groupby("dgp"):
            null = group["target_standardized_coefficient"].to_numpy(dtype=float)
            observed = float(empirical_row.target_standardized_coefficient)
            rows.append(
                {
                    "dataset_id": empirical_row.dataset_id,
                    "provider": empirical_row.provider,
                    "sample": empirical_row.sample,
                    "universe_type": empirical_row.universe_type,
                    "weighting_class": empirical_row.weighting_class,
                    "frequency": empirical_row.frequency,
                    "model": empirical_row.model,
                    "dgp": dgp,
                    "scenario": scenario,
                    "empirical_standardized_coefficient": observed,
                    "empirical_hac_p_value": empirical_row.target_p_value,
                    "null_repetitions": len(null),
                    "null_mean": float(np.mean(null)),
                    "null_std": float(np.std(null, ddof=1)),
                    "null_ci_lower": float(np.quantile(null, 0.025)),
                    "null_median": float(np.median(null)),
                    "null_ci_upper": float(np.quantile(null, 0.975)),
                    "empirical_percentile_in_null": float(np.mean(null <= observed)),
                    "negative_tail_monte_carlo_p": float(
                        (1 + np.sum(null <= observed)) / (len(null) + 1)
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result["negative_tail_q_value_bh_fdr"] = np.nan
    for _, indices in result.groupby("dataset_id").groups.items():
        result.loc[indices, "negative_tail_q_value_bh_fdr"] = benjamini_hochberg(
            result.loc[indices, "negative_tail_monte_carlo_p"]
        )
    result["more_negative_than_null_bh"] = (
        result["empirical_standardized_coefficient"].lt(0)
        & result["negative_tail_q_value_bh_fdr"].le(float(config["empirical"]["alpha"]))
    )
    return result


def _ols_hac(dependent: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inverse = np.linalg.pinv(design.T @ design)
    beta = inverse @ design.T @ dependent
    residual = dependent - design @ beta
    score = design * residual[:, None]
    meat = score.T @ score
    maxlags = _automatic_hac_lag(len(dependent))
    for lag in range(1, maxlags + 1):
        weight = 1.0 - lag / (maxlags + 1.0)
        covariance = score[lag:].T @ score[:-lag]
        meat += weight * (covariance + covariance.T)
    covariance_matrix = inverse @ meat @ inverse
    standard_error = np.sqrt(np.maximum(np.diag(covariance_matrix), 0.0))
    return beta, standard_error


def _automatic_hac_lag(nobs: int) -> int:
    return max(int(np.floor(4.0 * ((nobs / 100.0) ** (2.0 / 9.0)))), 1)


def _softmax_weights(state: np.ndarray, minimum: float) -> np.ndarray:
    shifted = state - np.max(state)
    weights = np.maximum(np.exp(shifted), minimum)
    return weights / np.sum(weights)


def _wilson_interval(successes: int, repetitions: int) -> tuple[float, float]:
    if repetitions <= 0:
        return np.nan, np.nan
    z_value = 1.959963984540054
    proportion = successes / repetitions
    denominator = 1.0 + z_value**2 / repetitions
    centre = (proportion + z_value**2 / (2.0 * repetitions)) / denominator
    half_width = (
        z_value
        * np.sqrt(
            proportion * (1.0 - proportion) / repetitions
            + z_value**2 / (4.0 * repetitions**2)
        )
        / denominator
    )
    return float(max(centre - half_width, 0.0)), float(min(centre + half_width, 1.0))


def _fpr_classification(rate: float, config: Mapping) -> str:
    if not np.isfinite(rate):
        return "diagnostic_only"
    if rate >= float(config["false_positive_material_threshold"]):
        return "materially_inflated"
    if rate > float(config["false_positive_inflated_threshold"]):
        return "inflated"
    return "within_preregistered_tolerance"

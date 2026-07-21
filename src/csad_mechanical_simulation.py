from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from csad_mechanical_derivation import (
    MECHANICAL_MODELS,
    build_gaussian_theory_table,
)
from csad_null_simulation import construct_market_and_csad, fit_fast_hac_model
from frequency_sensitivity import benjamini_hochberg


LOGGER = logging.getLogger(__name__)


def run_convergence_simulation(config: Mapping) -> dict[str, pd.DataFrame]:
    simulation = config["convergence_simulation"]
    scenarios = [
        {
            "id": f"n{int(assets)}_t{int(simulation['observations'])}_equal",
            "observations": int(simulation["observations"]),
            "assets": int(assets),
            "weighting": str(simulation["weighting"]),
        }
        for assets in simulation["assets"]
    ]
    replicates, diagnostics = _run_simulation_grid(
        dgp_names=[str(simulation["dgp"])],
        scenarios=scenarios,
        repetitions=int(simulation["repetitions"]),
        seed=int(simulation["seed"]),
        config=config,
        phase="convergence",
    )
    theory = build_gaussian_theory_table(
        simulation["assets"], float(config["theory"]["sigma"])
    )
    summary = summarize_simulation_replicates(
        replicates,
        confidence=float(config["theory"]["convergence_confidence"]),
        theory=theory,
    )
    gates = build_convergence_gates(summary, config["theory"])
    return {
        "convergence_replicates": replicates,
        "convergence_diagnostics": diagnostics,
        "convergence_summary": summary,
        "convergence_gates": gates,
    }


def run_robustness_simulation(config: Mapping) -> dict[str, pd.DataFrame]:
    simulation = config["robustness_simulation"]
    dgp_names = list(config["dgp"])
    replicates, diagnostics = _run_simulation_grid(
        dgp_names=dgp_names,
        scenarios=list(simulation["scenarios"]),
        repetitions=int(simulation["repetitions"]),
        seed=int(simulation["seed"]),
        config=config,
        phase="robustness",
    )
    summary = summarize_simulation_replicates(replicates, confidence=0.95)
    diagnostic_summary = summarize_diagnostics(diagnostics)
    robustness_cells, robustness_decision = build_symmetric_robustness_gate(
        summary, simulation
    )
    return {
        "robustness_replicates": replicates,
        "robustness_diagnostics": diagnostics,
        "robustness_summary": summary,
        "robustness_diagnostic_summary": diagnostic_summary,
        "symmetric_robustness_cells": robustness_cells,
        "symmetric_robustness_decision": robustness_decision,
    }


def simulate_mechanical_null_returns(
    dgp_name: str,
    dgp_config: Mapping,
    observations: int,
    assets: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if observations < 10:
        raise ValueError("observations must be at least 10")
    if assets < 3:
        raise ValueError("assets must be at least 3")

    if dgp_name == "independent_gaussian":
        scale = float(dgp_config["idiosyncratic_scale"])
        return rng.normal(0.0, scale, size=(observations, assets))

    loading = _factor_loadings(dgp_config, assets, rng)
    factor_scale = float(dgp_config.get("factor_scale", 0.0))
    idiosyncratic_scale = float(dgp_config.get("idiosyncratic_scale", 0.0))

    if dgp_name == "common_factor":
        factor = rng.normal(0.0, factor_scale, size=observations)
        idiosyncratic = rng.normal(
            0.0, idiosyncratic_scale, size=(observations, assets)
        )
        return factor[:, None] * loading[None, :] + idiosyncratic

    if dgp_name == "stochastic_volatility_factor":
        volatility = _stochastic_volatility(
            observations,
            float(dgp_config["log_volatility_phi"]),
            float(dgp_config["log_volatility_innovation"]),
            rng,
        )
        factor = rng.normal(0.0, factor_scale, size=observations)
        idiosyncratic = rng.normal(
            0.0, idiosyncratic_scale, size=(observations, assets)
        )
        return volatility[:, None] * (
            factor[:, None] * loading[None, :] + idiosyncratic
        )

    if dgp_name == "student_t_factor":
        degrees = float(dgp_config["degrees_of_freedom"])
        variance_scale = np.sqrt(degrees / (degrees - 2.0))
        factor = (
            rng.standard_t(degrees, size=observations) / variance_scale * factor_scale
        )
        idiosyncratic = (
            rng.standard_t(degrees, size=(observations, assets))
            / variance_scale
            * idiosyncratic_scale
        )
        return factor[:, None] * loading[None, :] + idiosyncratic

    if dgp_name == "skewed_factor":
        shape = float(dgp_config["lognormal_shape"])
        factor = _centered_lognormal(observations, shape, rng) * factor_scale
        idiosyncratic = (
            _centered_lognormal((observations, assets), shape, rng)
            * idiosyncratic_scale
        )
        return factor[:, None] * loading[None, :] + idiosyncratic

    if dgp_name == "jump_diffusion_factor":
        factor = rng.normal(0.0, factor_scale, size=observations)
        common_jump = rng.random(observations) < float(
            dgp_config["common_jump_probability"]
        )
        factor += common_jump * rng.normal(
            0.0, float(dgp_config["common_jump_scale"]), size=observations
        )
        idiosyncratic = rng.normal(
            0.0, idiosyncratic_scale, size=(observations, assets)
        )
        idiosyncratic_jump = rng.random((observations, assets)) < float(
            dgp_config["idiosyncratic_jump_probability"]
        )
        idiosyncratic += idiosyncratic_jump * rng.normal(
            0.0,
            float(dgp_config["idiosyncratic_jump_scale"]),
            size=(observations, assets),
        )
        return factor[:, None] * loading[None, :] + idiosyncratic

    if dgp_name == "time_varying_correlation":
        correlation = _time_varying_correlation(observations, dgp_config, rng)
        total_scale = float(dgp_config["total_scale"])
        factor = rng.normal(size=observations)
        idiosyncratic = rng.normal(size=(observations, assets))
        return total_scale * (
            np.sqrt(correlation)[:, None] * factor[:, None]
            + np.sqrt(1.0 - correlation)[:, None] * idiosyncratic
        )

    if dgp_name == "asymmetric_common_shock":
        factor_standard = rng.normal(size=observations)
        factor = factor_standard * factor_scale
        multiplier = np.where(
            factor_standard < 0.0,
            float(dgp_config["negative_volatility_multiplier"]),
            1.0,
        )
        idiosyncratic = (
            rng.normal(size=(observations, assets))
            * idiosyncratic_scale
            * multiplier[:, None]
        )
        return factor[:, None] * loading[None, :] + idiosyncratic

    raise ValueError(f"Unsupported mechanical-null DGP: {dgp_name}")


def simulate_research_weights(
    observations: int,
    assets: int,
    weighting: str,
    config: Mapping,
    rng: np.random.Generator,
) -> np.ndarray:
    if weighting == "equal":
        return np.full((observations, assets), 1.0 / assets, dtype=float)

    minimum = float(config["minimum_weight"])
    if weighting == "lagged_lognormal_liquidity":
        state = rng.normal(0.0, float(config["lognormal_sigma"]), size=assets)
        weights = np.empty((observations, assets), dtype=float)
        phi = float(config["liquidity_ar1"])
        innovation = float(config["liquidity_innovation"])
        for index in range(observations):
            weights[index] = _softmax(state, minimum)
            next_state = phi * state + rng.normal(0.0, innovation, size=assets)
            state = next_state - np.mean(next_state)
        return weights

    if weighting == "concentrated_static":
        top_weight = float(config["concentrated_top_weight"])
        if not 1.0 / assets < top_weight < 1.0:
            raise ValueError("concentrated_top_weight must be between equal weight and one")
        static = np.full(assets, (1.0 - top_weight) / (assets - 1), dtype=float)
        static[int(rng.integers(0, assets))] = top_weight
        return np.tile(static, (observations, 1))

    raise ValueError(f"Unsupported research weighting: {weighting}")


def summarize_simulation_replicates(
    replicates: pd.DataFrame,
    confidence: float,
    theory: pd.DataFrame | None = None,
) -> pd.DataFrame:
    grouping = ["phase", "dgp", "scenario", "observations", "assets", "weighting", "model"]
    rows: list[dict[str, object]] = []
    for keys, group in replicates.groupby(grouping, sort=False):
        row = dict(zip(grouping, keys, strict=True))
        coefficients = group["target_coefficient"].to_numpy(dtype=float)
        repetitions = len(coefficients)
        mean = float(np.mean(coefficients))
        standard_deviation = float(np.std(coefficients, ddof=1))
        monte_carlo_se = standard_deviation / np.sqrt(repetitions)
        critical = float(stats.t.ppf((1.0 + confidence) / 2.0, repetitions - 1))
        row.update(
            {
                "repetitions": repetitions,
                "mean_target_coefficient": mean,
                "median_target_coefficient": float(np.median(coefficients)),
                "target_coefficient_std": standard_deviation,
                "monte_carlo_standard_error": monte_carlo_se,
                "mean_ci_confidence": confidence,
                "mean_ci_lower": mean - critical * monte_carlo_se,
                "mean_ci_upper": mean + critical * monte_carlo_se,
                "negative_coefficient_rate": float(np.mean(coefficients < 0.0)),
                "raw_false_positive_rate": float(group["raw_false_positive"].mean()),
                "bh3_false_positive_rate": _mean_or_nan(group["bh3_false_positive"]),
                "bh4_false_positive_rate": float(group["bh4_false_positive"].mean()),
            }
        )
        for column in ("raw_false_positive", "bh3_false_positive", "bh4_false_positive"):
            valid = group[column].dropna().astype(bool)
            low, high = _wilson_interval(int(valid.sum()), len(valid))
            prefix = column.removesuffix("_false_positive")
            row[f"{prefix}_wilson_ci_lower"] = low
            row[f"{prefix}_wilson_ci_upper"] = high
        rows.append(row)

    summary = pd.DataFrame(rows)
    if theory is None:
        summary["theoretical_target_coefficient"] = np.nan
    else:
        summary = summary.merge(
            theory[["assets", "model", "theoretical_target_coefficient"]],
            on=["assets", "model"],
            how="left",
            validate="many_to_one",
        )
    summary["coefficient_bias"] = (
        summary["mean_target_coefficient"] - summary["theoretical_target_coefficient"]
    )
    nonzero = summary["theoretical_target_coefficient"].abs().gt(0.0)
    summary["absolute_relative_error"] = np.where(
        nonzero,
        summary["coefficient_bias"].abs()
        / summary["theoretical_target_coefficient"].abs(),
        np.nan,
    )
    summary["theory_inside_mean_ci"] = (
        summary["theoretical_target_coefficient"].ge(summary["mean_ci_lower"])
        & summary["theoretical_target_coefficient"].le(summary["mean_ci_upper"])
    )
    return summary


def build_convergence_gates(summary: pd.DataFrame, theory_config: Mapping) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    mechanical = {"no_intercept_csad", "scsad"}
    controls = {"standard_csad", "intercept_restored"}
    for row in summary.itertuples(index=False):
        result = {
            "assets": int(row.assets),
            "model": row.model,
            "theoretical_target_coefficient": row.theoretical_target_coefficient,
            "mean_target_coefficient": row.mean_target_coefficient,
            "absolute_relative_error": row.absolute_relative_error,
            "theory_inside_mean_ci": bool(row.theory_inside_mean_ci),
            "negative_coefficient_rate": row.negative_coefficient_rate,
            "raw_false_positive_rate": row.raw_false_positive_rate,
            "bh3_false_positive_rate": row.bh3_false_positive_rate,
        }
        if row.model in mechanical:
            result.update(
                {
                    "relative_error_pass": bool(
                        row.absolute_relative_error
                        <= float(theory_config["convergence_relative_tolerance"])
                    ),
                    "mean_ci_pass": bool(row.theory_inside_mean_ci),
                    "negative_rate_pass": bool(
                        row.negative_coefficient_rate
                        >= float(theory_config["negative_sign_rate_minimum"])
                    ),
                    "raw_fpr_pass": bool(
                        row.raw_false_positive_rate
                        >= float(theory_config["mechanical_raw_fpr_minimum"])
                    ),
                    "bh_fpr_pass": bool(
                        row.bh3_false_positive_rate
                        >= float(theory_config["mechanical_bh_fpr_minimum"])
                    ),
                    "gate_type": "mechanical_convergence",
                }
            )
            result["gate_pass"] = bool(
                result["relative_error_pass"]
                and result["mean_ci_pass"]
                and result["negative_rate_pass"]
                and result["raw_fpr_pass"]
                and result["bh_fpr_pass"]
            )
        elif row.model in controls:
            result.update(
                {
                    "relative_error_pass": np.nan,
                    "mean_ci_pass": bool(row.theory_inside_mean_ci),
                    "negative_rate_pass": np.nan,
                    "raw_fpr_pass": bool(
                        row.raw_false_positive_rate
                        <= float(theory_config["nominal_raw_fpr_maximum"])
                    ),
                    "bh_fpr_pass": np.nan,
                    "gate_type": "nominal_control",
                }
            )
            result["gate_pass"] = bool(result["raw_fpr_pass"])
        else:
            raise ValueError(f"Unexpected model in convergence summary: {row.model}")
        rows.append(result)
    return pd.DataFrame(rows)


def build_symmetric_robustness_gate(
    summary: pd.DataFrame,
    config: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    symmetric = summary.loc[
        summary["dgp"].isin(config["symmetric_dgps"])
        & summary["model"].isin(["no_intercept_csad", "scsad"])
    ].copy()
    pivot = symmetric.pivot_table(
        index=["dgp", "scenario", "observations", "assets", "weighting"],
        columns="model",
        values="negative_coefficient_rate",
        aggfunc="first",
    ).reset_index()
    threshold = float(config["symmetric_negative_rate_minimum"])
    pivot["no_intercept_negative_rate_pass"] = pivot["no_intercept_csad"].ge(threshold)
    pivot["scsad_negative_rate_pass"] = pivot["scsad"].ge(threshold)
    pivot["both_models_pass"] = (
        pivot["no_intercept_negative_rate_pass"] & pivot["scsad_negative_rate_pass"]
    )
    share = float(pivot["both_models_pass"].mean())
    required = float(config["symmetric_cell_share_minimum"])
    decision = pd.DataFrame(
        [
            {
                "symmetric_dgp_count": len(config["symmetric_dgps"]),
                "scenario_count": summary["scenario"].nunique(),
                "evaluated_cells": len(pivot),
                "passing_cells": int(pivot["both_models_pass"].sum()),
                "passing_cell_share": share,
                "required_cell_share": required,
                "negative_rate_threshold": threshold,
                "distributional_robustness_pass": bool(share >= required),
            }
        ]
    )
    return pivot, decision


def summarize_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    grouping = ["phase", "dgp", "scenario", "observations", "assets", "weighting"]
    numeric = [
        "market_mean",
        "market_std",
        "market_skew",
        "market_excess_kurtosis",
        "csad_mean",
        "csad_std",
        "corr_csad_market",
        "corr_csad_abs_market",
        "mean_weight_hhi",
        "max_weight",
    ]
    aggregations = {column: "mean" for column in numeric}
    aggregations["repetition"] = "size"
    result = diagnostics.groupby(grouping, as_index=False).agg(aggregations)
    return result.rename(columns={"repetition": "repetitions"})


def build_final_mechanical_decision(
    equation_verification: pd.DataFrame,
    convergence_gates: pd.DataFrame,
    robustness_decision: pd.DataFrame,
) -> pd.DataFrame:
    equation_pass = bool(equation_verification["equation_gate_pass"].all())
    mechanical = convergence_gates.loc[
        convergence_gates["gate_type"].eq("mechanical_convergence")
    ]
    controls = convergence_gates.loc[convergence_gates["gate_type"].eq("nominal_control")]
    convergence_pass = bool(mechanical["gate_pass"].all() and len(mechanical) == 6)
    controls_pass = bool(controls["gate_pass"].all() and len(controls) == 6)
    mechanical_fpr_pass = bool(
        mechanical["raw_fpr_pass"].astype(bool).all()
        and mechanical["bh_fpr_pass"].astype(bool).all()
    )
    confirmed = equation_pass and convergence_pass and controls_pass and mechanical_fpr_pass
    return pd.DataFrame(
        [
            {
                "equation_cells_passed": int(equation_verification["equation_gate_pass"].sum()),
                "equation_cells_total": len(equation_verification),
                "mechanical_convergence_cells_passed": int(mechanical["gate_pass"].sum()),
                "mechanical_convergence_cells_total": len(mechanical),
                "nominal_control_cells_passed": int(controls["gate_pass"].sum()),
                "nominal_control_cells_total": len(controls),
                "mechanical_fpr_gate_pass": mechanical_fpr_pass,
                "distributional_robustness_pass": bool(
                    robustness_decision.iloc[0]["distributional_robustness_pass"]
                ),
                "mechanical_null_confirmed": confirmed,
                "classification": (
                    "mechanical_null_confirmed"
                    if confirmed
                    else "mechanical_null_not_confirmed"
                ),
            }
        ]
    )


def _run_simulation_grid(
    dgp_names: Sequence[str],
    scenarios: Sequence[Mapping],
    repetitions: int,
    seed: int,
    config: Mapping,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha = float(config["models"]["alpha"])
    original_family = list(config["models"]["original_bh_family"])
    result_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for dgp_index, dgp_name in enumerate(dgp_names):
        dgp_config = config["dgp"][dgp_name]
        for scenario_index, scenario in enumerate(scenarios):
            observations = int(scenario["observations"])
            assets = int(scenario["assets"])
            weighting = str(scenario["weighting"])
            for repetition in range(repetitions):
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [seed, dgp_index, scenario_index, repetition]
                    )
                )
                returns = simulate_mechanical_null_returns(
                    dgp_name, dgp_config, observations, assets, rng
                )
                weights = simulate_research_weights(
                    observations, assets, weighting, config["weight_process"], rng
                )
                market, csad = construct_market_and_csad(returns, weights)
                fitted_rows = [
                    fit_fast_hac_model(csad, market, model)
                    for model in MECHANICAL_MODELS
                ]
                p_values = pd.Series(
                    {row["model"]: row["target_p_value"] for row in fitted_rows},
                    dtype=float,
                )
                q4 = benjamini_hochberg(p_values)
                q3 = benjamini_hochberg(p_values.loc[original_family])
                for fitted in fitted_rows:
                    model = str(fitted["model"])
                    coefficient = float(fitted["target_coefficient"])
                    fitted.update(
                        {
                            "phase": phase,
                            "dgp": dgp_name,
                            "scenario": str(scenario["id"]),
                            "observations": observations,
                            "assets": assets,
                            "weighting": weighting,
                            "repetition": repetition,
                            "q_value_bh3": (
                                float(q3.loc[model]) if model in original_family else np.nan
                            ),
                            "q_value_bh4": float(q4.loc[model]),
                        }
                    )
                    fitted["raw_false_positive"] = bool(
                        coefficient < 0.0 and fitted["target_p_value"] <= alpha
                    )
                    fitted["bh3_false_positive"] = (
                        bool(coefficient < 0.0 and fitted["q_value_bh3"] <= alpha)
                        if model in original_family
                        else np.nan
                    )
                    fitted["bh4_false_positive"] = bool(
                        coefficient < 0.0 and fitted["q_value_bh4"] <= alpha
                    )
                    result_rows.append(fitted)
                diagnostic_rows.append(
                    {
                        "phase": phase,
                        "dgp": dgp_name,
                        "scenario": str(scenario["id"]),
                        "observations": observations,
                        "assets": assets,
                        "weighting": weighting,
                        "repetition": repetition,
                        **_simulation_diagnostics(market, csad, weights),
                    }
                )
            LOGGER.info(
                "%s simulation complete: dgp=%s scenario=%s repetitions=%d",
                phase,
                dgp_name,
                scenario["id"],
                repetitions,
            )
    return pd.DataFrame(result_rows), pd.DataFrame(diagnostic_rows)


def _simulation_diagnostics(
    market: np.ndarray,
    csad: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    shares = weights / weights.sum(axis=1, keepdims=True)
    return {
        "market_mean": float(np.mean(market)),
        "market_std": float(np.std(market, ddof=1)),
        "market_skew": float(stats.skew(market, bias=False)),
        "market_excess_kurtosis": float(stats.kurtosis(market, fisher=True, bias=False)),
        "csad_mean": float(np.mean(csad)),
        "csad_std": float(np.std(csad, ddof=1)),
        "corr_csad_market": _safe_correlation(csad, market),
        "corr_csad_abs_market": _safe_correlation(csad, np.abs(market)),
        "mean_weight_hhi": float(np.mean(np.sum(shares**2, axis=1))),
        "max_weight": float(np.max(shares)),
    }


def _factor_loadings(
    config: Mapping,
    assets: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.normal(
        float(config.get("factor_loading_mean", 1.0)),
        float(config.get("factor_loading_std", 0.0)),
        size=assets,
    )


def _stochastic_volatility(
    observations: int,
    phi: float,
    innovation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    burn_in = 250
    state = np.zeros(observations + burn_in, dtype=float)
    stationary_scale = innovation / np.sqrt(max(1.0 - phi**2, 1e-12))
    state[0] = rng.normal(0.0, stationary_scale)
    for index in range(1, len(state)):
        state[index] = phi * state[index - 1] + rng.normal(0.0, innovation)
    log_volatility = state[burn_in:]
    return np.exp(log_volatility - 0.5 * stationary_scale**2)


def _centered_lognormal(
    size: int | tuple[int, ...],
    shape: float,
    rng: np.random.Generator,
) -> np.ndarray:
    normal = rng.normal(size=size)
    raw = np.exp(shape * normal)
    mean = np.exp(0.5 * shape**2)
    standard_deviation = np.sqrt((np.exp(shape**2) - 1.0) * np.exp(shape**2))
    return (raw - mean) / standard_deviation


def _time_varying_correlation(
    observations: int,
    config: Mapping,
    rng: np.random.Generator,
) -> np.ndarray:
    minimum = float(config["correlation_minimum"])
    maximum = float(config["correlation_maximum"])
    mean = float(config["correlation_mean"])
    phi = float(config["correlation_phi"])
    innovation = float(config["correlation_innovation"])
    normalized_mean = (mean - minimum) / (maximum - minimum)
    centre = np.log(normalized_mean / (1.0 - normalized_mean))
    latent = np.empty(observations, dtype=float)
    latent[0] = centre
    for index in range(1, observations):
        latent[index] = (
            centre + phi * (latent[index - 1] - centre) + rng.normal(0.0, innovation)
        )
    logistic = 1.0 / (1.0 + np.exp(-latent))
    return minimum + (maximum - minimum) * logistic


def _softmax(state: np.ndarray, minimum: float) -> np.ndarray:
    shifted = state - np.max(state)
    weights = np.maximum(np.exp(shifted), minimum)
    return weights / np.sum(weights)


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def _mean_or_nan(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else np.nan


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

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import gammaln, logsumexp

from frequency_sensitivity import benjamini_hochberg


MECHANISM_OUTCOMES = [
    "log_amihud_illiquidity",
    "zero_tick_share",
    "log_mean_intertrade_ms",
    "log_quote_volume",
    "abs_aggressor_imbalance",
]
GROUP_DIMENSIONS = [
    "pooled",
    "asset",
    "transaction_count_quintile",
    "zero_tick_share_quintile",
    "liquidity_quintile",
]
RAW_REQUIRED_COLUMNS = [
    "bucket_start",
    "symbol",
    "transaction_count",
    "zero_ticks",
    "zero_runs",
    "run_z_zero",
]
ANALYSIS_REQUIRED_COLUMNS = [
    "bucket_start",
    "symbol",
    "sample_split",
    "meets_minimum_trade_count",
    "zero_run_intensity",
    "scaled__zero_run_intensity",
    "utc_day",
    "hour_utc",
    "weekday_utc",
    *MECHANISM_OUTCOMES,
    *[f"scaled__{value}" for value in MECHANISM_OUTCOMES],
    "scaled__abs_bucket_return",
    "scaled__current_market_abs_return_loo",
]


@dataclass(frozen=True)
class NullSamplingDiagnostics:
    rows: int
    repetitions: int
    minimum_n: int
    maximum_n: int
    minimum_k: int
    maximum_k: int
    maximum_pmf_sum_error: float
    minimum_variance: float


def validate_frozen_config(config: Mapping) -> None:
    data = config["data"]
    analysis = config["analysis"]
    if list(data["symbols"]) != [
        "BTCUSDT",
        "ETHUSDT",
        "XRPUSDT",
        "SOLUSDT",
        "DOGEUSDT",
        "ADAUSDT",
        "AVAXUSDT",
    ]:
        raise ValueError("Mechanical-null universe must remain the frozen seven assets")
    if int(data["expected_rows"]) != 490_560:
        raise ValueError("Mechanical-null raw panel must contain 490,560 rows")
    if int(analysis["monte_carlo_repetitions"]) != 999:
        raise ValueError("Mechanical-null repetitions are frozen at 999")
    if int(analysis["stratification_quantiles"]) != 5:
        raise ValueError("Mechanical-null stratification is frozen at quintiles")
    if float(analysis["clustering_z_cutoff"]) != -1.96:
        raise ValueError("Mechanical-null clustering cutoff is frozen at -1.96")
    if list(analysis["mechanism_outcomes"]) != MECHANISM_OUTCOMES:
        raise ValueError("Mechanical-null mechanism outcomes have drifted")
    if config["output"]["base_dir"] != "outputs/v2/final_research_completion_v1":
        raise ValueError("Final research output directory is frozen")


def load_audit_frame(
    tick_path: str | Path,
    analysis_path: str | Path,
    config: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tick_source = Path(tick_path)
    analysis_source = Path(analysis_path)
    tick_columns = set(pq.ParquetFile(tick_source).schema.names)
    analysis_columns = set(pq.ParquetFile(analysis_source).schema.names)
    missing_tick = sorted(set(RAW_REQUIRED_COLUMNS).difference(tick_columns))
    missing_analysis = sorted(set(ANALYSIS_REQUIRED_COLUMNS).difference(analysis_columns))
    if missing_tick:
        raise ValueError(f"Raw tick frame is missing: {', '.join(missing_tick)}")
    if missing_analysis:
        raise ValueError(f"Zero-run analysis frame is missing: {', '.join(missing_analysis)}")

    raw = pd.read_parquet(tick_source, columns=RAW_REQUIRED_COLUMNS)
    analysis = pd.read_parquet(analysis_source, columns=ANALYSIS_REQUIRED_COLUMNS)
    raw["bucket_start"] = pd.to_datetime(raw["bucket_start"], utc=True)
    analysis["bucket_start"] = pd.to_datetime(analysis["bucket_start"], utc=True)
    if len(raw) != int(config["data"]["expected_rows"]):
        raise ValueError("Raw tick row count does not match the frozen config")
    if len(analysis) != len(raw):
        raise ValueError("Raw and zero-run analysis frames have different row counts")
    if raw.duplicated(["symbol", "bucket_start"]).any():
        raise ValueError("Raw tick frame contains duplicate row keys")
    if analysis.duplicated(["symbol", "bucket_start"]).any():
        raise ValueError("Zero-run analysis frame contains duplicate row keys")

    frame = analysis.merge(
        raw,
        on=["symbol", "bucket_start"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_raw"),
    )
    if len(frame) != len(raw):
        raise ValueError("Raw and analysis frame keys are not identical")
    if not np.allclose(
        frame["zero_run_intensity"], -frame["run_z_zero"], atol=1e-12, rtol=0.0
    ):
        raise ValueError("Stored zero-run intensity does not equal -run_z_zero")

    oos_start = as_utc_timestamp(config["data"]["expected_oos_start"])
    oos_end = as_utc_timestamp(config["data"]["expected_oos_end_exclusive"])
    oos = frame.loc[
        frame["sample_split"].eq("oos")
        & frame["bucket_start"].ge(oos_start)
        & frame["bucket_start"].lt(oos_end)
        & frame["meets_minimum_trade_count"]
    ].copy()
    minimum_transactions = int(config["analysis"]["minimum_transaction_count"])
    if not oos["transaction_count"].ge(minimum_transactions).all():
        raise ValueError("OOS audit frame violates the frozen minimum trade count")
    if set(oos["symbol"].unique()) != set(config["data"]["symbols"]):
        raise ValueError("OOS audit frame does not contain the frozen seven assets")

    recomputed = conditional_run_z(
        oos["zero_runs"].to_numpy(),
        oos["zero_ticks"].to_numpy(),
        oos["transaction_count"].to_numpy(),
    )
    differences = np.abs(recomputed - oos["run_z_zero"].to_numpy(dtype=float))
    integrity = pd.DataFrame(
        [
            {
                "raw_rows": int(len(raw)),
                "analysis_rows": int(len(analysis)),
                "oos_eligible_rows": int(len(oos)),
                "oos_start": oos["bucket_start"].min(),
                "oos_end": oos["bucket_start"].max(),
                "symbols": int(oos["symbol"].nunique()),
                "maximum_recomputed_run_z_difference": float(np.nanmax(differences)),
                "run_z_exact_match": bool(np.nanmax(differences) <= 1e-12),
            }
        ]
    )
    if not bool(integrity.loc[0, "run_z_exact_match"]):
        raise ValueError("Stored run_z_zero fails the frozen 1e-12 reconstruction gate")
    return oos.sort_values(["bucket_start", "symbol"]).reset_index(drop=True), integrity


def conditional_run_z(
    runs: np.ndarray | Sequence[int],
    category_count: np.ndarray | Sequence[int],
    n_transactions: np.ndarray | Sequence[int],
) -> np.ndarray:
    r, k, n = np.broadcast_arrays(
        np.asarray(runs, dtype=float),
        np.asarray(category_count, dtype=float),
        np.asarray(n_transactions, dtype=float),
    )
    m = n - k
    mean = k * (m + 1.0) / n
    variance = k * m * (k - 1.0) * (m + 1.0) / (n**2 * (n - 1.0))
    result = np.full(r.shape, np.nan, dtype=float)
    valid = (n > 1) & (k > 0) & (k < n) & (r >= 0) & (variance > 0)
    result[valid] = (r[valid] - mean[valid]) / np.sqrt(variance[valid])
    return result


def add_frozen_strata(
    frame: pd.DataFrame, quantiles: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    rows = []
    specifications = [
        ("transaction_count", "transaction_count_quintile"),
        ("zero_tick_share", "zero_tick_share_quintile"),
        ("log_quote_volume", "liquidity_quintile"),
    ]
    for source, target in specifications:
        labels, bins = pd.qcut(
            pd.to_numeric(result[source], errors="coerce"),
            q=quantiles,
            labels=False,
            retbins=True,
            duplicates="raise",
        )
        result[target] = labels.astype(int) + 1
        for index in range(len(bins) - 1):
            rows.append(
                {
                    "dimension": target,
                    "quintile": index + 1,
                    "lower_bound": float(bins[index]),
                    "upper_bound": float(bins[index + 1]),
                    "rows": int((result[target] == index + 1).sum()),
                }
            )
    return result, pd.DataFrame(rows)


def draw_stratified_audit_sample(
    frame: pd.DataFrame,
    maximum_rows_per_stratum: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    sample_parts = []
    summary_rows = []
    group_columns = [
        "symbol",
        "transaction_count_quintile",
        "zero_tick_share_quintile",
    ]
    for keys, subset in frame.groupby(group_columns, sort=True, observed=True):
        local = subset.sort_values(["bucket_start", "symbol"])
        sample_size = min(len(local), maximum_rows_per_stratum)
        positions = np.sort(rng.choice(len(local), size=sample_size, replace=False))
        selected = local.iloc[positions].copy()
        weight = len(local) / sample_size
        selected["sampling_weight"] = weight
        selected["population_stratum_rows"] = len(local)
        selected["sample_stratum_rows"] = sample_size
        sample_parts.append(selected)
        summary_rows.append(
            {
                "symbol": keys[0],
                "transaction_count_quintile": int(keys[1]),
                "zero_tick_share_quintile": int(keys[2]),
                "population_rows": int(len(local)),
                "sample_rows": int(sample_size),
                "sampling_weight": float(weight),
            }
        )
    sample = pd.concat(sample_parts, ignore_index=True)
    sample = sample.sort_values(["bucket_start", "symbol"]).reset_index(drop=True)
    if sample.duplicated(["symbol", "bucket_start"]).any():
        raise ValueError("Stratified audit sample contains duplicate row keys")
    return sample, pd.DataFrame(summary_rows)


def exact_run_count_pmf(n_transactions: int, category_count: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(n_transactions)
    k = int(category_count)
    m = n - k
    if n <= 1 or k <= 0 or k >= n:
        raise ValueError("Exact conditional run distribution requires 0 < k < n")
    maximum_runs = min(k, m + 1)
    runs = np.arange(1, maximum_runs + 1, dtype=int)
    log_probabilities = (
        log_combination(k - 1, runs - 1)
        + log_combination(m + 1, runs)
        - log_combination(n, k)
    )
    probabilities = np.exp(log_probabilities - logsumexp(log_probabilities))
    return runs, probabilities


def log_combination(n: int, k: np.ndarray | int) -> np.ndarray:
    values = np.asarray(k, dtype=float)
    return gammaln(n + 1.0) - gammaln(values + 1.0) - gammaln(n - values + 1.0)


def verify_preregistration_seal(
    protocol_path: str | Path,
    config_path: str | Path,
    seal_path: str | Path,
) -> dict:
    protocol = Path(protocol_path)
    config = Path(config_path)
    seal = json.loads(Path(seal_path).read_text(encoding="utf-8"))
    observed = {
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    }
    for key, digest in observed.items():
        if seal.get(key) != digest:
            raise ValueError(f"Preregistration seal mismatch: {key}")
    if seal.get("results_observed_at_seal") is not False:
        raise ValueError("Preregistration seal does not certify an unobserved null result")
    return {**seal, **observed, "verified": True}


def sample_exact_conditional_run_z(
    n_transactions: np.ndarray,
    category_count: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[np.ndarray, NullSamplingDiagnostics]:
    n_values = np.asarray(n_transactions, dtype=int)
    k_values = np.asarray(category_count, dtype=int)
    if n_values.shape != k_values.shape:
        raise ValueError("n and k arrays must have identical shapes")
    rng = np.random.default_rng(seed)
    draws = np.empty((len(n_values), repetitions), dtype=np.float32)
    maximum_error = 0.0
    minimum_variance = np.inf
    for index, (n, k) in enumerate(zip(n_values, k_values)):
        runs, probabilities = exact_run_count_pmf(int(n), int(k))
        maximum_error = max(maximum_error, abs(float(probabilities.sum()) - 1.0))
        cdf = np.cumsum(probabilities)
        cdf[-1] = 1.0
        sampled_runs = runs[np.searchsorted(cdf, rng.random(repetitions), side="left")]
        m = n - k
        mean = k * (m + 1.0) / n
        variance = k * m * (k - 1.0) * (m + 1.0) / (n**2 * (n - 1.0))
        if variance <= 0 or not np.isfinite(variance):
            raise ValueError(f"Degenerate exact run variance for n={n}, k={k}")
        minimum_variance = min(minimum_variance, float(variance))
        draws[index] = ((sampled_runs - mean) / np.sqrt(variance)).astype(np.float32)
    diagnostics = NullSamplingDiagnostics(
        rows=int(len(n_values)),
        repetitions=int(repetitions),
        minimum_n=int(n_values.min()),
        maximum_n=int(n_values.max()),
        minimum_k=int(k_values.min()),
        maximum_k=int(k_values.max()),
        maximum_pmf_sum_error=float(maximum_error),
        minimum_variance=float(minimum_variance),
    )
    return draws, diagnostics


def apply_frozen_intensity_scaler(
    raw_intensity: np.ndarray,
    scaler_path: str | Path,
) -> np.ndarray:
    payload = json.loads(Path(scaler_path).read_text(encoding="utf-8"))
    parameters = payload["columns"]["zero_run_intensity"]
    clipped = np.clip(
        raw_intensity,
        float(parameters["winsor_lower"]),
        float(parameters["winsor_upper"]),
    )
    return (
        clipped - float(parameters["mean"])
    ) / float(parameters["std"])


def build_group_definitions(frame: pd.DataFrame) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = [
        {"dimension": "pooled", "group": "all", "mask": np.ones(len(frame), dtype=bool)}
    ]
    for symbol in sorted(frame["symbol"].unique()):
        groups.append(
            {
                "dimension": "asset",
                "group": symbol,
                "mask": frame["symbol"].eq(symbol).to_numpy(),
            }
        )
    for column in (
        "transaction_count_quintile",
        "zero_tick_share_quintile",
        "liquidity_quintile",
    ):
        for value in range(1, 6):
            groups.append(
                {
                    "dimension": column,
                    "group": f"Q{value}",
                    "mask": frame[column].eq(value).to_numpy(),
                }
            )
    if len(groups) != 23:
        raise ValueError(f"Expected 23 frozen null groups, found {len(groups)}")
    return groups


def analyze_group_null_calibration(
    sample: pd.DataFrame,
    null_run_z: np.ndarray,
    config: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    repetitions = null_run_z.shape[1]
    cutoff = float(config["analysis"]["clustering_z_cutoff"])
    actual_intensity = sample["zero_run_intensity"].to_numpy(dtype=float)
    actual_tail = sample["run_z_zero"].le(cutoff).to_numpy(dtype=float)
    null_intensity = -null_run_z.astype(float)
    null_tail = (null_run_z <= cutoff).astype(float)
    weights = sample["sampling_weight"].to_numpy(dtype=float)
    minimum_rows = int(config["analysis"]["minimum_group_rows"])
    summary_rows = []
    replicate_rows = []
    for definition in build_group_definitions(sample):
        mask = np.asarray(definition["mask"], dtype=bool)
        local_weights = weights[mask]
        weight_sum = local_weights.sum()
        empirical_intensity = float(
            np.sum(local_weights * actual_intensity[mask]) / weight_sum
        )
        empirical_tail = float(np.sum(local_weights * actual_tail[mask]) / weight_sum)
        null_intensity_values = (
            local_weights @ null_intensity[mask] / weight_sum
        )
        null_tail_values = local_weights @ null_tail[mask] / weight_sum
        eligible = int(mask.sum()) >= minimum_rows
        intensity_p = upper_empirical_p(null_intensity_values, empirical_intensity)
        tail_p = upper_empirical_p(null_tail_values, empirical_tail)
        if not eligible:
            intensity_p = 1.0
            tail_p = 1.0
        summary_rows.append(
            {
                "dimension": definition["dimension"],
                "group": definition["group"],
                "sample_rows": int(mask.sum()),
                "estimated_population_rows": float(weight_sum),
                "inference_eligible": eligible,
                "empirical_mean_intensity": empirical_intensity,
                "null_mean_intensity": float(np.mean(null_intensity_values)),
                "null_intensity_q025": float(np.quantile(null_intensity_values, 0.025)),
                "null_intensity_q975": float(np.quantile(null_intensity_values, 0.975)),
                "intensity_empirical_percentile": float(
                    np.mean(null_intensity_values <= empirical_intensity)
                ),
                "intensity_empirical_p_value": intensity_p,
                "empirical_clustering_share": empirical_tail,
                "null_fpr_mean": float(np.mean(null_tail_values)),
                "null_fpr_q025": float(np.quantile(null_tail_values, 0.025)),
                "null_fpr_q975": float(np.quantile(null_tail_values, 0.975)),
                "tail_empirical_percentile": float(
                    np.mean(null_tail_values <= empirical_tail)
                ),
                "tail_empirical_p_value": tail_p,
            }
        )
        for draw in range(repetitions):
            replicate_rows.append(
                {
                    "dimension": definition["dimension"],
                    "group": definition["group"],
                    "draw": draw + 1,
                    "null_mean_intensity": float(null_intensity_values[draw]),
                    "null_fpr": float(null_tail_values[draw]),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary["intensity_q_value_bh_fdr_23"] = benjamini_hochberg(
        summary["intensity_empirical_p_value"]
    )
    summary["tail_q_value_bh_fdr_23"] = benjamini_hochberg(
        summary["tail_empirical_p_value"]
    )
    alpha = float(config["analysis"]["fdr_alpha"])
    percentile = float(config["analysis"]["empirical_percentile_threshold"])
    summary["supports_excess_intensity"] = (
        summary["inference_eligible"]
        & summary["intensity_q_value_bh_fdr_23"].le(alpha)
        & summary["intensity_empirical_percentile"].ge(percentile)
    )
    summary["supports_excess_tail"] = (
        summary["inference_eligible"]
        & summary["tail_q_value_bh_fdr_23"].le(alpha)
        & summary["tail_empirical_percentile"].ge(percentile)
    )
    return summary, pd.DataFrame(replicate_rows)


def upper_empirical_p(null_values: np.ndarray, observed: float) -> float:
    values = np.asarray(null_values, dtype=float)
    return float((1.0 + np.sum(values >= observed)) / (len(values) + 1.0))


def two_sided_empirical_p(null_values: np.ndarray, observed: float) -> float:
    values = np.asarray(null_values, dtype=float)
    lower = (1.0 + np.sum(values <= observed)) / (len(values) + 1.0)
    upper = (1.0 + np.sum(values >= observed)) / (len(values) + 1.0)
    return float(min(1.0, 2.0 * min(lower, upper)))


def analyze_mechanism_null(
    sample: pd.DataFrame,
    null_run_z: np.ndarray,
    scaler_path: str | Path,
    full_mechanism_decisions_path: str | Path,
    config: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis = config["analysis"]
    controls = [f"scaled__{value}" for value in analysis["mechanism_controls"]]
    fixed_effects = list(analysis["fixed_effects"])
    required = [
        "scaled__zero_run_intensity",
        "sampling_weight",
        *controls,
        *fixed_effects,
        *[f"scaled__{value}" for value in MECHANISM_OUTCOMES],
    ]
    model = sample.dropna(subset=required).copy()
    if len(model) != len(sample):
        raise ValueError("Mechanical-null audit sample has incomplete mechanism rows")
    design = build_control_design(model, controls, fixed_effects)
    weights = model["sampling_weight"].to_numpy(dtype=float)
    weights = weights / np.mean(weights)
    sqrt_weights = np.sqrt(weights)
    weighted_design = design * sqrt_weights[:, None]
    gram_inverse = np.linalg.pinv(weighted_design.T @ weighted_design)

    actual_x = model["scaled__zero_run_intensity"].to_numpy(dtype=float)
    null_raw_intensity = -null_run_z.astype(float)
    null_scaled = apply_frozen_intensity_scaler(null_raw_intensity, scaler_path)
    actual_x_weighted = actual_x * sqrt_weights
    null_x_weighted = null_scaled * sqrt_weights[:, None]
    actual_x_residual = residualize_weighted_vector(
        actual_x_weighted, weighted_design, gram_inverse
    )
    null_z_cross = weighted_design.T @ null_x_weighted
    null_projection = gram_inverse @ null_z_cross
    null_denominator = np.sum(null_x_weighted**2, axis=0) - np.sum(
        null_z_cross * null_projection, axis=0
    )
    if np.any(null_denominator <= 0):
        raise ValueError("Null residualized focal denominator is non-positive")

    full = pd.read_csv(full_mechanism_decisions_path).set_index("outcome")
    summary_rows = []
    draw_rows = []
    model_rows = []
    for outcome in MECHANISM_OUTCOMES:
        y = model[f"scaled__{outcome}"].to_numpy(dtype=float)
        y_weighted = y * sqrt_weights
        y_residual = residualize_weighted_vector(
            y_weighted, weighted_design, gram_inverse
        )
        actual_coefficient = float(
            np.dot(actual_x_residual, y_residual)
            / np.dot(actual_x_residual, actual_x_residual)
        )
        null_numerator = null_x_weighted.T @ y_residual
        null_coefficients = null_numerator / null_denominator
        p_value = two_sided_empirical_p(null_coefficients, actual_coefficient)
        full_coefficient = float(full.loc[outcome, "coefficient"])
        magnitude_ratio = (
            abs(actual_coefficient) / abs(full_coefficient)
            if full_coefficient != 0
            else np.nan
        )
        summary_rows.append(
            {
                "outcome": outcome,
                "full_oos_coefficient": full_coefficient,
                "full_oos_pass": bool(full.loc[outcome, "passes_predeclared_criteria"]),
                "audit_sample_coefficient": actual_coefficient,
                "same_sign_as_full": bool(
                    np.sign(actual_coefficient) == np.sign(full_coefficient)
                ),
                "audit_to_full_absolute_magnitude_ratio": magnitude_ratio,
                "null_mean_coefficient": float(np.mean(null_coefficients)),
                "null_std_coefficient": float(np.std(null_coefficients, ddof=1)),
                "null_q025": float(np.quantile(null_coefficients, 0.025)),
                "null_q975": float(np.quantile(null_coefficients, 0.975)),
                "outside_null_95_interval": bool(
                    actual_coefficient < np.quantile(null_coefficients, 0.025)
                    or actual_coefficient > np.quantile(null_coefficients, 0.975)
                ),
                "empirical_p_value": p_value,
            }
        )
        model_rows.append(
            {
                "outcome": outcome,
                "observations": int(len(model)),
                "control_terms": int(weighted_design.shape[1]),
                "condition_number": float(
                    np.sqrt(np.linalg.cond(weighted_design.T @ weighted_design))
                ),
                "weighted_effective_sample_size": float(
                    weights.sum() ** 2 / np.sum(weights**2)
                ),
            }
        )
        for draw, coefficient in enumerate(null_coefficients):
            draw_rows.append(
                {
                    "outcome": outcome,
                    "draw": draw + 1,
                    "null_coefficient": float(coefficient),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary["q_value_bh_fdr_5"] = benjamini_hochberg(summary["empirical_p_value"])
    summary["exceeds_count_conditioned_null"] = (
        summary["full_oos_pass"]
        & summary["same_sign_as_full"]
        & summary["audit_to_full_absolute_magnitude_ratio"].ge(
            float(analysis["mechanism_minimum_magnitude_ratio"])
        )
        & summary["q_value_bh_fdr_5"].le(float(analysis["fdr_alpha"]))
        & summary["outside_null_95_interval"]
    )
    return summary, pd.DataFrame(draw_rows), pd.DataFrame(model_rows)


def build_control_design(
    frame: pd.DataFrame,
    controls: Sequence[str],
    fixed_effects: Sequence[str],
) -> np.ndarray:
    continuous = frame[list(controls)].apply(pd.to_numeric, errors="coerce").astype(float)
    dummies = pd.get_dummies(
        frame[list(fixed_effects)].astype(str),
        columns=list(fixed_effects),
        drop_first=True,
        dtype=float,
    )
    return np.column_stack(
        [np.ones(len(frame), dtype=float), continuous.to_numpy(dtype=float), dummies.to_numpy(dtype=float)]
    )


def residualize_weighted_vector(
    weighted_values: np.ndarray,
    weighted_design: np.ndarray,
    gram_inverse: np.ndarray,
) -> np.ndarray:
    return weighted_values - weighted_design @ (
        gram_inverse @ (weighted_design.T @ weighted_values)
    )


def build_audit_decisions(
    group_summary: pd.DataFrame,
    mechanism_summary: pd.DataFrame,
    config: Mapping,
) -> pd.DataFrame:
    analysis = config["analysis"]
    pooled = group_summary.loc[
        group_summary["dimension"].eq("pooled")
        & group_summary["group"].eq("all")
    ].iloc[0]
    pooled_calibrated = (
        float(analysis["null_fpr_lower_bound"])
        <= pooled["null_fpr_mean"]
        <= float(analysis["null_fpr_upper_bound"])
    )
    calibrated_group_share = float(
        group_summary["null_fpr_mean"]
        .le(float(analysis["null_fpr_group_maximum"]))
        .mean()
    )
    calibration_pass = pooled_calibrated and calibrated_group_share >= float(
        analysis["minimum_calibrated_group_share"]
    )
    assets = group_summary.loc[group_summary["dimension"].eq("asset")]
    supporting_assets = int(
        (assets["supports_excess_intensity"] & assets["supports_excess_tail"]).sum()
    )
    excess_clustering_pass = bool(
        pooled["supports_excess_intensity"]
        and pooled["supports_excess_tail"]
        and supporting_assets >= int(analysis["minimum_supporting_assets"])
    )
    passed_mechanisms = mechanism_summary.loc[
        mechanism_summary["exceeds_count_conditioned_null"]
    ]
    mechanism_pass = bool(
        len(passed_mechanisms) >= int(analysis["mechanism_minimum_count"])
        and passed_mechanisms["outcome"]
        .isin(analysis["mechanism_independent_outcomes"])
        .any()
    )
    if not calibration_pass:
        final_classification = "cutoff_not_calibrated"
    elif not excess_clustering_pass:
        final_classification = "counts_explain_empirical_clustering"
    elif not mechanism_pass:
        final_classification = "clustering_beyond_counts_but_mechanism_not_distinct"
    else:
        final_classification = "clustering_and_mechanism_beyond_count_conditioned_null"
    return pd.DataFrame(
        [
            {
                "decision": "conditional_cutoff_calibration",
                "passed": calibration_pass,
                "value": float(pooled["null_fpr_mean"]),
                "supporting_count": int(
                    group_summary["null_fpr_mean"]
                    .le(float(analysis["null_fpr_group_maximum"]))
                    .sum()
                ),
                "required_count_or_share": float(
                    analysis["minimum_calibrated_group_share"]
                ),
                "classification": "calibrated" if calibration_pass else "not_calibrated",
            },
            {
                "decision": "empirical_excess_clustering",
                "passed": excess_clustering_pass,
                "value": float(pooled["empirical_clustering_share"]),
                "supporting_count": supporting_assets,
                "required_count_or_share": float(analysis["minimum_supporting_assets"]),
                "classification": "beyond_counts" if excess_clustering_pass else "not_distinct",
            },
            {
                "decision": "mechanism_beyond_conditional_null",
                "passed": mechanism_pass,
                "value": float(len(passed_mechanisms)),
                "supporting_count": int(len(passed_mechanisms)),
                "required_count_or_share": float(analysis["mechanism_minimum_count"]),
                "classification": "beyond_counts" if mechanism_pass else "not_distinct",
            },
            {
                "decision": "final_mechanical_null_audit",
                "passed": calibration_pass and excess_clustering_pass and mechanism_pass,
                "value": np.nan,
                "supporting_count": np.nan,
                "required_count_or_share": np.nan,
                "classification": final_classification,
            },
        ]
    )


def build_artifact_manifest(output_dir: str | Path) -> pd.DataFrame:
    base = Path(output_dir)
    rows = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.csv":
            continue
        rows.append(
            {
                "path": path.relative_to(base).as_posix(),
                "size": int(path.stat().st_size),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return pd.DataFrame(rows)


def as_utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper


def run_csad_regression(
    csad: pd.Series,
    market_return: pd.Series,
    cov_type: str = "HAC",
    hac_maxlags: int | str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RegressionResultsWrapper, dict]:
    regression_frame = prepare_regression_frame(csad, market_return)
    x = sm.add_constant(regression_frame[["abs_market_return", "market_return_sq"]])
    model = _fit_ols(
        dependent=regression_frame["csad"],
        design_matrix=x,
        cov_type=cov_type,
        hac_maxlags=hac_maxlags,
    )

    coefficient_table = _build_coefficient_table(model)
    diagnostics = _build_diagnostics_table(model)

    regression_frame["fitted_csad"] = model.fittedvalues
    regression_frame["residual"] = model.resid

    json_summary = build_regression_json_summary(model)
    return coefficient_table, diagnostics, regression_frame, model, json_summary


def run_no_intercept_csad_regression(
    csad: pd.Series,
    market_return: pd.Series,
    cov_type: str = "HAC",
    hac_maxlags: int | str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RegressionResultsWrapper, dict]:
    regression_frame = prepare_regression_frame(csad, market_return)
    x = regression_frame[["market_return", "abs_market_return", "market_return_sq"]]
    model = _fit_ols(
        dependent=regression_frame["csad"],
        design_matrix=x,
        cov_type=cov_type,
        hac_maxlags=hac_maxlags,
    )

    coefficient_table = _build_coefficient_table(model)
    diagnostics = _build_diagnostics_table(model)

    regression_frame["fitted_csad"] = model.fittedvalues
    regression_frame["residual"] = model.resid

    json_summary = build_generic_regression_json_summary(
        model=model,
        target_term="market_return_sq",
        interpretation_if_negative="quadratic term is negative, which is consistent with herding in the no-intercept model.",
        interpretation_if_not_negative="quadratic term is not negative, so the no-intercept model does not support herding.",
    )
    return coefficient_table, diagnostics, regression_frame, model, json_summary


def run_scsad_regression(
    csad: pd.Series,
    market_return: pd.Series,
    cov_type: str = "HAC",
    hac_maxlags: int | str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RegressionResultsWrapper, dict]:
    regression_frame = prepare_regression_frame(csad, market_return)
    regression_frame["scsad"] = np.where(
        regression_frame["market_return"] >= 0.0,
        regression_frame["csad"],
        -regression_frame["csad"],
    )
    regression_frame["market_return_cu"] = regression_frame["market_return"] ** 3

    x = sm.add_constant(regression_frame[["market_return", "market_return_sq", "market_return_cu"]])
    model = _fit_ols(
        dependent=regression_frame["scsad"],
        design_matrix=x,
        cov_type=cov_type,
        hac_maxlags=hac_maxlags,
    )

    coefficient_table = _build_coefficient_table(model)
    diagnostics = _build_diagnostics_table(model)

    regression_frame["fitted_scsad"] = model.fittedvalues
    regression_frame["residual"] = model.resid

    json_summary = build_generic_regression_json_summary(
        model=model,
        target_term="market_return_cu",
        interpretation_if_negative="cubic term is negative, which is consistent with herding in the SCSAD model.",
        interpretation_if_not_negative="cubic term is not negative, so the SCSAD model does not support herding.",
    )
    return coefficient_table, diagnostics, regression_frame, model, json_summary


def prepare_regression_frame(csad: pd.Series, market_return: pd.Series) -> pd.DataFrame:
    regression_frame = pd.concat(
        [csad.rename("csad"), market_return.rename("market_return")],
        axis=1,
        join="inner",
    ).dropna()
    regression_frame["abs_market_return"] = regression_frame["market_return"].abs()
    regression_frame["market_return_sq"] = regression_frame["market_return"] ** 2
    return regression_frame


def build_regression_json_summary(model: RegressionResultsWrapper) -> dict:
    return build_generic_regression_json_summary(
        model=model,
        target_term="market_return_sq",
        interpretation_if_negative="beta2 is negative, which is consistent with herding.",
        interpretation_if_not_negative="beta2 is not negative, so the baseline regression does not support herding.",
    )


def build_generic_regression_json_summary(
    model: RegressionResultsWrapper,
    target_term: str,
    interpretation_if_negative: str,
    interpretation_if_not_negative: str,
) -> dict:
    target_value = float(model.params.get(target_term))
    target_t_stat = float(model.tvalues.get(target_term))
    target_p_value = float(model.pvalues.get(target_term))

    interpretation = (
        interpretation_if_negative
        if target_value < 0
        else interpretation_if_not_negative
    )

    return {
        "coefficients": {key: float(value) for key, value in model.params.items()},
        "t_statistics": {key: float(value) for key, value in model.tvalues.items()},
        "p_values": {key: float(value) for key, value in model.pvalues.items()},
        "target_term": target_term,
        "target_value": target_value,
        "target_t_stat": target_t_stat,
        "target_p_value": target_p_value,
        "beta2": target_value if target_term == "market_return_sq" else float("nan"),
        "beta2_t_stat": target_t_stat if target_term == "market_return_sq" else float("nan"),
        "beta2_p_value": target_p_value if target_term == "market_return_sq" else float("nan"),
        "rsquared": float(model.rsquared),
        "adj_rsquared": float(model.rsquared_adj),
        "nobs": float(model.nobs),
        "cov_type": str(model.cov_type),
        "covariance_metadata": {
            key: value
            for key, value in dict(getattr(model, "cov_kwds", {}) or {}).items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
        "interpretation": interpretation,
    }


def run_rolling_csad_regression(
    csad: pd.Series,
    market_return: pd.Series,
    window: int,
    min_periods: int | None = None,
    cov_type: str = "HAC",
    hac_maxlags: int | str | None = None,
) -> pd.DataFrame:
    regression_frame = prepare_regression_frame(csad, market_return)
    if regression_frame.empty:
        return pd.DataFrame()

    window = int(window)
    min_periods = int(min_periods or window)
    if window <= 0 or len(regression_frame) < min_periods:
        return pd.DataFrame()

    results = []
    for end_idx in range(window, len(regression_frame) + 1):
        subset = regression_frame.iloc[end_idx - window:end_idx]
        if len(subset) < min_periods:
            continue

        x = sm.add_constant(subset[["abs_market_return", "market_return_sq"]])
        model = _fit_ols(subset["csad"], x, cov_type=cov_type, hac_maxlags=hac_maxlags)
        results.append(
            {
                "timestamp": subset.index[-1],
                "beta1": model.params.get("abs_market_return"),
                "beta2": model.params.get("market_return_sq"),
                "beta2_t_stat": model.tvalues.get("market_return_sq"),
                "beta2_p_value": model.pvalues.get("market_return_sq"),
                "rsquared": model.rsquared,
                "nobs": model.nobs,
                "cov_type": str(model.cov_type),
            }
        )

    if not results:
        return pd.DataFrame()

    rolling_results = pd.DataFrame(results).set_index("timestamp")
    rolling_results.index.name = "timestamp"
    return rolling_results


def _fit_ols(
    dependent: pd.Series,
    design_matrix: pd.DataFrame,
    cov_type: str = "HAC",
    hac_maxlags: int | str | None = None,
) -> RegressionResultsWrapper:
    cov_type_normalized = str(cov_type or "nonrobust").upper()
    if cov_type_normalized in {"NONROBUST", "OLS", "DEFAULT"}:
        return sm.OLS(dependent, design_matrix).fit()

    if cov_type_normalized in {"HAC", "NEWEY-WEST", "NEWEYWEST", "NW"}:
        resolved_maxlags = _resolve_hac_maxlags(len(dependent), hac_maxlags)
        return sm.OLS(dependent, design_matrix).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": resolved_maxlags},
        )

    raise ValueError(f"Unsupported regression covariance type: {cov_type}")


def _resolve_hac_maxlags(nobs: int, hac_maxlags: int | str | None) -> int:
    if hac_maxlags not in {None, "", "auto"}:
        return max(int(hac_maxlags), 1)

    if nobs <= 1:
        return 1

    automatic_lag = int(np.floor(4.0 * ((nobs / 100.0) ** (2.0 / 9.0))))
    return max(automatic_lag, 1)


def _build_coefficient_table(model: RegressionResultsWrapper) -> pd.DataFrame:
    confidence_interval = model.conf_int()
    coefficient_table = pd.DataFrame(
        {
            "coefficient": model.params,
            "std_error": model.bse,
            "t_stat": model.tvalues,
            "p_value": model.pvalues,
            "ci_lower": confidence_interval[0],
            "ci_upper": confidence_interval[1],
        }
    )
    coefficient_table.index.name = "term"
    return coefficient_table


def _build_diagnostics_table(model: RegressionResultsWrapper) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": ["rsquared", "adj_rsquared", "nobs", "aic", "bic", "f_pvalue"],
            "value": [
                model.rsquared,
                model.rsquared_adj,
                model.nobs,
                model.aic,
                model.bic,
                model.f_pvalue,
            ],
        }
    )

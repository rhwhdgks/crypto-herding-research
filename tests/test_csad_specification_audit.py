from __future__ import annotations

import numpy as np
import pandas as pd

from csad_null_simulation import (
    construct_market_and_csad,
    fit_fast_hac_model,
    simulate_null_returns,
    simulate_weights,
)
from csad_specification_audit import (
    assign_historical_volatility_regime,
    build_empirical_panel,
    fit_audit_model,
)


def _member_rows() -> pd.DataFrame:
    periods = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    returns = {
        periods[0]: [0.01, 0.02, -0.01],
        periods[1]: [-0.02, 0.01, 0.03],
        periods[2]: [0.00, -0.01, 0.02],
    }
    rows = []
    for period, values in returns.items():
        market = float(np.mean(values))
        csad = float(np.mean(np.abs(np.asarray(values) - market)))
        for symbol, value in zip(["BTCUSDT", "ETHUSDT", "XRPUSDT"], values, strict=True):
            rows.append(
                {
                    "period": period,
                    "symbol": symbol,
                    "asset_return": value,
                    "return_weight": 1.0,
                    "valid_return_weight": True,
                    "eligible": True,
                    "market_return": market,
                    "csad": csad,
                }
            )
    return pd.DataFrame(rows)


def test_empirical_panel_rebuilds_market_and_exact_equal_weight_loo() -> None:
    panel = build_empirical_panel(
        _member_rows(),
        {
            "id": "test",
            "provider": "test",
            "sample": "test",
            "universe_type": "fixed",
            "weighting_class": "equal",
        },
        frequency="daily",
    )
    assert panel.integrity["rebuild_matches_stored"]
    pd.testing.assert_series_equal(panel.market, panel.loo_market, check_names=False)
    expected_loo = panel.csad * 3.0 / 2.0
    pd.testing.assert_series_equal(panel.loo_csad, expected_loo, check_names=False)
    assert np.allclose(panel.metrics["weight_hhi"], 1.0 / 3.0)
    assert np.allclose(panel.metrics["btc_weight"], 1.0 / 3.0)


def test_intercept_restored_uses_same_no_intercept_regressors_plus_constant() -> None:
    market = pd.Series(np.linspace(-0.05, 0.05, 301))
    csad = 0.02 + 0.3 * market.abs() + 0.4 * market.pow(2)
    restored, coefficients = fit_audit_model(csad, market, "intercept_restored")
    assert abs(restored["intercept"] - 0.02) < 1e-10
    assert abs(restored["target_coefficient"] - 0.4) < 1e-8
    assert set(coefficients["term"]) == {
        "const",
        "market_return",
        "abs_market_return",
        "market_return_sq",
    }


def test_volatility_regime_uses_only_information_before_current_period() -> None:
    rng = np.random.default_rng(7)
    index = pd.date_range("2020-01-01", periods=400, freq="D", tz="UTC")
    market = pd.Series(rng.normal(0.0, 0.01, len(index)), index=index)
    config = {
        "quantiles": [1 / 3, 2 / 3],
        "daily": {"rolling_window": 30, "rolling_minimum": 20, "expanding_minimum": 180},
    }
    original = assign_historical_volatility_regime(market, "daily", config)
    changed = market.copy()
    timestamp = index[300]
    changed.loc[timestamp:] = 100.0
    audited = assign_historical_volatility_regime(changed, "daily", config)
    assert original.loc[timestamp, "lagged_realized_volatility"] == audited.loc[
        timestamp, "lagged_realized_volatility"
    ]
    assert original.loc[timestamp, "volatility_regime"] == audited.loc[
        timestamp, "volatility_regime"
    ]


def test_fast_hac_matches_statsmodels_hac_target_estimate_and_standard_error() -> None:
    rng = np.random.default_rng(11)
    returns = simulate_null_returns(
        "common_factor",
        {
            "factor_scale": 0.018,
            "idiosyncratic_scale": 0.025,
            "factor_loading_mean": 1.0,
            "factor_loading_std": 0.2,
        },
        observations=500,
        assets=14,
        rng=rng,
    )
    weights = np.ones_like(returns)
    market, csad = construct_market_and_csad(returns, weights)
    for model_name in (
        "standard_csad",
        "no_intercept_csad",
        "intercept_restored",
        "scsad",
    ):
        fast = fit_fast_hac_model(csad, market, model_name)
        reference, _ = fit_audit_model(
            pd.Series(csad),
            pd.Series(market),
            model_name,
            cov_type="HAC",
            hac_maxlags="auto",
        )
        assert np.isclose(
            fast["target_coefficient"],
            reference["target_coefficient"],
            rtol=1e-9,
            atol=1e-10,
        )
        assert np.isclose(
            fast["target_std_error"],
            reference["target_std_error"],
            rtol=1e-7,
            atol=1e-10,
        )


def test_lagged_liquidity_weights_are_positive_normalized_and_reproducible() -> None:
    returns = np.zeros((20, 5))
    config = {
        "lognormal_sigma": 1.25,
        "liquidity_ar1": 0.97,
        "liquidity_innovation": 0.12,
        "minimum_weight": 1e-12,
    }
    first = simulate_weights(
        returns, "lagged_lognormal_liquidity", config, np.random.default_rng(99)
    )
    second = simulate_weights(
        returns, "lagged_lognormal_liquidity", config, np.random.default_rng(99)
    )
    assert np.all(first > 0)
    assert np.allclose(first.sum(axis=1), 1.0)
    assert np.allclose(first, second)


def test_univariate_moderator_audit_is_full_rank_and_fdr_adjusted() -> None:
    from csad_specification_audit import _descriptive_univariate_meta_regressions

    rows = []
    for model in ("standard_csad", "no_intercept_csad", "scsad"):
        for index in range(12):
            rows.append(
                {
                    "model": model,
                    "target_standardized_coefficient": -0.1 + index * 0.01,
                    "provider": "a" if index < 6 else "b",
                    "frequency": "daily" if index % 2 == 0 else "weekly",
                    "universe_type": "fixed" if index < 6 else "pit",
                    "weighting_class": "equal" if index % 2 == 0 else "lagged",
                    "mean_active_assets": 14.0 + index,
                    "mean_weight_hhi": 0.1 + index * 0.01,
                    "mean_btc_weight": 0.2 + index * 0.01,
                }
            )
    coefficients, diagnostics = _descriptive_univariate_meta_regressions(pd.DataFrame(rows))
    estimated = diagnostics["status"].eq("estimated")
    assert estimated.all()
    assert not diagnostics.loc[estimated, "rank_deficient"].any()
    tested = coefficients["term"].ne("const")
    assert coefficients.loc[tested, "q_value_bh_fdr"].notna().all()

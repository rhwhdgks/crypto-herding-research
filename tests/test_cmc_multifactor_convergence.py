from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cmc_multifactor_convergence import (
    _empirical_expected_abs_deviation,
    build_point_in_time_factors,
    estimate_multifactor_models,
)


def test_factor_sorting_uses_lagged_characteristics_and_excludes_self() -> None:
    date = pd.Timestamp("2020-01-02", tz="UTC")
    rows = []
    for cmc_id in range(1, 41):
        rows.append(
            {
                "snapshot_date": date,
                "cmc_id": cmc_id,
                "name": f"Asset {cmc_id}",
                "symbol": f"A{cmc_id}",
                "asset_return": cmc_id / 1000.0,
                "lagged_size": float(cmc_id),
                "lagged_turnover": float(cmc_id),
                "lagged_momentum": float(cmc_id),
                "market_return": 0.02,
                "target_loo_market_return": 0.019,
                "is_target_member": True,
                "active_assets": 40,
                "lagged_market_cap_usd": float(cmc_id),
            }
        )
    history = pd.DataFrame(rows)
    enriched, factors, diagnostics = build_point_in_time_factors(
        history, tail_fraction=0.30, minimum_leg_assets=10
    )

    small = np.mean(np.arange(1, 13) / 1000.0)
    big = np.mean(np.arange(29, 41) / 1000.0)
    assert factors.loc[date, "factor_size"] == pytest.approx(small - big)
    asset_one = enriched.loc[enriched["cmc_id"].eq(1)].iloc[0]
    expected_loo_small = np.mean(np.arange(2, 13) / 1000.0)
    assert asset_one["factor_size"] == pytest.approx(expected_loo_small - big)
    assert asset_one["factor_market"] == pytest.approx(0.019)
    assert diagnostics["valid_days"].eq(1).all()


def test_multifactor_rolling_model_does_not_use_current_return() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=14, freq="D", tz="UTC")
    x = rng.normal(0.0, 0.02, (len(dates), 4))
    y = 0.001 + x @ np.array([1.1, 0.3, -0.2, 0.5])
    frame = pd.DataFrame(
        {
            "snapshot_date": dates,
            "cmc_id": 1,
            "name": "Asset",
            "symbol": "AAA",
            "asset_return": y,
            "market_return": x[:, 0],
            "is_target_member": True,
            "active_assets": 30,
            "lagged_market_cap_usd": 100.0,
            "factor_market": x[:, 0],
            "factor_size": x[:, 1],
            "factor_liquidity": x[:, 2],
            "factor_momentum": x[:, 3],
        }
    )
    kwargs = dict(
        factor_columns=[
            "factor_market",
            "factor_size",
            "factor_liquidity",
            "factor_momentum",
        ],
        window_observations=8,
        minimum_observations=6,
        maximum_condition_number=1e12,
        minimum_residual_sigma=1e-8,
    )
    baseline = estimate_multifactor_models(frame, **kwargs)
    shocked = frame.copy()
    shocked.loc[10, "asset_return"] = 10.0
    changed = estimate_multifactor_models(shocked, **kwargs)

    beta_columns = [column for column in baseline if column.startswith("beta_")]
    assert np.allclose(
        baseline.loc[10, beta_columns],
        changed.loc[10, beta_columns],
        equal_nan=True,
    )
    assert not np.allclose(
        baseline.loc[11, beta_columns],
        changed.loc[11, beta_columns],
        equal_nan=True,
    )


def test_empirical_counterfactual_excludes_current_error() -> None:
    x = np.arange(24, dtype=float).reshape(6, 4) / 100.0
    coefficient = np.tile(np.array([0.01, 1.0, 0.0, 0.0, 0.0]), (6, 1))
    y = 0.01 + x[:, 0]
    mu = np.full(6, 0.02)
    baseline, counts = _empirical_expected_abs_deviation(
        x, y, coefficient, mu, window=4, minimum_residuals=3
    )
    shocked_y = y.copy()
    shocked_y[4] = 100.0
    changed, _ = _empirical_expected_abs_deviation(
        x, shocked_y, coefficient, mu, window=4, minimum_residuals=3
    )

    assert changed[4] == pytest.approx(baseline[4])
    assert changed[5] != pytest.approx(baseline[5])
    assert counts[4] == 4


def test_multifactor_estimator_produces_normal_and_empirical_expectations() -> None:
    rng = np.random.default_rng(8)
    dates = pd.date_range("2020-01-01", periods=30, freq="D", tz="UTC")
    x = rng.normal(0.0, 0.02, (len(dates), 4))
    frame = pd.DataFrame(
        {
            "snapshot_date": dates,
            "cmc_id": 1,
            "name": "Asset",
            "symbol": "AAA",
            "asset_return": 0.002 + x @ np.array([1.0, 0.2, -0.1, 0.3]) + rng.normal(0, 0.001, len(dates)),
            "market_return": x[:, 0],
            "is_target_member": True,
            "active_assets": 30,
            "lagged_market_cap_usd": 100.0,
            "factor_market": x[:, 0],
            "factor_size": x[:, 1],
            "factor_liquidity": x[:, 2],
            "factor_momentum": x[:, 3],
        }
    )
    result = estimate_multifactor_models(
        frame,
        factor_columns=[
            "factor_market",
            "factor_size",
            "factor_liquidity",
            "factor_momentum",
        ],
        window_observations=15,
        minimum_observations=10,
        maximum_condition_number=1e12,
        minimum_residual_sigma=1e-8,
        empirical_minimum_residuals=8,
    )

    eligible = result.dropna(
        subset=["expected_abs_deviation", "expected_abs_deviation_empirical"]
    )
    assert not eligible.empty
    assert eligible["expected_abs_deviation"].gt(0.0).all()
    assert eligible["expected_abs_deviation_empirical"].gt(0.0).all()

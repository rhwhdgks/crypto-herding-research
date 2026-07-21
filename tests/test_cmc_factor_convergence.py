from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cmc_factor_convergence import (
    add_leave_one_out_market_return,
    aggregate_daily_convergence,
    build_fixed_regimes,
    estimate_point_in_time_factor_model,
    expected_absolute_normal,
    load_point_in_time_estimation_history,
    run_convergence_regressions,
)


def test_leave_one_out_market_return_excludes_current_asset() -> None:
    rows = pd.DataFrame(
        {
            "snapshot_date": [pd.Timestamp("2020-01-01", tz="UTC")] * 3,
            "cmc_id": [1, 2, 3],
            "asset_return": [0.10, 0.00, -0.10],
            "lagged_market_cap_usd": [1.0, 2.0, 1.0],
        }
    )
    result = add_leave_one_out_market_return(rows).set_index("cmc_id")

    assert result.loc[1, "market_return"] == pytest.approx(0.0)
    assert result.loc[1, "loo_market_return"] == pytest.approx(-0.10 / 3.0)
    assert result.loc[2, "loo_market_return"] == pytest.approx(0.0)
    assert result.loc[3, "loo_market_return"] == pytest.approx(0.10 / 3.0)


def test_rolling_model_uses_only_prior_observations() -> None:
    dates = pd.date_range("2020-01-01", periods=9, freq="D", tz="UTC")
    market = np.arange(1.0, 10.0) / 100.0
    frame = pd.DataFrame(
        {
            "snapshot_date": dates,
            "cmc_id": 1,
            "name": "Asset",
            "symbol": "AAA",
            "asset_return": 0.01 + 2.0 * market,
            "lagged_market_cap_usd": 1.0,
            "market_return": market,
            "loo_market_return": market,
            "active_assets": 20,
        }
    )
    baseline = estimate_point_in_time_factor_model(frame, 5, 3, 1e-12, 1e-8)
    shocked = frame.copy()
    shocked.loc[5, "asset_return"] = 99.0
    changed = estimate_point_in_time_factor_model(shocked, 5, 3, 1e-12, 1e-8)

    # Changing y_t cannot change alpha_t or beta_t, but it must affect t+1.
    assert changed.loc[5, "factor_alpha"] == pytest.approx(baseline.loc[5, "factor_alpha"])
    assert changed.loc[5, "factor_beta"] == pytest.approx(baseline.loc[5, "factor_beta"])
    assert changed.loc[6, "factor_beta"] != pytest.approx(baseline.loc[6, "factor_beta"])


def test_estimation_history_keeps_pre_membership_prices(tmp_path) -> None:
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    for number, date in enumerate(dates, start=1):
        pd.DataFrame(
            {
                "snapshot_date": [date],
                "cmc_id": [1],
                "name": ["Asset"],
                "symbol": ["AAA"],
                "price_usd": [float(number)],
            }
        ).to_parquet(snapshot_dir / f"{date.date()}.parquet", index=False)
    member = pd.DataFrame(
        {
            "snapshot_date": [pd.Timestamp("2020-01-04", tz="UTC")],
            "cmc_id": [1],
            "market_return": [0.01],
            "loo_market_return": [0.02],
            "active_assets": [20],
            "lagged_market_cap_usd": [100.0],
        }
    )
    history = load_point_in_time_estimation_history(
        member,
        snapshot_dir,
        "2020-01-01",
        "2020-01-04",
    )

    assert len(history) == 4
    assert history["is_target_member"].tolist() == [False, False, False, True]
    assert history.loc[:2, "loo_market_return"].isna().all()
    assert history.loc[3, "loo_market_return"] == pytest.approx(0.02)
    assert history.loc[3, "asset_return"] == pytest.approx(np.log(4.0 / 3.0))


def test_expected_absolute_normal_matches_special_cases() -> None:
    values = expected_absolute_normal(
        pd.Series([0.0, 2.0]),
        pd.Series([1.0, 1e-8]),
    )
    assert values[0] == pytest.approx(np.sqrt(2.0 / np.pi))
    assert values[1] == pytest.approx(2.0)


def test_daily_aggregation_applies_model_coverage_gate() -> None:
    frame = pd.DataFrame(
        {
            "snapshot_date": [pd.Timestamp("2020-01-01", tz="UTC")] * 3,
            "cmc_id": [1, 2, 3],
            "market_return": [0.01] * 3,
            "asset_return": [0.02, 0.00, 0.01],
            "factor_predicted_return": [0.015, 0.005, np.nan],
            "factor_prediction_error": [0.005, -0.005, np.nan],
            "factor_beta": [1.0, 1.0, np.nan],
            "factor_residual_sigma": [0.01, 0.01, np.nan],
            "observed_abs_deviation": [0.01, 0.01, 0.0],
            "factor_point_deviation": [0.005, 0.005, np.nan],
            "expected_abs_deviation": [0.012, 0.012, np.nan],
        }
    )
    accepted = aggregate_daily_convergence(frame, 2 / 3, 2).iloc[0]
    rejected = aggregate_daily_convergence(frame, 0.80, 2).iloc[0]

    assert bool(accepted["eligible_factor_day"])
    assert accepted["abnormal_convergence"] == pytest.approx(0.002)
    assert not bool(rejected["eligible_factor_day"])


def test_fixed_regime_regressions_produce_six_test_family() -> None:
    rng = np.random.default_rng(31)
    index = pd.date_range("2020-01-01", periods=720, freq="D", tz="UTC")
    market = rng.normal(0.0, 0.03, len(index))
    ratio = 0.01 + 15.0 * market**2 + rng.normal(0.0, 0.002, len(index))
    daily = pd.DataFrame(
        {
            "market_return": market,
            "convergence_ratio": ratio,
            "abnormal_convergence": ratio * 0.03,
            "eligible_factor_day": True,
        },
        index=index,
    )
    break_dates = pd.DataFrame(
        {
            "next_regime_start": pd.to_datetime(
                ["2020-05-01", "2020-09-01", "2021-01-01", "2021-05-01"],
                utc=True,
            )
        }
    )
    regimes = build_fixed_regimes("2020-01-01", "2021-12-20", break_dates)
    targets, coefficients, means = run_convergence_regressions(
        {"window_365": daily},
        regimes,
        {"hac_maxlags": "auto", "fdr_alpha": 0.05},
    )

    assert len(targets) == 6
    assert len(means) == 6
    assert targets["delta2_q_value_bh_fdr"].between(0.0, 1.0).all()
    assert set(coefficients["term"]) == {
        "const",
        "abs_market_return",
        "market_return_sq",
    }
    assert targets["delta2"].gt(0.0).all()

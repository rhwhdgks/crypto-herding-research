from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cmc_fixed_universe import (
    build_chunk_windows,
    build_fixed_panels,
    parse_individual_history_payload,
    run_fixed_regressions,
)


def test_chunk_windows_are_non_overlapping_and_limited_to_400_days() -> None:
    windows = build_chunk_windows(
        {"source_start": "2017-12-31", "source_end": "2020-03-01", "chunk_days": 400}
    )
    assert windows[0][0] == pd.Timestamp("2017-12-31")
    assert windows[-1][1] == pd.Timestamp("2020-03-01")
    assert all((end - start).days + 1 <= 400 for start, end in windows)
    assert all(windows[index][1] + pd.Timedelta(days=1) == windows[index + 1][0] for index in range(len(windows) - 1))


def test_individual_payload_is_filtered_to_checkpoint_window() -> None:
    payload = {
        "status": {"error_code": "0"},
        "data": {
            "id": 7,
            "name": "Alpha",
            "symbol": "OLD",
            "quotes": [
                _quote("2019-12-31", 9.0, 90.0),
                _quote("2020-01-01", 10.0, 100.0),
                _quote("2020-01-02", 11.0, 110.0),
            ],
        },
    }
    frame = parse_individual_history_payload(payload, "AAA", 7, "2020-01-01", "2020-01-02")
    assert frame["date"].tolist() == [
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2020-01-02", tz="UTC"),
    ]
    assert frame["research_symbol"].unique().tolist() == ["AAA"]
    assert frame["provider_symbol"].unique().tolist() == ["OLD"]


def test_daily_primary_and_lagged_sensitivity_use_different_weights() -> None:
    history = pd.DataFrame(
        [
            _history("2019-12-31", 1, "AAA", 100.0, 100.0),
            _history("2019-12-31", 2, "BBB", 100.0, 300.0),
            _history("2020-01-01", 1, "AAA", 110.0, 900.0),
            _history("2020-01-01", 2, "BBB", 90.0, 100.0),
            _history("2020-01-02", 1, "AAA", 121.0, 900.0),
            _history("2020-01-02", 2, "BBB", 81.0, 100.0),
        ]
    )
    analysis = {"start": "2020-01-01", "end": "2020-01-02", "minimum_active_assets": 2}
    primary = build_fixed_panels(
        history,
        {"return_method": "simple", "market_cap_weighting": "contemporaneous"},
        analysis,
    )
    sensitivity = build_fixed_panels(
        history,
        {"return_method": "log", "market_cap_weighting": "lagged"},
        analysis,
    )
    date = pd.Timestamp("2020-01-01", tz="UTC")
    expected_primary = (900.0 * 0.1 + 100.0 * -0.1) / 1000.0
    expected_sensitivity = (100.0 * np.log(1.1) + 300.0 * np.log(0.9)) / 400.0
    assert primary["daily_market"].loc[date] == pytest.approx(expected_primary)
    assert sensitivity["daily_market"].loc[date] == pytest.approx(expected_sensitivity)


def test_missing_calendar_day_is_not_treated_as_a_return() -> None:
    history = pd.DataFrame(
        [
            _history("2019-12-31", 1, "AAA", 100.0, 100.0),
            _history("2020-01-02", 1, "AAA", 120.0, 120.0),
        ]
    )
    panels = build_fixed_panels(
        history,
        {"return_method": "simple", "market_cap_weighting": "contemporaneous"},
        {"start": "2020-01-01", "end": "2020-01-02", "minimum_active_assets": 1},
    )
    assert panels["daily_market"].empty


def test_fixed_regression_reports_standardized_target_coefficient() -> None:
    rng = np.random.default_rng(11)
    daily_index = pd.date_range("2020-01-01", periods=120, freq="D", tz="UTC")
    weekly_index = pd.date_range("2020-01-06", periods=30, freq="7D", tz="UTC")
    daily_market = pd.Series(rng.normal(0, 0.02, len(daily_index)), index=daily_index)
    weekly_market = pd.Series(rng.normal(0, 0.04, len(weekly_index)), index=weekly_index)
    panels = {
        "test_variant": {
            "daily_market": daily_market,
            "daily_csad": 0.03 + 0.4 * daily_market.abs() + rng.normal(0, 0.001, len(daily_index)),
            "weekly_market": weekly_market,
            "weekly_csad": 0.06 + 0.3 * weekly_market.abs() + rng.normal(0, 0.002, len(weekly_index)),
        }
    }
    targets, _, _ = run_fixed_regressions(
        panels,
        {
            "frequencies": ["daily", "weekly"],
            "subperiods": [{"name": "full", "start": "2020-01-01", "end": "2022-12-31"}],
            "regression": {
                "models": ["standard_csad", "no_intercept_csad", "scsad"],
                "cov_type": "HAC",
                "hac_maxlags": "auto",
                "family_size_per_period_variant": 6,
                "fdr_alpha": 0.05,
            },
        },
    )
    assert targets["standardized_target_coefficient"].notna().all()
    assert np.isfinite(targets["standardized_target_coefficient"]).all()


def _quote(date: str, close: float, market_cap: float) -> dict:
    return {
        "timeOpen": f"{date}T00:00:00.000Z",
        "quote": {
            "name": "2781",
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
            "marketCap": market_cap,
            "circulatingSupply": 10.0,
        },
    }


def _history(date: str, cmc_id: int, symbol: str, close: float, cap: float) -> dict:
    return {
        "date": pd.Timestamp(date, tz="UTC"),
        "cmc_id": cmc_id,
        "research_symbol": symbol,
        "provider_symbol": symbol,
        "name": symbol,
        "open_usd": close,
        "high_usd": close,
        "low_usd": close,
        "close_usd": close,
        "volume_usd": 1.0,
        "market_cap_usd": cap,
        "circulating_supply": 10.0,
    }

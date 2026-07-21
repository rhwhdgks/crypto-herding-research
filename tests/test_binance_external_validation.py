from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from binance_external_validation import (
    build_binance_panels,
    evaluate_external_robustness,
    load_binance_daily_history,
)


def _history() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    rows = []
    values = {
        "AAAUSDT": ([100.0, 110.0, 121.0, 133.1], [1.0, 100.0, 1.0, 1.0]),
        "BBBUSDT": ([100.0, 90.0, 81.0, 72.9], [9.0, 100.0, 1.0, 1.0]),
    }
    for symbol, (closes, volumes) in values.items():
        for date, close, volume in zip(dates, closes, volumes, strict=True):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "close_usdt": close,
                    "base_volume": volume,
                    "turnover_proxy_usdt": close * volume,
                }
            )
    return pd.DataFrame(rows)


def _analysis() -> dict:
    return {
        "start": "2024-01-02",
        "end": "2024-01-04",
        "minimum_active_assets": 2,
    }


def test_lagged_turnover_weight_uses_previous_day_only() -> None:
    variant = {
        "name": "lagged_turnover_sensitivity",
        "return_method": "log",
        "market_weighting": "lagged_turnover",
    }
    panels = build_binance_panels(_history(), variant, _analysis())
    market = panels["daily_market"]
    up = np.log(1.1)
    down = np.log(0.9)
    expected_day_two = (up * 100.0 + down * 900.0) / 1000.0
    expected_day_three = (up * 11000.0 + down * 9000.0) / 20000.0
    assert np.isclose(market.loc[pd.Timestamp("2024-01-02", tz="UTC")], expected_day_two)
    assert np.isclose(market.loc[pd.Timestamp("2024-01-03", tz="UTC")], expected_day_three)


def test_load_binance_history_writes_hash_manifest(tmp_path: Path) -> None:
    for symbol in ("AAAUSDT", "BBBUSDT"):
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
                "close": [1.0, 1.1, 1.2],
                "volume": [10.0, 11.0, 12.0],
            }
        ).to_parquet(tmp_path / f"{symbol}_1d.parquet", index=False)
    source = {
        "data_dir": tmp_path,
        "timeframe": "1d",
        "source_start": "2024-01-01",
        "source_end": "2024-01-03",
        "symbols": ["AAAUSDT", "BBBUSDT"],
    }
    history, manifest = load_binance_daily_history(source)
    assert len(history) == 6
    assert manifest["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert set(history["symbol"]) == {"AAAUSDT", "BBBUSDT"}


def test_external_decision_requires_all_four_corrected_cells() -> None:
    rows = []
    for variant in ("primary", "sensitivity"):
        for frequency in ("daily", "weekly"):
            for model in ("no_intercept_csad", "scsad"):
                rows.append(
                    {
                        "variant": variant,
                        "period": "full",
                        "frequency": frequency,
                        "model": model,
                        "coefficient": -1.0,
                        "q_value_bh_fdr": 0.01,
                    }
                )
    targets = pd.DataFrame(rows)
    targets.loc[
        targets["variant"].eq("sensitivity") & targets["frequency"].eq("weekly"),
        "q_value_bh_fdr",
    ] = 0.10
    config = {
        "primary_variant": "primary",
        "sensitivity_variant": "sensitivity",
        "decision_period": "full",
        "required_models": ["no_intercept_csad", "scsad"],
        "required_frequencies": ["daily", "weekly"],
        "alpha": 0.05,
    }
    _, summary = evaluate_external_robustness(targets, config)
    result = summary.set_index("variant")
    assert bool(result.loc["primary", "all_required_cells_pass"])
    assert not bool(result.loc["sensitivity", "all_required_cells_pass"])
    assert result.loc["sensitivity", "passing_cells"] == 2

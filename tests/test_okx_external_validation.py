from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from okx_external_validation import (
    build_chunk_windows,
    build_okx_panels,
    parse_okx_candles_payload,
    parse_okx_instruments_payload,
)


def test_okx_chunk_windows_cover_exact_range_without_overlap() -> None:
    source = {
        "source_start": "2021-04-09",
        "source_end": "2026-04-09",
        "chunk_days": 100,
        "limit": 100,
    }
    windows = build_chunk_windows(source)
    assert len(windows) == 19
    assert windows[0][0] == pd.Timestamp("2021-04-09", tz="UTC")
    assert windows[-1][1] == pd.Timestamp("2026-04-09", tz="UTC")
    assert all(
        right[0] == left[1] + pd.Timedelta(days=1)
        for left, right in zip(windows, windows[1:])
    )


def test_okx_payload_parser_uses_quote_volume_and_rejects_unconfirmed() -> None:
    timestamp = int(pd.Timestamp("2024-01-02", tz="UTC").timestamp() * 1000)
    payload = {
        "code": "0",
        "msg": "",
        "data": [[str(timestamp), "10", "12", "9", "11", "5", "55", "55", "1"]],
    }
    instrument = {"instrument_id": "AAA-USDT", "research_symbol": "AAAUSDT"}
    frame = parse_okx_candles_payload(payload, instrument, "2024-01-01", "2024-01-03")
    assert frame.loc[0, "quote_volume_usdt"] == 55.0
    assert bool(frame.loc[0, "confirmed"])
    payload["data"][0][-1] = "0"
    with pytest.raises(ValueError, match="unconfirmed"):
        parse_okx_candles_payload(payload, instrument, "2024-01-01", "2024-01-03")


def test_okx_listing_aware_panel_adds_new_asset_without_backfill() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    rows = []
    values = {
        "AAAUSDT": ([100.0, 110.0, 121.0, 133.1], [100.0, 200.0, 300.0, 400.0]),
        "BBBUSDT": ([100.0, 90.0, 81.0, 72.9], [900.0, 800.0, 700.0, 600.0]),
    }
    for symbol, (closes, volumes) in values.items():
        for date, close, volume in zip(dates, closes, volumes, strict=True):
            rows.append(
                {
                    "date": date,
                    "instrument_id": symbol.replace("USDT", "-USDT"),
                    "symbol": symbol,
                    "close_usdt": close,
                    "quote_volume_usdt": volume,
                }
            )
    for date, close in zip(dates[1:], [50.0, 55.0, 60.5], strict=True):
        rows.append(
            {
                "date": date,
                "instrument_id": "CCC-USDT",
                "symbol": "CCCUSDT",
                "close_usdt": close,
                "quote_volume_usdt": 500.0,
            }
        )
    history = pd.DataFrame(rows)
    history["date"] = history["date"].astype(object)
    history["close_usdt"] = history["close_usdt"].astype(object)
    history["quote_volume_usdt"] = history["quote_volume_usdt"].astype(object)
    analysis = {
        "start": "2024-01-02",
        "end": "2024-01-04",
        "minimum_active_assets": 2,
    }
    variant = {
        "name": "lagged_quote_volume_sensitivity",
        "return_method": "log",
        "market_weighting": "lagged_quote_volume",
    }
    panels = build_okx_panels(history, variant, analysis)
    coverage = panels["daily_coverage"].set_index("period")
    assert coverage.loc[pd.Timestamp("2024-01-02", tz="UTC"), "active_assets"] == 2
    assert coverage.loc[pd.Timestamp("2024-01-03", tz="UTC"), "active_assets"] == 3
    expected = (np.log(1.1) * 200.0 + np.log(0.9) * 800.0 + np.log(1.1) * 500.0) / 1500.0
    assert np.isclose(
        panels["daily_market"].loc[pd.Timestamp("2024-01-03", tz="UTC")],
        expected,
    )


def test_okx_instrument_metadata_requires_live_spot_pairs() -> None:
    payload = {
        "code": "0",
        "data": [
            {
                "instId": "AAA-USDT",
                "instType": "SPOT",
                "state": "live",
                "baseCcy": "AAA",
                "quoteCcy": "USDT",
                "listTime": "1609459200000",
            }
        ],
    }
    instruments = [{"instrument_id": "AAA-USDT", "research_symbol": "AAAUSDT"}]
    metadata = parse_okx_instruments_payload(payload, instruments)
    assert metadata.loc[0, "list_time"] == pd.Timestamp("2021-01-01", tz="UTC")

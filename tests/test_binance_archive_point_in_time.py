from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from binance_archive_point_in_time import (
    build_listing_episodes,
    build_monthly_membership,
    build_point_in_time_panels,
    classify_symbol_prefixes,
    parse_binance_kline_zip,
    parse_s3_listing,
    run_descriptive_meta_analysis,
)


def test_s3_inventory_parser_supports_pagination_and_symbol_filtering() -> None:
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <IsTruncated>true</IsTruncated><NextMarker>data/spot/monthly/klines/BBB/</NextMarker>
      <CommonPrefixes><Prefix>data/spot/monthly/klines/AAAUSDT/</Prefix></CommonPrefixes>
      <CommonPrefixes><Prefix>data/spot/monthly/klines/AAADOWNUSDT/</Prefix></CommonPrefixes>
      <Contents><Key>data/spot/monthly/klines/AAAUSDT/1d/AAAUSDT-1d-2024-01.zip</Key><ETag>\"abc\"</ETag><Size>42</Size></Contents>
    </ListBucketResult>"""
    parsed = parse_s3_listing(xml)
    assert parsed["is_truncated"]
    assert parsed["next_marker"].endswith("BBB/")
    assert parsed["contents"][0]["etag"] == "abc"
    candidates = classify_symbol_prefixes(
        parsed["prefixes"],
        {"root_prefix": "data/spot/monthly/klines/"},
        {
            "quote_suffix": "USDT",
            "leveraged_suffixes": ["UP", "DOWN"],
            "excluded_base_assets": [],
            "excluded_wrapped_staked_assets": [],
        },
    ).set_index("symbol")
    assert bool(candidates.loc["AAAUSDT", "included"])
    assert not bool(candidates.loc["AAADOWNUSDT", "included"])


def test_kline_zip_parser_handles_millisecond_and_microsecond_timestamps(tmp_path: Path) -> None:
    for unit, multiplier in (("ms", 1_000), ("us", 1_000_000)):
        timestamp = int(pd.Timestamp("2025-01-02", tz="UTC").timestamp() * multiplier)
        row = f"{timestamp},1,2,0.5,1.5,10,{timestamp},15,3,4,6,0\n"
        path = tmp_path / f"AAAUSDT-1d-2025-01-{unit}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("AAAUSDT-1d-2025-01.csv", row)
        parsed = parse_binance_kline_zip(path, "AAAUSDT")
        assert parsed.loc[0, "date"] == pd.Timestamp("2025-01-02", tz="UTC")
        assert parsed.loc[0, "quote_volume_usdt"] == 15.0


def test_kline_parser_deduplicates_only_identical_source_rows(tmp_path: Path) -> None:
    timestamp = int(pd.Timestamp("2026-02-10", tz="UTC").timestamp() * 1_000_000)
    row = f"{timestamp},1,2,0.5,1.5,10,{timestamp},15,3,4,6,0\n"
    path = tmp_path / "AXSUSDT-1d-2026-02.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("AXSUSDT-1d-2026-02.csv", row + row)
    parsed = parse_binance_kline_zip(path, "AXSUSDT")
    assert len(parsed) == 1
    assert parsed.attrs["source_exact_duplicate_rows_removed"] == 1


def test_kline_parser_rejects_conflicting_duplicate_timestamp(tmp_path: Path) -> None:
    timestamp = int(pd.Timestamp("2026-02-10", tz="UTC").timestamp() * 1_000_000)
    first = f"{timestamp},1,2,0.5,1.5,10,{timestamp},15,3,4,6,0\n"
    second = f"{timestamp},1,2,0.5,1.6,10,{timestamp},16,3,4,6,0\n"
    path = tmp_path / "CONFLICTUSDT-1d-2026-02.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("CONFLICTUSDT-1d-2026-02.csv", first + second)
    with pytest.raises(ValueError, match="conflicting duplicate UTC dates"):
        parse_binance_kline_zip(path, "CONFLICTUSDT")


def test_listing_episode_breaks_ticker_after_long_gap() -> None:
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2022-05-01", "2022-05-02", "2022-05-20", "2022-05-21"], utc=True
            ),
            "symbol": ["LUNAUSDT"] * 4,
        }
    )
    result = build_listing_episodes(history, gap_days=7)
    assert result["asset_key"].tolist() == [
        "LUNAUSDT#01",
        "LUNAUSDT#01",
        "LUNAUSDT#02",
        "LUNAUSDT#02",
    ]


def _history_for_membership() -> pd.DataFrame:
    rows = []
    for symbol, volume, close_growth in (
        ("AAAUSDT", 100.0, 1.10),
        ("BBBUSDT", 200.0, 0.90),
        ("CCCUSDT", 300.0, 1.05),
    ):
        dates = pd.date_range("2024-01-01", "2024-02-04", freq="D", tz="UTC")
        close = 100.0
        for date in dates:
            if date.month == 2:
                close *= close_growth
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "base_asset": symbol.removesuffix("USDT"),
                    "listing_episode": 1,
                    "asset_key": f"{symbol}#01",
                    "close_usdt": close,
                    "quote_volume_usdt": volume,
                    "archive_key": f"{symbol}-{date:%Y-%m}.zip",
                }
            )
    return pd.DataFrame(rows)


def test_membership_uses_only_prior_month_and_tie_break_is_deterministic() -> None:
    history = _history_for_membership()
    membership, _ = build_monthly_membership(
        history,
        {"minimum_prior_month_days": 20, "top_n": 2},
        {"start": "2024-02-01", "end": "2024-02-29"},
    )
    assert membership["source_month"].unique().tolist() == ["2024-01"]
    assert membership["asset_key"].tolist() == ["CCCUSDT#01", "BBBUSDT#01"]
    assert membership["rank"].tolist() == [1, 2]


def test_lagged_liquidity_panel_uses_previous_day_weight_and_excludes_nonmember() -> None:
    history = _history_for_membership()
    membership, _ = build_monthly_membership(
        history,
        {"minimum_prior_month_days": 20, "top_n": 2},
        {"start": "2024-02-01", "end": "2024-02-04"},
    )
    panels = build_point_in_time_panels(
        history,
        membership,
        {"return_method": "log", "market_weighting": "lagged_quote_volume"},
        {"start": "2024-02-01", "end": "2024-02-04", "minimum_active_assets": 2},
    )
    expected = (np.log(1.05) * 300.0 + np.log(0.90) * 200.0) / 500.0
    assert np.isclose(panels["daily_market"].loc[pd.Timestamp("2024-02-02", tz="UTC")], expected)
    assert set(panels["daily_panel"].columns) == {"BBBUSDT#01", "CCCUSDT#01"}


def test_meta_analysis_reports_heterogeneity_and_four_cell_bh() -> None:
    rows = []
    for frequency in ("daily", "weekly"):
        for model in ("no_intercept_csad", "scsad"):
            for estimate_id, coefficient in (("a", -0.5), ("b", -1.0), ("c", 0.2)):
                rows.append(
                    {
                        "frequency": frequency,
                        "model": model,
                        "estimate_id": estimate_id,
                        "standardized_target_coefficient": coefficient,
                        "standardized_std_error": 0.2,
                    }
                )
    result = run_descriptive_meta_analysis(pd.DataFrame(rows))
    assert len(result) == 4
    assert result["i_squared"].between(0, 1).all()
    assert result["random_q_value_bh_fdr"].between(0, 1).all()

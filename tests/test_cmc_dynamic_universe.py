from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cmc_dynamic_universe import (
    apply_metadata_exclusions,
    build_daily_research_panel,
    build_monthly_dynamic_universe,
    parse_historical_snapshot_payload,
    run_dynamic_csad_regressions,
    validate_snapshot_frame,
)


def test_historical_payload_is_normalized_and_validated() -> None:
    payload = {
        "status": {"error_code": "0"},
        "data": [
            {
                "id": 1,
                "name": "Alpha",
                "symbol": "AAA",
                "slug": "alpha",
                "cmcRank": 1,
                "circulatingSupply": 10**30,
                "dateAdded": "2017-01-01T00:00:00Z",
                "quotes": [
                    {
                        "name": "2781",
                        "price": 10,
                        "marketCap": 100,
                        "volume24h": 5,
                        "lastUpdated": "2018-01-01T23:59:00Z",
                    }
                ],
            },
            {
                "id": 2,
                "name": "Beta",
                "symbol": "BBB",
                "slug": "beta",
                "cmcRank": 2,
                "circulatingSupply": 20,
                "dateAdded": "2017-01-01T00:00:00Z",
                "quotes": [
                    {
                        "name": "2781",
                        "price": 5,
                        "marketCap": 100,
                        "volume24h": 4,
                        "lastUpdated": "2018-01-01T23:59:00Z",
                    }
                ],
            },
        ],
    }
    frame = parse_historical_snapshot_payload(payload, "2018-01-01")

    validate_snapshot_frame(
        frame,
        "2018-01-01",
        {"retrieval_limit": 2, "minimum_positive_quote_share": 1.0},
    )
    assert frame["cmc_id"].tolist() == [1, 2]
    assert frame["rank"].tolist() == [1, 2]
    assert frame.loc[0, "market_cap_usd"] == pytest.approx(100.0)
    assert frame.loc[0, "circulating_supply"] == pytest.approx(1e30)


def test_metadata_exclusion_is_recomputed_from_cached_tags() -> None:
    metadata = pd.DataFrame(
        {
            "cmc_id": [1, 2],
            "tags_json": [json.dumps(["mineable"]), json.dumps(["stablecoin"])],
            "excluded_by_metadata_tag": [True, False],
        }
    )
    result = apply_metadata_exclusions(metadata, ["stablecoin"])
    assert result["excluded_by_metadata_tag"].tolist() == [False, True]


def test_monthly_universe_uses_only_previous_month_end_and_excludes_pegs() -> None:
    dates = pd.date_range("2020-01-02", "2020-01-31", freq="D", tz="UTC")
    rows = []
    for date in dates:
        for cmc_id, name, price, cap in [
            (1, "Alpha", 10.0, 500_000_000.0),
            (2, "Dollar", 1.0, 400_000_000.0),
            (3, "Peg", 1.01, 300_000_000.0),
            (4, "Small", 4.0, 50_000_000.0),
        ]:
            rows.append(
                {
                    "snapshot_date": date,
                    "cmc_id": cmc_id,
                    "name": name,
                    "symbol": name[:3].upper(),
                    "slug": name.lower(),
                    "rank": cmc_id,
                    "price_usd": price,
                    "market_cap_usd": cap,
                }
            )
    snapshots = pd.DataFrame(rows)
    metadata = pd.DataFrame(
        {
            "cmc_id": [1, 2, 3, 4],
            "tags_json": [
                json.dumps([]),
                json.dumps(["stablecoin"]),
                json.dumps([]),
                json.dumps([]),
            ],
            "excluded_by_metadata_tag": [False, True, False, False],
        }
    )
    universe_cfg = {
        "top_n": 4,
        "minimum_formation_market_cap_usd": 100_000_000,
        "excluded_slug_terms": ["stablecoin", "wrapped", "bridged", "staked"],
        "peg_filter": {
            "trailing_calendar_days": 30,
            "minimum_observations": 20,
            "median_price_lower_usd": 0.90,
            "median_price_upper_usd": 1.10,
            "maximum_price_ratio": 1.10,
        },
    }
    analysis_cfg = {
        "start": "2020-02-01",
        "end": "2020-02-29",
        "minimum_active_assets": 1,
    }

    membership, audit, _ = build_monthly_dynamic_universe(
        snapshots,
        metadata,
        universe_cfg,
        analysis_cfg,
    )

    assert membership["cmc_id"].tolist() == [1]
    assert audit.set_index("cmc_id").loc[2, "excluded_by_metadata_tag"]
    assert audit.set_index("cmc_id").loc[3, "excluded_by_peg_rule"]
    assert audit.set_index("cmc_id").loc[4, "excluded_by_market_cap"]
    assert membership["formation_date"].iloc[0] == pd.Timestamp("2020-01-31", tz="UTC")


def test_daily_panel_uses_exact_previous_day_and_lagged_market_cap_weights() -> None:
    snapshots = pd.DataFrame(
        [
            _snapshot_row("2020-01-31", 1, 100.0, 100.0),
            _snapshot_row("2020-01-31", 2, 100.0, 300.0),
            _snapshot_row("2020-02-01", 1, 110.0, 120.0),
            _snapshot_row("2020-02-01", 2, 90.0, 280.0),
            _snapshot_row("2020-02-03", 1, 121.0, 130.0),
            _snapshot_row("2020-02-03", 2, 81.0, 260.0),
        ]
    )
    membership = pd.DataFrame(
        {
            "month_start": [pd.Timestamp("2020-02-01", tz="UTC")] * 2,
            "cmc_id": [1, 2],
            "formation_date": [pd.Timestamp("2020-01-31", tz="UTC")] * 2,
            "formation_universe_size": [2, 2],
            "rank": [1, 2],
            "market_cap_usd": [100.0, 300.0],
        }
    )
    analysis_cfg = {
        "start": "2020-02-01",
        "end": "2020-02-03",
        "minimum_active_assets": 2,
        "minimum_membership_coverage": 1.0,
    }

    rows, panel, market_return, csad, coverage = build_daily_research_panel(
        snapshots,
        membership,
        analysis_cfg,
    )

    up = np.log(1.1)
    down = np.log(0.9)
    expected_market = (100.0 * up + 300.0 * down) / 400.0
    expected_csad = np.mean([abs(up - expected_market), abs(down - expected_market)])
    date = pd.Timestamp("2020-02-01", tz="UTC")
    assert market_return.loc[date] == pytest.approx(expected_market)
    assert csad.loc[date] == pytest.approx(expected_csad)
    assert panel.loc[pd.Timestamp("2020-02-03", tz="UTC")].isna().all()
    assert int(coverage["eligible_day"].sum()) == 1
    assert rows.loc[rows["snapshot_date"].eq(date), "lagged_market_cap_usd"].tolist() == [100.0, 300.0]


def test_regression_outputs_one_six_test_bh_family_per_period() -> None:
    rng = np.random.default_rng(7)
    daily_index = pd.date_range("2020-01-01", periods=400, freq="D", tz="UTC")
    weekly_index = pd.date_range("2020-01-06", periods=80, freq="7D", tz="UTC")
    daily_market = pd.Series(rng.normal(0, 0.03, len(daily_index)), index=daily_index)
    weekly_market = pd.Series(rng.normal(0, 0.05, len(weekly_index)), index=weekly_index)
    daily_csad = 0.03 + 0.4 * daily_market.abs() - daily_market.pow(2) + rng.normal(0, 0.001, len(daily_index))
    weekly_csad = 0.04 + 0.5 * weekly_market.abs() - weekly_market.pow(2) + rng.normal(0, 0.002, len(weekly_index))
    analysis_cfg = {
        "frequencies": ["daily", "weekly"],
        "regression": {
            "models": ["standard_csad", "no_intercept_csad", "scsad"],
            "cov_type": "HAC",
            "hac_maxlags": "auto",
            "family_size_per_period": 6,
            "fdr_alpha": 0.05,
        },
        "subperiods": [
            {"name": "full_sample", "start": "2020-01-01", "end": "2021-12-31"}
        ],
    }

    targets, coefficients, diagnostics = run_dynamic_csad_regressions(
        {"daily": (daily_csad, daily_market), "weekly": (weekly_csad, weekly_market)},
        analysis_cfg,
    )

    assert len(targets) == 6
    assert targets["q_value_bh_fdr"].between(0, 1).all()
    assert set(targets["model"]) == {"standard_csad", "no_intercept_csad", "scsad"}
    assert not coefficients.empty
    assert not diagnostics.empty


def _snapshot_row(date: str, cmc_id: int, price: float, market_cap: float) -> dict:
    return {
        "snapshot_date": pd.Timestamp(date, tz="UTC"),
        "cmc_id": cmc_id,
        "name": f"Asset {cmc_id}",
        "symbol": f"A{cmc_id}",
        "slug": f"asset-{cmc_id}",
        "rank": cmc_id,
        "price_usd": price,
        "market_cap_usd": market_cap,
    }

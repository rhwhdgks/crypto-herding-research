from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from itertools import combinations

from tick_event_schema import LegacyTickSchemaError, require_tick_schema_v2, validate_tick_analysis_config
from tick_herding import compute_conditional_run_z
from tick_lead_lag import _cluster_bootstrap_difference, build_lead_lag_frame, summarize_lead_lag
from tick_short_horizon import compute_bucket_tick_statistics, prepare_micro_herding_frame


def test_conditional_run_z_matches_formula() -> None:
    runs, category_count, total = 2, 4, 10
    other = total - category_count
    expected_mean = category_count * (other + 1) / total
    expected_variance = (
        category_count
        * other
        * (category_count - 1)
        * (other + 1)
        / ((total**2) * (total - 1))
    )
    expected = (runs - expected_mean) / np.sqrt(expected_variance)
    assert compute_conditional_run_z(runs, category_count, total) == pytest.approx(expected)


def test_conditional_run_formula_matches_exhaustive_binary_permutations() -> None:
    n, k = 7, 3
    run_counts = []
    for positions in combinations(range(n), k):
        sequence = np.zeros(n, dtype=int)
        sequence[list(positions)] = 1
        runs = int(np.sum((sequence == 1) & (np.r_[0, sequence[:-1]] == 0)))
        run_counts.append(runs)
    m = n - k
    expected_mean = k * (m + 1) / n
    expected_variance = k * m * (k - 1) * (m + 1) / ((n**2) * (n - 1))
    assert np.mean(run_counts) == pytest.approx(expected_mean)
    assert np.var(run_counts, ddof=0) == pytest.approx(expected_variance)


def test_clustered_and_alternating_sequences_have_expected_signs() -> None:
    assert compute_conditional_run_z(runs=1, category_count=4, n_transactions=8) < 0
    assert compute_conditional_run_z(runs=4, category_count=4, n_transactions=8) > 0
    assert compute_conditional_run_z(2, 4, 10) == compute_conditional_run_z(2, 4, 10)


@pytest.mark.parametrize("runs,count,total", [(0, 0, 10), (1, 10, 10), (1, 1, 1), (1, 1, 10)])
def test_conditional_run_z_degenerate_cases_are_nan(runs: int, count: int, total: int) -> None:
    assert np.isnan(compute_conditional_run_z(runs, count, total))


def test_run_side_and_price_direction_are_independent() -> None:
    timestamp = pd.date_range("2026-01-01", periods=10, freq="s", tz="UTC")
    ticks = pd.DataFrame(
        {
            "timestamp": timestamp,
            "price": [100, 99, 98, 97, 96, 95, 96, 97, 99, 101],
            "quantity": 1.0,
            "quote_quantity": 100.0,
            "is_buyer_maker": [True] * 5 + [False] * 5,
            "tick_type": ["down"] * 5 + ["up"] * 4 + ["zero"],
            "run_id": [1] * 5 + [2] * 4 + [3],
            "run_length": [5] * 5 + [4] * 4 + [1],
        }
    )
    record = compute_bucket_tick_statistics(ticks, "TEST", timestamp[0].floor("15min"), 15)
    assert record["run_clustering_side"] == "down"
    bucket = pd.DataFrame([record])
    config = {
        "analysis": {
            "session_hours_utc": [0],
            "run_clustering_score_percentile": 0.15,
            "lookback_days_for_threshold": 1,
            "min_trades_per_bucket": 1,
            "forward_horizons_minutes": [30],
            "flat_return_epsilon": 0.0,
        }
    }
    result = prepare_micro_herding_frame(bucket, config)
    assert result.loc[0, "run_clustering_side"] == "down"
    assert result.loc[0, "price_direction"] == "up"
    assert result.loc[0, "signal_timestamp"] == result.loc[0, "bucket_end"]


def test_up_run_clustering_can_coexist_with_negative_price_return() -> None:
    timestamp = pd.date_range("2026-01-01", periods=10, freq="s", tz="UTC")
    ticks = pd.DataFrame(
        {
            "timestamp": timestamp,
            "price": [100, 101, 102, 103, 104, 105, 104, 102, 99, 98],
            "quantity": 1.0,
            "quote_quantity": 100.0,
            "is_buyer_maker": False,
            "tick_type": ["up"] * 5 + ["down"] * 4 + ["zero"],
            "run_id": [1] * 5 + [2] * 4 + [3],
            "run_length": [5] * 5 + [4] * 4 + [1],
        }
    )
    record = compute_bucket_tick_statistics(ticks, "TEST", timestamp[0].floor("15min"), 15)
    result = prepare_micro_herding_frame(
        pd.DataFrame([record]),
        {
            "analysis": {
                "session_hours_utc": [0],
                "run_clustering_score_percentile": 0.15,
                "lookback_days_for_threshold": 1,
                "min_trades_per_bucket": 1,
                "forward_horizons_minutes": [30],
            }
        },
    )
    assert result.loc[0, "run_clustering_side"] == "up"
    assert result.loc[0, "price_direction"] == "down"


def test_forward_return_requires_exact_clock_bucket() -> None:
    starts = pd.to_datetime(
        ["2026-01-01T00:00:00Z", "2026-01-01T00:15:00Z", "2026-01-01T00:45:00Z"]
    )
    frame = pd.DataFrame(
        {
            "symbol": "TEST",
            "bucket_start": starts,
            "interval_minutes": 15,
            "transaction_count": 10,
            "last_price": [100.0, 101.0, 103.0],
            "bucket_return": [0.0, 0.01, 0.01],
            "run_clustering_score": [-1.0, -1.1, -1.2],
            "run_clustering_side": "down",
            "aggressor_imbalance": 0.0,
        }
    )
    config = {
        "analysis": {
            "session_hours_utc": [0],
            "run_clustering_score_percentile": 0.15,
            "lookback_days_for_threshold": 1,
            "min_trades_per_bucket": 1,
            "forward_horizons_minutes": [30],
        }
    }
    result = prepare_micro_herding_frame(frame, config).set_index("bucket_start")
    first = result.loc[pd.Timestamp("2026-01-01T00:00:00Z")]
    assert np.isnan(first["forward_return_30m"])
    assert not bool(first["horizon_is_exact_30m"])
    assert first["signal_timestamp"] == pd.Timestamp("2026-01-01T00:15:00Z")


def test_future_score_changes_do_not_change_past_thresholds() -> None:
    starts = pd.date_range("2026-01-01", periods=194, freq="15min", tz="UTC")
    base = pd.DataFrame(
        {
            "symbol": "TEST",
            "bucket_start": starts,
            "interval_minutes": 15,
            "transaction_count": 10,
            "last_price": 100.0,
            "bucket_return": 0.0,
            "run_clustering_score": np.linspace(-3.0, -1.0, len(starts)),
            "run_clustering_side": "down",
            "aggressor_imbalance": 0.0,
        }
    )
    config = {
        "analysis": {
            "session_hours_utc": list(range(24)),
            "run_clustering_score_percentile": 0.15,
            "lookback_days_for_threshold": 1,
            "min_trades_per_bucket": 1,
            "forward_horizons_minutes": [30],
        }
    }
    first = prepare_micro_herding_frame(base, config)
    changed = base.copy()
    changed.loc[changed.index[-1], "run_clustering_score"] = -999.0
    second = prepare_micro_herding_frame(changed, config)
    pd.testing.assert_series_equal(
        first.loc[: len(first) - 2, "run_clustering_threshold"],
        second.loc[: len(second) - 2, "run_clustering_threshold"],
    )


def test_legacy_cache_and_ambiguous_config_are_rejected() -> None:
    with pytest.raises(LegacyTickSchemaError):
        require_tick_schema_v2(pd.DataFrame({"event_label": ["micro_herding_down"]}))
    with pytest.raises(ValueError, match="Ambiguous"):
        validate_tick_analysis_config({"directions": ["down"]})


def test_lead_lag_uses_explicit_filter_and_cluster_bootstrap_metadata() -> None:
    timestamps = pd.date_range("2026-01-01", periods=16, freq="12h", tz="UTC")
    rows = []
    for symbol in ["LEADER", "TARGET"]:
        for index, timestamp in enumerate(timestamps):
            is_event = symbol == "LEADER" and index % 3 == 0
            rows.append(
                {
                    "symbol": symbol,
                    "bucket_start": timestamp,
                    "signal_timestamp": timestamp + pd.Timedelta(minutes=15),
                    "schema_version": 2,
                    "run_clustering_side": "down" if is_event else "up",
                    "price_direction": "up",
                    "aggressor_direction": "buy",
                    "is_micro_run_clustering_event": is_event,
                    "forward_return_30m": 0.01 if index % 3 == 0 else -0.001,
                    "hour_utc": timestamp.hour,
                    "is_target_session": True,
                    "meets_trade_count": True,
                    "bucket_return": 0.0,
                    "transaction_count": 100,
                    "run_clustering_score": -2.0,
                }
            )
    micro = pd.DataFrame(rows)
    lead_lag = build_lead_lag_frame(
        micro,
        target_symbol="TARGET",
        leader_symbols=["LEADER"],
        interval_minutes=15,
        forward_horizon_minutes=30,
        event_filter={"run_clustering_side": ["down"]},
    )
    summary = summarize_lead_lag(lead_lag, ["LEADER"])
    assert summary.loc[0, "block_method"] == "utc_day_cluster_bootstrap"
    assert summary.loc[0, "n_unique_days"] > 1
    assert not bool(summary.loc[0, "inference_eligible"])
    assert pd.isna(summary.loc[0, "p_value_block"])


def test_sparse_lead_lag_cells_are_excluded_from_inference() -> None:
    valid = pd.DataFrame(
        {
            "bucket_start": pd.date_range("2026-01-01", periods=20, freq="D", tz="UTC"),
            "target_forward_return": np.linspace(-0.01, 0.01, 20),
            "forward_horizon_minutes": 30,
        }
    )
    event_mask = pd.Series([True, True] + [False] * 18)
    result = _cluster_bootstrap_difference(valid, event_mask, seed=7)
    assert result["inference_eligible"] is False
    assert "insufficient_nonoverlapping_events" in result["inference_exclusion_reason"]
    assert np.isnan(result["p_value_block"])

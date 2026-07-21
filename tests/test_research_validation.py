from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_validation import (
    CandidateSpec,
    apply_thresholds,
    fit_quantile_thresholds,
    run_selection_aware_permutation,
    simulate_complete_basket,
)


def test_thresholds_are_fit_on_train_and_overlap_is_rejected() -> None:
    train = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC"),
            "flush": [1.0, 2.0, 3.0, 100.0],
        }
    )
    artifact = fit_quantile_thresholds(train, "timestamp", {"flush": 0.5})
    oos = pd.DataFrame({"timestamp": [pd.Timestamp("2025-01-06", tz="UTC")], "flush": [-999.0]})
    applied = apply_thresholds(oos, artifact, "timestamp")
    assert applied.loc[0, "flush_fitted_threshold"] == pytest.approx(2.5)
    overlapping = pd.DataFrame({"timestamp": [pd.Timestamp("2025-01-04", tz="UTC")]})
    with pytest.raises(ValueError, match="overlap"):
        apply_thresholds(overlapping, artifact, "timestamp")
    changed_oos = oos.assign(flush=999999.0)
    changed_applied = apply_thresholds(changed_oos, artifact, "timestamp")
    assert changed_applied.loc[0, "flush_fitted_threshold"] == applied.loc[0, "flush_fitted_threshold"]


def test_selection_aware_permutation_uses_add_one_p_value() -> None:
    count = 20
    frame = pd.DataFrame(
        {
            "leader": "DOGEUSDT",
            "bucket_start": pd.date_range("2025-01-01", periods=count, freq="15min", tz="UTC"),
            "target": ["ADAUSDT", "AVAXUSDT"] * 10,
            "price_direction": ["down"] * 5 + ["up"] * 15,
            "run_clustering_side": "down",
            "is_event": True,
            "hour_utc": 17,
            "forward_return": np.linspace(0.02, -0.01, count),
        }
    )
    candidates = (
        CandidateSpec("DOGEUSDT", ("ADAUSDT",), "down", "down", 30, (17,)),
        CandidateSpec("DOGEUSDT", ("AVAXUSDT",), "down", "down", 30, (17,)),
    )
    observed, metadata = run_selection_aware_permutation(frame, candidates, n_draws=19, seed=7)
    assert len(observed) == 2
    assert metadata["method"] == "selection_aware_circular_shift_max_stat"
    assert metadata["p_value_add_one"] >= 1 / 20
    assert set(metadata["shifted_state_columns"]) >= {
        "is_event",
        "price_direction",
        "run_clustering_side",
    }


def test_execution_requires_complete_basket_and_compounds_with_true_drawdown() -> None:
    timestamps = [
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:15:00Z"),
        pd.Timestamp("2025-01-01T01:00:00Z"),
    ]
    rows = [
        {"signal_timestamp": timestamps[0], "target": "A", "gross_return": 0.10},
        {"signal_timestamp": timestamps[0], "target": "B", "gross_return": 0.10},
        {"signal_timestamp": timestamps[1], "target": "A", "gross_return": 0.50},
        {"signal_timestamp": timestamps[2], "target": "A", "gross_return": -0.10},
        {"signal_timestamp": timestamps[2], "target": "B", "gross_return": -0.10},
    ]
    trades, summary = simulate_complete_basket(
        pd.DataFrame(rows),
        basket_targets=("A", "B"),
        holding_minutes=30,
        overlap_policy="skip",
        round_trip_fee=0.0,
    )
    assert len(trades) == 2
    assert summary["terminal_return"] == pytest.approx(1.1 * 0.9 - 1.0)
    assert summary["max_drawdown"] == pytest.approx(-0.10)
    assert summary["skipped_incomplete_basket"] == 1
    assert summary["arithmetic_return_sum"] != pytest.approx(summary["terminal_return"])


def test_overlap_and_cost_policy_are_monotone() -> None:
    timestamps = [pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:15:00Z")]
    rows = [
        {"signal_timestamp": timestamp, "target": target, "gross_return": 0.02}
        for timestamp in timestamps
        for target in ["A", "B"]
    ]
    low_cost_trades, low = simulate_complete_basket(
        pd.DataFrame(rows), ("A", "B"), 30, "skip_while_position_open", 0.001
    )
    _, high = simulate_complete_basket(
        pd.DataFrame(rows), ("A", "B"), 30, "skip_while_position_open", 0.01
    )
    assert low["trade_count"] == 1
    assert low["skipped_overlap"] == 1
    assert high["terminal_return"] <= low["terminal_return"]
    assert "equity_after" in low_cost_trades.columns


def test_maker_requires_fill_model() -> None:
    frame = pd.DataFrame(
        [{"signal_timestamp": "2025-01-01T00:00:00Z", "target": "A", "gross_return": 0.01}]
    )
    with pytest.raises(ValueError, match="fill_probability"):
        simulate_complete_basket(frame, ("A",), 30, "allow", 0.0, execution_mode="maker")

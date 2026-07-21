import pandas as pd
import pytest

from tick_raw_confirmatory import validate_confirmatory_raw_frame
from tick_semantic_validation import analyze_run_aggressor_semantics


def test_aggressor_proxy_uses_up_buy_and_down_sell() -> None:
    rows = []
    for symbol in ["A", "B"]:
        for index in range(100):
            side = "up" if index % 2 == 0 else "down"
            expected = "buy" if side == "up" else "sell"
            opposite = "sell" if expected == "buy" else "buy"
            rows.append(
                {
                    "symbol": symbol,
                    "is_micro_run_clustering_event": True,
                    "run_clustering_side": side,
                    "aggressor_direction": expected if index < 80 else opposite,
                    "aggressor_imbalance": 0.2,
                }
            )
    summary, table = analyze_run_aggressor_semantics(
        pd.DataFrame(rows),
        ["A", "B"],
        minimum_events=30,
        family_size=3,
        fdr_alpha=0.05,
        proxy_minimum_concordance=0.60,
    )
    pooled = summary.loc[summary["scope"] == "pooled"].iloc[0]
    assert pooled["directional_concordance"] == pytest.approx(0.8)
    assert bool(pooled["supports_aggressor_direction_proxy"])
    assert len(table) == 27


def test_confirmatory_validation_rejects_missing_aggressor() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A"],
            "bucket_start": [pd.Timestamp("2024-01-01T00:00:00Z")],
            "signal_timestamp": [pd.Timestamp("2024-01-01T00:15:00Z")],
            "schema_version": [2],
            "aggressor_imbalance": [float("nan")],
            "is_micro_run_clustering_event": [False],
            "is_control_bucket": [True],
        }
    )
    config = {
        "symbols": ["A"],
        "expected_rows": 1,
        "expected_start": "2024-01-01T00:00:00Z",
        "expected_end": "2024-01-01T00:00:00Z",
        "expected_interval_minutes": 15,
        "minimum_aggressor_available_share": 1.0,
    }
    with pytest.raises(ValueError, match="aggressor"):
        validate_confirmatory_raw_frame(frame, config)


def test_confirmatory_validation_rejects_incomplete_symbol_grid() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "bucket_start": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-01T00:30:00Z"], utc=True
            ),
            "signal_timestamp": pd.to_datetime(
                ["2024-01-01T00:15:00Z", "2024-01-01T00:45:00Z"], utc=True
            ),
            "schema_version": [2, 2],
            "aggressor_imbalance": [0.1, -0.1],
            "is_micro_run_clustering_event": [False, True],
            "is_control_bucket": [True, False],
        }
    )
    config = {
        "symbols": ["A"],
        "expected_rows": 2,
        "expected_start": "2024-01-01T00:00:00Z",
        "expected_end": "2024-01-01T00:30:00Z",
        "expected_interval_minutes": 15,
        "minimum_aggressor_available_share": 1.0,
    }
    with pytest.raises(ValueError, match="grid is incomplete"):
        validate_confirmatory_raw_frame(frame, config)

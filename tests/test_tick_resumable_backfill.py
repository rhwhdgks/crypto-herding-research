import pandas as pd
import pytest

from tick_resumable_backfill import _resolve_max_workers, monthly_cache_path, validate_monthly_cache


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "bucket_start": pd.to_datetime(
                ["2024-04-01T00:00:00Z", "2024-04-01T00:15:00Z"],
                utc=True,
            ),
            "interval_minutes": [15, 15],
            "transaction_count": [100, 120],
            "last_price": [1.0, 1.1],
            "aggressor_imbalance": [0.1, -0.2],
        }
    )


def test_monthly_cache_path_is_stable(tmp_path) -> None:
    path = monthly_cache_path(tmp_path, "BTCUSDT", "2024-04", 15)
    assert path == tmp_path / "BTCUSDT" / "BTCUSDT_2024-04_15m.parquet"


def test_validate_monthly_cache_accepts_complete_frame() -> None:
    validate_monthly_cache(_valid_frame(), "BTCUSDT", "2024-04", 15)


def test_validate_monthly_cache_rejects_wrong_month() -> None:
    frame = _valid_frame()
    frame.loc[1, "bucket_start"] = pd.Timestamp("2024-05-01T00:00:00Z")
    with pytest.raises(ValueError, match="outside"):
        validate_monthly_cache(frame, "BTCUSDT", "2024-04", 15)


def test_resolve_max_workers_is_bounded_by_symbol_count() -> None:
    assert _resolve_max_workers({"max_workers": 4}, 2) == 2


def test_resolve_max_workers_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _resolve_max_workers({"max_workers": 0}, 7)

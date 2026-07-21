from __future__ import annotations

import pandas as pd
import pytest

from event_detection import detect_events
from event_sentiment import attach_event_sentiment_features


def test_sentiment_requires_first_seen_and_uses_availability_time() -> None:
    events = pd.DataFrame({"timestamp": [pd.Timestamp("2025-01-02T00:00:00Z")]})
    historical = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2025-01-01T23:55:00Z")],
            "sentiment_score": [1.0],
            "sentiment_label": ["positive"],
        }
    )
    with pytest.raises(ValueError, match="first_seen"):
        attach_event_sentiment_features(events, historical, [15])
    historical["first_seen_at_utc"] = pd.Timestamp("2025-01-02T00:05:00Z")
    result = attach_event_sentiment_features(events, historical, [15])
    assert result.loc[0, "news_count_15m"] == 0


def test_baseline_event_is_named_low_dispersion_not_herding() -> None:
    index = pd.date_range("2025-01-01", periods=20, freq="min", tz="UTC")
    csad = pd.Series([0.01] * 10 + [0.001] * 10, index=index)
    market = pd.Series([0.001] * 20, index=index)
    config = {
        "lookback_window": 5,
        "min_history": 5,
        "volatility_window": 2,
        "cooldown_periods": 0,
        "low_dispersion": {
            "csad_low_percentile": 0.5,
            "market_abs_upper_percentile": 1.0,
        },
        "shock": {"abs_return_percentile": 1.0, "volatility_percentile": 1.0},
    }
    result = detect_events(csad, market, config)
    assert "is_low_dispersion_event" in result.columns
    assert "is_herding_event" not in result.columns
    assert "herding" not in set(result["event_type"])

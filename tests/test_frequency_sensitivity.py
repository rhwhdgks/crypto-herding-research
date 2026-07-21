import numpy as np
import pandas as pd

from frequency_sensitivity import (
    aggregate_complete_log_returns,
    benjamini_hochberg,
    run_frequency_sensitivity,
)


def test_aggregate_complete_log_returns_drops_partial_bins() -> None:
    index = pd.date_range("2024-04-08 00:01:00", periods=9, freq="1min", tz="UTC")
    panel = pd.DataFrame({"A": np.ones(9), "B": np.full(9, 2.0)}, index=index)

    result = aggregate_complete_log_returns(panel, 5)

    assert list(result.index) == [pd.Timestamp("2024-04-08 00:05:00", tz="UTC")]
    assert result.loc[result.index[0], "A"] == 5.0
    assert result.loc[result.index[0], "B"] == 10.0


def test_benjamini_hochberg_is_monotone_in_p_value_order() -> None:
    p_values = pd.Series([0.01, 0.04, 0.03, 0.002])

    adjusted = benjamini_hochberg(p_values)

    ordered = pd.DataFrame({"p": p_values, "q": adjusted}).sort_values("p")
    assert ordered["q"].is_monotonic_increasing
    assert np.allclose(adjusted.to_numpy(), [0.02, 0.04, 0.04, 0.008])


def test_frequency_family_size_is_validated() -> None:
    index = pd.date_range("2024-01-01", periods=120, freq="1min", tz="UTC")
    panel = pd.DataFrame(
        {
            "A": np.sin(np.arange(120) / 5) / 1000,
            "B": np.cos(np.arange(120) / 7) / 1000,
            "C": np.sin(np.arange(120) / 9) / 1000,
        },
        index=index,
    )
    analysis = {
        "frequencies": [{"label": "1m", "minutes": 1}],
        "universes": [
            {"name": "all", "display_name": "all", "symbols": ["A", "B", "C"]},
            {"name": "small", "display_name": "small", "symbols": ["A", "B"]},
        ],
    }

    try:
        run_frequency_sensitivity(
            panel,
            analysis_cfg=analysis,
            regression_cfg={"cov_type": "HAC", "hac_maxlags": 1},
            multiple_testing_cfg={"family_size": 3, "alpha": 0.05},
            decision_cfg={"required_q_value": 0.05},
        )
    except ValueError as error:
        assert "family_size" in str(error)
    else:
        raise AssertionError("Expected a family-size validation error")

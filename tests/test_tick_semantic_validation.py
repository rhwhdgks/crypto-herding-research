import numpy as np
import pandas as pd

from tick_semantic_validation import (
    _leave_one_out_cross_sectional_mean,
    analyze_run_price_semantics,
)


def test_leave_one_out_cross_sectional_mean_excludes_own_value() -> None:
    frame = pd.DataFrame(
        {
            "bucket_start": [pd.Timestamp("2024-01-01", tz="UTC")] * 3,
            "value": [1.0, 2.0, 6.0],
        }
    )

    result = _leave_one_out_cross_sectional_mean(frame, "value")

    assert np.allclose(result.to_numpy(), [4.0, 3.5, 1.5])


def test_run_price_proxy_requires_effect_size_not_only_significance() -> None:
    rows = []
    for symbol in ["A", "B"]:
        for index in range(100):
            side = "up" if index % 2 == 0 else "down"
            price = side if index < 55 else ("down" if side == "up" else "up")
            rows.append(
                {
                    "symbol": symbol,
                    "is_micro_run_clustering_event": True,
                    "run_clustering_side": side,
                    "price_direction": price,
                }
            )
    frame = pd.DataFrame(rows)

    summary, contingency = analyze_run_price_semantics(
        frame,
        expected_symbols=["A", "B"],
        minimum_events=30,
        family_size=3,
        fdr_alpha=0.05,
        proxy_minimum_concordance=0.60,
    )

    assert len(contingency) == 27
    assert not bool(summary.loc[summary["scope"] == "pooled", "supports_price_direction_proxy"].iloc[0])


def test_ineligible_directional_scope_is_not_reported_significant() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A"] * 10,
            "is_micro_run_clustering_event": [True] * 10,
            "run_clustering_side": ["up"] * 10,
            "price_direction": ["up"] * 10,
        }
    )

    summary, _ = analyze_run_price_semantics(
        frame,
        expected_symbols=["A"],
        minimum_events=30,
        family_size=2,
        fdr_alpha=0.05,
        proxy_minimum_concordance=0.60,
    )

    assert not summary["inference_eligible"].any()
    assert summary["binomial_q_value_bh_fdr"].isna().all()
    assert not summary["supports_price_direction_proxy"].any()

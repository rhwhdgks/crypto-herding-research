from __future__ import annotations

import pandas as pd
import pytest

from cmc_temporal_validation import (
    compare_historical_and_holdout,
    evaluate_temporal_persistence,
)


def test_temporal_persistence_requires_all_four_corrected_cells() -> None:
    rows = []
    for variant in ["replication_primary", "no_lookahead_sensitivity"]:
        for frequency in ["daily", "weekly"]:
            for model in ["no_intercept_csad", "scsad"]:
                rows.append(
                    {
                        "variant": variant,
                        "period": "holdout_full",
                        "frequency": frequency,
                        "model": model,
                        "coefficient": -1.0,
                        "q_value_bh_fdr": 0.01,
                    }
                )
    rows[3]["q_value_bh_fdr"] = 0.10
    detail, summary = evaluate_temporal_persistence(
        pd.DataFrame(rows),
        {
            "required_models": ["no_intercept_csad", "scsad"],
            "required_frequencies": ["daily", "weekly"],
            "alpha": 0.05,
        },
    )
    result = summary.set_index("variant")
    assert not bool(result.loc["replication_primary", "all_required_cells_pass"])
    assert result.loc["replication_primary", "passed_cells"] == 3
    assert bool(result.loc["no_lookahead_sensitivity", "all_required_cells_pass"])
    assert len(detail) == 8


def test_historical_holdout_comparison_preserves_cell_identity() -> None:
    historical = pd.DataFrame(
        [_target("replication_primary", "full_sample", "daily", "scsad", -2.0, True)]
    )
    holdout = pd.DataFrame(
        [_target("replication_primary", "holdout_full", "daily", "scsad", -1.0, True)]
    )
    result = compare_historical_and_holdout(
        historical,
        holdout,
        {
            "historical_variant": "replication_primary",
            "historical_period": "full_sample",
            "holdout_variant": "replication_primary",
            "holdout_period": "holdout_full",
        },
    )
    assert result.loc[0, "coefficient_difference"] == pytest.approx(1.0)
    assert result.loc[0, "absolute_magnitude_ratio"] == pytest.approx(0.5)
    assert result.loc[0, "standardized_absolute_magnitude_ratio"] == pytest.approx(0.5)
    assert bool(result.loc[0, "coefficient_sign_matches"])


def _target(variant: str, period: str, frequency: str, model: str, coefficient: float, support: bool) -> dict:
    return {
        "variant": variant,
        "period": period,
        "frequency": frequency,
        "model": model,
        "target_term": "market_return_cu",
        "coefficient": coefficient,
        "standardized_target_coefficient": coefficient / 10.0,
        "t_stat": -2.0,
        "q_value_bh_fdr": 0.01,
        "supports_herding": support,
        "nobs": 100,
    }

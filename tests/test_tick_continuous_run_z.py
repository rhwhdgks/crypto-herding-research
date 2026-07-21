import numpy as np
import pandas as pd
import pytest

from tick_continuous_run_z import (
    _attach_predeclared_decisions,
    apply_scaling_artifact,
    assign_sample_split,
    fit_scaling_artifact,
    leave_one_out_cross_sectional_mean,
    prepare_continuous_run_z_frame,
    scaled_column,
)


SPLIT = {
    "development_start": "2024-04-08T00:00:00Z",
    "oos_start": "2025-04-08T00:00:00Z",
    "oos_end_exclusive": "2026-04-08T00:00:00Z",
}


def test_assign_sample_split_uses_exact_half_open_boundaries() -> None:
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2024-04-07T23:45:00Z",
                "2024-04-08T00:00:00Z",
                "2025-04-07T23:45:00Z",
                "2025-04-08T00:00:00Z",
                "2026-04-07T23:45:00Z",
                "2026-04-08T00:00:00Z",
            ],
            utc=True,
        )
    )

    result = assign_sample_split(timestamps, SPLIT)

    assert result.tolist() == [
        "outside",
        "development",
        "development",
        "oos",
        "oos",
        "outside",
    ]


def test_prepare_frame_blocks_future_outcomes_that_cross_split_boundaries() -> None:
    buckets = pd.to_datetime(
        [
            "2025-04-07T23:00:00Z",
            "2025-04-07T23:30:00Z",
            "2026-04-07T23:00:00Z",
            "2026-04-07T23:30:00Z",
        ],
        utc=True,
    )
    rows = []
    for bucket in buckets:
        for symbol_index, symbol in enumerate(["A", "B"]):
            rows.append(
                {
                    "bucket_start": bucket,
                    "signal_timestamp": bucket + pd.Timedelta(minutes=15),
                    "symbol": symbol,
                    "transaction_count": 500,
                    "total_quote_quantity": 1_000 + symbol_index,
                    "bucket_return": 0.001 * (1 if symbol_index == 0 else -1),
                    "run_z_up": -1.0,
                    "run_z_down": -0.5,
                    "run_z_zero": 0.1,
                    "aggressor_imbalance": 0.2,
                    "price_direction": "up",
                    "forward_return_30m": 0.003,
                    "horizon_is_exact_30m": True,
                }
            )
    config = {
        "split": SPLIT,
        "analysis": {
            "horizon_minutes": 30,
            "trailing_volatility_buckets": 1,
            "minimum_transaction_count": 200,
        },
    }

    prepared = prepare_continuous_run_z_frame(pd.DataFrame(rows), config)
    availability = (
        prepared.groupby("bucket_start")["future_outcome_within_split"]
        .first()
        .to_dict()
    )

    assert availability[buckets[0]]
    assert not availability[buckets[1]]
    assert availability[buckets[2]]
    assert not availability[buckets[3]]


def test_scaling_artifact_is_fit_only_on_development_values() -> None:
    development = pd.DataFrame(
        {
            "bucket_start": pd.to_datetime(
                ["2024-04-08", "2024-04-09", "2024-04-10"], utc=True
            ),
            "feature": [1.0, 2.0, 3.0],
        }
    )
    artifact = fit_scaling_artifact(development, ["feature"], 0.0, 1.0)
    combined = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0, 1_000_000.0],
            "sample_split": ["development", "development", "development", "oos"],
        }
    )

    transformed = apply_scaling_artifact(combined, artifact)

    assert artifact.columns["feature"]["winsor_upper"] == pytest.approx(3.0)
    assert transformed.loc[3, scaled_column("feature")] == pytest.approx(
        transformed.loc[2, scaled_column("feature")]
    )


def test_leave_one_out_cross_sectional_mean_excludes_own_asset() -> None:
    frame = pd.DataFrame(
        {
            "bucket_start": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-01"], utc=True
            ),
            "value": [1.0, 2.0, np.nan],
        }
    )

    result = leave_one_out_cross_sectional_mean(frame, "value")

    assert result.iloc[0] == pytest.approx(2.0)
    assert result.iloc[1] == pytest.approx(1.0)
    assert result.iloc[2] == pytest.approx(1.5)


def test_predeclared_decision_requires_q_sign_and_effect_size() -> None:
    coefficients = pd.DataFrame(
        [
            {
                "split": "oos",
                "family": "construct_aggressor",
                "feature": "run_intensity_up",
                "coefficient": 0.03,
                "q_value_bh_fdr": 0.01,
                "expected_sign": "positive",
                "ci_lower": 0.01,
                "ci_upper": 0.05,
            },
            {
                "split": "oos",
                "family": "construct_aggressor",
                "feature": "run_intensity_down",
                "coefficient": 0.03,
                "q_value_bh_fdr": 0.01,
                "expected_sign": "negative",
                "ci_lower": 0.01,
                "ci_upper": 0.05,
            },
            {
                "split": "oos",
                "family": "future_excess_return",
                "feature": "run_intensity_zero",
                "coefficient": 0.0006,
                "q_value_bh_fdr": 0.04,
                "expected_sign": "none",
                "ci_lower": 0.0001,
                "ci_upper": 0.0011,
            },
            {
                "split": "development",
                "family": "future_excess_return",
                "feature": "run_intensity_zero",
                "coefficient": 0.0006,
                "q_value_bh_fdr": 0.04,
                "expected_sign": "none",
                "ci_lower": 0.0001,
                "ci_upper": 0.0011,
            },
        ]
    )
    analysis = {
        "construct_minimum_effect": 0.02,
        "economic_effect_threshold_bps": 5.0,
        "fdr_alpha": 0.05,
    }

    result = _attach_predeclared_decisions(coefficients, analysis)

    assert result["passes_split_criteria"].tolist() == [True, False, True, True]
    assert result["is_primary_oos_pass"].tolist() == [True, False, True, False]

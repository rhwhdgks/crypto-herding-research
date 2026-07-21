from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zero_run_microstructure import (
    FIXED_EFFECTS,
    FUTURE_CONTROLS,
    ScalingArtifact,
    apply_scaling_artifact,
    assign_sample_split,
    build_artifact_manifest,
    build_decision_tables,
    build_symbol_forward_outcomes,
    fit_scaling_artifact,
    outcome_within_split,
    scaled_column,
    validate_frozen_config,
)


SPLIT = {
    "development_start": "2024-04-08T00:00:00Z",
    "oos_start": "2025-04-08T00:00:00Z",
    "oos_end_exclusive": "2026-04-08T00:00:00Z",
}


def test_split_and_outcome_availability_use_strict_half_open_boundaries() -> None:
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2024-04-08T00:00:00Z",
                "2025-04-07T23:45:00Z",
                "2025-04-08T00:00:00Z",
                "2026-04-07T23:45:00Z",
            ],
            utc=True,
        )
    )
    split = assign_sample_split(timestamps, SPLIT)
    available_at = timestamps + pd.Timedelta(minutes=15)

    valid = outcome_within_split(split, available_at, SPLIT)

    assert split.tolist() == ["development", "development", "oos", "oos"]
    assert valid.tolist() == [True, False, True, False]


def test_forward_outcomes_start_at_last_known_minute_close(tmp_path: Path) -> None:
    timestamps = pd.date_range("2024-04-08T00:00:00Z", periods=45, freq="1min")
    log_prices = np.arange(len(timestamps), dtype=float) * 0.001
    ohlcv = pd.DataFrame(
        {"timestamp": timestamps, "close": 100.0 * np.exp(log_prices)}
    )
    path = tmp_path / "TEST_1m.parquet"
    ohlcv.to_parquet(path, index=False)
    tick = pd.DataFrame(
        {
            "symbol": ["TEST"],
            "bucket_start": pd.to_datetime(["2024-04-08T00:00:00Z"], utc=True),
            "signal_timestamp": pd.to_datetime(["2024-04-08T00:15:00Z"], utc=True),
            "last_price": [ohlcv.loc[14, "close"]],
            "sample_split": ["development"],
        }
    )

    result, quality = build_symbol_forward_outcomes(tick, path, [5, 15, 30], SPLIT)

    assert result.loc[0, "abs_forward_return_5m_bps"] == pytest.approx(50.0)
    assert result.loc[0, "abs_forward_return_15m_bps"] == pytest.approx(150.0)
    assert result.loc[0, "abs_forward_return_30m_bps"] == pytest.approx(300.0)
    assert result.loc[0, "realized_volatility_5m_bps"] == pytest.approx(
        np.sqrt(5) * 10.0
    )
    assert bool(result.loc[0, "horizon_is_exact_30m"])
    assert quality["maximum_tick_vs_1m_close_difference_bps"] == pytest.approx(0.0)


def test_scaler_is_fit_on_development_only() -> None:
    columns = sorted(
        {
            "zero_run_intensity",
            "log_amihud_illiquidity",
            "zero_tick_share",
            "log_mean_intertrade_ms",
            "log_quote_volume",
            "abs_aggressor_imbalance",
            "abs_bucket_return",
            "current_market_abs_return_loo",
            "trailing_volatility",
            "log_transaction_count",
        }
    )
    frame = pd.DataFrame(
        {
            "bucket_start": pd.to_datetime(
                ["2024-04-08", "2024-04-09", "2024-04-10", "2025-04-08"],
                utc=True,
            ),
            "sample_split": ["development"] * 3 + ["oos"],
            "meets_minimum_trade_count": [True] * 4,
        }
    )
    for index, column in enumerate(columns):
        frame[column] = np.asarray([1.0, 2.0, 3.0, 1_000_000.0]) + index
    config = {
        "analysis": {"winsor_lower_quantile": 0.0, "winsor_upper_quantile": 1.0}
    }

    artifact = fit_scaling_artifact(frame, config)
    transformed = apply_scaling_artifact(frame, artifact)

    development_max = frame.loc[
        frame["sample_split"].eq("development"), "zero_run_intensity"
    ].max()
    assert artifact.columns["zero_run_intensity"]["winsor_upper"] == pytest.approx(
        development_max
    )
    assert transformed.loc[3, scaled_column("zero_run_intensity")] == pytest.approx(
        transformed.loc[2, scaled_column("zero_run_intensity")]
    )


def test_future_controls_contain_no_future_outcomes() -> None:
    assert all("forward" not in value and "future" not in value for value in FUTURE_CONTROLS)
    assert FIXED_EFFECTS == ["symbol", "hour_utc", "weekday_utc"]


def test_decision_requires_all_six_gates_and_family_replication() -> None:
    coefficient_rows = []
    for family, prefix in (
        ("future_absolute_return", "abs_forward_return"),
        ("future_realized_volatility", "realized_volatility"),
    ):
        for horizon in [5, 15, 30]:
            coefficient_rows.append(
                {
                    "split": "oos",
                    "family": family,
                    "outcome": f"{prefix}_{horizon}m_bps",
                    "horizon_minutes": horizon,
                    "coefficient": 2.0,
                    "outcome_mean": 20.0,
                    "q_value_bh_fdr": 0.01,
                }
            )
    coefficient_rows.append(
        {
            "split": "oos",
            "family": "mechanism",
            "outcome": "zero_tick_share",
            "horizon_minutes": np.nan,
            "coefficient": 0.10,
            "outcome_mean": 0.0,
            "q_value_bh_fdr": 0.01,
        }
    )
    coefficients = pd.DataFrame(coefficient_rows)
    future_keys = coefficients.loc[coefficients["family"].ne("mechanism"), [
        "family", "outcome", "horizon_minutes"
    ]]
    prediction = future_keys.assign(rmse_improvement_percent=0.30)
    loao = future_keys[["family", "horizon_minutes"]].assign(
        same_sign_count=7, median_magnitude_ratio=1.0
    )
    permutation = future_keys.assign(
        q_value_bh_fdr=0.01,
        empirical_p_value=0.005,
    )
    placebo = future_keys.assign(coefficient=-0.1, q_value_bh_fdr=1.0)
    config = {
        "analysis": {
            "fdr_alpha": 0.05,
            "mechanism_minimum_standardized_effect": 0.05,
            "future_minimum_effect_bps": 1.0,
            "future_minimum_relative_effect": 0.05,
            "loao_minimum_same_sign_count": 6,
            "loao_minimum_median_magnitude_ratio": 0.5,
            "oos_rmse_improvement_percent": 0.25,
            "primary_horizon_minutes": 15,
        }
    }

    _, future, families = build_decision_tables(
        coefficients, prediction, loao, permutation, placebo, config
    )

    assert future["passes_predeclared_criteria"].all()
    assert families.loc[
        families["family"].isin(
            ["future_absolute_return", "future_realized_volatility"]
        ),
        "family_success",
    ].all()

    prediction.loc[prediction["horizon_minutes"].eq(15), "rmse_improvement_percent"] = 0.0
    _, future_failed, families_failed = build_decision_tables(
        coefficients, prediction, loao, permutation, placebo, config
    )
    assert not future_failed.loc[
        future_failed["horizon_minutes"].eq(15), "passes_predeclared_criteria"
    ].any()
    assert not families_failed.loc[
        families_failed["family"].isin(
            ["future_absolute_return", "future_realized_volatility"]
        ),
        "family_success",
    ].any()


def test_frozen_config_rejects_horizon_drift() -> None:
    config = {
        "data": {
            "symbols": [
                "BTCUSDT",
                "ETHUSDT",
                "XRPUSDT",
                "SOLUSDT",
                "DOGEUSDT",
                "ADAUSDT",
                "AVAXUSDT",
            ],
            "expected_rows": 490_560,
        },
        "analysis": {
            "horizons_minutes": [5, 15, 60],
            "primary_horizon_minutes": 15,
            "mechanism_family_size": 5,
            "future_family_size": 3,
            "permutation_repetitions": 199,
        },
        "output": {"base_dir": "outputs/v2/zero_run_microstructure_v1"},
    }
    with pytest.raises(ValueError, match="horizons"):
        validate_frozen_config(config)


def test_artifact_manifest_excludes_itself(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "artifact_manifest.csv").write_text("old")
    manifest = build_artifact_manifest(tmp_path)
    assert manifest["path"].tolist() == ["a.txt"]

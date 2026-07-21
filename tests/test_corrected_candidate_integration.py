from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from corrected_candidate import run_corrected_candidate_validation


def test_corrected_candidate_runner_writes_oos_artifacts(tmp_path: Path) -> None:
    timestamps = pd.date_range("2025-01-01", periods=32, freq="15min", tz="UTC")
    rows = []
    for symbol in ["DOGEUSDT", "ADAUSDT", "AVAXUSDT"]:
        for index, start in enumerate(timestamps):
            rows.append(
                {
                    "symbol": symbol,
                    "bucket_start": start,
                    "bucket_end": start + pd.Timedelta(minutes=15),
                    "signal_timestamp": start + pd.Timedelta(minutes=15),
                    "schema_version": 2,
                    "hour_utc": start.hour,
                    "is_micro_run_clustering_event": symbol == "DOGEUSDT" and index % 2 == 0,
                    "run_clustering_side": "down",
                    "price_direction": "down" if index % 2 == 0 else "up",
                    "aggressor_direction": "sell",
                    "forward_return_30m": 0.01 if index % 2 == 0 else -0.002,
                }
            )
    micro_path = tmp_path / "micro.csv"
    pd.DataFrame(rows).to_csv(micro_path, index=False)
    state_path = tmp_path / "state.csv"
    pd.DataFrame(
        {
            "bucket_start": timestamps,
            "funding_pre": 0.0002,
            "d_oi_event": [index / 1000 for index in range(len(timestamps))],
        }
    ).to_csv(state_path, index=False)
    config = {
        "input": {
            "micro_frame_path": str(micro_path),
            "futures_state_path": str(state_path),
            "state_columns": ["funding_pre", "d_oi_event"],
        },
        "sample_split": {
            "train_start": "2025-01-01T00:00:00Z",
            "train_end": "2025-01-01T03:45:00Z",
            "oos_start": "2025-01-01T04:00:00Z",
            "oos_end": "2025-01-01T07:45:00Z",
        },
        "threshold_fit": {"column": "d_oi_event", "quantile": 1 / 3},
        "candidate_family": [
            {
                "leader": "DOGEUSDT",
                "targets": ["ADAUSDT", "AVAXUSDT"],
                "price_direction": "down",
                "run_clustering_side": "down",
                "horizon_minutes": 30,
                "session_hours_utc": list(range(24)),
                "funding_threshold": 0.0001,
                "oi_rule": "above_fitted_threshold",
            }
        ],
        "permutation": {"n_draws": 19, "seed": 7, "statistic": "mean_return"},
        "execution": {
            "mode": "taker",
            "overlap_policy": "skip_while_position_open",
            "round_trip_fee": 0.001,
            "slippage": 0.0002,
            "funding": 0.0,
            "latency_minutes": 0.1,
        },
    }
    output = tmp_path / "out"
    result = run_corrected_candidate_validation(config, output)
    assert result["schema_version"] == 2
    assert (output / "threshold_artifact.json").exists()
    assert (output / "selection_aware_permutation.json").exists()
    assert (output / "execution_trade_log.csv").exists()
    artifact = json.loads((output / "threshold_artifact.json").read_text())
    assert artifact["fit_end"] < config["sample_split"]["oos_start"]

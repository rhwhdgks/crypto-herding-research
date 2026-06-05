from __future__ import annotations

from math import sqrt
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from utils import horizon_to_label


def run_event_study(
    analysis_frame: pd.DataFrame,
    market_return: pd.Series,
    holding_periods: Iterable[int],
    event_label_column: str = "event_type",
    event_types: list[str] | None = None,
    max_path_horizon: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holding_periods = [int(period) for period in holding_periods]
    forward_returns = compute_forward_returns(market_return, holding_periods)

    enriched = analysis_frame.join(forward_returns, how="left")
    if event_label_column not in enriched.columns:
        raise ValueError(f"Event label column not found: {event_label_column}")

    event_labels = enriched[event_label_column].fillna("none")
    if event_types is None:
        event_types = [label for label in event_labels.unique().tolist() if label != "none"]

    events = enriched[event_labels != "none"].copy()

    results = []
    for event_type in event_types:
        subset = events[events[event_label_column] == event_type]
        for horizon in holding_periods:
            column = f"forward_return_{horizon}m"
            sample = subset[column].dropna()
            results.append(
                {
                    event_label_column: event_type,
                    "horizon_minutes": horizon,
                    "horizon_label": horizon_to_label(horizon),
                    "count": int(sample.shape[0]),
                    "mean_return": sample.mean() if not sample.empty else np.nan,
                    "median_return": sample.median() if not sample.empty else np.nan,
                    "std_return": sample.std(ddof=1) if sample.shape[0] >= 2 else np.nan,
                    "t_stat": _compute_t_stat(sample),
                    "win_rate": (sample > 0).mean() if not sample.empty else np.nan,
                }
            )

    summary = pd.DataFrame(results)
    comparison = build_holding_period_comparison(summary, event_label_column)
    path_frame = compute_event_time_average_paths(
        events=events,
        market_return=market_return,
        event_label_column=event_label_column,
        event_types=event_types,
        max_horizon=int(max_path_horizon or max(holding_periods)),
    )
    return events, summary, comparison, path_frame


def compute_forward_returns(market_return: pd.Series, holding_periods: Iterable[int]) -> pd.DataFrame:
    values = market_return.to_numpy(dtype=float)
    n_obs = len(values)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))

    data = {}
    for horizon in holding_periods:
        horizon = int(horizon)
        forward_log_return = np.full(n_obs, np.nan)

        valid_positions = np.arange(n_obs)
        end_positions = valid_positions + 1 + horizon
        valid_mask = end_positions <= n_obs

        valid_indices = valid_positions[valid_mask]
        valid_end_positions = end_positions[valid_mask]
        forward_log_return[valid_indices] = cumulative[valid_end_positions] - cumulative[valid_indices + 1]

        data[f"forward_return_{horizon}m"] = np.expm1(forward_log_return)

    return pd.DataFrame(data, index=market_return.index)


def build_holding_period_comparison(summary: pd.DataFrame, event_label_column: str) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    comparison = summary.pivot(index="horizon_minutes", columns=event_label_column, values=["mean_return", "median_return", "std_return", "t_stat", "win_rate", "count"])
    comparison.columns = [f"{metric}__{label}" for metric, label in comparison.columns]
    comparison = comparison.reset_index()
    comparison["horizon_label"] = comparison["horizon_minutes"].apply(horizon_to_label)
    return comparison


def compute_event_time_average_paths(
    events: pd.DataFrame,
    market_return: pd.Series,
    event_label_column: str,
    event_types: list[str],
    max_horizon: int,
) -> pd.DataFrame:
    if events.empty or max_horizon <= 0:
        return pd.DataFrame()

    values = market_return.to_numpy(dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    event_positions = market_return.index.get_indexer(events.index)
    path_records = []

    for event_type in event_types:
        subset_positions = event_positions[events[event_label_column] == event_type]
        subset_positions = subset_positions[subset_positions >= 0]
        if subset_positions.size == 0:
            continue

        for step in range(1, max_horizon + 1):
            valid_positions = subset_positions[subset_positions + 1 + step <= len(values)]
            if valid_positions.size == 0:
                break

            forward_log = cumulative[valid_positions + 1 + step] - cumulative[valid_positions + 1]
            path_records.append(
                {
                    event_label_column: event_type,
                    "event_time_minutes": step,
                    "mean_cumulative_return": float(np.expm1(forward_log).mean()),
                    "contributing_event_count": int(valid_positions.size),
                }
            )

    return pd.DataFrame(path_records)


def _compute_t_stat(sample: pd.Series) -> float:
    if sample.shape[0] < 2:
        return np.nan

    std = sample.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan

    return float(sample.mean() / (std / sqrt(sample.shape[0])))

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from utils import timeframe_to_pandas_freq


def build_price_and_return_panels(
    asset_frames: Dict[str, pd.DataFrame],
    data_cfg: dict,
    panel_cfg: dict | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel_cfg = panel_cfg or {}
    panel_mode = str(panel_cfg.get("mode", "intersection")).lower()
    total_assets = max(len(asset_frames), 1)
    default_min_assets = total_assets if panel_mode == "intersection" else min(2, total_assets)
    min_active_assets = int(panel_cfg.get("min_active_assets", default_min_assets))

    expected_index = build_expected_timestamp_index(
        start=data_cfg["start"],
        end=data_cfg["end"],
        timeframe=data_cfg.get("timeframe", "1m"),
    )
    raw_close_prices = reindex_close_prices(asset_frames, expected_index)
    aligned_close_prices = apply_missing_timestamp_policy(
        raw_close_prices,
        policy=data_cfg.get("missing_timestamp_policy", "forward_fill_prices"),
        max_forward_fill_steps=data_cfg.get("max_forward_fill_steps"),
        panel_mode=panel_mode,
    )
    log_returns = compute_log_returns(aligned_close_prices, panel_mode=panel_mode)
    data_quality_summary = summarize_data_quality(
        asset_frames=asset_frames,
        raw_close_prices=raw_close_prices,
        aligned_close_prices=aligned_close_prices,
        log_returns=log_returns,
        policy=data_cfg.get("missing_timestamp_policy", "forward_fill_prices"),
        max_forward_fill_steps=data_cfg.get("max_forward_fill_steps"),
        panel_mode=panel_mode,
        min_active_assets=min_active_assets,
    )
    universe_coverage_summary, universe_transition_points = summarize_universe_coverage(
        raw_close_prices=raw_close_prices,
        aligned_close_prices=aligned_close_prices,
        log_returns=log_returns,
        panel_mode=panel_mode,
        min_active_assets=min_active_assets,
    )
    return (
        raw_close_prices,
        aligned_close_prices,
        log_returns,
        data_quality_summary,
        universe_coverage_summary,
        universe_transition_points,
    )


def build_expected_timestamp_index(start: str, end: str, timeframe: str) -> pd.DatetimeIndex:
    return pd.date_range(
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
        freq=timeframe_to_pandas_freq(timeframe),
        inclusive="left",
        name="timestamp",
    )


def reindex_close_prices(
    asset_frames: Dict[str, pd.DataFrame],
    expected_index: pd.DatetimeIndex,
    price_column: str = "close",
) -> pd.DataFrame:
    if not asset_frames:
        raise ValueError("asset_frames is empty")

    close_prices = pd.concat(
        {
            symbol: frame[price_column].reindex(expected_index)
            for symbol, frame in asset_frames.items()
        },
        axis=1,
    ).sort_index()
    close_prices.index.name = "timestamp"
    return close_prices


def apply_missing_timestamp_policy(
    raw_close_prices: pd.DataFrame,
    policy: str = "forward_fill_prices",
    max_forward_fill_steps: int | None = None,
    panel_mode: str = "intersection",
) -> pd.DataFrame:
    # Forward-filling is safe here because it uses only past prices and never future observations.
    if policy == "forward_fill_prices":
        aligned = raw_close_prices.ffill(limit=max_forward_fill_steps)
    elif policy == "drop_incomplete_rows":
        aligned = raw_close_prices.copy()
    elif policy == "none":
        aligned = raw_close_prices.copy()
    else:
        raise ValueError(f"Unsupported missing timestamp policy: {policy}")

    if panel_mode == "intersection":
        aligned = aligned.dropna(how="any")
    elif panel_mode == "expanding_universe":
        aligned = aligned.dropna(how="all")
    else:
        raise ValueError(f"Unsupported panel mode: {panel_mode}")

    aligned.index.name = "timestamp"
    return aligned


def compute_log_returns(price_frame: pd.DataFrame, panel_mode: str = "intersection") -> pd.DataFrame:
    if price_frame.empty:
        return price_frame.copy()

    if (price_frame <= 0).any().any():
        raise ValueError("Prices must be strictly positive to compute log returns")

    log_returns = np.log(price_frame).diff()
    if panel_mode == "intersection":
        log_returns = log_returns.dropna(how="any")
    elif panel_mode == "expanding_universe":
        log_returns = log_returns.dropna(how="all")
    else:
        raise ValueError(f"Unsupported panel mode: {panel_mode}")
    log_returns.index.name = "timestamp"
    return log_returns


def summarize_data_quality(
    asset_frames: Dict[str, pd.DataFrame],
    raw_close_prices: pd.DataFrame,
    aligned_close_prices: pd.DataFrame,
    log_returns: pd.DataFrame,
    policy: str,
    max_forward_fill_steps: int | None,
    panel_mode: str,
    min_active_assets: int,
) -> pd.DataFrame:
    expected_rows = len(raw_close_prices)
    records = []

    for symbol, frame in asset_frames.items():
        raw_series = raw_close_prices[symbol]
        missing_before_fill = int(raw_series.isna().sum())
        aligned_missing = int(aligned_close_prices[symbol].isna().sum()) if symbol in aligned_close_prices.columns else expected_rows

        records.append(
            {
                "symbol": symbol,
                "raw_rows_loaded": int(len(frame)),
                "expected_rows": expected_rows,
                "missing_before_policy": missing_before_fill,
                "coverage_before_policy": 1.0 - (missing_before_fill / expected_rows if expected_rows else 0.0),
                "missing_after_policy": aligned_missing,
                "coverage_after_policy": 1.0 - (aligned_missing / max(len(aligned_close_prices), 1)),
                "first_observation": frame.index.min(),
                "last_observation": frame.index.max(),
                "missing_timestamp_policy": policy,
                "max_forward_fill_steps": max_forward_fill_steps,
                "panel_mode": panel_mode,
                "min_active_assets": min_active_assets,
                "final_aligned_price_rows": int(len(aligned_close_prices)),
                "final_return_rows": int(len(log_returns)),
            }
        )

    records.append(
        {
            "symbol": "__panel__",
            "raw_rows_loaded": sum(len(frame) for frame in asset_frames.values()),
            "expected_rows": expected_rows,
            "missing_before_policy": int(raw_close_prices.isna().sum().sum()),
            "coverage_before_policy": float(raw_close_prices.notna().mean().mean()),
            "missing_after_policy": int(aligned_close_prices.isna().sum().sum()),
            "coverage_after_policy": float(aligned_close_prices.notna().mean().mean()) if not aligned_close_prices.empty else 0.0,
            "first_observation": raw_close_prices.index.min(),
            "last_observation": raw_close_prices.index.max(),
            "missing_timestamp_policy": policy,
            "max_forward_fill_steps": max_forward_fill_steps,
            "panel_mode": panel_mode,
            "min_active_assets": min_active_assets,
            "final_aligned_price_rows": int(len(aligned_close_prices)),
            "final_return_rows": int(len(log_returns)),
        }
    )

    return pd.DataFrame(records)


def summarize_universe_coverage(
    raw_close_prices: pd.DataFrame,
    aligned_close_prices: pd.DataFrame,
    log_returns: pd.DataFrame,
    panel_mode: str,
    min_active_assets: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total_assets = int(raw_close_prices.shape[1])
    price_counts = aligned_close_prices.notna().sum(axis=1) if not aligned_close_prices.empty else pd.Series(dtype=float)
    return_counts = log_returns.notna().sum(axis=1) if not log_returns.empty else pd.Series(dtype=float)

    candidate_rows = return_counts[return_counts >= min_active_assets]
    analysis_start = candidate_rows.index.min() if not candidate_rows.empty else pd.NaT
    analysis_end = candidate_rows.index.max() if not candidate_rows.empty else pd.NaT

    summary = pd.DataFrame(
        [
            {
                "panel_mode": panel_mode,
                "min_active_assets": int(min_active_assets),
                "total_assets": total_assets,
                "requested_start": raw_close_prices.index.min(),
                "requested_end": raw_close_prices.index.max(),
                "aligned_price_start": aligned_close_prices.index.min() if not aligned_close_prices.empty else pd.NaT,
                "aligned_price_end": aligned_close_prices.index.max() if not aligned_close_prices.empty else pd.NaT,
                "return_start": log_returns.index.min() if not log_returns.empty else pd.NaT,
                "return_end": log_returns.index.max() if not log_returns.empty else pd.NaT,
                "analysis_candidate_start": analysis_start,
                "analysis_candidate_end": analysis_end,
                "max_active_price_assets": int(price_counts.max()) if not price_counts.empty else 0,
                "max_active_return_assets": int(return_counts.max()) if not return_counts.empty else 0,
                "median_active_return_assets": float(return_counts.median()) if not return_counts.empty else 0.0,
                "rows_meeting_min_active_assets": int(candidate_rows.shape[0]),
                "share_rows_meeting_min_active_assets": float((return_counts >= min_active_assets).mean()) if not return_counts.empty else 0.0,
            }
        ]
    )

    transition_frame = pd.DataFrame(index=log_returns.index)
    transition_frame["available_price_assets"] = aligned_close_prices.notna().sum(axis=1).reindex(log_returns.index).fillna(0).astype(int)
    transition_frame["available_return_assets"] = return_counts.astype(int) if not return_counts.empty else 0
    transition_frame["available_return_share"] = (
        transition_frame["available_return_assets"] / total_assets if total_assets else 0.0
    )
    transition_frame["meets_min_active_assets"] = transition_frame["available_return_assets"] >= int(min_active_assets)
    transition_frame["panel_mode"] = panel_mode

    if transition_frame.empty:
        return summary, transition_frame.reset_index()

    change_mask = transition_frame["available_price_assets"].ne(transition_frame["available_price_assets"].shift())
    change_mask |= transition_frame["available_return_assets"].ne(transition_frame["available_return_assets"].shift())
    transition_frame = transition_frame.loc[change_mask].reset_index()
    return summary, transition_frame

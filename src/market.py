from __future__ import annotations

import numpy as np
import pandas as pd


def compute_active_asset_count(return_frame: pd.DataFrame) -> pd.Series:
    active_asset_count = return_frame.notna().sum(axis=1)
    active_asset_count.name = "active_asset_count"
    return active_asset_count


def compute_equal_weighted_market_return(
    return_frame: pd.DataFrame,
    min_active_assets: int = 1,
) -> pd.Series:
    active_asset_count = compute_active_asset_count(return_frame)
    market_return = return_frame.mean(axis=1, skipna=True)
    market_return = market_return.where(active_asset_count >= int(min_active_assets))
    market_return.name = "market_return"
    return market_return


def compute_market_cap_weighted_market_return(
    return_frame: pd.DataFrame,
    market_cap_frame: pd.DataFrame,
    min_active_assets: int = 1,
    weight_lag_periods: int = 0,
) -> pd.Series:
    aligned_returns, aligned_caps = return_frame.align(market_cap_frame, join="left", axis=0)
    aligned_caps = aligned_caps.reindex(columns=aligned_returns.columns)
    aligned_caps = aligned_caps.where(aligned_caps > 0)

    if int(weight_lag_periods) != 0:
        aligned_caps = aligned_caps.shift(int(weight_lag_periods))

    valid_mask = aligned_returns.notna() & aligned_caps.notna()
    weighted_returns = aligned_returns.where(valid_mask) * aligned_caps.where(valid_mask)
    weight_sums = aligned_caps.where(valid_mask).sum(axis=1, skipna=True)
    active_asset_count = valid_mask.sum(axis=1)

    market_return = weighted_returns.sum(axis=1, skipna=True) / weight_sums
    market_return = market_return.where((active_asset_count >= int(min_active_assets)) & (weight_sums > 0))
    market_return.name = "market_return"
    return market_return


def compute_market_index(market_return: pd.Series, start_value: float = 100.0) -> pd.Series:
    valid_market_return = market_return.dropna()
    market_index = pd.Series(index=market_return.index, dtype=float, name="market_index")
    if valid_market_return.empty:
        return market_index
    market_index.loc[valid_market_return.index] = start_value * np.exp(valid_market_return.cumsum())
    market_index.name = "market_index"
    return market_index

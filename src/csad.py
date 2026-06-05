from __future__ import annotations

import pandas as pd


def compute_csad(
    return_frame: pd.DataFrame,
    market_return: pd.Series,
    min_active_assets: int = 1,
) -> pd.Series:
    aligned_returns = return_frame.reindex(market_return.index)
    active_asset_count = aligned_returns.notna().sum(axis=1)
    csad = aligned_returns.sub(market_return, axis=0).abs().mean(axis=1, skipna=True)
    csad = csad.where(active_asset_count >= int(min_active_assets))
    csad.name = "csad"
    return csad

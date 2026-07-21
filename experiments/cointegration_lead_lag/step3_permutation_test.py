"""Permutation test — filter PnL의 통계적 유의성 결판.

가설: lead-lag event filter가 random보다 더 좋은 trade timing을 잡는가?
방법: DOGE event timestamp들을 1000번 random circular shift → 매번 backtest → null 분포 생성
검증: 실제 결과 PnL이 null 분포의 어느 percentile인지 (p-value)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Reuse step2_15m_backtest helpers
from step2_15m_backtest import (  # type: ignore
    COIN_A,
    COIN_B,
    DATA_DIR,
    END,
    LAG_BARS_SWEEP,
    MICRO_FRAME_PATH,
    OUT_DIR,
    SIG_FLOOR_Q,
    START,
    Z_ENTRY,
    Z_EXIT,
    backtest_one_window,
    max_drawdown,
)

N_PERMUTATIONS = 1000
SLIPPAGE_BPS = 5  # focus on 5bps for permutation (most realistic)
LAG_BARS_FOCUS = 2  # 30min — best from sweep
RANDOM_SEED = 42


def build_event_mask_from_indices(
    bar_indices: np.ndarray,
    n_bars: int,
    lag_bars: int,
    base_index: pd.DatetimeIndex,
) -> pd.Series:
    """벡터화 mask 생성 — index 정수 배열로부터 lag-window flag."""
    arr = np.zeros(n_bars, dtype=bool)
    for offset in range(lag_bars):
        clipped = np.clip(bar_indices + offset, 0, n_bars - 1)
        arr[clipped] = True
    return pd.Series(arr, index=base_index)


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    # Load data once
    a_df = pd.read_parquet(DATA_DIR / f"{COIN_A}_1m.parquet")[["timestamp", "close"]]
    b_df = pd.read_parquet(DATA_DIR / f"{COIN_B}_1m.parquet")[["timestamp", "close"]]
    a_df["timestamp"] = pd.to_datetime(a_df["timestamp"], utc=True)
    b_df["timestamp"] = pd.to_datetime(b_df["timestamp"], utc=True)
    df_1m = a_df.merge(b_df, on="timestamp", how="inner", suffixes=("_a", "_b"))
    df_1m = df_1m.rename(columns={"close_a": COIN_A, "close_b": COIN_B}).set_index("timestamp").sort_index()
    df_1m = df_1m.loc[pd.Timestamp(START, tz="UTC"):pd.Timestamp(END, tz="UTC")]
    df = df_1m.resample("15min").last().dropna()
    log_prices = np.log(df.astype(float))
    n_bars = len(log_prices)

    pairs = pd.read_csv(OUT_DIR / f"cointegration_15m_{COIN_A}_{COIN_B}.csv")
    pairs["start_ts"] = pd.to_datetime(pairs["start_ts"], utc=True)
    pairs = pairs.loc[pairs["passes_filter"]].copy()

    # Real DOGE down event indices in 15min frame
    mf = pd.read_csv(MICRO_FRAME_PATH)
    mf["bucket_start"] = pd.to_datetime(mf["bucket_start"], utc=True)
    real_events = pd.to_datetime(
        mf.loc[
            (mf["symbol"] == "DOGEUSDT")
            & mf["is_micro_run_clustering_event"].fillna(False).astype(bool)
            & mf["run_clustering_side"].eq("down")
            & mf["price_direction"].eq("down"),
            "bucket_start",
        ].values,
        utc=True,
    )
    # Map events to integer positions in our 15min frame
    bar_position = {ts: i for i, ts in enumerate(log_prices.index)}
    real_indices = np.array([bar_position[ts] for ts in real_events if ts in bar_position])
    print(f"real DOGE down events in frame: {len(real_indices)} / {len(real_events)} aligned")

    def run_backtest(indices: np.ndarray, lag_bars: int) -> dict:
        mask = build_event_mask_from_indices(indices, n_bars, lag_bars, log_prices.index)
        rows: list[dict] = []
        for _, row in pairs.iterrows():
            out = backtest_one_window(row.to_dict(), log_prices, SLIPPAGE_BPS, mask)
            if out is None:
                continue
            rows.append(out)
        if not rows:
            return {"cum_pnl": 0.0, "n_entries": 0, "sharpe": float("nan"), "mdd": float("nan")}
        sub = pd.DataFrame(rows)
        daily = sub.groupby("window_day")["pnl"].sum().sort_index()
        equity = daily.cumsum()
        mdd = max_drawdown(equity)
        mu = daily.mean()
        sd = daily.std(ddof=1)
        sharpe = float(np.sqrt(365) * mu / sd) if sd > 0 else float("nan")
        return {
            "cum_pnl": float(daily.sum()),
            "n_entries": int(sub["n_entries"].sum()),
            "sharpe": sharpe,
            "mdd": float(mdd),
        }

    # Real result
    real_result = run_backtest(real_indices, LAG_BARS_FOCUS)
    print(f"REAL (LAG={LAG_BARS_FOCUS}): cum_pnl={real_result['cum_pnl']:.6f}, "
          f"sharpe={real_result['sharpe']:.3f}, n_entries={real_result['n_entries']}")

    # Permutations: circular shift
    perm_results: list[dict] = []
    for i in range(N_PERMUTATIONS):
        shift = int(rng.integers(low=-n_bars + 1, high=n_bars))
        perm_indices = (real_indices + shift) % n_bars
        result = run_backtest(perm_indices, LAG_BARS_FOCUS)
        result["perm_id"] = i
        perm_results.append(result)
        if (i + 1) % 100 == 0:
            print(f"  permutation {i+1}/{N_PERMUTATIONS} done")

    perm_df = pd.DataFrame(perm_results)
    perm_df.to_csv(OUT_DIR / "permutation_results.csv", index=False)

    # Stats
    real_pnl = real_result["cum_pnl"]
    null_pnls = perm_df["cum_pnl"].dropna().values
    pct_above = float((null_pnls >= real_pnl).mean())
    pct_below = float((null_pnls <= real_pnl).mean())
    p_two_sided = 2.0 * min(pct_above, pct_below)

    null_sharpes = perm_df["sharpe"].dropna().values
    real_sharpe = real_result["sharpe"]
    sharpe_pct_above = float((null_sharpes >= real_sharpe).mean())

    summary = {
        "real_cum_pnl": real_pnl,
        "real_sharpe": real_sharpe,
        "real_n_entries": real_result["n_entries"],
        "n_permutations": int(len(perm_df)),
        "null_pnl_mean": float(null_pnls.mean()),
        "null_pnl_std": float(null_pnls.std(ddof=1)),
        "null_pnl_p5": float(np.percentile(null_pnls, 5)),
        "null_pnl_p95": float(np.percentile(null_pnls, 95)),
        "p_value_pnl_one_sided_above": pct_above,
        "p_value_pnl_two_sided": p_two_sided,
        "p_value_sharpe_one_sided_above": sharpe_pct_above,
    }
    pd.Series(summary).to_csv(OUT_DIR / "permutation_summary.csv")

    print()
    print("=== Permutation summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    print()
    if pct_above < 0.05:
        print("RESULT: Real PnL is in TOP 5% of null distribution → 신호 통계적 유의 (p<0.05)")
    elif pct_above < 0.20:
        print("RESULT: Real PnL is in top 5-20% of null distribution → 약한 신호 (marginal)")
    else:
        print("RESULT: Real PnL is within central 80% of null → 우연 가능성 매우 높음")


if __name__ == "__main__":
    raise SystemExit(main())

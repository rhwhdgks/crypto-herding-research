"""15분봉 AVAX|DOGE backtester — baseline vs lead-lag-filtered.

Sprint1 step10 robust 로직 (Z_ENTRY=3, Z_EXIT=0.5, sigma floor q20) + lead-lag event filter.
1분봉을 15min resample, 3-day cointegration window, 1-day forward trade.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "experiments/cointegration_lead_lag/outputs"
MICRO_FRAME_PATH = PROJECT_ROOT / "outputs/v2/tick/multi_asset_365d/lead_lag_matrix/intermediate/tick_micro_frame_15m.csv"

COIN_A = "AVAXUSDT"
COIN_B = "DOGEUSDT"
START = "2025-05-08"
END = "2026-04-09"

# 15min bar units
TRADE_BARS = 24 * 4  # 1-day trade window = 96 bars
Z_ENTRY = 3.0
Z_EXIT = 0.5
L_MIN_BARS, L_MAX_BARS = 4, 80  # ~1h to ~20h (15min)
MIN_TRADE_OBS = 40
FEE_BPS = 5.0
SLIPPAGE_BPS_LIST = [0, 5, 10]
LAG_BARS = 2  # legacy default (sweep으로 대체)
LAG_BARS_SWEEP = [1, 2, 4, 8, 16]  # 15min, 30min, 1h, 2h, 4h
SIG_FLOOR_Q = 0.2


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return float("nan")
    peak = equity.cummax()
    return float((equity - peak).min())


def load_event_mask_15m(coin: str, lag_bars: int, all_bars_index: pd.DatetimeIndex) -> pd.Series:
    """direct join: micro_frame의 bucket_start와 15min bar index가 같은 단위."""
    mf = pd.read_csv(MICRO_FRAME_PATH)
    mf["bucket_start"] = pd.to_datetime(mf["bucket_start"], utc=True)
    sym = coin if coin.endswith("USDT") else f"{coin}USDT"
    events = pd.to_datetime(
        mf.loc[
            (mf["symbol"] == sym)
            & mf["is_micro_run_clustering_event"].fillna(False).astype(bool)
            & mf["run_clustering_side"].eq("down")
            & mf["price_direction"].eq("down"),
            "bucket_start",
        ].values,
        utc=True,
    )
    base = pd.Series(False, index=all_bars_index)
    base.loc[base.index.isin(events)] = True
    # forward fill TRUE for `lag_bars` consecutive bars (event bar inclusive)
    if lag_bars <= 1:
        return base
    arr = base.values.copy()
    n = len(arr)
    for offset in range(1, lag_bars):
        # mark next `offset` bars as True if previous was a True event bar
        arr[offset:] |= base.values[:-offset] if offset < n else arr[offset:]
    return pd.Series(arr, index=all_bars_index)


def backtest_one_window(
    row: dict,
    log_prices: pd.DataFrame,
    slip_bps: float,
    event_mask: pd.Series | None,
) -> dict | None:
    beta = float(row["hedge_beta"])
    hl = float(row["half_life_bars"])
    if not np.isfinite(hl) or hl <= 0:
        return None

    train_start = int(row["start_idx"])
    train_end = int(row["end_idx"])
    trade_start = train_end
    trade_end = train_end + TRADE_BARS
    if trade_end > len(log_prices):
        return None

    train = log_prices.iloc[train_start:train_end][[COIN_A, COIN_B]].dropna()
    trade = log_prices.iloc[trade_start:trade_end][[COIN_A, COIN_B]].dropna()
    if len(trade) < MIN_TRADE_OBS:
        return None

    train_spread = train[COIN_A] - beta * train[COIN_B]
    trade_spread = trade[COIN_A] - beta * trade[COIN_B]
    L = clamp(int(round(2 * hl)), L_MIN_BARS, L_MAX_BARS)
    seed = train_spread.tail(L)
    spread_full = pd.concat([seed, trade_spread], axis=0)
    mu = spread_full.rolling(L).mean()
    sig = spread_full.rolling(L).std(ddof=1)
    z_full = (spread_full - mu) / sig
    z = z_full.loc[trade_spread.index].dropna()

    sig_trade = sig.loc[trade_spread.index]
    sig_floor = sig_trade.quantile(SIG_FLOOR_Q)
    valid_sigma = sig_trade > sig_floor
    z = z[valid_sigma.loc[z.index].fillna(False)]
    if len(z) < 10:
        return None

    trade_lp = trade.loc[z.index, [COIN_A, COIN_B]]
    rA = trade_lp[COIN_A].diff().fillna(0.0)
    rB = trade_lp[COIN_B].diff().fillna(0.0)
    cost_rate = (FEE_BPS + slip_bps) / 10000.0

    if event_mask is not None:
        local_mask = event_mask.reindex(z.index).fillna(False)
    else:
        local_mask = None

    pos, wA, wB = 0, 0.0, 0.0
    pnl, n_entries, n_filtered_out = 0.0, 0, 0

    for t in z.index:
        zt = float(z.loc[t])
        pnl += wA * float(rA.loc[t]) + wB * float(rB.loc[t])

        new_pos = pos
        if pos == 0:
            wants_short = zt > Z_ENTRY
            wants_long = zt < -Z_ENTRY
            if wants_short or wants_long:
                if local_mask is not None and not bool(local_mask.loc[t]):
                    n_filtered_out += 1
                else:
                    new_pos = -1 if wants_short else +1
        else:
            if (pos == +1 and zt >= -Z_EXIT) or (pos == -1 and zt <= Z_EXIT):
                new_pos = 0

        if new_pos != pos:
            new_wA = (1.0 if new_pos == +1 else (-1.0 if new_pos == -1 else 0.0))
            new_wB = (-beta * new_wA) if new_pos != 0 else 0.0
            turnover = abs(new_wA - wA) + abs(new_wB - wB)
            pnl -= cost_rate * turnover
            if pos == 0 and new_pos != 0:
                n_entries += 1
            pos, wA, wB = new_pos, new_wA, new_wB

    if pos != 0:
        turnover = abs(0.0 - wA) + abs(0.0 - wB)
        pnl -= cost_rate * turnover

    return {
        "window_day": int(row["window_day"]),
        "start_ts": row["start_ts"],
        "beta": beta,
        "half_life_bars": hl,
        "L": L,
        "slippage_bps": slip_bps,
        "pnl": pnl,
        "n_entries": n_entries,
        "n_filtered_out": n_filtered_out,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a_df = pd.read_parquet(DATA_DIR / f"{COIN_A}_1m.parquet")[["timestamp", "close"]]
    b_df = pd.read_parquet(DATA_DIR / f"{COIN_B}_1m.parquet")[["timestamp", "close"]]
    a_df["timestamp"] = pd.to_datetime(a_df["timestamp"], utc=True)
    b_df["timestamp"] = pd.to_datetime(b_df["timestamp"], utc=True)
    df_1m = a_df.merge(b_df, on="timestamp", how="inner", suffixes=("_a", "_b"))
    df_1m = df_1m.rename(columns={"close_a": COIN_A, "close_b": COIN_B}).set_index("timestamp").sort_index()
    df_1m = df_1m.loc[pd.Timestamp(START, tz="UTC"):pd.Timestamp(END, tz="UTC")]
    df = df_1m.resample("15min").last().dropna()
    log_prices = np.log(df.astype(float))

    pairs = pd.read_csv(OUT_DIR / f"cointegration_15m_{COIN_A}_{COIN_B}.csv")
    pairs["start_ts"] = pd.to_datetime(pairs["start_ts"], utc=True)
    pairs = pairs.loc[pairs["passes_filter"]].copy()
    print(f"using {len(pairs)} cointegrated windows (15min)")

    masks: list[tuple[str, pd.Series | None]] = [("baseline", None)]
    for lag_b in LAG_BARS_SWEEP:
        em = load_event_mask_15m("DOGEUSDT", lag_b, log_prices.index)
        masks.append((f"filtered_lag{lag_b}", em))
        print(f"LAG={lag_b} bars ({lag_b*15}min): {int(em.sum())} of {len(em)} bars flagged")

    all_results: list[dict] = []
    for mode_name, mask in masks:
        for _, row in pairs.iterrows():
            for slip in SLIPPAGE_BPS_LIST:
                out = backtest_one_window(row.to_dict(), log_prices, slip, mask)
                if out is None:
                    continue
                out["mode"] = mode_name
                all_results.append(out)

    res = pd.DataFrame(all_results)
    res.to_csv(OUT_DIR / "backtest_15m_all_rows.csv", index=False)

    summary_rows: list[dict] = []
    for (mode, slip), sub in res.groupby(["mode", "slippage_bps"]):
        daily = sub.groupby("window_day")["pnl"].sum().sort_index()
        equity = daily.cumsum()
        mdd = max_drawdown(equity)
        mu = daily.mean()
        sd = daily.std(ddof=1)
        sharpe = float(np.sqrt(365) * mu / sd) if sd > 0 else float("nan")
        summary_rows.append({
            "mode": mode,
            "slippage_bps": slip,
            "n_windows": int(len(sub)),
            "cum_pnl": float(daily.sum()),
            "avg_daily_pnl": float(mu),
            "sharpe": sharpe,
            "mdd": float(mdd),
            "total_entries": int(sub["n_entries"].sum()),
            "filtered_out": int(sub["n_filtered_out"].sum()),
        })

    summary = pd.DataFrame(summary_rows).sort_values(["mode", "slippage_bps"])
    summary.to_csv(OUT_DIR / "backtest_15m_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())

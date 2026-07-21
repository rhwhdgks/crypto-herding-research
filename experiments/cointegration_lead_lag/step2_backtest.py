"""AVAX|DOGE pair backtester — baseline vs lead-lag-filtered.

Sprint1 step9_backtesting.py의 z-score 진입/청산 로직을 그대로 유지.
Binance USDT 1분봉 + lead_lag_matrix의 micro_frame을 결합해서:
  - Baseline mode:   |z| > Z_ENTRY 진입, mean-reversion 청산
  - Filtered mode:   진입 조건에 "직전 LAG_MIN 분 이내 DOGE schema-v2 run-side-down and price-down event 발생" 추가

PnL은 두 다리 로그리턴 기반, 비용은 turnover * (fee+slip) per side.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "experiments/cointegration_lead_lag/outputs"
MICRO_FRAME_PATH = PROJECT_ROOT / "outputs/v2/tick/multi_asset_365d/lead_lag_matrix/intermediate/tick_micro_frame_15m.csv"

Z_ENTRY = 3.0  # Sprint1 step10 robust 버전 (churn 감소)
Z_EXIT = 0.5
L_MIN, L_MAX = 30, 500
MIN_TRADE_OBS = 600
FEE_BPS = 5.0
SLIPPAGE_BPS_LIST = [0, 5, 10]
LAG_MIN_DEFAULT = 30  # event 발생 후 30분 안에 진입하면 filter 통과
SIG_FLOOR_Q = 0.2  # spread sigma 하위 20%인 시점은 노이즈로 보고 진입 차단 (Sprint1 step10)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return float("nan")
    peak = equity.cummax()
    dd = equity - peak
    return float(dd.min())


def load_event_mask(coin: str, lag_minutes: int, all_minutes_index: pd.DatetimeIndex) -> pd.Series:
    """Return a boolean Series indexed by minute timestamps.

    True after an explicit run-side-down and price-down v2 event.
    """
    mf = pd.read_csv(MICRO_FRAME_PATH)
    mf["bucket_start"] = pd.to_datetime(mf["bucket_start"], utc=True)
    sym = coin if coin.endswith("USDT") else f"{coin}USDT"
    events = mf.loc[
        (mf["symbol"] == sym)
        & mf["is_micro_run_clustering_event"].fillna(False).astype(bool)
        & mf["run_clustering_side"].eq("down")
        & mf["price_direction"].eq("down"),
        "bucket_start",
    ]
    events = pd.to_datetime(events.values, utc=True)
    if len(events) == 0:
        return pd.Series(False, index=all_minutes_index)

    mask = pd.Series(False, index=all_minutes_index)
    idx_values = mask.index.values.astype("datetime64[ns]")
    lag = np.timedelta64(lag_minutes, "m")
    bool_array = np.zeros(len(idx_values), dtype=bool)
    for evt in events:
        evt64 = np.datetime64(evt.to_datetime64()) if hasattr(evt, "to_datetime64") else np.datetime64(evt)
        start_pos = np.searchsorted(idx_values, evt64, side="left")
        end_pos = np.searchsorted(idx_values, evt64 + lag, side="left")
        bool_array[start_pos:end_pos] = True
    mask[:] = bool_array
    return mask


def backtest_one_window(
    row: dict,
    log_prices: pd.DataFrame,
    coin_a: str,
    coin_b: str,
    slip_bps: float,
    event_mask: pd.Series | None,
) -> dict | None:
    beta = float(row["hedge_beta"])
    hl = float(row["half_life"])
    if not np.isfinite(hl) or hl <= 0:
        return None

    train_start = int(row["start_idx"])
    train_end = int(row["end_idx"])
    trade_start = train_end
    trade_end = train_end + 1440

    if trade_end > len(log_prices):
        return None

    train = log_prices.iloc[train_start:train_end][[coin_a, coin_b]].dropna()
    trade = log_prices.iloc[trade_start:trade_end][[coin_a, coin_b]].dropna()
    if len(trade) < MIN_TRADE_OBS:
        return None

    train_spread = train[coin_a] - beta * train[coin_b]
    trade_spread = trade[coin_a] - beta * trade[coin_b]

    L = clamp(int(round(2 * hl)), L_MIN, L_MAX)
    seed = train_spread.tail(L)
    spread_full = pd.concat([seed, trade_spread], axis=0)

    mu = spread_full.rolling(L).mean()
    sig = spread_full.rolling(L).std(ddof=1)
    z_full = (spread_full - mu) / sig
    z = z_full.loc[trade_spread.index].dropna()

    # Sigma floor: drop trade-window timestamps where sigma is in lowest q (Sprint1 step10)
    sig_trade = sig.loc[trade_spread.index]
    sig_floor = sig_trade.quantile(SIG_FLOOR_Q)
    valid_sigma = sig_trade > sig_floor
    z = z[valid_sigma.loc[z.index].fillna(False)]

    if len(z) < 50:
        return None

    trade_lp = trade.loc[z.index, [coin_a, coin_b]]
    if len(trade_lp) < 50:
        return None

    rA = trade_lp[coin_a].diff().fillna(0.0)
    rB = trade_lp[coin_b].diff().fillna(0.0)
    cost_rate = (FEE_BPS + slip_bps) / 10000.0

    pos = 0
    wA, wB = 0.0, 0.0
    pnl = 0.0
    n_entries = 0
    n_filtered_out = 0  # how many would-be entries were blocked by event filter

    if event_mask is not None:
        local_mask = event_mask.reindex(z.index).fillna(False)
    else:
        local_mask = None

    for t in z.index:
        zt = float(z.loc[t])
        pnl += wA * float(rA.loc[t]) + wB * float(rB.loc[t])

        new_pos = pos
        if pos == 0:
            wants_short = zt > Z_ENTRY
            wants_long = zt < -Z_ENTRY
            if wants_short or wants_long:
                if local_mask is not None and not bool(local_mask.loc[t]):
                    # Filter blocks this entry
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

    # EOD flat
    if pos != 0:
        turnover = abs(0.0 - wA) + abs(0.0 - wB)
        pnl -= cost_rate * turnover

    return {
        "window_day": int(row["window_day"]),
        "start_ts": row["start_ts"],
        "end_ts": row["end_ts"],
        "beta": beta,
        "half_life": hl,
        "L": L,
        "fee_bps": FEE_BPS,
        "slippage_bps": slip_bps,
        "pnl": pnl,
        "n_entries": n_entries,
        "n_filtered_out": n_filtered_out,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin-a", default="AVAXUSDT")
    parser.add_argument("--coin-b", default="DOGEUSDT")
    parser.add_argument("--filter-coin", default="DOGEUSDT", help="schema-v2 run-side-down and price-down 이벤트 기준 자산")
    parser.add_argument("--lag-minutes", type=int, default=LAG_MIN_DEFAULT)
    parser.add_argument(
        "--cointegration-csv",
        default=str(OUT_DIR / "cointegration_AVAXUSDT_DOGEUSDT.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    a_df = pd.read_parquet(DATA_DIR / f"{args.coin_a}_1m.parquet")[["timestamp", "close"]]
    b_df = pd.read_parquet(DATA_DIR / f"{args.coin_b}_1m.parquet")[["timestamp", "close"]]
    a_df = a_df.rename(columns={"close": args.coin_a})
    b_df = b_df.rename(columns={"close": args.coin_b})
    a_df["timestamp"] = pd.to_datetime(a_df["timestamp"], utc=True)
    b_df["timestamp"] = pd.to_datetime(b_df["timestamp"], utc=True)
    df = a_df.merge(b_df, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)

    pairs = pd.read_csv(args.cointegration_csv)
    pairs["start_ts"] = pd.to_datetime(pairs["start_ts"], utc=True)
    pairs["end_ts"] = pd.to_datetime(pairs["end_ts"], utc=True)
    pairs = pairs.loc[pairs["passes_filter"]].copy()
    print(f"using {len(pairs)} cointegrated windows")

    # Match screening's time filter so that start_idx/end_idx align with log_prices.iloc positions
    if len(pairs) > 0:
        first_ts = pairs["start_ts"].min()
        df = df.loc[df["timestamp"] >= first_ts].reset_index(drop=True)

    log_prices = pd.DataFrame(
        {
            args.coin_a: np.log(df[args.coin_a].astype(float).values),
            args.coin_b: np.log(df[args.coin_b].astype(float).values),
        },
        index=df["timestamp"],
    )

    # Build minute-level event mask once for the whole timeline
    event_mask = load_event_mask(args.filter_coin, args.lag_minutes, log_prices.index)
    print(f"event_mask: {int(event_mask.sum())} of {len(event_mask)} minutes flagged "
          f"(coin={args.filter_coin}, lag={args.lag_minutes}min)")

    all_results: list[dict] = []
    for mode_name, mask in [("baseline", None), ("filtered", event_mask)]:
        for _, row in pairs.iterrows():
            for slip in SLIPPAGE_BPS_LIST:
                # use positional iloc reset: row["start_idx"] / row["end_idx"] are integer
                row_dict = row.to_dict()
                # Convert slot to integer indices into log_prices (which is indexed by timestamp)
                # We need to translate start_idx/end_idx → reset to numeric position in log_prices
                # The screening script used df-level indices that map 1-to-1 to log_prices since df row order is preserved.
                out = backtest_one_window(row_dict, log_prices, args.coin_a, args.coin_b, slip, mask)
                if out is None:
                    continue
                out["mode"] = mode_name
                all_results.append(out)

    res = pd.DataFrame(all_results)
    out_all = OUT_DIR / "backtest_all_rows.csv"
    res.to_csv(out_all, index=False)
    print(f"saved: {out_all}")

    # Summary per (mode, slip)
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
    out_summary = OUT_DIR / "backtest_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"saved: {out_summary}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())

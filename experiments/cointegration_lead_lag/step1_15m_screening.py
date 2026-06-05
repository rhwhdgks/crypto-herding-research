"""15분봉으로 AVAX|DOGE cointegration screening — Sprint1 PPT setting과 정합.

1분봉 → 15min resample, 3-day rolling window (= 288 bars), 1-day step (= 96 bars).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller

DATA_DIR = Path("/home/jonghan/findalpha/herding/data")
OUT_DIR = Path("/home/jonghan/findalpha/herding/experiments/cointegration_lead_lag/outputs")

COIN_A = "AVAXUSDT"
COIN_B = "DOGEUSDT"
START = "2025-05-08"
END = "2026-04-09"

WINDOW_BARS = 3 * 24 * 4  # 3-day = 288 bars (15min)
STEP_BARS = 24 * 4  # 1-day = 96 bars
ADF_PVALUE_MAX = 0.05
HL_MIN = 2  # 15분 단위 → 30분 (Sprint1 30 min 기준)
HL_MAX = WINDOW_BARS * 0.5  # 144 bars = 36시간


def half_life(spread: pd.Series) -> float:
    s_lag = spread.shift(1).dropna()
    ds = spread.diff().dropna()
    aligned = pd.concat([s_lag, ds], axis=1).dropna()
    if len(aligned) < 30:
        return np.nan
    x = add_constant(aligned.iloc[:, 0].values)
    model = OLS(aligned.iloc[:, 1].values, x).fit()
    beta = model.params[1]
    if beta >= 0:
        return np.nan
    return float(-np.log(2) / beta)


def engle_granger(log_a: pd.Series, log_b: pd.Series):
    aligned = pd.concat([log_a, log_b], axis=1).dropna()
    if len(aligned) < 100:
        return np.nan, np.nan, np.nan
    x = add_constant(aligned.iloc[:, 1].values)
    model = OLS(aligned.iloc[:, 0].values, x).fit()
    beta = float(model.params[1])
    spread = aligned.iloc[:, 0] - beta * aligned.iloc[:, 1]
    try:
        adf_result = adfuller(spread.values, autolag="AIC")
        pvalue = float(adf_result[1])
    except Exception:
        return beta, np.nan, np.nan
    hl = half_life(spread)
    return beta, pvalue, hl


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a_df = pd.read_parquet(DATA_DIR / f"{COIN_A}_1m.parquet")[["timestamp", "close"]]
    b_df = pd.read_parquet(DATA_DIR / f"{COIN_B}_1m.parquet")[["timestamp", "close"]]
    a_df["timestamp"] = pd.to_datetime(a_df["timestamp"], utc=True)
    b_df["timestamp"] = pd.to_datetime(b_df["timestamp"], utc=True)
    df_1m = a_df.merge(b_df, on="timestamp", how="inner", suffixes=("_a", "_b"))
    df_1m = df_1m.rename(columns={"close_a": COIN_A, "close_b": COIN_B})
    df_1m = df_1m.set_index("timestamp").sort_index()

    start_ts = pd.Timestamp(START, tz="UTC")
    end_ts = pd.Timestamp(END, tz="UTC")
    df_1m = df_1m.loc[start_ts:end_ts]

    df = df_1m.resample("15min").last().dropna()
    print(f"15min frame: {len(df)} bars ({df.index.min()} ~ {df.index.max()})")

    log_prices = np.log(df.astype(float))

    rows: list[dict] = []
    window_day = 0
    for start_idx in range(0, len(df) - WINDOW_BARS, STEP_BARS):
        end_idx = start_idx + WINDOW_BARS
        sub_a = log_prices[COIN_A].iloc[start_idx:end_idx]
        sub_b = log_prices[COIN_B].iloc[start_idx:end_idx]
        beta, pvalue, hl = engle_granger(sub_a, sub_b)
        passes = (
            np.isfinite(pvalue)
            and pvalue < ADF_PVALUE_MAX
            and np.isfinite(hl)
            and HL_MIN <= hl <= HL_MAX
        )
        rows.append({
            "window_day": window_day,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_ts": df.index[start_idx],
            "end_ts": df.index[end_idx - 1],
            "coin_a": COIN_A,
            "coin_b": COIN_B,
            "hedge_beta": beta,
            "adf_pvalue": pvalue,
            "half_life_bars": hl,
            "half_life_minutes": hl * 15 if np.isfinite(hl) else np.nan,
            "n_obs": len(sub_a.dropna()),
            "passes_filter": bool(passes),
        })
        window_day += 1

    res = pd.DataFrame(rows)
    out_path = OUT_DIR / f"cointegration_15m_{COIN_A}_{COIN_B}.csv"
    res.to_csv(out_path, index=False)
    n_pass = int(res["passes_filter"].sum())
    print(f"saved: {out_path}")
    print(f"total windows: {len(res)}, tradable: {n_pass} ({n_pass / max(len(res), 1):.1%})")
    print(f"avg half_life (tradable): {res.loc[res['passes_filter'], 'half_life_minutes'].mean():.1f} min")


if __name__ == "__main__":
    raise SystemExit(main())

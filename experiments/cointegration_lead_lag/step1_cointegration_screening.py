"""AVAX|DOGE Binance USDT 1분봉 cointegration screening.

Sprint1의 step5_filter.py / step6_found.py 로직을 단일 페어에 맞춰 재구성.
3-day rolling window, 1-day step, ADF p-value < 0.05, half-life ∈ [30, window*0.5].
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller

DATA_DIR = Path("/home/jonghan/findalpha/herding/data")
OUT_DIR = Path("/home/jonghan/findalpha/herding/experiments/cointegration_lead_lag/outputs")

WINDOW_MINUTES = 3 * 1440  # 3-day window = 4320 minutes
STEP_MINUTES = 1440  # 1-day step
ADF_PVALUE_MAX = 0.05
HL_MIN = 30
HL_MAX = WINDOW_MINUTES * 0.5  # 2160


def half_life(spread: pd.Series) -> float:
    """Half-life of mean reversion via OU regression: Δs_t = α + β·s_{t-1} + ε."""
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


def engle_granger_step(log_a: pd.Series, log_b: pd.Series) -> tuple[float, float, float]:
    """Step1: OLS log_a ~ log_b → β + spread; Step2: ADF on spread.

    Returns (hedge_beta, adf_pvalue, half_life_minutes).
    """
    if len(log_a) < 100 or len(log_b) < 100:
        return np.nan, np.nan, np.nan
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin-a", default="AVAXUSDT")
    parser.add_argument("--coin-b", default="DOGEUSDT")
    parser.add_argument("--start", default="2025-05-08")
    parser.add_argument("--end", default="2026-04-09")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    a_path = DATA_DIR / f"{args.coin_a}_1m.parquet"
    b_path = DATA_DIR / f"{args.coin_b}_1m.parquet"
    a_df = pd.read_parquet(a_path)[["timestamp", "close"]].rename(columns={"close": args.coin_a})
    b_df = pd.read_parquet(b_path)[["timestamp", "close"]].rename(columns={"close": args.coin_b})
    a_df["timestamp"] = pd.to_datetime(a_df["timestamp"], utc=True)
    b_df["timestamp"] = pd.to_datetime(b_df["timestamp"], utc=True)

    df = a_df.merge(b_df, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(args.end, tz="UTC")
    df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)].reset_index(drop=True)

    # close prices
    log_a_full = np.log(df[args.coin_a].astype(float))
    log_b_full = np.log(df[args.coin_b].astype(float))
    df["log_a"] = log_a_full
    df["log_b"] = log_b_full

    n = len(df)
    print(f"merged length: {n} rows ({df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]})")

    rows: list[dict] = []
    window_day = 0
    for start_idx in range(0, n - WINDOW_MINUTES, STEP_MINUTES):
        end_idx = start_idx + WINDOW_MINUTES
        sub = df.iloc[start_idx:end_idx]
        beta, pvalue, hl = engle_granger_step(sub["log_a"], sub["log_b"])
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
            "start_ts": sub["timestamp"].iloc[0],
            "end_ts": sub["timestamp"].iloc[-1],
            "coin_a": args.coin_a,
            "coin_b": args.coin_b,
            "hedge_beta": beta,
            "adf_pvalue": pvalue,
            "half_life": hl,
            "n_obs": int(sub[[args.coin_a, args.coin_b]].dropna().shape[0]),
            "passes_filter": bool(passes),
        })
        window_day += 1

    res = pd.DataFrame(rows)
    out_path = OUT_DIR / f"cointegration_{args.coin_a}_{args.coin_b}.csv"
    res.to_csv(out_path, index=False)
    print(f"saved: {out_path}")
    n_pass = int(res["passes_filter"].sum())
    print(f"total windows: {len(res)}, tradable: {n_pass} ({n_pass / max(len(res), 1):.1%})")
    print(f"ADF p<0.05 only: {int((res['adf_pvalue'] < 0.05).sum())}")
    print(f"avg half_life (tradable): {res.loc[res['passes_filter'], 'half_life'].mean():.1f} min")


if __name__ == "__main__":
    raise SystemExit(main())

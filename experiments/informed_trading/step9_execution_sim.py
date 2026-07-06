"""Step 9 — crowded × no_flush 후보 실행 시뮬레이션 (비용 포함).

규칙 (모두 진입 시점 관측 가능):
  진입: DOGE down micro-herding event 버킷 종료 시점
        + funding_pre > 0.0001 (crowded long)
        + d_oi_event > flush tercile 컷 (OI 미붕괴)
  포지션: 알트 4종(AVAX/ADA/XRP/SOL) 동일가중 long basket, 30분 보유
  변형: DOGE 포함 5종 basket (DOGE 자체 반등 포획)

비용 시나리오 (USDT-M 선물 왕복):
  gross   0.00%   |   maker 0.04% (0.02%×2)   |   taker 0.10% (0.05%×2)

주의: 이것은 최종 승격이 아니라 경제성 1차 확정. flush 컷은 전기간 tercile 이라
mild look-ahead (실전은 rolling 컷 필요) — 컷 자체는 ±0.001 근방에서 안정적.
출력: experiments/informed_trading/outputs/execution_sim_summary.csv
      experiments/informed_trading/outputs/execution_sim_events.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/jonghan/findalpha/herding")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tick_lead_lag import build_lead_lag_frame, _compute_t_stat  # type: ignore

MICRO_FRAME = PROJECT_ROOT / "outputs/tick/multi_asset_5y/lead_lag_matrix/intermediate/tick_micro_frame_15m.csv"
STATE_PATH = PROJECT_ROOT / "experiments/informed_trading/outputs/doge_futures_state_15m.csv"
OUT_DIR = PROJECT_ROOT / "experiments/informed_trading/outputs"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT"]
BASKET = ["AVAXUSDT", "ADAUSDT", "XRPUSDT", "SOLUSDT"]
DIRECTION = "down"
HORIZON = 30
INTERVAL = 15
DEFAULT_FUNDING = 0.0001
FEES = {"gross": 0.0, "maker_rt": 0.0004, "taker_rt": 0.0010}

USECOLS = [
    "symbol", "bucket_start", "event_label", f"forward_return_{HORIZON}m",
    "hour_utc", "is_target_session", "meets_trade_count",
    "bucket_return", "transaction_count", "herding_score",
]


def main() -> None:
    print("micro frame 로드 중...")
    mf = pd.read_csv(MICRO_FRAME, usecols=USECOLS)
    mf["bucket_start"] = pd.to_datetime(mf["bucket_start"], utc=True)

    state = pd.read_csv(STATE_PATH)
    state["bucket_start"] = pd.to_datetime(state["bucket_start"], utc=True)
    state = state[["bucket_start", "funding_pre", "d_oi_event", "oi_end"]]

    # basket frame: 타깃별 forward return 을 bucket_start 로 병합
    merged: pd.DataFrame | None = None
    for target in BASKET:
        leaders = [s for s in SYMBOLS if s != target]
        llf = build_lead_lag_frame(
            micro_frame=mf, target_symbol=target, leader_symbols=leaders,
            interval_minutes=INTERVAL, forward_horizon_minutes=HORIZON,
            event_direction=DIRECTION,
        )
        valid = llf.loc[llf["non_target_control"] & llf["target_forward_return"].notna()]
        col = f"fwd_{target.replace('USDT', '').lower()}"
        part = valid[["bucket_start", "doge_event", "target_forward_return"]].rename(
            columns={"target_forward_return": col})
        merged = part if merged is None else merged.merge(part, on=["bucket_start", "doge_event"], how="outer")

    # DOGE 자체 forward return (이벤트 버킷의 DOGE +30min)
    doge = mf.loc[mf["symbol"] == "DOGEUSDT", ["bucket_start", f"forward_return_{HORIZON}m"]].rename(
        columns={f"forward_return_{HORIZON}m": "fwd_doge"})
    merged = merged.merge(doge, on="bucket_start", how="left")
    merged = merged.merge(state, on="bucket_start", how="left")
    merged = merged.loc[merged["oi_end"].notna()].copy()
    merged["doge_event"] = merged["doge_event"].fillna(False)

    fwd_cols = [f"fwd_{t.replace('USDT', '').lower()}" for t in BASKET]
    merged["basket4"] = merged[fwd_cols].mean(axis=1)
    merged["basket5"] = merged[fwd_cols + ["fwd_doge"]].mean(axis=1)

    events = merged.loc[
        merged["doge_event"] & merged["funding_pre"].notna() & merged["d_oi_event"].notna()
    ].copy()
    flush_cut = events["d_oi_event"].quantile(1 / 3)
    cell = events.loc[
        (events["funding_pre"] > DEFAULT_FUNDING) & (events["d_oi_event"] > flush_cut)
    ].sort_values("bucket_start").copy()
    cell["year"] = cell["bucket_start"].dt.year
    print(f"cell 이벤트: {len(cell)}  ({cell['bucket_start'].min().date()} ~ {cell['bucket_start'].max().date()})")

    rows: list[dict] = []
    for basket_name in ["basket4", "basket5"]:
        gross = cell[basket_name].dropna()
        t_stat = _compute_t_stat(gross)
        for fee_name, fee in FEES.items():
            net = gross - fee
            cum = net.cumsum()
            dd = float((cum.cummax() - cum).max())
            rows.append({
                "basket": basket_name, "fee_scenario": fee_name, "fee_rt": fee,
                "n": int(len(net)),
                "mean_pct": float(net.mean()) * 100,
                "median_pct": float(net.median()) * 100,
                "sd_pct": float(net.std()) * 100,
                "win_rate": float((net > 0).mean()),
                "total_return_pct": float(net.sum()) * 100,
                "max_drawdown_pct": dd * 100,
                "t_stat_gross": float(t_stat),
                "sharpe_per_trade": float(net.mean() / net.std()) if net.std() > 0 else np.nan,
            })

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "execution_sim_summary.csv", index=False)
    cell[["bucket_start", "year", "funding_pre", "d_oi_event", "basket4", "basket5"] + fwd_cols + ["fwd_doge"]].to_csv(
        OUT_DIR / "execution_sim_events.csv", index=False)

    print("\n=== 실행 시뮬 (30min hold, 이벤트당 %) ===")
    for _, r in out.iterrows():
        print(f"  {r['basket']} | {r['fee_scenario']:>8}: mean={r['mean_pct']:+.4f}%  med={r['median_pct']:+.4f}%  "
              f"win={r['win_rate']:.3f}  total={r['total_return_pct']:+.1f}%  maxDD={r['max_drawdown_pct']:.2f}%")

    print("\n=== 연도별 (basket4, gross) ===")
    yearly = cell.groupby("year")["basket4"].agg(["count", "mean", lambda s: (s > 0).mean()])
    yearly.columns = ["n", "mean", "win_rate"]
    for y, r in yearly.iterrows():
        print(f"  {y}: n={int(r['n']):>3}  mean={r['mean']*100:+.4f}%  win={r['win_rate']:.3f}")

    print(f"\nsaved → {OUT_DIR / 'execution_sim_summary.csv'}")


if __name__ == "__main__":
    raise SystemExit(main())

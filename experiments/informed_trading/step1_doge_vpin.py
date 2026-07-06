"""Step 1 — DOGE 15분 버킷별 order-flow imbalance (VPIN-proxy) 계산.

선행연구 (Herding, information cascades, and cryptocurrencies) 의 informed-trading
axis (VPIN + SUR) 를 우리 lead-lag matrix 와 결합하기 위한 첫 단계.

핵심 질문(확장방향 A):
  "DOGE down → AVAX/ADA 30분 공동 반응이 informed trading 에 driving 되는가,
   아니면 단순 noise herding 인가?"

이를 위해 leader(DOGE) 의 15분 버킷별 informed-flow 강도를 정량화한다.
aggTrades 는 taker 방향(aggressor side)을 정확히 준다:
  is_buyer_maker == True  → 매도 aggressor (SELL volume)
  is_buyer_maker == False → 매수 aggressor (BUY volume)

버킷 지표:
  order_imbalance = buy_vol - sell_vol            (부호: 음수 = 순매도)
  toxicity        = |buy_vol - sell_vol| / total  (단일 버킷 VPIN, 0~1)
  vpin_50         = 직전 50버킷 toxicity 이동평균  (Easley-LdP VPIN smoothing 근사)

주의: 이것은 canonical equal-volume-bucket VPIN 이 아니라, event frame(15분 시간
버킷)에 정합하도록 만든 15분 시간버킷 order-flow imbalance / VPIN-proxy 이다.
buy/sell 분류는 BVC 근사가 아니라 aggTrades 의 실제 aggressor side 를 쓰므로 정확.

트랙 분리: baseline / lead_lag_matrix 로직을 건드리지 않고 leader 측 보조 feature 만 생성.
출력: experiments/informed_trading/outputs/doge_vpin_15m.csv
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/jonghan/findalpha/herding")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tick_data import iter_tick_archive_chunks  # type: ignore

LEADER = "DOGEUSDT"
ARCHIVE_DIR = PROJECT_ROOT / "data/tick_archive" / LEADER / "aggTrades" / "monthly"
OUT_DIR = PROJECT_ROOT / "experiments/informed_trading/outputs"
OUT_PATH = OUT_DIR / "doge_vpin_15m.csv"

BUCKET = "15min"
CHUNKSIZE = 3_000_000
VPIN_WINDOW = 50  # 직전 N개 버킷 smoothing


def month_range(start: str, end: str) -> list[str]:
    idx = pd.date_range(start=start, end=end, freq="MS")
    return [ts.strftime("%Y-%m") for ts in idx]


def accumulate_month(zip_path: Path, buy_acc: dict, sell_acc: dict, cnt_acc: dict) -> int:
    rows = 0
    for chunk in iter_tick_archive_chunks(zip_path, trade_kind="aggTrades", chunksize=CHUNKSIZE):
        rows += len(chunk)
        bucket = chunk["timestamp"].dt.floor(BUCKET)
        is_sell = chunk["is_buyer_maker"] == True  # noqa: E712  (aggressor = seller)
        buy = chunk.loc[~is_sell].groupby(bucket[~is_sell])["quantity"].sum()
        sell = chunk.loc[is_sell].groupby(bucket[is_sell])["quantity"].sum()
        cnt = chunk.groupby(bucket)["quantity"].count()
        for ts, v in buy.items():
            buy_acc[ts] += float(v)
        for ts, v in sell.items():
            sell_acc[ts] += float(v)
        for ts, v in cnt.items():
            cnt_acc[ts] += int(v)
    return rows


def main() -> None:
    start = sys.argv[1] if len(sys.argv) > 1 else "2021-05-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-04-01"
    months = month_range(start, end)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"DOGE VPIN-proxy: {len(months)} months ({months[0]} ~ {months[-1]})")
    buy_acc: dict = defaultdict(float)
    sell_acc: dict = defaultdict(float)
    cnt_acc: dict = defaultdict(int)

    t0 = time.time()
    total_rows = 0
    for i, ym in enumerate(months, 1):
        zip_path = ARCHIVE_DIR / f"{LEADER}-aggTrades-{ym}.zip"
        if not zip_path.exists():
            print(f"  [{i}/{len(months)}] {ym}: MISSING, skip")
            continue
        rows = accumulate_month(zip_path, buy_acc, sell_acc, cnt_acc)
        total_rows += rows
        dt = time.time() - t0
        print(f"  [{i}/{len(months)}] {ym}: {rows:>10,} trades  "
              f"(cum {total_rows:>12,}, {dt:6.1f}s, {len(buy_acc):,} buckets)", flush=True)

    if not buy_acc:
        raise SystemExit("No data accumulated — check archive paths.")

    buckets = sorted(set(buy_acc) | set(sell_acc))
    df = pd.DataFrame({
        "bucket_start": buckets,
        "buy_vol": [buy_acc.get(b, 0.0) for b in buckets],
        "sell_vol": [sell_acc.get(b, 0.0) for b in buckets],
        "trade_count": [cnt_acc.get(b, 0) for b in buckets],
    })
    df["bucket_start"] = pd.to_datetime(df["bucket_start"], utc=True)
    df["total_vol"] = df["buy_vol"] + df["sell_vol"]
    df["order_imbalance"] = df["buy_vol"] - df["sell_vol"]
    df["signed_imbalance_ratio"] = df["order_imbalance"] / df["total_vol"].replace(0, np.nan)
    df["toxicity"] = df["order_imbalance"].abs() / df["total_vol"].replace(0, np.nan)
    df = df.sort_values("bucket_start").reset_index(drop=True)
    df["vpin_50"] = df["toxicity"].rolling(VPIN_WINDOW, min_periods=VPIN_WINDOW // 2).mean()

    df.to_csv(OUT_PATH, index=False)
    print()
    print(f"=== DOGE VPIN-proxy 완료 ({time.time()-t0:.1f}s) ===")
    print(f"buckets: {len(df):,}  |  범위: {df['bucket_start'].min()} ~ {df['bucket_start'].max()}")
    print(f"toxicity  mean={df['toxicity'].mean():.4f}  median={df['toxicity'].median():.4f}  "
          f"p90={df['toxicity'].quantile(0.90):.4f}")
    print(f"vpin_50   mean={df['vpin_50'].mean():.4f}  p90={df['vpin_50'].quantile(0.90):.4f}")
    print(f"saved → {OUT_PATH}")


if __name__ == "__main__":
    raise SystemExit(main())

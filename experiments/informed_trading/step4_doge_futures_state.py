"""Step 4 — DOGE 선물 레버리지 상태변수 수집 + 15분 그리드 구축.

방향 A (레버리지 캐스케이드 축):
  step2/3 결론 = 살아남은 공동반응은 noise-driven diffuse beta.
  용의자 메커니즘 = 선물 청산 캐스케이드 (강제 flow 는 방향은 쏠려도 정보성이 없음).
  이를 검증하려면 OHLCV 밖의 새 정보축이 필요 — funding rate / open interest.

데이터 (Binance 공개 아카이브, data.binance.vision):
  - fundingRate: monthly zip, 8h 간격 (2020-07~)
  - metrics: daily zip, 5분 간격 (2021-12-01~) — sum_open_interest 등

산출 15분 변수 (look-ahead 구분이 핵심):
  [사전 관측 가능 — ex-ante 필터 후보]
    funding_pre    : bucket_start 시점까지 공표된 마지막 funding rate (ffill)
    d_oi_24h_pre   : 직전 24h OI 증감률 (bucket_start 기준, 레버리지 buildup)
  [동시 — 메커니즘 진단용, ex-ante 아님]
    d_oi_event     : 이벤트 버킷 내 OI 증감률 (급감 = long 청산 flush 시그니처)

출력:
  data/futures_archive/DOGEUSDT/{fundingRate,metrics}/  (raw zip, gitignored)
  experiments/informed_trading/outputs/doge_futures_state_15m.csv
"""
from __future__ import annotations

import io
import sys
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/jonghan/findalpha/herding")
SYMBOL = "DOGEUSDT"
BASE = "https://data.binance.vision/data/futures/um"
RAW_DIR = PROJECT_ROOT / "data/futures_archive" / SYMBOL
OUT_DIR = PROJECT_ROOT / "experiments/informed_trading/outputs"
OUT_PATH = OUT_DIR / "doge_futures_state_15m.csv"

FUNDING_START, FUNDING_END = "2021-05-01", "2026-04-01"   # monthly
METRICS_START, METRICS_END = "2021-12-01", "2026-04-30"   # daily (archive 시작이 2021-12)


def download(url: str, dest: Path) -> str:
    if dest.exists():
        return "cache"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url) as resp, dest.open("wb") as fh:
            fh.write(resp.read())
        return "download"
    except HTTPError as exc:
        dest.unlink(missing_ok=True)
        if exc.code == 404:
            return "missing"
        raise


def read_zip_csv(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        member = [n for n in zf.namelist() if n.endswith(".csv")][0]
        return pd.read_csv(io.BytesIO(zf.read(member)))


def collect_funding() -> pd.DataFrame:
    months = pd.date_range(FUNDING_START, FUNDING_END, freq="MS")
    frames = []
    print(f"funding: {len(months)} monthly files")
    for ts in months:
        ym = ts.strftime("%Y-%m")
        name = f"{SYMBOL}-fundingRate-{ym}.zip"
        dest = RAW_DIR / "fundingRate" / name
        status = download(f"{BASE}/monthly/fundingRate/{SYMBOL}/{name}", dest)
        if status == "missing":
            print(f"  {ym}: MISSING")
            continue
        frames.append(read_zip_csv(dest))
    df = pd.concat(frames, ignore_index=True)
    df["calc_time"] = pd.to_datetime(pd.to_numeric(df["calc_time"]), unit="ms", utc=True)
    df = df[["calc_time", "last_funding_rate"]].sort_values("calc_time").reset_index(drop=True)
    print(f"funding rows: {len(df):,}  ({df['calc_time'].min()} ~ {df['calc_time'].max()})")
    return df


def collect_metrics() -> pd.DataFrame:
    days = pd.date_range(METRICS_START, METRICS_END, freq="D")
    frames = []
    missing = 0
    t0 = time.time()
    print(f"metrics: {len(days)} daily files")
    for i, ts in enumerate(days, 1):
        ymd = ts.strftime("%Y-%m-%d")
        name = f"{SYMBOL}-metrics-{ymd}.zip"
        dest = RAW_DIR / "metrics" / name
        status = download(f"{BASE}/daily/metrics/{SYMBOL}/{name}", dest)
        if status == "missing":
            missing += 1
            continue
        frames.append(read_zip_csv(dest))
        if i % 100 == 0:
            print(f"  [{i}/{len(days)}] {ymd}  ({time.time()-t0:6.1f}s, missing={missing})", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df["create_time"] = pd.to_datetime(df["create_time"], utc=True, format="mixed")
    df = df[["create_time", "sum_open_interest", "sum_open_interest_value",
             "sum_taker_long_short_vol_ratio"]].sort_values("create_time").reset_index(drop=True)
    print(f"metrics rows: {len(df):,}  missing days: {missing}  "
          f"({df['create_time'].min()} ~ {df['create_time'].max()})")
    return df


def build_state(funding: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    # 15분 그리드: metrics 5분봉의 각 버킷 마지막 관측 = oi_end
    metrics = metrics.copy()
    metrics["bucket_start"] = metrics["create_time"].dt.floor("15min")
    grp = metrics.groupby("bucket_start")
    state = pd.DataFrame({
        "oi_end": grp["sum_open_interest"].last(),
        "oi_value_end": grp["sum_open_interest_value"].last(),
        "taker_ls_mean": grp["sum_taker_long_short_vol_ratio"].mean(),
        "n_obs_5m": grp["sum_open_interest"].count(),
    }).reset_index().sort_values("bucket_start").reset_index(drop=True)

    # 그리드 결손을 명시적으로 두고 shift 기반 변수 생성 (reindex로 갭 보전 안 함 — 갭은 NaN)
    full_idx = pd.date_range(state["bucket_start"].min(), state["bucket_start"].max(), freq="15min", tz="UTC")
    state = state.set_index("bucket_start").reindex(full_idx)
    state.index.name = "bucket_start"

    oi = state["oi_end"]
    state["oi_start"] = oi.shift(1)                             # 버킷 시작 시점 OI (직전 버킷 종료값)
    state["d_oi_event"] = oi / state["oi_start"] - 1.0          # 동시: 이벤트 버킷 내 OI 변화
    state["d_oi_24h_pre"] = state["oi_start"] / state["oi_start"].shift(96) - 1.0  # ex-ante buildup

    state = state.reset_index()
    # merge_asof 는 양쪽 datetime 해상도가 같아야 함 (ms vs us mismatch 방지)
    state["bucket_start"] = state["bucket_start"].dt.as_unit("ns")
    funding = funding.copy()
    funding["calc_time"] = funding["calc_time"].dt.as_unit("ns")

    # funding: bucket_start 시점까지 공표된 마지막 rate (merge_asof backward = ffill, look-ahead 없음)
    state = pd.merge_asof(
        state.sort_values("bucket_start"),
        funding.rename(columns={"calc_time": "bucket_start", "last_funding_rate": "funding_pre"}),
        on="bucket_start", direction="backward",
    )
    return state


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    funding = collect_funding()
    metrics = collect_metrics()
    state = build_state(funding, metrics)
    state.to_csv(OUT_PATH, index=False)

    valid = state["oi_end"].notna()
    print()
    print(f"=== DOGE futures state 15m 완료 ===")
    print(f"buckets: {len(state):,} (OI 관측 {int(valid.sum()):,})")
    print(f"범위: {state['bucket_start'].min()} ~ {state['bucket_start'].max()}")
    print(f"funding_pre  mean={state['funding_pre'].mean():+.6f}  p10={state['funding_pre'].quantile(0.1):+.6f}  "
          f"p90={state['funding_pre'].quantile(0.9):+.6f}")
    print(f"d_oi_event   p10={state['d_oi_event'].quantile(0.1):+.5f}  p90={state['d_oi_event'].quantile(0.9):+.5f}")
    print(f"d_oi_24h_pre p10={state['d_oi_24h_pre'].quantile(0.1):+.5f}  p90={state['d_oi_24h_pre'].quantile(0.9):+.5f}")
    print(f"saved → {OUT_PATH}")


if __name__ == "__main__":
    raise SystemExit(main())

"""Step 8 — crowded × no_flush 셀 permutation test (circular shift null).

배경: 이 셀은 조건부 스캔 끝에 발견됐다 (다중비교 우려). 정면 검정:
  "DOGE down event 와 레버리지 상태(funding_pre, d_oi_event)의 시간 정렬이
   정보를 갖는가, 아니면 같은 크기의 아무 부분집합이나 비슷하게 나오는가?"

방법:
  - 통계량: 알트 4종(AVAX/ADA/XRP/SOL) 동일가중 basket 의 +30min delta
    (타깃별 쪼개기 다중비교 제거 — basket 이 primary, 타깃별은 참고)
  - Null: state 시계열(funding_pre, d_oi_event)을 15분 그리드 위에서
    circular shift (k ∈ [96, N-96] 랜덤, 1000회). 두 시계열의 자기상관/군집
    구조를 보존한 채 이벤트와의 정렬만 파괴 — iid label shuffle 보다 보수적.
  - 각 draw 에서 flush tercile 컷도 재계산 (발견 절차 자체를 null 에서 재현)
  - control(비이벤트 버킷 basket 평균)은 라벨과 무관하므로 고정.

출력: experiments/informed_trading/outputs/permutation_crowded_noflush.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/jonghan/findalpha/herding")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tick_lead_lag import build_lead_lag_frame  # type: ignore

MICRO_FRAME = PROJECT_ROOT / "outputs/tick/multi_asset_5y/lead_lag_matrix/intermediate/tick_micro_frame_15m.csv"
STATE_PATH = PROJECT_ROOT / "experiments/informed_trading/outputs/doge_futures_state_15m.csv"
OUT_DIR = PROJECT_ROOT / "experiments/informed_trading/outputs"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT"]
BASKET = ["AVAXUSDT", "ADAUSDT", "XRPUSDT", "SOLUSDT"]
DIRECTION = "down"
HORIZON = 30
INTERVAL = 15
DEFAULT_FUNDING = 0.0001
N_SHUFFLES = 1000
SEED = 42

USECOLS = [
    "symbol", "bucket_start", "event_label", f"forward_return_{HORIZON}m",
    "hour_utc", "is_target_session", "meets_trade_count",
    "bucket_return", "transaction_count", "herding_score",
]


def build_basket_frame(mf: pd.DataFrame) -> pd.DataFrame:
    """타깃별 valid frame 을 bucket_start 기준으로 합쳐 basket forward return 구성."""
    merged: pd.DataFrame | None = None
    for target in BASKET:
        leaders = [s for s in SYMBOLS if s != target]
        llf = build_lead_lag_frame(
            micro_frame=mf,
            target_symbol=target,
            leader_symbols=leaders,
            interval_minutes=INTERVAL,
            forward_horizon_minutes=HORIZON,
            event_direction=DIRECTION,
        )
        valid = llf.loc[llf["non_target_control"] & llf["target_forward_return"].notna()].copy()
        col = f"fwd_{target.replace('USDT', '').lower()}"
        part = valid[["bucket_start", "doge_event", "target_forward_return"]].rename(
            columns={"target_forward_return": col}
        )
        if merged is None:
            merged = part
        else:
            merged = merged.merge(part, on=["bucket_start", "doge_event"], how="outer")
    fwd_cols = [f"fwd_{t.replace('USDT', '').lower()}" for t in BASKET]
    merged["basket_fwd"] = merged[fwd_cols].mean(axis=1)
    merged["doge_event"] = merged["doge_event"].fillna(False)
    return merged.sort_values("bucket_start").reset_index(drop=True)


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("micro frame 로드 중...")
    mf = pd.read_csv(MICRO_FRAME, usecols=USECOLS)
    mf["bucket_start"] = pd.to_datetime(mf["bucket_start"], utc=True)

    state = pd.read_csv(STATE_PATH)
    state["bucket_start"] = pd.to_datetime(state["bucket_start"], utc=True)
    grid_start = state["bucket_start"].iloc[0]
    n_grid = len(state)
    funding_arr = state["funding_pre"].to_numpy()
    doi_arr = state["d_oi_event"].to_numpy()
    print(f"state grid: {n_grid:,} buckets from {grid_start}")

    frame = build_basket_frame(mf)
    # state 커버리지로 제한 + 그리드 인덱스 부여
    frame["grid_idx"] = ((frame["bucket_start"] - grid_start) / pd.Timedelta(minutes=15)).astype("int64")
    frame = frame.loc[(frame["grid_idx"] >= 0) & (frame["grid_idx"] < n_grid)].copy()
    frame = frame.loc[frame["basket_fwd"].notna()].copy()

    ev_mask = frame["doge_event"].to_numpy()
    basket_fwd = frame["basket_fwd"].to_numpy()
    ev_idx = frame.loc[frame["doge_event"], "grid_idx"].to_numpy()
    ev_fwd = basket_fwd[ev_mask]
    control_mean = float(basket_fwd[~ev_mask].mean())
    n_control = int((~ev_mask).sum())
    print(f"basket frame: {len(frame):,} buckets  |  events {len(ev_idx):,}  control {n_control:,}")
    print(f"control basket mean: {control_mean*100:+.4f}%")

    def cell_delta(shift: int) -> tuple[float, int]:
        fp = funding_arr[(ev_idx + shift) % n_grid]
        doi = doi_arr[(ev_idx + shift) % n_grid]
        valid = ~np.isnan(fp) & ~np.isnan(doi)
        if valid.sum() < 30:
            return np.nan, 0
        flush_cut = np.quantile(doi[valid], 1 / 3)
        sel = valid & (fp > DEFAULT_FUNDING) & (doi > flush_cut)
        if sel.sum() < 10:
            return np.nan, int(sel.sum())
        return float(ev_fwd[sel].mean() - control_mean), int(sel.sum())

    obs_delta, obs_n = cell_delta(0)
    print(f"\n관측값 (shift=0): basket delta={obs_delta*100:+.4f}%  n_cell={obs_n}")

    shifts = rng.integers(96, n_grid - 96, size=N_SHUFFLES)
    null_deltas = np.full(N_SHUFFLES, np.nan)
    null_ns = np.zeros(N_SHUFFLES, dtype=int)
    for i, k in enumerate(shifts):
        null_deltas[i], null_ns[i] = cell_delta(int(k))
        if (i + 1) % 200 == 0:
            print(f"  shuffle {i+1}/{N_SHUFFLES}...", flush=True)

    ok = ~np.isnan(null_deltas)
    nd = null_deltas[ok]
    p_one = float((nd >= obs_delta).mean())
    p_two = float((np.abs(nd) >= abs(obs_delta)).mean())

    print(f"\n=== Permutation 결과 (circular shift, {int(ok.sum())} valid draws) ===")
    print(f"null delta:  mean={nd.mean()*100:+.4f}%  sd={nd.std()*100:.4f}%  "
          f"p95={np.quantile(nd, 0.95)*100:+.4f}%  p99={np.quantile(nd, 0.99)*100:+.4f}%")
    print(f"null n_cell: mean={null_ns[ok].mean():.0f}")
    print(f"관측 delta:  {obs_delta*100:+.4f}%  (n={obs_n})")
    print(f"p-value (one-sided, null ≥ obs): {p_one:.4f}")
    print(f"p-value (two-sided, |null| ≥ |obs|): {p_two:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "shift": np.concatenate([[0], shifts]),
        "delta": np.concatenate([[obs_delta], null_deltas]),
        "n_cell": np.concatenate([[obs_n], null_ns]),
        "is_observed": [True] + [False] * N_SHUFFLES,
    }).to_csv(OUT_DIR / "permutation_crowded_noflush.csv", index=False)
    print(f"\nsaved → {OUT_DIR / 'permutation_crowded_noflush.csv'}")

    verdict = "통과 (정렬이 정보를 가짐)" if p_one < 0.05 else "기각 실패 — null 과 구분 안 됨"
    print(f"\n판정: {verdict}")


if __name__ == "__main__":
    raise SystemExit(main())

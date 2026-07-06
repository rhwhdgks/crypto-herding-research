"""Step 6 — 구현 가능한 결합 필터: funding regime × OI flush.

step5 발견:
  - funding_pre 단조 기울기 (ex-ante): 높을수록 반등 강함
  - d_oi_event flush 에 반등 집중 (이벤트 버킷 종료 시점에 관측 완료 → 진입 시점 구현 가능)

이 step 은 둘을 결합한 "청산 캐스케이드 프로파일"을 검정한다:
  funding regime 은 분위수 대신 경제적 카테고리로:
    negative : funding < 0        (숏이 지불 — long 과열 아님)
    default  : 0 ≤ f ≤ 0.0001    (기본요율 근방 — 중립)
    crowded  : f > 0.0001        (기본요율 초과 — long 과열, 프리미엄 지불 중)
  OI flush 는 이벤트 표본 내 3분위 중 최하위 (OI 급감).

주의: 결합 셀은 표본이 작아진다 (n 명시). 이것은 최종 승격이 아니라
"수수료 장벽에 도달 가능한 프로파일이 존재하는가"의 1차 스크리닝이다.
출력: experiments/informed_trading/outputs/combined_filter_summary.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/jonghan/findalpha/herding")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tick_lead_lag import build_lead_lag_frame, _compute_difference_t_stat  # type: ignore

MICRO_FRAME = PROJECT_ROOT / "outputs/tick/multi_asset_5y/lead_lag_matrix/intermediate/tick_micro_frame_15m.csv"
STATE_PATH = PROJECT_ROOT / "experiments/informed_trading/outputs/doge_futures_state_15m.csv"
OUT_DIR = PROJECT_ROOT / "experiments/informed_trading/outputs"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT"]
TARGETS = ["AVAXUSDT", "ADAUSDT"]
DIRECTION = "down"
HORIZON = 30
INTERVAL = 15
DEFAULT_FUNDING = 0.0001  # Binance 기본요율

USECOLS = [
    "symbol", "bucket_start", "event_label", f"forward_return_{HORIZON}m",
    "hour_utc", "is_target_session", "meets_trade_count",
    "bucket_return", "transaction_count", "herding_score",
]


def welch_row(event: pd.Series, control: pd.Series) -> dict:
    event = event.dropna()
    control = control.dropna()
    return {
        "n_event": int(event.shape[0]),
        "n_control": int(control.shape[0]),
        "event_mean": float(event.mean()) if len(event) else np.nan,
        "control_mean": float(control.mean()) if len(control) else np.nan,
        "delta": float(event.mean() - control.mean()) if len(event) and len(control) else np.nan,
        "t_stat": float(_compute_difference_t_stat(event, control)),
        "win_rate": float((event > 0).mean()) if len(event) else np.nan,
    }


def funding_regime(f: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(f < 0, "negative", np.where(f <= DEFAULT_FUNDING, "default", "crowded")),
        index=f.index,
    )


def main() -> None:
    print("micro frame 로드 중...")
    mf = pd.read_csv(MICRO_FRAME, usecols=USECOLS)
    mf["bucket_start"] = pd.to_datetime(mf["bucket_start"], utc=True)

    state = pd.read_csv(STATE_PATH)
    state["bucket_start"] = pd.to_datetime(state["bucket_start"], utc=True)
    state = state[["bucket_start", "d_oi_event", "funding_pre", "oi_end"]]

    rows: list[dict] = []
    for target in TARGETS:
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
        valid = valid.merge(state, on="bucket_start", how="left")
        valid = valid.loc[valid["oi_end"].notna()].copy()

        doge_event = valid["doge_event"].fillna(False)
        control_ret = valid.loc[~doge_event, "target_forward_return"]
        events = valid.loc[doge_event & valid["funding_pre"].notna() & valid["d_oi_event"].notna()].copy()

        events["f_regime"] = funding_regime(events["funding_pre"])
        flush_cut = events["d_oi_event"].quantile(1 / 3)
        events["oi_flush"] = events["d_oi_event"] <= flush_cut

        print(f"\n=== DOGE down → {target.replace('USDT','')} — funding regime × OI flush ===")
        print(f"  flush 컷 (하위 3분위): d_oi_event ≤ {flush_cut:+.5f}")
        dist = events["f_regime"].value_counts()
        print(f"  funding regime 분포: {dict(dist)}")

        for regime in ["negative", "default", "crowded"]:
            for flush_state, flush_label in [(True, "flush"), (False, "no_flush")]:
                cell = events.loc[(events["f_regime"] == regime) & (events["oi_flush"] == flush_state)]
                res = welch_row(cell["target_forward_return"], control_ret)
                res.update({"target": target, "f_regime": regime, "oi_flush": flush_label})
                rows.append(res)
            # regime 전체도 저장
            cell = events.loc[events["f_regime"] == regime]
            res = welch_row(cell["target_forward_return"], control_ret)
            res.update({"target": target, "f_regime": regime, "oi_flush": "ALL"})
            rows.append(res)

        sub = [r for r in rows if r["target"] == target]
        for r in sub:
            if r["oi_flush"] == "ALL":
                continue
            print(f"  {r['f_regime']:>8} × {r['oi_flush']:<8}: delta={r['delta']*100:+.4f}%  "
                  f"t={r['t_stat']:+.2f}  n={r['n_event']:>4}  win={r['win_rate']:.3f}"
                  if r["n_event"] else
                  f"  {r['f_regime']:>8} × {r['oi_flush']:<8}: (표본 없음)")

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "combined_filter_summary.csv", index=False)
    print(f"\nsaved → {OUT_DIR / 'combined_filter_summary.csv'}")

    print("\n=== 핵심: crowded × flush (청산 캐스케이드 프로파일) vs 수수료 장벽 ===")
    for target in TARGETS:
        cf = out[(out["target"] == target) & (out["f_regime"] == "crowded") & (out["oi_flush"] == "flush")]
        if cf.empty or not cf.iloc[0]["n_event"]:
            continue
        r = cf.iloc[0]
        print(f"  {target}: delta={r['delta']*100:+.4f}%/30min  t={r['t_stat']:+.2f}  n={r['n_event']}  "
              f"win={r['win_rate']:.3f}  (taker 왕복 ~0.15% / maker 왕복 ~0.04%)")


if __name__ == "__main__":
    raise SystemExit(main())

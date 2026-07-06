"""Step 5 — 레버리지 상태 조건부 lead-lag 분해 (청산 캐스케이드 가설 검증).

가설:
  H-A1 (메커니즘): DOGE down event 중 OI가 급감한 버킷(= long 청산 flush)일수록
        AVAX/ADA 공동 mean reversion 이 강하다.
  H-A2 (ex-ante): 이벤트 직전 funding 이 높고(crowded long) / 직전 24h OI buildup 이
        클수록 flush 후 반등이 강하다 — 사전 관측 가능하므로 alpha 각도가 있음.

사전 긴장 포인트 (정직하게 기록):
  청산은 one-sided market order flow → toxicity 가 높게 찍혀야 정상.
  그런데 step2 에서 공동반응은 LOW toxicity 에 집중됐다.
  → 만약 H-A1 이 기각되면 "청산 캐스케이드 메커니즘"도 기각되는 것이고,
    d_oi_event 와 toxicity 의 상관으로 이 정합성을 직접 확인한다.

방법: step2/3 과 동일 framework. 상태변수 커버리지(2021-12~)로 이벤트/컨트롤 모두 제한.
출력: experiments/informed_trading/outputs/leverage_conditional_summary.csv
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
VPIN_PATH = PROJECT_ROOT / "experiments/informed_trading/outputs/doge_vpin_15m.csv"
OUT_DIR = PROJECT_ROOT / "experiments/informed_trading/outputs"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT"]
TARGETS = ["AVAXUSDT", "ADAUSDT"]
DIRECTION = "down"
HORIZON = 30
INTERVAL = 15

COND_VARS = {
    "d_oi_event": "동시: 이벤트 버킷 OI 변화 (급감=청산 flush)",
    "funding_pre": "ex-ante: 직전 공표 funding rate",
    "d_oi_24h_pre": "ex-ante: 직전 24h OI buildup",
}

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


def main() -> None:
    if not STATE_PATH.exists():
        raise SystemExit(f"먼저 step4 를 실행하세요 — {STATE_PATH} 없음")

    print("micro frame 로드 중...")
    mf = pd.read_csv(MICRO_FRAME, usecols=USECOLS)
    mf["bucket_start"] = pd.to_datetime(mf["bucket_start"], utc=True)

    state = pd.read_csv(STATE_PATH)
    state["bucket_start"] = pd.to_datetime(state["bucket_start"], utc=True)
    state = state[["bucket_start", "d_oi_event", "funding_pre", "d_oi_24h_pre", "oi_end"]]

    vpin = pd.read_csv(VPIN_PATH)
    vpin["bucket_start"] = pd.to_datetime(vpin["bucket_start"], utc=True)
    vpin = vpin[["bucket_start", "toxicity"]]

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
        valid = valid.merge(vpin, on="bucket_start", how="left")
        # 상태변수 커버리지(2021-12~)로 이벤트·컨트롤 모두 제한 — 기간 mismatch 방지
        valid = valid.loc[valid["oi_end"].notna()].copy()

        doge_event = valid["doge_event"].fillna(False)
        control_ret = valid.loc[~doge_event, "target_forward_return"]
        events = valid.loc[doge_event].copy()

        # 커버리지 축소 후 headline 재현 (비교 기준선)
        base = welch_row(events["target_forward_return"], control_ret)
        base.update({"target": target, "cond_var": "ALL", "bucket": "all_events"})
        rows.append(base)
        print(f"\n=== {LEADER_LABEL(target)} (state 커버리지 2021-12~ 재현) ===")
        print(f"  ALL: delta={base['delta']*100:+.4f}%  t={base['t_stat']:+.2f}  n={base['n_event']}")

        # 정합성 진단: 청산 flush ↔ toxicity
        if target == TARGETS[0]:
            ev_diag = events.loc[events["d_oi_event"].notna() & events["toxicity"].notna()]
            sp = ev_diag["d_oi_event"].corr(ev_diag["toxicity"], method="spearman")
            oi_drop_share = float((ev_diag["d_oi_event"] < 0).mean())
            print(f"\n=== 진단 (DOGE down 이벤트 내) ===")
            print(f"  d_oi_event vs toxicity spearman: {sp:+.3f}")
            print(f"  이벤트 중 OI 감소 비중: {oi_drop_share:.1%}")

        for cond_var, desc in COND_VARS.items():
            ev = events.loc[events[cond_var].notna()].copy()
            if len(ev) < 60:
                print(f"  [{cond_var}] 표본 부족 ({len(ev)}), skip")
                continue
            q33, q67 = ev[cond_var].quantile([1 / 3, 2 / 3]).values
            groups = {
                "low":  ev.loc[ev[cond_var] <= q33],
                "mid":  ev.loc[(ev[cond_var] > q33) & (ev[cond_var] <= q67)],
                "high": ev.loc[ev[cond_var] > q67],
            }
            print(f"\n  [{cond_var}] {desc}")
            print(f"    3분위 컷: q33={q33:+.5f}  q67={q67:+.5f}")
            for label, g in groups.items():
                res = welch_row(g["target_forward_return"], control_ret)
                res.update({"target": target, "cond_var": cond_var, "bucket": label,
                            "cut_low": float(q33), "cut_high": float(q67),
                            "cond_mean": float(g[cond_var].mean())})
                rows.append(res)
                print(f"    {label:>4}: delta={res['delta']*100:+.4f}%  t={res['t_stat']:+.2f}  "
                      f"n={res['n_event']:>4}  win={res['win_rate']:.3f}  "
                      f"{cond_var}̄={res['cond_mean']:+.5f}")

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "leverage_conditional_summary.csv", index=False)
    print(f"\nsaved → {OUT_DIR / 'leverage_conditional_summary.csv'}")

    print("\n=== 해석 요약 ===")
    for target in TARGETS:
        for cond_var in COND_VARS:
            sub = out[(out["target"] == target) & (out["cond_var"] == cond_var)]
            lo = sub.loc[sub["bucket"] == "low", ["delta", "t_stat"]]
            hi = sub.loc[sub["bucket"] == "high", ["delta", "t_stat"]]
            if lo.empty or hi.empty:
                continue
            print(f"  {target} / {cond_var:>13}: "
                  f"low={lo['delta'].values[0]*100:+.4f}% (t={lo['t_stat'].values[0]:+.2f})  "
                  f"high={hi['delta'].values[0]*100:+.4f}% (t={hi['t_stat'].values[0]:+.2f})")


def LEADER_LABEL(target: str) -> str:
    return f"DOGE down → {target.replace('USDT', '')}"


if __name__ == "__main__":
    raise SystemExit(main())

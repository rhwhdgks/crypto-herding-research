"""Step 2 — DOGE informed-flow(VPIN-proxy) 로 조건부 lead-lag 분해.

확장방향 A 핵심 검정:
  headline = "DOGE down event → AVAX/ADA +30min 공동반응" (5y t=3.42, BH-FDR q=0.05 통과)
  질문     = 이 공동반응이 DOGE 의 informed trading(high toxicity)에 집중되는가?

방법:
  1. tick_lead_lag.build_lead_lag_frame 로 headline cell 을 그대로 재현
     (target=AVAX/ADA, leader=DOGE, direction=down) — 같은 session/control 정의 사용.
  2. step1 의 DOGE 15분 toxicity 를 bucket_start 로 join.
  3. DOGE down EVENT 버킷을 DOGE toxicity 3분위(low/mid/high)로 분할.
  4. 각 분위의 AVAX/ADA +30min 평균을 공통 control(doge_event=False)과 비교, Welch t.
     → 효과가 high-toxicity 에 monotonic 하게 집중되면 informed-driven,
       분위 무관하게 균일하면 noise herding 쪽 증거.

트랙 분리: baseline / lead_lag_matrix 산출물을 수정하지 않는다. 읽기만 함.
출력: experiments/informed_trading/outputs/conditional_leadlag_summary.csv
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
VPIN_PATH = PROJECT_ROOT / "experiments/informed_trading/outputs/doge_vpin_15m.csv"
OUT_DIR = PROJECT_ROOT / "experiments/informed_trading/outputs"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT"]
LEADER = "DOGEUSDT"
TARGETS = ["AVAXUSDT", "ADAUSDT"]  # BH-FDR q=0.05 통과 cell
DIRECTION = "down"
HORIZON = 30
INTERVAL = 15

USECOLS = [
    "symbol", "bucket_start", "event_label", f"forward_return_{HORIZON}m",
    "hour_utc", "is_target_session", "meets_trade_count",
    "bucket_return", "transaction_count", "herding_score",
]


def welch(event: pd.Series, control: pd.Series) -> dict:
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
    if not VPIN_PATH.exists():
        raise SystemExit(f"먼저 step1 을 실행하세요 — {VPIN_PATH} 없음")

    print("micro frame 로드 중...")
    mf = pd.read_csv(MICRO_FRAME, usecols=USECOLS)
    mf["bucket_start"] = pd.to_datetime(mf["bucket_start"], utc=True)
    print(f"  {len(mf):,} rows, {mf['symbol'].nunique()} symbols")

    vpin = pd.read_csv(VPIN_PATH)
    vpin["bucket_start"] = pd.to_datetime(vpin["bucket_start"], utc=True)
    vpin = vpin[["bucket_start", "toxicity", "vpin_50", "order_imbalance", "signed_imbalance_ratio"]]
    print(f"  vpin: {len(vpin):,} buckets, {vpin['bucket_start'].min()} ~ {vpin['bucket_start'].max()}")

    leader_prefix = LEADER.lower().replace("usdt", "")  # 'doge'
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
        valid = valid.merge(vpin, on="bucket_start", how="left")

        doge_event = valid[f"{leader_prefix}_event"].fillna(False)
        control_ret = valid.loc[~doge_event, "target_forward_return"]

        # --- (0) headline 재현 (전체 DOGE down event vs control) ---
        overall = welch(valid.loc[doge_event, "target_forward_return"], control_ret)
        overall.update({"target": target, "cond_var": "ALL", "bucket": "all_events"})
        rows.append(overall)
        print(f"\n=== {LEADER} down → {target} (5y 재현) ===")
        print(f"  ALL events: delta={overall['delta']*100:+.4f}%  t={overall['t_stat']:+.2f}  "
              f"n={overall['n_event']} (control {overall['n_control']})")

        # --- 조건부 분해: DOGE toxicity / vpin_50 3분위 ---
        for cond_var in ["toxicity", "vpin_50"]:
            ev = valid.loc[doge_event].copy()
            ev = ev.loc[ev[cond_var].notna()]
            if len(ev) < 30:
                print(f"  [{cond_var}] event 표본 부족 ({len(ev)}), skip")
                continue
            q33, q67 = ev[cond_var].quantile([1 / 3, 2 / 3]).values
            groups = {
                "low":  ev.loc[ev[cond_var] <= q33],
                "mid":  ev.loc[(ev[cond_var] > q33) & (ev[cond_var] <= q67)],
                "high": ev.loc[ev[cond_var] > q67],
            }
            print(f"  [{cond_var}] 3분위 컷: q33={q33:.4f}  q67={q67:.4f}")
            for label, g in groups.items():
                res = welch(g["target_forward_return"], control_ret)
                res.update({"target": target, "cond_var": cond_var, "bucket": label,
                            "cut_low": float(q33), "cut_high": float(q67),
                            "cond_mean": float(g[cond_var].mean())})
                rows.append(res)
                print(f"    {label:>4}: delta={res['delta']*100:+.4f}%  t={res['t_stat']:+.2f}  "
                      f"n={res['n_event']:>4}  win={res['win_rate']:.3f}  "
                      f"{cond_var}̄={res['cond_mean']:.4f}")

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "conditional_leadlag_summary.csv", index=False)
    print(f"\nsaved → {OUT_DIR / 'conditional_leadlag_summary.csv'}")

    # 요약 해석 힌트
    print("\n=== 해석 ===")
    for target in TARGETS:
        for cond_var in ["toxicity", "vpin_50"]:
            sub = out[(out["target"] == target) & (out["cond_var"] == cond_var)]
            if sub.empty:
                continue
            lo = sub.loc[sub["bucket"] == "low", "delta"]
            hi = sub.loc[sub["bucket"] == "high", "delta"]
            if lo.empty or hi.empty:
                continue
            lo_v, hi_v = lo.values[0], hi.values[0]
            verdict = "informed-driven(high>low)" if hi_v > lo_v else "noise/역방향(low≥high)"
            print(f"  {target} / {cond_var}: low={lo_v*100:+.4f}%  high={hi_v*100:+.4f}%  → {verdict}")


if __name__ == "__main__":
    raise SystemExit(main())

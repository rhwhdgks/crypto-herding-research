"""Step 10 — crowded × no_flush 규칙의 leader 확장 (SOL/ADA/XRP/AVAX).

질문:
  1. DOGE에서 발견한 "crowded funding × OI no-flush → 알트 공동반등" 구조가
     다른 leader의 레버리지 상태로도 성립하는가? (구조 일반화 = 요행 반박 강화)
  2. leader별 crowded funding 레짐 현황 — DOGE는 2024-12 이후 휴면인데,
     **지금 발화 가능한** leader 변형이 존재하는가?

방법: step5/6와 동일 framework, leader만 교체.
  - leader down micro-herding event (5y micro frame 재사용)
  - leader 자신의 funding_pre > 0.0001 & d_oi_event > leader별 tercile 컷
  - basket: {DOGE,AVAX,ADA,XRP,SOL} 5종 동일가중 +30min (leader 포함, 전 leader 공통)
  - control: 해당 leader 비이벤트 버킷
  - 레짐 현황: 최근 90일 crowded 버킷 비중 (funding은 2026-07 초까지 확장)

출력: experiments/informed_trading/outputs/leader_expansion_summary.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/jonghan/findalpha/herding")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from tick_lead_lag import _compute_difference_t_stat  # type: ignore
from run_leverage_candidate_tracker import build_futures_state  # type: ignore

MICRO_FRAME = PROJECT_ROOT / "outputs/tick/multi_asset_5y/lead_lag_matrix/intermediate/tick_micro_frame_15m.csv"
OUT_DIR = PROJECT_ROOT / "experiments/informed_trading/outputs"

LEADERS = ["SOLUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "DOGEUSDT"]  # DOGE는 기준 재현용
BASKET = ["DOGEUSDT", "AVAXUSDT", "ADAUSDT", "XRPUSDT", "SOLUSDT"]
HORIZON = 30
DEFAULT_FUNDING = 0.0001
STATE_START = "2021-12-01"
STATE_END = "2026-07-04"   # 레짐 현황까지 확인 (in-sample 셀 통계는 2026-04까지의 이벤트가 지배)
FS_CFG = {"base_url": "https://data.binance.vision/data/futures/um",
          "local_data_dir": "data/futures_archive"}

USECOLS = ["symbol", "bucket_start", "event_label", f"forward_return_{HORIZON}m",
           "is_target_session", "meets_trade_count"]


def welch(event: pd.Series, control: pd.Series) -> dict:
    event, control = event.dropna(), control.dropna()
    return {
        "n_event": int(len(event)), "n_control": int(len(control)),
        "delta": float(event.mean() - control.mean()) if len(event) and len(control) else np.nan,
        "t_stat": float(_compute_difference_t_stat(event, control)),
        "win_rate": float((event > 0).mean()) if len(event) else np.nan,
    }


def main() -> None:
    print("micro frame 로드 중...")
    mf = pd.read_csv(MICRO_FRAME, usecols=USECOLS)
    mf["bucket_start"] = pd.to_datetime(mf["bucket_start"], utc=True).dt.as_unit("ns")

    fwd_col = f"forward_return_{HORIZON}m"
    fwd_wide = mf.loc[mf["symbol"].isin(BASKET)].pivot_table(
        index="bucket_start", columns="symbol", values=fwd_col)
    fwd_wide["basket5"] = fwd_wide[BASKET].mean(axis=1)
    basket = fwd_wide[["basket5"]].reset_index()

    rows: list[dict] = []
    for leader in LEADERS:
        print(f"\n=== leader: {leader} ===")
        print("  futures state 빌드 (다운로드 포함)...")
        state = build_futures_state(leader, STATE_START, STATE_END, FS_CFG)
        state = state[["bucket_start", "funding_pre", "d_oi_event", "oi_end"]]

        lf = mf.loc[mf["symbol"] == leader,
                    ["bucket_start", "event_label", "is_target_session", "meets_trade_count"]].copy()
        lf = lf.merge(state, on="bucket_start", how="left").merge(basket, on="bucket_start", how="left")
        lf = lf.loc[lf["oi_end"].notna() & lf["basket5"].notna()].copy()
        # step5/6와 동일하게 valid universe = 세션 + 거래수 충족
        lf = lf.loc[lf["is_target_session"] & lf["meets_trade_count"]].copy()

        is_event = lf["event_label"] == "micro_herding_down"
        events = lf.loc[is_event & lf["funding_pre"].notna() & lf["d_oi_event"].notna()].copy()
        control = lf.loc[~is_event, "basket5"]

        base = welch(events["basket5"], control)
        flush_cut = float(events["d_oi_event"].quantile(1 / 3))
        cell_sel = (events["funding_pre"] > DEFAULT_FUNDING) & (events["d_oi_event"] > flush_cut)
        cell = welch(events.loc[cell_sel, "basket5"], control)

        # 레짐 현황: 최근 90일 crowded 비중 + 활동 기간
        recent = state.loc[state["bucket_start"] >= state["bucket_start"].max() - pd.Timedelta(days=90)]
        crowded_recent = float((recent["funding_pre"] > DEFAULT_FUNDING).mean())
        crowded_all = state.loc[state["funding_pre"] > DEFAULT_FUNDING, "bucket_start"]
        active_range = (f"{crowded_all.min().date()} ~ {crowded_all.max().date()}"
                        if len(crowded_all) else "없음")

        row = {"leader": leader, "flush_cut": flush_cut,
               "all_delta": base["delta"], "all_t": base["t_stat"], "all_n": base["n_event"],
               "cell_delta": cell["delta"], "cell_t": cell["t_stat"], "cell_n": cell["n_event"],
               "cell_win": cell["win_rate"],
               "crowded_share_90d": crowded_recent, "crowded_range": active_range}
        rows.append(row)
        print(f"  ALL down events : delta={base['delta']*100:+.4f}%  t={base['t_stat']:+.2f}  n={base['n_event']}")
        print(f"  crowded×no_flush: delta={cell['delta']*100:+.4f}%  t={cell['t_stat']:+.2f}  "
              f"n={cell['n_event']}  win={cell['win_rate'] if cell['n_event'] else float('nan'):.3f}")
        print(f"  레짐: 최근 90일 crowded {crowded_recent:.1%}  (crowded 기간: {active_range})")

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "leader_expansion_summary.csv", index=False)
    print(f"\nsaved → {OUT_DIR / 'leader_expansion_summary.csv'}")

    print("\n=== 요약 ===")
    for _, r in out.iterrows():
        status = "ACTIVE" if r["crowded_share_90d"] > 0.05 else "dormant"
        print(f"  {r['leader'].replace('USDT',''):>5}: cell {r['cell_delta']*100:+.4f}% "
              f"(t={r['cell_t']:+.2f}, n={int(r['cell_n'])})  |  최근 레짐 {status} "
              f"({r['crowded_share_90d']:.1%})")


if __name__ == "__main__":
    raise SystemExit(main())

"""Step 7 — "crowded × no_flush" 셀 robustness 스크리닝.

step6 발견: funding > 기본요율(crowded long) 상태에서 DOGE down event 가 OI flush 없이
발생하면 AVAX/ADA +30min 반등이 +0.16% (t≈2.3) — 첫 taker-fee 초과 ex-ante 셀.

주의: 다중비교 스캔 끝에 나온 셀이므로 승격 전 검증 필요. 이 step 은:
  1. 타깃 확장 — 전체 6개 타깃 (XRP/SOL 추가 + BTC/ETH falsification:
     BTC/ETH 는 알트 공동반응을 안 따라갔으므로 여기서도 약해야 정합적)
  2. 시간 반분할 — state 커버리지(2021-12~2026-04)를 반으로 나눠 부호/강도 유지 확인

출력: experiments/informed_trading/outputs/crowded_noflush_robustness.csv
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
TARGETS = ["AVAXUSDT", "ADAUSDT", "XRPUSDT", "SOLUSDT", "ETHUSDT", "BTCUSDT"]
DIRECTION = "down"
HORIZON = 30
INTERVAL = 15
DEFAULT_FUNDING = 0.0001

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
        "delta": float(event.mean() - control.mean()) if len(event) and len(control) else np.nan,
        "t_stat": float(_compute_difference_t_stat(event, control)),
        "win_rate": float((event > 0).mean()) if len(event) else np.nan,
    }


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
        events = valid.loc[doge_event & valid["funding_pre"].notna() & valid["d_oi_event"].notna()].copy()

        flush_cut = events["d_oi_event"].quantile(1 / 3)
        sel = (events["funding_pre"] > DEFAULT_FUNDING) & (events["d_oi_event"] > flush_cut)
        cell = events.loc[sel]
        control_ret = valid.loc[~doge_event, "target_forward_return"]

        # 전체 기간
        res = welch_row(cell["target_forward_return"], control_ret)
        res.update({"target": target, "window": "full"})
        rows.append(res)

        # 시간 반분할 (이벤트·컨트롤 모두 같은 half 로 제한)
        midpoint = valid["bucket_start"].min() + (valid["bucket_start"].max() - valid["bucket_start"].min()) / 2
        for half, mask in [("h1", valid["bucket_start"] < midpoint), ("h2", valid["bucket_start"] >= midpoint)]:
            v_half = valid.loc[mask]
            e_half = cell.loc[cell["bucket_start"] < midpoint] if half == "h1" else cell.loc[cell["bucket_start"] >= midpoint]
            c_half = v_half.loc[~v_half["doge_event"].fillna(False), "target_forward_return"]
            res = welch_row(e_half["target_forward_return"], c_half)
            res.update({"target": target, "window": half})
            rows.append(res)

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "crowded_noflush_robustness.csv", index=False)

    print("\n=== crowded × no_flush 셀 — 타깃 확장 + 반분할 ===")
    print(f"(midpoint 분할, ALT 4종은 지지 기대 / BTC·ETH 는 약해야 정합)")
    for target in TARGETS:
        sub = out[out["target"] == target]
        f = sub[sub["window"] == "full"].iloc[0]
        h1 = sub[sub["window"] == "h1"].iloc[0]
        h2 = sub[sub["window"] == "h2"].iloc[0]
        tshort = target.replace("USDT", "")
        print(f"  {tshort:>5}: full delta={f['delta']*100:+.4f}% t={f['t_stat']:+.2f} n={f['n_event']:>3}  ||  "
              f"H1 t={h1['t_stat']:+.2f} (n={h1['n_event']})  H2 t={h2['t_stat']:+.2f} (n={h2['n_event']})")
    print(f"\nsaved → {OUT_DIR / 'crowded_noflush_robustness.csv'}")


if __name__ == "__main__":
    raise SystemExit(main())

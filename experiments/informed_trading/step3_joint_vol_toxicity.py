"""Step 3 — Vol regime × DOGE toxicity 결합 분해 (confound 해소).

배경 (두 결과의 긴장):
  - step6_vol_regime:  headline 효과는 HIGH vol에서 강함 (t=2.76 vs 2.06)
  - step2 (informed):  headline 효과는 LOW toxicity에서 강함 (t=2.4~3.4 vs 1.0~1.4)

질문:
  1. toxicity와 vol regime이 상관(교락)인가, 독립 축인가?
  2. 독립이라면 "HIGH vol × LOW toxicity" = '변동성은 크지만 flow는 balanced'
     = 유동성 dip 프로파일에 효과가 집중되는가? (더 날카로운 이벤트 정의 후보)

방법:
  - vol regime: step6와 동일 — BTC bucket_return rolling 96-bar std, 전체표본 median split
  - toxicity tercile: step2와 동일 — DOGE down event 표본 내 3분위 (전체 이벤트 기준 고정 컷)
  - control: 같은 vol regime 내 non-event 버킷 (regime별 baseline 차이 제거)
  - 각 (vol × tox) 셀의 target +30min forward return 을 regime-matched control과 Welch t 비교

트랙 분리: 읽기 전용 (micro frame / vpin csv). baseline·matrix 미변경.
출력: experiments/informed_trading/outputs/joint_vol_toxicity_summary.csv
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
TARGETS = ["AVAXUSDT", "ADAUSDT"]
DIRECTION = "down"
HORIZON = 30
INTERVAL = 15
ROLLING_BARS = 96  # step6와 동일: 24h × 4 buckets/hour

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
    print("micro frame 로드 중...")
    mf = pd.read_csv(MICRO_FRAME, usecols=USECOLS)
    mf["bucket_start"] = pd.to_datetime(mf["bucket_start"], utc=True)

    vpin = pd.read_csv(VPIN_PATH)
    vpin["bucket_start"] = pd.to_datetime(vpin["bucket_start"], utc=True)
    vpin = vpin[["bucket_start", "toxicity", "vpin_50"]]

    # --- vol regime (step6와 동일 정의) ---
    btc = mf.loc[mf["symbol"] == "BTCUSDT"].sort_values("bucket_start").copy()
    btc["vol_24h"] = btc["bucket_return"].rolling(ROLLING_BARS).std()
    median_vol = btc["vol_24h"].median()
    btc["vol_regime"] = np.where(btc["vol_24h"] >= median_vol, "high", "low")
    regime_frame = btc[["bucket_start", "vol_24h", "vol_regime"]]
    print(f"BTC 24h-vol median: {median_vol:.6f}")

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
        valid = valid.merge(regime_frame, on="bucket_start", how="left")
        valid = valid.loc[valid["vol_regime"].notna()]

        doge_event = valid["doge_event"].fillna(False)
        events = valid.loc[doge_event & valid["toxicity"].notna()].copy()

        # --- (1) 교락 진단 ---
        if target == TARGETS[0]:
            corr_all = valid["toxicity"].corr(valid["vol_24h"], method="spearman")
            corr_ev = events["toxicity"].corr(events["vol_24h"], method="spearman")
            print(f"\n=== 교락 진단 (toxicity vs BTC vol_24h, spearman) ===")
            print(f"  전체 버킷: {corr_all:+.3f}   |   DOGE down 이벤트 내: {corr_ev:+.3f}")

        # toxicity tercile: step2와 동일 — 전체 이벤트 표본 기준 고정 컷
        q33, q67 = events["toxicity"].quantile([1 / 3, 2 / 3]).values
        events["tox_bucket"] = np.where(
            events["toxicity"] <= q33, "low",
            np.where(events["toxicity"] <= q67, "mid", "high"),
        )

        if target == TARGETS[0]:
            ct = pd.crosstab(events["vol_regime"], events["tox_bucket"], normalize="index")
            print(f"\n=== 이벤트 분포: vol regime × tox tercile (행 비율) ===")
            print(ct.round(3).to_string())

        print(f"\n=== {target}: vol regime × toxicity 분해 ===")
        for regime in ["low", "high"]:
            reg_control = valid.loc[
                (~doge_event) & (valid["vol_regime"] == regime), "target_forward_return"
            ]
            # regime 전체 (step6 재현 대응치, regime-matched control 기준)
            reg_events = events.loc[events["vol_regime"] == regime]
            res = welch_row(reg_events["target_forward_return"], reg_control)
            res.update({"target": target, "vol_regime": regime, "tox_bucket": "ALL"})
            rows.append(res)
            print(f"  [{regime:>4} vol] ALL : delta={res['delta']*100:+.4f}%  t={res['t_stat']:+.2f}  n={res['n_event']:>4}")

            for tox in ["low", "mid", "high"]:
                cell = reg_events.loc[reg_events["tox_bucket"] == tox]
                res = welch_row(cell["target_forward_return"], reg_control)
                res.update({"target": target, "vol_regime": regime, "tox_bucket": tox})
                rows.append(res)
                print(f"  [{regime:>4} vol] {tox:>4}: delta={res['delta']*100:+.4f}%  t={res['t_stat']:+.2f}  "
                      f"n={res['n_event']:>4}  win={res['win_rate']:.3f}" if res["n_event"] else
                      f"  [{regime:>4} vol] {tox:>4}: (표본 없음)")

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "joint_vol_toxicity_summary.csv", index=False)
    print(f"\nsaved → {OUT_DIR / 'joint_vol_toxicity_summary.csv'}")

    # --- 핵심 셀 요약 ---
    print("\n=== 핵심 비교: high-vol × low-tox (유동성 dip 프로파일) ===")
    for target in TARGETS:
        sub = out[(out["target"] == target)]
        hv_lt = sub[(sub["vol_regime"] == "high") & (sub["tox_bucket"] == "low")]
        hv_ht = sub[(sub["vol_regime"] == "high") & (sub["tox_bucket"] == "high")]
        lv_lt = sub[(sub["vol_regime"] == "low") & (sub["tox_bucket"] == "low")]
        if hv_lt.empty:
            continue
        a, b, c = hv_lt.iloc[0], hv_ht.iloc[0], lv_lt.iloc[0]
        print(f"  {target}:")
        print(f"    high-vol × low-tox : delta={a['delta']*100:+.4f}%  t={a['t_stat']:+.2f}  n={a['n_event']}")
        print(f"    high-vol × high-tox: delta={b['delta']*100:+.4f}%  t={b['t_stat']:+.2f}  n={b['n_event']}")
        print(f"    low-vol  × low-tox : delta={c['delta']*100:+.4f}%  t={c['t_stat']:+.2f}  n={c['n_event']}")


if __name__ == "__main__":
    raise SystemExit(main())

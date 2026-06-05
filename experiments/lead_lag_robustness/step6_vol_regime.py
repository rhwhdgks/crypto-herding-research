"""Vol regime split — 논문 가설 (저변동성에서 herding 강함) 검증.

방법:
1. BTC를 시장 proxy로 사용
2. BTC bucket_return의 rolling 24h (96 bar) std → vol regime
3. Median split: low vs high vol regime
4. 각 regime에서 lead-lag matrix 재계산
5. 비교: 어느 regime에서 더 강한 신호?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/jonghan/findalpha/herding")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tick_lead_lag import run_lead_lag_matrix  # type: ignore

MICRO_FRAME = PROJECT_ROOT / "outputs/tick/multi_asset_5y/lead_lag_matrix/intermediate/tick_micro_frame_15m.csv"
OUT_DIR = PROJECT_ROOT / "experiments/lead_lag_robustness/outputs"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT"]
ROLLING_BARS = 96  # 24 hours × 4 buckets/hour


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mf = pd.read_csv(MICRO_FRAME)
    mf["bucket_start"] = pd.to_datetime(mf["bucket_start"], utc=True)

    # BTC vol proxy
    btc = mf.loc[mf["symbol"] == "BTCUSDT"].sort_values("bucket_start").copy()
    btc["abs_ret"] = btc["bucket_return"].abs()
    btc["vol_24h"] = btc["bucket_return"].rolling(ROLLING_BARS).std()
    median_vol = btc["vol_24h"].median()
    print(f"BTC 24h-vol median: {median_vol:.6f}")

    # Build regime mask (bucket_start → regime)
    btc["regime"] = np.where(btc["vol_24h"] >= median_vol, "high", "low")
    regime_map = btc.set_index("bucket_start")["regime"]

    mf["regime"] = mf["bucket_start"].map(regime_map)
    mf_low = mf[mf["regime"] == "low"].copy()
    mf_high = mf[mf["regime"] == "high"].copy()
    print(f"low vol bars: {len(mf_low):,}")
    print(f"high vol bars: {len(mf_high):,}")

    print("\nrunning matrix on low vol regime...")
    m_low = run_lead_lag_matrix(mf_low, SYMBOLS, 15, 30, ["up", "down"])
    print(f"  low cells: {len(m_low)}")
    print("running matrix on high vol regime...")
    m_high = run_lead_lag_matrix(mf_high, SYMBOLS, 15, 30, ["up", "down"])
    print(f"  high cells: {len(m_high)}")

    m_low.to_csv(OUT_DIR / "matrix_5y_lowvol.csv", index=False)
    m_high.to_csv(OUT_DIR / "matrix_5y_highvol.csv", index=False)

    merged = m_low.merge(m_high, on=["target", "leader", "direction"], suffixes=("_low", "_high"))
    merged["same_sign"] = np.sign(merged["delta_mean_return_low"]) == np.sign(merged["delta_mean_return_high"])
    merged.to_csv(OUT_DIR / "vol_regime_comparison.csv", index=False)

    n = len(merged)
    n_strong_low = int((merged["delta_t_stat_low"].abs() >= 2.0).sum())
    n_strong_high = int((merged["delta_t_stat_high"].abs() >= 2.0).sum())
    n_strong_low_15 = int((merged["delta_t_stat_low"].abs() >= 1.5).sum())
    n_strong_high_15 = int((merged["delta_t_stat_high"].abs() >= 1.5).sum())

    print()
    print(f"=== Vol regime split (n={n} cells) ===")
    print(f"|t|>=1.5 in low vol:           {n_strong_low_15}/{n}")
    print(f"|t|>=1.5 in high vol:          {n_strong_high_15}/{n}")
    print(f"|t|>=2.0 in low vol:           {n_strong_low}/{n}")
    print(f"|t|>=2.0 in high vol:          {n_strong_high}/{n}")

    # Compare top edges: are they stronger in low or high vol?
    full = pd.read_csv(PROJECT_ROOT / "outputs/tick/multi_asset_5y/lead_lag_matrix/lead_lag_matrix_summary.csv")
    full = full.reindex(full["delta_t_stat"].abs().sort_values(ascending=False).index).head(10)
    print("\n=== Top 10 5y full-period edges — vol regime breakdown ===")
    for _, fr in full.iterrows():
        match = merged[
            (merged["target"] == fr["target"])
            & (merged["leader"] == fr["leader"])
            & (merged["direction"] == fr["direction"])
        ]
        if match.empty:
            continue
        m = match.iloc[0]
        leader_short = fr["leader"].replace("USDT", "")
        target_short = fr["target"].replace("USDT", "")
        print(
            f"{leader_short:>5} {fr['direction']:>4} → {target_short:<5} | "
            f"full t={fr['delta_t_stat']:+.2f}  ||  "
            f"LOW t={m['delta_t_stat_low']:+.2f}  HIGH t={m['delta_t_stat_high']:+.2f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())

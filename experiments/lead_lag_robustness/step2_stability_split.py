"""Lead-lag stability split — 11개월 데이터를 전반/후반으로 나눠 매트릭스 재계산.

목적: top edges가 양 기간에서 같은 부호 + 비슷한 magnitude로 나오는지.
같은 방향이면 robust signal, 다르면 overfitting / time-decay.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/jonghan/findalpha/herding")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tick_lead_lag import run_lead_lag_matrix  # type: ignore

MICRO_FRAME = PROJECT_ROOT / "outputs/tick/multi_asset_365d/lead_lag_matrix/intermediate/tick_micro_frame_15m.csv"
OUT_DIR = PROJECT_ROOT / "experiments/lead_lag_robustness/outputs"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT"]
INTERVAL = 15
HORIZON = 30


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mf = pd.read_csv(MICRO_FRAME)
    mf["bucket_start"] = pd.to_datetime(mf["bucket_start"], utc=True)

    # Split point: midpoint
    t_min, t_max = mf["bucket_start"].min(), mf["bucket_start"].max()
    midpoint = t_min + (t_max - t_min) / 2
    print(f"range: {t_min} ~ {t_max}")
    print(f"split: first half ends at {midpoint}")

    first_half = mf.loc[mf["bucket_start"] < midpoint].copy()
    second_half = mf.loc[mf["bucket_start"] >= midpoint].copy()
    print(f"first_half rows: {len(first_half):,}")
    print(f"second_half rows: {len(second_half):,}")

    print("\nrunning matrix on first half...")
    m1 = run_lead_lag_matrix(first_half, SYMBOLS, INTERVAL, HORIZON, ["up", "down"])
    print(f"first half cells: {len(m1)}")
    print("\nrunning matrix on second half...")
    m2 = run_lead_lag_matrix(second_half, SYMBOLS, INTERVAL, HORIZON, ["up", "down"])
    print(f"second half cells: {len(m2)}")

    m1.to_csv(OUT_DIR / "matrix_first_half.csv", index=False)
    m2.to_csv(OUT_DIR / "matrix_second_half.csv", index=False)

    # Merge for comparison
    merged = m1.merge(
        m2,
        on=["target", "leader", "direction"],
        suffixes=("_h1", "_h2"),
    )
    merged["same_sign"] = (
        np.sign(merged["delta_mean_return_h1"]) == np.sign(merged["delta_mean_return_h2"])
    )
    merged["both_strong"] = (
        (merged["delta_t_stat_h1"].abs() >= 1.5) & (merged["delta_t_stat_h2"].abs() >= 1.5)
    )
    merged["both_strong_same_sign"] = merged["both_strong"] & merged["same_sign"]
    merged["sign_flip"] = ~merged["same_sign"]

    merged.to_csv(OUT_DIR / "stability_split_comparison.csv", index=False)

    # Stats
    n = len(merged)
    n_same = int(merged["same_sign"].sum())
    n_both_strong = int(merged["both_strong"].sum())
    n_both_strong_same = int(merged["both_strong_same_sign"].sum())
    n_h1_strong = int((merged["delta_t_stat_h1"].abs() >= 1.5).sum())
    n_h2_strong = int((merged["delta_t_stat_h2"].abs() >= 1.5).sum())

    print(f"\n=== Stability split results (n={n} cells) ===")
    print(f"Same sign in both halves:           {n_same}/{n} ({n_same/n:.1%})")
    print(f"|t|>=1.5 in first half:             {n_h1_strong}/{n}")
    print(f"|t|>=1.5 in second half:            {n_h2_strong}/{n}")
    print(f"|t|>=1.5 in BOTH halves:            {n_both_strong}/{n}")
    print(f"|t|>=1.5 in both AND same sign:     {n_both_strong_same}/{n}")

    # Track key edges from full-period top
    print("\n=== Top 5 full-period edges — stability check ===")
    full = pd.read_csv(
        PROJECT_ROOT / "outputs/tick/multi_asset_365d/lead_lag_matrix/lead_lag_matrix_summary.csv"
    )
    full = full.reindex(full["delta_t_stat"].abs().sort_values(ascending=False).index).head(10)
    for _, fr in full.iterrows():
        match = merged[
            (merged["target"] == fr["target"])
            & (merged["leader"] == fr["leader"])
            & (merged["direction"] == fr["direction"])
        ]
        if match.empty:
            continue
        m = match.iloc[0]
        same = "✓" if m["same_sign"] else "✗"
        leader_short = fr["leader"].replace("USDT", "")
        target_short = fr["target"].replace("USDT", "")
        print(
            f"{leader_short:>5} {fr['direction']:>4} → {target_short:<5} | "
            f"full t={fr['delta_t_stat']:+.2f}  ||  "
            f"H1 t={m['delta_t_stat_h1']:+.2f}  H2 t={m['delta_t_stat_h2']:+.2f}  "
            f"same_sign={same}"
        )

    print(f"\nsaved comparison to: {OUT_DIR/'stability_split_comparison.csv'}")


if __name__ == "__main__":
    raise SystemExit(main())

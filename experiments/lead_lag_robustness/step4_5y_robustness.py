"""5년 데이터 robustness 검증 — 다중비교 보정 + stability split.

1년 결과와 직접 비교 가능하도록 같은 framework로 재실행.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path("/home/jonghan/findalpha/herding")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from tick_lead_lag import run_lead_lag_matrix  # type: ignore

INPUT_5Y = PROJECT_ROOT / "outputs/tick/multi_asset_5y/lead_lag_matrix/lead_lag_matrix_summary.csv"
MICRO_FRAME_5Y = PROJECT_ROOT / "outputs/tick/multi_asset_5y/lead_lag_matrix/intermediate/tick_micro_frame_15m.csv"
OUT_DIR = PROJECT_ROOT / "experiments/lead_lag_robustness/outputs"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT"]


def benjamini_hochberg(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    passed = ranked <= thresholds
    if not passed.any():
        return np.zeros(n, dtype=bool)
    max_idx = int(np.where(passed)[0].max())
    rejected_mask = np.zeros(n, dtype=bool)
    rejected_mask[order[: max_idx + 1]] = True
    return rejected_mask


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # === 1. Multiple comparison correction ===
    df = pd.read_csv(INPUT_5Y)
    print(f"5y matrix loaded: {len(df)} cells")

    t_stats = df["delta_t_stat"].values
    p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(t_stats)))
    df["p_value_normal_approx"] = p_values
    n_tests = len(df)
    bonf_alpha = 0.05 / n_tests

    df["bonferroni_05"] = df["p_value_normal_approx"] < bonf_alpha
    df["bh_fdr_05"] = benjamini_hochberg(p_values, alpha=0.05)
    df["bh_fdr_10"] = benjamini_hochberg(p_values, alpha=0.10)
    df["bh_fdr_20"] = benjamini_hochberg(p_values, alpha=0.20)

    df.sort_values("p_value_normal_approx").to_csv(OUT_DIR / "multiple_comparison_5y.csv", index=False)

    print()
    print(f"=== 5y Multiple comparison (n_tests={n_tests}) ===")
    print(f"Uncorrected p<0.05:        {int((df['p_value_normal_approx']<0.05).sum())} cells")
    print(f"Uncorrected p<0.01:        {int((df['p_value_normal_approx']<0.01).sum())} cells")
    print(f"Bonferroni α=0.05 (p<{bonf_alpha:.5f}): {int(df['bonferroni_05'].sum())} cells")
    print(f"BH-FDR q=0.05:             {int(df['bh_fdr_05'].sum())} cells")
    print(f"BH-FDR q=0.10:             {int(df['bh_fdr_10'].sum())} cells")
    print(f"BH-FDR q=0.20:             {int(df['bh_fdr_20'].sum())} cells")

    print()
    print("=== Top 10 by |t-stat| (5y) ===")
    cols = ["leader", "target", "direction", "delta_mean_return", "delta_t_stat",
            "p_value_normal_approx", "bonferroni_05", "bh_fdr_05", "bh_fdr_10", "bh_fdr_20"]
    top10 = df.reindex(df["delta_t_stat"].abs().sort_values(ascending=False).index).head(10)
    print(top10[cols].to_string(index=False))

    # === 2. Stability split ===
    print("\n=== 5y Stability split ===")
    mf = pd.read_csv(MICRO_FRAME_5Y)
    mf["bucket_start"] = pd.to_datetime(mf["bucket_start"], utc=True)
    t_min, t_max = mf["bucket_start"].min(), mf["bucket_start"].max()
    midpoint = t_min + (t_max - t_min) / 2
    print(f"range: {t_min} ~ {t_max}")
    print(f"split at: {midpoint}")

    h1 = mf.loc[mf["bucket_start"] < midpoint].copy()
    h2 = mf.loc[mf["bucket_start"] >= midpoint].copy()
    print(f"h1: {len(h1):,} rows, h2: {len(h2):,} rows")

    print("computing matrix on h1 (~2.5y)...")
    m1 = run_lead_lag_matrix(h1, SYMBOLS, 15, 30, ["up", "down"])
    print(f"  h1 cells: {len(m1)}")
    print("computing matrix on h2 (~2.5y)...")
    m2 = run_lead_lag_matrix(h2, SYMBOLS, 15, 30, ["up", "down"])
    print(f"  h2 cells: {len(m2)}")

    m1.to_csv(OUT_DIR / "matrix_5y_h1.csv", index=False)
    m2.to_csv(OUT_DIR / "matrix_5y_h2.csv", index=False)

    merged = m1.merge(m2, on=["target", "leader", "direction"], suffixes=("_h1", "_h2"))
    merged["same_sign"] = np.sign(merged["delta_mean_return_h1"]) == np.sign(merged["delta_mean_return_h2"])
    merged["both_strong"] = (merged["delta_t_stat_h1"].abs() >= 1.5) & (merged["delta_t_stat_h2"].abs() >= 1.5)
    merged["both_strong_same_sign"] = merged["both_strong"] & merged["same_sign"]
    merged.to_csv(OUT_DIR / "stability_split_5y.csv", index=False)

    n = len(merged)
    print()
    print(f"=== 5y Stability split (n={n} cells) ===")
    print(f"Same sign in both halves:           {int(merged['same_sign'].sum())}/{n} ({merged['same_sign'].mean():.1%})")
    print(f"|t|>=1.5 in h1:                     {int((merged['delta_t_stat_h1'].abs()>=1.5).sum())}/{n}")
    print(f"|t|>=1.5 in h2:                     {int((merged['delta_t_stat_h2'].abs()>=1.5).sum())}/{n}")
    print(f"|t|>=1.5 in BOTH:                   {int(merged['both_strong'].sum())}/{n}")
    print(f"|t|>=1.5 in both AND same sign:     {int(merged['both_strong_same_sign'].sum())}/{n}")
    print(f"|t|>=2.0 in both AND same sign:     "
          f"{int(((merged['delta_t_stat_h1'].abs()>=2.0)&(merged['delta_t_stat_h2'].abs()>=2.0)&merged['same_sign']).sum())}/{n}")

    print("\n=== Top 10 full-period — 5y stability check ===")
    full = pd.read_csv(INPUT_5Y)
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


if __name__ == "__main__":
    raise SystemExit(main())

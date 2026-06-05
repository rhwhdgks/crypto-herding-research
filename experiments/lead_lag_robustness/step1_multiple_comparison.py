"""다중비교 보정 — 84-cell lead-lag matrix의 t-stat에 BH-FDR / Bonferroni 적용.

입력: outputs/tick/multi_asset_365d/lead_lag_matrix/lead_lag_matrix_summary.csv (84행)
출력: experiments/lead_lag_robustness/outputs/multiple_comparison_results.csv
      + 콘솔에 핵심 결과 요약
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

INPUT = Path(
    "/home/jonghan/findalpha/herding/outputs/tick/multi_asset_365d/lead_lag_matrix/lead_lag_matrix_summary.csv"
)
OUT_DIR = Path("/home/jonghan/findalpha/herding/experiments/lead_lag_robustness/outputs")


def benjamini_hochberg(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    """BH-FDR procedure. Returns boolean array — True if rejected (significant)."""
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

    df = pd.read_csv(INPUT)
    print(f"loaded {len(df)} cells from {INPUT.name}")

    # Compute two-sided p-value from t-stat using normal approximation (large samples ~700-1000)
    t_stats = df["delta_t_stat"].values
    p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(t_stats)))
    df["p_value_normal_approx"] = p_values

    n_tests = len(df)
    bonferroni_alpha_05 = 0.05 / n_tests
    bonferroni_alpha_10 = 0.10 / n_tests

    df["bonferroni_05"] = df["p_value_normal_approx"] < bonferroni_alpha_05
    df["bonferroni_10"] = df["p_value_normal_approx"] < bonferroni_alpha_10
    df["bh_fdr_05"] = benjamini_hochberg(p_values, alpha=0.05)
    df["bh_fdr_10"] = benjamini_hochberg(p_values, alpha=0.10)
    df["bh_fdr_20"] = benjamini_hochberg(p_values, alpha=0.20)

    out_path = OUT_DIR / "multiple_comparison_results.csv"
    df.sort_values("p_value_normal_approx").to_csv(out_path, index=False)
    print(f"saved: {out_path}")

    # Summary
    n_uncorrected_05 = int((df["p_value_normal_approx"] < 0.05).sum())
    n_uncorrected_10 = int((df["p_value_normal_approx"] < 0.10).sum())
    n_bonf_05 = int(df["bonferroni_05"].sum())
    n_bonf_10 = int(df["bonferroni_10"].sum())
    n_bh_05 = int(df["bh_fdr_05"].sum())
    n_bh_10 = int(df["bh_fdr_10"].sum())
    n_bh_20 = int(df["bh_fdr_20"].sum())

    print()
    print(f"=== Multiple comparison correction (n_tests = {n_tests}) ===")
    print(f"Uncorrected p<0.05:        {n_uncorrected_05} cells")
    print(f"Uncorrected p<0.10:        {n_uncorrected_10} cells")
    print(f"Bonferroni α=0.05 (p<{bonferroni_alpha_05:.5f}): {n_bonf_05} cells")
    print(f"Bonferroni α=0.10 (p<{bonferroni_alpha_10:.5f}): {n_bonf_10} cells")
    print(f"BH-FDR q=0.05:             {n_bh_05} cells")
    print(f"BH-FDR q=0.10:             {n_bh_10} cells")
    print(f"BH-FDR q=0.20:             {n_bh_20} cells")
    print()
    print("=== Top 5 by |t-stat| ===")
    cols = ["leader", "target", "direction", "delta_mean_return", "delta_t_stat",
            "p_value_normal_approx", "bonferroni_05", "bh_fdr_05", "bh_fdr_10", "bh_fdr_20"]
    top5 = df.reindex(df["delta_t_stat"].abs().sort_values(ascending=False).index).head(5)
    print(top5[cols].to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())

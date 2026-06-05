"""시기 구분 lead-lag matrix — 3-period split (winter / recovery / current).

P1: 2021-05 ~ 2022-12 (2021 bull 후반 + 2022 crypto winter)
P2: 2023-01 ~ 2024-06 (recovery + halving 전후)
P3: 2024-07 ~ 2026-05 (post-halving / current bull)
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

PERIODS = [
    ("P1_winter", "2021-05-01", "2022-12-31"),
    ("P2_recovery", "2023-01-01", "2024-06-30"),
    ("P3_current", "2024-07-01", "2026-05-31"),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mf = pd.read_csv(MICRO_FRAME)
    mf["bucket_start"] = pd.to_datetime(mf["bucket_start"], utc=True)

    results = {}
    for name, start, end in PERIODS:
        s = pd.Timestamp(start, tz="UTC")
        e = pd.Timestamp(end, tz="UTC")
        sub = mf.loc[(mf["bucket_start"] >= s) & (mf["bucket_start"] <= e)].copy()
        print(f"\n=== {name} ({start} ~ {end}) ===")
        print(f"rows: {len(sub):,}")
        m = run_lead_lag_matrix(sub, SYMBOLS, 15, 30, ["up", "down"])
        results[name] = m
        m.to_csv(OUT_DIR / f"matrix_5y_{name}.csv", index=False)
        n_strong = int((m["delta_t_stat"].abs() >= 1.5).sum())
        n_strong20 = int((m["delta_t_stat"].abs() >= 2.0).sum())
        print(f"  |t|>=1.5: {n_strong}/{len(m)},  |t|>=2.0: {n_strong20}/{len(m)}")

    # Compare top 5y edges across periods
    full = pd.read_csv(PROJECT_ROOT / "outputs/tick/multi_asset_5y/lead_lag_matrix/lead_lag_matrix_summary.csv")
    full = full.reindex(full["delta_t_stat"].abs().sort_values(ascending=False).index).head(10)

    print("\n=== Top 10 5y edges — period breakdown ===")
    print(f"{'Edge':<28} {'5y t':>7} {'P1 t':>7} {'P2 t':>7} {'P3 t':>7}")
    rows: list[dict] = []
    for _, fr in full.iterrows():
        leader_short = fr["leader"].replace("USDT", "")
        target_short = fr["target"].replace("USDT", "")
        edge_label = f"{leader_short} {fr['direction']:>4} → {target_short}"
        period_t = {}
        for name, _, _ in PERIODS:
            match = results[name][
                (results[name]["target"] == fr["target"])
                & (results[name]["leader"] == fr["leader"])
                & (results[name]["direction"] == fr["direction"])
            ]
            if not match.empty:
                period_t[name] = float(match.iloc[0]["delta_t_stat"])
            else:
                period_t[name] = float("nan")
        print(
            f"{edge_label:<28} {fr['delta_t_stat']:+7.2f} "
            f"{period_t['P1_winter']:+7.2f} {period_t['P2_recovery']:+7.2f} {period_t['P3_current']:+7.2f}"
        )
        rows.append({
            "edge": edge_label,
            "leader": fr["leader"],
            "target": fr["target"],
            "direction": fr["direction"],
            "t_5y": fr["delta_t_stat"],
            "t_P1_winter": period_t["P1_winter"],
            "t_P2_recovery": period_t["P2_recovery"],
            "t_P3_current": period_t["P3_current"],
        })

    pd.DataFrame(rows).to_csv(OUT_DIR / "period_top_edges_breakdown.csv", index=False)


if __name__ == "__main__":
    raise SystemExit(main())

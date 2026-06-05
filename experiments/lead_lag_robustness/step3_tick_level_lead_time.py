"""Tick-level lead time 측정 — DOGE event 발생 시점 기준 AVAX 평균 반응 curve.

목적: lead-lag matrix의 +0.0914% (DOGE down → AVAX +30min)이
- t=0 이전부터 이미 같이 움직이면 → 공통 shock (lead-lag ❌)
- t=0 직후 단조증가면 → 진짜 microstructure lead-lag (✓)

방법:
1. AVAX, DOGE 1분봉 log return 시계열
2. DOGE micro_herding_down event 875건의 bucket_start를 anchor
3. 각 anchor 기준 [-60min, +60min] window의 AVAX cumulative return
4. 모든 event 평균 → event-triggered average response curve
5. + Cross-correlation function (CCF) for ±60min lags
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("/home/jonghan/findalpha/herding/data")
MICRO_FRAME = Path(
    "/home/jonghan/findalpha/herding/outputs/tick/multi_asset_365d/lead_lag_matrix/intermediate/tick_micro_frame_15m.csv"
)
OUT_DIR = Path("/home/jonghan/findalpha/herding/experiments/lead_lag_robustness/outputs")

WINDOW_MINUTES_BEFORE = 60
WINDOW_MINUTES_AFTER = 60


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load 1m closes
    avax = pd.read_parquet(DATA_DIR / "AVAXUSDT_1m.parquet")[["timestamp", "close"]]
    doge = pd.read_parquet(DATA_DIR / "DOGEUSDT_1m.parquet")[["timestamp", "close"]]
    for d in (avax, doge):
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    df = avax.merge(doge, on="timestamp", how="inner", suffixes=("_avax", "_doge"))
    df = df.set_index("timestamp").sort_index()
    df = df.loc[pd.Timestamp("2025-05-08", tz="UTC"):pd.Timestamp("2026-04-09", tz="UTC")]
    df = df.rename(columns={"close_avax": "avax", "close_doge": "doge"}).dropna()
    df["avax_log_ret"] = np.log(df["avax"]).diff()
    df["doge_log_ret"] = np.log(df["doge"]).diff()
    print(f"merged 1m: {len(df):,} rows ({df.index.min()} ~ {df.index.max()})")

    # 2) DOGE down events
    mf = pd.read_csv(MICRO_FRAME)
    mf["bucket_start"] = pd.to_datetime(mf["bucket_start"], utc=True)
    events = mf.loc[
        (mf["symbol"] == "DOGEUSDT") & (mf["event_label"] == "micro_herding_down"),
        "bucket_start",
    ]
    events = pd.to_datetime(events.values, utc=True)
    events = events[(events >= df.index.min()) & (events <= df.index.max() - pd.Timedelta(minutes=WINDOW_MINUTES_AFTER))]
    print(f"DOGE down events in window: {len(events)}")

    # 3) Event-triggered windows
    avax_ret = df["avax_log_ret"]
    doge_ret = df["doge_log_ret"]

    offsets = np.arange(-WINDOW_MINUTES_BEFORE, WINDOW_MINUTES_AFTER + 1)
    avax_matrix = np.full((len(events), len(offsets)), np.nan)
    doge_matrix = np.full((len(events), len(offsets)), np.nan)

    for i, evt in enumerate(events):
        for j, off in enumerate(offsets):
            target_ts = evt + pd.Timedelta(minutes=int(off))
            if target_ts in avax_ret.index:
                avax_matrix[i, j] = avax_ret.loc[target_ts]
                doge_matrix[i, j] = doge_ret.loc[target_ts]

    avax_mean_ret_per_min = np.nanmean(avax_matrix, axis=0)
    doge_mean_ret_per_min = np.nanmean(doge_matrix, axis=0)

    # cumulative from t=0 (event_ts is offset 0; pre-event window is negative)
    # Anchor cumulative at offset=0 → 0
    zero_idx = int(np.where(offsets == 0)[0][0])
    avax_cum = np.cumsum(avax_mean_ret_per_min) - np.cumsum(avax_mean_ret_per_min)[zero_idx]
    doge_cum = np.cumsum(doge_mean_ret_per_min) - np.cumsum(doge_mean_ret_per_min)[zero_idx]

    # Save curves
    curve_df = pd.DataFrame({
        "offset_minutes": offsets,
        "avax_mean_log_ret": avax_mean_ret_per_min,
        "doge_mean_log_ret": doge_mean_ret_per_min,
        "avax_cum_from_event": avax_cum,
        "doge_cum_from_event": doge_cum,
        "n_events_with_data": np.sum(~np.isnan(avax_matrix), axis=0),
    })
    curve_df.to_csv(OUT_DIR / "event_triggered_curve.csv", index=False)

    # 4) Cross-correlation function (full sample, ±60min lags)
    max_lag = 60
    avax_full = df["avax_log_ret"].fillna(0).values
    doge_full = df["doge_log_ret"].fillna(0).values
    avax_full = (avax_full - avax_full.mean()) / avax_full.std(ddof=1)
    doge_full = (doge_full - doge_full.mean()) / doge_full.std(ddof=1)
    n = len(avax_full)
    ccf = np.zeros(2 * max_lag + 1)
    for k, lag in enumerate(range(-max_lag, max_lag + 1)):
        if lag >= 0:
            x = doge_full[: n - lag]
            y = avax_full[lag:]
        else:
            x = doge_full[-lag:]
            y = avax_full[: n + lag]
        ccf[k] = float(np.mean(x * y))

    ccf_df = pd.DataFrame({"lag_minutes": np.arange(-max_lag, max_lag + 1), "ccf": ccf})
    ccf_df.to_csv(OUT_DIR / "ccf_avax_doge.csv", index=False)
    peak_lag = int(ccf_df.iloc[ccf_df["ccf"].abs().idxmax()]["lag_minutes"])
    peak_val = float(ccf_df["ccf"].abs().max())

    # 5) Diagnostics
    pre_event_avax_jump = float(avax_cum[zero_idx] - avax_cum[zero_idx - 5])  # 5분 직전 AVAX cumulative
    post_event_avax_30m = float(avax_cum[zero_idx + 30] - avax_cum[zero_idx])
    post_event_avax_60m = float(avax_cum[zero_idx + 60] - avax_cum[zero_idx])
    pre_event_doge = float(doge_cum[zero_idx] - doge_cum[zero_idx - 5])
    post_event_doge_30m = float(doge_cum[zero_idx + 30] - doge_cum[zero_idx])

    # 6) Plot
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(offsets, avax_cum * 100, label="AVAX (target)", color="#54A24B", linewidth=2)
    ax1.plot(offsets, doge_cum * 100, label="DOGE (leader)", color="#E45756", linewidth=2)
    ax1.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7, label="event ts (t=0)")
    ax1.axvline(15, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="bucket end (t=15)")
    ax1.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax1.set_xlabel("Offset from DOGE down event (minutes)")
    ax1.set_ylabel("Cumulative log return (%) — anchored at event_ts")
    ax1.set_title(f"Event-triggered avg response (n={len(events)} DOGE down events)")
    ax1.legend()
    ax1.grid(alpha=0.2)

    ax2 = axes[1]
    ax2.plot(ccf_df["lag_minutes"], ccf_df["ccf"], color="#4C78A8", linewidth=1.5)
    ax2.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax2.axvline(peak_lag, color="red", linestyle=":", linewidth=1, alpha=0.7, label=f"peak lag={peak_lag}min")
    ax2.set_xlabel("Lag (DOGE leads → / AVAX leads ←) minutes")
    ax2.set_ylabel("Cross-correlation")
    ax2.set_title("CCF: Cov(DOGE_t, AVAX_{t+lag}) — full sample, 1min returns")
    ax2.legend()
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    plot_path = OUT_DIR / "tick_level_lead_time.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 7) Summary
    print()
    print("=== Event-triggered diagnostics ===")
    print(f"Pre-event AVAX cumulative (t=-5min → t=0):   {pre_event_avax_jump*100:+.4f}%")
    print(f"Pre-event DOGE cumulative (t=-5min → t=0):   {pre_event_doge*100:+.4f}%")
    print(f"Post-event AVAX (t=0 → t=+30min):            {post_event_avax_30m*100:+.4f}%")
    print(f"Post-event AVAX (t=0 → t=+60min):            {post_event_avax_60m*100:+.4f}%")
    print(f"Post-event DOGE (t=0 → t=+30min):            {post_event_doge_30m*100:+.4f}%")
    print()
    print(f"=== CCF (full sample, ±{max_lag}min lags) ===")
    print(f"Peak |CCF| = {peak_val:.4f} at lag = {peak_lag} min")
    print(f"CCF at lag=0:  {float(ccf_df.loc[ccf_df['lag_minutes']==0, 'ccf'].iloc[0]):.4f}")
    print(f"CCF at lag=+1: {float(ccf_df.loc[ccf_df['lag_minutes']==1, 'ccf'].iloc[0]):.4f}")
    print(f"CCF at lag=+5: {float(ccf_df.loc[ccf_df['lag_minutes']==5, 'ccf'].iloc[0]):.4f}")
    print(f"CCF at lag=-1: {float(ccf_df.loc[ccf_df['lag_minutes']==-1, 'ccf'].iloc[0]):.4f}")
    print(f"CCF at lag=-5: {float(ccf_df.loc[ccf_df['lag_minutes']==-5, 'ccf'].iloc[0]):.4f}")
    print()
    print(f"saved plot: {plot_path}")
    print(f"saved curve: {OUT_DIR/'event_triggered_curve.csv'}")
    print(f"saved ccf: {OUT_DIR/'ccf_avax_doge.csv'}")
    print()

    # Interpretation
    print("=== Interpretation ===")
    if abs(pre_event_avax_jump) >= 0.0005:  # 0.05% or larger
        print("- Pre-event AVAX 누적이 |0.05%| 이상 → event 직전부터 같이 움직임 (공통 shock 의심)")
    else:
        print("- Pre-event AVAX 누적이 작음 → event 직전엔 AVAX 큰 변화 없음")
    if peak_lag > 0:
        print(f"- CCF peak at lag={peak_lag} (DOGE → AVAX) → 1분봉에서 DOGE가 AVAX 선행")
    elif peak_lag < 0:
        print(f"- CCF peak at lag={peak_lag} (AVAX → DOGE) → 오히려 AVAX가 DOGE 선행")
    else:
        print("- CCF peak at lag=0 → 동시 반응 (lead-lag ❌)")


if __name__ == "__main__":
    raise SystemExit(main())

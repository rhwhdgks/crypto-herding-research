"""5년 tick-level CCF + event-triggered curve.

step3과 동일 방식이지만 5년 데이터로.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MICRO_FRAME = PROJECT_ROOT / "outputs/v2/tick/multi_asset_5y/lead_lag_matrix/intermediate/tick_micro_frame_15m.csv"
OUT_DIR = PROJECT_ROOT / "experiments/lead_lag_robustness/outputs"

WINDOW_BEFORE = 60
WINDOW_AFTER = 60
START = "2021-05-09"
END = "2026-04-09"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    avax = pd.read_parquet(DATA_DIR / "AVAXUSDT_1m.parquet")[["timestamp", "close"]]
    doge = pd.read_parquet(DATA_DIR / "DOGEUSDT_1m.parquet")[["timestamp", "close"]]
    for d in (avax, doge):
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    df = avax.merge(doge, on="timestamp", how="inner", suffixes=("_avax", "_doge"))
    df = df.set_index("timestamp").sort_index()
    df = df.loc[pd.Timestamp(START, tz="UTC"):pd.Timestamp(END, tz="UTC")]
    df = df.rename(columns={"close_avax": "avax", "close_doge": "doge"}).dropna()
    df["avax_log_ret"] = np.log(df["avax"]).diff()
    df["doge_log_ret"] = np.log(df["doge"]).diff()
    print(f"merged 1m: {len(df):,} rows ({df.index.min()} ~ {df.index.max()})")

    mf = pd.read_csv(MICRO_FRAME)
    mf["bucket_start"] = pd.to_datetime(mf["bucket_start"], utc=True)
    events = pd.to_datetime(
        mf.loc[
            (mf["symbol"] == "DOGEUSDT")
            & mf["is_micro_run_clustering_event"].fillna(False).astype(bool)
            & mf["run_clustering_side"].eq("down")
            & mf["price_direction"].eq("down"),
            "bucket_start",
        ].values,
        utc=True,
    )
    events = events[(events >= df.index.min()) & (events <= df.index.max() - pd.Timedelta(minutes=WINDOW_AFTER))]
    print(f"DOGE down events in window: {len(events)}")

    avax_ret = df["avax_log_ret"]
    doge_ret = df["doge_log_ret"]
    offsets = np.arange(-WINDOW_BEFORE, WINDOW_AFTER + 1)

    avax_matrix = np.full((len(events), len(offsets)), np.nan)
    doge_matrix = np.full((len(events), len(offsets)), np.nan)
    avax_idx = avax_ret.index
    avax_pos = {ts: i for i, ts in enumerate(avax_idx)}
    avax_arr = avax_ret.values
    doge_arr = doge_ret.values

    for i, evt in enumerate(events):
        anchor = avax_pos.get(evt)
        if anchor is None:
            continue
        for j, off in enumerate(offsets):
            pos = anchor + int(off)
            if 0 <= pos < len(avax_arr):
                avax_matrix[i, j] = avax_arr[pos]
                doge_matrix[i, j] = doge_arr[pos]

    avax_mean = np.nanmean(avax_matrix, axis=0)
    doge_mean = np.nanmean(doge_matrix, axis=0)

    zero_idx = int(np.where(offsets == 0)[0][0])
    avax_cum = np.cumsum(avax_mean) - np.cumsum(avax_mean)[zero_idx]
    doge_cum = np.cumsum(doge_mean) - np.cumsum(doge_mean)[zero_idx]

    pd.DataFrame({
        "offset_minutes": offsets,
        "avax_mean_log_ret": avax_mean,
        "doge_mean_log_ret": doge_mean,
        "avax_cum_from_event": avax_cum,
        "doge_cum_from_event": doge_cum,
        "n_events": np.sum(~np.isnan(avax_matrix), axis=0),
    }).to_csv(OUT_DIR / "event_triggered_curve_5y.csv", index=False)

    # CCF (sample to keep runtime reasonable on 5y data)
    max_lag = 60
    sample_size = min(len(df), 1_000_000)
    sample_idx = np.random.default_rng(42).choice(len(df) - max_lag - 1, size=sample_size, replace=False)
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
    ccf_df.to_csv(OUT_DIR / "ccf_avax_doge_5y.csv", index=False)
    peak_lag = int(ccf_df.iloc[ccf_df["ccf"].abs().idxmax()]["lag_minutes"])
    peak_val = float(ccf_df["ccf"].abs().max())

    pre_avax = float(avax_cum[zero_idx] - avax_cum[zero_idx - 5])
    post_avax_30 = float(avax_cum[zero_idx + 30] - avax_cum[zero_idx])
    post_doge_30 = float(doge_cum[zero_idx + 30] - doge_cum[zero_idx])

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
    ax1.set_title(f"5y Event-triggered avg response (n={len(events)})")
    ax1.legend()
    ax1.grid(alpha=0.2)

    ax2 = axes[1]
    ax2.plot(ccf_df["lag_minutes"], ccf_df["ccf"], color="#4C78A8", linewidth=1.5)
    ax2.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax2.axvline(peak_lag, color="red", linestyle=":", linewidth=1, alpha=0.7, label=f"peak lag={peak_lag}min")
    ax2.set_xlabel("Lag (DOGE leads → / AVAX leads ←) minutes")
    ax2.set_ylabel("Cross-correlation")
    ax2.set_title("5y CCF: Cov(DOGE_t, AVAX_{t+lag}), 1min returns")
    ax2.legend()
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    plot_path = OUT_DIR / "tick_level_lead_time_5y.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print()
    print(f"=== 5y Event-triggered ===")
    print(f"Pre-event AVAX (t=-5→0):  {pre_avax*100:+.4f}%")
    print(f"Post-event AVAX (t=0→30): {post_avax_30*100:+.4f}%")
    print(f"Post-event DOGE (t=0→30): {post_doge_30*100:+.4f}%")
    print(f"=== 5y CCF ===")
    print(f"Peak |CCF| = {peak_val:.4f} at lag = {peak_lag} min")
    print(f"CCF at lag=0:  {float(ccf_df.loc[ccf_df['lag_minutes']==0,'ccf'].iloc[0]):.4f}")
    print(f"CCF at lag=+1: {float(ccf_df.loc[ccf_df['lag_minutes']==1,'ccf'].iloc[0]):.4f}")
    print(f"CCF at lag=-1: {float(ccf_df.loc[ccf_df['lag_minutes']==-1,'ccf'].iloc[0]):.4f}")


if __name__ == "__main__":
    raise SystemExit(main())

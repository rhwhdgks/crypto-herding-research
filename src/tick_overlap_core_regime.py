from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_overlap_core_trade_log(base_dir: str | Path) -> pd.DataFrame:
    path = Path(base_dir) / "paper_trade_log.csv"
    frame = pd.read_csv(path, parse_dates=["bucket_start", "entry_timestamp", "exit_timestamp"])
    if frame.empty:
        return frame
    for column in ["bucket_start", "entry_timestamp", "exit_timestamp"]:
        timestamp = frame[column]
        if timestamp.dt.tz is None:
            frame[column] = timestamp.dt.tz_localize("UTC")
        else:
            frame[column] = timestamp.dt.tz_convert("UTC")
    return frame.sort_values("entry_timestamp").reset_index(drop=True)


def load_trade_sample_with_prior_features(base_dir: str | Path) -> pd.DataFrame:
    path = Path(base_dir) / "trade_sample.csv"
    frame = pd.read_csv(path, parse_dates=["bucket_start", "entry_timestamp", "exit_timestamp"])
    if frame.empty:
        return frame
    for column in ["bucket_start", "entry_timestamp", "exit_timestamp"]:
        timestamp = frame[column]
        if timestamp.dt.tz is None:
            frame[column] = timestamp.dt.tz_localize("UTC")
        else:
            frame[column] = timestamp.dt.tz_convert("UTC")

    frame = frame.sort_values(["symbol", "bucket_start"]).reset_index(drop=True)
    frame["prior_bucket_return"] = frame.groupby("symbol")["bucket_return"].shift(1)
    frame["prior_drop_magnitude"] = (-frame["prior_bucket_return"]).clip(lower=0.0)
    keep_columns = [
        "symbol",
        "bucket_start",
        "bucket_return",
        "prior_bucket_return",
        "prior_drop_magnitude",
        "transaction_count",
        "total_quantity",
        "run_clustering_score",
    ]
    return frame[keep_columns].copy()


def enrich_overlap_core_trades(
    overlap_trades: pd.DataFrame,
    trade_sample: pd.DataFrame,
) -> pd.DataFrame:
    if overlap_trades.empty:
        return overlap_trades.copy()
    enriched = overlap_trades.merge(
        trade_sample,
        on=["symbol", "bucket_start"],
        how="left",
        validate="one_to_one",
    )
    enriched["entry_timestamp"] = pd.to_datetime(enriched["entry_timestamp"], utc=True)
    enriched["calendar_year"] = enriched["entry_timestamp"].dt.year
    enriched["calendar_month"] = enriched["entry_timestamp"].dt.month
    enriched["calendar_month_label"] = enriched["entry_timestamp"].dt.strftime("%m월")
    enriched["calendar_quarter"] = "Q" + enriched["entry_timestamp"].dt.quarter.astype(str)
    enriched["trade_date"] = enriched["entry_timestamp"].dt.normalize()
    return enriched.sort_values("entry_timestamp").reset_index(drop=True)


def summarize_group(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for group_key, group in frame.groupby(group_columns, sort=True, dropna=False):
        ordered = group.sort_values("entry_timestamp").copy()
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "trade_count": int(len(ordered)),
                "mean_net_return": float(ordered["net_return"].mean()),
                "median_net_return": float(ordered["net_return"].median()),
                "win_rate": float((ordered["net_return"] > 0).mean()),
                "terminal_cumulative_net_return": float((1.0 + ordered["net_return"]).prod() - 1.0),
                "t_stat": _compute_t_stat(ordered["net_return"]),
                "mean_strength_ratio": float(ordered["strength_ratio"].mean()),
            }
        )
        if "prior_drop_magnitude" in ordered.columns:
            row["mean_prior_drop_magnitude"] = float(ordered["prior_drop_magnitude"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def assign_quantile_bucket(series: pd.Series, bucket_count: int, prefix: str) -> pd.Series:
    valid = series.dropna().astype(float)
    if valid.empty:
        return pd.Series(pd.NA, index=series.index, dtype="object")

    unique_count = int(valid.nunique())
    bucket_count = max(1, min(int(bucket_count), unique_count))
    if bucket_count == 1:
        return pd.Series([f"{prefix}_Q1"] * len(series), index=series.index, dtype="object")

    ranks = valid.rank(method="first")
    labels = [f"{prefix}_Q{i}" for i in range(1, bucket_count + 1)]
    buckets = pd.qcut(ranks, q=bucket_count, labels=labels)
    result = pd.Series(pd.NA, index=series.index, dtype="object")
    result.loc[valid.index] = buckets.astype(str)
    return result


def summarize_strength_and_prior_buckets(
    frame: pd.DataFrame,
    strength_bucket_count: int,
    prior_drop_bucket_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    enriched = frame.copy()
    enriched["strength_bucket"] = assign_quantile_bucket(
        enriched["strength_ratio"],
        bucket_count=int(strength_bucket_count),
        prefix="strength",
    )
    enriched["prior_drop_bucket"] = assign_quantile_bucket(
        enriched["prior_drop_magnitude"],
        bucket_count=int(prior_drop_bucket_count),
        prefix="prior_drop",
    )
    strength_summary = summarize_group(
        enriched.dropna(subset=["strength_bucket"]).copy(),
        ["strength_bucket"],
    ).sort_values("strength_bucket")
    prior_summary = summarize_group(
        enriched.dropna(subset=["prior_drop_bucket"]).copy(),
        ["prior_drop_bucket"],
    ).sort_values("prior_drop_bucket")
    return strength_summary, prior_summary


def summarize_rolling_windows(frame: pd.DataFrame, window_days_list: list[int]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    start_date = frame["trade_date"].dropna().min()
    end_date = frame["trade_date"].dropna().max()
    if pd.isna(start_date) or pd.isna(end_date):
        return pd.DataFrame()
    calendar_dates = list(pd.date_range(start=start_date, end=end_date, freq="1D", tz="UTC"))
    rows: list[dict] = []
    for window_days in sorted({int(value) for value in window_days_list}):
        if window_days <= 0 or window_days > len(calendar_dates):
            continue
        for end_idx in range(window_days - 1, len(calendar_dates)):
            window_dates = calendar_dates[end_idx - window_days + 1 : end_idx + 1]
            subset = frame[frame["trade_date"].isin(window_dates)].copy()
            if subset.empty:
                rows.append(
                    {
                        "window_days": int(window_days),
                        "window_start": min(window_dates),
                        "window_end": max(window_dates),
                        "trade_count": 0,
                        "mean_net_return": np.nan,
                        "win_rate": np.nan,
                        "terminal_cumulative_net_return": np.nan,
                        "t_stat": np.nan,
                    }
                )
                continue
            ordered = subset.sort_values("entry_timestamp").copy()
            rows.append(
                {
                    "window_days": int(window_days),
                    "window_start": min(window_dates),
                    "window_end": max(window_dates),
                    "trade_count": int(len(ordered)),
                    "mean_net_return": float(ordered["net_return"].mean()),
                    "win_rate": float((ordered["net_return"] > 0).mean()),
                    "terminal_cumulative_net_return": float((1.0 + ordered["net_return"]).prod() - 1.0),
                    "t_stat": _compute_t_stat(ordered["net_return"]),
                }
            )
    return pd.DataFrame(rows)


def build_overlap_core_regime_report(
    sample_frame: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    month_summary: pd.DataFrame,
    quarter_summary: pd.DataFrame,
    hour_summary: pd.DataFrame,
    strength_summary: pd.DataFrame,
    prior_summary: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    plot_paths: list[str],
) -> str:
    lines = ["# Tick Overlap Core Regime 보고서", ""]
    lines.append("## 표본")
    lines.extend(
        [
            "- 규칙: `prev_neg AND ratio_1_40_16_18`",
            "- 심볼: `XRPUSDT`",
            f"- 거래 수: {int(len(sample_frame))}건",
            f"- 시작: {sample_frame['entry_timestamp'].min()}",
            f"- 종료: {sample_frame['entry_timestamp'].max()}",
            "",
        ]
    )

    lines.append("## 연도별")
    if yearly_summary.empty:
        lines.append("- 결과가 없습니다.")
    else:
        best_year = yearly_summary.sort_values("mean_net_return", ascending=False).iloc[0]
        worst_year = yearly_summary.sort_values("mean_net_return", ascending=True).iloc[0]
        lines.append(
            f"- 최고 연도: {int(best_year['calendar_year'])} | 평균 순수익 {best_year['mean_net_return']:.4%}, "
            f"누적 {best_year['terminal_cumulative_net_return']:.2%}, 거래 {int(best_year['trade_count'])}건"
        )
        lines.append(
            f"- 최저 연도: {int(worst_year['calendar_year'])} | 평균 순수익 {worst_year['mean_net_return']:.4%}, "
            f"누적 {worst_year['terminal_cumulative_net_return']:.2%}, 거래 {int(worst_year['trade_count'])}건"
        )
    lines.append("")

    lines.append("## 월/분기")
    if not month_summary.empty:
        best_month = month_summary.sort_values("mean_net_return", ascending=False).iloc[0]
        worst_month = month_summary.sort_values("mean_net_return", ascending=True).iloc[0]
        lines.append(
            f"- 가장 좋은 달(계절성): {best_month['calendar_month_label']} | 평균 순수익 {best_month['mean_net_return']:.4%}, 승률 {best_month['win_rate']:.2%}"
        )
        lines.append(
            f"- 가장 약한 달(계절성): {worst_month['calendar_month_label']} | 평균 순수익 {worst_month['mean_net_return']:.4%}, 승률 {worst_month['win_rate']:.2%}"
        )
    if not quarter_summary.empty:
        top_quarter = quarter_summary.sort_values("mean_net_return", ascending=False).iloc[0]
        lines.append(
            f"- 가장 좋은 분기: {top_quarter['calendar_quarter']} | 평균 순수익 {top_quarter['mean_net_return']:.4%}, 누적 {top_quarter['terminal_cumulative_net_return']:.2%}"
        )
    lines.append("")

    lines.append("## 시간대")
    if not hour_summary.empty:
        top_hour = hour_summary.sort_values("mean_net_return", ascending=False).iloc[0]
        bottom_hour = hour_summary.sort_values("mean_net_return", ascending=True).iloc[0]
        lines.append(
            f"- 가장 좋은 시각대: UTC {int(top_hour['hour_utc'])}시 | 평균 순수익 {top_hour['mean_net_return']:.4%}, 승률 {top_hour['win_rate']:.2%}, 거래 {int(top_hour['trade_count'])}건"
        )
        lines.append(
            f"- 가장 약한 시각대: UTC {int(bottom_hour['hour_utc'])}시 | 평균 순수익 {bottom_hour['mean_net_return']:.4%}, 승률 {bottom_hour['win_rate']:.2%}, 거래 {int(bottom_hour['trade_count'])}건"
        )
    lines.append("")

    lines.append("## 이벤트 강도")
    if not strength_summary.empty:
        top_strength = strength_summary.sort_values("mean_net_return", ascending=False).iloc[0]
        lines.append(
            f"- 가장 좋은 strength bucket: {top_strength['strength_bucket']} | 평균 순수익 {top_strength['mean_net_return']:.4%}, "
            f"평균 strength ratio {top_strength['mean_strength_ratio']:.2f}, 거래 {int(top_strength['trade_count'])}건"
        )
    if not prior_summary.empty:
        top_prior = prior_summary.sort_values("mean_net_return", ascending=False).iloc[0]
        lines.append(
            f"- 가장 좋은 prior-drop bucket: {top_prior['prior_drop_bucket']} | 평균 순수익 {top_prior['mean_net_return']:.4%}, "
            f"평균 직전 하락폭 {top_prior['mean_prior_drop_magnitude']:.4%}, 거래 {int(top_prior['trade_count'])}건"
        )
    lines.append("")

    lines.append("## Rolling Regime")
    if rolling_summary.empty:
        lines.append("- 결과가 없습니다.")
    else:
        for window_days in sorted(rolling_summary["window_days"].dropna().unique().tolist()):
            subset = rolling_summary[rolling_summary["window_days"] == window_days].copy()
            best = subset.sort_values("mean_net_return", ascending=False).iloc[0]
            worst = subset.sort_values("mean_net_return", ascending=True).iloc[0]
            positive_share = float((subset["mean_net_return"] > 0).mean())
            lines.append(
                f"- {int(window_days)}일: 양(+) window 비중 {positive_share:.2%}, "
                f"최고 {best['window_start']} ~ {best['window_end']} ({best['mean_net_return']:.4%}), "
                f"최저 {worst['window_start']} ~ {worst['window_end']} ({worst['mean_net_return']:.4%})"
            )
    lines.append("")

    lines.append("## 해석")
    if not strength_summary.empty and not hour_summary.empty:
        top_strength = strength_summary.sort_values("mean_net_return", ascending=False).iloc[0]
        top_hour = hour_summary.sort_values("mean_net_return", ascending=False).iloc[0]
        strongest_bucket = strength_summary.sort_values("mean_strength_ratio", ascending=False).iloc[0]
        if top_strength["strength_bucket"] == strongest_bucket["strength_bucket"]:
            lines.append(
                f"- 5년 표본에서도 overlap_core는 강도가 센 이벤트일수록 더 잘 작동하는 경향이 있습니다. "
                f"특히 `{top_strength['strength_bucket']}`에서 평균 strength ratio {top_strength['mean_strength_ratio']:.2f}, 평균 순수익 {top_strength['mean_net_return']:.4%}가 나왔습니다."
            )
        else:
            lines.append(
                f"- 5년 표본에서는 초강한 이벤트보다 `threshold 근처의 깔끔한 이벤트`가 더 잘 작동합니다. "
                f"가장 좋았던 `{top_strength['strength_bucket']}`의 평균 strength ratio는 {top_strength['mean_strength_ratio']:.2f}였습니다."
            )
        lines.append(
            f"- 시간대는 UTC {int(top_hour['hour_utc'])}시가 가장 좋았습니다. 이 시각대 평균 순수익은 {top_hour['mean_net_return']:.4%}입니다."
        )
    if not yearly_summary.empty:
        positive_year_share = float((yearly_summary["mean_net_return"] > 0).mean())
        lines.append(
            f"- 연도별 평균 순수익 양(+) 비중은 {positive_year_share:.2%}입니다. 즉 최근 1년만의 우연이라기보다 장기 반복성이 어느 정도 있습니다."
        )
    lines.append(
        "- 따라서 다음 실험은 overlap_core 전체를 그대로 쓰기보다, 강한 strength bucket과 유리한 시간대를 우선 필터로 두고 비용/체결 가능성을 다시 확인하는 방향이 적절합니다."
    )
    lines.append("")

    lines.append("## 플롯")
    for plot_path in plot_paths:
        lines.append(f"- {plot_path}")
    lines.append("")
    return "\n".join(lines)


def plot_yearly_summary(yearly_summary: pd.DataFrame, path: str | Path) -> None:
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(10, 5))
    if yearly_summary.empty:
        _render_empty_plot(fig, ax, "연도별 overlap_core", "표시할 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    colors = ["#54A24B" if value >= 0 else "#E45756" for value in yearly_summary["mean_net_return"]]
    ax.bar(yearly_summary["calendar_year"].astype(str), yearly_summary["mean_net_return"] * 100.0, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("평균 순수익 (%)")
    ax.set_title("연도별 overlap_core 평균 순수익")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_hour_summary(hour_summary: pd.DataFrame, path: str | Path) -> None:
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(8, 5))
    if hour_summary.empty:
        _render_empty_plot(fig, ax, "시간대별 overlap_core", "표시할 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    colors = ["#54A24B" if value >= 0 else "#E45756" for value in hour_summary["mean_net_return"]]
    ax.bar(hour_summary["hour_utc"].astype(str), hour_summary["mean_net_return"] * 100.0, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("UTC 시각")
    ax.set_ylabel("평균 순수익 (%)")
    ax.set_title("시간대별 overlap_core 평균 순수익")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bucket_summary(summary: pd.DataFrame, bucket_column: str, title: str, path: str | Path) -> None:
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(9, 5))
    if summary.empty:
        _render_empty_plot(fig, ax, title, "표시할 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    colors = ["#54A24B" if value >= 0 else "#E45756" for value in summary["mean_net_return"]]
    ax.bar(summary[bucket_column].astype(str), summary["mean_net_return"] * 100.0, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("평균 순수익 (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_summary(rolling_summary: pd.DataFrame, path: str | Path) -> None:
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(11, 5))
    if rolling_summary.empty:
        _render_empty_plot(fig, ax, "rolling overlap_core", "표시할 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    plot_frame = rolling_summary.copy()
    plot_frame["window_end"] = pd.to_datetime(plot_frame["window_end"], utc=True)
    for window_days, subset in plot_frame.groupby("window_days"):
        ax.plot(
            subset["window_end"],
            subset["mean_net_return"] * 100.0,
            marker="o",
            linewidth=1.2,
            markersize=2.5,
            label=f"{int(window_days)}일",
        )
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("평균 순수익 (%)")
    ax.set_title("overlap_core rolling 평균 순수익")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _compute_t_stat(series: pd.Series) -> float:
    clean = pd.Series(series).dropna().astype(float)
    if clean.shape[0] < 2:
        return np.nan
    std = clean.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return float(clean.mean() / (std / np.sqrt(clean.shape[0])))


def _configure_korean_matplotlib_font() -> None:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False


def _render_empty_plot(fig: plt.Figure, ax: plt.Axes, title: str, message: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tick_subset_candidates import build_candidate_masks, enrich_trade_features, load_trade_sample


DISPLAY_LABELS = {
    "prev_neg": "prev_neg",
    "ratio_1_40_16_18": "ratio_1_40_16_18",
}


def build_candidate_trade_log(
    base_dir: str | Path,
    selected_candidates: list[str],
    round_trip_cost_bps: float,
) -> pd.DataFrame:
    trades = load_trade_sample(base_dir)
    trades = enrich_trade_features(trades)
    candidate_masks = build_candidate_masks(trades)
    cost_decimal = float(round_trip_cost_bps) / 10000.0

    trade_frames: list[pd.DataFrame] = []
    for candidate_name in selected_candidates:
        if candidate_name not in candidate_masks:
            continue
        subset = trades.loc[candidate_masks[candidate_name]].copy()
        if subset.empty:
            continue
        subset["candidate_name"] = candidate_name
        subset["candidate_label"] = DISPLAY_LABELS.get(candidate_name, candidate_name)
        subset["round_trip_cost_bps"] = float(round_trip_cost_bps)
        subset["net_return"] = subset["gross_return"].astype(float) - cost_decimal
        trade_frames.append(subset)

    if not trade_frames:
        return pd.DataFrame()
    return pd.concat(trade_frames, axis=0, ignore_index=True).sort_values(["candidate_name", "entry_timestamp"]).reset_index(drop=True)


def summarize_candidate_performance(
    trade_log: pd.DataFrame,
    recent_days_list: list[int],
) -> pd.DataFrame:
    if trade_log.empty:
        return pd.DataFrame()

    last_date = trade_log["entry_timestamp"].dt.normalize().max()
    period_masks: dict[str, pd.Series] = {"full_sample": pd.Series(True, index=trade_log.index)}
    for recent_days in sorted({int(value) for value in recent_days_list}):
        start = last_date - pd.Timedelta(days=max(recent_days - 1, 0))
        period_masks[f"recent_{recent_days}d"] = trade_log["entry_timestamp"].dt.normalize() >= start

    rows: list[dict] = []
    for candidate_name, candidate_frame in trade_log.groupby("candidate_name", sort=True):
        for period_name, period_mask in period_masks.items():
            subset = candidate_frame.loc[period_mask.loc[candidate_frame.index]].copy()
            if subset.empty:
                continue
            ordered = subset.sort_values("entry_timestamp").copy()
            equity = (1.0 + ordered["net_return"]).cumprod()
            running_max = equity.cummax()
            drawdown = equity / running_max - 1.0
            net_std = ordered["net_return"].std(ddof=1)
            sharpe_like = np.nan
            if ordered.shape[0] >= 2 and not pd.isna(net_std) and net_std != 0.0:
                sharpe_like = ordered["net_return"].mean() / net_std * math.sqrt(ordered.shape[0])

            rows.append(
                {
                    "candidate_name": candidate_name,
                    "candidate_label": DISPLAY_LABELS.get(candidate_name, candidate_name),
                    "period_name": period_name,
                    "trade_count": int(len(ordered)),
                    "mean_gross_return": ordered["gross_return"].mean(),
                    "mean_net_return": ordered["net_return"].mean(),
                    "median_net_return": ordered["net_return"].median(),
                    "net_win_rate": (ordered["net_return"] > 0).mean(),
                    "terminal_cumulative_net_return": float(equity.iloc[-1] - 1.0),
                    "max_drawdown": float(drawdown.min()),
                    "sharpe_like": sharpe_like,
                    "best_trade_net_return": ordered["net_return"].max(),
                    "worst_trade_net_return": ordered["net_return"].min(),
                }
            )
    return pd.DataFrame(rows)


def build_monthly_summary(trade_log: pd.DataFrame) -> pd.DataFrame:
    if trade_log.empty:
        return pd.DataFrame()

    frame = trade_log.copy()
    frame["month"] = frame["entry_timestamp"].dt.to_period("M").astype(str)
    rows: list[dict] = []
    for (candidate_name, month), group in frame.groupby(["candidate_name", "month"], sort=True):
        ordered = group.sort_values("entry_timestamp").copy()
        month_curve = (1.0 + ordered["net_return"]).cumprod()
        rows.append(
            {
                "candidate_name": candidate_name,
                "candidate_label": DISPLAY_LABELS.get(candidate_name, candidate_name),
                "month": month,
                "trade_count": int(len(ordered)),
                "mean_net_return": ordered["net_return"].mean(),
                "month_cumulative_net_return": float(month_curve.iloc[-1] - 1.0),
                "month_win_rate": (ordered["net_return"] > 0).mean(),
            }
        )
    return pd.DataFrame(rows)


def build_equity_curves(trade_log: pd.DataFrame) -> pd.DataFrame:
    if trade_log.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for candidate_name, group in trade_log.groupby("candidate_name", sort=True):
        ordered = group.sort_values("entry_timestamp").copy()
        ordered["equity_curve"] = (1.0 + ordered["net_return"]).cumprod()
        ordered["cumulative_net_return"] = ordered["equity_curve"] - 1.0
        for trade_number, (_, row) in enumerate(ordered.iterrows(), start=1):
            rows.append(
                {
                    "candidate_name": candidate_name,
                    "candidate_label": DISPLAY_LABELS.get(candidate_name, candidate_name),
                    "entry_timestamp": row["entry_timestamp"],
                    "exit_timestamp": row["exit_timestamp"],
                    "trade_number": trade_number,
                    "net_return": row["net_return"],
                    "equity_curve": row["equity_curve"],
                    "cumulative_net_return": row["cumulative_net_return"],
                }
            )
    return pd.DataFrame(rows)


def build_candidate_paper_report(
    summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    round_trip_cost_bps: float,
    plot_paths: list[str],
) -> str:
    lines = ["# Tick Candidate Paper Simulation", ""]
    lines.append("## 설정")
    lines.extend(
        [
            "- 대상 규칙은 이미 고정된 후보 두 개입니다.",
            f"- round-trip 비용: {float(round_trip_cost_bps):.0f} bps",
            "- 이 결과는 연구 후보를 시간순으로 추적한 paper-trading 형태의 시뮬레이션입니다.",
            "",
        ]
    )

    lines.append("## 기간별 요약")
    focus_periods = ["full_sample", "recent_30d", "recent_60d", "recent_90d"]
    for candidate_name in summary["candidate_name"].drop_duplicates().tolist():
        candidate_rows = summary[summary["candidate_name"] == candidate_name].copy()
        lines.append(f"- {DISPLAY_LABELS.get(candidate_name, candidate_name)}")
        for period_name in focus_periods:
            row = candidate_rows[candidate_rows["period_name"] == period_name]
            if row.empty:
                continue
            record = row.iloc[0]
            lines.append(
                f"  - {period_name}: 거래 {int(record['trade_count'])}건, 평균 순수익 {record['mean_net_return']:.4%}, "
                f"승률 {record['net_win_rate']:.2%}, 누적 {record['terminal_cumulative_net_return']:.2%}, "
                f"최대낙폭 {record['max_drawdown']:.2%}"
            )
    lines.append("")

    lines.append("## 월별 안정성")
    if monthly_summary.empty:
        lines.append("- 결과가 없습니다.")
    else:
        for candidate_name, group in monthly_summary.groupby("candidate_name", sort=True):
            positive_share = float((group["month_cumulative_net_return"] > 0).mean())
            mean_month = float(group["month_cumulative_net_return"].mean())
            lines.append(
                f"- {DISPLAY_LABELS.get(candidate_name, candidate_name)}: 양(+) 월 비중 {positive_share:.2%}, "
                f"월 평균 누적 순수익 {mean_month:.4%}, 월 수 {int(len(group))}"
            )
    lines.append("")

    lines.append("## 해석")
    if not summary.empty:
        full = summary[summary["period_name"] == "full_sample"].copy().sort_values("terminal_cumulative_net_return", ascending=False)
        recent60 = summary[summary["period_name"] == "recent_60d"].copy().sort_values("mean_net_return", ascending=False)
        if not full.empty:
            top_full = full.iloc[0]
            lines.append(
                f"- 전체 기간 누적 기준으로는 `{top_full['candidate_label']}`가 가장 강합니다."
            )
        if not recent60.empty:
            top_recent = recent60.iloc[0]
            lines.append(
                f"- 최근 60일 평균 순수익 기준으로도 `{top_recent['candidate_label']}`가 우위입니다."
            )
        lines.append("- 따라서 다음 단계에서는 두 후보를 모두 버리기보다, 회전이 많은 `prev_neg`와 고확신 저회전인 `ratio_1_40_16_18`를 병렬 추적하는 구성이 좋습니다.")
    lines.append("")

    lines.append("## 플롯")
    for plot_path in plot_paths:
        lines.append(f"- {plot_path}")
    lines.append("")
    return "\n".join(lines)


def plot_equity_curve(curves: pd.DataFrame, path: str | Path, recent_days: int | None = None) -> None:
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(11, 5))
    if curves.empty:
        _render_empty_plot(fig, ax, "Paper Simulation Equity Curve", "표시할 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    plot_frame = curves.copy()
    plot_frame["entry_timestamp"] = pd.to_datetime(plot_frame["entry_timestamp"], utc=True)
    if recent_days is not None:
        cutoff = plot_frame["entry_timestamp"].max().normalize() - pd.Timedelta(days=max(int(recent_days) - 1, 0))
        plot_frame = plot_frame[plot_frame["entry_timestamp"].dt.normalize() >= cutoff].copy()

    for candidate_name, group in plot_frame.groupby("candidate_name", sort=True):
        ordered = group.sort_values("entry_timestamp")
        ax.plot(
            ordered["entry_timestamp"],
            ordered["cumulative_net_return"] * 100.0,
            linewidth=1.5,
            label=DISPLAY_LABELS.get(candidate_name, candidate_name),
        )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("누적 순수익 (%)")
    ax.set_title("후보 규칙 누적 순수익 곡선" if recent_days is None else f"후보 규칙 최근 {int(recent_days)}일 누적 순수익")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_bars(monthly_summary: pd.DataFrame, path: str | Path) -> None:
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(12, 5))
    if monthly_summary.empty:
        _render_empty_plot(fig, ax, "월별 누적 순수익", "표시할 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    pivot = monthly_summary.pivot(index="month", columns="candidate_label", values="month_cumulative_net_return").sort_index()
    months = pivot.index.tolist()
    x = np.arange(len(months))
    candidate_labels = list(pivot.columns)
    width = 0.35 if len(candidate_labels) > 1 else 0.6
    offsets = np.linspace(-width / 2, width / 2, len(candidate_labels)) if len(candidate_labels) > 1 else [0.0]
    for offset, label in zip(offsets, candidate_labels):
        ax.bar(x + offset, pivot[label] * 100.0, width=width / max(len(candidate_labels), 1), label=label)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=35, ha="right")
    ax.set_ylabel("월 누적 순수익 (%)")
    ax.set_title("후보 규칙 월별 누적 순수익")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _configure_korean_matplotlib_font() -> None:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False


def _render_empty_plot(fig: plt.Figure, ax: plt.Axes, title: str, message: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()

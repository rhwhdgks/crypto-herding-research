from __future__ import annotations

from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tick_event_schema import build_run_side_event_mask


def build_tick_trade_frame(
    frame: pd.DataFrame,
    event_label: str,
    horizon_minutes: int,
    enforce_non_overlap: bool,
) -> pd.DataFrame:
    return_column = f"forward_return_{int(horizon_minutes)}m"
    if return_column not in frame.columns:
        raise ValueError(f"Column not found: {return_column}")

    event_mask = build_run_side_event_mask(frame, event_label)

    trades = frame.loc[event_mask].copy()
    trades = trades.dropna(subset=[return_column]).sort_values(["symbol", "bucket_start"]).reset_index(drop=True)
    if trades.empty:
        return pd.DataFrame()

    trades["entry_timestamp"] = pd.to_datetime(trades["signal_timestamp"], utc=True)
    trades["exit_timestamp"] = trades["entry_timestamp"] + pd.Timedelta(minutes=int(horizon_minutes))
    trades["gross_return"] = trades[return_column].astype(float)
    trades["signal_side"] = "long"

    if not enforce_non_overlap:
        return trades

    kept: list[dict] = []
    for symbol, symbol_frame in trades.groupby("symbol", sort=True):
        next_entry_time = None
        for _, row in symbol_frame.iterrows():
            entry = pd.Timestamp(row["entry_timestamp"])
            if next_entry_time is not None and entry < next_entry_time:
                continue
            kept.append(row.to_dict())
            next_entry_time = pd.Timestamp(row["exit_timestamp"])

    return pd.DataFrame(kept).sort_values(["symbol", "entry_timestamp"]).reset_index(drop=True) if kept else pd.DataFrame()


def build_period_masks(trades: pd.DataFrame, holdout_days_list: list[int]) -> dict[str, pd.Series]:
    normalized_dates = sorted(trades["entry_timestamp"].dt.normalize().unique().tolist())
    masks: dict[str, pd.Series] = {"full_sample": pd.Series(True, index=trades.index)}

    for holdout_days in sorted({int(value) for value in holdout_days_list}):
        holdout_days = min(holdout_days, len(normalized_dates))
        holdout_dates = set(normalized_dates[-holdout_days:])
        development_dates = set(normalized_dates[:-holdout_days]) if holdout_days < len(normalized_dates) else set()
        masks[f"holdout_{holdout_days}d"] = trades["entry_timestamp"].dt.normalize().isin(holdout_dates)
        masks[f"development_{holdout_days}d"] = trades["entry_timestamp"].dt.normalize().isin(development_dates)

    return masks


def summarize_cost_grid(
    trades: pd.DataFrame,
    holdout_days_list: list[int],
    cost_bps_grid: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()

    period_masks = build_period_masks(trades, holdout_days_list=holdout_days_list)
    rows: list[dict] = []
    break_even_rows: list[dict] = []

    for period_name, period_mask in period_masks.items():
        subset = trades.loc[period_mask].copy()
        if subset.empty:
            continue

        gross_sample = subset["gross_return"].dropna()
        break_even_bps = float(gross_sample.mean() * 10000.0) if not gross_sample.empty else np.nan
        break_even_rows.append(
            {
                "period_name": period_name,
                "trade_count": int(gross_sample.shape[0]),
                "mean_gross_return": gross_sample.mean() if not gross_sample.empty else np.nan,
                "gross_t_stat": _compute_t_stat(gross_sample),
                "gross_win_rate": (gross_sample > 0).mean() if not gross_sample.empty else np.nan,
                "break_even_round_trip_bps": break_even_bps,
            }
        )

        for cost_bps in cost_bps_grid:
            round_trip_cost = float(cost_bps) / 10000.0
            net_sample = gross_sample - round_trip_cost
            rows.append(
                {
                    "period_name": period_name,
                    "round_trip_cost_bps": float(cost_bps),
                    "trade_count": int(net_sample.shape[0]),
                    "mean_net_return": net_sample.mean() if not net_sample.empty else np.nan,
                    "median_net_return": net_sample.median() if not net_sample.empty else np.nan,
                    "net_t_stat": _compute_t_stat(net_sample),
                    "net_win_rate": (net_sample > 0).mean() if not net_sample.empty else np.nan,
                    "terminal_cumulative_net_return": _compute_terminal_cumulative_return(net_sample),
                    "max_drawdown": _compute_max_drawdown(net_sample),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(break_even_rows)


def build_cost_sanity_report(
    rule_label: str,
    symbol_label: str,
    horizon_minutes: int,
    cost_summary: pd.DataFrame,
    break_even_summary: pd.DataFrame,
    plot_paths: list[str],
) -> str:
    lines = ["# Tick 비용 Sanity Check", ""]
    lines.append("## 설정")
    lines.extend(
        [
            f"- 규칙: {rule_label}",
            f"- 심볼: {symbol_label}",
            f"- 보유기간: {int(horizon_minutes)}분",
            "- 해석 기준은 이벤트 발생 시 진입하는 long-only non-overlap trade입니다.",
            "- 비용은 round-trip 기준으로 일괄 차감했습니다.",
            "",
        ]
    )

    lines.append("## Break-even")
    if break_even_summary.empty:
        lines.append("- 결과가 없습니다.")
    else:
        for _, row in break_even_summary.iterrows():
            lines.append(
                f"- {row['period_name']}: 평균 총수익 {row['mean_gross_return']:.4%}, "
                f"gross t={row['gross_t_stat']:.2f}, 승률 {row['gross_win_rate']:.2%}, "
                f"손익분기 round-trip 비용 {row['break_even_round_trip_bps']:.2f} bps"
            )
    lines.append("")

    lines.append("## 비용별 순수익")
    if cost_summary.empty:
        lines.append("- 결과가 없습니다.")
    else:
        focus_periods = ["full_sample", "development_30d", "holdout_30d", "holdout_60d"]
        for period_name in focus_periods:
            subset = cost_summary[cost_summary["period_name"] == period_name].copy()
            if subset.empty:
                continue
            lines.append(f"- {period_name}")
            for _, row in subset.iterrows():
                lines.append(
                    f"  - {row['round_trip_cost_bps']:.0f} bps: 평균 순수익 {row['mean_net_return']:.4%}, "
                    f"net t={row['net_t_stat']:.2f}, 승률 {row['net_win_rate']:.2%}, "
                    f"누적 {row['terminal_cumulative_net_return']:.2%}"
                )
    lines.append("")

    lines.append("## 해석")
    if not break_even_summary.empty:
        full_row = break_even_summary[break_even_summary["period_name"] == "full_sample"]
        holdout_30 = break_even_summary[break_even_summary["period_name"] == "holdout_30d"]
        holdout_60 = break_even_summary[break_even_summary["period_name"] == "holdout_60d"]
        if not full_row.empty:
            lines.append(
                f"- 전체 1년 기준 손익분기 비용은 {float(full_row.iloc[0]['break_even_round_trip_bps']):.2f} bps입니다."
            )
        if not holdout_30.empty:
            lines.append(
                f"- 최근 30일 기준 손익분기 비용은 {float(holdout_30.iloc[0]['break_even_round_trip_bps']):.2f} bps입니다."
            )
        if not holdout_60.empty:
            lines.append(
                f"- 최근 60일 기준 손익분기 비용은 {float(holdout_60.iloc[0]['break_even_round_trip_bps']):.2f} bps입니다."
            )
        lines.append("- 따라서 최근 30일 강세가 실제 체결비용까지 감당하는지, 그리고 60일 이상으로 늘리면 edge가 유지되는지가 핵심 판단 포인트입니다.")
    lines.append("")

    lines.append("## 플롯")
    for plot_path in plot_paths:
        lines.append(f"- {plot_path}")
    lines.append("")
    return "\n".join(lines)


def plot_cost_curves(cost_summary: pd.DataFrame, path: str | Path) -> None:
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(10, 5))
    if cost_summary.empty:
        _render_empty_plot(fig, ax, "Tick 비용 곡선", "표시할 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    focus_periods = ["full_sample", "development_30d", "holdout_30d", "holdout_60d"]
    for period_name in focus_periods:
        subset = cost_summary[cost_summary["period_name"] == period_name].copy()
        if subset.empty:
            continue
        ax.plot(
            subset["round_trip_cost_bps"],
            subset["mean_net_return"] * 100.0,
            marker="o",
            linewidth=1.5,
            markersize=4.0,
            label=period_name,
        )
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Round-trip 비용 (bps)")
    ax.set_ylabel("평균 순수익 (%)")
    ax.set_title("비용 가정별 평균 순수익")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _compute_terminal_cumulative_return(sample: pd.Series) -> float:
    if sample.empty:
        return np.nan
    return float((1.0 + sample).cumprod().iloc[-1] - 1.0)


def _compute_max_drawdown(sample: pd.Series) -> float:
    if sample.empty:
        return np.nan
    equity = (1.0 + sample).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def _compute_t_stat(sample: pd.Series) -> float:
    if sample.shape[0] < 2:
        return np.nan
    std = sample.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return float(sample.mean() / (std / sqrt(sample.shape[0])))


def _configure_korean_matplotlib_font() -> None:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False


def _render_empty_plot(fig: plt.Figure, ax: plt.Axes, title: str, message: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()

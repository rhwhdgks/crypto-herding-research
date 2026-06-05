from __future__ import annotations

from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tick_candidate_paper_sim import build_equity_curves, build_monthly_summary, plot_equity_curve, plot_monthly_bars
from tick_cost_sanity import summarize_cost_grid


RULE_LABEL = "overlap_core"
RULE_DESCRIPTION = "prev_neg AND ratio_1_40_16_18"


def load_overlap_core_trades(base_dir: str | Path) -> pd.DataFrame:
    path = Path(base_dir) / "partition_trade_log.csv"
    frame = pd.read_csv(path, parse_dates=["bucket_start", "entry_timestamp", "exit_timestamp"])
    if frame.empty:
        return frame
    for column in ["bucket_start", "entry_timestamp", "exit_timestamp"]:
        timestamp = frame[column]
        if timestamp.dt.tz is None:
            frame[column] = timestamp.dt.tz_localize("UTC")
        else:
            frame[column] = timestamp.dt.tz_convert("UTC")
    overlap = frame[frame["partition_name"] == "ratio_overlap"].copy()
    if overlap.empty:
        return overlap
    overlap["candidate_name"] = RULE_LABEL
    overlap["candidate_label"] = RULE_LABEL
    return overlap.sort_values("entry_timestamp").reset_index(drop=True)


def summarize_overlap_oos(
    trades: pd.DataFrame,
    round_trip_cost_bps: float,
    holdout_days_list: list[int],
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    frame = trades.copy()
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    frame["net_return"] = frame["gross_return"].astype(float) - (float(round_trip_cost_bps) / 10000.0)
    normalized_dates = sorted(frame["entry_timestamp"].dt.normalize().unique().tolist())

    rows: list[dict] = []
    rows.append(_build_period_row(frame, "full_sample"))
    for holdout_days in sorted({int(value) for value in holdout_days_list}):
        holdout_days = min(holdout_days, len(normalized_dates))
        holdout_dates = set(normalized_dates[-holdout_days:])
        development_dates = set(normalized_dates[:-holdout_days]) if holdout_days < len(normalized_dates) else set()
        holdout_frame = frame[frame["entry_timestamp"].dt.normalize().isin(holdout_dates)].copy()
        development_frame = frame[frame["entry_timestamp"].dt.normalize().isin(development_dates)].copy()
        rows.append(_build_period_row(development_frame, f"development_{holdout_days}d"))
        rows.append(_build_period_row(holdout_frame, f"holdout_{holdout_days}d"))
    return pd.DataFrame(rows)


def summarize_overlap_blocks(
    trades: pd.DataFrame,
    round_trip_cost_bps: float,
    block_days: int,
    step_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()

    frame = trades.copy()
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    frame["net_return"] = frame["gross_return"].astype(float) - (float(round_trip_cost_bps) / 10000.0)
    start_date = frame["entry_timestamp"].dt.normalize().min()
    end_date = frame["entry_timestamp"].dt.normalize().max()
    if pd.isna(start_date) or pd.isna(end_date):
        return pd.DataFrame(), pd.DataFrame()
    calendar = list(pd.date_range(start=start_date, end=end_date, freq="1D", tz="UTC"))
    if len(calendar) < int(block_days):
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict] = []
    fold_id = 1
    max_start = len(calendar) - int(block_days)
    if max_start < 0:
        return pd.DataFrame(), pd.DataFrame()

    for start_idx in range(0, max_start + 1, int(step_days)):
        block_dates = calendar[start_idx : start_idx + int(block_days)]
        if len(block_dates) < int(block_days):
            continue
        subset = frame[frame["entry_timestamp"].dt.normalize().isin(block_dates)].copy()
        rows.append(
            {
                "fold_id": int(fold_id),
                "window_start": min(block_dates),
                "window_end": max(block_dates),
                "trade_count": int(len(subset)),
                "mean_net_return": float(subset["net_return"].mean()) if not subset.empty else np.nan,
                "median_net_return": float(subset["net_return"].median()) if not subset.empty else np.nan,
                "win_rate": float((subset["net_return"] > 0).mean()) if not subset.empty else np.nan,
                "terminal_cumulative_net_return": _compute_terminal_cumulative_return(subset["net_return"]),
                "t_stat": _compute_t_stat(subset["net_return"]),
            }
        )
        fold_id += 1

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summary = pd.DataFrame(
        [
            {
                "blocks_with_trades": int((detail["trade_count"] > 0).sum()),
                "positive_block_share": float((detail["mean_net_return"] > 0).mean()),
                "mean_block_net_return": float(detail["mean_net_return"].mean()),
                "median_block_trade_count": float(detail["trade_count"].median()),
                "max_block_net_return": float(detail["mean_net_return"].max()),
                "min_block_net_return": float(detail["mean_net_return"].min()),
            }
        ]
    )
    return detail, summary


def prepare_overlap_paper_sim_trade_log(trades: pd.DataFrame, round_trip_cost_bps: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    frame = trades.copy()
    frame["candidate_name"] = RULE_LABEL
    frame["candidate_label"] = RULE_LABEL
    frame["round_trip_cost_bps"] = float(round_trip_cost_bps)
    frame["net_return"] = frame["gross_return"].astype(float) - (float(round_trip_cost_bps) / 10000.0)
    return frame.sort_values("entry_timestamp").reset_index(drop=True)


def build_overlap_core_report(
    oos_summary: pd.DataFrame,
    break_even_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    block_summary: pd.DataFrame,
    distinct_event_days: int,
    round_trip_cost_bps: float,
    plot_paths: list[str],
) -> str:
    lines = ["# Tick 겹침 핵심 후보 검증", ""]
    lines.append("## 규칙")
    lines.extend(
        [
            f"- 규칙 이름: `{RULE_LABEL}`",
            f"- 정의: `{RULE_DESCRIPTION}`",
            "- 의미: 직전 15분이 음수였고, 이벤트 강도가 threshold 대비 충분히 강하며, 시각대가 UTC 16-18인 XRP 이벤트만 남긴 규칙입니다.",
            f"- 이벤트가 존재한 날짜 수: {int(distinct_event_days)}일",
            f"- 기준 비용: round-trip {float(round_trip_cost_bps):.1f} bps",
            "",
        ]
    )

    lines.append("## OOS 요약")
    if oos_summary.empty:
        lines.append("- 결과가 없습니다.")
    else:
        for _, row in oos_summary.iterrows():
            lines.append(
                f"- {row['period_name']}: 거래 {int(row['trade_count'])}건, 평균 순수익 {row['mean_net_return']:.4%}, "
                f"승률 {row['win_rate']:.2%}, 누적 {row['terminal_cumulative_net_return']:.2%}, t={row['t_stat']:.2f}"
            )
    lines.append("")

    lines.append("## 비용 내구성")
    if break_even_summary.empty:
        lines.append("- 결과가 없습니다.")
    else:
        for _, row in break_even_summary.iterrows():
            lines.append(
                f"- {row['period_name']}: 평균 총수익 {row['mean_gross_return']:.4%}, "
                f"손익분기 비용 {row['break_even_round_trip_bps']:.2f} bps"
            )
    if not cost_summary.empty:
        full = cost_summary[cost_summary["period_name"] == "full_sample"].copy()
        holdout = cost_summary[cost_summary["period_name"] == "holdout_30d"].copy()
        if not full.empty:
            lines.append(
                f"- full_sample 기준 2bps에서 평균 순수익 {float(full[full['round_trip_cost_bps'] == 2]['mean_net_return'].iloc[0]):.4%}, "
                f"10bps에서는 {float(full[full['round_trip_cost_bps'] == 10]['mean_net_return'].iloc[0]):.4%}입니다."
            )
        if not holdout.empty:
            lines.append(
                f"- holdout_30d 기준 2bps에서 평균 순수익 {float(holdout[holdout['round_trip_cost_bps'] == 2]['mean_net_return'].iloc[0]):.4%}, "
                f"10bps에서는 {float(holdout[holdout['round_trip_cost_bps'] == 10]['mean_net_return'].iloc[0]):.4%}입니다."
            )
    lines.append("")

    lines.append("## Paper Simulation")
    if monthly_summary.empty:
        lines.append("- 결과가 없습니다.")
    else:
        positive_share = float((monthly_summary["month_cumulative_net_return"] > 0).mean())
        mean_month = float(monthly_summary["month_cumulative_net_return"].mean())
        lines.append(
            f"- 양(+) 월 비중 {positive_share:.2%}, 월 평균 누적 순수익 {mean_month:.4%}, 월 수 {int(len(monthly_summary))}"
        )
    if not block_summary.empty:
        block = block_summary.iloc[0]
        lines.append(
            f"- 30일 블록 기준 양(+) 블록 비중 {float(block['positive_block_share']):.2%}, "
            f"블록 평균 순수익 {float(block['mean_block_net_return']):.4%}"
        )
    lines.append("")

    lines.append("## 해석")
    if not oos_summary.empty:
        hold30 = oos_summary[oos_summary["period_name"] == "holdout_30d"]
        hold60 = oos_summary[oos_summary["period_name"] == "holdout_60d"]
        full = oos_summary[oos_summary["period_name"] == "full_sample"]
        if not full.empty:
            lines.append(
                f"- 전체 표본에서는 평균 순수익 {float(full.iloc[0]['mean_net_return']):.4%}로, 기존 `prev_neg` 전체보다 훨씬 압축된 고확신 규칙인지 확인할 수 있습니다."
            )
        if not hold30.empty and not hold60.empty:
            hold30_mean = float(hold30.iloc[0]["mean_net_return"])
            hold60_mean = float(hold60.iloc[0]["mean_net_return"])
            if hold30_mean > 0 and hold60_mean > 0:
                lines.append("- 최근 30일과 60일 holdout 모두 양(+)이면, 최근 구간에서도 규칙이 무너지지 않았다고 볼 수 있습니다.")
            elif hold30_mean > 0 and hold60_mean <= 0:
                lines.append("- 최근 30일은 좋지만 60일로 넓히면 약해져, 최근 구간 의존 가능성을 열어둬야 합니다.")
            else:
                lines.append("- holdout에서도 아직 일관된 우위가 약해 추가 추적이 더 필요합니다.")
    lines.append("- 이 리포트는 겹침 핵심 65건이 단순한 우연인지, 아니면 시간순 누적 기준에서도 추적할 가치가 있는지 보는 용도입니다.")
    lines.append("")

    lines.append("## 플롯")
    for plot_path in plot_paths:
        lines.append(f"- {plot_path}")
    lines.append("")
    return "\n".join(lines)


def plot_overlap_oos(oos_summary: pd.DataFrame, path: str | Path) -> None:
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(10, 5))
    if oos_summary.empty:
        _render_empty_plot(fig, ax, "겹침 핵심 OOS", "표시할 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    plot_frame = oos_summary[oos_summary["period_name"] != "full_sample"].copy()
    colors = ["#54A24B" if value >= 0 else "#E45756" for value in plot_frame["mean_net_return"]]
    ax.bar(plot_frame["period_name"], plot_frame["mean_net_return"] * 100.0, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("평균 순수익 (%)")
    ax.set_title("겹침 핵심 OOS 평균 순수익")
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_overlap_blocks(block_detail: pd.DataFrame, path: str | Path) -> None:
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(10, 5))
    if block_detail.empty:
        _render_empty_plot(fig, ax, "겹침 핵심 30일 블록", "표시할 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    plot_frame = block_detail.copy()
    labels = [f"F{int(value)}" for value in plot_frame["fold_id"]]
    colors = ["#54A24B" if value >= 0 else "#E45756" for value in plot_frame["mean_net_return"]]
    ax.bar(labels, plot_frame["mean_net_return"] * 100.0, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("평균 순수익 (%)")
    ax.set_title("겹침 핵심 30일 블록 평균 순수익")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_period_row(frame: pd.DataFrame, period_name: str) -> dict:
    if frame.empty:
        return {
            "period_name": period_name,
            "trade_count": 0,
            "mean_gross_return": np.nan,
            "mean_net_return": np.nan,
            "median_net_return": np.nan,
            "win_rate": np.nan,
            "terminal_cumulative_net_return": np.nan,
            "t_stat": np.nan,
        }
    net = frame["net_return"]
    return {
        "period_name": period_name,
        "trade_count": int(len(frame)),
        "mean_gross_return": float(frame["gross_return"].mean()),
        "mean_net_return": float(net.mean()),
        "median_net_return": float(net.median()),
        "win_rate": float((net > 0).mean()),
        "terminal_cumulative_net_return": _compute_terminal_cumulative_return(net),
        "t_stat": _compute_t_stat(net),
    }


def _compute_terminal_cumulative_return(sample: pd.Series) -> float:
    if sample.empty:
        return np.nan
    return float((1.0 + sample).cumprod().iloc[-1] - 1.0)


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

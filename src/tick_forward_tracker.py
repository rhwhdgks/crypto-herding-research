from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_paper_trade_log(base_dir: str | Path) -> pd.DataFrame:
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


def build_signal_log(trades: pd.DataFrame, as_of_utc: pd.Timestamp) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    frame = trades.copy().sort_values("entry_timestamp").reset_index(drop=True)
    frame["signal_id"] = [f"overlap_core_{idx:04d}" for idx in range(1, len(frame) + 1)]
    frame["holding_minutes"] = (
        (frame["exit_timestamp"] - frame["entry_timestamp"]).dt.total_seconds() / 60.0
    ).astype(int)
    frame["age_minutes"] = ((as_of_utc - frame["entry_timestamp"]).dt.total_seconds() / 60.0).astype(float)
    frame["status"] = np.where(
        frame["exit_timestamp"] <= as_of_utc,
        "closed",
        np.where(frame["entry_timestamp"] <= as_of_utc, "active", "scheduled"),
    )
    frame["realized_net_return"] = np.where(frame["status"] == "closed", frame["net_return"], np.nan)
    frame["realized_gross_return"] = np.where(frame["status"] == "closed", frame["gross_return"], np.nan)
    frame["signal_date"] = frame["entry_timestamp"].dt.normalize()
    return frame


def summarize_tracker_periods(signal_log: pd.DataFrame, recent_days_list: list[int], as_of_utc: pd.Timestamp) -> pd.DataFrame:
    if signal_log.empty:
        return pd.DataFrame()

    closed = signal_log[signal_log["status"] == "closed"].copy()
    if closed.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    rows.append(_build_tracker_row(closed, "full_sample"))
    as_of_date = as_of_utc.normalize()
    for days in sorted({int(value) for value in recent_days_list}):
        cutoff = as_of_date - pd.Timedelta(days=max(days - 1, 0))
        subset = closed[closed["entry_timestamp"].dt.normalize() >= cutoff].copy()
        rows.append(_build_tracker_row(subset, f"recent_{days}d"))
    return pd.DataFrame(rows)


def build_recent_signal_views(signal_log: pd.DataFrame, recent_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if signal_log.empty:
        return pd.DataFrame(), pd.DataFrame()
    active = signal_log[signal_log["status"] == "active"].copy().sort_values("entry_timestamp", ascending=False)
    recent = signal_log[signal_log["status"] == "closed"].copy().sort_values("entry_timestamp", ascending=False).head(int(recent_count))
    return active.reset_index(drop=True), recent.reset_index(drop=True)


def build_daily_tracker_summary(signal_log: pd.DataFrame) -> pd.DataFrame:
    if signal_log.empty:
        return pd.DataFrame()
    closed = signal_log[signal_log["status"] == "closed"].copy()
    if closed.empty:
        return pd.DataFrame()
    closed["trade_date"] = closed["entry_timestamp"].dt.normalize()
    rows: list[dict] = []
    running_equity = 1.0
    for trade_date, group in closed.groupby("trade_date", sort=True):
        ordered = group.sort_values("entry_timestamp")
        day_curve = (1.0 + ordered["realized_net_return"]).cumprod()
        day_return = float(day_curve.iloc[-1] - 1.0)
        running_equity *= float(day_curve.iloc[-1])
        rows.append(
            {
                "trade_date": trade_date,
                "trade_count": int(len(ordered)),
                "mean_net_return": float(ordered["realized_net_return"].mean()),
                "day_cumulative_net_return": day_return,
                "running_cumulative_net_return": float(running_equity - 1.0),
                "win_rate": float((ordered["realized_net_return"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_tracker_report(
    signal_log: pd.DataFrame,
    tracker_summary: pd.DataFrame,
    active_signals: pd.DataFrame,
    recent_signals: pd.DataFrame,
    daily_summary: pd.DataFrame,
    as_of_utc: pd.Timestamp,
    plot_paths: list[str],
) -> str:
    lines = ["# Overlap Core Forward Paper Tracker", ""]
    lines.append("## 기준 시각")
    lines.append(f"- as_of_utc: {as_of_utc}")
    lines.append("")

    lines.append("## 현재 상태")
    if signal_log.empty:
        lines.append("- 신호 로그가 없습니다.")
    else:
        last_signal = signal_log.sort_values("entry_timestamp").iloc[-1]
        lines.append(f"- 누적 신호 수: {int(len(signal_log))}건")
        lines.append(f"- 종료 신호 수: {int((signal_log['status'] == 'closed').sum())}건")
        lines.append(f"- 활성 신호 수: {int((signal_log['status'] == 'active').sum())}건")
        lines.append(
            f"- 마지막 신호: {last_signal['entry_timestamp']} | 상태 {last_signal['status']} | "
            f"최근 순수익 {float(last_signal['net_return']):.4%}"
        )
    lines.append("")

    lines.append("## 기간별 추적 요약")
    if tracker_summary.empty:
        lines.append("- 요약 결과가 없습니다.")
    else:
        for _, row in tracker_summary.iterrows():
            lines.append(
                f"- {row['period_name']}: 거래 {int(row['trade_count'])}건, 평균 순수익 {row['mean_net_return']:.4%}, "
                f"승률 {row['win_rate']:.2%}, 누적 {row['terminal_cumulative_net_return']:.2%}"
            )
    lines.append("")

    lines.append("## 활성 신호")
    if active_signals.empty:
        lines.append("- 현재 활성 신호는 없습니다.")
    else:
        for _, row in active_signals.iterrows():
            lines.append(
                f"- {row['signal_id']}: {row['entry_timestamp']} 진입, {row['exit_timestamp']} 예정 종료, "
                f"강도비율 {float(row['strength_ratio']):.2f}"
            )
    lines.append("")

    lines.append("## 최근 종료 신호")
    if recent_signals.empty:
        lines.append("- 최근 종료 신호가 없습니다.")
    else:
        for _, row in recent_signals.head(10).iterrows():
            lines.append(
                f"- {row['signal_id']}: {row['entry_timestamp']} | net {float(row['realized_net_return']):.4%} | "
                f"strength {float(row['strength_ratio']):.2f}"
            )
    lines.append("")

    lines.append("## 일별 흐름")
    if daily_summary.empty:
        lines.append("- 일별 요약이 없습니다.")
    else:
        positive_days = float((daily_summary["day_cumulative_net_return"] > 0).mean())
        lines.append(
            f"- 신호 발생일 기준 양(+) 일 비중 {positive_days:.2%}, 마지막 누적 순수익 {float(daily_summary.iloc[-1]['running_cumulative_net_return']):.2%}"
        )
    lines.append("")

    lines.append("## 플롯")
    for plot_path in plot_paths:
        lines.append(f"- {plot_path}")
    lines.append("")
    return "\n".join(lines)


def plot_tracker_cumulative(daily_summary: pd.DataFrame, path: str | Path) -> None:
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(10, 5))
    if daily_summary.empty:
        _render_empty_plot(fig, ax, "Forward Tracker Cumulative", "표시할 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    ax.plot(daily_summary["trade_date"], daily_summary["running_cumulative_net_return"] * 100.0, linewidth=1.6, color="#4C78A8")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_title("Overlap Core 누적 순수익 추적")
    ax.set_ylabel("누적 순수익 (%)")
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_recent_signals(recent_signals: pd.DataFrame, path: str | Path) -> None:
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(11, 5))
    if recent_signals.empty:
        _render_empty_plot(fig, ax, "Recent Signals", "표시할 최근 신호가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    plot_frame = recent_signals.sort_values("entry_timestamp").copy()
    colors = ["#54A24B" if value >= 0 else "#E45756" for value in plot_frame["realized_net_return"]]
    labels = [str(value)[5:16] for value in plot_frame["entry_timestamp"]]
    ax.bar(labels, plot_frame["realized_net_return"] * 100.0, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_title("최근 종료 신호 순수익")
    ax.set_ylabel("순수익 (%)")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_tracker_row(frame: pd.DataFrame, period_name: str) -> dict:
    if frame.empty:
        return {
            "period_name": period_name,
            "trade_count": 0,
            "mean_net_return": np.nan,
            "median_net_return": np.nan,
            "win_rate": np.nan,
            "terminal_cumulative_net_return": np.nan,
        }
    net = frame["realized_net_return"]
    return {
        "period_name": period_name,
        "trade_count": int(len(frame)),
        "mean_net_return": float(net.mean()),
        "median_net_return": float(net.median()),
        "win_rate": float((net > 0).mean()),
        "terminal_cumulative_net_return": float((1.0 + net).prod() - 1.0),
    }


def _configure_korean_matplotlib_font() -> None:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False


def _render_empty_plot(fig: plt.Figure, ax: plt.Axes, title: str, message: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()

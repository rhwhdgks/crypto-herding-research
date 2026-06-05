from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_variant_trade_log(base_dir: str | Path, selected_variants: list[str]) -> pd.DataFrame:
    path = Path(base_dir) / "variant_trade_log.csv"
    frame = pd.read_csv(path, parse_dates=["bucket_start", "entry_timestamp", "exit_timestamp"])
    if frame.empty:
        return frame
    for column in ["bucket_start", "entry_timestamp", "exit_timestamp"]:
        timestamp = frame[column]
        if timestamp.dt.tz is None:
            frame[column] = timestamp.dt.tz_localize("UTC")
        else:
            frame[column] = timestamp.dt.tz_convert("UTC")
    if selected_variants:
        frame = frame[frame["variant_name"].isin(selected_variants)].copy()
    return frame.sort_values(["variant_name", "entry_timestamp"]).reset_index(drop=True)


def build_variant_signal_log(trades: pd.DataFrame, as_of_utc: pd.Timestamp) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    signal_frames: list[pd.DataFrame] = []
    for variant_name, group in trades.groupby("variant_name", sort=True):
        ordered = group.sort_values("entry_timestamp").reset_index(drop=True).copy()
        ordered["signal_id"] = [f"{variant_name}_{idx:04d}" for idx in range(1, len(ordered) + 1)]
        signal_frames.append(ordered)

    frame = pd.concat(signal_frames, axis=0, ignore_index=True).sort_values(
        ["variant_name", "entry_timestamp"]
    ).reset_index(drop=True)
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


def summarize_variant_tracker_periods(
    signal_log: pd.DataFrame,
    recent_days_list: list[int],
    as_of_utc: pd.Timestamp,
) -> pd.DataFrame:
    if signal_log.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    closed = signal_log[signal_log["status"] == "closed"].copy()
    if closed.empty:
        return pd.DataFrame()

    as_of_date = as_of_utc.normalize()
    for variant_name, group in closed.groupby("variant_name", sort=True):
        rows.append(_build_tracker_row(group, variant_name, "full_sample"))
        for days in sorted({int(value) for value in recent_days_list}):
            cutoff = as_of_date - pd.Timedelta(days=max(days - 1, 0))
            subset = group[group["entry_timestamp"].dt.normalize() >= cutoff].copy()
            rows.append(_build_tracker_row(subset, variant_name, f"recent_{days}d"))
    return pd.DataFrame(rows)


def build_variant_recent_views(signal_log: pd.DataFrame, recent_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if signal_log.empty:
        return pd.DataFrame(), pd.DataFrame()
    active = (
        signal_log[signal_log["status"] == "active"]
        .copy()
        .sort_values(["variant_name", "entry_timestamp"], ascending=[True, False])
        .reset_index(drop=True)
    )
    recent_frames: list[pd.DataFrame] = []
    for variant_name, group in signal_log[signal_log["status"] == "closed"].groupby("variant_name", sort=True):
        recent_frames.append(group.sort_values("entry_timestamp", ascending=False).head(int(recent_count)))
    recent = pd.concat(recent_frames, axis=0, ignore_index=True) if recent_frames else pd.DataFrame()
    if not recent.empty:
        recent = recent.sort_values(["variant_name", "entry_timestamp"], ascending=[True, False]).reset_index(drop=True)
    return active, recent


def build_variant_daily_summary(signal_log: pd.DataFrame) -> pd.DataFrame:
    if signal_log.empty:
        return pd.DataFrame()
    closed = signal_log[signal_log["status"] == "closed"].copy()
    if closed.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for variant_name, group in closed.groupby("variant_name", sort=True):
        ordered_group = group.sort_values("entry_timestamp").copy()
        running_equity = 1.0
        for trade_date, date_group in ordered_group.groupby(ordered_group["entry_timestamp"].dt.normalize(), sort=True):
            ordered = date_group.sort_values("entry_timestamp")
            day_curve = (1.0 + ordered["realized_net_return"]).cumprod()
            running_equity *= float(day_curve.iloc[-1])
            rows.append(
                {
                    "variant_name": variant_name,
                    "trade_date": trade_date,
                    "trade_count": int(len(ordered)),
                    "mean_net_return": float(ordered["realized_net_return"].mean()),
                    "day_cumulative_net_return": float(day_curve.iloc[-1] - 1.0),
                    "running_cumulative_net_return": float(running_equity - 1.0),
                    "win_rate": float((ordered["realized_net_return"] > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def build_variant_tracker_report(
    signal_log: pd.DataFrame,
    tracker_summary: pd.DataFrame,
    active_signals: pd.DataFrame,
    recent_signals: pd.DataFrame,
    daily_summary: pd.DataFrame,
    as_of_utc: pd.Timestamp,
    plot_paths: list[str],
) -> str:
    lines = ["# Tick Variant Tracker", ""]
    lines.append("## 기준 시각")
    lines.append(f"- as_of_utc: {as_of_utc}")
    lines.append("")

    lines.append("## 현재 상태")
    if signal_log.empty:
        lines.append("- 신호 로그가 없습니다.")
    else:
        for variant_name, group in signal_log.groupby("variant_name", sort=True):
            last_signal = group.sort_values("entry_timestamp").iloc[-1]
            lines.append(
                f"- {variant_name}: 누적 {int(len(group))}건, 종료 {int((group['status'] == 'closed').sum())}건, "
                f"활성 {int((group['status'] == 'active').sum())}건, 마지막 {last_signal['entry_timestamp']} | net {float(last_signal['net_return']):.4%}"
            )
    lines.append("")

    lines.append("## 기간별 추적 요약")
    if tracker_summary.empty:
        lines.append("- 요약 결과가 없습니다.")
    else:
        for variant_name, group in tracker_summary.groupby("variant_name", sort=True):
            lines.append(f"- {variant_name}")
            for _, row in group.iterrows():
                lines.append(
                    f"  - {row['period_name']}: 거래 {int(row['trade_count'])}건, 평균 순수익 {row['mean_net_return']:.4%}, "
                    f"승률 {row['win_rate']:.2%}, 누적 {row['terminal_cumulative_net_return']:.2%}"
                )
    lines.append("")

    lines.append("## 활성 신호")
    if active_signals.empty:
        lines.append("- 현재 활성 신호는 없습니다.")
    else:
        for _, row in active_signals.iterrows():
            lines.append(
                f"- {row['variant_name']} | {row['signal_id']} | {row['entry_timestamp']} -> {row['exit_timestamp']} | strength {float(row['strength_ratio']):.2f}"
            )
    lines.append("")

    lines.append("## 최근 종료 신호")
    if recent_signals.empty:
        lines.append("- 최근 종료 신호가 없습니다.")
    else:
        for variant_name, group in recent_signals.groupby("variant_name", sort=True):
            lines.append(f"- {variant_name}")
            for _, row in group.head(6).iterrows():
                lines.append(
                    f"  - {row['signal_id']}: {row['entry_timestamp']} | net {float(row['realized_net_return']):.4%} | strength {float(row['strength_ratio']):.2f}"
                )
    lines.append("")

    lines.append("## 일별 흐름")
    if daily_summary.empty:
        lines.append("- 일별 요약이 없습니다.")
    else:
        for variant_name, group in daily_summary.groupby("variant_name", sort=True):
            positive_days = float((group["day_cumulative_net_return"] > 0).mean())
            last_cum = float(group.sort_values("trade_date").iloc[-1]["running_cumulative_net_return"])
            lines.append(f"- {variant_name}: 양(+) 일 비중 {positive_days:.2%}, 마지막 누적 순수익 {last_cum:.2%}")
    lines.append("")

    lines.append("## 플롯")
    for plot_path in plot_paths:
        lines.append(f"- {plot_path}")
    lines.append("")
    return "\n".join(lines)


def plot_variant_tracker_cumulative(daily_summary: pd.DataFrame, path: str | Path) -> None:
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(10, 5))
    if daily_summary.empty:
        _render_empty_plot(fig, ax, "Dual Tracker Cumulative", "표시할 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    for variant_name, group in daily_summary.groupby("variant_name", sort=True):
        ordered = group.sort_values("trade_date")
        ax.plot(
            ordered["trade_date"],
            ordered["running_cumulative_net_return"] * 100.0,
            linewidth=1.6,
            label=variant_name,
        )
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_title("규칙별 누적 순수익 추적")
    ax.set_ylabel("누적 순수익 (%)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_variant_recent_signals(recent_signals: pd.DataFrame, path: str | Path) -> None:
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(12, 5))
    if recent_signals.empty:
        _render_empty_plot(fig, ax, "Dual Tracker Recent", "표시할 최근 신호가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    plot_frame = recent_signals.copy()
    plot_frame["label"] = plot_frame["variant_name"] + " | " + plot_frame["entry_timestamp"].astype(str).str[5:16]
    colors = ["#54A24B" if value >= 0 else "#E45756" for value in plot_frame["realized_net_return"]]
    ax.bar(plot_frame["label"], plot_frame["realized_net_return"] * 100.0, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_title("최근 종료 신호 순수익")
    ax.set_ylabel("순수익 (%)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_tracker_row(frame: pd.DataFrame, variant_name: str, period_name: str) -> dict:
    if frame.empty:
        return {
            "variant_name": variant_name,
            "period_name": period_name,
            "trade_count": 0,
            "mean_net_return": np.nan,
            "median_net_return": np.nan,
            "win_rate": np.nan,
            "terminal_cumulative_net_return": np.nan,
        }
    net = frame["realized_net_return"]
    return {
        "variant_name": variant_name,
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

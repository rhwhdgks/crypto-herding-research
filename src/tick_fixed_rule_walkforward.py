from __future__ import annotations

from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_walkforward_summary(
    frame: pd.DataFrame,
    event_label: str,
    horizon_minutes: int,
    block_days: int,
    step_days: int,
) -> pd.DataFrame:
    return_column = f"forward_return_{int(horizon_minutes)}m"
    if return_column not in frame.columns:
        raise ValueError(f"Column not found: {return_column}")

    unique_dates = sorted(frame["bucket_start"].dt.normalize().unique().tolist())
    if not unique_dates:
        return pd.DataFrame()

    event_mask = frame["is_micro_herding_event"] if event_label == "all" else frame["event_label"] == f"micro_herding_{event_label}"
    control_mask = frame["is_control_bucket"]

    rows: list[dict] = []
    fold_id = 1
    max_start = len(unique_dates) - int(block_days)
    if max_start < 0:
        return pd.DataFrame()

    for start_idx in range(0, max_start + 1, int(step_days)):
        block_dates = unique_dates[start_idx : start_idx + int(block_days)]
        if len(block_dates) < int(block_days):
            continue
        block_mask = frame["bucket_start"].dt.normalize().isin(block_dates)
        subset = frame.loc[block_mask].copy()
        event_sample = subset.loc[event_mask.loc[subset.index], return_column].dropna()
        control_sample = subset.loc[control_mask.loc[subset.index], return_column].dropna()

        rows.append(
            {
                "fold_id": int(fold_id),
                "window_start": min(block_dates),
                "window_end": max(block_dates),
                "event_count": int(event_sample.shape[0]),
                "control_count": int(control_sample.shape[0]),
                "event_mean_return": event_sample.mean() if not event_sample.empty else np.nan,
                "control_mean_return": control_sample.mean() if not control_sample.empty else np.nan,
                "delta_mean_return": (
                    event_sample.mean() - control_sample.mean()
                    if not event_sample.empty and not control_sample.empty
                    else np.nan
                ),
                "event_t_stat": _compute_t_stat(event_sample),
                "control_t_stat": _compute_t_stat(control_sample),
                "delta_t_stat": _compute_difference_t_stat(event_sample, control_sample),
                "event_win_rate": (event_sample > 0).mean() if not event_sample.empty else np.nan,
                "control_win_rate": (control_sample > 0).mean() if not control_sample.empty else np.nan,
                "delta_win_rate": (
                    (event_sample > 0).mean() - (control_sample > 0).mean()
                    if not event_sample.empty and not control_sample.empty
                    else np.nan
                ),
            }
        )
        fold_id += 1

    return pd.DataFrame(rows)


def build_walkforward_report(
    summary: pd.DataFrame,
    interval_minutes: int,
    event_label: str,
    horizon_minutes: int,
    symbols: list[str] | None,
    block_days: int,
    step_days: int,
    plot_paths: list[str],
) -> str:
    lines = ["# Tick 고정 룰 순차 블록 검증", ""]
    lines.append("## 고정 룰")
    lines.extend(
        [
            f"- interval: {int(interval_minutes)}분 버킷",
            f"- event_label: {event_label}",
            f"- 목표 반응 horizon: {int(horizon_minutes)}분",
            f"- 심볼 범위: {', '.join(symbols) if symbols else '전체'}",
            f"- 블록 길이: {int(block_days)}일",
            f"- 블록 이동 간격: {int(step_days)}일",
            "- 이 검증은 같은 rule을 순차적인 미래 구간들에 반복 적용해, 특정 한 달에만 좋았는지 확인합니다.",
            "",
        ]
    )

    lines.append("## 블록별 결과")
    if summary.empty:
        lines.append("- 결과가 없습니다.")
    else:
        for _, row in summary.iterrows():
            lines.append(
                f"- fold {int(row['fold_id'])}: {row['window_start']} ~ {row['window_end']} | "
                f"이벤트 {row['event_mean_return']:.4%}, 대조군 {row['control_mean_return']:.4%}, "
                f"차이 {row['delta_mean_return']:.4%}, 차이 t={row['delta_t_stat']:.2f}, "
                f"이벤트 {int(row['event_count'])}건 / 대조군 {int(row['control_count'])}건"
            )
    lines.append("")

    lines.append("## 해석")
    if not summary.empty:
        valid = summary.dropna(subset=["delta_mean_return"]).copy()
        positive_share = float((valid["delta_mean_return"] > 0).mean()) if not valid.empty else np.nan
        best = valid.sort_values("delta_mean_return", ascending=False).iloc[0] if not valid.empty else None
        worst = valid.sort_values("delta_mean_return", ascending=True).iloc[0] if not valid.empty else None
        if best is not None:
            lines.append(
                f"- 최고 블록은 fold {int(best['fold_id'])} ({best['window_start']} ~ {best['window_end']})이고, "
                f"차이는 {best['delta_mean_return']:.4%}입니다."
            )
        if worst is not None:
            lines.append(
                f"- 최저 블록은 fold {int(worst['fold_id'])} ({worst['window_start']} ~ {worst['window_end']})이고, "
                f"차이는 {worst['delta_mean_return']:.4%}입니다."
            )
        if not np.isnan(positive_share):
            lines.append(f"- 양(+) 블록 비중은 {positive_share:.2%}입니다.")
        if not np.isnan(positive_share) and positive_share >= 0.6:
            lines.append("- 양(+) 블록 비중이 절반을 넘어서, 고정 룰의 일관성이 어느 정도 남아 있습니다.")
        else:
            lines.append("- 양(+) 블록 비중이 낮아, 최근 성과가 특정 regime에 치우쳤을 가능성을 열어둬야 합니다.")
    lines.append("- 즉 이 리포트는 '최근 몇 주의 반등'이 아니라, 같은 규칙이 여러 30일 블록에서 얼마나 반복되는지 보는 용도입니다.")
    lines.append("")

    lines.append("## 플롯")
    for plot_path in plot_paths:
        lines.append(f"- {plot_path}")
    lines.append("")
    return "\n".join(lines)


def plot_walkforward_delta(summary: pd.DataFrame, path: str | Path) -> None:
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(10, 5))
    if summary.empty:
        _render_empty_plot(fig, ax, "Tick 고정 룰 순차 블록 차이", "표시할 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    plot_frame = summary.copy()
    labels = [f"F{int(value)}" for value in plot_frame["fold_id"]]
    colors = ["#54A24B" if value >= 0 else "#E45756" for value in plot_frame["delta_mean_return"]]
    ax.bar(labels, plot_frame["delta_mean_return"] * 100.0, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("이벤트-대조군 차이 (%)")
    ax.set_title("Tick 고정 룰 30일 블록별 차이")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _compute_t_stat(sample: pd.Series) -> float:
    if sample.shape[0] < 2:
        return np.nan
    std = sample.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return float(sample.mean() / (std / sqrt(sample.shape[0])))


def _compute_difference_t_stat(event_sample: pd.Series, control_sample: pd.Series) -> float:
    if event_sample.shape[0] < 2 or control_sample.shape[0] < 2:
        return np.nan
    event_var = event_sample.var(ddof=1)
    control_var = control_sample.var(ddof=1)
    if np.isnan(event_var) or np.isnan(control_var):
        return np.nan
    denominator = np.sqrt((event_var / event_sample.shape[0]) + (control_var / control_sample.shape[0]))
    if denominator == 0 or np.isnan(denominator):
        return np.nan
    return float((event_sample.mean() - control_sample.mean()) / denominator)


def _configure_korean_matplotlib_font() -> None:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False


def _render_empty_plot(fig: plt.Figure, ax: plt.Axes, title: str, message: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tick_fixed_rule_oos import load_micro_frame


def compute_rolling_regime_summary(
    frame: pd.DataFrame,
    event_label: str,
    horizon_minutes: int,
    window_days_list: list[int],
) -> pd.DataFrame:
    return_column = f"forward_return_{int(horizon_minutes)}m"
    if return_column not in frame.columns:
        raise ValueError(f"Column not found: {return_column}")

    event_mask = frame["is_micro_herding_event"] if event_label == "all" else frame["event_label"] == f"micro_herding_{event_label}"
    control_mask = frame["is_control_bucket"]

    unique_dates = sorted(frame["bucket_start"].dt.normalize().unique().tolist())
    rows: list[dict] = []

    for window_days in window_days_list:
        if window_days > len(unique_dates):
            continue
        for end_idx in range(window_days - 1, len(unique_dates)):
            window_dates = set(unique_dates[end_idx - window_days + 1 : end_idx + 1])
            subset = frame[frame["bucket_start"].dt.normalize().isin(window_dates)].copy()
            event_sample = subset.loc[event_mask.loc[subset.index], return_column].dropna()
            control_sample = subset.loc[control_mask.loc[subset.index], return_column].dropna()
            rows.append(
                {
                    "window_days": int(window_days),
                    "window_start": min(window_dates),
                    "window_end": max(window_dates),
                    "event_count": int(event_sample.shape[0]),
                    "control_count": int(control_sample.shape[0]),
                    "event_mean_return": event_sample.mean() if not event_sample.empty else np.nan,
                    "control_mean_return": control_sample.mean() if not control_sample.empty else np.nan,
                    "delta_mean_return": (
                        event_sample.mean() - control_sample.mean()
                        if not event_sample.empty and not control_sample.empty
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_regime_report(regime_summary: pd.DataFrame, plot_paths: list[str]) -> str:
    lines = ["# Tick 고정 룰 Regime 점검", ""]
    lines.append("## 목적")
    lines.extend(
        [
            "- 더 긴 표본에서 fixed rule 성과가 왜 구간마다 달라지는지 보기 위해 rolling window 차이를 계산했습니다.",
            "- 관심 값은 항상 `이벤트 평균수익률 - 같은 세션 대조군 평균수익률`입니다.",
            "",
        ]
    )

    for window_days in sorted(regime_summary["window_days"].dropna().unique().tolist()):
        subset = regime_summary[regime_summary["window_days"] == window_days].copy()
        if subset.empty:
            continue
        best = subset.sort_values("delta_mean_return", ascending=False).iloc[0]
        worst = subset.sort_values("delta_mean_return", ascending=True).iloc[0]
        positive_share = float((subset["delta_mean_return"] > 0).mean())
        lines.append(f"## {window_days}일 rolling")
        lines.append(
            f"- 최고 구간: {best['window_start']} ~ {best['window_end']} | 차이 {best['delta_mean_return']:.4%}"
        )
        lines.append(
            f"- 최저 구간: {worst['window_start']} ~ {worst['window_end']} | 차이 {worst['delta_mean_return']:.4%}"
        )
        lines.append(f"- 양(+) window 비중: {positive_share:.2%}")
        lines.append("")

    lines.append("## 해석")
    if not regime_summary.empty:
        twenty = regime_summary[regime_summary["window_days"] == 20].copy()
        if not twenty.empty:
            last_row = twenty.sort_values("window_end").iloc[-1]
            first_row = twenty.sort_values("window_end").iloc[0]
            if float(last_row["delta_mean_return"]) > 0 and float(first_row["delta_mean_return"]) < 0:
                lines.append("- 최근으로 갈수록 delta가 개선된 흔적이 있어, fixed rule은 현재 국면에서만 유효할 가능성이 있습니다.")
            else:
                lines.append("- rolling delta가 한쪽 방향으로만 움직이지 않아, 현재 신호는 아직 불안정합니다.")
    lines.append("- 즉 현재 단계의 정직한 해석은 '최근 구간의 개선은 보이지만, 더 긴 표본 전체로는 아직 regime dependence가 크다'입니다.")
    lines.append("")

    lines.append("## 플롯")
    for plot_path in plot_paths:
        lines.append(f"- {plot_path}")
    lines.append("")
    return "\n".join(lines)


def plot_regime_lines(regime_summary: pd.DataFrame, path: str | Path) -> None:
    configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(11, 5))
    if regime_summary.empty:
        _render_empty_plot(fig, ax, "Tick fixed rule regime", "표시할 regime 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    plot_frame = regime_summary.copy()
    plot_frame["window_end"] = pd.to_datetime(plot_frame["window_end"], utc=True)
    for window_days, subset in plot_frame.groupby("window_days"):
        ax.plot(
            subset["window_end"],
            subset["delta_mean_return"] * 100.0,
            marker="o",
            linewidth=1.3,
            markersize=3.0,
            label=f"{int(window_days)}일",
        )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("이벤트-대조군 차이 (%)")
    ax.set_title("Tick fixed rule rolling regime")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def configure_korean_matplotlib_font() -> None:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False


def _render_empty_plot(fig: plt.Figure, ax: plt.Axes, title: str, message: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()

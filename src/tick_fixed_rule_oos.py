from __future__ import annotations

from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_micro_frame(base_dir: str | Path, interval_minutes: int) -> pd.DataFrame:
    path = Path(base_dir) / "intermediate" / f"tick_micro_frame_{int(interval_minutes)}m.csv"
    frame = pd.read_csv(path, parse_dates=["bucket_start"])
    if frame.empty:
        return frame
    timestamp = frame["bucket_start"]
    if timestamp.dt.tz is None:
        timestamp = timestamp.dt.tz_localize("UTC")
    else:
        timestamp = timestamp.dt.tz_convert("UTC")
    frame["bucket_start"] = timestamp
    return frame.sort_values(["symbol", "bucket_start"]).reset_index(drop=True)


def build_period_masks(frame: pd.DataFrame, holdout_days: int) -> dict[str, pd.Series]:
    normalized_dates = sorted(frame["bucket_start"].dt.normalize().unique().tolist())
    holdout_days = min(int(holdout_days), len(normalized_dates))
    holdout_dates = set(normalized_dates[-holdout_days:])
    development_dates = set(normalized_dates[:-holdout_days]) if holdout_days < len(normalized_dates) else set()

    masks: dict[str, pd.Series] = {
        "full_sample": pd.Series(True, index=frame.index),
        "development": frame["bucket_start"].dt.normalize().isin(development_dates),
        "holdout": frame["bucket_start"].dt.normalize().isin(holdout_dates),
    }

    if len(normalized_dates) >= 3:
        chunk = len(normalized_dates) // 3
        first_dates = set(normalized_dates[:chunk])
        second_dates = set(normalized_dates[chunk : 2 * chunk])
        third_dates = set(normalized_dates[2 * chunk :])
        masks["block_1"] = frame["bucket_start"].dt.normalize().isin(first_dates)
        masks["block_2"] = frame["bucket_start"].dt.normalize().isin(second_dates)
        masks["block_3"] = frame["bucket_start"].dt.normalize().isin(third_dates)
    return masks


def summarize_fixed_rule(
    frame: pd.DataFrame,
    event_label: str,
    horizon_minutes: int,
    holdout_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return_column = f"forward_return_{int(horizon_minutes)}m"
    if return_column not in frame.columns:
        raise ValueError(f"Column not found: {return_column}")

    period_masks = build_period_masks(frame, holdout_days=holdout_days)
    summary_rows: list[dict] = []
    symbol_rows: list[dict] = []

    event_mask = frame["is_micro_herding_event"] if event_label == "all" else frame["event_label"] == f"micro_herding_{event_label}"
    control_mask = frame["is_control_bucket"]

    for period_name, period_mask in period_masks.items():
        period_frame = frame[period_mask].copy()
        if period_frame.empty:
            continue

        event_sample = period_frame.loc[event_mask.loc[period_frame.index], return_column].dropna()
        control_sample = period_frame.loc[control_mask.loc[period_frame.index], return_column].dropna()
        summary_rows.append(
            _build_summary_row(
                period_name=period_name,
                symbol="pooled",
                event_sample=event_sample,
                control_sample=control_sample,
            )
        )

        for symbol, symbol_frame in period_frame.groupby("symbol"):
            symbol_event_sample = symbol_frame.loc[event_mask.loc[symbol_frame.index], return_column].dropna()
            symbol_control_sample = symbol_frame.loc[control_mask.loc[symbol_frame.index], return_column].dropna()
            symbol_rows.append(
                _build_summary_row(
                    period_name=period_name,
                    symbol=symbol,
                    event_sample=symbol_event_sample,
                    control_sample=symbol_control_sample,
                )
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(symbol_rows)


def build_fixed_rule_report(
    interval_minutes: int,
    event_label: str,
    horizon_minutes: int,
    holdout_days: int,
    symbols: list[str] | None,
    summary: pd.DataFrame,
    symbol_summary: pd.DataFrame,
    plot_paths: list[str],
) -> str:
    lines = ["# Tick 고정 룰 OOS 검증", ""]
    lines.append("## 고정 룰")
    lines.extend(
        [
            f"- interval: {int(interval_minutes)}분 버킷",
            f"- event_label: {event_label}",
            f"- 목표 반응 horizon: {int(horizon_minutes)}분",
            f"- holdout: 마지막 {int(holdout_days)}일",
            f"- 심볼 범위: {', '.join(symbols) if symbols else '전체'}",
            "- 이 검증은 더 이상 조합을 고르지 않고, 미리 정한 rule을 그대로 평가합니다.",
            "",
        ]
    )

    lines.append("## 기간별 pooled 결과")
    if summary.empty:
        lines.append("- 요약 결과가 없습니다.")
    else:
        for _, row in summary.iterrows():
            lines.append(
                f"- {row['period_name']}: 이벤트 {row['event_mean_return']:.4%}, 대조군 {row['control_mean_return']:.4%}, "
                f"차이 {row['delta_mean_return']:.4%}, 차이 t={row['delta_t_stat']:.2f}, "
                f"이벤트 {int(row['event_count'])}건 / 대조군 {int(row['control_count'])}건"
            )
    lines.append("")

    lines.append("## Holdout 심볼별 결과")
    holdout = symbol_summary[symbol_summary["period_name"] == "holdout"].copy()
    if holdout.empty:
        lines.append("- holdout 심볼별 결과가 없습니다.")
    else:
        for _, row in holdout.iterrows():
            lines.append(
                f"- {row['symbol']}: 이벤트 {row['event_mean_return']:.4%}, 대조군 {row['control_mean_return']:.4%}, "
                f"차이 {row['delta_mean_return']:.4%}, 차이 t={row['delta_t_stat']:.2f}"
            )
    lines.append("")

    lines.append("## 해석")
    if not summary.empty:
        holdout_row = summary[summary["period_name"] == "holdout"]
        development_row = summary[summary["period_name"] == "development"]
        if not development_row.empty and not holdout_row.empty:
            dev_delta = float(development_row.iloc[0]["delta_mean_return"])
            holdout_delta = float(holdout_row.iloc[0]["delta_mean_return"])
            if dev_delta > 0 and holdout_delta > 0:
                lines.append("- development와 holdout 모두 이벤트-대조군 차이가 양(+)이라서, rule이 한 구간에만 의존하지는 않습니다.")
            elif dev_delta > 0 and holdout_delta <= 0:
                lines.append("- development에서는 좋았지만 holdout에서 약해져, 아직 자동매매로 바로 넘기기엔 이릅니다.")
            else:
                lines.append("- fixed rule의 우위가 아직 안정적이지 않아 추가 OOS가 더 필요합니다.")
        best_holdout = holdout.sort_values(["delta_mean_return", "delta_t_stat"], ascending=False).iloc[0] if not holdout.empty else None
        if best_holdout is not None:
            lines.append(
                f"- 현재 holdout에서 가장 좋은 심볼은 {best_holdout['symbol']}이고, 이벤트-대조군 차이는 {best_holdout['delta_mean_return']:.4%}입니다."
            )
    lines.append("- 이 단계는 전략 구현이 아니라, short-horizon alpha 후보를 고정 룰로 검증하는 연구 단계입니다.")
    lines.append("")

    lines.append("## 플롯")
    for plot_path in plot_paths:
        lines.append(f"- {plot_path}")
    lines.append("")
    return "\n".join(lines)


def filter_micro_frame_by_symbols(frame: pd.DataFrame, symbols: list[str] | None) -> pd.DataFrame:
    if not symbols:
        return frame.copy()
    symbol_set = {str(symbol) for symbol in symbols}
    return frame[frame["symbol"].isin(symbol_set)].copy().reset_index(drop=True)


def plot_period_delta(summary: pd.DataFrame, path: str | Path) -> None:
    configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(10, 5))
    if summary.empty:
        _render_empty_plot(fig, ax, "고정 룰 기간별 차이", "표시할 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    plot_frame = summary.copy()
    colors = ["#54A24B" if value >= 0 else "#E45756" for value in plot_frame["delta_mean_return"]]
    ax.bar(plot_frame["period_name"], plot_frame["delta_mean_return"] * 100.0, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("이벤트-대조군 차이 (%)")
    ax.set_title("Tick 고정 룰 기간별 차이")
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_holdout_symbol_delta(symbol_summary: pd.DataFrame, path: str | Path) -> None:
    configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(8, 5))
    holdout = symbol_summary[symbol_summary["period_name"] == "holdout"].copy()
    if holdout.empty:
        _render_empty_plot(fig, ax, "Holdout 심볼별 차이", "표시할 holdout 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    colors = ["#54A24B" if value >= 0 else "#E45756" for value in holdout["delta_mean_return"]]
    ax.bar(holdout["symbol"], holdout["delta_mean_return"] * 100.0, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("이벤트-대조군 차이 (%)")
    ax.set_title("Holdout 심볼별 고정 룰 차이")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def configure_korean_matplotlib_font() -> None:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False


def _build_summary_row(period_name: str, symbol: str, event_sample: pd.Series, control_sample: pd.Series) -> dict:
    return {
        "period_name": period_name,
        "symbol": symbol,
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


def _render_empty_plot(fig: plt.Figure, ax: plt.Axes, title: str, message: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()

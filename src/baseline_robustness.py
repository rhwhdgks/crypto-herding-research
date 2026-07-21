from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from event_study import run_event_study
from regression import run_csad_regression
from utils import horizon_to_label


def configure_korean_matplotlib_font() -> None:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False


def load_baseline_intermediate_outputs(base_dir: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    base_dir = Path(base_dir)
    analysis_frame = pd.read_csv(
        base_dir / "intermediate" / "analysis_frame.csv",
        parse_dates=["timestamp"],
    ).set_index("timestamp")
    market_return = pd.read_csv(
        base_dir / "intermediate" / "market_return_series.csv",
        parse_dates=["timestamp"],
    ).set_index("timestamp")["market_return"]

    analysis_frame = analysis_frame.sort_index()
    market_return = market_return.sort_index()
    analysis_frame = analysis_frame.loc[analysis_frame.index.intersection(market_return.index)]
    market_return = market_return.loc[analysis_frame.index]
    return analysis_frame, market_return


def build_chronological_segment_specs(index: pd.DatetimeIndex) -> list[dict]:
    specs: list[dict] = []
    total = len(index)
    if total == 0:
        return specs

    specs.append(
        {
            "segment_group": "chronological",
            "segment_name": "full_sample",
            "display_name": "전체 표본",
            "mask": pd.Series(True, index=index),
            "contiguous": True,
        }
    )

    half_masks = {
        "first_half": (0, total // 2),
        "second_half": (total // 2, total),
    }
    half_display = {
        "first_half": "전반부",
        "second_half": "후반부",
    }
    for name, (start, end) in half_masks.items():
        mask = pd.Series(False, index=index)
        mask.iloc[start:end] = True
        specs.append(
            {
                "segment_group": "chronological",
                "segment_name": name,
                "display_name": half_display[name],
                "mask": mask,
                "contiguous": True,
            }
        )

    quarter_indices = np.array_split(np.arange(total), 4)
    for idx, positions in enumerate(quarter_indices, start=1):
        mask = pd.Series(False, index=index)
        mask.iloc[positions] = True
        specs.append(
            {
                "segment_group": "chronological",
                "segment_name": f"q{idx}",
                "display_name": f"4분할 구간 {idx}",
                "mask": mask,
                "contiguous": True,
            }
        )
    return specs


def build_session_segment_specs(index: pd.DatetimeIndex) -> list[dict]:
    hour = pd.Series(index.hour, index=index)
    return [
        {
            "segment_group": "session",
            "segment_name": "utc_00_07",
            "display_name": "UTC 00-07 / KST 09-16",
            "mask": (hour >= 0) & (hour < 8),
            "contiguous": False,
        },
        {
            "segment_group": "session",
            "segment_name": "utc_08_15",
            "display_name": "UTC 08-15 / KST 17-00",
            "mask": (hour >= 8) & (hour < 16),
            "contiguous": False,
        },
        {
            "segment_group": "session",
            "segment_name": "utc_16_23",
            "display_name": "UTC 16-23 / KST 01-08",
            "mask": (hour >= 16) & (hour < 24),
            "contiguous": False,
        },
    ]


def build_day_type_segment_specs(index: pd.DatetimeIndex) -> list[dict]:
    day_of_week = pd.Series(index.dayofweek, index=index)
    return [
        {
            "segment_group": "day_type",
            "segment_name": "weekday",
            "display_name": "주중",
            "mask": day_of_week < 5,
            "contiguous": False,
        },
        {
            "segment_group": "day_type",
            "segment_name": "weekend",
            "display_name": "주말",
            "mask": day_of_week >= 5,
            "contiguous": False,
        },
    ]


def build_volatility_segment_specs(
    analysis_frame: pd.DataFrame,
    low_quantile: float = 0.30,
    high_quantile: float = 0.70,
) -> list[dict]:
    volatility = analysis_frame["rolling_volatility"]
    valid = volatility.dropna()
    if valid.empty:
        return []

    low_threshold = float(valid.quantile(low_quantile))
    high_threshold = float(valid.quantile(high_quantile))

    return [
        {
            "segment_group": "volatility_regime",
            "segment_name": "low_vol",
            "display_name": f"저변동성 하위 {int(low_quantile * 100)}%",
            "mask": volatility <= low_threshold,
            "contiguous": False,
            "threshold_value": low_threshold,
        },
        {
            "segment_group": "volatility_regime",
            "segment_name": "mid_vol",
            "display_name": "중간 변동성",
            "mask": (volatility > low_threshold) & (volatility < high_threshold),
            "contiguous": False,
            "threshold_value": np.nan,
        },
        {
            "segment_group": "volatility_regime",
            "segment_name": "high_vol",
            "display_name": f"고변동성 상위 {int((1.0 - high_quantile) * 100)}%",
            "mask": volatility >= high_threshold,
            "contiguous": False,
            "threshold_value": high_threshold,
        },
    ]


def summarize_segments(
    analysis_frame: pd.DataFrame,
    market_return: pd.Series,
    segment_specs: list[dict],
    holding_periods: Iterable[int],
    focus_horizons: Iterable[int] = (15, 120, 1440),
    min_observations: int = 10_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holding_periods = [int(period) for period in holding_periods]
    focus_horizons = {int(period) for period in focus_horizons}

    regression_rows: list[dict] = []
    event_rows: list[dict] = []
    best_rows: list[dict] = []

    for spec in segment_specs:
        raw_mask = pd.Series(spec["mask"], index=analysis_frame.index).fillna(False)
        subset = analysis_frame.loc[raw_mask].copy()
        if len(subset) < min_observations:
            continue

        regression_results, _, _, _, regression_json = run_csad_regression(
            subset["csad"],
            subset["market_return"],
        )
        beta2_row = regression_results.loc["market_return_sq"]

        event_market_return = subset["market_return"] if spec.get("contiguous", False) else market_return
        _, event_summary, _, _ = run_event_study(
            analysis_frame=subset,
            market_return=event_market_return,
            holding_periods=holding_periods,
            event_label_column="event_type",
            event_types=["low_dispersion", "shock"],
            max_path_horizon=max(holding_periods),
        )

        regression_rows.append(
            {
                "segment_group": spec["segment_group"],
                "segment_name": spec["segment_name"],
                "display_name": spec["display_name"],
                "start": subset.index.min(),
                "end": subset.index.max(),
                "observations": int(len(subset)),
                "beta2": float(regression_json["beta2"]),
                "beta2_t_stat": float(regression_json["beta2_t_stat"]),
                "beta2_p_value": float(regression_json["beta2_p_value"]),
                "rsquared": float(regression_json["rsquared"]),
                "low_dispersion_count": int((subset["event_type"] == "low_dispersion").sum()),
                "shock_count": int((subset["event_type"] == "shock").sum()),
                "negative_beta2": bool(float(regression_json["beta2"]) < 0.0),
            }
        )

        if event_summary.empty:
            continue

        event_summary = event_summary.copy()
        event_summary.insert(0, "segment_group", spec["segment_group"])
        event_summary.insert(1, "segment_name", spec["segment_name"])
        event_summary.insert(2, "display_name", spec["display_name"])
        event_summary.insert(3, "start", subset.index.min())
        event_summary.insert(4, "end", subset.index.max())
        event_summary.insert(5, "observations", int(len(subset)))
        event_summary.insert(6, "beta2", float(regression_json["beta2"]))
        event_summary.insert(7, "beta2_t_stat", float(regression_json["beta2_t_stat"]))
        event_summary.insert(8, "beta2_p_value", float(regression_json["beta2_p_value"]))
        event_rows.extend(event_summary.to_dict("records"))

        for event_type in ["low_dispersion", "shock"]:
            subset_summary = event_summary[event_summary["event_type"] == event_type].dropna(subset=["mean_return"])
            if subset_summary.empty:
                continue

            best_row = subset_summary.sort_values("mean_return", ascending=False).iloc[0]
            best_rows.append(
                {
                    "segment_group": spec["segment_group"],
                    "segment_name": spec["segment_name"],
                    "display_name": spec["display_name"],
                    "start": subset.index.min(),
                    "end": subset.index.max(),
                    "observations": int(len(subset)),
                    "beta2": float(regression_json["beta2"]),
                    "beta2_t_stat": float(regression_json["beta2_t_stat"]),
                    "event_type": event_type,
                    "event_count": int(best_row["count"]),
                    "best_horizon": best_row["horizon_label"],
                    "best_mean_return": float(best_row["mean_return"]),
                    "best_t_stat": float(best_row["t_stat"]) if pd.notna(best_row["t_stat"]) else np.nan,
                    "best_win_rate": float(best_row["win_rate"]) if pd.notna(best_row["win_rate"]) else np.nan,
                }
            )

    regression_summary = pd.DataFrame(regression_rows)
    event_summary = pd.DataFrame(event_rows)
    focus_summary = event_summary[event_summary["horizon_minutes"].isin(focus_horizons)].copy() if not event_summary.empty else pd.DataFrame()
    best_summary = pd.DataFrame(best_rows)
    return regression_summary, best_summary, focus_summary


def compute_rolling_beta2(
    analysis_frame: pd.DataFrame,
    window_minutes: int,
    step_minutes: int,
    min_observations: int | None = None,
) -> pd.DataFrame:
    min_observations = int(min_observations or window_minutes)
    if len(analysis_frame) < min_observations:
        return pd.DataFrame()

    rows: list[dict] = []
    for start_position in range(0, len(analysis_frame) - window_minutes + 1, step_minutes):
        subset = analysis_frame.iloc[start_position:start_position + window_minutes]
        if len(subset) < min_observations:
            continue

        _, _, _, _, regression_json = run_csad_regression(
            subset["csad"],
            subset["market_return"],
        )
        rows.append(
            {
                "window_start": subset.index.min(),
                "window_end": subset.index.max(),
                "window_minutes": int(window_minutes),
                "step_minutes": int(step_minutes),
                "observations": int(len(subset)),
                "beta2": float(regression_json["beta2"]),
                "beta2_t_stat": float(regression_json["beta2_t_stat"]),
                "beta2_p_value": float(regression_json["beta2_p_value"]),
                "negative_beta2": bool(float(regression_json["beta2"]) < 0.0),
                "significant_negative_beta2": bool(
                    float(regression_json["beta2"]) < 0.0 and float(regression_json["beta2_p_value"]) < 0.05
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_group_beta2(
    regression_summary: pd.DataFrame,
    segment_group: str,
    order: list[str],
    path: str | Path,
    title: str,
) -> None:
    configure_korean_matplotlib_font()
    subset = regression_summary[regression_summary["segment_group"] == segment_group].copy()
    subset = subset.set_index("segment_name").reindex(order).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    if subset.empty:
        _render_empty_plot(fig, ax, title, "표시할 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    colors = np.where(subset["beta2"] < 0.0, "#54A24B", "#E45756")
    ax.bar(subset["display_name"], subset["beta2"], color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("beta2")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    fig.autofmt_xdate(rotation=20)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_beta2(rolling_summary: pd.DataFrame, path: str | Path, title: str) -> None:
    configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(11, 5))
    if rolling_summary.empty:
        _render_empty_plot(fig, ax, title, "표시할 rolling 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    ordered = rolling_summary.sort_values("window_end")
    ax.plot(ordered["window_end"], ordered["beta2"], color="#4C78A8", linewidth=1.4)
    negative = ordered[ordered["beta2"] < 0.0]
    if not negative.empty:
        ax.scatter(negative["window_end"], negative["beta2"], color="#E45756", s=18, label="beta2 < 0")

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("beta2")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    if not negative.empty:
        ax.legend(frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_focus_returns(
    focus_summary: pd.DataFrame,
    segment_group: str,
    horizon_minutes: int,
    order: list[str],
    path: str | Path,
    title: str,
) -> None:
    configure_korean_matplotlib_font()
    subset = focus_summary[
        (focus_summary["segment_group"] == segment_group)
        & (focus_summary["horizon_minutes"] == int(horizon_minutes))
    ].copy()
    subset = subset.set_index(["segment_name", "event_type"]).sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    if subset.empty:
        _render_empty_plot(fig, axes[0], title, "표시할 조건부 event-study 결과가 없습니다.")
        axes[1].axis("off")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    for axis, event_type in zip(axes, ["low_dispersion", "shock"]):
        values = []
        labels = []
        for segment_name in order:
            key = (segment_name, event_type)
            if key not in subset.index:
                continue
            row = subset.loc[key]
            labels.append(row["display_name"])
            values.append(float(row["mean_return"]) * 100.0)

        colors = ["#54A24B" if value >= 0.0 else "#E45756" for value in values]
        axis.bar(labels, values, color=colors)
        axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
        axis.set_title(f"{event_type} / {horizon_to_label(horizon_minutes)}")
        axis.set_ylabel("평균 선행 수익률 (%)")
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", rotation=20)

    fig.suptitle(title)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_baseline_robustness_report(
    regression_summary: pd.DataFrame,
    best_summary: pd.DataFrame,
    focus_summary: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    plot_paths: list[str],
) -> str:
    lines = ["# Baseline Robustness 분석", ""]

    chrono = regression_summary[regression_summary["segment_group"] == "chronological"].copy()
    lines.append("## 전체 결론")
    if chrono.empty:
        lines.append("- chronological 분석 결과가 없습니다.")
    else:
        full_row = chrono[chrono["segment_name"] == "full_sample"].iloc[0]
        negative_count = int((chrono["beta2"] < 0.0).sum())
        lines.extend(
            [
                f"- 전체 표본 beta2는 {full_row['beta2']:.4f}이며, t-통계량은 {full_row['beta2_t_stat']:.2f}입니다.",
                f"- 반기/분기 cut {len(chrono) - 1}개 중 beta2가 음수인 구간은 {negative_count}개였습니다.",
                "- 즉 full sample 결과가 단순 평균 착시라기보다, baseline CSAD 자체가 최근 2년 Binance 1분봉에서는 일관되게 양(+)의 beta2를 보이는 쪽에 가깝습니다.",
            ]
        )
    lines.append("")

    lines.append("## 반기·분기 분해")
    if chrono.empty:
        lines.append("- 반기·분기 결과가 없습니다.")
    else:
        quarter_rows = chrono[chrono["segment_name"].isin(["q1", "q2", "q3", "q4"])].copy()
        for _, row in quarter_rows.iterrows():
            lines.append(
                f"- {row['display_name']} ({row['start']} ~ {row['end']}): beta2 {row['beta2']:.4f}, "
                f"HAC t={row['beta2_t_stat']:.2f}, low_dispersion {int(row['low_dispersion_count'])}건, "
                f"shock {int(row['shock_count'])}건"
            )

        quarter_1d = focus_summary[
            (focus_summary["segment_group"] == "chronological")
            & (focus_summary["segment_name"].isin(["q1", "q2", "q3", "q4"]))
            & (focus_summary["horizon_minutes"] == 1440)
        ].copy()
        if not quarter_1d.empty:
            top_positive = quarter_1d.sort_values("mean_return", ascending=False).iloc[0]
            most_negative = quarter_1d.sort_values("mean_return", ascending=True).iloc[0]
            lines.extend(
                [
                    f"- 1일 보유 기준 최고 구간은 {top_positive['display_name']} / {top_positive['event_type']}로 "
                    f"평균 {top_positive['mean_return']:.4%}, UTC-day block p={top_positive['p_value_block']:.4f}, "
                    f"95% CI [{top_positive['confidence_interval_lower']:.4%}, {top_positive['confidence_interval_upper']:.4%}]였습니다.",
                    f"- 가장 약한 구간은 {most_negative['display_name']} / {most_negative['event_type']}로 "
                    f"평균 {most_negative['mean_return']:.4%}, UTC-day block p={most_negative['p_value_block']:.4f}, "
                    f"95% CI [{most_negative['confidence_interval_lower']:.4%}, {most_negative['confidence_interval_upper']:.4%}]였습니다.",
                    "- 이 분할은 사후 exploratory diagnostic이며 여러 구간 중 극값을 선택했으므로 별도 FDR 없이 예측 신호로 해석하지 않습니다.",
                ]
            )
    lines.append("")

    lines.append("## 시간대와 주중·주말")
    session_1d = focus_summary[
        (focus_summary["segment_group"] == "session")
        & (focus_summary["horizon_minutes"] == 1440)
    ].copy()
    if session_1d.empty:
        lines.append("- 시간대 분석 결과가 없습니다.")
    else:
        for event_type in ["low_dispersion", "shock"]:
            subset = session_1d[session_1d["event_type"] == event_type].sort_values("mean_return", ascending=False)
            if subset.empty:
                continue
            best = subset.iloc[0]
            worst = subset.iloc[-1]
            lines.append(_describe_extreme_pair(event_type, "시간대", best, worst))

    day_type_1d = focus_summary[
        (focus_summary["segment_group"] == "day_type")
        & (focus_summary["horizon_minutes"] == 1440)
    ].copy()
    if not day_type_1d.empty:
        for event_type in ["low_dispersion", "shock"]:
            subset = day_type_1d[day_type_1d["event_type"] == event_type].sort_values("mean_return", ascending=False)
            if subset.empty:
                continue
            best = subset.iloc[0]
            worst = subset.iloc[-1]
            lines.append(_describe_extreme_pair(event_type, "요일 구분", best, worst))
    lines.append("")

    lines.append("## 변동성 상태")
    volatility_1d = focus_summary[
        (focus_summary["segment_group"] == "volatility_regime")
        & (focus_summary["horizon_minutes"] == 1440)
    ].copy()
    if volatility_1d.empty:
        lines.append("- 변동성 상태 분석 결과가 없습니다.")
    else:
        for event_type in ["low_dispersion", "shock"]:
            subset = volatility_1d[volatility_1d["event_type"] == event_type].sort_values("mean_return", ascending=False)
            if subset.empty:
                continue
            best = subset.iloc[0]
            worst = subset.iloc[-1]
            lines.append(_describe_extreme_pair(event_type, "변동성 상태", best, worst))
        lines.append("- 변동성 상태 차이는 사후 진단입니다. 새 조건을 도입하려면 별도 train 구간에서 사전 등록하고 이후 OOS에 고정해야 합니다.")
    lines.append("")

    lines.append("## 30일 Rolling beta2")
    if rolling_summary.empty:
        lines.append("- rolling 회귀 결과가 없습니다.")
    else:
        negative_windows = int(rolling_summary["negative_beta2"].sum())
        significant_negative_windows = int(rolling_summary["significant_negative_beta2"].sum())
        total_windows = int(len(rolling_summary))
        lines.extend(
            [
                f"- 30일 창을 1일 간격으로 민 결과 전체 {total_windows}개 창 중 beta2<0 창은 {negative_windows}개였습니다.",
                f"- 그중 5% 유의수준에서 유의한 음수 beta2 창은 {significant_negative_windows}개였습니다.",
                f"- rolling beta2 범위는 {rolling_summary['beta2'].min():.4f} ~ {rolling_summary['beta2'].max():.4f}였습니다.",
            ]
        )
        negative_subset = rolling_summary[rolling_summary["negative_beta2"]].sort_values("window_start")
        if not negative_subset.empty:
            first_negative = negative_subset.iloc[0]
            last_negative = negative_subset.iloc[-1]
            lines.extend(
                [
                    f"- 음수 창은 주로 {first_negative['window_start']} ~ {first_negative['window_end']}와 "
                    f"{last_negative['window_start']} ~ {last_negative['window_end']} 부근에 나타났습니다.",
                    "- 즉 허딩 해석이 완전히 없는 것은 아니지만, 최근 2년 전체를 대표하는 안정적 현상이라기보다 드문 국소 구간에 가깝습니다.",
                ]
            )
    lines.append("")

    lines.append("## 실무적 해석")
    lines.extend(
        [
            "- baseline full sample을 그대로 신호로 쓰는 것은 설득력이 약합니다. 회귀는 거의 전 구간에서 양(+)의 beta2를 보입니다.",
            "- 분기·시간대·변동성 분해는 후보 생성용 exploratory 결과이며, 같은 표본에서 threshold를 다시 맞추면 과적합입니다.",
            "- 새 regime이나 sentiment 조건은 train에서 고정한 뒤 엄격히 이후 OOS에서만 평가합니다.",
        ]
    )
    lines.append("")

    lines.append("## 생성 플롯")
    for plot_path in plot_paths:
        lines.append(f"- {plot_path}")
    lines.append("")

    return "\n".join(lines)


def _render_empty_plot(fig: plt.Figure, ax: plt.Axes, title: str, message: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def _describe_extreme_pair(event_type: str, context: str, best: pd.Series, worst: pd.Series) -> str:
    best_value = float(best["mean_return"])
    worst_value = float(worst["mean_return"])

    if best_value > 0.0:
        best_phrase = f"가장 강한 쪽은 {best['display_name']} ({best_value:.4%}, block p={best['p_value_block']:.4f})"
    else:
        best_phrase = f"가장 덜 약한 쪽은 {best['display_name']} ({best_value:.4%}, block p={best['p_value_block']:.4f})"

    if worst_value < 0.0:
        worst_phrase = f"가장 약한 쪽은 {worst['display_name']} ({worst_value:.4%}, block p={worst['p_value_block']:.4f})"
    else:
        worst_phrase = f"가장 덜 강한 쪽은 {worst['display_name']} ({worst_value:.4%}, block p={worst['p_value_block']:.4f})"

    return f"- {event_type} 1일 기준 {context}에서는 {best_phrase}, {worst_phrase}입니다."

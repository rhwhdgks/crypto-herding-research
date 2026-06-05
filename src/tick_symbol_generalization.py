from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_generalization_focus_summary(
    pooled_summary: pd.DataFrame,
    symbol_summary: pd.DataFrame,
    focus_horizon_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_focus = pooled_summary.loc[pooled_summary["horizon_minutes"] == int(focus_horizon_minutes)].copy()
    symbol_focus = symbol_summary.loc[symbol_summary["horizon_minutes"] == int(focus_horizon_minutes)].copy()
    if symbol_focus.empty:
        return pooled_focus, symbol_focus

    symbol_focus["direction_score"] = np.where(
        symbol_focus["delta_mean_return"] > 0,
        symbol_focus["delta_t_stat"].fillna(0.0),
        -symbol_focus["delta_t_stat"].abs().fillna(0.0),
    )
    symbol_focus["signal_quality"] = np.select(
        [
            (symbol_focus["delta_mean_return"] > 0) & (symbol_focus["delta_t_stat"] >= 1.5),
            (symbol_focus["delta_mean_return"] > 0) & (symbol_focus["delta_t_stat"] > 0),
            symbol_focus["delta_mean_return"] < 0,
        ],
        [
            "positive_supported",
            "positive_weak",
            "negative",
        ],
        default="neutral",
    )
    return pooled_focus, symbol_focus.sort_values(
        ["event_label", "delta_mean_return", "delta_t_stat"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_generalization_report(
    config: dict,
    backfill_summary: pd.DataFrame,
    pooled_focus: pd.DataFrame,
    symbol_focus: pd.DataFrame,
    plot_paths: list[str],
) -> str:
    lines = ["# 다심볼 Tick 일반화 연구", ""]
    lines.append("## 설정")
    lines.append(f"- 종목: {', '.join(config['data']['symbols'])}")
    lines.append(f"- 구간: {config['data'].get('resolved_start_utc', config['data'].get('start'))} ~ {config['data'].get('resolved_end_utc', config['data'].get('end'))}")
    lines.append(f"- 이벤트 버킷: {', '.join(str(v) + 'm' for v in config['analysis']['interval_minutes'])}")
    lines.append(f"- 선행 수익률 horizon: {', '.join(str(v) + 'm' for v in config['analysis']['forward_horizons_minutes'])}")
    lines.append("")

    if not backfill_summary.empty:
        ok = backfill_summary[backfill_summary["status"] == "ok"]
        lines.append("## Tick 백필")
        lines.append(f"- 전체 작업 수: {int(len(backfill_summary))}")
        lines.append(f"- 성공: {int(len(ok))}")
        lines.append(f"- 새 다운로드: {int((ok['source_used'] == 'download').sum())}")
        lines.append(f"- 캐시 재사용: {int((ok['source_used'] == 'cache').sum())}")
        lines.append("")

    lines.append("## Pooled 결과")
    if pooled_focus.empty:
        lines.append("- pooled 요약이 없습니다.")
    else:
        for _, row in pooled_focus.iterrows():
            lines.append(
                f"- {row['event_label']} / {row['horizon_label']}: 차이 {row['delta_mean_return']:.4%}, "
                f"차이 t={row['delta_t_stat']:.2f}, 이벤트 {int(row['event_count'])}건"
            )
    lines.append("")

    lines.append("## 심볼별 핵심 비교")
    if symbol_focus.empty:
        lines.append("- 심볼별 요약이 없습니다.")
    else:
        for event_label in ["up", "all", "down"]:
            subset = symbol_focus.loc[symbol_focus["event_label"] == event_label].copy()
            if subset.empty:
                continue
            lines.append(f"- {event_label} 이벤트")
            for _, row in subset.iterrows():
                lines.append(
                    f"  {row['symbol']}: 차이 {row['delta_mean_return']:.4%}, "
                    f"차이 t={row['delta_t_stat']:.2f}, 이벤트 {int(row['event_count'])}건, 판정 {row['signal_quality']}"
                )
    lines.append("")

    lines.append("## 해석")
    if not symbol_focus.empty:
        up_subset = symbol_focus.loc[symbol_focus["event_label"] == "up"].sort_values(
            ["delta_mean_return", "delta_t_stat"], ascending=False
        )
        if not up_subset.empty:
            best_up = up_subset.iloc[0]
            lines.append(
                f"- up micro-herding 기준 가장 좋은 심볼은 {best_up['symbol']}이며, "
                f"{best_up['horizon_label']} 차이는 {best_up['delta_mean_return']:.4%}입니다."
            )
        positive_up = symbol_focus.loc[
            (symbol_focus["event_label"] == "up") & (symbol_focus["delta_mean_return"] > 0),
            "symbol",
        ].nunique()
        lines.append(
            f"- up micro-herding의 양(+) 반응이 나온 심볼은 {int(positive_up)}개입니다."
        )
        lines.append("- 이 모드는 XRP에서 본 short-horizon 구조가 다른 심볼에도 반복되는지 확인하기 위한 일반화 실험입니다.")
    lines.append("")

    lines.append("## 플롯")
    for plot_path in plot_paths:
        lines.append(f"- {plot_path}")
    lines.append("")
    return "\n".join(lines)


def plot_symbol_generalization(
    symbol_focus: pd.DataFrame,
    event_label: str,
    path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    subset = symbol_focus.loc[symbol_focus["event_label"] == event_label].copy()
    if subset.empty:
        ax.text(0.5, 0.5, "표시할 결과가 없습니다.", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    subset = subset.sort_values("delta_mean_return", ascending=False)
    colors = ["#54A24B" if value >= 0 else "#E45756" for value in subset["delta_mean_return"]]
    ax.bar(subset["symbol"], subset["delta_mean_return"] * 100.0, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_title(f"{event_label} micro-herding 심볼별 차이")
    ax.set_ylabel("이벤트-대조군 차이 (%)")
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

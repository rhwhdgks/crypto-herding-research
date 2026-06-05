from __future__ import annotations

from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_shifted_daily_context(daily_event_frame: pd.DataFrame) -> pd.DataFrame:
    if daily_event_frame.empty:
        return pd.DataFrame(columns=["trade_date", "prior_daily_event", "prior_daily_csad", "prior_daily_market_return"])

    context = daily_event_frame[["event_type", "csad", "market_return"]].copy()
    context = context.rename(
        columns={
            "event_type": "prior_daily_event",
            "csad": "prior_daily_csad",
            "market_return": "prior_daily_market_return",
        }
    )
    context.index = pd.to_datetime(context.index, utc=True) + pd.Timedelta(days=1)
    context.index.name = "trade_date"
    return context.reset_index()


def merge_tick_trades_with_daily_context(
    trade_log: pd.DataFrame,
    daily_context: pd.DataFrame,
    candidate_names: list[str],
) -> pd.DataFrame:
    if trade_log.empty:
        return pd.DataFrame()

    frame = trade_log.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], utc=True).dt.normalize()
    frame = frame.loc[frame["variant_name"].isin(candidate_names)].copy()
    merged = frame.merge(daily_context, on="trade_date", how="left")
    merged["prior_daily_event"] = merged["prior_daily_event"].fillna("none")
    return merged


def summarize_paper_bridge(bridge_frame: pd.DataFrame) -> pd.DataFrame:
    if bridge_frame.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for (variant_name, prior_daily_event), subset in bridge_frame.groupby(["variant_name", "prior_daily_event"], dropna=False):
        sample = subset["net_return"].dropna()
        rows.append(
            {
                "variant_name": variant_name,
                "prior_daily_event": prior_daily_event,
                "trade_count": int(sample.shape[0]),
                "mean_net_return": sample.mean() if not sample.empty else np.nan,
                "t_stat": _compute_t_stat(sample),
                "win_rate": (sample > 0).mean() if not sample.empty else np.nan,
            }
        )
    summary = pd.DataFrame(rows)

    baseline = summary.loc[summary["prior_daily_event"] == "none", ["variant_name", "mean_net_return"]].rename(
        columns={"mean_net_return": "baseline_none_mean"}
    )
    summary = summary.merge(baseline, on="variant_name", how="left")
    summary["delta_vs_none"] = summary["mean_net_return"] - summary["baseline_none_mean"]
    return summary.sort_values(["variant_name", "delta_vs_none"], ascending=[True, False]).reset_index(drop=True)


def build_paper_bridge_report(
    config: dict,
    regression_summary: dict,
    bridge_summary: pd.DataFrame,
    plot_paths: list[str],
) -> str:
    lines = ["# Paper-like와 Tick 연결 연구", ""]
    lines.append("## 설정")
    lines.append(f"- 일봉 구간: {config['data'].get('resolved_start_utc', config['data'].get('start'))} ~ {config['data'].get('resolved_end_utc', config['data'].get('end'))}")
    lines.append(f"- tick 후보: {', '.join(config['tick']['candidate_names'])}")
    lines.append("- 전일 일봉 CSAD 상태를 다음 날 tick trade에 연결했습니다.")
    lines.append("")

    lines.append("## 일봉 CSAD 회귀 배경")
    lines.append(f"- beta2: {regression_summary.get('beta2', np.nan):.6f}")
    lines.append(f"- beta2 t-stat: {regression_summary.get('beta2_t_stat', np.nan):.2f}")
    lines.append(f"- 해석: {regression_summary.get('interpretation', 'n/a')}")
    lines.append("")

    lines.append("## 후보별 전일 상태 조건부 성과")
    if bridge_summary.empty:
        lines.append("- 연결 결과가 없습니다.")
    else:
        for variant_name, subset in bridge_summary.groupby("variant_name"):
            lines.append(f"- {variant_name}")
            for _, row in subset.iterrows():
                lines.append(
                    f"  전일 {row['prior_daily_event']}: 평균 {row['mean_net_return']:.4%}, "
                    f"none 대비 {row['delta_vs_none']:.4%}, t={row['t_stat']:.2f}, 거래 {int(row['trade_count'])}건"
                )
    lines.append("")

    lines.append("## 해석")
    if not bridge_summary.empty:
        positive = bridge_summary.loc[bridge_summary["prior_daily_event"] != "none"].sort_values(
            ["delta_vs_none", "t_stat"], ascending=False
        )
        if not positive.empty:
            best = positive.iloc[0]
            lines.append(
                f"- 가장 잘 맞는 상위 시장 상태는 {best['variant_name']}의 전일 {best['prior_daily_event']} 조건이며, "
                f"none 대비 차이는 {best['delta_vs_none']:.4%}입니다."
            )
        lines.append("- 이 모드는 논문 유사 저빈도 herding 상태가 intraday tick 규칙의 성과를 강화하는지 확인하는 연결 실험입니다.")
    lines.append("")

    lines.append("## 플롯")
    for plot_path in plot_paths:
        lines.append(f"- {plot_path}")
    lines.append("")
    return "\n".join(lines)


def plot_paper_bridge_summary(summary: pd.DataFrame, path: str | Path) -> None:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 5))
    if summary.empty:
        ax.text(0.5, 0.5, "표시할 결과가 없습니다.", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    plot_frame = summary.copy()
    plot_frame["label"] = plot_frame["variant_name"] + " / " + plot_frame["prior_daily_event"]
    colors = ["#54A24B" if value >= 0 else "#E45756" for value in plot_frame["delta_vs_none"]]
    ax.bar(plot_frame["label"], plot_frame["delta_vs_none"] * 100.0, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("none 대비 차이 (%)")
    ax.set_title("전일 일봉 CSAD 상태가 tick 후보 성과에 주는 영향")
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _compute_t_stat(sample: pd.Series) -> float:
    if sample.shape[0] < 2:
        return np.nan
    std = sample.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return float(sample.mean() / (std / sqrt(sample.shape[0])))

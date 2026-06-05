from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tick_candidate_paper_sim import build_monthly_summary
from tick_overlap_core_regime import assign_quantile_bucket


VARIANT_LABELS = {
    "base_overlap_core": "base_overlap_core",
    "time_17_18": "time_17_18",
    "prior_drop_q4": "prior_drop_q4",
    "strength_q1_q2": "strength_q1_q2",
    "time_17_18_prior_drop_q4": "time_17_18_prior_drop_q4",
    "triple_focus": "triple_focus",
}


def load_overlap_core_regime_sample(base_dir: str | Path) -> pd.DataFrame:
    path = Path(base_dir) / "overlap_core_regime_sample.csv"
    frame = pd.read_csv(path, parse_dates=["bucket_start", "entry_timestamp", "exit_timestamp", "trade_date"])
    if frame.empty:
        return frame
    for column in ["bucket_start", "entry_timestamp", "exit_timestamp", "trade_date"]:
        timestamp = frame[column]
        if timestamp.dt.tz is None:
            frame[column] = timestamp.dt.tz_localize("UTC")
        else:
            frame[column] = timestamp.dt.tz_convert("UTC")
    return frame.sort_values("entry_timestamp").reset_index(drop=True)


def enrich_variant_buckets(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    enriched = frame.copy()
    enriched["strength_bucket"] = assign_quantile_bucket(enriched["strength_ratio"], 4, "strength")
    enriched["prior_drop_bucket"] = assign_quantile_bucket(enriched["prior_drop_magnitude"], 4, "prior_drop")

    strength_cut = float(enriched["strength_ratio"].quantile(0.50))
    prior_cut = float(enriched["prior_drop_magnitude"].quantile(0.75))
    thresholds = {
        "strength_q2_upper": strength_cut,
        "prior_drop_q4_lower": prior_cut,
    }
    return enriched, thresholds


def build_variant_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    time_17_18 = frame["hour_utc"].isin([17, 18])
    prior_drop_q4 = frame["prior_drop_bucket"] == "prior_drop_Q4"
    strength_q1_q2 = frame["strength_bucket"].isin(["strength_Q1", "strength_Q2"])
    return {
        "base_overlap_core": pd.Series(True, index=frame.index),
        "time_17_18": time_17_18,
        "prior_drop_q4": prior_drop_q4,
        "strength_q1_q2": strength_q1_q2,
        "time_17_18_prior_drop_q4": time_17_18 & prior_drop_q4,
        "triple_focus": time_17_18 & prior_drop_q4 & strength_q1_q2,
    }


def build_variant_trade_log(frame: pd.DataFrame, variant_masks: dict[str, pd.Series]) -> pd.DataFrame:
    trade_frames: list[pd.DataFrame] = []
    for variant_name, mask in variant_masks.items():
        subset = frame.loc[mask].copy()
        if subset.empty:
            continue
        subset["variant_name"] = variant_name
        subset["variant_label"] = VARIANT_LABELS.get(variant_name, variant_name)
        trade_frames.append(subset)
    if not trade_frames:
        return pd.DataFrame()
    return pd.concat(trade_frames, axis=0, ignore_index=True).sort_values(["variant_name", "entry_timestamp"]).reset_index(drop=True)


def summarize_variant_oos(
    trade_log: pd.DataFrame,
    holdout_days_list: list[int],
) -> pd.DataFrame:
    if trade_log.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for variant_name, group in trade_log.groupby("variant_name", sort=True):
        last_date = group["entry_timestamp"].dt.normalize().max()
        rows.append(_build_period_row(group, variant_name, "full_sample"))
        for holdout_days in sorted({int(value) for value in holdout_days_list}):
            holdout_start = last_date - pd.Timedelta(days=max(int(holdout_days) - 1, 0))
            holdout_mask = group["entry_timestamp"].dt.normalize() >= holdout_start
            development_mask = ~holdout_mask
            rows.append(
                _build_period_row(
                    group.loc[development_mask].copy(),
                    variant_name,
                    f"development_{holdout_days}d",
                )
            )
            rows.append(
                _build_period_row(
                    group.loc[holdout_mask].copy(),
                    variant_name,
                    f"holdout_{holdout_days}d",
                )
            )
    return pd.DataFrame(rows)


def summarize_variant_months(trade_log: pd.DataFrame) -> pd.DataFrame:
    if trade_log.empty:
        return pd.DataFrame()
    monthly = trade_log.copy()
    monthly = monthly.drop(columns=["candidate_name", "candidate_label"], errors="ignore")
    monthly = monthly.rename(columns={"variant_name": "candidate_name", "variant_label": "candidate_label"})
    return build_monthly_summary(monthly)


def summarize_variant_block_stability(trade_log: pd.DataFrame, block_days: int) -> pd.DataFrame:
    if trade_log.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for variant_name, group in trade_log.groupby("variant_name", sort=True):
        start_date = group["entry_timestamp"].dt.normalize().min()
        end_date = group["entry_timestamp"].dt.normalize().max()
        calendar = list(pd.date_range(start=start_date, end=end_date, freq="1D", tz="UTC"))
        block_means: list[float] = []
        counts: list[int] = []
        for start_idx in range(0, max(len(calendar) - int(block_days) + 1, 0), int(block_days)):
            block_dates = calendar[start_idx : start_idx + int(block_days)]
            if len(block_dates) < int(block_days):
                continue
            subset = group[group["entry_timestamp"].dt.normalize().isin(block_dates)].copy()
            if subset.empty:
                continue
            block_means.append(float(subset["net_return"].mean()))
            counts.append(int(len(subset)))
        if counts:
            rows.append(
                {
                    "variant_name": variant_name,
                    "blocks_with_trades": int(len(counts)),
                    "positive_block_share": float((pd.Series(block_means) > 0).mean()),
                    "mean_block_net_return": float(pd.Series(block_means).mean()),
                    "median_block_trade_count": float(pd.Series(counts).median()),
                }
            )
    return pd.DataFrame(rows)


def summarize_variant_costs(
    trade_log: pd.DataFrame,
    holdout_days_list: list[int],
    cost_bps_grid: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trade_log.empty:
        return pd.DataFrame(), pd.DataFrame()

    cost_frames: list[pd.DataFrame] = []
    break_even_frames: list[pd.DataFrame] = []
    for variant_name, group in trade_log.groupby("variant_name", sort=True):
        cost_summary, break_even_summary = _summarize_cost_grid_calendar(
            group[["entry_timestamp", "gross_return"]].copy(),
            holdout_days_list=holdout_days_list,
            cost_bps_grid=cost_bps_grid,
        )
        if not cost_summary.empty:
            cost_summary["variant_name"] = variant_name
            cost_frames.append(cost_summary)
        if not break_even_summary.empty:
            break_even_summary["variant_name"] = variant_name
            break_even_frames.append(break_even_summary)

    cost_summary = pd.concat(cost_frames, axis=0, ignore_index=True) if cost_frames else pd.DataFrame()
    break_even_summary = pd.concat(break_even_frames, axis=0, ignore_index=True) if break_even_frames else pd.DataFrame()
    return cost_summary, break_even_summary


def build_variant_report(
    thresholds: dict[str, float],
    oos_summary: pd.DataFrame,
    block_summary: pd.DataFrame,
    break_even_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    plot_paths: list[str],
) -> str:
    lines = ["# Tick Overlap Core Variant 비교", ""]
    lines.append("## 비교 규칙")
    lines.extend(
        [
            "- 기준선: `base_overlap_core` = `prev_neg AND ratio_1_40_16_18` 전체",
            "- `time_17_18`: overlap_core 중 UTC 17~18시만 유지",
            f"- `prior_drop_q4`: overlap_core 중 직전 15분 하락폭 상위 25%만 유지 (컷오프 {thresholds['prior_drop_q4_lower']:.4%})",
            f"- `strength_q1_q2`: overlap_core 중 strength 하위 50%만 유지 (상한 {thresholds['strength_q2_upper']:.2f})",
            "- `time_17_18_prior_drop_q4`: UTC 17~18시이면서 직전 하락폭 상위 25%",
            "- `triple_focus`: UTC 17~18시 + 직전 하락폭 상위 25% + strength 하위 50%",
            "",
        ]
    )

    lines.append("## OOS 비교")
    focus_periods = ["full_sample", "holdout_30d", "holdout_60d"]
    for period_name in focus_periods:
        subset = oos_summary[oos_summary["period_name"] == period_name].copy()
        if subset.empty:
            continue
        ordered = subset.sort_values(["mean_net_return", "trade_count"], ascending=[False, False])
        lines.append(f"- {period_name}")
        for _, row in ordered.iterrows():
            lines.append(
                f"  - {row['variant_name']}: 거래 {int(row['trade_count'])}건, 평균 순수익 {row['mean_net_return']:.4%}, "
                f"승률 {row['win_rate']:.2%}, 누적 {row['terminal_cumulative_net_return']:.2%}, t={row['t_stat']:.2f}"
            )
    lines.append("")

    lines.append("## 비용 내구성")
    if break_even_summary.empty:
        lines.append("- 결과가 없습니다.")
    else:
        full = break_even_summary[break_even_summary["period_name"] == "full_sample"].copy()
        ordered = full.sort_values("break_even_round_trip_bps", ascending=False)
        for _, row in ordered.iterrows():
            lines.append(
                f"- {row['variant_name']}: full_sample 손익분기 비용 {row['break_even_round_trip_bps']:.2f} bps, "
                f"평균 총수익 {row['mean_gross_return']:.4%}"
            )
    lines.append("")

    lines.append("## 안정성")
    if not block_summary.empty:
        ordered = block_summary.sort_values(["positive_block_share", "mean_block_net_return"], ascending=False)
        for _, row in ordered.iterrows():
            lines.append(
                f"- {row['variant_name']}: 양(+) 30일 블록 비중 {row['positive_block_share']:.2%}, "
                f"블록 평균 순수익 {row['mean_block_net_return']:.4%}, 중앙 블록 거래 수 {row['median_block_trade_count']:.0f}건"
            )
    if not monthly_summary.empty:
        lines.append("- 월별 안정성 요약:")
        for variant_name, group in monthly_summary.groupby("candidate_name", sort=True):
            positive_share = float((group["month_cumulative_net_return"] > 0).mean())
            lines.append(
                f"  - {variant_name}: 양(+) 월 비중 {positive_share:.2%}, 월 평균 누적 순수익 {group['month_cumulative_net_return'].mean():.4%}"
            )
    lines.append("")

    lines.append("## 해석")
    full = oos_summary[oos_summary["period_name"] == "full_sample"].copy()
    hold60 = oos_summary[oos_summary["period_name"] == "holdout_60d"].copy()
    if not full.empty:
        top_full = full.sort_values("mean_net_return", ascending=False).iloc[0]
        lines.append(
            f"- 전체 기간 기준 가장 강한 변형은 `{top_full['variant_name']}`입니다. 평균 순수익 {top_full['mean_net_return']:.4%}, 거래 {int(top_full['trade_count'])}건입니다."
        )
    if not hold60.empty:
        top_hold = hold60.sort_values("mean_net_return", ascending=False).iloc[0]
        lines.append(
            f"- holdout_60d 기준으로도 `{top_hold['variant_name']}`가 가장 강합니다. 평균 순수익 {top_hold['mean_net_return']:.4%}입니다."
        )
    lines.append("- 따라서 다음 단계는 성과가 가장 좋은 단일 변형을 메인 후보로 두고, 기준선 overlap_core 대비 거래 수 감소와 비용 내구성의 trade-off를 함께 봐야 합니다.")
    lines.append("")

    lines.append("## 플롯")
    for plot_path in plot_paths:
        lines.append(f"- {plot_path}")
    lines.append("")
    return "\n".join(lines)


def plot_variant_oos(oos_summary: pd.DataFrame, path: str | Path) -> None:
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(11, 5))
    if oos_summary.empty:
        _render_empty_plot(fig, ax, "variant OOS", "표시할 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    subset = oos_summary[oos_summary["period_name"].isin(["full_sample", "holdout_60d"])].copy()
    pivot = subset.pivot(index="variant_name", columns="period_name", values="mean_net_return").fillna(np.nan)
    names = pivot.index.tolist()
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, pivot.get("full_sample", pd.Series(index=names)).fillna(0.0) * 100.0, width=width, label="full_sample", color="#4C78A8")
    ax.bar(x + width / 2, pivot.get("holdout_60d", pd.Series(index=names)).fillna(0.0) * 100.0, width=width, label="holdout_60d", color="#54A24B")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("평균 순수익 (%)")
    ax.set_title("Overlap Core 변형별 OOS 비교")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_variant_break_even(break_even_summary: pd.DataFrame, path: str | Path) -> None:
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(10, 5))
    if break_even_summary.empty:
        _render_empty_plot(fig, ax, "variant break-even", "표시할 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    subset = break_even_summary[break_even_summary["period_name"] == "full_sample"].copy().sort_values("break_even_round_trip_bps", ascending=False)
    colors = ["#54A24B" if value >= 0 else "#E45756" for value in subset["break_even_round_trip_bps"]]
    ax.bar(subset["variant_name"], subset["break_even_round_trip_bps"], color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("손익분기 비용 (bps)")
    ax.set_title("Overlap Core 변형별 손익분기 비용")
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=15)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_period_row(subset: pd.DataFrame, variant_name: str, period_name: str) -> dict:
    if subset.empty:
        return {
            "variant_name": variant_name,
            "period_name": period_name,
            "trade_count": 0,
            "mean_gross_return": np.nan,
            "mean_net_return": np.nan,
            "median_net_return": np.nan,
            "win_rate": np.nan,
            "terminal_cumulative_net_return": np.nan,
            "t_stat": np.nan,
        }
    ordered = subset.sort_values("entry_timestamp").copy()
    return {
        "variant_name": variant_name,
        "period_name": period_name,
        "trade_count": int(len(ordered)),
        "mean_gross_return": float(ordered["gross_return"].mean()),
        "mean_net_return": float(ordered["net_return"].mean()),
        "median_net_return": float(ordered["net_return"].median()),
        "win_rate": float((ordered["net_return"] > 0).mean()),
        "terminal_cumulative_net_return": float((1.0 + ordered["net_return"]).prod() - 1.0),
        "t_stat": _compute_t_stat(ordered["net_return"]),
    }


def _compute_t_stat(series: pd.Series) -> float:
    clean = pd.Series(series).dropna().astype(float)
    if clean.shape[0] < 2:
        return np.nan
    std = clean.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return float(clean.mean() / (std / np.sqrt(clean.shape[0])))


def _summarize_cost_grid_calendar(
    trades: pd.DataFrame,
    holdout_days_list: list[int],
    cost_bps_grid: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()

    last_date = trades["entry_timestamp"].dt.normalize().max()
    period_masks: dict[str, pd.Series] = {"full_sample": pd.Series(True, index=trades.index)}
    for holdout_days in sorted({int(value) for value in holdout_days_list}):
        holdout_start = last_date - pd.Timedelta(days=max(int(holdout_days) - 1, 0))
        holdout_mask = trades["entry_timestamp"].dt.normalize() >= holdout_start
        period_masks[f"holdout_{holdout_days}d"] = holdout_mask
        period_masks[f"development_{holdout_days}d"] = ~holdout_mask

    rows: list[dict] = []
    break_even_rows: list[dict] = []
    for period_name, period_mask in period_masks.items():
        subset = trades.loc[period_mask].copy()
        if subset.empty:
            continue
        gross_sample = subset["gross_return"].dropna().astype(float)
        break_even_rows.append(
            {
                "period_name": period_name,
                "trade_count": int(gross_sample.shape[0]),
                "mean_gross_return": gross_sample.mean(),
                "gross_t_stat": _compute_t_stat(gross_sample),
                "gross_win_rate": (gross_sample > 0).mean(),
                "break_even_round_trip_bps": float(gross_sample.mean() * 10000.0),
            }
        )
        for cost_bps in cost_bps_grid:
            cost_decimal = float(cost_bps) / 10000.0
            net_sample = gross_sample - cost_decimal
            rows.append(
                {
                    "period_name": period_name,
                    "round_trip_cost_bps": float(cost_bps),
                    "trade_count": int(net_sample.shape[0]),
                    "mean_net_return": net_sample.mean(),
                    "median_net_return": net_sample.median(),
                    "net_t_stat": _compute_t_stat(net_sample),
                    "net_win_rate": (net_sample > 0).mean(),
                    "terminal_cumulative_net_return": float((1.0 + net_sample).prod() - 1.0),
                    "max_drawdown": _compute_max_drawdown(net_sample),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(break_even_rows)


def _compute_max_drawdown(series: pd.Series) -> float:
    clean = pd.Series(series).dropna().astype(float)
    if clean.empty:
        return np.nan
    equity = (1.0 + clean).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def _configure_korean_matplotlib_font() -> None:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False


def _render_empty_plot(fig: plt.Figure, ax: plt.Axes, title: str, message: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()

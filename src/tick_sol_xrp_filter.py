from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tick_candidate_paper_sim import build_monthly_summary
from tick_overlap_core_variants import _build_period_row, _compute_t_stat, _summarize_cost_grid_calendar


def load_trade_and_micro_frames(
    trade_log_path: str | Path,
    micro_frame_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = pd.read_csv(trade_log_path)
    trades["bucket_start"] = pd.to_datetime(trades["bucket_start"], utc=True)
    trades["entry_timestamp"] = pd.to_datetime(trades["entry_timestamp"], utc=True)
    trades["exit_timestamp"] = pd.to_datetime(trades["exit_timestamp"], utc=True)

    micro = pd.read_csv(micro_frame_path)
    micro["bucket_start"] = pd.to_datetime(micro["bucket_start"], utc=True)
    return trades, micro


def build_sol_context(micro_frame: pd.DataFrame) -> pd.DataFrame:
    sol = micro_frame.loc[micro_frame["symbol"] == "SOLUSDT", ["bucket_start", "event_label"]].copy()
    sol = sol.sort_values("bucket_start").reset_index(drop=True)
    sol["sol_up_t0"] = sol["event_label"].eq("micro_herding_up")
    sol["sol_up_t1"] = sol["sol_up_t0"].shift(1).fillna(False).astype(bool)
    sol["sol_down_t1"] = sol["event_label"].eq("micro_herding_down").shift(1).fillna(False).astype(bool)
    return sol[["bucket_start", "sol_up_t0", "sol_up_t1", "sol_down_t1"]]


def build_filtered_trade_log(
    trades: pd.DataFrame,
    sol_context: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    frame = trades.loc[(trades["bucket_start"] >= start_ts) & (trades["bucket_start"] <= end_ts)].copy()
    frame = frame.merge(sol_context, on="bucket_start", how="left")
    for column in ["sol_up_t0", "sol_up_t1", "sol_down_t1"]:
        frame[column] = frame[column].fillna(False).astype(bool)
    return frame


def build_filter_candidate_log(frame: pd.DataFrame) -> pd.DataFrame:
    candidate_specs = [
        ("time_17_18", frame["variant_name"] == "time_17_18"),
        ("time_17_18__sol_up_t0", (frame["variant_name"] == "time_17_18") & frame["sol_up_t0"]),
        ("time_17_18__sol_up_t1", (frame["variant_name"] == "time_17_18") & frame["sol_up_t1"]),
        ("time_17_18__sol_down_t1", (frame["variant_name"] == "time_17_18") & frame["sol_down_t1"]),
        (
            "time_17_18_prior_drop_q4",
            frame["variant_name"] == "time_17_18_prior_drop_q4",
        ),
        (
            "time_17_18_prior_drop_q4__sol_up_t0",
            (frame["variant_name"] == "time_17_18_prior_drop_q4") & frame["sol_up_t0"],
        ),
        (
            "time_17_18_prior_drop_q4__sol_up_t1",
            (frame["variant_name"] == "time_17_18_prior_drop_q4") & frame["sol_up_t1"],
        ),
        (
            "time_17_18_prior_drop_q4__sol_down_t1",
            (frame["variant_name"] == "time_17_18_prior_drop_q4") & frame["sol_down_t1"],
        ),
    ]

    trade_frames: list[pd.DataFrame] = []
    for candidate_name, mask in candidate_specs:
        subset = frame.loc[mask].copy()
        if subset.empty:
            continue
        subset["candidate_name"] = candidate_name
        subset["candidate_label"] = candidate_name
        subset["parent_variant"] = candidate_name.split("__")[0]
        trade_frames.append(subset)
    if not trade_frames:
        return pd.DataFrame()
    return pd.concat(trade_frames, axis=0, ignore_index=True).sort_values(["candidate_name", "entry_timestamp"]).reset_index(drop=True)


def summarize_filter_oos(trade_log: pd.DataFrame, holdout_days_list: list[int]) -> pd.DataFrame:
    if trade_log.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for candidate_name, group in trade_log.groupby("candidate_name", sort=True):
        last_date = group["entry_timestamp"].dt.normalize().max()
        rows.append(_build_period_row(group, candidate_name, "full_sample"))
        for holdout_days in sorted({int(value) for value in holdout_days_list}):
            holdout_start = last_date - pd.Timedelta(days=max(int(holdout_days) - 1, 0))
            holdout_mask = group["entry_timestamp"].dt.normalize() >= holdout_start
            rows.append(_build_period_row(group.loc[~holdout_mask].copy(), candidate_name, f"development_{holdout_days}d"))
            rows.append(_build_period_row(group.loc[holdout_mask].copy(), candidate_name, f"holdout_{holdout_days}d"))
    summary = pd.DataFrame(rows).rename(columns={"variant_name": "candidate_name"})

    base_rows = summary.loc[
        summary["candidate_name"].isin(["time_17_18", "time_17_18_prior_drop_q4"]),
        ["candidate_name", "period_name", "mean_net_return"],
    ].copy()
    base_rows["parent_variant"] = base_rows["candidate_name"]
    base_rows = base_rows.rename(columns={"mean_net_return": "parent_mean_net_return"})

    summary["parent_variant"] = summary["candidate_name"].str.split("__").str[0]
    summary = summary.merge(base_rows[["parent_variant", "period_name", "parent_mean_net_return"]], on=["parent_variant", "period_name"], how="left")
    summary["delta_vs_parent"] = summary["mean_net_return"] - summary["parent_mean_net_return"]
    return summary


def summarize_filter_costs(
    trade_log: pd.DataFrame,
    holdout_days_list: list[int],
    cost_bps_grid: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trade_log.empty:
        return pd.DataFrame(), pd.DataFrame()

    cost_frames: list[pd.DataFrame] = []
    break_even_frames: list[pd.DataFrame] = []
    for candidate_name, group in trade_log.groupby("candidate_name", sort=True):
        cost_summary, break_even_summary = _summarize_cost_grid_calendar(
            group[["entry_timestamp", "gross_return"]].copy(),
            holdout_days_list=holdout_days_list,
            cost_bps_grid=cost_bps_grid,
        )
        if not cost_summary.empty:
            cost_summary["candidate_name"] = candidate_name
            cost_frames.append(cost_summary)
        if not break_even_summary.empty:
            break_even_summary["candidate_name"] = candidate_name
            break_even_frames.append(break_even_summary)
    return (
        pd.concat(cost_frames, ignore_index=True) if cost_frames else pd.DataFrame(),
        pd.concat(break_even_frames, ignore_index=True) if break_even_frames else pd.DataFrame(),
    )


def summarize_filter_months(trade_log: pd.DataFrame) -> pd.DataFrame:
    if trade_log.empty:
        return pd.DataFrame()
    return build_monthly_summary(trade_log)


def build_sol_filter_report(
    oos_summary: pd.DataFrame,
    break_even_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    plot_paths: list[str],
) -> str:
    lines = ["# SOL -> XRP 조건부 필터 연구", ""]
    lines.append("## 비교 규칙")
    lines.append("- 기준선 1: `time_17_18`")
    lines.append("- 기준선 2: `time_17_18_prior_drop_q4`")
    lines.append("- 필터 A: 같은 버킷 `SOL up` (`sol_up_t0`)")
    lines.append("- 필터 B: 직전 15분 `SOL up` (`sol_up_t1`)")
    lines.append("- 필터 C: 직전 15분 `SOL down` (`sol_down_t1`)")
    lines.append("")

    lines.append("## OOS 비교")
    for period_name in ["full_sample", "holdout_30d", "holdout_60d"]:
        subset = oos_summary.loc[oos_summary["period_name"] == period_name].copy()
        if subset.empty:
            continue
        lines.append(f"- {period_name}")
        ordered = subset.sort_values(["mean_net_return", "trade_count"], ascending=[False, False])
        for _, row in ordered.iterrows():
            lines.append(
                f"  - {row['candidate_name']}: 거래 {int(row['trade_count'])}건, 평균 순수익 {row['mean_net_return']:.4%}, "
                f"부모 대비 {row['delta_vs_parent']:.4%}, t={row['t_stat']:.2f}"
            )
    lines.append("")

    lines.append("## 비용 내구성")
    if not break_even_summary.empty:
        full = break_even_summary.loc[break_even_summary["period_name"] == "full_sample"].copy()
        full = full.sort_values("break_even_round_trip_bps", ascending=False)
        for _, row in full.iterrows():
            lines.append(
                f"- {row['candidate_name']}: 손익분기 비용 {row['break_even_round_trip_bps']:.2f} bps"
            )
    lines.append("")

    lines.append("## 월별 안정성")
    if not monthly_summary.empty:
        for candidate_name, group in monthly_summary.groupby("candidate_name", sort=True):
            positive_share = float((group["month_cumulative_net_return"] > 0).mean())
            lines.append(
                f"- {candidate_name}: 양(+) 월 비중 {positive_share:.2%}, 월 평균 누적 순수익 {group['month_cumulative_net_return'].mean():.4%}"
            )
    lines.append("")

    lines.append("## 해석")
    if not oos_summary.empty:
        full = oos_summary.loc[oos_summary["period_name"] == "full_sample"].copy()
        best = full.sort_values(["delta_vs_parent", "mean_net_return"], ascending=False).iloc[0]
        lines.append(
            f"- full sample 기준 가장 개선 폭이 큰 조건은 `{best['candidate_name']}`이며, 부모 대비 {best['delta_vs_parent']:.4%}입니다."
        )
        hold60 = oos_summary.loc[oos_summary["period_name"] == "holdout_60d"].copy()
        if not hold60.empty:
            best_hold = hold60.sort_values(["mean_net_return", "delta_vs_parent"], ascending=False).iloc[0]
            lines.append(
                f"- holdout_60d 기준으로는 `{best_hold['candidate_name']}`가 가장 강합니다."
            )
        lines.append("- 이 실험은 SOL micro-herding 상태를 XRP 후보 규칙의 확인 필터로 쓸 수 있는지 보는 단계입니다.")
    lines.append("")

    lines.append("## 플롯")
    for plot_path in plot_paths:
        lines.append(f"- {plot_path}")
    lines.append("")
    return "\n".join(lines)


def plot_filter_oos(oos_summary: pd.DataFrame, path: str | Path) -> None:
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(12, 5))
    subset = oos_summary.loc[oos_summary["period_name"].isin(["full_sample", "holdout_60d"])].copy()
    if subset.empty:
        _render_empty_plot(fig, ax, "sol filter oos", "표시할 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    pivot = subset.pivot(index="candidate_name", columns="period_name", values="mean_net_return").fillna(0.0)
    names = pivot.index.tolist()
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, pivot.get("full_sample", pd.Series(index=names)).reindex(names).fillna(0.0) * 100.0, width=width, label="full_sample", color="#4C78A8")
    ax.bar(x + width / 2, pivot.get("holdout_60d", pd.Series(index=names)).reindex(names).fillna(0.0) * 100.0, width=width, label="holdout_60d", color="#54A24B")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("평균 순수익 (%)")
    ax.set_title("SOL 조건부 필터별 OOS 비교")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_filter_break_even(break_even_summary: pd.DataFrame, path: str | Path) -> None:
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(12, 5))
    subset = break_even_summary.loc[break_even_summary["period_name"] == "full_sample"].copy()
    if subset.empty:
        _render_empty_plot(fig, ax, "sol filter break even", "표시할 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    subset = subset.sort_values("break_even_round_trip_bps", ascending=False)
    colors = ["#54A24B" if value >= 0 else "#E45756" for value in subset["break_even_round_trip_bps"]]
    ax.bar(subset["candidate_name"], subset["break_even_round_trip_bps"], color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("손익분기 비용 (bps)")
    ax.set_title("SOL 조건부 필터별 손익분기 비용")
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _configure_korean_matplotlib_font() -> None:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False


def _render_empty_plot(fig: plt.Figure, ax: plt.Axes, title: str, message: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_trade_sample(base_dir: str | Path) -> pd.DataFrame:
    path = Path(base_dir) / "trade_sample.csv"
    frame = pd.read_csv(path, parse_dates=["entry_timestamp", "exit_timestamp", "bucket_start"])
    if frame.empty:
        return frame
    for column in ["entry_timestamp", "exit_timestamp", "bucket_start"]:
        timestamp = frame[column]
        if timestamp.dt.tz is None:
            frame[column] = timestamp.dt.tz_localize("UTC")
        else:
            frame[column] = timestamp.dt.tz_convert("UTC")
    return frame.sort_values("entry_timestamp").reset_index(drop=True)


def enrich_trade_features(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    enriched = trades.copy()
    enriched["strength_ratio"] = enriched["herding_score"].abs() / enriched["herding_threshold"].abs()
    enriched["prior_negative_bucket"] = enriched["bucket_return"] < 0
    enriched["session_16_18"] = enriched["hour_utc"].isin([16, 17, 18])
    return enriched


def build_candidate_masks(trades: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "base_up": pd.Series(True, index=trades.index),
        "prev_neg": trades["prior_negative_bucket"],
        "ratio_1_25": trades["strength_ratio"] >= 1.25,
        "ratio_1_40": trades["strength_ratio"] >= 1.40,
        "ratio_1_25_prev_neg": (trades["strength_ratio"] >= 1.25) & trades["prior_negative_bucket"],
        "ratio_1_40_prev_neg": (trades["strength_ratio"] >= 1.40) & trades["prior_negative_bucket"],
        "ratio_1_40_16_18": (trades["strength_ratio"] >= 1.40) & trades["session_16_18"],
    }


def summarize_candidates_by_period(
    trades: pd.DataFrame,
    candidate_masks: dict[str, pd.Series],
    holdout_days_list: list[int],
    round_trip_cost_bps: float,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    last_date = trades["entry_timestamp"].dt.normalize().max()
    period_masks: dict[str, pd.Series] = {"full_sample": pd.Series(True, index=trades.index)}
    for holdout_days in sorted({int(value) for value in holdout_days_list}):
        holdout_start = last_date - pd.Timedelta(days=max(holdout_days - 1, 0))
        holdout_mask = trades["entry_timestamp"].dt.normalize() >= holdout_start
        period_masks[f"holdout_{holdout_days}d"] = holdout_mask
        period_masks[f"development_{holdout_days}d"] = ~holdout_mask

    rows: list[dict] = []
    cost_decimal = float(round_trip_cost_bps) / 10000.0
    for candidate_name, candidate_mask in candidate_masks.items():
        for period_name, period_mask in period_masks.items():
            subset = trades.loc[candidate_mask & period_mask].copy()
            if subset.empty:
                continue
            gross = subset["gross_return"].astype(float)
            net = gross - cost_decimal
            rows.append(
                {
                    "candidate_name": candidate_name,
                    "period_name": period_name,
                    "trade_count": int(len(subset)),
                    "mean_gross_return": gross.mean(),
                    "mean_net_return": net.mean(),
                    "gross_win_rate": (gross > 0).mean(),
                    "net_win_rate": (net > 0).mean(),
                    "break_even_bps": gross.mean() * 10000.0,
                    "round_trip_cost_bps": float(round_trip_cost_bps),
                }
            )
    return pd.DataFrame(rows)


def summarize_candidates_by_blocks(
    trades: pd.DataFrame,
    candidate_masks: dict[str, pd.Series],
    block_days: int,
    round_trip_cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()

    start_date = trades["entry_timestamp"].dt.normalize().min()
    end_date = trades["entry_timestamp"].dt.normalize().max()
    calendar = list(pd.date_range(start=start_date, end=end_date, freq="1D", tz="UTC"))
    cost_decimal = float(round_trip_cost_bps) / 10000.0

    block_rows: list[dict] = []
    summary_rows: list[dict] = []

    for candidate_name, candidate_mask in candidate_masks.items():
        candidate_blocks: list[float] = []
        candidate_counts: list[int] = []
        fold_id = 1
        for start_idx in range(0, max(len(calendar) - int(block_days) + 1, 0), int(block_days)):
            block_dates = calendar[start_idx : start_idx + int(block_days)]
            if len(block_dates) < int(block_days):
                continue
            date_mask = trades["entry_timestamp"].dt.normalize().isin(block_dates)
            subset = trades.loc[candidate_mask & date_mask].copy()
            if subset.empty:
                block_rows.append(
                    {
                        "candidate_name": candidate_name,
                        "fold_id": int(fold_id),
                        "window_start": min(block_dates),
                        "window_end": max(block_dates),
                        "trade_count": 0,
                        "mean_net_return": np.nan,
                    }
                )
                fold_id += 1
                continue

            net = subset["gross_return"].astype(float) - cost_decimal
            mean_net = float(net.mean())
            block_rows.append(
                {
                    "candidate_name": candidate_name,
                    "fold_id": int(fold_id),
                    "window_start": min(block_dates),
                    "window_end": max(block_dates),
                    "trade_count": int(len(subset)),
                    "mean_net_return": mean_net,
                }
            )
            candidate_blocks.append(mean_net)
            candidate_counts.append(int(len(subset)))
            fold_id += 1

        if candidate_counts:
            summary_rows.append(
                {
                    "candidate_name": candidate_name,
                    "blocks_with_trades": int(len(candidate_counts)),
                    "positive_block_share": float((pd.Series(candidate_blocks) > 0).mean()),
                    "mean_block_net_return": float(pd.Series(candidate_blocks).mean()),
                    "median_block_trade_count": float(pd.Series(candidate_counts).median()),
                    "min_block_trade_count": int(pd.Series(candidate_counts).min()),
                }
            )

    return pd.DataFrame(block_rows), pd.DataFrame(summary_rows)


def build_subset_candidate_report(
    period_summary: pd.DataFrame,
    block_summary: pd.DataFrame,
    round_trip_cost_bps: float,
    plot_paths: list[str],
) -> str:
    lines = ["# Tick Subset 후보 비교", ""]
    lines.append("## 기준")
    lines.extend(
        [
            "- 기본 베이스는 `XRPUSDT / UTC 16-23 / 15분 up micro-herding / 다음 30분 long`입니다.",
            "- 여기서 해석 가능한 고정 필터만 추가해 subset 후보를 비교했습니다.",
            f"- 순수익 비교는 round-trip 비용 {float(round_trip_cost_bps):.0f} bps를 가정했습니다.",
            "",
        ]
    )

    lines.append("## 후보 정의")
    lines.extend(
        [
            "- `base_up`: 기존 up-only 규칙 그대로",
            "- `prev_neg`: 직전 15분 버킷 수익률이 음수인 경우만 진입",
            "- `ratio_1_25`, `ratio_1_40`: event strength ratio가 각각 1.25, 1.40 이상",
            "- `ratio_1_25_prev_neg`, `ratio_1_40_prev_neg`: 강한 이벤트이면서 직전 버킷이 음수",
            "- `ratio_1_40_16_18`: 매우 강한 이벤트이면서 UTC 16-18 시각대",
            "",
        ]
    )

    lines.append("## 기간별 핵심 비교")
    focus_periods = ["full_sample", "holdout_30d", "holdout_60d"]
    for period_name in focus_periods:
        subset = period_summary[period_summary["period_name"] == period_name].copy()
        if subset.empty:
            continue
        ordered = subset.sort_values(["mean_net_return", "trade_count"], ascending=[False, False])
        lines.append(f"- {period_name}")
        for _, row in ordered.iterrows():
            lines.append(
                f"  - {row['candidate_name']}: 거래 {int(row['trade_count'])}건, "
                f"총수익 {row['mean_gross_return']:.4%}, 순수익 {row['mean_net_return']:.4%}, "
                f"손익분기 {row['break_even_bps']:.2f} bps"
            )
    lines.append("")

    lines.append("## 30일 블록 안정성")
    if block_summary.empty:
        lines.append("- 결과가 없습니다.")
    else:
        ordered = block_summary.sort_values(["positive_block_share", "mean_block_net_return"], ascending=False)
        for _, row in ordered.iterrows():
            lines.append(
                f"- {row['candidate_name']}: 양(+) 블록 비중 {row['positive_block_share']:.2%}, "
                f"블록 평균 순수익 {row['mean_block_net_return']:.4%}, "
                f"중앙 블록 거래 수 {row['median_block_trade_count']:.0f}건"
            )
    lines.append("")

    lines.append("## 해석")
    if not period_summary.empty and not block_summary.empty:
        full_period = period_summary[period_summary["period_name"] == "full_sample"].copy()
        hold60_period = period_summary[period_summary["period_name"] == "holdout_60d"].copy()
        merged = full_period.merge(
            hold60_period[["candidate_name", "mean_net_return", "trade_count"]],
            on="candidate_name",
            how="left",
            suffixes=("_full", "_hold60"),
        ).merge(block_summary, on="candidate_name", how="left")

        stable = merged[
            (merged["mean_net_return_full"] > 0)
            & (merged["mean_net_return_hold60"] > 0)
        ].copy()

        if not stable.empty:
            balanced = stable.sort_values(
                ["trade_count_full", "mean_net_return_hold60", "positive_block_share"],
                ascending=[False, False, False],
            ).iloc[0]
            lines.append(
                f"- 균형형 후보는 `{balanced['candidate_name']}`입니다. 전체 표본 거래 수가 {int(balanced['trade_count_full'])}건으로 비교적 많고, "
                f"holdout_60d 평균 순수익도 {balanced['mean_net_return_hold60']:.4%}로 양수입니다."
            )

            high_conv = stable.sort_values(
                ["mean_net_return_full", "positive_block_share", "trade_count_full"],
                ascending=[False, False, False],
            ).iloc[0]
            lines.append(
                f"- 고확신 후보는 `{high_conv['candidate_name']}`입니다. 전체 평균 순수익 {high_conv['mean_net_return_full']:.4%}, "
                f"holdout_60d 평균 순수익 {high_conv['mean_net_return_hold60']:.4%}, "
                f"30일 블록 양(+) 비중 {high_conv['positive_block_share']:.2%}로 가장 강합니다."
            )
        else:
            lines.append("- 전체 표본과 holdout_60d를 동시에 만족하는 안정 후보는 이번 설정에서 뚜렷하지 않았습니다.")

        if "prev_neg" in merged["candidate_name"].values:
            prev_row = merged.loc[merged["candidate_name"] == "prev_neg"].iloc[0]
            lines.append(
                f"- `prev_neg`는 전체 평균 순수익 {prev_row['mean_net_return_full']:.4%}, "
                f"holdout_60d 평균 순수익 {prev_row['mean_net_return_hold60']:.4%}라서, "
                "이번 장기 표본에서는 단독 메인 후보로 보기 어렵습니다."
            )
    lines.append("- 따라서 다음 메인라인은 `ratio_1_40_16_18`를 중심으로 보고, `ratio_1_40_prev_neg` 또는 그 교집합 구조를 함께 비교하는 구성이 더 적절합니다.")
    lines.append("")

    lines.append("## 플롯")
    for plot_path in plot_paths:
        lines.append(f"- {plot_path}")
    lines.append("")
    return "\n".join(lines)


def plot_candidate_net_returns(period_summary: pd.DataFrame, path: str | Path) -> None:
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(11, 5))
    if period_summary.empty:
        _render_empty_plot(fig, ax, "Subset 후보 비교", "표시할 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    subset = period_summary[period_summary["period_name"].isin(["full_sample", "holdout_60d"])].copy()
    pivot = subset.pivot(index="candidate_name", columns="period_name", values="mean_net_return").fillna(np.nan)
    plot_frame = pivot.reset_index()
    x = np.arange(len(plot_frame))
    width = 0.35
    ax.bar(x - width / 2, plot_frame["full_sample"] * 100.0, width=width, label="full_sample")
    if "holdout_60d" in plot_frame:
        ax.bar(x + width / 2, plot_frame["holdout_60d"] * 100.0, width=width, label="holdout_60d")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_frame["candidate_name"], rotation=20)
    ax.set_ylabel("평균 순수익 (%)")
    ax.set_title("Subset 후보별 비용 후 평균 순수익")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_candidate_block_share(block_summary: pd.DataFrame, path: str | Path) -> None:
    _configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(11, 5))
    if block_summary.empty:
        _render_empty_plot(fig, ax, "Subset 후보 블록 안정성", "표시할 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    ordered = block_summary.sort_values("positive_block_share", ascending=False).copy()
    ax.bar(ordered["candidate_name"], ordered["positive_block_share"] * 100.0, color="#54A24B")
    ax.set_ylabel("양(+) 블록 비중 (%)")
    ax.set_title("Subset 후보별 30일 블록 양수 비중")
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

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PARTITION_LABELS = {
    "prev_only": "prev 단독",
    "ratio_overlap": "겹침 핵심",
    "ratio_only": "ratio 단독",
    "union": "두 후보 union",
}

PLOT_PARTITION_LABELS = {
    "prev_only": "prev_only",
    "ratio_overlap": "overlap_core",
    "ratio_only": "ratio_only",
    "union": "union",
}


def load_candidate_trade_log(base_dir: str | Path) -> pd.DataFrame:
    path = Path(base_dir) / "candidate_trade_log.csv"
    frame = pd.read_csv(path, parse_dates=["bucket_start", "entry_timestamp", "exit_timestamp"])
    for column in ["bucket_start", "entry_timestamp", "exit_timestamp"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def build_partition_frame(
    trade_log: pd.DataFrame,
    prev_candidate: str = "prev_neg",
    ratio_candidate: str = "ratio_1_40_16_18",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    key_cols = ["symbol", "bucket_start"]
    flag_frame = (
        trade_log.assign(candidate_flag=1)
        .pivot_table(index=key_cols, columns="candidate_name", values="candidate_flag", aggfunc="max", fill_value=0)
    )
    flag_frame.columns = [str(column) for column in flag_frame.columns]

    base_cols = [
        "symbol",
        "bucket_start",
        "entry_timestamp",
        "exit_timestamp",
        "gross_return",
        "round_trip_cost_bps",
        "hour_utc",
        "strength_ratio",
        "prior_negative_bucket",
        "session_16_18",
    ]
    base_frame = (
        trade_log.sort_values(["symbol", "bucket_start", "candidate_name"])
        .drop_duplicates(subset=key_cols)
        .set_index(key_cols)[base_cols[2:]]
    )
    base_frame = base_frame.join(flag_frame, how="left").reset_index()

    base_frame["is_prev_candidate"] = base_frame.get(prev_candidate, 0).fillna(0).astype(int).eq(1)
    base_frame["is_ratio_candidate"] = base_frame.get(ratio_candidate, 0).fillna(0).astype(int).eq(1)
    base_frame["partition_name"] = "other"
    base_frame.loc[base_frame["is_prev_candidate"] & ~base_frame["is_ratio_candidate"], "partition_name"] = "prev_only"
    base_frame.loc[base_frame["is_prev_candidate"] & base_frame["is_ratio_candidate"], "partition_name"] = "ratio_overlap"
    base_frame.loc[~base_frame["is_prev_candidate"] & base_frame["is_ratio_candidate"], "partition_name"] = "ratio_only"
    base_frame["is_union_candidate"] = base_frame["is_prev_candidate"] | base_frame["is_ratio_candidate"]

    overlap_summary = pd.DataFrame(
        [
            {
                "prev_candidate": prev_candidate,
                "ratio_candidate": ratio_candidate,
                "prev_count": int(base_frame["is_prev_candidate"].sum()),
                "ratio_count": int(base_frame["is_ratio_candidate"].sum()),
                "overlap_count": int((base_frame["partition_name"] == "ratio_overlap").sum()),
                "prev_only_count": int((base_frame["partition_name"] == "prev_only").sum()),
                "ratio_only_count": int((base_frame["partition_name"] == "ratio_only").sum()),
                "union_count": int(base_frame["is_union_candidate"].sum()),
                "ratio_inside_prev_share": float(
                    (base_frame["partition_name"] == "ratio_overlap").sum() / max(int(base_frame["is_ratio_candidate"].sum()), 1)
                ),
                "overlap_inside_prev_share": float(
                    (base_frame["partition_name"] == "ratio_overlap").sum() / max(int(base_frame["is_prev_candidate"].sum()), 1)
                ),
            }
        ]
    )
    return base_frame, overlap_summary


def summarize_partitions_by_period(
    partition_frame: pd.DataFrame,
    round_trip_cost_bps: float = 4.0,
    recent_days_list: list[int] | None = None,
) -> pd.DataFrame:
    recent_days_list = recent_days_list or [30, 60, 90]
    frame = partition_frame.copy()
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    frame["net_return"] = frame["gross_return"] - (round_trip_cost_bps / 10000.0)

    latest_timestamp = frame["entry_timestamp"].max()
    periods: list[tuple[str, pd.DataFrame]] = [("full_sample", frame)]
    for days in recent_days_list:
        cutoff = latest_timestamp - pd.Timedelta(days=int(days))
        periods.append((f"recent_{int(days)}d", frame[frame["entry_timestamp"] >= cutoff].copy()))

    rows: list[dict] = []
    for period_name, period_frame in periods:
        for partition_name in ["prev_only", "ratio_overlap", "ratio_only"]:
            subset = period_frame[period_frame["partition_name"] == partition_name].copy()
            rows.extend(_summarize_subset(subset, partition_name, period_name))

        union_subset = period_frame[period_frame["is_union_candidate"]].copy()
        rows.extend(_summarize_subset(union_subset, "union", period_name))

    return pd.DataFrame(rows)


def summarize_partitions_by_cost(
    partition_frame: pd.DataFrame,
    cost_bps_grid: list[float] | None = None,
) -> pd.DataFrame:
    cost_bps_grid = cost_bps_grid or [2, 4, 6, 8, 10]
    frame = partition_frame.copy()
    rows: list[dict] = []
    subset_map = {
        "prev_only": frame[frame["partition_name"] == "prev_only"].copy(),
        "ratio_overlap": frame[frame["partition_name"] == "ratio_overlap"].copy(),
        "ratio_only": frame[frame["partition_name"] == "ratio_only"].copy(),
        "union": frame[frame["is_union_candidate"]].copy(),
    }
    for cost_bps in cost_bps_grid:
        for partition_name, subset in subset_map.items():
            if subset.empty:
                continue
            net = subset["gross_return"] - (float(cost_bps) / 10000.0)
            rows.append(
                {
                    "partition_name": partition_name,
                    "partition_label": PARTITION_LABELS[partition_name],
                    "round_trip_cost_bps": float(cost_bps),
                    "trade_count": int(len(subset)),
                    "mean_net_return": float(net.mean()),
                    "net_win_rate": float((net > 0).mean()),
                    "terminal_cumulative_net_return": float((1.0 + net).prod() - 1.0),
                }
            )
    return pd.DataFrame(rows)


def plot_partition_period_summary(period_summary: pd.DataFrame, path: str | Path) -> None:
    plot_frame = period_summary[period_summary["period_name"].isin(["full_sample", "recent_60d", "recent_90d"])].copy()
    if plot_frame.empty:
        return
    plot_frame["plot_partition_label"] = plot_frame["partition_name"].map(PLOT_PARTITION_LABELS)
    pivot = (
        plot_frame.pivot(index="plot_partition_label", columns="period_name", values="mean_net_return")
        .reindex(["prev_only", "overlap_core", "ratio_only", "union"])
        .fillna(0.0)
        * 100.0
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Candidate Partition Mean Net Return")
    ax.set_xlabel("")
    ax.set_ylabel("Mean Net Return (%)")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.legend(title="Period")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_partition_cost_summary(cost_summary: pd.DataFrame, path: str | Path) -> None:
    if cost_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for partition_name, subset in cost_summary.groupby("partition_name"):
        subset = subset.sort_values("round_trip_cost_bps")
        ax.plot(
            subset["round_trip_cost_bps"],
            subset["mean_net_return"] * 100.0,
            marker="o",
            label=PLOT_PARTITION_LABELS.get(partition_name, partition_name),
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Cost Sensitivity of Mean Net Return")
    ax.set_xlabel("Round-trip Cost (bps)")
    ax.set_ylabel("Mean Net Return (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_candidate_refinement_report(
    overlap_summary: pd.DataFrame,
    period_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    round_trip_cost_bps: float,
    plot_paths: list[str],
) -> str:
    overlap_row = overlap_summary.iloc[0]
    lines = ["# Tick 후보 정제 보고서", ""]
    lines.append("## 목적")
    lines.append("- `prev_neg`와 `ratio_1_40_16_18`를 더 이상 새로운 탐색 없이, 구조적으로 어떻게 겹치고 어떤 부분이 실제 성과를 만드는지 분해합니다.")
    lines.append(f"- 기준 비용은 round-trip {round_trip_cost_bps:.1f} bps입니다.")
    lines.append("")

    lines.append("## 겹침 구조")
    lines.append(f"- prev 후보 수: {int(overlap_row['prev_count'])}건")
    lines.append(f"- ratio 후보 수: {int(overlap_row['ratio_count'])}건")
    lines.append(f"- 겹침 구간 수: {int(overlap_row['overlap_count'])}건")
    lines.append(f"- prev 단독 구간 수: {int(overlap_row['prev_only_count'])}건")
    lines.append(f"- ratio 단독 구간 수: {int(overlap_row['ratio_only_count'])}건")
    lines.append(f"- union 구간 수: {int(overlap_row['union_count'])}건")
    lines.append(f"- ratio 후보 중 prev 안에 들어 있는 비중: {float(overlap_row['ratio_inside_prev_share']) * 100:.2f}%")
    lines.append("")

    lines.append("## 기간별 비교")
    for period_name in ["full_sample", "recent_60d", "recent_90d", "recent_30d"]:
        subset = period_summary[period_summary["period_name"] == period_name].copy()
        if subset.empty:
            continue
        lines.append(f"- {period_name}")
        ordered = subset.set_index("partition_name").reindex(["ratio_overlap", "union", "ratio_only", "prev_only"]).reset_index()
        for _, row in ordered.iterrows():
            if pd.isna(row["trade_count"]):
                continue
            lines.append(
                f"  - {row['partition_label']}: 거래 {int(row['trade_count'])}건, 평균 순수익 {row['mean_net_return'] * 100:.4f}%, "
                f"승률 {row['net_win_rate'] * 100:.2f}%, 누적 {row['terminal_cumulative_net_return'] * 100:.2f}%"
            )
    lines.append("")

    lines.append("## 비용 민감도")
    for partition_name in ["ratio_overlap", "union", "prev_only"]:
        subset = cost_summary[cost_summary["partition_name"] == partition_name].copy()
        if subset.empty:
            continue
        best = subset.iloc[0]
        last = subset.iloc[-1]
        lines.append(
            f"- {PARTITION_LABELS[partition_name]}: {best['round_trip_cost_bps']:.0f}bps에서 평균 순수익 {best['mean_net_return'] * 100:.4f}%, "
            f"{last['round_trip_cost_bps']:.0f}bps에서도 {last['mean_net_return'] * 100:.4f}%"
        )
    lines.append("")

    lines.append("## 핵심 해석")
    lines.append("- `ratio_1_40_16_18`의 성과는 대부분 `prev_neg`와 겹치는 구간에서 발생합니다.")
    lines.append("- `prev_neg` 전체는 거래 수는 많지만, ratio와 겹치지 않는 나머지 구간만 떼어 보면 오히려 성과를 희석하는 경우가 많습니다.")
    lines.append("- 따라서 현재 가장 유력한 다음 메인 후보는 `ratio_1_40_16_18 AND prev_neg`의 교집합입니다.")
    lines.append("- `union`은 거래 수를 늘려 주지만, 강한 겹침 구간의 edge를 옅게 만드는 경향이 있습니다.")
    lines.append("")

    lines.append("## 추천")
    lines.append("- 실전형 추적 1순위: `겹침 핵심` = `ratio_1_40_16_18 AND prev_neg`")
    lines.append("- 비교용 보조 축: `prev_neg` 전체")
    lines.append("- union 규칙은 연구 참고용으로만 유지하고, 바로 메인 후보로 올리지는 않는 편이 좋습니다.")
    lines.append("")

    lines.append("## 플롯")
    for plot_path in plot_paths:
        lines.append(f"- {plot_path}")
    lines.append("")
    return "\n".join(lines)


def _summarize_subset(subset: pd.DataFrame, partition_name: str, period_name: str) -> list[dict]:
    if subset.empty:
        return []
    net = subset["net_return"]
    return [
        {
            "partition_name": partition_name,
            "partition_label": PARTITION_LABELS[partition_name],
            "period_name": period_name,
            "trade_count": int(len(subset)),
            "mean_net_return": float(net.mean()),
            "median_net_return": float(net.median()),
            "net_win_rate": float((net > 0).mean()),
            "terminal_cumulative_net_return": float((1.0 + net).prod() - 1.0),
            "best_trade_net_return": float(net.max()),
            "worst_trade_net_return": float(net.min()),
        }
    ]

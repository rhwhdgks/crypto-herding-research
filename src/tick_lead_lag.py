from __future__ import annotations

from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_lead_lag_frame(
    micro_frame: pd.DataFrame,
    target_symbol: str,
    leader_symbols: list[str],
    interval_minutes: int,
    forward_horizon_minutes: int,
    event_direction: str = "up",
) -> pd.DataFrame:
    event_label_value = f"micro_herding_{event_direction}"
    target_frame = micro_frame.loc[micro_frame["symbol"] == target_symbol].copy()
    if target_frame.empty:
        return pd.DataFrame()

    target_frame = target_frame.sort_values("bucket_start").reset_index(drop=True)
    target_frame["target_event"] = target_frame["event_label"].eq(event_label_value)
    target_frame["target_forward_return"] = target_frame.get(f"forward_return_{int(forward_horizon_minutes)}m")

    keep_columns = [
        "bucket_start",
        "hour_utc",
        "is_target_session",
        "meets_trade_count",
        "target_forward_return",
        "target_event",
        "bucket_return",
        "transaction_count",
        "herding_score",
    ]
    result = target_frame[keep_columns].copy()

    for leader in leader_symbols:
        leader_frame = micro_frame.loc[micro_frame["symbol"] == leader].copy()
        leader_frame = leader_frame.sort_values("bucket_start").reset_index(drop=True)
        leader_frame = leader_frame[
            [
                "bucket_start",
                "event_label",
                "bucket_return",
                "herding_score",
                "transaction_count",
            ]
        ].copy()
        leader_prefix = leader.lower().replace("usdt", "")
        leader_frame[f"{leader_prefix}_event"] = leader_frame["event_label"].eq(event_label_value)
        leader_frame = leader_frame.rename(
            columns={
                "bucket_return": f"{leader_prefix}_bucket_return",
                "herding_score": f"{leader_prefix}_herding_score",
                "transaction_count": f"{leader_prefix}_transaction_count",
            }
        )
        leader_frame = leader_frame.drop(columns=["event_label"])
        result = result.merge(leader_frame, on="bucket_start", how="left")

    event_columns = [f"{leader.lower().replace('usdt', '')}_event" for leader in leader_symbols]
    result["leaders_available"] = result[event_columns].notna().all(axis=1)
    result["any_leader_event"] = result[event_columns].fillna(False).any(axis=1)
    result["all_leaders_event"] = result[event_columns].fillna(False).all(axis=1)
    result["non_target_control"] = result["is_target_session"] & result["meets_trade_count"] & result["leaders_available"]
    return result


def summarize_lead_lag(
    lead_lag_frame: pd.DataFrame,
    leader_symbols: list[str],
) -> pd.DataFrame:
    if lead_lag_frame.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    scenario_specs = []
    for leader in leader_symbols:
        leader_prefix = leader.lower().replace("usdt", "")
        scenario_specs.append((f"{leader_prefix}_event", lead_lag_frame[f"{leader_prefix}_event"].fillna(False)))
    if len(leader_symbols) >= 2:
        scenario_specs.append(("any_leader_event", lead_lag_frame["any_leader_event"].fillna(False)))
        scenario_specs.append(("all_leaders_event", lead_lag_frame["all_leaders_event"].fillna(False)))

    valid_frame = lead_lag_frame.loc[
        lead_lag_frame["non_target_control"] & lead_lag_frame["target_forward_return"].notna()
    ].copy()
    control_all = valid_frame["target_forward_return"]

    for scenario_name, scenario_mask in scenario_specs:
        event_sample = valid_frame.loc[scenario_mask.loc[valid_frame.index], "target_forward_return"].dropna()
        control_sample = valid_frame.loc[~scenario_mask.loc[valid_frame.index], "target_forward_return"].dropna()
        rows.append(
            {
                "scenario_name": scenario_name,
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
                "baseline_mean_return": control_all.mean() if not control_all.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["delta_mean_return", "delta_t_stat"], ascending=False).reset_index(drop=True)


def run_lead_lag_matrix(
    micro_frame: pd.DataFrame,
    symbols: list[str],
    interval_minutes: int,
    forward_horizon_minutes: int,
    directions: list[str] | None = None,
) -> pd.DataFrame:
    if directions is None:
        directions = ["up", "down"]

    rows: list[dict] = []
    for direction in directions:
        for target in symbols:
            leaders = [sym for sym in symbols if sym != target]
            lead_lag = build_lead_lag_frame(
                micro_frame=micro_frame,
                target_symbol=target,
                leader_symbols=leaders,
                interval_minutes=interval_minutes,
                forward_horizon_minutes=forward_horizon_minutes,
                event_direction=direction,
            )
            if lead_lag.empty:
                continue
            summary = summarize_lead_lag(lead_lag, leader_symbols=leaders)
            if summary.empty:
                continue
            per_leader = summary[~summary["scenario_name"].isin({"any_leader_event", "all_leaders_event"})]
            prefix_to_symbol = {sym.lower().replace("usdt", ""): sym for sym in leaders}
            for _, row in per_leader.iterrows():
                prefix = row["scenario_name"].removesuffix("_event")
                leader_symbol = prefix_to_symbol.get(prefix)
                if leader_symbol is None:
                    continue
                rows.append(
                    {
                        "target": target,
                        "leader": leader_symbol,
                        "direction": direction,
                        "event_count": int(row["event_count"]),
                        "control_count": int(row["control_count"]),
                        "event_mean_return": float(row["event_mean_return"]),
                        "control_mean_return": float(row["control_mean_return"]),
                        "delta_mean_return": float(row["delta_mean_return"]),
                        "delta_t_stat": float(row["delta_t_stat"]),
                        "event_win_rate": float(row["event_win_rate"]) if pd.notna(row["event_win_rate"]) else np.nan,
                        "baseline_mean_return": float(row["baseline_mean_return"]) if pd.notna(row["baseline_mean_return"]) else np.nan,
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["direction", "target", "leader"]).reset_index(drop=True)


def build_lead_lag_matrix_report(
    config: dict,
    backfill_summary: pd.DataFrame,
    matrix_summary: pd.DataFrame,
    plot_paths: list[str],
    top_n: int = 10,
) -> str:
    symbols = ", ".join(config["analysis"]["symbols"])
    interval = int(config["analysis"]["interval_minutes"][0])
    horizon = int(config["analysis"]["forward_horizons_minutes"][0])

    lines = ["# Tick Lead-Lag Matrix 연구", ""]
    lines.append("## 설정")
    lines.append(f"- 심볼 universe: {symbols}")
    lines.append(
        f"- 구간: {config['data'].get('resolved_start_utc', config['data'].get('start'))} ~ "
        f"{config['data'].get('resolved_end_utc', config['data'].get('end'))}"
    )
    lines.append(f"- 버킷: {interval}분")
    lines.append(f"- 타깃 선행 수익률: {horizon}분")
    lines.append("")

    if not backfill_summary.empty:
        ok = backfill_summary.loc[backfill_summary["status"] == "ok"]
        lines.append("## Tick 백필")
        lines.append(f"- 성공: {int(len(ok))} / {int(len(backfill_summary))}")
        lines.append(f"- 새 다운로드: {int((ok['source_used'] == 'download').sum())}")
        lines.append(f"- 캐시 재사용: {int((ok['source_used'] == 'cache').sum())}")
        lines.append("")

    if matrix_summary.empty:
        lines.append("## 결과")
        lines.append("- 결과가 없습니다.")
        return "\n".join(lines)

    ranked = matrix_summary.assign(abs_t=matrix_summary["delta_t_stat"].abs())
    ranked = ranked.sort_values("abs_t", ascending=False).head(top_n)
    lines.append(f"## 상위 Directed Edges (|delta_t_stat| 기준 top-{top_n})")
    for _, row in ranked.iterrows():
        leader_short = row["leader"].replace("USDT", "")
        target_short = row["target"].replace("USDT", "")
        lines.append(
            f"- {leader_short} {row['direction']} → {target_short}: "
            f"delta {row['delta_mean_return']:.4%}, t={row['delta_t_stat']:.2f}, "
            f"n={int(row['event_count'])}건"
        )
    lines.append("")

    for direction in sorted(matrix_summary["direction"].unique()):
        direction_frame = matrix_summary.loc[matrix_summary["direction"] == direction]
        pivot = direction_frame.pivot(index="target", columns="leader", values="delta_mean_return")
        t_pivot = direction_frame.pivot(index="target", columns="leader", values="delta_t_stat")
        count_pivot = direction_frame.pivot(index="target", columns="leader", values="event_count")
        lines.append(f"## {direction.capitalize()}-event 매트릭스 (delta_mean_return)")
        lines.append("")
        lines.append(_pivot_to_markdown(pivot, lambda v: f"{v:.4%}"))
        lines.append("")
        lines.append(f"## {direction.capitalize()}-event 매트릭스 (delta_t_stat)")
        lines.append("")
        lines.append(_pivot_to_markdown(t_pivot, lambda v: f"{v:.2f}"))
        lines.append("")
        lines.append(f"## {direction.capitalize()}-event 매트릭스 (event_count)")
        lines.append("")
        lines.append(_pivot_to_markdown(count_pivot, lambda v: f"{int(v)}"))
        lines.append("")

    strong = matrix_summary.loc[matrix_summary["delta_t_stat"].abs() >= 1.5]
    lines.append("## 해석")
    lines.append(f"- |t|>=1.5인 directed edge 수: {len(strong)}건 (양방향 합계)")
    if not strong.empty:
        strong_sorted = strong.sort_values("delta_t_stat", ascending=False)
        top_pos = strong_sorted.iloc[0]
        bot_neg = strong_sorted.iloc[-1]
        lines.append(
            f"- 최강 양(+) edge: {top_pos['leader'].replace('USDT','')} {top_pos['direction']} → "
            f"{top_pos['target'].replace('USDT','')} (delta {top_pos['delta_mean_return']:.4%}, t={top_pos['delta_t_stat']:.2f})"
        )
        lines.append(
            f"- 최강 음(-) edge: {bot_neg['leader'].replace('USDT','')} {bot_neg['direction']} → "
            f"{bot_neg['target'].replace('USDT','')} (delta {bot_neg['delta_mean_return']:.4%}, t={bot_neg['delta_t_stat']:.2f})"
        )
    lines.append("- 양방향 대칭성 확인: A→B와 B→A가 같은 방향으로 둘 다 유의하면 공통 shock 가능성. 한쪽만 유의하면 lead-lag 해석 가능.")
    lines.append("")

    lines.append("## 플롯")
    for plot_path in plot_paths:
        lines.append(f"- {plot_path}")
    lines.append("")
    return "\n".join(lines)


def plot_lead_lag_matrix(
    matrix_summary: pd.DataFrame,
    direction: str,
    path: str | Path,
) -> None:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False
    direction_frame = matrix_summary.loc[matrix_summary["direction"] == direction]
    fig, ax = plt.subplots(figsize=(9, 7))
    if direction_frame.empty:
        ax.text(0.5, 0.5, "표시할 결과가 없습니다.", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    pivot = direction_frame.pivot(index="target", columns="leader", values="delta_mean_return")
    t_pivot = direction_frame.pivot(index="target", columns="leader", values="delta_t_stat")
    pivot = pivot * 100.0
    max_abs = float(np.nanmax(np.abs(pivot.values))) if pivot.size else 0.1
    if max_abs == 0 or np.isnan(max_abs):
        max_abs = 0.1

    image = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([col.replace("USDT", "") for col in pivot.columns], rotation=0)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([idx.replace("USDT", "") for idx in pivot.index])
    ax.set_xlabel("Leader (event 발생 심볼)")
    ax.set_ylabel("Target (다음 " + "30" + "분 반응 심볼)")
    ax.set_title(f"{direction.capitalize()}-event lead-lag 매트릭스 (% delta)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            t_value = t_pivot.values[i, j]
            if np.isnan(value):
                continue
            text = f"{value:.3f}\nt={t_value:.2f}"
            color = "black" if abs(value) < max_abs * 0.6 else "white"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(image, ax=ax, label="delta_mean_return (%)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_lead_lag_report(
    config: dict,
    backfill_summary: pd.DataFrame,
    lead_lag_summary: pd.DataFrame,
    plot_paths: list[str],
) -> str:
    leaders = ", ".join(config["analysis"]["leader_symbols"])
    target = config["analysis"]["target_symbol"]

    lines = ["# Tick Lead-Lag 연구", ""]
    lines.append("## 설정")
    lines.append(f"- 리더 심볼: {leaders}")
    lines.append(f"- 타깃 심볼: {target}")
    lines.append(f"- 구간: {config['data'].get('resolved_start_utc', config['data'].get('start'))} ~ {config['data'].get('resolved_end_utc', config['data'].get('end'))}")
    lines.append(f"- 버킷: {int(config['analysis']['interval_minutes'][0])}분")
    lines.append(f"- 타깃 선행 수익률: {int(config['analysis']['forward_horizons_minutes'][0])}분")
    lines.append("")

    if not backfill_summary.empty:
        ok = backfill_summary.loc[backfill_summary["status"] == "ok"]
        lines.append("## Tick 백필")
        lines.append(f"- 성공: {int(len(ok))} / {int(len(backfill_summary))}")
        lines.append(f"- 새 다운로드: {int((ok['source_used'] == 'download').sum())}")
        lines.append(f"- 캐시 재사용: {int((ok['source_used'] == 'cache').sum())}")
        lines.append("")

    lines.append("## 시나리오별 결과")
    if lead_lag_summary.empty:
        lines.append("- 결과가 없습니다.")
    else:
        for _, row in lead_lag_summary.iterrows():
            lines.append(
                f"- {row['scenario_name']}: 이벤트 {row['event_mean_return']:.4%}, "
                f"대조군 {row['control_mean_return']:.4%}, 차이 {row['delta_mean_return']:.4%}, "
                f"차이 t={row['delta_t_stat']:.2f}, 이벤트 {int(row['event_count'])}건"
            )
    lines.append("")

    lines.append("## 해석")
    if not lead_lag_summary.empty:
        best = lead_lag_summary.iloc[0]
        lines.append(
            f"- 현재 가장 강한 lead-lag 조건은 {best['scenario_name']}이며, "
            f"XRP 다음 30분 반응 차이는 {best['delta_mean_return']:.4%}입니다."
        )
        lines.append("- 이 결과는 리더 심볼의 micro-herding 이벤트가 XRP short-horizon 반응을 선행 설명하는지 보는 실험입니다.")
    lines.append("")

    lines.append("## 플롯")
    for plot_path in plot_paths:
        lines.append(f"- {plot_path}")
    lines.append("")
    return "\n".join(lines)


def plot_lead_lag_summary(summary: pd.DataFrame, path: str | Path) -> None:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 5))
    if summary.empty:
        ax.text(0.5, 0.5, "표시할 결과가 없습니다.", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    ordered = summary.sort_values("delta_mean_return", ascending=False)
    colors = ["#54A24B" if value >= 0 else "#E45756" for value in ordered["delta_mean_return"]]
    ax.bar(ordered["scenario_name"], ordered["delta_mean_return"] * 100.0, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("이벤트-대조군 차이 (%)")
    ax.set_title("리더 이벤트가 XRP 다음 30분에 주는 영향")
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _pivot_to_markdown(pivot: pd.DataFrame, formatter) -> str:
    columns = [str(col).replace("USDT", "") for col in pivot.columns]
    header = "| target | " + " | ".join(columns) + " |"
    separator = "| --- | " + " | ".join(["---"] * len(columns)) + " |"
    body_lines: list[str] = []
    for index_value, row in pivot.iterrows():
        cells: list[str] = []
        for value in row.values:
            cells.append(formatter(value) if pd.notna(value) else "")
        body_lines.append(f"| {str(index_value).replace('USDT', '')} | " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *body_lines])


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
    denominator = np.sqrt((event_var / event_sample.shape[0]) + (control_var / control_sample.shape[0]))
    if denominator == 0 or np.isnan(denominator):
        return np.nan
    return float((event_sample.mean() - control_sample.mean()) / denominator)

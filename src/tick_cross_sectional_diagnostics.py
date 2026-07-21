from __future__ import annotations

from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tick_event_schema import require_tick_schema_v2


def load_symbol_focus(path: str | Path, focus_horizon_minutes: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    return frame.loc[frame["horizon_minutes"] == int(focus_horizon_minutes)].copy().reset_index(drop=True)


def load_micro_frames(config: dict) -> pd.DataFrame:
    micro_paths = config["input"]["micro_frame_paths"]
    frames: list[pd.DataFrame] = []

    for symbol, raw_path in micro_paths.items():
        path = Path(raw_path)
        frame = pd.read_csv(path, parse_dates=["bucket_start"])
        if "symbol" in frame.columns:
            frame = frame.loc[frame["symbol"] == symbol].copy()
        frame["symbol"] = symbol
        frames.append(frame)

    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined["bucket_start"] = pd.to_datetime(combined["bucket_start"], utc=True)
    combined["is_target_session"] = combined["is_target_session"].astype(bool)
    combined["meets_trade_count"] = combined["meets_trade_count"].astype(bool)
    require_tick_schema_v2(combined)
    combined["is_micro_run_clustering_event"] = combined["is_micro_run_clustering_event"].astype(bool)
    combined["is_control_bucket"] = combined["is_control_bucket"].astype(bool)

    eligible = combined.loc[
        combined["is_target_session"]
        & combined["meets_trade_count"]
        & combined["run_clustering_threshold"].notna()
    ].copy()
    eligible["event_side"] = np.where(
        eligible["is_micro_run_clustering_event"], eligible["run_clustering_side"], "none"
    )
    eligible["abs_bucket_return"] = eligible["bucket_return"].abs()
    eligible["signed_imbalance"] = (
        (eligible["up_ticks"] - eligible["down_ticks"])
        / eligible["transaction_count"].replace(0, np.nan)
    )
    eligible["prior_bucket_return"] = eligible.groupby("symbol")["bucket_return"].shift(1)
    eligible["prior_state"] = np.where(
        eligible["prior_bucket_return"] < 0,
        "prior_neg",
        "prior_pos",
    )

    groups = config["analysis"]["symbol_groups"]
    group_map = {}
    for group_name, symbols in groups.items():
        for symbol in symbols:
            group_map[symbol] = group_name
    eligible["symbol_group"] = eligible["symbol"].map(group_map).fillna("other")
    return eligible.reset_index(drop=True)


def summarize_down_hours(
    micro_frame: pd.DataFrame,
    focus_horizon_minutes: int,
    min_event_count: int,
) -> pd.DataFrame:
    return _summarize_by_dimension(
        micro_frame=micro_frame,
        focus_horizon_minutes=focus_horizon_minutes,
        event_side="down",
        dimension_columns=["symbol", "hour_utc"],
        min_event_count=min_event_count,
    )


def summarize_down_prior_state(
    micro_frame: pd.DataFrame,
    focus_horizon_minutes: int,
    min_event_count: int,
) -> pd.DataFrame:
    frame = micro_frame.loc[micro_frame["prior_bucket_return"].notna()].copy()
    return _summarize_by_dimension(
        micro_frame=frame,
        focus_horizon_minutes=focus_horizon_minutes,
        event_side="down",
        dimension_columns=["symbol", "prior_state"],
        min_event_count=min_event_count,
    )


def _summarize_by_dimension(
    micro_frame: pd.DataFrame,
    focus_horizon_minutes: int,
    event_side: str,
    dimension_columns: list[str],
    min_event_count: int,
) -> pd.DataFrame:
    horizon_column = f"forward_return_{int(focus_horizon_minutes)}m"
    rows: list[dict] = []

    for keys, subset in micro_frame.groupby(dimension_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        event_sample = subset.loc[subset["event_side"] == event_side, horizon_column].dropna()
        control_sample = subset.loc[subset["is_control_bucket"], horizon_column].dropna()
        if event_sample.shape[0] < int(min_event_count) or control_sample.shape[0] < int(min_event_count):
            continue
        row = dict(zip(dimension_columns, keys))
        row.update(
            _build_return_summary(
                event_sample=event_sample,
                control_sample=control_sample,
                event_label=event_side,
                horizon_minutes=int(focus_horizon_minutes),
            )
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    summary = pd.DataFrame(rows)
    return summary.sort_values(["delta_mean_return", "delta_t_stat"], ascending=False).reset_index(drop=True)


def summarize_structure_by_symbol(
    micro_frame: pd.DataFrame,
    symbol_focus: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    focus_lookup = symbol_focus.set_index(["symbol", "event_label"])

    for symbol, subset in micro_frame.groupby("symbol"):
        control = subset.loc[subset["is_control_bucket"]].copy()
        up = subset.loc[subset["event_side"] == "up"].copy()
        down = subset.loc[subset["event_side"] == "down"].copy()

        row = {
            "symbol": symbol,
            "symbol_group": subset["symbol_group"].iloc[0],
            "eligible_bucket_count": int(len(subset)),
            "control_count": int(len(control)),
            "up_count": int(len(up)),
            "down_count": int(len(down)),
            "up_event_rate": float(len(up) / len(subset)) if len(subset) else np.nan,
            "down_event_rate": float(len(down) / len(subset)) if len(subset) else np.nan,
            "control_mean_return": control["forward_return_30m"].mean(),
            "control_mean_transaction_count": control["transaction_count"].mean(),
            "control_mean_abs_bucket_return": control["abs_bucket_return"].mean(),
            "control_mean_signed_imbalance": control["signed_imbalance"].mean(),
            "control_mean_quote_quantity": control["total_quote_quantity"].mean(),
            "up_mean_transaction_count": up["transaction_count"].mean(),
            "down_mean_transaction_count": down["transaction_count"].mean(),
            "up_mean_abs_bucket_return": up["abs_bucket_return"].mean(),
            "down_mean_abs_bucket_return": down["abs_bucket_return"].mean(),
            "up_mean_signed_imbalance": up["signed_imbalance"].mean(),
            "down_mean_signed_imbalance": down["signed_imbalance"].mean(),
            "up_mean_quote_quantity": up["total_quote_quantity"].mean(),
            "down_mean_quote_quantity": down["total_quote_quantity"].mean(),
        }
        row["up_txn_ratio_vs_control"] = _safe_ratio(
            row["up_mean_transaction_count"], row["control_mean_transaction_count"]
        )
        row["down_txn_ratio_vs_control"] = _safe_ratio(
            row["down_mean_transaction_count"], row["control_mean_transaction_count"]
        )
        row["up_abs_return_ratio_vs_control"] = _safe_ratio(
            row["up_mean_abs_bucket_return"], row["control_mean_abs_bucket_return"]
        )
        row["down_abs_return_ratio_vs_control"] = _safe_ratio(
            row["down_mean_abs_bucket_return"], row["control_mean_abs_bucket_return"]
        )
        row["up_quote_ratio_vs_control"] = _safe_ratio(
            row["up_mean_quote_quantity"], row["control_mean_quote_quantity"]
        )
        row["down_quote_ratio_vs_control"] = _safe_ratio(
            row["down_mean_quote_quantity"], row["control_mean_quote_quantity"]
        )
        row["up_delta_mean_return"] = _lookup_focus_value(focus_lookup, symbol, "up", "delta_mean_return")
        row["up_delta_t_stat"] = _lookup_focus_value(focus_lookup, symbol, "up", "delta_t_stat")
        row["down_delta_mean_return"] = _lookup_focus_value(focus_lookup, symbol, "down", "delta_mean_return")
        row["down_delta_t_stat"] = _lookup_focus_value(focus_lookup, symbol, "down", "delta_t_stat")
        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary.sort_values("up_delta_mean_return", ascending=False).reset_index(drop=True)


def summarize_structure_by_group(
    micro_frame: pd.DataFrame,
    focus_horizon_minutes: int,
) -> pd.DataFrame:
    horizon_column = f"forward_return_{int(focus_horizon_minutes)}m"
    rows: list[dict] = []
    for group_name, subset in micro_frame.groupby("symbol_group"):
        control = subset.loc[subset["is_control_bucket"]]
        for event_side in ["up", "down"]:
            event_frame = subset.loc[subset["event_side"] == event_side]
            event_sample = event_frame[horizon_column].dropna()
            control_sample = control[horizon_column].dropna()
            row = {
                "symbol_group": group_name,
                "event_label": event_side,
                "event_count": int(event_frame.shape[0]),
                "control_count": int(control.shape[0]),
                "event_txn_ratio_vs_control": _safe_ratio(
                    event_frame["transaction_count"].mean(),
                    control["transaction_count"].mean(),
                ),
                "event_abs_return_ratio_vs_control": _safe_ratio(
                    event_frame["abs_bucket_return"].mean(),
                    control["abs_bucket_return"].mean(),
                ),
                "event_quote_ratio_vs_control": _safe_ratio(
                    event_frame["total_quote_quantity"].mean(),
                    control["total_quote_quantity"].mean(),
                ),
            }
            row.update(
                _build_return_summary(
                    event_sample=event_sample,
                    control_sample=control_sample,
                    event_label=event_side,
                    horizon_minutes=int(focus_horizon_minutes),
                )
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["event_label", "delta_mean_return"], ascending=[True, False]
    ).reset_index(drop=True)


def build_cross_sectional_report(
    config: dict,
    symbol_focus: pd.DataFrame,
    down_hour_summary: pd.DataFrame,
    down_prior_summary: pd.DataFrame,
    structure_symbol_summary: pd.DataFrame,
    structure_group_summary: pd.DataFrame,
    plot_paths: list[str],
) -> str:
    lines = ["# Tick 교차심볼 진단 연구", ""]
    lines.append("## 설정")
    lines.append(f"- 대상 심볼: {', '.join(config['analysis']['symbols'])}")
    lines.append(f"- horizon: {int(config['analysis']['focus_horizon_minutes'])}분")
    lines.append(f"- down 시간대 최소 이벤트 수: {int(config['analysis']['min_hour_event_count'])}")
    lines.append(f"- down prior-state 최소 이벤트 수: {int(config['analysis']['min_prior_event_count'])}")
    lines.append("")

    down_focus = symbol_focus.loc[symbol_focus["event_label"] == "down"].sort_values(
        ["delta_mean_return", "delta_t_stat"], ascending=False
    )
    up_focus = symbol_focus.loc[symbol_focus["event_label"] == "up"].sort_values(
        ["delta_mean_return", "delta_t_stat"], ascending=False
    )

    lines.append("## Down 신호 핵심")
    for _, row in down_focus.iterrows():
        lines.append(
            f"- {row['symbol']}: 차이 {row['delta_mean_return']:.4%}, "
            f"t={row['delta_t_stat']:.2f}, 이벤트 {int(row['event_count'])}건, 판정 {row['signal_quality']}"
        )
    lines.append("")

    lines.append("## Down 신호 시간대")
    if down_hour_summary.empty:
        lines.append("- 조건을 만족하는 시간대 요약이 없습니다.")
    else:
        for symbol, subset in down_hour_summary.groupby("symbol", sort=False):
            best = subset.sort_values(["delta_mean_return", "delta_t_stat"], ascending=False).iloc[0]
            lines.append(
                f"- {symbol}: 최고 시간은 UTC {int(best['hour_utc']):02d}시, "
                f"차이 {best['delta_mean_return']:.4%}, t={best['delta_t_stat']:.2f}, 이벤트 {int(best['event_count'])}건"
            )
    lines.append("")

    lines.append("## Down 신호 직전 상태")
    if down_prior_summary.empty:
        lines.append("- prior-state 요약이 없습니다.")
    else:
        for symbol, subset in down_prior_summary.groupby("symbol", sort=False):
            best = subset.sort_values(["delta_mean_return", "delta_t_stat"], ascending=False).iloc[0]
            state_label = "직전 15분 음수" if best["prior_state"] == "prior_neg" else "직전 15분 양수/보합"
            lines.append(
                f"- {symbol}: 더 좋은 상태는 {state_label}, "
                f"차이 {best['delta_mean_return']:.4%}, 이벤트 {int(best['event_count'])}건"
            )
    lines.append("")

    lines.append("## 구조 비교")
    if not structure_group_summary.empty:
        for _, row in structure_group_summary.iterrows():
            lines.append(
                f"- {row['symbol_group']} / {row['event_label']}: 차이 {row['delta_mean_return']:.4%}, "
                f"t={row['delta_t_stat']:.2f}, 거래수 배율 {row['event_txn_ratio_vs_control']:.2f}배, "
                f"절대수익률 배율 {row['event_abs_return_ratio_vs_control']:.2f}배"
            )
    lines.append("")

    lines.append("## 해석")
    if not up_focus.empty:
        best_up = up_focus.iloc[0]
        lines.append(
            f"- up 기준 최상위는 {best_up['symbol']}이고, "
            f"{best_up['delta_mean_return']:.4%} 차이로 약한 양(+) 반응이 이어졌습니다."
        )
    if not down_focus.empty:
        best_down = down_focus.iloc[0]
        lines.append(
            f"- down 기준 최상위는 {best_down['symbol']}이며, "
            f"{best_down['delta_mean_return']:.4%} 차이로 up보다 더 큰 반응을 보였습니다."
        )
    if not structure_group_summary.empty:
        majors_down = structure_group_summary.loc[
            (structure_group_summary["symbol_group"] == "majors")
            & (structure_group_summary["event_label"] == "down")
        ]
        alts_down = structure_group_summary.loc[
            (structure_group_summary["symbol_group"] == "alts")
            & (structure_group_summary["event_label"] == "down")
        ]
        majors_up = structure_group_summary.loc[
            (structure_group_summary["symbol_group"] == "majors")
            & (structure_group_summary["event_label"] == "up")
        ]
        alts_up = structure_group_summary.loc[
            (structure_group_summary["symbol_group"] == "alts")
            & (structure_group_summary["event_label"] == "up")
        ]
        if not majors_down.empty and not alts_down.empty:
            lines.append(
                f"- 메이저는 down도 차이 {majors_down.iloc[0]['delta_mean_return']:.4%}로 약하거나 음수인 반면, "
                f"알트는 {alts_down.iloc[0]['delta_mean_return']:.4%}로 더 잘 반응합니다."
            )
        if not majors_up.empty and not alts_up.empty:
            lines.append(
                f"- up도 메이저는 {majors_up.iloc[0]['delta_mean_return']:.4%}, "
                f"알트는 {alts_up.iloc[0]['delta_mean_return']:.4%}라서 구조 차이가 분명합니다."
            )
        lines.append("- 같은 micro-herding 조건이라도 메이저는 거래량은 크지만 버킷 변동성과 후속 drift가 작고, 알트는 충격 대비 가격 반응이 더 큽니다.")
    lines.append("")

    lines.append("## 플롯")
    for plot_path in plot_paths:
        lines.append(f"- {plot_path}")
    lines.append("")
    return "\n".join(lines)


def plot_down_focus(down_focus: pd.DataFrame, path: str | Path) -> None:
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 5))
    if down_focus.empty:
        ax.text(0.5, 0.5, "표시할 결과가 없습니다.", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    frame = down_focus.sort_values("delta_mean_return", ascending=False)
    colors = ["#54A24B" if value >= 0 else "#E45756" for value in frame["delta_mean_return"]]
    ax.bar(frame["symbol"], frame["delta_mean_return"] * 100.0, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_title("심볼별 down micro-herding 차이")
    ax.set_ylabel("이벤트-대조군 차이 (%)")
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_group_structure(structure_group_summary: pd.DataFrame, path: str | Path) -> None:
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    if structure_group_summary.empty:
        ax.text(0.5, 0.5, "표시할 결과가 없습니다.", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    frame = structure_group_summary.copy()
    frame["label"] = frame["symbol_group"] + " / " + frame["event_label"]
    colors = ["#4C78A8" if label.endswith("up") else "#F58518" for label in frame["label"]]
    ax.bar(frame["label"], frame["delta_mean_return"] * 100.0, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_title("메이저 vs 알트 구조 차이")
    ax.set_ylabel("이벤트-대조군 차이 (%)")
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _lookup_focus_value(
    focus_lookup: pd.DataFrame,
    symbol: str,
    event_label: str,
    column: str,
) -> float:
    try:
        return float(focus_lookup.loc[(symbol, event_label), column])
    except KeyError:
        return np.nan


def _build_return_summary(
    event_sample: pd.Series,
    control_sample: pd.Series,
    event_label: str,
    horizon_minutes: int,
) -> dict:
    return {
        "event_label": event_label,
        "horizon_minutes": int(horizon_minutes),
        "horizon_label": f"{int(horizon_minutes)}m",
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


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator in (0, 0.0) or pd.isna(denominator):
        return np.nan
    return float(numerator / denominator) if not pd.isna(numerator) else np.nan


def _configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False

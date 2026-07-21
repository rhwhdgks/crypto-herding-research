from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tick_event_schema import build_event_mask, require_tick_schema_v2


def load_candidate_micro_frames(micro_frame_paths: dict[str, str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol, raw_path in micro_frame_paths.items():
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
    eligible["prior_bucket_return"] = eligible.groupby("symbol")["bucket_return"].shift(1)
    eligible["prior_state"] = np.where(
        eligible["prior_bucket_return"] < 0,
        "prior_neg",
        np.where(eligible["prior_bucket_return"].notna(), "prior_pos", "unknown"),
    )
    eligible["strength_ratio"] = (
        eligible["run_clustering_score"].abs()
        / eligible["run_clustering_threshold"].abs().replace(0.0, np.nan)
    )
    return eligible.reset_index(drop=True)


def build_candidate_variant_trade_log(
    micro_frame: pd.DataFrame,
    candidates: list[dict],
    focus_horizon_minutes: int,
    round_trip_cost_bps: float,
) -> pd.DataFrame:
    horizon_column = f"forward_return_{int(focus_horizon_minutes)}m"
    cost_decimal = float(round_trip_cost_bps) / 10000.0
    frames: list[pd.DataFrame] = []

    for candidate in candidates:
        if "event_side" in candidate:
            raise ValueError("candidate.event_side is ambiguous; use candidate.event_filter")
        event_filter = candidate.get("event_filter", {})
        symbol = str(candidate["symbol"]).strip().upper()
        variant_name = str(candidate["variant_name"])
        variant_label = str(candidate.get("variant_label", variant_name))
        mask = (
            (micro_frame["symbol"] == symbol)
            & build_event_mask(micro_frame, event_filter)
            & micro_frame[horizon_column].notna()
        )

        hours = [int(value) for value in candidate.get("hours_utc", [])]
        if hours:
            mask &= micro_frame["hour_utc"].isin(hours)

        prior_state = candidate.get("prior_state")
        if prior_state:
            mask &= micro_frame["prior_state"] == str(prior_state)

        min_strength_ratio = candidate.get("min_strength_ratio")
        if min_strength_ratio is not None:
            mask &= micro_frame["strength_ratio"] >= float(min_strength_ratio)

        max_strength_ratio = candidate.get("max_strength_ratio")
        if max_strength_ratio is not None:
            mask &= micro_frame["strength_ratio"] <= float(max_strength_ratio)

        subset = micro_frame.loc[mask].copy()
        if subset.empty:
            continue

        subset["variant_name"] = variant_name
        subset["variant_label"] = variant_label
        subset["candidate_filter"] = str(event_filter)
        subset["entry_timestamp"] = subset["signal_timestamp"]
        subset["exit_timestamp"] = subset["signal_timestamp"] + pd.Timedelta(minutes=int(focus_horizon_minutes))
        subset["gross_return"] = subset[horizon_column].astype(float)
        subset["round_trip_cost_bps"] = float(round_trip_cost_bps)
        subset["net_return"] = subset["gross_return"] - cost_decimal
        subset["candidate_hours"] = ",".join(str(value) for value in hours) if hours else ""
        frames.append(
            subset[
                [
                    "variant_name",
                    "variant_label",
                    "symbol",
                    "candidate_filter",
                    "bucket_start",
                    "entry_timestamp",
                    "exit_timestamp",
                    "gross_return",
                    "net_return",
                    "round_trip_cost_bps",
                    "hour_utc",
                    "prior_state",
                    "strength_ratio",
                    "bucket_return",
                    "prior_bucket_return",
                    "transaction_count",
                    "total_quantity",
                    "total_quote_quantity",
                    "run_clustering_score",
                    "run_clustering_threshold",
                    "run_clustering_side",
                    "price_direction",
                    "aggressor_direction",
                    "event_label",
                    "candidate_hours",
                ]
            ].copy()
        )

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True).sort_values(
        ["variant_name", "entry_timestamp"]
    ).reset_index(drop=True)


def summarize_candidate_basket(trade_log: pd.DataFrame, recent_days_list: list[int]) -> pd.DataFrame:
    if trade_log.empty:
        return pd.DataFrame()

    frame = trade_log.copy()
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    last_date = frame["entry_timestamp"].dt.normalize().max()
    period_masks: dict[str, pd.Series] = {"full_sample": pd.Series(True, index=frame.index)}

    for recent_days in sorted({int(value) for value in recent_days_list}):
        cutoff = last_date - pd.Timedelta(days=max(recent_days - 1, 0))
        period_masks[f"recent_{recent_days}d"] = frame["entry_timestamp"].dt.normalize() >= cutoff

    rows: list[dict] = []
    for variant_name, group in frame.groupby("variant_name", sort=True):
        for period_name, period_mask in period_masks.items():
            subset = group.loc[period_mask.loc[group.index]].copy()
            if subset.empty:
                continue
            ordered = subset.sort_values("entry_timestamp")
            equity = (1.0 + ordered["net_return"]).cumprod()
            rows.append(
                {
                    "variant_name": variant_name,
                    "variant_label": ordered["variant_label"].iloc[0],
                    "period_name": period_name,
                    "trade_count": int(len(ordered)),
                    "mean_gross_return": ordered["gross_return"].mean(),
                    "mean_net_return": ordered["net_return"].mean(),
                    "median_net_return": ordered["net_return"].median(),
                    "net_win_rate": (ordered["net_return"] > 0).mean(),
                    "terminal_cumulative_net_return": float(equity.iloc[-1] - 1.0),
                    "avg_strength_ratio": ordered["strength_ratio"].mean(),
                }
            )
    return pd.DataFrame(rows)


def build_candidate_basket_report(
    summary: pd.DataFrame,
    recent_signals: pd.DataFrame,
    as_of_utc: pd.Timestamp,
    plot_paths: list[str],
) -> str:
    lines = ["# Tick Candidate Basket Tracker", ""]
    lines.append("## 기준 시각")
    lines.append(f"- as_of_utc: {as_of_utc}")
    lines.append("")

    lines.append("## 기간별 요약")
    if summary.empty:
        lines.append("- 요약 결과가 없습니다.")
    else:
        for variant_name, group in summary.groupby("variant_name", sort=True):
            label = group["variant_label"].iloc[0]
            lines.append(f"- {label}")
            for _, row in group.iterrows():
                lines.append(
                    f"  - {row['period_name']}: 거래 {int(row['trade_count'])}건, 평균 순수익 {row['mean_net_return']:.4%}, "
                    f"승률 {row['net_win_rate']:.2%}, 누적 {row['terminal_cumulative_net_return']:.2%}, "
                    f"평균 강도비율 {row['avg_strength_ratio']:.2f}"
                )
    lines.append("")

    lines.append("## 최근 종료 신호")
    if recent_signals.empty:
        lines.append("- 최근 종료 신호가 없습니다.")
    else:
        for variant_name, group in recent_signals.groupby("variant_name", sort=True):
            label = group["variant_label"].iloc[0]
            lines.append(f"- {label}")
            for _, row in group.head(6).iterrows():
                lines.append(
                    f"  - {row['entry_timestamp']} | net {float(row['realized_net_return']):.4%} | "
                    f"강도비율 {float(row['strength_ratio']):.2f}"
                )
    lines.append("")

    lines.append("## 해석")
    if not summary.empty:
        full = summary.loc[summary["period_name"] == "full_sample"].sort_values(
            "mean_net_return", ascending=False
        )
        top = full.iloc[0]
        lines.append(
            f"- 현재 basket에서 전체 평균 순수익이 가장 좋은 규칙은 `{top['variant_label']}`입니다."
        )
        lines.append("- 이 tracker는 여러 심볼 후보를 같은 포맷으로 계속 관찰하기 위한 실전 전 단계 모니터링입니다.")
    lines.append("")

    lines.append("## 플롯")
    for plot_path in plot_paths:
        lines.append(f"- {plot_path}")
    lines.append("")
    return "\n".join(lines)

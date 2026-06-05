from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tick_data import (
    iter_tick_archive_chunks,
    iter_tick_dates,
    iter_tick_months,
    load_tick_day,
    load_tick_month,
    resolve_tick_date_window,
)
from tick_herding import classify_tick_runs


def build_tick_short_horizon_dataset(config: dict) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    data_cfg = resolve_tick_date_window(config["data"])
    intervals = [int(value) for value in config["analysis"]["interval_minutes"]]
    symbols = list(data_cfg["symbols"])
    dates = iter_tick_dates(data_cfg["start"], data_cfg["end"])
    month_starts = iter_tick_months(data_cfg["start"], data_cfg["end"])
    prefer_monthly_archives = bool(data_cfg.get("prefer_monthly_archives", False))
    monthly_chunk_rows = int(data_cfg.get("monthly_chunk_rows", 1_000_000))
    start_ts = pd.Timestamp(data_cfg["start"], tz="UTC")
    end_ts = pd.Timestamp(data_cfg["end"], tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    bucket_frames_by_interval: dict[int, list[pd.DataFrame]] = {interval: [] for interval in intervals}
    load_records: list[dict] = []
    previous_price_by_interval: dict[tuple[str, int], float | None] = {
        (symbol, interval): None for symbol in symbols for interval in intervals
    }

    for symbol in symbols:
        if prefer_monthly_archives:
            daily_dates_by_month = _build_daily_tail_dates(dates)
            for month_start in month_starts:
                month_key = month_start.strftime("%Y-%m")
                month_end = (month_start + pd.offsets.MonthEnd(1)).normalize()
                use_monthly = month_end < end_ts.normalize()
                if use_monthly:
                    try:
                        archive_path, _, load_summary = load_tick_month(symbol=symbol, month_start=month_start, data_cfg=data_cfg)
                    except FileNotFoundError:
                        use_monthly = False
                    else:
                        rows_loaded = 0
                        month_frames, previous_price_by_interval = process_month_archive_to_bucket_frames(
                            archive_path=archive_path,
                            trade_kind=str(data_cfg.get("trade_kind", "aggTrades")),
                            symbol=symbol,
                            intervals=intervals,
                            previous_price_by_interval=previous_price_by_interval,
                            monthly_chunk_rows=monthly_chunk_rows,
                            start_ts=start_ts,
                            end_ts=end_ts,
                        )
                        rows_loaded = int(sum(len(frame) for frame in month_frames.values()))
                        load_summary["rows_loaded"] = rows_loaded
                        load_records.append(load_summary)
                        for interval, frames in month_frames.items():
                            if frames:
                                bucket_frames_by_interval[interval].extend(frames)
                if not use_monthly:
                    for date in daily_dates_by_month.get(month_key, []):
                        try:
                            trade_frame, load_summary = load_tick_day(symbol=symbol, date=date, data_cfg=data_cfg)
                        except FileNotFoundError:
                            continue
                        load_records.append(load_summary)
                        for interval in intervals:
                            bucket_frame, last_price = build_bucket_features_for_day(
                                trade_frame=trade_frame,
                                symbol=symbol,
                                interval_minutes=interval,
                                previous_price=previous_price_by_interval[(symbol, interval)],
                            )
                            previous_price_by_interval[(symbol, interval)] = last_price
                            if not bucket_frame.empty:
                                bucket_frames_by_interval[interval].append(bucket_frame)
        else:
            for date in dates:
                try:
                    trade_frame, load_summary = load_tick_day(symbol=symbol, date=date, data_cfg=data_cfg)
                except FileNotFoundError:
                    continue

                load_records.append(load_summary)
                for interval in intervals:
                    bucket_frame, last_price = build_bucket_features_for_day(
                        trade_frame=trade_frame,
                        symbol=symbol,
                        interval_minutes=interval,
                        previous_price=previous_price_by_interval[(symbol, interval)],
                    )
                    previous_price_by_interval[(symbol, interval)] = last_price
                    if not bucket_frame.empty:
                        bucket_frames_by_interval[interval].append(bucket_frame)

    final_frames: dict[int, pd.DataFrame] = {}
    for interval, frames in bucket_frames_by_interval.items():
        final_frames[interval] = (
            pd.concat(frames, axis=0, ignore_index=True).sort_values(["symbol", "bucket_start"]).reset_index(drop=True)
            if frames
            else pd.DataFrame()
        )
    return final_frames, pd.DataFrame(load_records)


def process_month_archive_to_bucket_frames(
    archive_path: Path,
    trade_kind: str,
    symbol: str,
    intervals: list[int],
    previous_price_by_interval: dict[tuple[str, int], float | None],
    monthly_chunk_rows: int,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> tuple[dict[int, list[pd.DataFrame]], dict[tuple[str, int], float | None]]:
    month_frames: dict[int, list[pd.DataFrame]] = {interval: [] for interval in intervals}
    carry_frame = pd.DataFrame()

    for chunk in iter_tick_archive_chunks(archive_path=archive_path, trade_kind=trade_kind, chunksize=monthly_chunk_rows):
        filtered = chunk[(chunk["timestamp"] >= start_ts) & (chunk["timestamp"] <= end_ts)].copy()
        if filtered.empty:
            continue
        combined = pd.concat([carry_frame, filtered], axis=0, ignore_index=True) if not carry_frame.empty else filtered
        combined["trade_date"] = combined["timestamp"].dt.normalize()
        last_date = combined["trade_date"].iloc[-1]
        process_frame = combined[combined["trade_date"] != last_date].copy()
        carry_frame = combined[combined["trade_date"] == last_date].copy().drop(columns=["trade_date"])

        if process_frame.empty:
            continue
        for _, day_frame in process_frame.groupby("trade_date", sort=True):
            day_frame = day_frame.drop(columns=["trade_date"]).copy()
            for interval in intervals:
                bucket_frame, last_price = build_bucket_features_for_day(
                    trade_frame=day_frame,
                    symbol=symbol,
                    interval_minutes=interval,
                    previous_price=previous_price_by_interval[(symbol, interval)],
                )
                previous_price_by_interval[(symbol, interval)] = last_price
                if not bucket_frame.empty:
                    month_frames[interval].append(bucket_frame)

    if not carry_frame.empty:
        for interval in intervals:
            bucket_frame, last_price = build_bucket_features_for_day(
                trade_frame=carry_frame.copy(),
                symbol=symbol,
                interval_minutes=interval,
                previous_price=previous_price_by_interval[(symbol, interval)],
            )
            previous_price_by_interval[(symbol, interval)] = last_price
            if not bucket_frame.empty:
                month_frames[interval].append(bucket_frame)

    return month_frames, previous_price_by_interval


def _build_daily_tail_dates(dates: list[pd.Timestamp]) -> dict[str, list[pd.Timestamp]]:
    mapping: dict[str, list[pd.Timestamp]] = {}
    for date in dates:
        mapping.setdefault(date.strftime("%Y-%m"), []).append(date)
    return mapping


def build_bucket_features_for_day(
    trade_frame: pd.DataFrame,
    symbol: str,
    interval_minutes: int,
    previous_price: float | None,
) -> tuple[pd.DataFrame, float | None]:
    if trade_frame.empty:
        return pd.DataFrame(), previous_price

    frame = trade_frame.sort_values("timestamp").reset_index(drop=True).copy()
    bucket_frequency = f"{int(interval_minutes)}min"
    frame["bucket_start"] = frame["timestamp"].dt.floor(bucket_frequency)

    bucket_records: list[dict] = []
    carry_price = previous_price
    for bucket_start, bucket_ticks in frame.groupby("bucket_start", sort=True):
        classified = classify_tick_runs(bucket_ticks.reset_index(drop=True), previous_price=carry_price)
        bucket_records.append(
            compute_bucket_tick_statistics(
                classified_ticks=classified,
                symbol=symbol,
                bucket_start=pd.Timestamp(bucket_start),
                interval_minutes=int(interval_minutes),
            )
        )
        carry_price = float(bucket_ticks["price"].iloc[-1])

    bucket_frame = pd.DataFrame(bucket_records)
    return bucket_frame, carry_price


def compute_bucket_tick_statistics(
    classified_ticks: pd.DataFrame,
    symbol: str,
    bucket_start: pd.Timestamp,
    interval_minutes: int,
) -> dict:
    if classified_ticks.empty:
        return {
            "symbol": symbol,
            "bucket_start": bucket_start,
            "interval_minutes": int(interval_minutes),
        }

    n_transactions = int(len(classified_ticks))
    run_frame = classified_ticks[["tick_type", "run_id", "run_length"]].drop_duplicates()
    run_counts = run_frame["tick_type"].value_counts().to_dict()
    tick_counts = classified_ticks["tick_type"].value_counts().to_dict()

    h_p = _compute_herding_intensity(int(run_counts.get("up", 0)), n_transactions)
    h_n = _compute_herding_intensity(int(run_counts.get("down", 0)), n_transactions)
    h_z = _compute_herding_intensity(int(run_counts.get("zero", 0)), n_transactions)

    score_components = {"up": h_p, "down": h_n}
    dominant_side = min(score_components, key=lambda key: score_components[key] if pd.notna(score_components[key]) else np.inf)
    herding_score = score_components[dominant_side]

    first_price = float(classified_ticks["price"].iloc[0])
    last_price = float(classified_ticks["price"].iloc[-1])

    return {
        "symbol": symbol,
        "bucket_start": bucket_start,
        "interval_minutes": int(interval_minutes),
        "first_timestamp": classified_ticks["timestamp"].min(),
        "last_timestamp": classified_ticks["timestamp"].max(),
        "transaction_count": n_transactions,
        "first_price": first_price,
        "last_price": last_price,
        "bucket_return": float(last_price / first_price - 1.0) if first_price > 0 else np.nan,
        "total_quantity": float(classified_ticks["quantity"].sum()),
        "total_quote_quantity": float(classified_ticks["quote_quantity"].sum()),
        "up_ticks": int(tick_counts.get("up", 0)),
        "down_ticks": int(tick_counts.get("down", 0)),
        "zero_ticks": int(tick_counts.get("zero", 0)),
        "up_runs": int(run_counts.get("up", 0)),
        "down_runs": int(run_counts.get("down", 0)),
        "zero_runs": int(run_counts.get("zero", 0)),
        "h_p": h_p,
        "h_n": h_n,
        "h_z": h_z,
        "herding_score": herding_score,
        "dominant_side": dominant_side,
    }


def prepare_micro_herding_frame(bucket_frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    if bucket_frame.empty:
        return pd.DataFrame()

    analysis_cfg = config["analysis"]
    session_hours = set(int(value) for value in analysis_cfg["session_hours_utc"])
    percentile = float(analysis_cfg["herding_score_percentile"])
    lookback_days = int(analysis_cfg["lookback_days_for_threshold"])
    min_trades = int(analysis_cfg["min_trades_per_bucket"])

    frame = bucket_frame.copy().sort_values(["symbol", "bucket_start"]).reset_index(drop=True)
    frame["bucket_start"] = pd.to_datetime(frame["bucket_start"], utc=True)
    frame["hour_utc"] = frame["bucket_start"].dt.hour
    frame["session_utc"] = frame["hour_utc"].map(_map_session_label)
    frame["is_target_session"] = frame["hour_utc"].isin(session_hours)
    frame["meets_trade_count"] = frame["transaction_count"] >= min_trades

    for horizon in [int(value) for value in analysis_cfg["forward_horizons_minutes"]]:
        if horizon % int(frame["interval_minutes"].iloc[0]) != 0:
            continue
        step = int(horizon // int(frame["interval_minutes"].iloc[0]))
        frame[f"forward_return_{horizon}m"] = frame.groupby("symbol")["last_price"].shift(-step) / frame["last_price"] - 1.0

    window_buckets = int((lookback_days * 24 * 60) / int(frame["interval_minutes"].iloc[0]))
    frame["herding_threshold"] = (
        frame.groupby("symbol")["herding_score"]
        .transform(lambda series: series.rolling(window=window_buckets, min_periods=window_buckets).quantile(percentile).shift(1))
    )
    frame["is_micro_herding_event"] = (
        frame["is_target_session"]
        & frame["meets_trade_count"]
        & frame["herding_threshold"].notna()
        & (frame["herding_score"] <= frame["herding_threshold"])
    )
    frame["is_control_bucket"] = (
        frame["is_target_session"]
        & frame["meets_trade_count"]
        & frame["herding_threshold"].notna()
        & (~frame["is_micro_herding_event"])
    )
    frame["event_label"] = np.where(
        frame["is_micro_herding_event"],
        "micro_herding_" + frame["dominant_side"].astype(str),
        "none",
    )
    return frame


def summarize_micro_herding(
    micro_frame: pd.DataFrame,
    forward_horizons: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if micro_frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    horizons = [int(value) for value in forward_horizons if f"forward_return_{int(value)}m" in micro_frame.columns]
    pooled_rows: list[dict] = []
    symbol_rows: list[dict] = []

    event_specs = [
        ("all", micro_frame["is_micro_herding_event"]),
        ("up", micro_frame["event_label"] == "micro_herding_up"),
        ("down", micro_frame["event_label"] == "micro_herding_down"),
    ]

    for horizon in horizons:
        return_column = f"forward_return_{horizon}m"
        control_sample = micro_frame.loc[micro_frame["is_control_bucket"], return_column].dropna()
        for label, event_mask in event_specs:
            event_sample = micro_frame.loc[event_mask, return_column].dropna()
            pooled_rows.append(
                _build_summary_row(
                    symbol="pooled",
                    event_label=label,
                    horizon_minutes=horizon,
                    event_sample=event_sample,
                    control_sample=control_sample,
                )
            )

        for symbol, symbol_frame in micro_frame.groupby("symbol"):
            symbol_control_sample = symbol_frame.loc[symbol_frame["is_control_bucket"], return_column].dropna()
            for label, event_mask in event_specs:
                local_mask = event_mask.loc[symbol_frame.index]
                event_sample = symbol_frame.loc[local_mask, return_column].dropna()
                symbol_rows.append(
                    _build_summary_row(
                        symbol=symbol,
                        event_label=label,
                        horizon_minutes=horizon,
                        event_sample=event_sample,
                        control_sample=symbol_control_sample,
                    )
                )

    return pd.DataFrame(pooled_rows), pd.DataFrame(symbol_rows)


def build_tick_short_horizon_report(
    config: dict,
    pooled_summary_by_interval: dict[int, pd.DataFrame],
    symbol_summary_by_interval: dict[int, pd.DataFrame],
    plot_paths: list[str],
) -> str:
    lines = ["# Tick 단기 미시구조 연구", ""]
    lines.append("## 설정")
    lines.extend(
        [
            f"- 종목: {', '.join(config['data']['symbols'])}",
            f"- 거래 데이터 종류: {config['data'].get('trade_kind', 'aggTrades')}",
            f"- 구간: {config['data'].get('start', 'resolved')} ~ {config['data'].get('end', 'resolved')}",
            f"- 버킷: {', '.join(str(v) + 'm' for v in config['analysis']['interval_minutes'])}",
            f"- 목표 세션 UTC: {min(config['analysis']['session_hours_utc'])}~{max(config['analysis']['session_hours_utc'])}시",
            f"- trailing herding score percentile: {float(config['analysis']['herding_score_percentile']):.2f}",
            "",
        ]
    )

    for interval in sorted(pooled_summary_by_interval):
        pooled = pooled_summary_by_interval[interval]
        lines.append(f"## {interval}분 버킷 요약")
        if pooled.empty:
            lines.append("- 결과가 없습니다.")
            lines.append("")
            continue
        for _, row in pooled.iterrows():
            lines.append(
                f"- {row['event_label']} / {row['horizon_label']}: 이벤트 {row['event_mean_return']:.4%}, "
                f"대조군 {row['control_mean_return']:.4%}, 차이 {row['delta_mean_return']:.4%}, "
                f"차이 t={row['delta_t_stat']:.2f}, 이벤트 {int(row['event_count'])}건 / 대조군 {int(row['control_count'])}건"
            )
        lines.append("")

    lines.append("## 해석")
    fifteen = pooled_summary_by_interval.get(15, pd.DataFrame())
    thirty = pooled_summary_by_interval.get(30, pd.DataFrame())
    if not fifteen.empty:
        best_15 = fifteen.sort_values(["delta_mean_return", "delta_t_stat"], ascending=False).iloc[0]
        lines.append(
            f"- 15분 버킷에서는 {best_15['event_label']} 이벤트의 {best_15['horizon_label']} 반응이 가장 좋았고, "
            f"이벤트-대조군 차이는 {best_15['delta_mean_return']:.4%}입니다."
        )
    if not thirty.empty:
        best_30 = thirty.sort_values(["delta_mean_return", "delta_t_stat"], ascending=False).iloc[0]
        lines.append(
            f"- 30분 버킷에서는 {best_30['event_label']} 이벤트의 {best_30['horizon_label']} 반응이 가장 좋았고, "
            f"이벤트-대조군 차이는 {best_30['delta_mean_return']:.4%}입니다."
        )
    lines.append("- 이 실험은 자동매매 구현이 아니라, short-horizon alpha가 tick run 구조에서 더 또렷해지는지 확인하는 연구 단계입니다.")
    lines.append("")

    lines.append("## 플롯")
    for plot_path in plot_paths:
        lines.append(f"- {plot_path}")
    lines.append("")
    return "\n".join(lines)


def plot_tick_short_horizon_bars(summary: pd.DataFrame, interval_minutes: int, path: str | Path) -> None:
    configure_korean_matplotlib_font()
    fig, ax = plt.subplots(figsize=(10, 5))
    if summary.empty:
        _render_empty_plot(fig, ax, f"{interval_minutes}분 버킷", "표시할 미시구조 결과가 없습니다.")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    plot_frame = summary.copy()
    plot_frame["label"] = plot_frame["event_label"] + " / " + plot_frame["horizon_label"]
    colors = ["#54A24B" if value >= 0 else "#E45756" for value in plot_frame["delta_mean_return"]]
    ax.bar(plot_frame["label"], plot_frame["delta_mean_return"] * 100.0, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("이벤트-대조군 차이 (%)")
    ax.set_title(f"{interval_minutes}분 버킷 tick micro-herding 결과")
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def configure_korean_matplotlib_font() -> None:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False


def _build_summary_row(
    symbol: str,
    event_label: str,
    horizon_minutes: int,
    event_sample: pd.Series,
    control_sample: pd.Series,
) -> dict:
    return {
        "symbol": symbol,
        "event_label": event_label,
        "horizon_minutes": int(horizon_minutes),
        "horizon_label": _horizon_to_label(int(horizon_minutes)),
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


def _map_session_label(hour: int) -> str:
    hour = int(hour)
    if 0 <= hour <= 7:
        return "00-07"
    if 8 <= hour <= 15:
        return "08-15"
    return "16-23"


def _compute_herding_intensity(runs: int, n_transactions: int) -> float:
    if n_transactions <= 0:
        return np.nan
    p_value = 1.0 / 3.0
    variance = p_value * (1.0 - p_value) - 3.0 * (p_value**2) * ((1.0 - p_value) ** 2)
    if variance <= 0:
        return np.nan
    chi = ((runs + 0.5) - (n_transactions * p_value * (1.0 - p_value))) / np.sqrt(n_transactions)
    return float(chi / np.sqrt(variance))


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


def _horizon_to_label(minutes: int) -> str:
    if minutes % 60 == 0:
        return f"{minutes // 60}h" if minutes < 1440 else f"{minutes // 1440}d"
    return f"{minutes}m"


def _render_empty_plot(fig: plt.Figure, ax: plt.Axes, title: str, message: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()

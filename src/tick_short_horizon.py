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
from tick_event_schema import (
    TICK_EVENT_SCHEMA_VERSION,
    TICK_PIPELINE_VERSION,
    build_event_mask,
    require_tick_schema_v2,
    validate_tick_analysis_config,
)
from tick_herding import classify_tick_runs, compute_conditional_run_z


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
                try:
                    archive_path, _, load_summary = load_tick_month(
                        symbol=symbol,
                        month_start=month_start,
                        data_cfg=data_cfg,
                    )
                except FileNotFoundError:
                    use_monthly = False
                else:
                    use_monthly = True
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
                    load_summary["rows_loaded"] = int(
                        sum(len(frame) for frame in month_frames.values())
                    )
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

    run_scores = {
        side: compute_conditional_run_z(
            runs=int(run_counts.get(side, 0)),
            category_count=int(tick_counts.get(side, 0)),
            n_transactions=n_transactions,
        )
        for side in ("up", "down", "zero")
    }
    valid_scores = {side: score for side, score in run_scores.items() if pd.notna(score)}
    run_clustering_side = min(valid_scores, key=valid_scores.get) if valid_scores else "none"
    run_clustering_score = valid_scores.get(run_clustering_side, np.nan)

    first_price = float(classified_ticks["price"].iloc[0])
    last_price = float(classified_ticks["price"].iloc[-1])
    bucket_end = bucket_start + pd.Timedelta(minutes=int(interval_minutes))
    maker_side = classified_ticks.get("is_buyer_maker", pd.Series(index=classified_ticks.index, dtype="boolean"))
    quote_volume = pd.to_numeric(classified_ticks.get("quote_quantity"), errors="coerce")
    available = maker_side.notna() & quote_volume.notna()
    buy_quote_volume = float(quote_volume.loc[available & maker_side.eq(False)].sum())
    sell_quote_volume = float(quote_volume.loc[available & maker_side.eq(True)].sum())
    aggressor_total = buy_quote_volume + sell_quote_volume
    aggressor_imbalance = (
        (buy_quote_volume - sell_quote_volume) / aggressor_total if aggressor_total > 0 else np.nan
    )

    return {
        "symbol": symbol,
        "bucket_start": bucket_start,
        "bucket_end": bucket_end,
        "signal_timestamp": bucket_end,
        "interval_minutes": int(interval_minutes),
        "schema_version": TICK_EVENT_SCHEMA_VERSION,
        "pipeline_version": TICK_PIPELINE_VERSION,
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
        "run_z_up": run_scores["up"],
        "run_z_down": run_scores["down"],
        "run_z_zero": run_scores["zero"],
        "run_clustering_score": run_clustering_score,
        "run_clustering_side": run_clustering_side,
        "aggressor_imbalance": aggressor_imbalance,
    }


def prepare_micro_herding_frame(bucket_frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    if bucket_frame.empty:
        return pd.DataFrame()

    analysis_cfg = config["analysis"]
    validate_tick_analysis_config(analysis_cfg)
    session_hours = set(int(value) for value in analysis_cfg["session_hours_utc"])
    percentile = float(analysis_cfg["run_clustering_score_percentile"])
    lookback_days = int(analysis_cfg["lookback_days_for_threshold"])
    min_trades = int(analysis_cfg["min_trades_per_bucket"])
    flat_return_epsilon = float(analysis_cfg.get("flat_return_epsilon", 0.0))

    frame = _reindex_complete_utc_grid(bucket_frame)
    frame["bucket_start"] = pd.to_datetime(frame["bucket_start"], utc=True)
    interval_minutes = int(frame["interval_minutes"].dropna().iloc[0])
    frame["bucket_end"] = frame["bucket_start"] + pd.to_timedelta(interval_minutes, unit="m")
    frame["signal_timestamp"] = frame["bucket_end"]
    frame["schema_version"] = TICK_EVENT_SCHEMA_VERSION
    frame["pipeline_version"] = TICK_PIPELINE_VERSION
    frame["hour_utc"] = frame["bucket_start"].dt.hour
    frame["session_utc"] = frame["hour_utc"].map(_map_session_label)
    frame["is_target_session"] = frame["hour_utc"].isin(session_hours)
    frame["meets_trade_count"] = frame["transaction_count"].fillna(0) >= min_trades
    frame["price_direction"] = np.select(
        [frame["bucket_return"] > flat_return_epsilon, frame["bucket_return"] < -flat_return_epsilon],
        ["up", "down"],
        default="flat",
    )
    frame.loc[frame["bucket_return"].isna(), "price_direction"] = "flat"
    balance_epsilon = float(analysis_cfg.get("aggressor_balance_epsilon", 0.0))
    frame["aggressor_direction"] = np.select(
        [frame["aggressor_imbalance"] > balance_epsilon, frame["aggressor_imbalance"] < -balance_epsilon],
        ["buy", "sell"],
        default="balanced",
    )
    frame.loc[frame["aggressor_imbalance"].isna(), "aggressor_direction"] = "unavailable"

    for horizon in [int(value) for value in analysis_cfg["forward_horizons_minutes"]]:
        if horizon % interval_minutes != 0:
            continue
        frame = _attach_exact_forward_return(frame, horizon)

    window_buckets = int((lookback_days * 24 * 60) / interval_minutes)
    frame["run_clustering_threshold"] = (
        frame.groupby("symbol")["run_clustering_score"]
        .transform(lambda series: series.rolling(window=window_buckets, min_periods=window_buckets).quantile(percentile).shift(1))
    )
    frame["is_micro_run_clustering_event"] = (
        frame["is_target_session"]
        & frame["meets_trade_count"]
        & frame["run_clustering_threshold"].notna()
        & (frame["run_clustering_score"] <= frame["run_clustering_threshold"])
    )
    frame["is_control_bucket"] = (
        frame["is_target_session"]
        & frame["meets_trade_count"]
        & frame["run_clustering_threshold"].notna()
        & (~frame["is_micro_run_clustering_event"])
    )
    frame["event_label"] = np.where(
        frame["is_micro_run_clustering_event"],
        "micro_run_clustering__run_side_"
        + frame["run_clustering_side"].astype(str)
        + "__price_"
        + frame["price_direction"].astype(str),
        "none",
    )
    return frame


def rebuild_bucket_schema_v2_from_run_counts(bucket_frame: pd.DataFrame) -> pd.DataFrame:
    """Explicitly rebuild v2 statistics from archived raw run/tick counts.

    Legacy fixed-p score columns are ignored. Aggressor fields are unavailable because
    historical bucket aggregates did not retain buyer-maker volume splits.
    """
    required = {
        "symbol",
        "bucket_start",
        "interval_minutes",
        "transaction_count",
        "bucket_return",
        "last_price",
        "up_ticks",
        "down_ticks",
        "zero_ticks",
        "up_runs",
        "down_runs",
        "zero_runs",
    }
    missing = sorted(required.difference(bucket_frame.columns))
    if missing:
        raise ValueError(f"Cannot rebuild v2 bucket schema; missing: {', '.join(missing)}")
    frame = bucket_frame.copy()
    frame["bucket_start"] = pd.to_datetime(frame["bucket_start"], utc=True)
    for side in ("up", "down", "zero"):
        n = pd.to_numeric(frame["transaction_count"], errors="coerce").to_numpy(dtype=float)
        k = pd.to_numeric(frame[f"{side}_ticks"], errors="coerce").to_numpy(dtype=float)
        r = pd.to_numeric(frame[f"{side}_runs"], errors="coerce").to_numpy(dtype=float)
        m = n - k
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = k * (m + 1.0) / n
            variance = k * m * (k - 1.0) * (m + 1.0) / ((n**2) * (n - 1.0))
            values = (r - mean) / np.sqrt(variance)
        degenerate = (n <= 1) | (k <= 0) | (k >= n) | (r < 0) | (variance <= 0) | ~np.isfinite(values)
        values[degenerate] = np.nan
        frame[f"run_z_{side}"] = values
    score_columns = ["run_z_up", "run_z_down", "run_z_zero"]
    scores = frame[score_columns]
    frame["run_clustering_score"] = scores.min(axis=1, skipna=True)
    all_missing = scores.isna().all(axis=1)
    safe_scores = scores.fillna(np.inf)
    frame["run_clustering_side"] = safe_scores.idxmin(axis=1).str.removeprefix("run_z_")
    frame.loc[all_missing, "run_clustering_side"] = "none"
    frame.loc[all_missing, "run_clustering_score"] = np.nan
    interval = pd.to_timedelta(frame["interval_minutes"], unit="m")
    frame["bucket_end"] = frame["bucket_start"] + interval
    frame["signal_timestamp"] = frame["bucket_end"]
    frame["aggressor_imbalance"] = np.nan
    frame["schema_version"] = TICK_EVENT_SCHEMA_VERSION
    frame["pipeline_version"] = TICK_PIPELINE_VERSION
    legacy_columns = [
        column
        for column in ["h_p", "h_n", "h_z", "herding_score", "dominant_side"]
        if column in frame.columns
    ]
    return frame.drop(columns=legacy_columns)


def _reindex_complete_utc_grid(bucket_frame: pd.DataFrame) -> pd.DataFrame:
    source = bucket_frame.copy()
    source["bucket_start"] = pd.to_datetime(source["bucket_start"], utc=True)
    interval_values = pd.to_numeric(source["interval_minutes"], errors="coerce").dropna().unique()
    if len(interval_values) != 1:
        raise ValueError("A micro frame must contain exactly one interval_minutes value")
    interval_minutes = int(interval_values[0])
    frames: list[pd.DataFrame] = []
    for symbol, group in source.groupby("symbol", sort=False):
        ordered = group.sort_values("bucket_start").drop_duplicates("bucket_start", keep="last")
        grid = pd.date_range(
            ordered["bucket_start"].min(),
            ordered["bucket_start"].max(),
            freq=f"{interval_minutes}min",
            tz="UTC",
        )
        complete = ordered.set_index("bucket_start").reindex(grid)
        complete.index.name = "bucket_start"
        complete["symbol"] = symbol
        complete["interval_minutes"] = interval_minutes
        complete["is_observed_bucket"] = complete["transaction_count"].notna()
        frames.append(complete.reset_index())
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "bucket_start"]).reset_index(drop=True)


def _attach_exact_forward_return(frame: pd.DataFrame, horizon_minutes: int) -> pd.DataFrame:
    result = frame.copy()
    horizon = int(horizon_minutes)
    lookup = result[["symbol", "bucket_start", "last_price"]].rename(
        columns={"bucket_start": "target_bucket_start", "last_price": "target_last_price"}
    )
    result["target_bucket_start"] = result["bucket_start"] + pd.to_timedelta(horizon, unit="m")
    result = result.merge(lookup, on=["symbol", "target_bucket_start"], how="left", validate="many_to_one")
    exact = result["last_price"].notna() & result["target_last_price"].notna()
    result[f"forward_return_{horizon}m"] = np.where(
        exact,
        result["target_last_price"] / result["last_price"] - 1.0,
        np.nan,
    )
    result[f"requested_horizon_minutes_{horizon}m"] = horizon
    result[f"realized_horizon_minutes_{horizon}m"] = np.where(exact, horizon, np.nan)
    result[f"horizon_is_exact_{horizon}m"] = exact
    return result.drop(columns=["target_bucket_start", "target_last_price"])


def summarize_micro_herding(
    micro_frame: pd.DataFrame,
    forward_horizons: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if micro_frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    horizons = [int(value) for value in forward_horizons if f"forward_return_{int(value)}m" in micro_frame.columns]
    pooled_rows: list[dict] = []
    symbol_rows: list[dict] = []

    require_tick_schema_v2(micro_frame)
    event_specs = [("all", build_event_mask(micro_frame))]
    event_specs.extend(
        (f"run_side_{side}", build_event_mask(micro_frame, {"run_clustering_side": [side]}))
        for side in ("up", "down", "zero")
    )

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
            f"- trailing run-clustering score percentile: {float(config['analysis']['run_clustering_score_percentile']):.2f}",
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

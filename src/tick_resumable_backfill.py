from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from tick_data import iter_tick_months, load_tick_month, resolve_tick_date_window
from utils import save_config_snapshot, save_dataframe, save_provenance_manifest


ProgressCallback = Callable[[dict], None]


def run_resumable_raw_backfill(
    config: dict,
    project_root: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    from tick_short_horizon import prepare_micro_herding_frame, summarize_micro_herding

    project_root = Path(project_root)
    data_cfg = resolve_tick_date_window(config["data"])
    intervals = [int(value) for value in config["analysis"]["interval_minutes"]]
    if len(intervals) != 1:
        raise ValueError("Resumable semantic backfill requires exactly one interval")
    interval = intervals[0]
    symbols = list(data_cfg["symbols"])
    months = iter_tick_months(data_cfg["start"], data_cfg["end"])
    output_dir = project_root / config["output"]["base_dir"]
    cache_dir = output_dir / "monthly_cache"
    intermediate_dir = output_dir / "intermediate"
    cache_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, output_dir / "config_snapshot.yaml")

    start_ts = pd.Timestamp(data_cfg["start"], tz="UTC")
    end_ts = pd.Timestamp(data_cfg["end"], tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    total_jobs = len(symbols) * len(months)
    progress_records: list[dict] = []
    progress_lock = threading.Lock()

    def record_progress(record: dict) -> None:
        with progress_lock:
            progress_records.append(record)
            state = _build_state(
                status="running",
                completed_jobs=len(progress_records),
                total_jobs=total_jobs,
                last_record=record,
            )
            _write_progress(output_dir, progress_records, state)
            if progress_callback is not None:
                progress_callback(state)

    max_workers = _resolve_max_workers(config.get("backfill", {}), len(symbols))
    if max_workers == 1:
        for symbol in symbols:
            _process_symbol_months(
                symbol=symbol,
                months=months,
                interval=interval,
                data_cfg=data_cfg,
                cache_dir=cache_dir,
                start_ts=start_ts,
                end_ts=end_ts,
                record_callback=record_progress,
            )
    else:
        # Months remain sequential inside each symbol so the prior-price state is exact.
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tick-symbol") as executor:
            futures = {
                executor.submit(
                    _process_symbol_months,
                    symbol,
                    months,
                    interval,
                    data_cfg,
                    cache_dir,
                    start_ts,
                    end_ts,
                    record_progress,
                ): symbol
                for symbol in symbols
            }
            for future in as_completed(futures):
                future.result()

    bucket_frame = assemble_monthly_caches(cache_dir, symbols, months, interval)
    expected_rows = _expected_bucket_rows(start_ts, end_ts, interval, len(symbols))
    if len(bucket_frame) != expected_rows:
        raise ValueError(
            f"Assembled bucket rows={len(bucket_frame):,} do not match expected complete grid={expected_rows:,}"
        )
    bucket_path = intermediate_dir / f"tick_bucket_features_{interval}m.parquet"
    _atomic_write_parquet(bucket_frame, bucket_path)
    micro_frame = prepare_micro_herding_frame(bucket_frame, config)
    micro_path = intermediate_dir / f"tick_micro_frame_{interval}m.parquet"
    _atomic_write_parquet(micro_frame, micro_path)

    pooled_summary, symbol_summary = summarize_micro_herding(
        micro_frame,
        config["analysis"]["forward_horizons_minutes"],
    )
    save_dataframe(
        pd.DataFrame(progress_records),
        output_dir / "tick_data_load_summary.csv",
        index=False,
    )
    save_dataframe(
        pooled_summary,
        output_dir / f"tick_micro_pooled_summary_{interval}m.csv",
        index=False,
    )
    save_dataframe(
        symbol_summary,
        output_dir / f"tick_micro_symbol_summary_{interval}m.csv",
        index=False,
    )
    input_manifest_path = _write_cache_manifest(cache_dir, output_dir / "input_manifest.json")
    save_provenance_manifest(
        config,
        output_dir / "provenance.json",
        schema_version=2,
        pipeline_version="tick-raw-resumable-backfill-v1",
        statistical_method="conditional-run-z; exact-clock-forward-return; monthly raw checkpoints",
        input_manifest_path=input_manifest_path,
        train_start=data_cfg["start"],
        train_end=data_cfg["end"],
    )
    final_state = _build_state(
        status="complete",
        completed_jobs=total_jobs,
        total_jobs=total_jobs,
        last_record=progress_records[-1] if progress_records else None,
    )
    final_state.update(
        {
            "bucket_rows": int(len(bucket_frame)),
            "micro_rows": int(len(micro_frame)),
            "expected_rows": int(expected_rows),
            "aggressor_available_share": float(micro_frame["aggressor_imbalance"].notna().mean()),
            "bucket_output": str(bucket_path),
            "micro_output": str(micro_path),
        }
    )
    _write_progress(output_dir, progress_records, final_state)
    return final_state


def _process_symbol_months(
    symbol: str,
    months: list[pd.Timestamp],
    interval: int,
    data_cfg: dict,
    cache_dir: Path,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    record_callback: Callable[[dict], None],
) -> None:
    from tick_short_horizon import process_month_archive_to_bucket_frames

    previous_price_by_interval: dict[tuple[str, int], float | None] = {
        (symbol, interval): None
    }
    for month_start in months:
        month_key = month_start.strftime("%Y-%m")
        cache_path = monthly_cache_path(cache_dir, symbol, month_key, interval)
        if cache_path.is_file():
            month_frame = pd.read_parquet(cache_path)
            validate_monthly_cache(month_frame, symbol, month_key, interval)
            cache_status = "reused"
            archive_path = _monthly_archive_path(data_cfg, symbol, month_key)
            source_used = "checkpoint"
        else:
            archive_path, source_used, _ = load_tick_month(
                symbol=symbol,
                month_start=month_start,
                data_cfg=data_cfg,
            )
            month_frames, previous_price_by_interval = process_month_archive_to_bucket_frames(
                archive_path=archive_path,
                trade_kind=str(data_cfg.get("trade_kind", "aggTrades")),
                symbol=symbol,
                intervals=[interval],
                previous_price_by_interval=previous_price_by_interval,
                monthly_chunk_rows=int(data_cfg.get("monthly_chunk_rows", 1_000_000)),
                start_ts=start_ts,
                end_ts=end_ts,
            )
            frames = month_frames.get(interval, [])
            month_frame = (
                pd.concat(frames, ignore_index=True).sort_values("bucket_start").reset_index(drop=True)
                if frames
                else pd.DataFrame()
            )
            if month_frame.empty:
                raise ValueError(f"No bucket rows produced for {symbol} {month_key}")
            validate_monthly_cache(month_frame, symbol, month_key, interval)
            _atomic_write_parquet(month_frame, cache_path)
            cache_status = "created"

        last_prices = pd.to_numeric(month_frame.get("last_price"), errors="coerce").dropna()
        if not last_prices.empty:
            previous_price_by_interval[(symbol, interval)] = float(last_prices.iloc[-1])
        record_callback(
            {
                "symbol": symbol,
                "month": month_key,
                "interval_minutes": interval,
                "cache_status": cache_status,
                "bucket_rows": int(len(month_frame)),
                "bucket_start": str(pd.to_datetime(month_frame["bucket_start"], utc=True).min()),
                "bucket_end": str(pd.to_datetime(month_frame["bucket_start"], utc=True).max()),
                "archive_path": str(archive_path),
                "archive_size_bytes": int(archive_path.stat().st_size) if archive_path.is_file() else None,
                "source_used": source_used,
                "cache_path": str(cache_path),
                "cache_sha256": _sha256_file(cache_path),
            }
        )


def _resolve_max_workers(backfill_cfg: dict, symbol_count: int) -> int:
    requested = int(backfill_cfg.get("max_workers", 1))
    if requested < 1:
        raise ValueError("backfill.max_workers must be at least 1")
    return min(requested, max(int(symbol_count), 1))


def monthly_cache_path(
    cache_dir: str | Path,
    symbol: str,
    month_key: str,
    interval: int,
) -> Path:
    return Path(cache_dir) / symbol / f"{symbol}_{month_key}_{int(interval)}m.parquet"


def validate_monthly_cache(
    frame: pd.DataFrame,
    symbol: str,
    month_key: str,
    interval: int,
) -> None:
    required = {
        "symbol",
        "bucket_start",
        "interval_minutes",
        "transaction_count",
        "last_price",
        "aggressor_imbalance",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Monthly cache is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"Monthly cache is empty: {symbol} {month_key}")
    if set(frame["symbol"].dropna().astype(str).unique()) != {symbol}:
        raise ValueError(f"Monthly cache symbol mismatch: {symbol} {month_key}")
    intervals = set(
        pd.to_numeric(frame["interval_minutes"], errors="coerce").dropna().astype(int).unique()
    )
    if intervals != {int(interval)}:
        raise ValueError(f"Monthly cache interval mismatch: {symbol} {month_key}")
    timestamps = pd.to_datetime(frame["bucket_start"], utc=True)
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError(f"Monthly cache timestamps are invalid: {symbol} {month_key}")
    if not timestamps.dt.strftime("%Y-%m").eq(month_key).all():
        raise ValueError(f"Monthly cache contains rows outside {month_key}: {symbol}")


def assemble_monthly_caches(
    cache_dir: str | Path,
    symbols: list[str],
    months: list[pd.Timestamp],
    interval: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        for month_start in months:
            month_key = month_start.strftime("%Y-%m")
            path = monthly_cache_path(cache_dir, symbol, month_key, interval)
            if not path.is_file():
                raise FileNotFoundError(f"Missing monthly checkpoint: {path}")
            frame = pd.read_parquet(path)
            validate_monthly_cache(frame, symbol, month_key, interval)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "bucket_start"]).reset_index(drop=True)


def load_backfill_state(output_dir: str | Path) -> dict:
    path = Path(output_dir) / "backfill_state.json"
    if not path.is_file():
        return {"status": "not_started", "completed_jobs": 0, "total_jobs": 0}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _expected_bucket_rows(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    interval: int,
    symbol_count: int,
) -> int:
    starts = pd.date_range(
        start=start_ts.floor(f"{interval}min"),
        end=end_ts.floor(f"{interval}min"),
        freq=f"{interval}min",
        tz="UTC",
    )
    return int(len(starts) * int(symbol_count))


def _monthly_archive_path(data_cfg: dict, symbol: str, month_key: str) -> Path:
    trade_kind = str(data_cfg.get("trade_kind", "aggTrades"))
    return (
        Path(data_cfg.get("local_data_dir", "data/tick_archive"))
        / symbol
        / trade_kind
        / "monthly"
        / f"{symbol}-{trade_kind}-{month_key}.zip"
    )


def _build_state(
    status: str,
    completed_jobs: int,
    total_jobs: int,
    last_record: dict | None,
) -> dict:
    return {
        "status": status,
        "completed_jobs": int(completed_jobs),
        "total_jobs": int(total_jobs),
        "progress_share": completed_jobs / total_jobs if total_jobs else 0.0,
        "last_completed_symbol": last_record.get("symbol") if last_record else None,
        "last_completed_month": last_record.get("month") if last_record else None,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write_progress(output_dir: Path, records: list[dict], state: dict) -> None:
    progress_path = output_dir / "backfill_progress.csv"
    progress_tmp = progress_path.with_suffix(".csv.tmp")
    pd.DataFrame(records).to_csv(progress_tmp, index=False)
    progress_tmp.replace(progress_path)
    state_path = output_dir / "backfill_state.json"
    state_tmp = state_path.with_suffix(".json.tmp")
    with state_tmp.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)
    state_tmp.replace(state_path)


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_cache_manifest(cache_dir: Path, destination: Path) -> Path:
    entries = []
    for path in sorted(cache_dir.glob("*/*.parquet")):
        entries.append(
            {
                "path": str(path),
                "size": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    temporary = destination.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump({"files": entries}, handle, indent=2)
    temporary.replace(destination)
    return destination

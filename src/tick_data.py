from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pandas as pd


LOGGER = logging.getLogger(__name__)

RAW_TRADE_COLUMNS = [
    "trade_id",
    "price",
    "quantity",
    "quote_quantity",
    "trade_time",
    "is_buyer_maker",
    "is_best_match",
]

AGG_TRADE_COLUMNS = [
    "aggregate_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "trade_time",
    "is_buyer_maker",
    "is_best_match",
]


def resolve_tick_date_window(data_cfg: dict) -> dict:
    resolved = dict(data_cfg)

    end_value = resolved.get("end", "latest")
    if end_value in {None, "", "latest", "yesterday"}:
        end_ts = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
    elif end_value in {"safe_latest", "archive_safe_latest"}:
        # Binance public daily archives can lag by roughly a day, especially earlier in the UTC day.
        end_ts = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=2)
    else:
        end_ts = pd.Timestamp(end_value)
        end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
        end_ts = end_ts.normalize()

    if resolved.get("start") not in {None, ""}:
        start_ts = pd.Timestamp(resolved["start"])
        start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
        start_ts = start_ts.normalize()
    elif resolved.get("lookback_days") is not None:
        lookback_days = int(resolved["lookback_days"])
        start_ts = end_ts - pd.Timedelta(days=max(lookback_days - 1, 0))
    else:
        raise ValueError("Tick data requires data.start or data.lookback_days")

    if end_ts < start_ts:
        raise ValueError("Resolved tick date window is invalid: end must be on or after start")

    resolved["start"] = start_ts.strftime("%Y-%m-%d")
    resolved["end"] = end_ts.strftime("%Y-%m-%d")
    resolved["resolved_start_utc"] = str(start_ts)
    resolved["resolved_end_utc"] = str(end_ts)
    return resolved


def iter_tick_dates(start_date: str, end_date: str) -> list[pd.Timestamp]:
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")
    return list(pd.date_range(start=start_ts, end=end_ts, freq="1D", tz="UTC"))


def iter_tick_months(start_date: str, end_date: str) -> list[pd.Timestamp]:
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")
    start_month = pd.Timestamp(year=start_ts.year, month=start_ts.month, day=1, tz="UTC")
    end_month = pd.Timestamp(year=end_ts.year, month=end_ts.month, day=1, tz="UTC")
    return list(pd.date_range(start=start_month, end=end_month, freq="MS", tz="UTC"))


def load_tick_day(
    symbol: str,
    date: pd.Timestamp,
    data_cfg: dict,
) -> tuple[pd.DataFrame, dict]:
    trade_kind = str(data_cfg.get("trade_kind", "trades"))
    archive_path, source_used = ensure_tick_archive(symbol=symbol, date=date, data_cfg=data_cfg)
    frame = _read_tick_archive(archive_path=archive_path, trade_kind=trade_kind)
    summary = {
        "symbol": symbol,
        "trade_kind": trade_kind,
        "date": date.strftime("%Y-%m-%d"),
        "archive_path": str(archive_path),
        "source_used": source_used,
        "rows_loaded": int(len(frame)),
        "archive_size_bytes": int(archive_path.stat().st_size),
    }
    return frame, summary


def load_tick_month(
    symbol: str,
    month_start: pd.Timestamp,
    data_cfg: dict,
) -> tuple[Path, str, dict]:
    trade_kind = str(data_cfg.get("trade_kind", "trades"))
    archive_path, source_used = ensure_tick_month_archive(symbol=symbol, month_start=month_start, data_cfg=data_cfg)
    summary = {
        "symbol": symbol,
        "trade_kind": trade_kind,
        "month": month_start.strftime("%Y-%m"),
        "archive_path": str(archive_path),
        "source_used": source_used,
        "archive_size_bytes": int(archive_path.stat().st_size),
    }
    return archive_path, source_used, summary


def ensure_tick_archive(
    symbol: str,
    date: pd.Timestamp,
    data_cfg: dict,
) -> tuple[Path, str]:
    trade_kind = str(data_cfg.get("trade_kind", "trades"))
    local_dir = Path(data_cfg.get("local_data_dir", "data/tick_pilot")) / symbol / trade_kind
    local_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"{symbol}-{trade_kind}-{date.strftime('%Y-%m-%d')}.zip"
    archive_path = local_dir / archive_name

    if archive_path.exists() and not data_cfg.get("overwrite_existing", False):
        return archive_path, "cache"

    url = build_tick_archive_url(symbol=symbol, date=date, trade_kind=trade_kind)
    LOGGER.info("Downloading tick archive for %s on %s from %s.", symbol, date.strftime("%Y-%m-%d"), url)
    try:
        with urlopen(url) as response, archive_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except HTTPError as exc:
        if archive_path.exists():
            archive_path.unlink(missing_ok=True)
        raise FileNotFoundError(f"Tick archive is not available: {url}") from exc

    return archive_path, "download"


def ensure_tick_month_archive(
    symbol: str,
    month_start: pd.Timestamp,
    data_cfg: dict,
) -> tuple[Path, str]:
    trade_kind = str(data_cfg.get("trade_kind", "trades"))
    local_dir = Path(data_cfg.get("local_data_dir", "data/tick_pilot")) / symbol / trade_kind / "monthly"
    local_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"{symbol}-{trade_kind}-{month_start.strftime('%Y-%m')}.zip"
    archive_path = local_dir / archive_name

    if archive_path.exists() and not data_cfg.get("overwrite_existing", False):
        return archive_path, "cache"

    url = build_tick_month_archive_url(symbol=symbol, month_start=month_start, trade_kind=trade_kind)
    LOGGER.info("Downloading monthly tick archive for %s on %s from %s.", symbol, month_start.strftime("%Y-%m"), url)
    try:
        with urlopen(url) as response, archive_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except HTTPError as exc:
        if archive_path.exists():
            archive_path.unlink(missing_ok=True)
        raise FileNotFoundError(f"Monthly tick archive is not available: {url}") from exc

    return archive_path, "download"


def build_tick_archive_url(symbol: str, date: pd.Timestamp, trade_kind: str) -> str:
    return (
        "https://data.binance.vision/"
        f"data/spot/daily/{trade_kind}/{symbol}/{symbol}-{trade_kind}-{date.strftime('%Y-%m-%d')}.zip"
    )


def build_tick_month_archive_url(symbol: str, month_start: pd.Timestamp, trade_kind: str) -> str:
    return (
        "https://data.binance.vision/"
        f"data/spot/monthly/{trade_kind}/{symbol}/{symbol}-{trade_kind}-{month_start.strftime('%Y-%m')}.zip"
    )


def _read_tick_archive(archive_path: Path, trade_kind: str) -> pd.DataFrame:
    column_names = RAW_TRADE_COLUMNS if trade_kind == "trades" else AGG_TRADE_COLUMNS
    with zipfile.ZipFile(archive_path) as zf:
        member = _select_archive_member(zf)
        with zf.open(member) as handle:
            frame = pd.read_csv(
                handle,
                header=None,
                names=column_names,
                low_memory=False,
            )
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "price", "quantity", "quote_quantity", "is_buyer_maker", "is_best_match"])

    return normalize_tick_frame(frame)


def iter_tick_archive_chunks(
    archive_path: Path,
    trade_kind: str,
    chunksize: int,
):
    column_names = RAW_TRADE_COLUMNS if trade_kind == "trades" else AGG_TRADE_COLUMNS
    with zipfile.ZipFile(archive_path) as zf:
        member = _select_archive_member(zf)
        with zf.open(member) as handle:
            for chunk in pd.read_csv(
                handle,
                header=None,
                names=column_names,
                low_memory=False,
                chunksize=int(chunksize),
            ):
                normalized = normalize_tick_frame(chunk)
                if not normalized.empty:
                    yield normalized


def normalize_tick_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "price", "quantity", "quote_quantity", "is_buyer_maker", "is_best_match"])

    normalized = frame.copy()
    normalized["price"] = pd.to_numeric(normalized["price"], errors="coerce")
    normalized["quantity"] = pd.to_numeric(normalized["quantity"], errors="coerce")
    if "quote_quantity" in normalized.columns:
        normalized["quote_quantity"] = pd.to_numeric(normalized["quote_quantity"], errors="coerce")
    else:
        normalized["quote_quantity"] = normalized["price"] * normalized["quantity"]

    timestamp_values = pd.to_numeric(normalized["trade_time"], errors="coerce")
    timestamp_unit = "us" if timestamp_values.max() >= 1_000_000_000_000_000 else "ms"
    normalized["timestamp"] = pd.to_datetime(timestamp_values, unit=timestamp_unit, utc=True)
    normalized["is_buyer_maker"] = normalized["is_buyer_maker"].astype(str).str.lower().map({"true": True, "false": False})
    normalized["is_best_match"] = normalized["is_best_match"].astype(str).str.lower().map({"true": True, "false": False})

    normalized = normalized.dropna(subset=["timestamp", "price", "quantity"]).sort_values("timestamp")
    normalized = normalized.reset_index(drop=True)
    return normalized[["timestamp", "price", "quantity", "quote_quantity", "is_buyer_maker", "is_best_match"]]


def _select_archive_member(zf: zipfile.ZipFile) -> str:
    csv_members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
    if not csv_members:
        raise FileNotFoundError("No CSV file found inside tick archive")
    csv_members = sorted(csv_members, key=lambda name: (name.count("/"), len(name)))
    return csv_members[0]

from __future__ import annotations

import io
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import ccxt
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils import load_table_file, timeframe_to_minutes, timeframe_to_pandas_freq


LOGGER = logging.getLogger(__name__)
REQUESTS_SESSION = requests.Session()
REQUESTS_SESSION.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=5,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
    ),
)
REQUESTS_SESSION.headers.update({"User-Agent": "crypto-herding-research/1.0"})


def load_multi_asset_ohlcv(config: dict) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    data_cfg = config["data"]
    source = str(data_cfg.get("source", "binance")).lower()
    prefer_local_files = bool(data_cfg.get("prefer_local_files", True))
    fallback_to_mock = bool(data_cfg.get("fallback_to_mock", True))
    mock_frames: Dict[str, pd.DataFrame] | None = None

    symbol_frames: Dict[str, pd.DataFrame] = {}
    load_records = []

    for symbol in data_cfg["symbols"]:
        source_used = "unknown"
        source_path = ""

        local_path = _resolve_local_file_path(data_cfg, symbol) if prefer_local_files else None
        if local_path is not None:
            frame = load_local_ohlcv(local_path)
            if _frame_covers_requested_range(frame, data_cfg["start"], data_cfg["end"], data_cfg.get("timeframe", "1m")):
                source_used = "local"
                source_path = str(local_path)
            elif _frame_has_requested_end(frame, data_cfg["end"], data_cfg.get("timeframe", "1m")) and data_cfg.get("allow_partial_history", False):
                LOGGER.info(
                    "Local file for %s starts after the requested start, but allow_partial_history is enabled. Using the available local history.",
                    symbol,
                )
                source_used = "local_partial_history"
                source_path = str(local_path)
            elif source == "binance":
                LOGGER.info("Local cache for %s is partial. Refreshing missing range from Binance.", symbol)
                frame = _refresh_local_cache_from_binance(frame, symbol, data_cfg)
                _cache_downloaded_frame(frame, symbol, data_cfg)
                source_used = "binance_refresh"
                source_path = str(local_path)
            elif source == "local":
                LOGGER.warning(
                    "Local file for %s does not fully cover the requested range. Continuing with the available local data only.",
                    symbol,
                )
                source_used = "local_partial"
                source_path = str(local_path)
            else:
                source_used = "local_partial"
                source_path = str(local_path)
        else:
            if source == "mock":
                if mock_frames is None:
                    LOGGER.info("Using deterministic mock OHLCV data.")
                    mock_frames = generate_mock_multi_asset_ohlcv(data_cfg)
                frame = mock_frames[symbol]
                source_used = "mock"
            elif source == "binance":
                try:
                    LOGGER.info("Fetching OHLCV data from Binance for %s.", symbol)
                    frame = fetch_symbol_ohlcv_from_binance(symbol, data_cfg)
                    _cache_downloaded_frame(frame, symbol, data_cfg)
                    source_used = "binance"
                except Exception as exc:
                    if not fallback_to_mock:
                        raise
                    LOGGER.warning("Data load failed for %s (%s). Falling back to mock data.", symbol, exc)
                    if mock_frames is None:
                        mock_frames = generate_mock_multi_asset_ohlcv(data_cfg)
                    frame = mock_frames[symbol]
                    source_used = "mock"
            elif source == "local":
                raise FileNotFoundError(
                    f"No local file was found for {symbol}. Provide data.local_file_map or place a matching file under {data_cfg.get('local_data_dir', 'data')}."
                )
            else:
                raise ValueError(f"Unsupported data source: {source}")

        normalized = normalize_ohlcv_frame(frame, data_cfg["start"], data_cfg["end"])
        symbol_frames[symbol] = normalized
        load_records.append(
            {
                "symbol": symbol,
                "source_used": source_used,
                "source_path": source_path,
                "rows_loaded": int(len(normalized)),
                "start_timestamp": normalized.index.min(),
                "end_timestamp": normalized.index.max(),
            }
        )

    load_summary = pd.DataFrame(load_records)
    return symbol_frames, load_summary


def load_local_ohlcv(path: str | Path) -> pd.DataFrame:
    frame = load_table_file(path)
    return normalize_ohlcv_frame(frame)


def fetch_symbol_ohlcv_from_binance(symbol: str, data_cfg: dict) -> pd.DataFrame:
    start_ms = _parse_timestamp_to_ms(data_cfg["start"])
    end_ms = _parse_timestamp_to_ms(data_cfg["end"])
    timeframe = data_cfg.get("timeframe", "1m")
    exchange_symbol = _normalize_symbol_for_exchange(symbol)

    if end_ms <= start_ms:
        raise ValueError("data.end must be greater than data.start")

    if data_cfg.get("use_bulk_downloads", True) and timeframe == "1m":
        return fetch_symbol_ohlcv_via_bulk_download(
            symbol=symbol,
            data_cfg=data_cfg,
            start_ms=start_ms,
            end_ms=end_ms,
        )

    exchange = ccxt.binance({"enableRateLimit": True})
    limit = int(data_cfg.get("exchange_limit", 1000))
    return _fetch_symbol_ohlcv(
        exchange=exchange,
        symbol=exchange_symbol,
        timeframe=timeframe,
        start_ms=start_ms,
        end_ms=end_ms,
        limit=limit,
    )


def fetch_symbol_ohlcv_via_bulk_download(
    symbol: str,
    data_cfg: dict,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    timeframe = data_cfg.get("timeframe", "1m")
    if timeframe != "1m":
        raise ValueError("Bulk download path currently supports only 1m timeframe")

    symbol_code = _normalize_symbol_for_filename(symbol)
    start_ts = pd.to_datetime(start_ms, unit="ms", utc=True)
    end_ts = pd.to_datetime(end_ms - 1, unit="ms", utc=True)
    month_starts = pd.date_range(
        start=pd.Timestamp(year=start_ts.year, month=start_ts.month, day=1, tz="UTC"),
        end=pd.Timestamp(year=end_ts.year, month=end_ts.month, day=1, tz="UTC"),
        freq="MS",
    )

    monthly_frames = []
    last_monthly_timestamp = None
    for idx, month_start in enumerate(month_starts, start=1):
        year_month = month_start.strftime("%Y-%m")
        LOGGER.info("Bulk downloading %s %s (%d/%d).", symbol, year_month, idx, len(month_starts))
        monthly_frame = _download_binance_monthly_klines(symbol_code, timeframe, year_month)
        if monthly_frame is None:
            continue
        monthly_frames.append(monthly_frame)
        last_monthly_timestamp = monthly_frame.index.max()

    combined = pd.concat(monthly_frames, axis=0) if monthly_frames else pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if not combined.empty:
        combined = normalize_ohlcv_frame(combined)

    requested_end_ts = pd.to_datetime(end_ms, unit="ms", utc=True)
    if combined.empty or combined.index.max() + pd.Timedelta(minutes=1) < requested_end_ts:
        exchange = ccxt.binance({"enableRateLimit": True})
        limit = int(data_cfg.get("exchange_limit", 1000))
        api_start_ms = start_ms
        if last_monthly_timestamp is not None:
            api_start_ms = max(start_ms, int((last_monthly_timestamp + pd.Timedelta(minutes=1)).timestamp() * 1000))
        if api_start_ms < end_ms:
            try:
                tail_frame = _fetch_symbol_ohlcv(
                    exchange=exchange,
                    symbol=_normalize_symbol_for_exchange(symbol),
                    timeframe=timeframe,
                    start_ms=api_start_ms,
                    end_ms=end_ms,
                    limit=limit,
                )
            except ValueError as exc:
                if "No OHLCV rows returned" not in str(exc):
                    raise
                LOGGER.info(
                    "No Binance API tail rows returned for %s after %s. Using monthly bulk history only.",
                    symbol,
                    pd.to_datetime(api_start_ms, unit="ms", utc=True),
                )
            else:
                combined = pd.concat([combined, tail_frame], axis=0) if not combined.empty else tail_frame

    return normalize_ohlcv_frame(combined, pd.to_datetime(start_ms, unit="ms", utc=True), pd.to_datetime(end_ms, unit="ms", utc=True))


def _fetch_symbol_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    limit: int,
) -> pd.DataFrame:
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    if timeframe_ms <= 0:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    all_rows = []
    since = start_ms

    while since < end_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break

        valid_rows = [row for row in batch if row[0] < end_ms]
        all_rows.extend(valid_rows)

        last_timestamp = batch[-1][0]
        next_since = last_timestamp + timeframe_ms
        if next_since <= since:
            break
        since = next_since

        if len(batch) < limit:
            break

    if not all_rows:
        raise ValueError(f"No OHLCV rows returned for {symbol}")

    frame = pd.DataFrame(
        all_rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    return normalize_ohlcv_frame(frame, pd.to_datetime(start_ms, unit="ms", utc=True), pd.to_datetime(end_ms, unit="ms", utc=True))


def normalize_ohlcv_frame(
    frame: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [_sanitize_column_name(column) for column in normalized.columns]

    if "timestamp" in normalized.columns:
        timestamp = _parse_timestamp_series(normalized["timestamp"])
        normalized = normalized.drop(columns=["timestamp"])
        normalized.index = timestamp
    elif not isinstance(normalized.index, pd.DatetimeIndex):
        for candidate in ["datetime", "date", "time", "time_open", "timeclose", "time_close"]:
            if candidate in normalized.columns:
                timestamp = _parse_timestamp_series(normalized[candidate])
                normalized = normalized.drop(columns=[candidate])
                normalized.index = timestamp
                break
        else:
            raise ValueError("OHLCV frame must contain a timestamp column or DatetimeIndex")
    else:
        if normalized.index.tz is None:
            normalized.index = normalized.index.tz_localize("UTC")
        else:
            normalized.index = normalized.index.tz_convert("UTC")

    required_columns = ["open", "high", "low", "close", "volume"]
    for column in required_columns:
        if column not in normalized.columns:
            if column == "volume":
                normalized[column] = 0.0
            else:
                raise ValueError(f"OHLCV frame is missing required column: {column}")

    extra_columns = [column for column in ["market_cap", "marketcap"] if column in normalized.columns]
    normalized = normalized[required_columns + extra_columns].sort_index()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    normalized = normalized.apply(pd.to_numeric, errors="coerce")
    if "marketcap" in normalized.columns and "market_cap" not in normalized.columns:
        normalized = normalized.rename(columns={"marketcap": "market_cap"})
    normalized.index.name = "timestamp"

    if start is not None:
        start_ts = pd.Timestamp(start)
        start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
        normalized = normalized[normalized.index >= start_ts]
    if end is not None:
        end_ts = pd.Timestamp(end)
        end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
        normalized = normalized[normalized.index < end_ts]

    normalized = normalized.dropna(subset=["open", "high", "low", "close"])
    if normalized.empty:
        raise ValueError("Normalized OHLCV frame is empty after filtering")

    return normalized


def generate_mock_multi_asset_ohlcv(data_cfg: dict) -> Dict[str, pd.DataFrame]:
    symbols = data_cfg["symbols"]
    start = pd.Timestamp(data_cfg["start"], tz="UTC")
    end = pd.Timestamp(data_cfg["end"], tz="UTC")
    seed = int(data_cfg.get("seed", 42))
    timeframe = data_cfg.get("timeframe", "1m")
    freq = timeframe_to_pandas_freq(timeframe)

    index = pd.date_range(start=start, end=end, freq=freq, inclusive="left")
    if len(index) < 10:
        raise ValueError("Mock data range is too short")

    rng = np.random.default_rng(seed)
    n_obs = len(index)

    market_factor = rng.normal(loc=0.0, scale=0.0012, size=n_obs)
    regime = np.sin(np.linspace(0.0, 8.0 * np.pi, n_obs)) * 0.00035
    market_factor = market_factor + regime

    base_prices = {
        "BTC/USDT": 43000.0,
        "ETH/USDT": 2300.0,
        "XRP/USDT": 0.55,
        "BNB/USDT": 310.0,
        "SOL/USDT": 95.0,
    }

    symbol_frames: Dict[str, pd.DataFrame] = {}
    for idx, symbol in enumerate(symbols):
        beta = 0.85 + 0.08 * idx
        idiosyncratic_scale = 0.00055 + 0.00008 * idx
        noise = rng.normal(loc=0.0, scale=idiosyncratic_scale, size=n_obs)
        drift = 0.00001 * np.cos(np.linspace(0.0, 5.0 * np.pi, n_obs) + idx)
        log_returns = beta * market_factor + noise + drift

        close = base_prices.get(symbol, 100.0 + 10.0 * idx) * np.exp(np.cumsum(log_returns))
        open_ = np.roll(close, 1)
        open_[0] = close[0] / np.exp(log_returns[0])

        intrabar_noise = np.abs(rng.normal(loc=0.0, scale=0.0015, size=n_obs))
        high = np.maximum(open_, close) * (1.0 + intrabar_noise)
        low = np.minimum(open_, close) * (1.0 - intrabar_noise)
        volume = rng.lognormal(mean=8.0 + 0.05 * idx, sigma=0.35, size=n_obs)

        frame = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=index,
        )
        frame.index.name = "timestamp"
        symbol_frames[symbol] = frame

    return symbol_frames


def _resolve_local_file_path(data_cfg: dict, symbol: str) -> Path | None:
    local_file_map = data_cfg.get("local_file_map", {})
    if symbol in local_file_map:
        candidate = Path(local_file_map[symbol])
        if candidate.exists():
            return candidate

    local_dir = Path(data_cfg.get("local_data_dir", "data"))
    timeframe = data_cfg.get("timeframe", "1m")
    normalized_symbol = _normalize_symbol_for_filename(symbol)
    normalized_timeframe = timeframe.replace("/", "").replace("-", "").replace("_", "").upper()
    candidate_stems = [
        f"{normalized_symbol}_{timeframe}",
        f"{symbol.replace('/', '_')}_{timeframe}",
        f"{symbol.replace('/', '')}_{timeframe}",
    ]

    for stem in candidate_stems:
        for suffix in [".parquet", ".csv"]:
            candidate = local_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate

    for suffix in ["*.parquet", "*.csv"]:
        for candidate in local_dir.rglob(suffix):
            compact_stem = candidate.stem.replace("_", "").replace("-", "").upper()
            if normalized_symbol in compact_stem and normalized_timeframe in compact_stem:
                return candidate

    return None


def _normalize_symbol_for_filename(symbol: str) -> str:
    return symbol.replace("/", "").replace("-", "").upper()


def _normalize_symbol_for_exchange(symbol: str) -> str:
    if "/" in symbol:
        return symbol

    normalized = symbol.replace("-", "").replace("_", "").upper()
    quote_candidates = ["USDT", "BUSD", "USDC", "BTC", "ETH", "BNB", "USD"]
    for quote in quote_candidates:
        if normalized.endswith(quote) and len(normalized) > len(quote):
            return f"{normalized[:-len(quote)]}/{quote}"

    raise ValueError(f"Unable to infer Binance symbol format for {symbol!r}")


def _parse_timestamp_to_ms(value: str) -> int:
    timestamp = pd.Timestamp(value, tz="UTC")
    return int(timestamp.timestamp() * 1000)


def probe_symbol_history_availability(symbol: str, timeframe: str = "1m") -> dict:
    symbol_code = _normalize_symbol_for_filename(symbol)
    exchange_symbol = _normalize_symbol_for_exchange(symbol)
    start_month = _probe_earliest_available_month(symbol_code, timeframe)

    exchange = ccxt.binance({"enableRateLimit": True})
    latest_rows = exchange.fetch_ohlcv(exchange_symbol, timeframe=timeframe, limit=2)
    latest_timestamp = pd.to_datetime(latest_rows[-1][0], unit="ms", utc=True) if latest_rows else pd.NaT

    first_timestamp = pd.NaT
    if start_month is not None:
        monthly_frame = _download_binance_monthly_klines(symbol_code, timeframe, start_month)
        if monthly_frame is not None and not monthly_frame.empty:
            first_timestamp = monthly_frame.index.min()

    return {
        "symbol": symbol,
        "earliest_month": start_month,
        "earliest_timestamp": first_timestamp,
        "latest_timestamp": latest_timestamp,
    }


def _parse_timestamp_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        max_abs = numeric.abs().max()
        if max_abs > 1e14:
            return pd.to_datetime(numeric, unit="ns", utc=True)
        if max_abs > 1e11:
            return pd.to_datetime(numeric, unit="ms", utc=True)
        if max_abs > 1e8:
            return pd.to_datetime(numeric, unit="s", utc=True)
    return pd.to_datetime(values, utc=True)


def _sanitize_column_name(column: object) -> str:
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
    )


def _cache_downloaded_frame(frame: pd.DataFrame, symbol: str, data_cfg: dict) -> None:
    if not data_cfg.get("cache_downloads", True):
        return

    local_dir = Path(data_cfg.get("local_data_dir", "data"))
    local_dir.mkdir(parents=True, exist_ok=True)

    timeframe = data_cfg.get("timeframe", "1m")
    cache_format = str(data_cfg.get("cache_format", "parquet")).lower()
    filename = f"{_normalize_symbol_for_filename(symbol)}_{timeframe}"
    cache_frame = frame.reset_index()

    if cache_format == "csv":
        cache_path = local_dir / f"{filename}.csv"
        cache_frame.to_csv(cache_path, index=False)
        return

    cache_path = local_dir / f"{filename}.parquet"
    cache_frame.to_parquet(cache_path, index=False)


def _refresh_local_cache_from_binance(
    existing_frame: pd.DataFrame,
    symbol: str,
    data_cfg: dict,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(data_cfg["start"])
    start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
    end_ts = pd.Timestamp(data_cfg["end"])
    end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
    timeframe = data_cfg.get("timeframe", "1m")
    tolerance = pd.Timedelta(minutes=timeframe_to_minutes(timeframe))

    has_requested_start = existing_frame.index.min() <= start_ts
    has_requested_end = existing_frame.index.max() >= (end_ts - tolerance)

    if not has_requested_start:
        LOGGER.info("Local cache for %s is missing early history. Refreshing full requested range.", symbol)
        return fetch_symbol_ohlcv_from_binance(symbol, data_cfg)

    if has_requested_end:
        return normalize_ohlcv_frame(existing_frame, start_ts, end_ts)

    refresh_cfg = dict(data_cfg)
    refresh_cfg["start"] = (existing_frame.index.max() + tolerance).isoformat()
    refresh_cfg["end"] = end_ts.isoformat()

    try:
        tail_frame = fetch_symbol_ohlcv_from_binance(symbol, refresh_cfg)
    except ValueError as exc:
        if "No OHLCV rows returned" not in str(exc) and "Normalized OHLCV frame is empty" not in str(exc):
            raise
        LOGGER.info(
            "No additional Binance rows were available for %s after %s. Keeping current local cache.",
            symbol,
            existing_frame.index.max(),
        )
        return normalize_ohlcv_frame(existing_frame, start_ts, end_ts)

    combined = pd.concat([existing_frame, tail_frame], axis=0)
    return normalize_ohlcv_frame(combined, start_ts, end_ts)


def _frame_covers_requested_range(
    frame: pd.DataFrame,
    start: str,
    end: str,
    timeframe: str,
) -> bool:
    if frame.empty:
        return False
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    tolerance = pd.Timedelta(minutes=timeframe_to_minutes(timeframe))
    return frame.index.min() <= start_ts and frame.index.max() >= (end_ts - tolerance)


def _frame_has_requested_end(frame: pd.DataFrame, end: str, timeframe: str) -> bool:
    if frame.empty:
        return False
    end_ts = pd.Timestamp(end, tz="UTC")
    tolerance = pd.Timedelta(minutes=timeframe_to_minutes(timeframe))
    return frame.index.max() >= (end_ts - tolerance)


def _download_binance_monthly_klines(symbol_code: str, timeframe: str, year_month: str) -> pd.DataFrame | None:
    url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol_code}/{timeframe}/{symbol_code}-{timeframe}-{year_month}.zip"
    response = _request_with_retries("GET", url, timeout=60)
    if response.status_code == 404:
        return None
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_members = [name for name in archive.namelist() if name.endswith(".csv")]
        if not csv_members:
            raise ValueError(f"No CSV file found inside archive: {url}")
        with archive.open(csv_members[0]) as handle:
            frame = pd.read_csv(handle, header=None)

    frame = frame.iloc[:, :6]
    frame.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    return normalize_ohlcv_frame(frame)


def _probe_earliest_available_month(symbol_code: str, timeframe: str) -> str | None:
    current = datetime.now(timezone.utc)
    year_months = []
    year = 2016
    month = 1
    while (year < current.year) or (year == current.year and month <= current.month):
        year_months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1

    for year_month in year_months:
        url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol_code}/{timeframe}/{symbol_code}-{timeframe}-{year_month}.zip"
        response = _request_with_retries("HEAD", url, timeout=20)
        if response.status_code == 200:
            return year_month
    return None


def _request_with_retries(method: str, url: str, timeout: int) -> requests.Response:
    return REQUESTS_SESSION.request(method=method, url=url, timeout=timeout)

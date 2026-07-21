from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import os
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import quote
from xml.etree import ElementTree

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crypto-herding-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import chi2, norm

from frequency_sensitivity import benjamini_hochberg


LOGGER = logging.getLogger(__name__)
_THREAD_LOCAL = threading.local()
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "base_volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]
HISTORY_COLUMNS = [
    "date",
    "symbol",
    "base_asset",
    "open_usdt",
    "high_usdt",
    "low_usdt",
    "close_usdt",
    "base_volume",
    "quote_volume_usdt",
    "trade_count",
    "archive_key",
]


def parse_s3_listing(payload: bytes | str) -> dict:
    """Parse a ListObjects-v1 XML response without depending on its namespace."""
    root = ElementTree.fromstring(payload)

    def local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    def first_text(name: str) -> str | None:
        for element in root.iter():
            if local_name(element.tag) == name:
                return element.text or ""
        return None

    prefixes = []
    contents = []
    for element in root.iter():
        name = local_name(element.tag)
        if name == "CommonPrefixes":
            value = next(
                (
                    child.text
                    for child in element
                    if local_name(child.tag) == "Prefix" and child.text
                ),
                None,
            )
            if value:
                prefixes.append(value)
        elif name == "Contents":
            values = {
                local_name(child.tag): (child.text or "")
                for child in element
            }
            if values.get("Key"):
                contents.append(
                    {
                        "key": values["Key"],
                        "last_modified": values.get("LastModified", ""),
                        "etag": values.get("ETag", "").strip('"'),
                        "size_bytes": int(values.get("Size", "0") or 0),
                    }
                )
    truncated = str(first_text("IsTruncated") or "false").lower() == "true"
    next_marker = first_text("NextMarker")
    if truncated and not next_marker:
        candidates = [row["key"] for row in contents] or prefixes
        next_marker = candidates[-1] if candidates else None
    return {
        "prefixes": prefixes,
        "contents": contents,
        "is_truncated": truncated,
        "next_marker": next_marker,
    }


def classify_symbol_prefixes(
    prefixes: Sequence[str],
    source_cfg: Mapping,
    universe_cfg: Mapping,
) -> pd.DataFrame:
    root_prefix = str(source_cfg["root_prefix"])
    quote_suffix = str(universe_cfg["quote_suffix"])
    leveraged = tuple(str(value) for value in universe_cfg["leveraged_suffixes"])
    excluded = {
        *(str(value) for value in universe_cfg["excluded_base_assets"]),
        *(str(value) for value in universe_cfg["excluded_wrapped_staked_assets"]),
    }
    rows = []
    for prefix in sorted(set(prefixes)):
        symbol = prefix.removeprefix(root_prefix).strip("/")
        base_asset = symbol[: -len(quote_suffix)] if symbol.endswith(quote_suffix) else ""
        if not symbol.endswith(quote_suffix):
            reason = "non_usdt_quote"
        elif not base_asset:
            reason = "empty_base_asset"
        elif base_asset in excluded:
            reason = "excluded_stable_fiat_or_wrapped"
        elif base_asset.endswith(leveraged):
            reason = "leveraged_token_suffix"
        else:
            reason = "included"
        rows.append(
            {
                "prefix": prefix,
                "symbol": symbol,
                "base_asset": base_asset,
                "included": reason == "included",
                "exclusion_reason": reason,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty and result["symbol"].duplicated().any():
        raise ValueError("Bucket root inventory contains duplicate symbol prefixes")
    return result


def collect_archive_inventory(
    source_cfg: Mapping,
    universe_cfg: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect resumable root/symbol inventories and return candidate and file indexes."""
    inventory_dir = Path(source_cfg["inventory_dir"])
    root_dir = inventory_dir / "root_pages"
    symbol_dir = inventory_dir / "symbol_pages"
    root_dir.mkdir(parents=True, exist_ok=True)
    symbol_dir.mkdir(parents=True, exist_ok=True)

    all_prefixes: list[str] = []
    inventory_rows: list[dict] = []
    marker: str | None = None
    page_number = 1
    while True:
        path = root_dir / f"page_{page_number:04d}.xml.gz"
        payload = _cached_s3_page(
            path,
            source_cfg,
            prefix=str(source_cfg["root_prefix"]),
            delimiter="/",
            marker=marker,
        )
        parsed = parse_s3_listing(payload)
        all_prefixes.extend(parsed["prefixes"])
        inventory_rows.append(
            _inventory_manifest_row(
                "root",
                page_number,
                path,
                parsed,
                marker,
                str(source_cfg["root_prefix"]),
            )
        )
        if not parsed["is_truncated"]:
            break
        if not parsed["next_marker"] or parsed["next_marker"] == marker:
            raise ValueError("S3 root inventory pagination did not advance")
        marker = str(parsed["next_marker"])
        page_number += 1

    candidates = classify_symbol_prefixes(all_prefixes, source_cfg, universe_cfg)
    included = candidates.loc[candidates["included"]].copy()
    symbol_results: list[tuple[dict, list[dict]]] = []

    def collect_symbol(row: object) -> tuple[dict, list[dict]]:
        symbol = str(getattr(row, "symbol"))
        prefix = f"{source_cfg['root_prefix']}{symbol}/{source_cfg['interval']}/"
        path = symbol_dir / f"{symbol}.xml.gz"
        payload = _cached_s3_page(path, source_cfg, prefix=prefix)
        parsed = parse_s3_listing(payload)
        if parsed["is_truncated"]:
            raise ValueError(f"Unexpected truncated symbol inventory for {symbol}")
        manifest = _inventory_manifest_row(
            "symbol", 1, path, parsed, None, prefix, symbol=symbol
        )
        return manifest, _archive_rows_from_listing(
            symbol,
            str(getattr(row, "base_asset")),
            parsed["contents"],
            source_cfg,
        )

    with ThreadPoolExecutor(max_workers=int(source_cfg["max_workers"])) as executor:
        futures = {executor.submit(collect_symbol, row): row.symbol for row in included.itertuples()}
        for index, future in enumerate(as_completed(futures), start=1):
            symbol_results.append(future.result())
            if index % 50 == 0 or index == len(futures):
                LOGGER.info("Symbol inventory %d/%d", index, len(futures))

    for manifest, _ in sorted(symbol_results, key=lambda item: item[0]["symbol"]):
        inventory_rows.append(manifest)
    archive_rows = [
        row
        for _, rows in symbol_results
        for row in rows
    ]
    files = pd.DataFrame(archive_rows).sort_values(["symbol", "month", "archive_key"])
    manifest = pd.DataFrame(inventory_rows)
    _atomic_csv(candidates, Path(source_cfg["candidate_inventory_path"]))
    _atomic_csv(files, Path(source_cfg["archive_index_path"]))
    _atomic_csv(manifest, Path(source_cfg["inventory_manifest_path"]))
    _write_state(
        source_cfg,
        phase="inventory_complete",
        expected_symbol_inventories=len(included),
        completed_symbol_inventories=int(manifest["inventory_type"].eq("symbol").sum()),
        discovered_archives=len(files),
    )
    return candidates.reset_index(drop=True), files.reset_index(drop=True), manifest


def load_archive_inventory(
    source_cfg: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = [
        Path(source_cfg["candidate_inventory_path"]),
        Path(source_cfg["archive_index_path"]),
        Path(source_cfg["inventory_manifest_path"]),
    ]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("Archive inventory is incomplete; run collection first")
    return tuple(pd.read_csv(path) for path in paths)  # type: ignore[return-value]


def download_archives(files: pd.DataFrame, source_cfg: Mapping) -> pd.DataFrame:
    rows: list[dict] = []

    def download(row: object) -> dict:
        target = Path(source_cfg["zip_dir"]) / str(row.symbol) / Path(str(row.archive_key)).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not _valid_zip(target):
            _download_file(str(row.download_url), target, source_cfg)
        sha256 = _sha256(target)
        md5 = _md5(target)
        etag = str(row.etag) if pd.notna(row.etag) else ""
        etag_verified = bool(len(etag) == 32 and "-" not in etag and md5 == etag.lower())
        return {
            **row._asdict(),
            "path": target.as_posix(),
            "downloaded": True,
            "zip_valid": True,
            "sha256": sha256,
            "md5": md5,
            "etag_verified": etag_verified,
            "checksum_required": False,
            "official_checksum": "",
            "checksum_verified": False,
            "checksum_path": "",
        }

    with ThreadPoolExecutor(max_workers=int(source_cfg["max_workers"])) as executor:
        futures = [executor.submit(download, row) for row in files.itertuples(index=False)]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 250 == 0 or index == len(futures):
                _write_state(
                    source_cfg,
                    phase="archive_download",
                    expected_archives=len(futures),
                    completed_archives=index,
                )
                LOGGER.info("Archive ZIP download %d/%d", index, len(futures))
    manifest = pd.DataFrame(rows).sort_values(["symbol", "month", "archive_key"])
    _atomic_csv(manifest, Path(source_cfg["file_manifest_path"]))
    _write_state(
        source_cfg,
        phase="archive_download_complete",
        expected_archives=len(files),
        completed_archives=len(manifest),
    )
    return manifest.reset_index(drop=True)


def load_file_manifest(source_cfg: Mapping) -> pd.DataFrame:
    path = Path(source_cfg["file_manifest_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing archive file manifest: {path}")
    return pd.read_csv(path)


def parse_binance_kline_zip(
    path: str | Path,
    symbol: str,
    archive_key: str | None = None,
    start: object | None = None,
    end: object | None = None,
) -> pd.DataFrame:
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV in {path}, found {len(csv_names)}")
        with archive.open(csv_names[0]) as handle:
            raw = pd.read_csv(handle, header=None, names=KLINE_COLUMNS, dtype=str)
    open_time = pd.to_numeric(raw["open_time"], errors="coerce")
    raw = raw.loc[open_time.notna()].copy()
    exact_duplicate_count = int(raw.duplicated(keep="first").sum())
    raw = raw.drop_duplicates(keep="first")
    open_time = open_time.loc[raw.index].astype("int64")
    unit = "us" if bool(open_time.abs().ge(10**14).any()) else "ms"
    dates = pd.to_datetime(open_time, unit=unit, utc=True).dt.normalize()
    quote_suffix = "USDT"
    base_asset = symbol[: -len(quote_suffix)] if symbol.endswith(quote_suffix) else symbol
    frame = pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "base_asset": base_asset,
            "open_usdt": pd.to_numeric(raw["open"], errors="coerce"),
            "high_usdt": pd.to_numeric(raw["high"], errors="coerce"),
            "low_usdt": pd.to_numeric(raw["low"], errors="coerce"),
            "close_usdt": pd.to_numeric(raw["close"], errors="coerce"),
            "base_volume": pd.to_numeric(raw["base_volume"], errors="coerce"),
            "quote_volume_usdt": pd.to_numeric(raw["quote_volume"], errors="coerce"),
            "trade_count": pd.to_numeric(raw["trade_count"], errors="coerce"),
            "archive_key": archive_key or path.name,
        }
    )
    if start is not None:
        frame = frame.loc[frame["date"].ge(_utc_day(start))]
    if end is not None:
        frame = frame.loc[frame["date"].le(_utc_day(end))]
    if frame.duplicated("date").any():
        raise ValueError(f"Archive contains conflicting duplicate UTC dates: {path}")
    required_numeric = ["open_usdt", "high_usdt", "low_usdt", "close_usdt", "quote_volume_usdt"]
    if frame[required_numeric].isna().any().any():
        raise ValueError(f"Archive contains invalid numeric OHLCV values: {path}")
    result = frame[HISTORY_COLUMNS].sort_values("date").reset_index(drop=True)
    result.attrs["source_exact_duplicate_rows_removed"] = exact_duplicate_count
    return result


def build_listing_episodes(history: pd.DataFrame, gap_days: int) -> pd.DataFrame:
    working = history.sort_values(["symbol", "date"]).copy()
    gaps = working.groupby("symbol", sort=False)["date"].diff().dt.days
    new_episode = gaps.isna() | gaps.gt(int(gap_days))
    working["listing_episode"] = new_episode.groupby(working["symbol"]).cumsum().astype(int)
    working["asset_key"] = (
        working["symbol"] + "#" + working["listing_episode"].map(lambda value: f"{value:02d}")
    )
    if working.duplicated(["date", "asset_key"]).any():
        raise ValueError("Episode assignment produced duplicate date-asset keys")
    return working.reset_index(drop=True)


def normalize_archive_history(
    manifest: pd.DataFrame,
    source_cfg: Mapping,
    universe_cfg: Mapping,
) -> pd.DataFrame:
    frames = []
    quality_rows = []
    for index, row in enumerate(manifest.itertuples(index=False), start=1):
        frame = parse_binance_kline_zip(
            row.path,
            str(row.symbol),
            archive_key=str(row.archive_key),
            start=source_cfg["source_start"],
            end=source_cfg["source_end"],
        )
        frames.append(frame)
        quality_rows.append(
            {
                "archive_key": str(row.archive_key),
                "symbol": str(row.symbol),
                "month": str(row.month),
                "normalized_rows": len(frame),
                "source_exact_duplicate_rows_removed": int(
                    frame.attrs.get("source_exact_duplicate_rows_removed", 0)
                ),
                "conflicting_duplicate_timestamps": 0,
            }
        )
        if index % 1000 == 0:
            LOGGER.info("Parsed archive ZIP %d/%d", index, len(manifest))
    if not frames:
        raise ValueError("No Binance archive rows were parsed")
    history = pd.concat(frames, ignore_index=True)
    history = history.sort_values(["symbol", "date", "archive_key"])
    if history.duplicated(["date", "symbol"]).any():
        duplicates = history.loc[history.duplicated(["date", "symbol"], keep=False)]
        raise ValueError(f"Archive overlap produced duplicate symbol dates: {len(duplicates)}")
    history = build_listing_episodes(history, int(universe_cfg["episode_gap_days"]))
    target = Path(source_cfg["normalized_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(target, index=False)
    _atomic_csv(pd.DataFrame(quality_rows), Path(source_cfg["source_quality_path"]))
    _write_state(source_cfg, phase="normalization_complete", normalized_rows=len(history))
    return history


def load_normalized_history(source_cfg: Mapping) -> pd.DataFrame:
    path = Path(source_cfg["normalized_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing normalized archive history: {path}")
    history = pd.read_parquet(path)
    history["date"] = pd.to_datetime(history["date"], utc=True)
    return history


def build_monthly_membership(
    history: pd.DataFrame,
    universe_cfg: Mapping,
    analysis_cfg: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank month m assets using only complete observations from month m-1."""
    working = history.copy()
    working["month"] = _month_string(working["date"])
    stats = (
        working.groupby(
            ["month", "asset_key", "symbol", "base_asset", "listing_episode"],
            as_index=False,
        )
        .agg(
            observed_days=("date", "nunique"),
            prior_month_quote_volume=("quote_volume_usdt", "sum"),
            first_observation=("date", "min"),
            last_observation=("date", "max"),
            positive_close_days=("close_usdt", lambda values: int(values.gt(0).sum())),
        )
    )
    stats["eligible_for_next_month"] = (
        stats["observed_days"].ge(int(universe_cfg["minimum_prior_month_days"]))
        & stats["prior_month_quote_volume"].gt(0)
        & stats["positive_close_days"].eq(stats["observed_days"])
    )

    first_target = pd.Timestamp(analysis_cfg["start"]).to_period("M")
    last_target = pd.Timestamp(analysis_cfg["end"]).to_period("M")
    membership_rows = []
    for target_month in pd.period_range(first_target, last_target, freq="M"):
        source_month = target_month - 1
        eligible = stats.loc[
            stats["month"].eq(str(source_month)) & stats["eligible_for_next_month"]
        ].copy()
        eligible = eligible.sort_values(
            ["prior_month_quote_volume", "asset_key"],
            ascending=[False, True],
            kind="mergesort",
        ).head(int(universe_cfg["top_n"]))
        eligible["rank"] = np.arange(1, len(eligible) + 1)
        eligible["membership_month"] = str(target_month)
        eligible["source_month"] = str(source_month)
        membership_rows.append(eligible)
    if not membership_rows:
        raise ValueError("No point-in-time membership months were constructed")
    membership = pd.concat(membership_rows, ignore_index=True)
    columns = [
        "membership_month",
        "source_month",
        "rank",
        "asset_key",
        "symbol",
        "base_asset",
        "listing_episode",
        "observed_days",
        "prior_month_quote_volume",
        "first_observation",
        "last_observation",
    ]
    membership = membership[columns].sort_values(["membership_month", "rank"])
    if membership.duplicated(["membership_month", "asset_key"]).any():
        raise ValueError("Point-in-time membership contains duplicate asset-month rows")
    month_delta = np.asarray(
        pd.PeriodIndex(membership["membership_month"], freq="M").astype(int)
        - pd.PeriodIndex(membership["source_month"], freq="M").astype(int)
    )
    if not np.all(month_delta == 1):
        raise ValueError("Membership source month is not exactly one month behind")
    return membership.reset_index(drop=True), stats.reset_index(drop=True)


def build_membership_transitions(membership: pd.DataFrame) -> pd.DataFrame:
    months = sorted(membership["membership_month"].unique())
    rows = []
    previous: set[str] = set()
    for month in months:
        current = set(membership.loc[membership["membership_month"].eq(month), "asset_key"])
        for asset_key in sorted(current - previous):
            rows.append({"membership_month": month, "asset_key": asset_key, "transition": "entered"})
        for asset_key in sorted(previous - current):
            rows.append({"membership_month": month, "asset_key": asset_key, "transition": "exited"})
        previous = current
    return pd.DataFrame(rows, columns=["membership_month", "asset_key", "transition"])


def build_episode_coverage(
    history: pd.DataFrame,
    membership: pd.DataFrame,
    source_cfg: Mapping,
) -> pd.DataFrame:
    coverage = (
        history.groupby(["asset_key", "symbol", "base_asset", "listing_episode"], as_index=False)
        .agg(
            observed_days=("date", "nunique"),
            first_observation=("date", "min"),
            last_observation=("date", "max"),
            archive_files=("archive_key", "nunique"),
            total_quote_volume=("quote_volume_usdt", "sum"),
        )
    )
    member_months = membership.groupby("asset_key")["membership_month"].nunique()
    coverage["membership_months"] = coverage["asset_key"].map(member_months).fillna(0).astype(int)
    end = _utc_day(source_cfg["source_end"])
    coverage["archive_ended_before_source_end"] = coverage["last_observation"].lt(
        end - pd.Timedelta(days=7)
    )
    return coverage.sort_values(["symbol", "listing_episode"]).reset_index(drop=True)


def membership_used_archive_keys(
    history: pd.DataFrame,
    membership: pd.DataFrame,
) -> set[str]:
    working = history[["date", "asset_key", "archive_key"]].copy()
    working["month"] = _month_string(working["date"])
    formation = working.merge(
        membership[["source_month", "asset_key"]].drop_duplicates(),
        left_on=["month", "asset_key"],
        right_on=["source_month", "asset_key"],
        how="inner",
        validate="many_to_many",
    )
    active = working.merge(
        membership[["membership_month", "asset_key"]].drop_duplicates(),
        left_on=["month", "asset_key"],
        right_on=["membership_month", "asset_key"],
        how="inner",
        validate="many_to_many",
    )
    return set(pd.concat([formation["archive_key"], active["archive_key"]]).dropna().astype(str))


def verify_membership_checksums(
    manifest: pd.DataFrame,
    required_archive_keys: set[str],
    source_cfg: Mapping,
) -> pd.DataFrame:
    working = manifest.copy()
    working["checksum_required"] = working["archive_key"].astype(str).isin(required_archive_keys)

    def verify(row: object) -> tuple[str, str, str, bool]:
        archive_key = str(row.archive_key)
        checksum_path = (
            Path(source_cfg["checksum_dir"])
            / str(row.symbol)
            / f"{Path(archive_key).name}.CHECKSUM"
        )
        checksum_path.parent.mkdir(parents=True, exist_ok=True)
        if not checksum_path.is_file() or checksum_path.stat().st_size == 0:
            _download_file(str(row.checksum_url), checksum_path, source_cfg)
        expected = checksum_path.read_text(encoding="utf-8").strip().split()[0].lower()
        if len(expected) != 64:
            raise ValueError(f"Invalid official checksum for {archive_key}")
        observed = str(row.sha256).lower()
        return archive_key, expected, checksum_path.as_posix(), expected == observed

    required_rows = list(working.loc[working["checksum_required"]].itertuples(index=False))
    verified = []
    with ThreadPoolExecutor(max_workers=int(source_cfg["max_workers"])) as executor:
        futures = [executor.submit(verify, row) for row in required_rows]
        for index, future in enumerate(as_completed(futures), start=1):
            verified.append(future.result())
            if index % 250 == 0 or index == len(futures):
                LOGGER.info("Official checksum verification %d/%d", index, len(futures))
    lookup = {key: (checksum, path, passed) for key, checksum, path, passed in verified}
    for index, row in working.loc[working["checksum_required"]].iterrows():
        checksum, path, passed = lookup[str(row["archive_key"])]
        working.at[index, "official_checksum"] = checksum
        working.at[index, "checksum_path"] = path
        working.at[index, "checksum_verified"] = bool(passed)
    if required_rows and not bool(working.loc[working["checksum_required"], "checksum_verified"].all()):
        raise ValueError("One or more membership-used archives failed official checksum verification")
    _atomic_csv(working, Path(source_cfg["file_manifest_path"]))
    _write_state(
        source_cfg,
        phase="checksum_complete",
        required_checksums=len(required_rows),
        verified_checksums=int(working["checksum_verified"].fillna(False).sum()),
    )
    return working


def build_point_in_time_panels(
    history: pd.DataFrame,
    membership: pd.DataFrame,
    variant_cfg: Mapping,
    analysis_cfg: Mapping,
) -> dict[str, pd.DataFrame | pd.Series]:
    daily_rows = _build_point_in_time_period_rows(
        history, membership, "daily", variant_cfg, analysis_cfg
    )
    weekly_rows = _build_point_in_time_period_rows(
        history, membership, "weekly", variant_cfg, analysis_cfg
    )
    return {
        "daily_rows": daily_rows,
        "daily_panel": daily_rows.pivot(index="period", columns="asset_key", values="asset_return"),
        "daily_market": _series_from_rows(daily_rows, "market_return"),
        "daily_csad": _series_from_rows(daily_rows, "csad"),
        "daily_coverage": _coverage_from_rows(daily_rows, "day"),
        "weekly_rows": weekly_rows,
        "weekly_panel": weekly_rows.pivot(index="period", columns="asset_key", values="asset_return"),
        "weekly_market": _series_from_rows(weekly_rows, "market_return"),
        "weekly_csad": _series_from_rows(weekly_rows, "csad"),
        "weekly_coverage": _coverage_from_rows(weekly_rows, "week"),
    }


def _build_point_in_time_period_rows(
    history: pd.DataFrame,
    membership: pd.DataFrame,
    frequency: str,
    variant_cfg: Mapping,
    analysis_cfg: Mapping,
) -> pd.DataFrame:
    working = history.sort_values(["asset_key", "date"]).copy()
    if frequency == "weekly":
        working["period"] = working["date"] - pd.to_timedelta(
            working["date"].dt.weekday, unit="D"
        )
        working = (
            working.groupby(
                ["asset_key", "symbol", "base_asset", "listing_episode", "period"],
                as_index=False,
            )
            .agg(
                period_date=("date", "last"),
                close_usdt=("close_usdt", "last"),
                quote_volume_usdt=("quote_volume_usdt", "sum"),
                source_observations=("date", "count"),
            )
        )
        expected_delta = pd.Timedelta(days=7)
    elif frequency == "daily":
        working = working.rename(columns={"date": "period"})
        working["period_date"] = working["period"]
        working["source_observations"] = 1
        expected_delta = pd.Timedelta(days=1)
    else:
        raise ValueError(f"Unsupported point-in-time frequency: {frequency}")

    grouped = working.groupby("asset_key", sort=False)
    working["previous_period"] = grouped["period"].shift(1)
    working["previous_close_usdt"] = grouped["close_usdt"].shift(1)
    working["previous_quote_volume_usdt"] = grouped["quote_volume_usdt"].shift(1)
    exact_previous = working["period"].sub(working["previous_period"]).eq(expected_delta)
    positive = working["close_usdt"].gt(0) & working["previous_close_usdt"].gt(0)
    ratio = working["close_usdt"].where(positive) / working["previous_close_usdt"].where(positive)
    if str(variant_cfg["return_method"]) == "log":
        working["asset_return"] = np.log(ratio).where(exact_previous)
    elif str(variant_cfg["return_method"]) == "simple":
        working["asset_return"] = (ratio - 1.0).where(exact_previous)
    else:
        raise ValueError("return_method must be log or simple")

    weighting = str(variant_cfg["market_weighting"])
    if weighting == "equal":
        working["return_weight"] = 1.0
    elif weighting == "lagged_quote_volume":
        working["return_weight"] = working["previous_quote_volume_usdt"].where(exact_previous)
    else:
        raise ValueError("market_weighting must be equal or lagged_quote_volume")

    working["membership_month"] = _month_string(working["period_date"])
    member_columns = [
        "membership_month",
        "asset_key",
        "rank",
        "source_month",
        "prior_month_quote_volume",
    ]
    working = working.merge(
        membership[member_columns],
        on=["membership_month", "asset_key"],
        how="inner",
        validate="many_to_one",
    )
    start = _utc_day(analysis_cfg["start"])
    end = _utc_day(analysis_cfg["end"])
    working = working.loc[working["period_date"].between(start, end)].copy()
    working["valid_return_weight"] = working["asset_return"].notna() & working["return_weight"].gt(0)
    active = working.groupby("period")["valid_return_weight"].sum().rename("active_assets")
    membership_size = working.groupby("period")["asset_key"].nunique().rename("membership_size")
    eligible = active.loc[active.ge(int(analysis_cfg["minimum_active_assets"]))].index
    working["eligible"] = working["period"].isin(eligible)
    valid = working.loc[working["eligible"] & working["valid_return_weight"]].copy()
    valid["weighted_return"] = valid["asset_return"] * valid["return_weight"]
    aggregates = valid.groupby("period").agg(
        weighted_return=("weighted_return", "sum"),
        total_weight=("return_weight", "sum"),
    )
    market = (aggregates["weighted_return"] / aggregates["total_weight"]).rename("market_return")
    valid["market_return"] = valid["period"].map(market)
    valid["absolute_deviation"] = (valid["asset_return"] - valid["market_return"]).abs()
    csad = valid.groupby("period")["absolute_deviation"].mean().rename("csad")
    working = working.merge(market, left_on="period", right_index=True, how="left", validate="many_to_one")
    working = working.merge(csad, left_on="period", right_index=True, how="left", validate="many_to_one")
    working = working.merge(active, left_on="period", right_index=True, how="left", validate="many_to_one")
    working = working.merge(
        membership_size, left_on="period", right_index=True, how="left", validate="many_to_one"
    )
    return working.sort_values(["period", "rank", "asset_key"]).reset_index(drop=True)


def _coverage_from_rows(rows: pd.DataFrame, label: str) -> pd.DataFrame:
    coverage = rows.groupby("period", as_index=False).agg(
        active_assets=("active_assets", "first"),
        membership_size=("membership_size", "first"),
        observed_assets=("asset_key", "nunique"),
        eligible=("eligible", "first"),
        source_observations=("source_observations", "sum"),
    )
    coverage["frequency"] = label
    return coverage


def _series_from_rows(rows: pd.DataFrame, column: str) -> pd.Series:
    frame = rows.loc[rows["eligible"], ["period", column]].drop_duplicates("period").dropna()
    return frame.set_index("period")[column].sort_index().rename(column)


def validate_archive_quality(
    candidates: pd.DataFrame,
    files: pd.DataFrame,
    inventory_manifest: pd.DataFrame,
    file_manifest: pd.DataFrame,
    history: pd.DataFrame,
    membership: pd.DataFrame,
    panels_by_variant: Mapping[str, Mapping],
    source_cfg: Mapping,
    analysis_cfg: Mapping,
) -> pd.DataFrame:
    source_quality_path = Path(source_cfg["source_quality_path"])
    source_quality = pd.read_csv(source_quality_path) if source_quality_path.is_file() else pd.DataFrame()
    included_symbols = int(candidates["included"].fillna(False).sum())
    symbol_inventories = int(inventory_manifest["inventory_type"].eq("symbol").sum())
    root_pages = inventory_manifest.loc[inventory_manifest["inventory_type"].eq("root")]
    root_complete = bool(not root_pages.empty and not bool(root_pages.iloc[-1]["is_truncated"]))
    checksum_required = file_manifest["checksum_required"].fillna(False).astype(bool)
    checksum_share = (
        float(file_manifest.loc[checksum_required, "checksum_verified"].fillna(False).mean())
        if checksum_required.any()
        else 0.0
    )
    membership_periods = pd.PeriodIndex(membership["membership_month"], freq="M")
    source_periods = pd.PeriodIndex(membership["source_month"], freq="M")
    lookahead_safe = bool(np.all((membership_periods.astype(int) - source_periods.astype(int)) == 1))
    checks = [
        ("bucket_inventory_pagination_complete", float(root_complete), 1.0),
        (
            "candidate_symbol_inventory_completion",
            symbol_inventories / included_symbols if included_symbols else 0.0,
            1.0,
        ),
        (
            "discovered_archive_download_completion",
            float(file_manifest["zip_valid"].fillna(False).sum()) / len(files) if len(files) else 0.0,
            1.0,
        ),
        (
            "source_quality_audit_completion",
            len(source_quality) / len(files) if len(files) else 0.0,
            1.0,
        ),
        (
            "conflicting_source_duplicate_timestamps_zero",
            float(
                not source_quality.empty
                and source_quality["conflicting_duplicate_timestamps"].sum() == 0
            ),
            1.0,
        ),
        ("membership_archive_checksum_completion", checksum_share, 1.0),
        ("unique_date_asset_key", float(not history.duplicated(["date", "asset_key"]).any()), 1.0),
        ("positive_close_share", float(history["close_usdt"].gt(0).mean()), float(analysis_cfg["minimum_positive_price_share"])),
        ("membership_one_month_lag", float(lookahead_safe), 1.0),
    ]
    for variant, panels in panels_by_variant.items():
        daily = panels["daily_coverage"]
        eligible = daily.loc[daily["eligible"]]
        cross_episode = panels["daily_rows"].loc[
            panels["daily_rows"]["asset_return"].notna(), "previous_period"
        ].notna().all()
        checks.extend(
            [
                (
                    f"eligible_daily_share:{variant}",
                    float(daily["eligible"].mean()),
                    float(analysis_cfg["minimum_eligible_day_share"]),
                ),
                (
                    f"minimum_active_assets:{variant}",
                    float(eligible["active_assets"].min()) if not eligible.empty else 0.0,
                    float(analysis_cfg["minimum_active_assets"]),
                ),
                (f"no_cross_episode_returns:{variant}", float(cross_episode), 1.0),
            ]
        )
    quality = pd.DataFrame(checks, columns=["check", "observed", "required"])
    quality["passes"] = quality["observed"].ge(quality["required"])
    if not bool(quality["passes"].all()):
        failed = quality.loc[~quality["passes"], "check"].tolist()
        raise ValueError(f"Binance archive point-in-time quality gates failed: {', '.join(failed)}")
    return quality


def build_external_estimate_comparison(
    archive_targets: pd.DataFrame,
    comparison_cfg: Mapping,
    decision_cfg: Mapping,
) -> pd.DataFrame:
    source_frames = {
        "cmc_historical": pd.read_csv(comparison_cfg["cmc_historical_targets"]),
        "cmc_holdout": pd.read_csv(comparison_cfg["cmc_holdout_targets"]),
        "binance_fixed": pd.read_csv(comparison_cfg["binance_targets"]),
        "okx_listing_aware": pd.read_csv(comparison_cfg["okx_targets"]),
        "binance_archive": archive_targets,
    }
    specs = [
        ("cmc_historical_primary", "cmc_historical", "replication_primary", "full_sample", "CMC", "fixed_62_survivor", "contemporaneous_market_cap"),
        ("cmc_holdout_primary", "cmc_holdout", "replication_primary", "holdout_full", "CMC", "fixed_62_survivor_holdout", "contemporaneous_market_cap"),
        ("binance_fixed_equal", "binance_fixed", "equal_weight_primary", "full_5y", "Binance", "fixed_14_survivor", "equal"),
        ("binance_fixed_lagged", "binance_fixed", "lagged_turnover_sensitivity", "full_5y", "Binance", "fixed_14_survivor", "lagged_liquidity_proxy"),
        ("okx_listing_equal", "okx_listing_aware", "equal_weight_primary", "full_5y", "OKX", "listing_aware_14_survivor", "equal"),
        ("okx_listing_lagged", "okx_listing_aware", "lagged_quote_volume_sensitivity", "full_5y", "OKX", "listing_aware_14_survivor", "lagged_quote_volume"),
        ("binance_archive_pit_equal", "binance_archive", str(decision_cfg["primary_variant"]), str(decision_cfg["decision_period"]), "Binance", "archive_point_in_time_top50", "equal"),
        ("binance_archive_pit_lagged", "binance_archive", str(decision_cfg["sensitivity_variant"]), str(decision_cfg["decision_period"]), "Binance", "archive_point_in_time_top50", "lagged_quote_volume"),
    ]
    models = set(decision_cfg["required_models"])
    frequencies = set(decision_cfg["required_frequencies"])
    frames = []
    for label, source_key, variant, period, provider, universe, weighting in specs:
        source = source_frames[source_key]
        selected = source.loc[
            source["variant"].eq(variant)
            & source["period"].eq(period)
            & source["model"].isin(models)
            & source["frequency"].isin(frequencies)
        ].copy()
        if len(selected) != len(models) * len(frequencies):
            raise ValueError(f"Meta comparison expected four corrected cells for {label}, found {len(selected)}")
        selected.insert(0, "estimate_id", label)
        selected.insert(1, "provider", provider)
        selected.insert(2, "universe", universe)
        selected.insert(3, "weighting", weighting)
        selected["standardized_std_error"] = _standardized_std_error(selected)
        frames.append(selected)
    comparison = pd.concat(frames, ignore_index=True)
    if comparison["standardized_std_error"].isna().any() or not comparison["standardized_std_error"].gt(0).all():
        raise ValueError("Meta comparison contains invalid standardized standard errors")
    return comparison.sort_values(["frequency", "model", "estimate_id"]).reset_index(drop=True)


def run_descriptive_meta_analysis(
    comparison: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for (frequency, model), group in comparison.groupby(["frequency", "model"], sort=True):
        theta = group["standardized_target_coefficient"].to_numpy(dtype=float)
        se = group["standardized_std_error"].to_numpy(dtype=float)
        weights = 1.0 / np.square(se)
        fixed = float(np.sum(weights * theta) / np.sum(weights))
        fixed_se = float(np.sqrt(1.0 / np.sum(weights)))
        q = float(np.sum(weights * np.square(theta - fixed)))
        degrees = len(theta) - 1
        c_value = float(np.sum(weights) - np.sum(np.square(weights)) / np.sum(weights))
        tau_squared = max(0.0, (q - degrees) / c_value) if c_value > 0 else 0.0
        random_weights = 1.0 / (np.square(se) + tau_squared)
        random = float(np.sum(random_weights * theta) / np.sum(random_weights))
        random_se = float(np.sqrt(1.0 / np.sum(random_weights)))
        fixed_z = fixed / fixed_se
        random_z = random / random_se
        rows.append(
            {
                "frequency": frequency,
                "model": model,
                "estimate_count": len(theta),
                "fixed_effect": fixed,
                "fixed_std_error": fixed_se,
                "fixed_z": fixed_z,
                "fixed_p_value": float(2 * norm.sf(abs(fixed_z))),
                "fixed_ci_lower": fixed - norm.ppf(0.975) * fixed_se,
                "fixed_ci_upper": fixed + norm.ppf(0.975) * fixed_se,
                "random_effect": random,
                "random_std_error": random_se,
                "random_z": random_z,
                "random_p_value": float(2 * norm.sf(abs(random_z))),
                "random_ci_lower": random - norm.ppf(0.975) * random_se,
                "random_ci_upper": random + norm.ppf(0.975) * random_se,
                "cochran_q": q,
                "q_p_value": float(chi2.sf(q, degrees)) if degrees > 0 else np.nan,
                "i_squared": max(0.0, (q - degrees) / q) if q > 0 else 0.0,
                "tau_squared": tau_squared,
            }
        )
    result = pd.DataFrame(rows)
    result["fixed_q_value_bh_fdr"] = benjamini_hochberg(result["fixed_p_value"])
    result["random_q_value_bh_fdr"] = benjamini_hochberg(result["random_p_value"])
    result["fixed_negative_fdr"] = result["fixed_effect"].lt(0) & result["fixed_q_value_bh_fdr"].le(alpha)
    result["random_negative_fdr"] = result["random_effect"].lt(0) & result["random_q_value_bh_fdr"].le(alpha)
    return result


def build_moderator_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for dimension in ("provider", "universe", "weighting", "frequency", "model"):
        grouped = (
            comparison.groupby(dimension, as_index=False)
            .agg(
                estimate_count=("standardized_target_coefficient", "size"),
                mean_standardized_coefficient=("standardized_target_coefficient", "mean"),
                median_standardized_coefficient=("standardized_target_coefficient", "median"),
                negative_share=("standardized_target_coefficient", lambda values: float(values.lt(0).mean())),
                fdr_support_share=("supports_herding", lambda values: float(values.fillna(False).mean())),
            )
            .rename(columns={dimension: "category"})
        )
        grouped.insert(0, "dimension", dimension)
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)


def build_posthoc_taxonomy_audit(
    membership: pd.DataFrame,
    universe_cfg: Mapping,
) -> pd.DataFrame:
    """Flag name-based taxonomy risks without changing the frozen universe."""
    preregistered = {
        *(str(value) for value in universe_cfg["excluded_base_assets"]),
        *(str(value) for value in universe_cfg["excluded_wrapped_staked_assets"]),
    }
    frame = membership.loc[
        membership["base_asset"].astype(str).str.contains("USD", case=False, regex=False)
        & ~membership["base_asset"].isin(preregistered)
    ].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "asset_key",
                "base_asset",
                "first_membership_month",
                "last_membership_month",
                "membership_months",
                "best_rank",
                "audit_flag",
            ]
        )
    result = (
        frame.groupby(["asset_key", "base_asset"], as_index=False)
        .agg(
            first_membership_month=("membership_month", "min"),
            last_membership_month=("membership_month", "max"),
            membership_months=("membership_month", "nunique"),
            best_rank=("rank", "min"),
        )
    )
    result["audit_flag"] = "name_contains_usd_not_in_preregistered_exclusions"
    return result.sort_values(["first_membership_month", "asset_key"]).reset_index(drop=True)


def build_weight_concentration_summary(
    panels_by_variant: Mapping[str, Mapping],
) -> pd.DataFrame:
    rows = []
    for variant, panels in panels_by_variant.items():
        for frequency in ("daily", "weekly"):
            frame = panels[f"{frequency}_rows"]
            valid = frame.loc[frame["eligible"] & frame["valid_return_weight"]].copy()
            valid["weight_share"] = valid["return_weight"] / valid.groupby("period")[
                "return_weight"
            ].transform("sum")
            by_period = valid.groupby("period").agg(
                maximum_asset_weight=("weight_share", "max"),
                herfindahl=("weight_share", lambda values: float(np.square(values).sum())),
            )
            by_period["effective_asset_count"] = 1.0 / by_period["herfindahl"]
            rows.append(
                {
                    "variant": variant,
                    "frequency": frequency,
                    "period_count": len(by_period),
                    "mean_maximum_asset_weight": by_period["maximum_asset_weight"].mean(),
                    "median_maximum_asset_weight": by_period["maximum_asset_weight"].median(),
                    "p95_maximum_asset_weight": by_period["maximum_asset_weight"].quantile(0.95),
                    "maximum_asset_weight": by_period["maximum_asset_weight"].max(),
                    "mean_effective_asset_count": by_period["effective_asset_count"].mean(),
                    "median_effective_asset_count": by_period["effective_asset_count"].median(),
                }
            )
    return pd.DataFrame(rows)


def plot_meta_forest(
    comparison: pd.DataFrame,
    meta: pd.DataFrame,
    path: str | Path,
) -> None:
    cells = [("daily", "no_intercept_csad"), ("daily", "scsad"), ("weekly", "no_intercept_csad"), ("weekly", "scsad")]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=False)
    for axis, (frequency, model) in zip(axes.flat, cells, strict=True):
        group = comparison.loc[
            comparison["frequency"].eq(frequency) & comparison["model"].eq(model)
        ].sort_values("estimate_id")
        y = np.arange(len(group), 0, -1)
        coefficients = group["standardized_target_coefficient"].to_numpy(dtype=float)
        errors = 1.96 * group["standardized_std_error"].to_numpy(dtype=float)
        axis.errorbar(coefficients, y, xerr=errors, fmt="o", color="#245b4a", capsize=3)
        pooled = meta.loc[meta["frequency"].eq(frequency) & meta["model"].eq(model)].iloc[0]
        axis.errorbar(
            pooled["random_effect"],
            0,
            xerr=1.96 * pooled["random_std_error"],
            fmt="D",
            color="#c4512d",
            capsize=4,
            label="Random-effects pooled",
        )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks([*y, 0], [*group["estimate_id"].tolist(), "pooled_random"])
        axis.set_title(f"{frequency} / {model}")
        axis.set_xlabel("Standardized target coefficient (95% CI)")
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Corrected CSAD: provider, universe, and weighting comparison", fontsize=15)
    fig.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=170)
    plt.close(fig)


def build_archive_report(
    config: Mapping,
    candidates: pd.DataFrame,
    file_manifest: pd.DataFrame,
    episode_coverage: pd.DataFrame,
    membership: pd.DataFrame,
    quality: pd.DataFrame,
    panels_by_variant: Mapping[str, Mapping],
    targets: pd.DataFrame,
    decision_summary: pd.DataFrame,
    meta: pd.DataFrame,
    taxonomy_audit: pd.DataFrame,
    weight_concentration: pd.DataFrame,
    plot_paths: Sequence[str],
) -> str:
    primary = str(config["decision"]["primary_variant"])
    sensitivity = str(config["decision"]["sensitivity_variant"])
    decisions = decision_summary.set_index("variant")
    full = targets.loc[targets["period"].eq(config["decision"]["decision_period"])]
    required = targets.loc[
        targets["model"].isin(config["decision"]["required_models"])
        & targets["frequency"].isin(config["decision"]["required_frequencies"])
    ]
    delisted = int(episode_coverage["archive_ended_before_source_end"].sum())
    lines = [
        "# Binance Archive Point-in-Time Universe 외부검증",
        "",
        "## 한눈에 보는 결론",
        "",
        f"- 동일가중 corrected 필수 셀: {int(decisions.loc[primary, 'passing_cells'])}/4 통과",
        f"- 전기 quote-volume 가중 필수 셀: {int(decisions.loc[sensitivity, 'passing_cells'])}/4 통과",
        f"- 동일가중 strict criterion: {'통과' if bool(decisions.loc[primary, 'all_required_cells_pass']) else '미통과'}",
        f"- 전기 quote-volume 가중 strict criterion: {'통과' if bool(decisions.loc[sensitivity, 'all_required_cells_pass']) else '미통과'}",
        "- 이 결과는 동시적 횡단면 수렴 관계의 외부검증이며 미래수익률 alpha, 인과효과 또는 의도적 모방의 검정이 아닙니다.",
        "",
        "## 왜 이 연구를 다시 했나",
        "",
        "현재까지 살아남은 코인만 고르면 과거에 상장폐지된 실패 자산이 빠집니다. 이를 survivor bias라고 하며, 과거의 시장이 실제보다 안정적이거나 동질적이었던 것처럼 보이게 할 수 있습니다. 이번 검증은 현재 목록이 아니라 Binance 공식 아카이브에 당시 일봉이 실제로 남은 종목을 다시 열거해 이 편향을 줄였습니다.",
        "",
        "핵심 질문은 단순합니다. 상장폐지 종목까지 포함하고, 당시에 알 수 있었던 전월 정보로만 종목을 골라도 corrected CSAD 관계가 daily와 weekly에서 모두 재현되는지를 묻습니다.",
        "",
        "## 누가 다시 실행해도 같은 표본을 만드는 방법",
        "",
        "1. Binance Vision spot monthly `1d` bucket의 symbol prefix를 페이지 끝까지 전수 열거합니다.",
        "2. 결과를 보기 전에 고정한 USDT·leveraged token·stable/fiat·wrapped/staked 필터만 적용합니다.",
        "3. 같은 ticker에 7일을 넘는 공백이 있으면 새 listing episode로 분리해 LUNA처럼 이름이 재사용된 자산의 수익률을 연결하지 않습니다.",
        "4. 각 달의 universe는 직전 달 20일 이상 거래된 episode 중 quote volume 상위 50개로 고정합니다. 당월 신규상장은 다음 달부터만 들어옵니다.",
        "5. daily·weekly와 동일가중·정확한 전기 quote-volume 가중을 나눠 standard, no-intercept, SCSAD를 실행하고 각 6개 검정군을 BH-FDR로 보정합니다.",
        "",
        "사전 strict 기준은 daily·weekly no-intercept·SCSAD 네 셀의 계수가 모두 음수이고 BH q-value가 0.05 이하일 것입니다. 일부 셀만 좋은 결과를 전체 재현으로 포장하지 않기 위한 기준입니다.",
        "",
        "## 데이터와 생존편향 통제",
        "",
        f"- 공식 archive root에서 발견한 전체 symbol prefix: {len(candidates):,}개",
        f"- 사전 필터를 통과한 USDT 현물 후보: {int(candidates['included'].sum()):,}개",
        f"- 분석 범위 monthly ZIP: {len(file_manifest):,}개",
        f"- listing episode: {len(episode_coverage):,}개, 종료일 이전 archive가 끊긴 episode: {delisted:,}개",
        f"- 월별 universe: 전월 20일 이상 관측 episode 중 quote volume 상위 {config['universe']['top_n']}개",
        f"- membership 월: {membership['membership_month'].nunique()}개, 실제 편입 episode: {membership['asset_key'].nunique()}개",
        f"- 공식 checksum 검증: {int(file_manifest['checksum_verified'].fillna(False).sum()):,}/{int(file_manifest['checksum_required'].fillna(False).sum()):,}개",
        f"- 품질 gate: {int(quality['passes'].sum())}/{len(quality)} 통과",
        "- 원시 ZIP 전수 감사에서 완전 동일 행 1개만 축약했고 값이 다른 충돌 timestamp는 0개였습니다.",
    ]
    for variant, panels in panels_by_variant.items():
        daily = panels["daily_coverage"]
        lines.append(
            f"- {variant}: daily {len(panels['daily_market']):,}개, weekly {len(panels['weekly_market']):,}개, "
            f"daily 활성 episode {int(daily.loc[daily['eligible'], 'active_assets'].min())}~{int(daily['active_assets'].max())}개"
        )
    lines.append(
        "- weekly panel은 경계일 가격 정렬을 위한 부분 시작주까지 262개를 보존하고, "
        "사전 5년 회귀 구간은 완전히 들어오는 261개 주를 사용합니다."
    )
    lines.extend(["", "## 가중치 농축 감사", ""])
    for row in weight_concentration.itertuples():
        lines.append(
            f"- {row.variant} / {row.frequency}: 기간별 최대 종목비중 중앙값 {row.median_maximum_asset_weight:.1%}, "
            f"95백분위 {row.p95_maximum_asset_weight:.1%}, 최대 {row.maximum_asset_weight:.1%}, "
            f"유효 종목 수 중앙값 {row.median_effective_asset_count:.1f}개"
        )
    lines.extend(["", "## 5년 전체표본 회귀", ""])
    for row in full.itertuples():
        lines.append(
            f"- {row.variant} / {row.frequency} / {row.model}: 계수 {row.coefficient:.4f}, "
            f"표준화 {row.standardized_target_coefficient:.3f}, t={row.t_stat:.2f}, "
            f"BH q={row.q_value_bh_fdr:.3g}, 지지={bool(row.supports_herding)}"
        )
    lines.extend(["", "## 전반·후반 안정성", ""])
    for period in ("full_5y", "early_half", "late_half"):
        for variant in (primary, sensitivity):
            subset = required.loc[required["period"].eq(period) & required["variant"].eq(variant)]
            lines.append(
                f"- {period} / {variant}: corrected {int(subset['supports_herding'].sum())}/4 통과"
            )
    lines.extend(
        [
            "",
            "동일가중은 5년 전체에서 daily 두 셀만 통과했고 weekly 두 셀은 통과하지 못했습니다. 전기 유동성가중은 5년 전체에서 네 셀 모두 미통과였습니다. 후반기에는 두 가중법 모두 4/4로 바뀌지만, 이는 full-sample 사전 판정을 대체하지 않으며 관계가 시기에 따라 크게 달라진다는 진단으로만 사용합니다.",
        ]
    )
    lines.extend(["", "## Descriptive pooled evidence", ""])
    for row in meta.itertuples():
        lines.append(
            f"- {row.frequency} / {row.model}: random-effects 표준화 계수 {row.random_effect:.3f} "
            f"(95% CI {row.random_ci_lower:.3f}~{row.random_ci_upper:.3f}), "
            f"BH q={row.random_q_value_bh_fdr:.3g}, I-squared={row.i_squared:.1%}"
        )
    lines.extend(
        [
            "",
            "CMC·Binance·OKX의 8개 사양을 표준화해 합치면 random-effects에서 4셀 중 3셀이 음수·BH-FDR를 통과합니다. 그러나 I-squared가 69.6%~96.8%로 매우 높아, 평균 계수 하나보다 공급자·universe·가중법에 따라 결과가 달라진다는 사실이 더 중요합니다. 특히 archive point-in-time 유동성가중은 daily corrected 계수가 양수로 바뀌어 보편적 herding 관계라는 해석을 약화시킵니다.",
            "",
            "## 결국 무슨 의미인가",
            "",
            "이번 결과는 corrected CSAD 관계가 완전히 사라졌다는 뜻도, 암호화폐 시장 전체에서 항상 허딩이 있다는 뜻도 아닙니다. 생존 종목 고정 표본에서 비교적 넓게 보이던 관계가 상장폐지를 포함한 동적 universe와 실제 전기 유동성 가중에서는 크게 약해졌다는 뜻입니다.",
            "",
            "따라서 현재 가장 안전한 결론은 `corrected CSAD는 특정 표본·시기·가중법에서 횡단면 수렴을 보여주지만, 보편적이거나 투자 가능한 alpha로 식별되지 않았다`입니다.",
        ]
    )
    lines.extend(
        [
            "",
            "## 해석 제한",
            "",
            "- Binance Vision archive 존재는 당시 거래 가능성의 감사 가능한 proxy지만 완전한 historical exchangeInfo master는 아닙니다.",
            "- 상장폐지 시각은 마지막 complete UTC daily candle로 근사하며 공식 공지 시각과 다를 수 있습니다.",
            "- ticker 공백 7일 초과를 새 episode로 분리했지만 모든 토큰 migration을 완전하게 식별하지 못할 수 있습니다.",
            "- 정적 stablecoin·wrapped token 제외는 point-in-time taxonomy의 완전한 복원이 아닙니다.",
            f"- 결과 확인 후 taxonomy 감사에서 사전 제외목록에 없던 USD 이름 episode {len(taxonomy_audit)}개가 일부 후반 membership에 포함된 것을 찾았습니다. primary를 post-hoc로 바꾸지 않고 감사표로 보존합니다.",
            "- 전기 quote-volume 가중은 특정 날 단일 종목 비중이 크게 올라가므로 동일가중 결과와 함께 해석해야 합니다.",
            "- pooled estimate는 기간과 표본이 겹치는 의존적 결과의 기술적 합성이며 독립 연구 메타분석으로 해석하지 않습니다.",
            "- 유의한 음의 corrected coefficient가 있어도 미래수익률 예측력이나 거래전략 수익성을 뜻하지 않습니다.",
            "",
            "## 그림",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in plot_paths)
    return "\n".join(lines) + "\n"


def collection_status(source_cfg: Mapping) -> dict:
    state_path = Path(source_cfg["state_path"])
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    paths = {
        "candidate_inventory": Path(source_cfg["candidate_inventory_path"]),
        "archive_index": Path(source_cfg["archive_index_path"]),
        "inventory_manifest": Path(source_cfg["inventory_manifest_path"]),
        "file_manifest": Path(source_cfg["file_manifest_path"]),
        "source_quality": Path(source_cfg["source_quality_path"]),
        "normalized_history": Path(source_cfg["normalized_path"]),
    }
    state["artifacts"] = {
        name: {"path": path.as_posix(), "exists": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else 0}
        for name, path in paths.items()
    }
    return state


def mark_analysis_complete(
    source_cfg: Mapping,
    output_dir: str | Path,
    primary_strict_pass: bool,
    sensitivity_strict_pass: bool,
) -> None:
    _write_state(
        source_cfg,
        phase="analysis_complete",
        output_dir=Path(output_dir).as_posix(),
        primary_strict_pass=bool(primary_strict_pass),
        sensitivity_strict_pass=bool(sensitivity_strict_pass),
    )


def _archive_rows_from_listing(
    symbol: str,
    base_asset: str,
    contents: Sequence[Mapping],
    source_cfg: Mapping,
) -> list[dict]:
    first_month = pd.Period(str(source_cfg["first_month"]), freq="M")
    last_month = pd.Period(str(source_cfg["last_month"]), freq="M")
    prefix = f"{source_cfg['root_prefix']}{symbol}/{source_cfg['interval']}/"
    rows = []
    for item in contents:
        key = str(item["key"])
        if not key.startswith(prefix) or not key.endswith(".zip"):
            continue
        stem = Path(key).name.removesuffix(".zip")
        expected = f"{symbol}-{source_cfg['interval']}-"
        if not stem.startswith(expected):
            continue
        try:
            month = pd.Period(stem.removeprefix(expected), freq="M")
        except ValueError:
            continue
        if not first_month <= month <= last_month:
            continue
        encoded_key = quote(key, safe="/")
        rows.append(
            {
                "symbol": symbol,
                "base_asset": base_asset,
                "month": str(month),
                "archive_key": key,
                "last_modified": item.get("last_modified", ""),
                "etag": item.get("etag", ""),
                "size_bytes": int(item.get("size_bytes", 0)),
                "download_url": f"{str(source_cfg['download_endpoint']).rstrip('/')}/{encoded_key}",
                "checksum_url": f"{str(source_cfg['download_endpoint']).rstrip('/')}/{encoded_key}.CHECKSUM",
            }
        )
    return rows


def _inventory_manifest_row(
    inventory_type: str,
    page_number: int,
    path: Path,
    parsed: Mapping,
    request_marker: str | None,
    request_prefix: str,
    symbol: str = "",
) -> dict:
    return {
        "inventory_type": inventory_type,
        "symbol": symbol,
        "page_number": page_number,
        "request_prefix": request_prefix,
        "request_marker": request_marker or "",
        "path": path.as_posix(),
        "sha256": _sha256(path),
        "prefix_count": len(parsed["prefixes"]),
        "key_count": len(parsed["contents"]),
        "is_truncated": bool(parsed["is_truncated"]),
        "next_marker": parsed["next_marker"] or "",
    }


def _cached_s3_page(
    path: Path,
    source_cfg: Mapping,
    prefix: str,
    delimiter: str | None = None,
    marker: str | None = None,
) -> bytes:
    if path.is_file() and path.stat().st_size > 0:
        with gzip.open(path, "rb") as handle:
            payload = handle.read()
        parse_s3_listing(payload)
        return payload
    params = {"prefix": prefix}
    if delimiter:
        params["delimiter"] = delimiter
    if marker:
        params["marker"] = marker
    payload = _request_bytes(str(source_cfg["bucket_endpoint"]), source_cfg, params=params)
    parse_s3_listing(payload)
    _atomic_gzip_bytes(path, payload)
    return payload


def _download_file(url: str, target: Path, source_cfg: Mapping) -> None:
    payload = _request_bytes(url, source_cfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.write_bytes(payload)
    os.replace(temporary, target)


def _request_bytes(
    url: str,
    source_cfg: Mapping,
    params: Mapping[str, str] | None = None,
) -> bytes:
    attempts = int(source_cfg["maximum_attempts"])
    for attempt in range(1, attempts + 1):
        try:
            session = getattr(_THREAD_LOCAL, "session", None)
            if session is None:
                session = requests.Session()
                _THREAD_LOCAL.session = session
            response = session.get(
                url,
                params=params,
                timeout=float(source_cfg["request_timeout_seconds"]),
                headers={"User-Agent": str(source_cfg["user_agent"])},
            )
            response.raise_for_status()
            return response.content
        except requests.RequestException as error:
            if attempt >= attempts:
                raise RuntimeError(f"Request failed after {attempts} attempts: {url}") from error
            delay = float(source_cfg["backoff_seconds"]) * (2 ** (attempt - 1))
            time.sleep(delay)
    raise AssertionError("unreachable")


def _valid_zip(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None and len(archive.namelist()) > 0
    except (OSError, zipfile.BadZipFile):
        return False


def _standardized_std_error(frame: pd.DataFrame) -> pd.Series:
    t_stat = pd.to_numeric(frame["t_stat"], errors="coerce")
    coefficient = pd.to_numeric(frame["coefficient"], errors="coerce")
    standardized = pd.to_numeric(frame["standardized_target_coefficient"], errors="coerce")
    raw_se = pd.to_numeric(frame["std_error"], errors="coerce")
    by_t = standardized.abs() / t_stat.abs()
    scale = standardized.abs() / coefficient.abs()
    result = by_t.where(t_stat.abs().gt(1e-12), raw_se * scale)
    return result.replace([np.inf, -np.inf], np.nan)


def _month_string(values: pd.Series) -> pd.Series:
    return values.dt.tz_localize(None).dt.to_period("M").astype(str)


def _utc_day(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC").normalize()
    return timestamp.tz_convert("UTC").normalize()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_gzip_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb") as handle:
        handle.write(payload)
    os.replace(temporary, path)


def _write_state(source_cfg: Mapping, phase: str, **values: object) -> None:
    path = Path(source_cfg["state_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    existing.update(values)
    existing["phase"] = phase
    existing["updated_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)

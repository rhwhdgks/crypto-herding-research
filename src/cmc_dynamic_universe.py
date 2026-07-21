from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

from frequency_sensitivity import benjamini_hochberg
from regression import (
    run_csad_regression,
    run_no_intercept_csad_regression,
    run_scsad_regression,
)


LOGGER = logging.getLogger(__name__)
SNAPSHOT_COLUMNS = [
    "snapshot_date",
    "cmc_id",
    "name",
    "symbol",
    "slug",
    "rank",
    "price_usd",
    "market_cap_usd",
    "volume_24h_usd",
    "circulating_supply",
    "date_added",
    "last_updated",
]
MODEL_SPECS = {
    "standard_csad": (run_csad_regression, "market_return_sq"),
    "no_intercept_csad": (run_no_intercept_csad_regression, "market_return_sq"),
    "scsad": (run_scsad_regression, "market_return_cu"),
}


def collect_cmc_snapshots(
    source_cfg: Mapping,
    excluded_metadata_tags: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = Path(source_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    dates = requested_snapshot_dates(source_cfg)
    completed: list[pd.Timestamp] = []
    missing: list[pd.Timestamp] = []
    for date in dates:
        path = snapshot_cache_path(cache_dir, date)
        if _valid_cached_snapshot(path, date, source_cfg):
            completed.append(date)
        else:
            missing.append(date)

    _write_collection_state(source_cfg, dates, completed, [], "running")
    LOGGER.info(
        "CMC snapshot collection: expected=%d cached=%d missing=%d",
        len(dates),
        len(completed),
        len(missing),
    )

    failures: list[dict] = []
    if missing:
        with ThreadPoolExecutor(max_workers=int(source_cfg["max_workers"])) as executor:
            futures = {
                executor.submit(_download_snapshot_checkpoint, date, source_cfg): date
                for date in missing
            }
            for count, future in enumerate(as_completed(futures), start=1):
                date = futures[future]
                try:
                    future.result()
                    completed.append(date)
                except Exception as exc:  # noqa: BLE001 - collection state retains the failure
                    failures.append({"date": date.strftime("%Y-%m-%d"), "error": str(exc)})
                    LOGGER.error("CMC snapshot failed for %s: %s", date.date(), exc)
                if count % 25 == 0 or count == len(missing):
                    LOGGER.info(
                        "CMC snapshot progress: downloaded=%d/%d failures=%d",
                        count,
                        len(missing),
                        len(failures),
                    )
                    _write_collection_state(
                        source_cfg,
                        dates,
                        completed,
                        failures,
                        "running" if count < len(missing) else "validating",
                    )

    if failures:
        _write_collection_state(source_cfg, dates, completed, failures, "failed")
        raise RuntimeError(f"CMC snapshot collection failed for {len(failures)} dates")

    manifest = build_snapshot_manifest(source_cfg)
    if len(manifest) != len(dates):
        raise ValueError(
            f"CMC snapshot manifest rows={len(manifest):,}, expected={len(dates):,}"
        )
    metadata = collect_current_metadata(
        source_cfg,
        excluded_metadata_tags=excluded_metadata_tags,
    )
    _write_collection_state(source_cfg, dates, completed, [], "complete")
    return manifest, metadata


def requested_snapshot_dates(source_cfg: Mapping) -> pd.DatetimeIndex:
    start = pd.Timestamp(source_cfg["source_start"])
    end = pd.Timestamp(source_cfg["source_end"])
    if start > end:
        raise ValueError("CMC source_start must not be after source_end")
    return pd.date_range(start, end, freq="D")


def snapshot_cache_path(cache_dir: str | Path, date: object) -> Path:
    timestamp = pd.Timestamp(date)
    return (
        Path(cache_dir)
        / f"year={timestamp.year:04d}"
        / f"month={timestamp.month:02d}"
        / f"{timestamp.strftime('%Y-%m-%d')}.parquet"
    )


def parse_historical_snapshot_payload(
    payload: Mapping,
    date: object,
    usd_convert_id: int = 2781,
) -> pd.DataFrame:
    status = payload.get("status", {})
    if str(status.get("error_code", "0")) != "0":
        raise ValueError(f"CMC payload error: {status.get('error_message', 'unknown')}")
    records = payload.get("data")
    if not isinstance(records, list):
        raise ValueError("CMC historical payload data must be a list")

    rows = []
    quote_name = str(usd_convert_id)
    snapshot_date = pd.Timestamp(date).normalize()
    for record in records:
        quotes = record.get("quotes", [])
        quote = next(
            (item for item in quotes if str(item.get("name")) == quote_name),
            None,
        )
        quote = quote or {}
        rows.append(
            {
                "snapshot_date": snapshot_date,
                "cmc_id": int(record["id"]),
                "name": str(record.get("name", "")),
                "symbol": str(record.get("symbol", "")),
                "slug": str(record.get("slug", "")),
                "rank": pd.to_numeric(record.get("cmcRank"), errors="coerce"),
                "price_usd": _safe_float(quote.get("price")),
                "market_cap_usd": _safe_float(quote.get("marketCap")),
                "volume_24h_usd": _safe_float(quote.get("volume24h")),
                "circulating_supply": _safe_float(record.get("circulatingSupply")),
                "date_added": pd.to_datetime(record.get("dateAdded"), utc=True),
                "last_updated": pd.to_datetime(
                    quote.get("lastUpdated") or record.get("lastUpdated"), utc=True
                ),
            }
        )
    frame = pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)
    if not frame.empty:
        frame["rank"] = frame["rank"].astype("Int64")
        frame = frame.sort_values(["rank", "cmc_id"]).reset_index(drop=True)
    return frame


def validate_snapshot_frame(frame: pd.DataFrame, date: object, source_cfg: Mapping) -> None:
    missing = sorted(set(SNAPSHOT_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"CMC snapshot missing columns: {', '.join(missing)}")
    expected_rows = int(
        source_cfg.get("minimum_snapshot_rows", source_cfg["retrieval_limit"])
    )
    if len(frame) < expected_rows:
        raise ValueError(f"CMC snapshot rows={len(frame)}, required at least {expected_rows}")
    expected_date = pd.Timestamp(date).normalize()
    actual_dates = pd.to_datetime(frame["snapshot_date"]).dt.normalize().unique()
    if len(actual_dates) != 1 or pd.Timestamp(actual_dates[0]) != expected_date:
        raise ValueError("CMC snapshot date does not match its checkpoint date")
    if frame["cmc_id"].duplicated().any():
        raise ValueError("CMC snapshot contains duplicate CMC IDs")
    if frame["rank"].duplicated().any():
        raise ValueError("CMC snapshot contains duplicate ranks")
    positive = frame["price_usd"].gt(0) & frame["market_cap_usd"].gt(0)
    minimum_share = float(source_cfg.get("minimum_positive_quote_share", 0.95))
    if float(positive.mean()) < minimum_share:
        raise ValueError(
            f"CMC positive quote share={positive.mean():.2%}, minimum={minimum_share:.2%}"
        )


def build_snapshot_manifest(source_cfg: Mapping) -> pd.DataFrame:
    cache_dir = Path(source_cfg["cache_dir"])
    rows = []
    for date in requested_snapshot_dates(source_cfg):
        path = snapshot_cache_path(cache_dir, date)
        if not _valid_cached_snapshot(path, date, source_cfg):
            raise ValueError(f"Missing or invalid CMC checkpoint: {path}")
        checkpoint = pd.read_parquet(
            path,
            columns=["rank", "price_usd", "market_cap_usd"],
        )
        positive = checkpoint["price_usd"].gt(0) & checkpoint["market_cap_usd"].gt(0)
        retrieval_limit = int(source_cfg["retrieval_limit"])
        observed_ranks = set(checkpoint["rank"].dropna().astype(int))
        rows.append(
            {
                "snapshot_date": date,
                "path": path.as_posix(),
                "rows": int(pq.ParquetFile(path).metadata.num_rows),
                "minimum_rank": int(checkpoint["rank"].min()),
                "maximum_rank": int(checkpoint["rank"].max()),
                "missing_ranks_through_limit": len(
                    set(range(1, retrieval_limit + 1)).difference(observed_ranks)
                ),
                "positive_quote_share": float(positive.mean()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    manifest = pd.DataFrame(rows)
    destination = Path(source_cfg["manifest_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(destination, index=False)
    return manifest


def collect_current_metadata(
    source_cfg: Mapping,
    excluded_metadata_tags: Sequence[str] = (),
    force: bool = False,
) -> pd.DataFrame:
    destination = Path(source_cfg["metadata_path"])
    if destination.is_file() and not force:
        frame = pd.read_parquet(destination)
        return apply_metadata_exclusions(frame, excluded_metadata_tags)
    params = {
        "start": 1,
        "limit": int(source_cfg["current_metadata_limit"]),
        "sortBy": "market_cap",
        "sortType": "desc",
        "convert": "USD",
        "cryptoType": "all",
        "tagType": "all",
        "audited": "false",
    }
    payload = _request_json(source_cfg["current_listing_endpoint"], params, source_cfg)
    data = payload.get("data", {})
    records = data.get("cryptoCurrencyList", []) if isinstance(data, dict) else []
    if len(records) < 1000:
        raise ValueError("CMC current metadata response is unexpectedly small")
    excluded_tags = {str(tag) for tag in excluded_metadata_tags}
    rows = []
    for record in records:
        tags = sorted(str(tag) for tag in (record.get("tags") or []))
        rows.append(
            {
                "cmc_id": int(record["id"]),
                "metadata_name": str(record.get("name", "")),
                "metadata_symbol": str(record.get("symbol", "")),
                "metadata_slug": str(record.get("slug", "")),
                "tags_json": json.dumps(tags, ensure_ascii=True),
                "excluded_by_metadata_tag": bool(excluded_tags.intersection(tags)),
                "metadata_fetched_at_utc": datetime.now(timezone.utc),
            }
        )
    frame = pd.DataFrame(rows).sort_values("cmc_id").reset_index(drop=True)
    frame = apply_metadata_exclusions(frame, excluded_metadata_tags)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_parquet_write(frame, destination)
    return frame


def apply_metadata_exclusions(
    metadata: pd.DataFrame,
    excluded_metadata_tags: Sequence[str],
) -> pd.DataFrame:
    frame = metadata.copy()
    excluded_tags = {str(tag) for tag in excluded_metadata_tags}

    def has_excluded_tag(raw_tags: object) -> bool:
        try:
            tags = set(json.loads(str(raw_tags)))
        except (TypeError, ValueError, json.JSONDecodeError):
            tags = set()
        return bool(excluded_tags.intersection(tags))

    frame["excluded_by_metadata_tag"] = frame["tags_json"].map(has_excluded_tag)
    return frame


def collection_status(source_cfg: Mapping) -> dict:
    dates = requested_snapshot_dates(source_cfg)
    cache_dir = Path(source_cfg["cache_dir"])
    completed = sum(snapshot_cache_path(cache_dir, date).is_file() for date in dates)
    state_path = Path(source_cfg["state_path"])
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    return {
        "expected": len(dates),
        "completed": int(completed),
        "remaining": int(len(dates) - completed),
        "completion_share": float(completed / len(dates)),
        "state": state,
    }


def load_snapshot_history(source_cfg: Mapping) -> pd.DataFrame:
    frames = []
    cache_dir = Path(source_cfg["cache_dir"])
    for date in requested_snapshot_dates(source_cfg):
        path = snapshot_cache_path(cache_dir, date)
        if not path.is_file():
            raise ValueError(f"CMC checkpoint is missing: {path}")
        frames.append(pd.read_parquet(path, columns=SNAPSHOT_COLUMNS))
    history = pd.concat(frames, ignore_index=True)
    history["snapshot_date"] = pd.to_datetime(history["snapshot_date"], utc=True)
    history["date_added"] = pd.to_datetime(history["date_added"], utc=True)
    history["last_updated"] = pd.to_datetime(history["last_updated"], utc=True)
    return history.sort_values(["snapshot_date", "rank", "cmc_id"]).reset_index(drop=True)


def build_monthly_dynamic_universe(
    snapshots: pd.DataFrame,
    metadata: pd.DataFrame,
    universe_cfg: Mapping,
    analysis_cfg: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis_start = _utc_timestamp(analysis_cfg["start"])
    analysis_end = _utc_timestamp(analysis_cfg["end"])
    month_starts = pd.date_range(
        _month_start(analysis_start),
        _month_start(analysis_end),
        freq="MS",
    )
    metadata_lookup = metadata[["cmc_id", "tags_json", "excluded_by_metadata_tag"]]
    audit_frames = []
    membership_frames = []
    peg_cfg = universe_cfg["peg_filter"]
    slug_terms = [str(term).lower() for term in universe_cfg["excluded_slug_terms"]]

    for month_start in month_starts:
        formation_date = month_start - pd.Timedelta(days=1)
        candidates = snapshots.loc[
            snapshots["snapshot_date"].eq(formation_date)
            & snapshots["rank"].le(int(universe_cfg["top_n"])),
            [
                "cmc_id",
                "name",
                "symbol",
                "slug",
                "rank",
                "price_usd",
                "market_cap_usd",
            ],
        ].copy()
        if candidates.empty:
            raise ValueError(f"No CMC formation snapshot for {formation_date.date()}")
        candidates = candidates.merge(metadata_lookup, on="cmc_id", how="left")
        candidates["excluded_by_metadata_tag"] = candidates[
            "excluded_by_metadata_tag"
        ].fillna(False)
        normalized_text = (
            candidates["name"].fillna("").str.lower()
            + " "
            + candidates["slug"].fillna("").str.lower()
        )
        candidates["excluded_by_slug"] = False
        for term in slug_terms:
            candidates["excluded_by_slug"] |= normalized_text.str.contains(
                term, regex=False
            )
        candidates = candidates.merge(
            identify_point_in_time_pegs(
                snapshots,
                candidates["cmc_id"],
                formation_date,
                peg_cfg,
            ),
            on="cmc_id",
            how="left",
        )
        candidates["excluded_by_peg_rule"] = candidates[
            "excluded_by_peg_rule"
        ].fillna(False)
        candidates["excluded_by_market_cap"] = candidates["market_cap_usd"].lt(
            float(universe_cfg["minimum_formation_market_cap_usd"])
        )
        candidates["selected"] = ~(
            candidates["excluded_by_metadata_tag"]
            | candidates["excluded_by_slug"]
            | candidates["excluded_by_peg_rule"]
            | candidates["excluded_by_market_cap"]
        )
        reason_columns = [
            ("excluded_by_metadata_tag", "metadata_tag"),
            ("excluded_by_slug", "slug_rule"),
            ("excluded_by_peg_rule", "peg_rule"),
            ("excluded_by_market_cap", "market_cap_floor"),
        ]
        candidates["exclusion_reason"] = candidates.apply(
            lambda row: "|".join(label for column, label in reason_columns if row[column]),
            axis=1,
        )
        candidates["month_start"] = month_start
        candidates["formation_date"] = formation_date
        candidates["metadata_available"] = candidates["tags_json"].notna()
        audit_frames.append(candidates)

        selected = candidates.loc[candidates["selected"]].copy()
        if len(selected) < int(analysis_cfg["minimum_active_assets"]):
            raise ValueError(
                f"Selected CMC universe for {month_start.date()} has only {len(selected)} assets"
            )
        selected["formation_universe_size"] = len(selected)
        membership_frames.append(selected)

    audit = pd.concat(audit_frames, ignore_index=True)
    membership = pd.concat(membership_frames, ignore_index=True)
    turnover = build_universe_turnover(membership)
    return membership, audit, turnover


def identify_point_in_time_pegs(
    snapshots: pd.DataFrame,
    cmc_ids: Sequence[int] | pd.Series,
    formation_date: object,
    peg_cfg: Mapping,
) -> pd.DataFrame:
    end = _utc_timestamp(formation_date)
    start = end - pd.Timedelta(days=int(peg_cfg["trailing_calendar_days"]) - 1)
    subset = snapshots.loc[
        snapshots["cmc_id"].isin(list(cmc_ids))
        & snapshots["snapshot_date"].between(start, end),
        ["cmc_id", "price_usd"],
    ].copy()
    stats = subset.groupby("cmc_id")["price_usd"].agg(
        peg_observations="count",
        peg_median_price="median",
        peg_min_price="min",
        peg_max_price="max",
    )
    stats["peg_price_ratio"] = stats["peg_max_price"] / stats["peg_min_price"]
    stats["excluded_by_peg_rule"] = (
        stats["peg_observations"].ge(int(peg_cfg["minimum_observations"]))
        & stats["peg_median_price"].between(
            float(peg_cfg["median_price_lower_usd"]),
            float(peg_cfg["median_price_upper_usd"]),
        )
        & stats["peg_price_ratio"].le(float(peg_cfg["maximum_price_ratio"]))
    )
    return stats.reset_index()


def build_universe_turnover(membership: pd.DataFrame) -> pd.DataFrame:
    rows = []
    previous: set[int] | None = None
    for month_start, subset in membership.groupby("month_start", sort=True):
        current = set(subset["cmc_id"].astype(int))
        if previous is None:
            entered = len(current)
            exited = 0
            jaccard = np.nan
        else:
            entered = len(current.difference(previous))
            exited = len(previous.difference(current))
            union = current.union(previous)
            jaccard = len(current.intersection(previous)) / len(union) if union else np.nan
        rows.append(
            {
                "month_start": month_start,
                "universe_size": len(current),
                "entered": entered,
                "exited": exited,
                "jaccard_similarity": jaccard,
            }
        )
        previous = current
    return pd.DataFrame(rows)


def build_daily_research_panel(
    snapshots: pd.DataFrame,
    membership: pd.DataFrame,
    analysis_cfg: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    working = snapshots.sort_values(["cmc_id", "snapshot_date"]).copy()
    grouped = working.groupby("cmc_id", sort=False)
    working["previous_date"] = grouped["snapshot_date"].shift(1)
    working["previous_price_usd"] = grouped["price_usd"].shift(1)
    working["lagged_market_cap_usd"] = grouped["market_cap_usd"].shift(1)
    exact_previous_day = working["snapshot_date"].sub(working["previous_date"]).eq(
        pd.Timedelta(days=1)
    )
    positive_prices = working["price_usd"].gt(0) & working["previous_price_usd"].gt(0)
    valid_price_ratio = (
        working["price_usd"].where(positive_prices)
        / working["previous_price_usd"].where(positive_prices)
    )
    working["asset_return"] = np.log(valid_price_ratio).where(exact_previous_day)
    working["lagged_market_cap_usd"] = working["lagged_market_cap_usd"].where(
        exact_previous_day & working["lagged_market_cap_usd"].gt(0)
    )

    start = _utc_timestamp(analysis_cfg["start"])
    end = _utc_timestamp(analysis_cfg["end"])
    working = working.loc[working["snapshot_date"].between(start, end)].copy()
    working["month_start"] = _month_start_series(working["snapshot_date"])
    membership_columns = [
        "month_start",
        "cmc_id",
        "formation_date",
        "formation_universe_size",
        "rank",
        "market_cap_usd",
    ]
    member_rows = working.merge(
        membership[membership_columns].rename(
            columns={
                "rank": "formation_rank",
                "market_cap_usd": "formation_market_cap_usd",
            }
        ),
        on=["month_start", "cmc_id"],
        how="inner",
        validate="many_to_one",
    )
    member_rows["valid_return_weight"] = (
        member_rows["asset_return"].notna()
        & member_rows["lagged_market_cap_usd"].gt(0)
    )
    coverage = _build_daily_coverage(member_rows, membership, analysis_cfg)
    eligible_dates = set(
        coverage.loc[coverage["eligible_day"], "snapshot_date"].tolist()
    )
    member_rows["eligible_day"] = member_rows["snapshot_date"].isin(eligible_dates)
    valid = member_rows.loc[
        member_rows["eligible_day"] & member_rows["valid_return_weight"]
    ].copy()
    valid["weighted_return"] = (
        valid["asset_return"] * valid["lagged_market_cap_usd"]
    )
    aggregates = valid.groupby("snapshot_date").agg(
        weighted_return_sum=("weighted_return", "sum"),
        weight_sum=("lagged_market_cap_usd", "sum"),
    )
    market_return = (
        aggregates["weighted_return_sum"] / aggregates["weight_sum"]
    ).rename("market_return")
    valid = valid.merge(market_return, left_on="snapshot_date", right_index=True)
    valid["absolute_deviation"] = (
        valid["asset_return"] - valid["market_return"]
    ).abs()
    csad = valid.groupby("snapshot_date")["absolute_deviation"].mean().rename("csad")
    return_panel = member_rows.pivot(
        index="snapshot_date", columns="cmc_id", values="asset_return"
    ).sort_index()
    ineligible = return_panel.index.difference(pd.DatetimeIndex(sorted(eligible_dates)))
    return_panel.loc[ineligible] = np.nan
    return member_rows, return_panel, market_return, csad, coverage


def build_weekly_research_panel(
    daily_member_rows: pd.DataFrame,
    analysis_cfg: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    daily = daily_member_rows.loc[daily_member_rows["eligible_day"]].copy()
    daily["week_start"] = daily["snapshot_date"] - pd.to_timedelta(
        daily["snapshot_date"].dt.weekday, unit="D"
    )
    calendar = pd.date_range(
        _utc_timestamp(analysis_cfg["start"]),
        _utc_timestamp(analysis_cfg["end"]),
        freq="D",
    )
    calendar_frame = pd.DataFrame({"snapshot_date": calendar})
    calendar_frame["week_start"] = calendar_frame["snapshot_date"] - pd.to_timedelta(
        calendar_frame["snapshot_date"].dt.weekday, unit="D"
    )
    expected_days = calendar_frame.groupby("week_start").size().rename("expected_days")

    weekly = daily.groupby(["week_start", "cmc_id"], as_index=False).agg(
        name=("name", "last"),
        symbol=("symbol", "last"),
        weekly_return=("asset_return", lambda values: values.sum(min_count=1)),
        return_observations=("asset_return", "count"),
        first_lagged_market_cap_usd=("lagged_market_cap_usd", "first"),
    )
    weekly = weekly.merge(expected_days, on="week_start", how="left")
    weekly["minimum_required_days"] = np.ceil(
        weekly["expected_days"] * float(analysis_cfg["minimum_membership_coverage"])
    ).astype(int)
    weekly["valid_return_weight"] = (
        weekly["return_observations"].ge(weekly["minimum_required_days"])
        & weekly["weekly_return"].notna()
        & weekly["first_lagged_market_cap_usd"].gt(0)
    )
    coverage = weekly.groupby("week_start").agg(
        active_assets=("valid_return_weight", "sum"),
        observed_assets=("cmc_id", "nunique"),
    ).reset_index()
    coverage["eligible_week"] = coverage["active_assets"].ge(
        int(analysis_cfg["minimum_active_assets"])
    )
    eligible_weeks = set(coverage.loc[coverage["eligible_week"], "week_start"])
    weekly["eligible_week"] = weekly["week_start"].isin(eligible_weeks)
    valid = weekly.loc[weekly["eligible_week"] & weekly["valid_return_weight"]].copy()
    valid["weighted_return"] = (
        valid["weekly_return"] * valid["first_lagged_market_cap_usd"]
    )
    aggregates = valid.groupby("week_start").agg(
        weighted_return_sum=("weighted_return", "sum"),
        weight_sum=("first_lagged_market_cap_usd", "sum"),
    )
    market_return = (
        aggregates["weighted_return_sum"] / aggregates["weight_sum"]
    ).rename("market_return")
    valid = valid.merge(market_return, left_on="week_start", right_index=True)
    valid["absolute_deviation"] = (
        valid["weekly_return"] - valid["market_return"]
    ).abs()
    csad = valid.groupby("week_start")["absolute_deviation"].mean().rename("csad")
    return_panel = weekly.pivot(
        index="week_start", columns="cmc_id", values="weekly_return"
    ).sort_index()
    ineligible = return_panel.index.difference(pd.DatetimeIndex(sorted(eligible_weeks)))
    return_panel.loc[ineligible] = np.nan
    return weekly, return_panel, market_return, csad, coverage


def validate_analysis_quality(
    manifest: pd.DataFrame,
    membership: pd.DataFrame,
    daily_coverage: pd.DataFrame,
    source_cfg: Mapping,
    analysis_cfg: Mapping,
) -> pd.DataFrame:
    expected_snapshots = len(requested_snapshot_dates(source_cfg))
    minimum_assets = int(analysis_cfg["minimum_active_assets"])
    eligible_share = float(daily_coverage["eligible_day"].mean())
    checks = pd.DataFrame(
        [
            {
                "check": "snapshot_completion",
                "observed": len(manifest) / expected_snapshots,
                "required": 1.0,
                "passes": len(manifest) == expected_snapshots,
            },
            {
                "check": "minimum_monthly_universe",
                "observed": membership.groupby("month_start")["cmc_id"].nunique().min(),
                "required": minimum_assets,
                "passes": membership.groupby("month_start")["cmc_id"].nunique().min()
                >= minimum_assets,
            },
            {
                "check": "eligible_daily_share",
                "observed": eligible_share,
                "required": float(analysis_cfg["minimum_eligible_day_share"]),
                "passes": eligible_share
                >= float(analysis_cfg["minimum_eligible_day_share"]),
            },
        ]
    )
    if not bool(checks["passes"].all()):
        failures = checks.loc[~checks["passes"], "check"].tolist()
        raise ValueError(f"CMC analysis quality gates failed: {', '.join(failures)}")
    return checks


def run_dynamic_csad_regressions(
    series_by_frequency: Mapping[str, tuple[pd.Series, pd.Series]],
    analysis_cfg: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regression_cfg = analysis_cfg["regression"]
    target_rows = []
    coefficient_frames = []
    diagnostic_frames = []
    for period in analysis_cfg["subperiods"]:
        start = _utc_timestamp(period["start"])
        end = _utc_timestamp(period["end"]) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        for frequency in analysis_cfg["frequencies"]:
            csad, market_return = series_by_frequency[frequency]
            period_csad = csad.loc[(csad.index >= start) & (csad.index <= end)]
            period_market = market_return.loc[
                (market_return.index >= start) & (market_return.index <= end)
            ]
            for model_name in regression_cfg["models"]:
                model_fn, target_term = MODEL_SPECS[model_name]
                coefficients, diagnostics, _, model, _ = model_fn(
                    period_csad,
                    period_market,
                    cov_type=regression_cfg["cov_type"],
                    hac_maxlags=regression_cfg["hac_maxlags"],
                )
                coefficient_frame = coefficients.reset_index()
                coefficient_frame["period"] = period["name"]
                coefficient_frame["frequency"] = frequency
                coefficient_frame["model"] = model_name
                coefficient_frames.append(coefficient_frame)
                diagnostic_frame = diagnostics.copy()
                diagnostic_frame["period"] = period["name"]
                diagnostic_frame["frequency"] = frequency
                diagnostic_frame["model"] = model_name
                diagnostic_frames.append(diagnostic_frame)
                ci = model.conf_int().loc[target_term]
                target_rows.append(
                    {
                        "period": period["name"],
                        "frequency": frequency,
                        "model": model_name,
                        "target_term": target_term,
                        "coefficient": float(model.params[target_term]),
                        "std_error": float(model.bse[target_term]),
                        "t_stat": float(model.tvalues[target_term]),
                        "p_value": float(model.pvalues[target_term]),
                        "ci_lower": float(ci.iloc[0]),
                        "ci_upper": float(ci.iloc[1]),
                        "nobs": int(model.nobs),
                        "rsquared": float(model.rsquared),
                    }
                )
    targets = pd.DataFrame(target_rows)
    expected_size = int(regression_cfg["family_size_per_period"])
    for period_name, indices in targets.groupby("period").groups.items():
        if len(indices) != expected_size:
            raise ValueError(
                f"CMC regression family {period_name} has {len(indices)} tests, expected {expected_size}"
            )
        targets.loc[indices, "q_value_bh_fdr"] = benjamini_hochberg(
            targets.loc[indices, "p_value"]
        )
    targets["negative_target"] = targets["coefficient"].lt(0)
    targets["supports_herding"] = (
        targets["negative_target"]
        & targets["q_value_bh_fdr"].le(float(regression_cfg["fdr_alpha"]))
    )
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    return targets, coefficients, diagnostics


def build_paper_benchmark_comparison(
    targets: pd.DataFrame,
    benchmark_cfg: Mapping,
) -> pd.DataFrame:
    paper = pd.DataFrame(benchmark_cfg["values"]).rename(
        columns={
            "coefficient": "paper_coefficient",
            "t_stat": "paper_t_stat",
            "nobs": "paper_nobs",
        }
    )
    ours = targets.rename(
        columns={
            "coefficient": "our_coefficient",
            "t_stat": "our_t_stat",
            "nobs": "our_nobs",
        }
    )
    columns = [
        "period",
        "frequency",
        "model",
        "our_coefficient",
        "our_t_stat",
        "our_nobs",
        "q_value_bh_fdr",
        "supports_herding",
    ]
    comparison = paper.merge(
        ours[columns],
        on=["period", "frequency", "model"],
        how="left",
        validate="one_to_one",
    )
    comparison["coefficient_difference"] = (
        comparison["our_coefficient"] - comparison["paper_coefficient"]
    )
    comparison["coefficient_sign_matches"] = np.sign(
        comparison["our_coefficient"]
    ).eq(np.sign(comparison["paper_coefficient"]))
    comparison["paper_significant_5pct"] = comparison["paper_t_stat"].abs().ge(1.96)
    comparison["paper_herding_support"] = (
        comparison["paper_coefficient"].lt(0)
        & comparison["paper_significant_5pct"]
    )
    return comparison


def build_cmc_dynamic_report(
    config: Mapping,
    manifest: pd.DataFrame,
    metadata: pd.DataFrame,
    membership: pd.DataFrame,
    turnover: pd.DataFrame,
    daily_coverage: pd.DataFrame,
    weekly_coverage: pd.DataFrame,
    quality_checks: pd.DataFrame,
    targets: pd.DataFrame,
    benchmark_comparison: pd.DataFrame,
    daily_csad: pd.Series,
    weekly_csad: pd.Series,
    plot_paths: Sequence[str],
) -> str:
    full = targets.loc[targets["period"].eq("full_sample")]
    full_support = int(full["supports_herding"].sum())
    model_agreement = full.groupby("frequency")["supports_herding"].all()
    lines = [
        "# CMC 동적 Universe Herding 재현 보고서",
        "",
        "## 한 문장 결론",
        "",
    ]
    if full_support == 0:
        lines.append(
            "> Full-sample daily·weekly 6개 사전 검정에서 BH-FDR를 통과한 음의 herding 계수는 없습니다."
        )
    elif bool(model_agreement.all()):
        lines.append(
            "> Daily·weekly의 Standard, no-intercept, SCSAD가 모두 음수이고 BH-FDR를 통과해 모형 간 일치된 herding 근거를 보입니다."
        )
    else:
        lines.append(
            "> 일부 corrected specification에서만 herding 계수가 통과해 method-sensitive evidence로 해석합니다."
        )
    lines.extend(
        [
            "",
            "## 재현 설계",
            "",
            f"- CMC snapshot: {config['source']['source_start']} ~ {config['source']['source_end']}",
            f"- 분석: {config['analysis']['start']} ~ {config['analysis']['end']}",
            "- universe: 전월 말 Top-200, 월간 고정, $100M market-cap floor",
            "- 제외: stablecoin, wrapped token, liquid-staking derivative, point-in-time peg rule",
            "- market return: 전일 시가총액 가중",
            "- CSAD: 활성 자산 단순 평균 절대편차",
            "- inference: Newey-West HAC, period별 6-test BH-FDR",
            "",
            "## 데이터 품질",
            "",
            f"- snapshot: {len(manifest):,}일, {int(manifest['rows'].sum()):,}행",
            f"- 500행 미만 partial snapshot: {int(manifest['rows'].lt(500).sum()):,}일; "
            f"최소 {int(manifest['rows'].min()):,}행",
            f"- 최소 양수 USD quote 비율: {manifest['positive_quote_share'].min():.2%}",
            f"- current metadata: {len(metadata):,}자산",
            f"- 월별 universe: 최소 {membership.groupby('month_start')['cmc_id'].nunique().min():,}, "
            f"중앙 {membership.groupby('month_start')['cmc_id'].nunique().median():.0f}, "
            f"최대 {membership.groupby('month_start')['cmc_id'].nunique().max():,}",
            f"- 월별 교체: 평균 진입 {turnover['entered'].iloc[1:].mean():.1f}, "
            f"평균 이탈 {turnover['exited'].iloc[1:].mean():.1f}",
            f"- daily 적격일: {int(daily_coverage['eligible_day'].sum()):,}/"
            f"{len(daily_coverage):,} ({daily_coverage['eligible_day'].mean():.2%})",
            f"- weekly 적격주: {int(weekly_coverage['eligible_week'].sum()):,}/"
            f"{len(weekly_coverage):,} ({weekly_coverage['eligible_week'].mean():.2%})",
            "",
            "### Quality gates",
            "",
            "| check | observed | required | passes |",
            "|---|---:|---:|---|",
        ]
    )
    for row in quality_checks.itertuples(index=False):
        lines.append(
            f"| {row.check} | {row.observed:.4f} | {row.required:.4f} | {bool(row.passes)} |"
        )
    lines.extend(
        [
            "",
            "## Full-sample 핵심 결과",
            "",
            "| frequency | model | target | coefficient | HAC t | p | BH q | herding support |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in full.sort_values(["frequency", "model"]).itertuples(index=False):
        lines.append(
            f"| {row.frequency} | {row.model} | {row.target_term} | {row.coefficient:.6f} | "
            f"{row.t_stat:.3f} | {row.p_value:.4g} | {row.q_value_bh_fdr:.4g} | "
            f"{bool(row.supports_herding)} |"
        )
    lines.extend(
        [
            "",
            "## 선행논문 수치 대조",
            "",
            f"- 일별 평균 CSAD: 우리 {daily_csad.mean():.2%}, 논문 2.94%",
            f"- 주별 평균 CSAD: 우리 {weekly_csad.mean():.2%}, 논문 7.94%",
            "- 논문은 raw Newey-West 유의성, 우리는 period별 6-test BH-FDR까지 적용한 더 엄격한 판정입니다.",
            "",
            "| period | frequency | model | paper coef | our coef | paper t | our t | sign match |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in benchmark_comparison.itertuples(index=False):
        lines.append(
            f"| {row.period} | {row.frequency} | {row.model} | "
            f"{row.paper_coefficient:.3f} | {row.our_coefficient:.3f} | "
            f"{row.paper_t_stat:.3f} | {row.our_t_stat:.3f} | "
            f"{bool(row.coefficient_sign_matches)} |"
        )
    lines.extend(["", "## Subperiod 진단", ""])
    for period in ["pre_covid", "covid", "post_covid"]:
        subset = targets.loc[targets["period"].eq(period)]
        labels = subset.loc[subset["supports_herding"], ["frequency", "model"]]
        supported = ", ".join(labels["frequency"] + ":" + labels["model"])
        lines.append(
            f"- {period}: {len(labels)}/6 통과"
            + (f" ({supported})" if supported else "")
        )
    lines.extend(
        [
            "",
            "## 해석 제약",
            "",
            "- 이 분석은 CMC point-in-time dynamic extension이며 원논문 fixed 62-coin sample의 완전 복제가 아닙니다.",
            "- Current metadata tag로 과거 자산 유형을 분류하는 후행적 한계가 있어 point-in-time peg rule을 병행했습니다.",
            "- CMC public endpoint의 22개 partial snapshot을 임의 보간하지 않았고, 행 수·누락 rank를 manifest에 보존했습니다.",
            "- 음의 계수는 허딩의 필요조건으로만 해석하며 intentional imitation을 직접 증명하지 않습니다.",
            "- 이 연구는 수익률 alpha 백테스트가 아닙니다.",
            "",
            "## 그림",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in plot_paths)
    lines.append("")
    return "\n".join(lines)


def plot_universe_diagnostics(turnover: pd.DataFrame, path: str | Path) -> None:
    plt = _get_pyplot()
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(turnover["month_start"], turnover["universe_size"], color="#1f5a75")
    axes[0].set_ylabel("Universe size")
    axes[0].grid(alpha=0.25)
    axes[1].plot(turnover["month_start"], turnover["entered"], label="entered", color="#d0773c")
    axes[1].plot(turnover["month_start"], turnover["exited"], label="exited", color="#8a4f3d")
    axes[1].set_ylabel("Monthly assets")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    figure.suptitle("CMC point-in-time dynamic universe")
    figure.tight_layout()
    _save_figure(figure, path, plt)


def plot_csad_series(
    daily_csad: pd.Series,
    daily_market: pd.Series,
    path: str | Path,
) -> None:
    plt = _get_pyplot()
    frame = pd.concat(
        [daily_csad.rename("csad"), daily_market.abs().rename("abs_market_return")],
        axis=1,
    ).dropna()
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.scatter(
        frame["abs_market_return"],
        frame["csad"],
        s=8,
        alpha=0.35,
        color="#1f5a75",
    )
    axis.set_xlabel("Absolute market return")
    axis.set_ylabel("CSAD")
    axis.set_title("CMC dynamic universe: daily dispersion")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, path, plt)


def plot_target_coefficients(targets: pd.DataFrame, path: str | Path) -> None:
    plt = _get_pyplot()
    full = targets.loc[targets["period"].eq("full_sample")].copy()
    full["label"] = full["frequency"] + "\n" + full["model"]
    figure, axis = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(full))
    values = full["coefficient"].to_numpy(dtype=float)
    lower = full["ci_lower"].to_numpy(dtype=float)
    upper = full["ci_upper"].to_numpy(dtype=float)
    colors = np.where(full["supports_herding"], "#a33a2b", "#1f5a75")
    for position, value, low, high, color in zip(x, values, lower, upper, colors):
        axis.errorbar(
            position,
            value,
            yerr=[[value - low], [high - value]],
            fmt="o",
            color=color,
            capsize=4,
        )
    axis.axhline(0, color="#333333", linewidth=1)
    axis.set_xticks(x, full["label"])
    axis.set_ylabel("Target coefficient")
    axis.set_title("Full-sample corrected CSAD target coefficients")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, path, plt)


def _download_snapshot_checkpoint(date: pd.Timestamp, source_cfg: Mapping) -> Path:
    params = {
        "date": date.strftime("%Y-%m-%d"),
        "start": 1,
        "limit": int(source_cfg["retrieval_limit"]),
        "convertId": int(source_cfg["usd_convert_id"]),
    }
    payload = _request_json(source_cfg["endpoint"], params, source_cfg)
    frame = parse_historical_snapshot_payload(
        payload,
        date,
        usd_convert_id=int(source_cfg["usd_convert_id"]),
    )
    validate_snapshot_frame(frame, date, source_cfg)
    destination = snapshot_cache_path(source_cfg["cache_dir"], date)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_parquet_write(frame, destination)
    return destination


def _request_json(url: str, params: Mapping, source_cfg: Mapping) -> dict:
    maximum_attempts = int(source_cfg["maximum_attempts"])
    for attempt in range(1, maximum_attempts + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers={"User-Agent": str(source_cfg["user_agent"])},
                timeout=float(source_cfg["request_timeout_seconds"]),
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(f"HTTP {response.status_code}")
            response.raise_for_status()
            payload = response.json()
            status = payload.get("status", {})
            if str(status.get("error_code", "0")) != "0":
                raise ValueError(status.get("error_message", "CMC payload error"))
            time.sleep(float(source_cfg["minimum_request_interval_seconds"]))
            return payload
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            if attempt >= maximum_attempts:
                raise RuntimeError(
                    f"CMC request failed after {maximum_attempts} attempts: {exc}"
                ) from exc
            time.sleep(float(source_cfg["backoff_seconds"]) * (2 ** (attempt - 1)))
    raise AssertionError("unreachable")


def _valid_cached_snapshot(path: Path, date: object, source_cfg: Mapping) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_parquet(path)
        validate_snapshot_frame(frame, date, source_cfg)
    except (OSError, ValueError, TypeError):
        return False
    return True


def _write_collection_state(
    source_cfg: Mapping,
    dates: Sequence[pd.Timestamp],
    completed: Sequence[pd.Timestamp],
    failures: Sequence[dict],
    status: str,
) -> None:
    destination = Path(source_cfg["state_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "expected_dates": len(dates),
        "completed_dates": len(set(pd.Timestamp(date) for date in completed)),
        "failed_dates": list(failures),
        "source_start": str(source_cfg["source_start"]),
        "source_end": str(source_cfg["source_end"]),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_text_write(json.dumps(payload, ensure_ascii=False, indent=2), destination)


def _build_daily_coverage(
    member_rows: pd.DataFrame,
    membership: pd.DataFrame,
    analysis_cfg: Mapping,
) -> pd.DataFrame:
    start = _utc_timestamp(analysis_cfg["start"])
    end = _utc_timestamp(analysis_cfg["end"])
    calendar = pd.DataFrame({"snapshot_date": pd.date_range(start, end, freq="D")})
    calendar["month_start"] = _month_start_series(calendar["snapshot_date"])
    expected = membership.groupby("month_start")["cmc_id"].nunique().rename(
        "formation_universe_size"
    )
    observed = member_rows.groupby("snapshot_date").agg(
        observed_members=("cmc_id", "nunique"),
        active_assets=("valid_return_weight", "sum"),
    )
    coverage = calendar.merge(expected, on="month_start", how="left").merge(
        observed, left_on="snapshot_date", right_index=True, how="left"
    )
    coverage[["observed_members", "active_assets"]] = coverage[
        ["observed_members", "active_assets"]
    ].fillna(0)
    coverage["membership_coverage"] = (
        coverage["active_assets"] / coverage["formation_universe_size"]
    )
    coverage["eligible_day"] = (
        coverage["active_assets"].ge(int(analysis_cfg["minimum_active_assets"]))
        & coverage["membership_coverage"].ge(
            float(analysis_cfg["minimum_membership_coverage"])
        )
    )
    return coverage


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _safe_float(value: object) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return converted if math.isfinite(converted) else float("nan")


def _month_start(value: object) -> pd.Timestamp:
    timestamp = _utc_timestamp(value)
    return pd.Timestamp(
        year=timestamp.year,
        month=timestamp.month,
        day=1,
        tz="UTC",
    )


def _month_start_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.dt.strftime("%Y-%m-01"), utc=True)


def _atomic_parquet_write(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, destination)


def _atomic_text_write(content: str, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, destination)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_pyplot():
    import matplotlib.pyplot as plt

    return plt


def _save_figure(figure, path: str | Path, plt) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

LOGGER = logging.getLogger(__name__)
RECENT_SEARCH_URL = "https://api.x.com/2/tweets/search/recent"


def load_x_cache(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    if not resolved.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(resolved)
    if "created_at" in frame.columns:
        frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
    return frame


def save_x_cache(frame: pd.DataFrame, path: str | Path) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(resolved, index=False)


def load_x_state(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        return {}
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_x_state(state: dict[str, Any], path: str | Path) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)


def _request_json(url: str, bearer_token: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {bearer_token}",
            "User-Agent": "crypto-herding-x-research/1.0",
        },
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _to_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _normalize_posts(payload: dict[str, Any], query_name: str, symbol: str, query_text: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in payload.get("data", []) or []:
        metrics = item.get("public_metrics", {}) or {}
        rows.append(
            {
                "query_name": query_name,
                "symbol": symbol,
                "query_text": query_text,
                "post_id": str(item.get("id")),
                "author_id": str(item.get("author_id")) if item.get("author_id") is not None else None,
                "created_at": item.get("created_at"),
                "lang": item.get("lang"),
                "text": item.get("text", ""),
                "like_count": int(metrics.get("like_count", 0) or 0),
                "reply_count": int(metrics.get("reply_count", 0) or 0),
                "repost_count": int(metrics.get("retweet_count", 0) or 0),
                "quote_count": int(metrics.get("quote_count", 0) or 0),
                "impression_count": int(metrics.get("impression_count", 0) or 0),
                "bookmark_count": int(metrics.get("bookmark_count", 0) or 0),
                "post_url": f"https://x.com/i/web/status/{item.get('id')}",
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
    return frame


def _build_recent_search_params(
    query_cfg: dict[str, Any],
    collection_cfg: dict[str, Any],
    since_id: str | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "query": str(query_cfg["query"]),
        "max_results": int(collection_cfg.get("max_results_per_request", 100)),
        "tweet.fields": "id,author_id,created_at,lang,public_metrics,text",
    }

    if since_id:
        params["since_id"] = since_id
        return params

    end_value = collection_cfg.get("end")
    if end_value not in {None, "", "latest", "now"}:
        end_ts = _to_utc_timestamp(end_value)
    else:
        end_ts = pd.Timestamp.now(tz="UTC").floor("min")
    params["end_time"] = end_ts.isoformat().replace("+00:00", "Z")

    if collection_cfg.get("start") not in {None, ""}:
        start_ts = _to_utc_timestamp(collection_cfg["start"])
    else:
        lookback_days = int(collection_cfg.get("initial_lookback_days", 7))
        start_ts = end_ts - pd.Timedelta(days=lookback_days)
    params["start_time"] = start_ts.isoformat().replace("+00:00", "Z")
    return params


def collect_recent_search_posts(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    collection_cfg = dict(config["collection"])
    cache_path = Path(collection_cfg["cache_path"])
    state_path = Path(collection_cfg["state_path"])
    bearer_token = os.environ.get(str(collection_cfg.get("bearer_token_env", "X_BEARER_TOKEN")), "")
    cache_frame = load_x_cache(cache_path)

    if not bearer_token:
        LOGGER.warning("X_BEARER_TOKEN이 없어 API 수집을 건너뜁니다. 로컬 캐시만 사용합니다.")
        skipped = pd.DataFrame(
            [
                {
                    "query_name": item["name"],
                    "symbol": item["symbol"],
                    "request_count": 0,
                    "new_posts": 0,
                    "status": "skipped_no_token",
                }
                for item in collection_cfg.get("queries", [])
            ]
        )
        return cache_frame, skipped, load_x_state(state_path)

    state = load_x_state(state_path)
    logs: list[dict[str, Any]] = []
    collected_frames: list[pd.DataFrame] = []
    pause_seconds = float(collection_cfg.get("request_pause_seconds", 0.5))

    for query_cfg in collection_cfg.get("queries", []):
        query_name = str(query_cfg["name"])
        symbol = str(query_cfg["symbol"]).upper()
        prior_state = state.get(query_name, {})
        since_id = prior_state.get("newest_id")
        base_params = _build_recent_search_params(query_cfg, collection_cfg, since_id)

        newest_id = None
        next_token = None
        request_count = 0
        new_posts = 0
        max_pages = int(collection_cfg.get("max_pages_per_query", 5))

        while request_count < max_pages:
            params = dict(base_params)
            if next_token:
                params["next_token"] = next_token

            url = f"{RECENT_SEARCH_URL}?{urlencode(params)}"
            payload = _request_json(url, bearer_token)
            request_count += 1

            frame = _normalize_posts(
                payload=payload,
                query_name=query_name,
                symbol=symbol,
                query_text=str(query_cfg["query"]),
            )
            if not frame.empty:
                new_posts += len(frame)
                collected_frames.append(frame)

            meta = payload.get("meta", {}) or {}
            if newest_id is None:
                newest_id = meta.get("newest_id")
            next_token = meta.get("next_token")
            if not next_token:
                break
            time.sleep(pause_seconds)

        state[query_name] = {
            "symbol": symbol,
            "query": str(query_cfg["query"]),
            "newest_id": newest_id or prior_state.get("newest_id"),
            "updated_at_utc": str(pd.Timestamp.now(tz="UTC")),
        }
        logs.append(
            {
                "query_name": query_name,
                "symbol": symbol,
                "request_count": request_count,
                "new_posts": new_posts,
                "status": "ok",
                "since_id_used": since_id,
                "newest_id": newest_id,
            }
        )
        time.sleep(pause_seconds)

    if collected_frames:
        fresh = pd.concat(collected_frames, axis=0, ignore_index=True)
        combined = pd.concat([cache_frame, fresh], axis=0, ignore_index=True)
        combined = combined.drop_duplicates(subset=["post_id"], keep="last").sort_values("created_at")
        save_x_cache(combined, cache_path)
    else:
        combined = cache_frame

    save_x_state(state, state_path)
    return combined.reset_index(drop=True), pd.DataFrame(logs), state

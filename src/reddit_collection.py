from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

LOGGER = logging.getLogger(__name__)
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; HerdingResearchBot/1.0; +https://www.reddit.com/dev/api/)",
}


def _normalize_asset(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    return text.replace("/", "").replace("-", "").replace("_", "")


def build_reddit_search_url(
    *,
    subreddit: str,
    query: str,
    sort: str,
    time_filter: str,
    limit: int,
    after: str | None,
    result_type: str = "link",
) -> str:
    params = {
        "q": query,
        "restrict_sr": "on",
        "sort": sort,
        "t": time_filter,
        "limit": int(limit),
        "type": result_type,
    }
    if after:
        params["after"] = after
    return f"https://www.reddit.com/r/{subreddit}/search.json?{urlencode(params)}"


def _fetch_reddit_json(url: str, timeout: int) -> dict:
    request = Request(url, headers=DEFAULT_HEADERS)
    with urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8", errors="ignore")
    return json.loads(payload)


def fetch_reddit_query(
    *,
    query_name: str,
    subreddit: str,
    query: str,
    asset: str | None,
    sort: str,
    time_filter: str,
    limit: int,
    max_pages: int,
    timeout: int,
    sleep_seconds: float,
) -> tuple[pd.DataFrame, int]:
    rows: list[dict] = []
    request_count = 0
    after: str | None = None

    for _ in range(max(int(max_pages), 1)):
        request_count += 1
        url = build_reddit_search_url(
            subreddit=subreddit,
            query=query,
            sort=sort,
            time_filter=time_filter,
            limit=limit,
            after=after,
        )
        data = _fetch_reddit_json(url, timeout=timeout)
        listing = data.get("data", {})
        children = listing.get("children", [])
        if not children:
            break

        for child in children:
            post = dict(child.get("data", {}))
            title = str(post.get("title", "") or "").strip()
            created_utc = post.get("created_utc")
            permalink = str(post.get("permalink", "") or "").strip()
            if not title or created_utc in {None, ""}:
                continue
            timestamp = pd.to_datetime(float(created_utc), unit="s", utc=True, errors="coerce")
            if pd.isna(timestamp):
                continue
            rows.append(
                {
                    "timestamp": timestamp,
                    "source": f"reddit:r/{subreddit}",
                    "asset": _normalize_asset(asset),
                    "subreddit": subreddit,
                    "headline": title,
                    "title": title,
                    "author": post.get("author"),
                    "score": int(post.get("score", 0) or 0),
                    "num_comments": int(post.get("num_comments", 0) or 0),
                    "permalink": f"https://www.reddit.com{permalink}" if permalink else None,
                    "query_name": query_name,
                    "query_text": query,
                    "post_fullname": post.get("name"),
                    "post_id": post.get("id"),
                    "collected_at_utc": pd.Timestamp.now(tz="UTC"),
                    "collection_source": "reddit_public_json",
                }
            )

        after = listing.get("after")
        if not after:
            break
        if sleep_seconds > 0:
            time.sleep(float(sleep_seconds))

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, request_count
    dedupe_cols = ["timestamp", "subreddit", "headline", "permalink"]
    frame = frame.drop_duplicates(subset=dedupe_cols, keep="first").sort_values("timestamp")
    return frame.reset_index(drop=True), request_count


def load_existing_reddit(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    if not resolved.exists():
        return pd.DataFrame(
            columns=[
                "timestamp",
                "source",
                "asset",
                "subreddit",
                "headline",
                "title",
                "author",
                "score",
                "num_comments",
                "permalink",
                "query_name",
                "query_text",
                "post_fullname",
                "post_id",
                "collected_at_utc",
                "collection_source",
            ]
        )

    frame = pd.read_csv(resolved)
    for column in ["timestamp", "collected_at_utc"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if "asset" in frame.columns:
        frame["asset"] = frame["asset"].apply(_normalize_asset)
    return frame


def collect_reddit_posts(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    collection_cfg = dict(config.get("reddit_collection", {}))
    queries = list(collection_cfg.get("queries", []))
    if not queries:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[pd.DataFrame] = []
    logs: list[dict] = []
    for query_cfg in queries:
        query_name = str(query_cfg["name"])
        subreddit = str(query_cfg["subreddit"])
        query = str(query_cfg["query"])
        asset = _normalize_asset(query_cfg.get("asset"))
        try:
            frame, request_count = fetch_reddit_query(
                query_name=query_name,
                subreddit=subreddit,
                query=query,
                asset=asset,
                sort=str(collection_cfg.get("sort", "new")),
                time_filter=str(collection_cfg.get("time_filter", "year")),
                limit=int(collection_cfg.get("limit", 100)),
                max_pages=int(collection_cfg.get("max_pages", 3)),
                timeout=int(collection_cfg.get("timeout_seconds", 60)),
                sleep_seconds=float(collection_cfg.get("sleep_seconds", 1.0)),
            )
            rows.append(frame)
            logs.append(
                {
                    "query_name": query_name,
                    "subreddit": subreddit,
                    "asset": asset,
                    "query": query,
                    "post_count": int(len(frame)),
                    "request_count": int(request_count),
                    "status": "ok",
                }
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Reddit 수집 실패: %s | %s", query_name, exc)
            logs.append(
                {
                    "query_name": query_name,
                    "subreddit": subreddit,
                    "asset": asset,
                    "query": query,
                    "post_count": 0,
                    "request_count": 0,
                    "status": f"error: {exc}",
                }
            )

    combined = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
    if not combined.empty:
        combined["asset"] = combined["asset"].apply(_normalize_asset)
        dedupe_cols = ["timestamp", "subreddit", "headline", "permalink"]
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="first").sort_values("timestamp")
    return combined.reset_index(drop=True), pd.DataFrame(logs)


def merge_and_save_reddit(existing: pd.DataFrame, fresh: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    combined = pd.concat([existing, fresh], axis=0, ignore_index=True) if not existing.empty or not fresh.empty else pd.DataFrame()
    if not combined.empty:
        if "asset" in combined.columns:
            combined["asset"] = combined["asset"].apply(_normalize_asset)
        for column in ["timestamp", "collected_at_utc"]:
            if column in combined.columns:
                combined[column] = pd.to_datetime(combined[column], utc=True, errors="coerce")
        dedupe_cols = ["timestamp", "subreddit", "headline", "permalink"]
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="last").sort_values("timestamp")

    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(resolved, index=False)
    return combined.reset_index(drop=True)


def build_reddit_collection_report(
    *,
    collection_log: pd.DataFrame,
    merged_posts: pd.DataFrame,
    output_paths: list[str],
) -> str:
    lines = ["# Reddit 데이터 수집 리포트", ""]
    lines.append("## 수집 결과")
    if collection_log.empty:
        lines.append("- 수집 로그가 없습니다.")
    else:
        for _, row in collection_log.iterrows():
            lines.append(
                f"- {row['query_name']} / r/{row['subreddit']} "
                f"({row['asset'] if pd.notna(row['asset']) else 'market'}): "
                f"{int(row['post_count'])}건, 요청 {int(row['request_count'])}회, 상태 {row['status']}"
            )
    lines.append("")

    lines.append("## 누적 파일 상태")
    lines.append(f"- 총 post 수: {len(merged_posts)}")
    if not merged_posts.empty:
        lines.append(f"- 구간: {merged_posts['timestamp'].min()} ~ {merged_posts['timestamp'].max()}")
        asset_counts = merged_posts["asset"].fillna("market").value_counts().to_dict()
        subreddit_counts = merged_posts["subreddit"].fillna("unknown").value_counts().to_dict()
        lines.append(f"- asset 분포: {asset_counts}")
        lines.append(f"- subreddit 분포: {subreddit_counts}")
    lines.append("")

    lines.append("## 출력물")
    for path in output_paths:
        lines.append(f"- {path}")
    return "\n".join(lines) + "\n"

from __future__ import annotations

import html
import json
import logging
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import pandas as pd

LOGGER = logging.getLogger(__name__)
DEFAULT_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; HerdingResearchBot/1.0; +https://gdeltproject.org/)",
}


def _normalize_asset(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    return text.replace("/", "").replace("-", "").replace("_", "")


def build_google_news_url(query: str, language: str, region: str, edition: str) -> str:
    return (
        "https://news.google.com/rss/search?"
        f"q={quote(query)}&hl={language}&gl={region}&ceid={edition}"
    )


def build_gdelt_doc_url(
    *,
    query: str,
    start_datetime: str | None,
    end_datetime: str | None,
    maxrecords: int,
    sort: str,
    mode: str = "artlist",
    output_format: str = "json",
    timespan: str | None = None,
) -> str:
    normalized_query = str(query).strip()
    if " OR " in normalized_query.upper() and not (
        normalized_query.startswith("(") and normalized_query.endswith(")")
    ):
        normalized_query = f"({normalized_query})"
    params = {
        "query": normalized_query,
        "mode": mode,
        "format": output_format,
        "maxrecords": int(maxrecords),
        "sort": sort,
    }
    if timespan:
        params["timespan"] = timespan
    else:
        params["startdatetime"] = start_datetime
        params["enddatetime"] = end_datetime
    return f"https://api.gdeltproject.org/api/v2/doc/doc?{urlencode(params)}"


def _parse_pubdate(value: str | None) -> pd.Timestamp | pd.NaT:
    if not value:
        return pd.NaT
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return pd.NaT
    if dt.tzinfo is None:
        return pd.Timestamp(dt).tz_localize("UTC")
    return pd.Timestamp(dt).tz_convert("UTC")


def _parse_gdelt_seendate(value: str | None) -> pd.Timestamp | pd.NaT:
    if not value:
        return pd.NaT
    try:
        return pd.to_datetime(str(value), format="%Y%m%dT%H%M%SZ", utc=True, errors="raise")
    except (TypeError, ValueError):
        return pd.NaT


def _fetch_text(url: str, timeout: int) -> str:
    request = Request(url, headers=DEFAULT_HTTP_HEADERS)
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


def _fetch_json(url: str, timeout: int, retries: int, retry_backoff_seconds: float) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            payload = _fetch_text(url, timeout=timeout)
            if not payload.strip().startswith("{"):
                raise json.JSONDecodeError("non-json-payload", payload, 0)
            return json.loads(payload)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:  # noqa: PERF203
            last_error = exc
            retryable = (
                (isinstance(exc, HTTPError) and exc.code in {429, 500, 502, 503, 504})
                or isinstance(exc, (URLError, TimeoutError, json.JSONDecodeError))
            )
            if attempt >= retries or not retryable:
                raise
            sleep_seconds = float(retry_backoff_seconds) * (attempt + 1)
            LOGGER.warning("GDELT 재시도 예정: %s | %.1fs 대기", exc, sleep_seconds)
            time.sleep(sleep_seconds)
    if last_error is not None:
        raise last_error
    raise RuntimeError("GDELT 요청 실패")


def _strip_title_source(title: str, source: str | None) -> str:
    headline = html.unescape(title or "").strip()
    if not headline:
        return headline
    if source:
        suffix = f" - {source}"
        if headline.endswith(suffix):
            headline = headline[: -len(suffix)].strip()
    return headline


def fetch_google_news_query(
    *,
    query_name: str,
    query: str,
    asset: str | None,
    language: str,
    region: str,
    edition: str,
    rss_source: str,
) -> pd.DataFrame:
    url = build_google_news_url(query=query, language=language, region=region, edition=edition)
    xml_text = urlopen(url, timeout=60).read().decode("utf-8", errors="ignore")
    root = ET.fromstring(xml_text)

    rows: list[dict] = []
    for item in root.findall("./channel/item"):
        title = item.findtext("title")
        link = item.findtext("link")
        guid = item.findtext("guid")
        pub_date = _parse_pubdate(item.findtext("pubDate"))
        source_elem = item.find("source")
        source = source_elem.text.strip() if source_elem is not None and source_elem.text else rss_source
        headline = _strip_title_source(title or "", source)
        if not headline or pd.isna(pub_date):
            continue
        rows.append(
            {
                "timestamp": pub_date,
                "source": source,
                "asset": _normalize_asset(asset),
                "headline": headline,
                "rss_source": rss_source,
                "query_name": query_name,
                "query_text": query,
                "guid": guid,
                "link": link,
                "collected_at_utc": pd.Timestamp.now(tz="UTC"),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("timestamp").reset_index(drop=True)


def _iter_gdelt_windows(
    *,
    end_timestamp: pd.Timestamp,
    lookback_days: int,
    window_days: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    end_ts = pd.Timestamp(end_timestamp).tz_convert("UTC")
    start_ts = end_ts - pd.Timedelta(days=int(lookback_days))
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start_ts
    while cursor < end_ts:
        window_end = min(cursor + pd.Timedelta(days=int(window_days)), end_ts)
        windows.append((cursor, window_end))
        cursor = window_end
    return windows


def _format_gdelt_timespan(lookback_days: int) -> str:
    if lookback_days % 30 == 0:
        return f"{max(1, lookback_days // 30)}months"
    if lookback_days % 7 == 0:
        return f"{max(1, lookback_days // 7)}weeks"
    return f"{max(1, lookback_days)}days"


def fetch_gdelt_query(
    *,
    query_name: str,
    query: str,
    asset: str | None,
    source_name: str,
    lookback_days: int,
    window_days: int,
    maxrecords: int,
    sort: str,
    timeout: int,
    retries: int,
    retry_backoff_seconds: float,
    sleep_seconds: float,
    end_timestamp: pd.Timestamp,
    request_strategy: str,
) -> tuple[pd.DataFrame, int]:
    rows: list[dict] = []
    request_count = 0
    strategy = str(request_strategy).strip().lower()
    if strategy == "timespan":
        windows = [(pd.NaT, pd.NaT)]
    else:
        windows = _iter_gdelt_windows(
            end_timestamp=end_timestamp,
            lookback_days=lookback_days,
            window_days=window_days,
        )

    for window_start, window_end in windows:
        request_count += 1
        url = build_gdelt_doc_url(
            query=query,
            start_datetime=None if strategy == "timespan" else window_start.strftime("%Y%m%d%H%M%S"),
            end_datetime=None if strategy == "timespan" else window_end.strftime("%Y%m%d%H%M%S"),
            maxrecords=maxrecords,
            sort=sort,
            timespan=_format_gdelt_timespan(lookback_days) if strategy == "timespan" else None,
        )
        data = _fetch_json(
            url=url,
            timeout=timeout,
            retries=retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        articles = list(data.get("articles", []))
        for article in articles:
            headline = html.unescape(str(article.get("title", "") or "")).strip()
            timestamp = _parse_gdelt_seendate(article.get("seendate"))
            link = str(article.get("url", "") or "").strip()
            domain = str(article.get("domain", "") or "").strip()
            if not headline or not link or pd.isna(timestamp):
                continue
            rows.append(
                {
                    "timestamp": timestamp,
                    "source": domain or source_name,
                    "asset": _normalize_asset(asset),
                    "headline": headline,
                    "rss_source": source_name,
                    "query_name": query_name,
                    "query_text": query,
                    "guid": link,
                    "link": link,
                    "collected_at_utc": pd.Timestamp.now(tz="UTC"),
                    "collection_source": "gdelt",
                    "language": article.get("language"),
                    "source_country": article.get("sourcecountry"),
                }
            )
        if sleep_seconds > 0:
            time.sleep(float(sleep_seconds))

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, request_count
    dedupe_cols = ["timestamp", "source", "headline", "link"]
    frame = frame.drop_duplicates(subset=dedupe_cols, keep="first").sort_values("timestamp")
    return frame.reset_index(drop=True), request_count


def load_existing_news(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    if not resolved.exists():
        return pd.DataFrame(
            columns=[
                "timestamp",
                "source",
                "asset",
                "headline",
                "rss_source",
                "query_name",
                "query_text",
                "guid",
                "link",
                "collected_at_utc",
                "collection_source",
                "language",
                "source_country",
            ]
        )

    frame = pd.read_csv(resolved)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if "collected_at_utc" in frame.columns:
        frame["collected_at_utc"] = pd.to_datetime(frame["collected_at_utc"], utc=True, errors="coerce")
    if "asset" in frame.columns:
        frame["asset"] = frame["asset"].apply(_normalize_asset)
    return frame


def collect_news_headlines(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    collection_cfg = dict(config["sentiment_extension"]["news_collection"])
    google_cfg = dict(collection_cfg.get("google_news", {}))
    gdelt_cfg = dict(collection_cfg.get("gdelt", {}))
    queries = list(collection_cfg.get("queries", []))
    end_timestamp = pd.Timestamp.now(tz="UTC")

    rows: list[pd.DataFrame] = []
    logs: list[dict] = []
    for query_cfg in queries:
        query_name = str(query_cfg["name"])
        query = str(query_cfg["query"])
        asset = _normalize_asset(query_cfg.get("asset"))
        if bool(google_cfg.get("enabled", True)):
            try:
                frame = fetch_google_news_query(
                    query_name=query_name,
                    query=query,
                    asset=asset,
                    language=str(google_cfg.get("language", "en-US")),
                    region=str(google_cfg.get("region", "US")),
                    edition=str(google_cfg.get("edition", "US:en")),
                    rss_source=str(query_cfg.get("rss_source", "google_news")),
                )
                if not frame.empty:
                    frame["collection_source"] = "google_news"
                rows.append(frame)
                logs.append(
                    {
                        "collection_source": "google_news",
                        "query_name": query_name,
                        "asset": asset,
                        "query": query,
                        "headline_count": int(len(frame)),
                        "request_count": 1,
                        "status": "ok",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("뉴스 수집 실패: %s | google | %s", query_name, exc)
                logs.append(
                    {
                        "collection_source": "google_news",
                        "query_name": query_name,
                        "asset": asset,
                        "query": query,
                        "headline_count": 0,
                        "request_count": 1,
                        "status": f"error: {exc}",
                    }
                )

        if bool(gdelt_cfg.get("enabled", False)):
            gdelt_query = str(query_cfg.get("gdelt_query", query))
            try:
                frame, request_count = fetch_gdelt_query(
                    query_name=query_name,
                    query=gdelt_query,
                    asset=asset,
                    source_name=str(query_cfg.get("gdelt_source", "gdelt")),
                    lookback_days=int(gdelt_cfg.get("lookback_days", 90)),
                    window_days=int(gdelt_cfg.get("window_days", 7)),
                    maxrecords=int(gdelt_cfg.get("max_records_per_window", 75)),
                    sort=str(gdelt_cfg.get("sort", "datedesc")),
                    timeout=int(gdelt_cfg.get("timeout_seconds", 60)),
                    retries=int(gdelt_cfg.get("retries", 2)),
                    retry_backoff_seconds=float(gdelt_cfg.get("retry_backoff_seconds", 3.0)),
                    sleep_seconds=float(gdelt_cfg.get("sleep_seconds", 0.75)),
                    end_timestamp=end_timestamp,
                    request_strategy=str(gdelt_cfg.get("request_strategy", "timespan")),
                )
                rows.append(frame)
                logs.append(
                    {
                        "collection_source": "gdelt",
                        "query_name": query_name,
                        "asset": asset,
                        "query": gdelt_query,
                        "headline_count": int(len(frame)),
                        "request_count": int(request_count),
                        "status": "ok",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("뉴스 수집 실패: %s | gdelt | %s", query_name, exc)
                logs.append(
                    {
                        "collection_source": "gdelt",
                        "query_name": query_name,
                        "asset": asset,
                        "query": gdelt_query,
                        "headline_count": 0,
                        "request_count": 0,
                        "status": f"error: {exc}",
                    }
                )

    combined = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
    if not combined.empty:
        combined["asset"] = combined["asset"].apply(_normalize_asset)
        dedupe_cols = ["timestamp", "source", "headline", "link"]
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="first").sort_values("timestamp")
    return combined.reset_index(drop=True), pd.DataFrame(logs)


def merge_and_save_news(existing: pd.DataFrame, fresh: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    combined = pd.concat([existing, fresh], axis=0, ignore_index=True) if not existing.empty or not fresh.empty else pd.DataFrame()
    if not combined.empty:
        if "asset" in combined.columns:
            combined["asset"] = combined["asset"].apply(_normalize_asset)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
        if "collected_at_utc" in combined.columns:
            combined["collected_at_utc"] = pd.to_datetime(combined["collected_at_utc"], utc=True, errors="coerce")
        dedupe_cols = ["timestamp", "source", "headline", "link"]
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="last").sort_values("timestamp")

    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(resolved, index=False)
    return combined.reset_index(drop=True)


def build_news_collection_report(
    collection_log: pd.DataFrame,
    merged_news: pd.DataFrame,
    output_paths: list[str],
) -> str:
    lines = ["# 뉴스 헤드라인 수집 리포트", ""]
    lines.append("## 수집 결과")
    if collection_log.empty:
        lines.append("- 수집 로그가 없습니다.")
    else:
        for _, row in collection_log.iterrows():
            lines.append(
                f"- {row['collection_source']} / {row['query_name']} "
                f"({row['asset'] if pd.notna(row['asset']) else 'market'}): "
                f"{int(row['headline_count'])}건, 요청 {int(row['request_count'])}회, 상태 {row['status']}"
            )
    lines.append("")

    lines.append("## 누적 파일 상태")
    lines.append(f"- 총 headline 수: {len(merged_news)}")
    if not merged_news.empty:
        lines.append(f"- 구간: {merged_news['timestamp'].min()} ~ {merged_news['timestamp'].max()}")
        asset_counts = merged_news["asset"].fillna("market").value_counts().to_dict()
        lines.append(f"- asset 분포: {asset_counts}")
        if "collection_source" in merged_news.columns:
            source_counts = merged_news["collection_source"].fillna("unknown").value_counts().to_dict()
            lines.append(f"- 수집 소스 분포: {source_counts}")
    lines.append("")

    lines.append("## 출력물")
    for path in output_paths:
        lines.append(f"- {path}")
    return "\n".join(lines) + "\n"

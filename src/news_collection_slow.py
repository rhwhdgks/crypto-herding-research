from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pandas as pd

from news_collection import fetch_gdelt_query, load_existing_news, merge_and_save_news
from utils import save_json

LOGGER = logging.getLogger(__name__)


def _build_query_lookup(config: dict) -> dict[str, dict]:
    collection_cfg = dict(config["sentiment_extension"]["news_collection"])
    queries = list(collection_cfg.get("queries", []))
    return {str(query_cfg["name"]): dict(query_cfg) for query_cfg in queries}


def _default_state(query_names: list[str]) -> dict[str, Any]:
    return {
        "query_names": list(query_names),
        "next_index": 0,
        "completed_cycles": 0,
        "completed_runs": 0,
        "last_query_name": None,
        "last_status": None,
        "last_run_at_utc": None,
        "last_headline_count": 0,
    }


def load_gdelt_slow_state(path: str | Path, query_names: list[str]) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        return _default_state(query_names)

    frame = pd.read_json(resolved, typ="series")
    state = frame.to_dict()
    if state.get("query_names") != list(query_names):
        return _default_state(query_names)
    state["next_index"] = int(state.get("next_index", 0)) % max(len(query_names), 1)
    state["completed_cycles"] = int(state.get("completed_cycles", 0))
    state["completed_runs"] = int(state.get("completed_runs", 0))
    state["last_headline_count"] = int(state.get("last_headline_count", 0))
    return state


def _save_state(state: dict[str, Any], path: str | Path) -> None:
    save_json(state, path)


def _select_query_batch(state: dict[str, Any], query_names: list[str], queries_per_run: int) -> list[str]:
    if not query_names:
        return []
    start_index = int(state.get("next_index", 0)) % len(query_names)
    batch: list[str] = []
    for offset in range(max(int(queries_per_run), 1)):
        batch.append(query_names[(start_index + offset) % len(query_names)])
    return batch


def _advance_state_after_success(state: dict[str, Any], query_names: list[str], steps: int) -> None:
    if not query_names:
        state["next_index"] = 0
        return
    previous = int(state.get("next_index", 0)) % len(query_names)
    updated = previous + int(steps)
    state["completed_cycles"] = int(state.get("completed_cycles", 0)) + (updated // len(query_names))
    state["next_index"] = updated % len(query_names)


def collect_gdelt_slow_batch(
    config: dict,
    *,
    query_name_override: str | None = None,
    queries_per_run_override: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    collection_cfg = dict(config["sentiment_extension"]["news_collection"])
    gdelt_cfg = dict(collection_cfg.get("gdelt", {}))
    slow_cfg = dict(collection_cfg.get("gdelt_slow", {}))
    query_lookup = _build_query_lookup(config)

    configured_query_names = list(slow_cfg.get("query_names", query_lookup.keys()))
    query_names = [name for name in configured_query_names if name in query_lookup]
    if not query_names:
        raise ValueError("gdelt_slow.query_names에 유효한 query가 없습니다.")

    state_path = Path(slow_cfg.get("state_path", "data/news/gdelt_slow_state.json"))
    state = load_gdelt_slow_state(state_path, query_names)

    if query_name_override:
        selected_query_names = [query_name_override]
    else:
        selected_query_names = _select_query_batch(
            state=state,
            query_names=query_names,
            queries_per_run=queries_per_run_override or int(slow_cfg.get("queries_per_run", 1)),
        )

    end_timestamp = pd.Timestamp.now(tz="UTC")
    rows: list[pd.DataFrame] = []
    logs: list[dict[str, Any]] = []
    completed_steps = 0
    stop_reason: str | None = None

    for idx, query_name in enumerate(selected_query_names):
        query_cfg = dict(query_lookup[query_name])
        asset = query_cfg.get("asset")
        gdelt_query = str(query_cfg.get("gdelt_query", query_cfg.get("query")))
        if idx > 0:
            sleep_seconds = float(slow_cfg.get("inter_query_sleep_seconds", 15.0))
            if sleep_seconds > 0:
                LOGGER.info("다음 GDELT query 전 %.1fs 대기", sleep_seconds)
                import time

                time.sleep(sleep_seconds)

        try:
            frame, request_count = fetch_gdelt_query(
                query_name=query_name,
                query=gdelt_query,
                asset=asset,
                source_name=str(query_cfg.get("gdelt_source", "gdelt")),
                lookback_days=int(slow_cfg.get("lookback_days", gdelt_cfg.get("lookback_days", 90))),
                window_days=int(slow_cfg.get("window_days", gdelt_cfg.get("window_days", 7))),
                maxrecords=int(slow_cfg.get("max_records_per_query", gdelt_cfg.get("max_records_per_window", 50))),
                sort=str(slow_cfg.get("sort", gdelt_cfg.get("sort", "datedesc"))),
                timeout=int(slow_cfg.get("timeout_seconds", gdelt_cfg.get("timeout_seconds", 60))),
                retries=int(slow_cfg.get("retries", gdelt_cfg.get("retries", 1))),
                retry_backoff_seconds=float(
                    slow_cfg.get("retry_backoff_seconds", gdelt_cfg.get("retry_backoff_seconds", 5.0))
                ),
                sleep_seconds=0.0,
                end_timestamp=end_timestamp,
                request_strategy=str(slow_cfg.get("request_strategy", gdelt_cfg.get("request_strategy", "timespan"))),
            )
            rows.append(frame)
            logs.append(
                {
                    "collection_source": "gdelt_slow",
                    "query_name": query_name,
                    "asset": asset,
                    "query": gdelt_query,
                    "headline_count": int(len(frame)),
                    "request_count": int(request_count),
                    "status": "ok",
                }
            )
            completed_steps += 1
        except HTTPError as exc:
            logs.append(
                {
                    "collection_source": "gdelt_slow",
                    "query_name": query_name,
                    "asset": asset,
                    "query": gdelt_query,
                    "headline_count": 0,
                    "request_count": 0,
                    "status": f"error: {exc}",
                }
            )
            stop_reason = f"http_{exc.code}"
            LOGGER.warning("GDELT 저속 수집 중단: %s | %s", query_name, exc)
            if int(exc.code) == 429 and bool(slow_cfg.get("stop_on_rate_limit", True)):
                break
            if bool(slow_cfg.get("advance_on_non_rate_limit_error", True)):
                completed_steps += 1
        except Exception as exc:  # noqa: BLE001
            logs.append(
                {
                    "collection_source": "gdelt_slow",
                    "query_name": query_name,
                    "asset": asset,
                    "query": gdelt_query,
                    "headline_count": 0,
                    "request_count": 0,
                    "status": f"error: {exc}",
                }
            )
            LOGGER.warning("GDELT 저속 수집 실패: %s | %s", query_name, exc)
            if bool(slow_cfg.get("advance_on_non_rate_limit_error", True)):
                completed_steps += 1

    if query_name_override is None:
        _advance_state_after_success(state, query_names, completed_steps)

    latest_log = logs[-1] if logs else {}
    state["completed_runs"] = int(state.get("completed_runs", 0)) + 1
    state["last_query_name"] = latest_log.get("query_name")
    state["last_status"] = latest_log.get("status")
    state["last_run_at_utc"] = str(pd.Timestamp.now(tz="UTC"))
    state["last_headline_count"] = int(latest_log.get("headline_count", 0) or 0)
    if stop_reason is not None:
        state["last_stop_reason"] = stop_reason
    _save_state(state, state_path)

    combined = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
    if not combined.empty:
        dedupe_cols = ["timestamp", "source", "headline", "link"]
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="first").sort_values("timestamp")
    return combined.reset_index(drop=True), pd.DataFrame(logs), state


def build_gdelt_slow_report(
    *,
    state: dict[str, Any],
    collection_log: pd.DataFrame,
    merged_news: pd.DataFrame,
    output_paths: list[str],
) -> str:
    lines = ["# GDELT 저속 수집 리포트", ""]
    lines.append("## 상태")
    lines.append(f"- 다음 query index: {state.get('next_index')}")
    lines.append(f"- 완료 cycle 수: {state.get('completed_cycles')}")
    lines.append(f"- 완료 run 수: {state.get('completed_runs')}")
    lines.append(f"- 마지막 query: {state.get('last_query_name')}")
    lines.append(f"- 마지막 상태: {state.get('last_status')}")
    lines.append(f"- 마지막 headline 수: {state.get('last_headline_count')}")
    if state.get("last_run_at_utc"):
        lines.append(f"- 마지막 실행 시각: {state.get('last_run_at_utc')}")
    lines.append("")

    lines.append("## 이번 실행")
    if collection_log.empty:
        lines.append("- 실행 로그가 없습니다.")
    else:
        for _, row in collection_log.iterrows():
            lines.append(
                f"- {row['query_name']} ({row['asset'] if pd.notna(row['asset']) else 'market'}): "
                f"{int(row['headline_count'])}건, 요청 {int(row['request_count'])}회, 상태 {row['status']}"
            )
    lines.append("")

    lines.append("## 누적 뉴스 파일")
    lines.append(f"- 총 headline 수: {len(merged_news)}")
    if not merged_news.empty:
        lines.append(f"- 구간: {merged_news['timestamp'].min()} ~ {merged_news['timestamp'].max()}")
        if "collection_source" in merged_news.columns:
            source_counts = merged_news["collection_source"].fillna("unknown").value_counts().to_dict()
            lines.append(f"- 수집 소스 분포: {source_counts}")
    lines.append("")

    lines.append("## 출력물")
    for path in output_paths:
        lines.append(f"- {path}")
    return "\n".join(lines) + "\n"

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from tick_data import (
    ensure_tick_archive,
    ensure_tick_month_archive,
    iter_tick_dates,
    iter_tick_months,
    resolve_tick_date_window,
)


LOGGER = logging.getLogger(__name__)


def build_backfill_plan(data_cfg: dict) -> tuple[dict, list[dict]]:
    resolved = resolve_tick_date_window(data_cfg)
    start_ts = pd.Timestamp(resolved["start"], tz="UTC")
    end_ts = pd.Timestamp(resolved["end"], tz="UTC")

    start_month = pd.Timestamp(year=start_ts.year, month=start_ts.month, day=1, tz="UTC")
    end_month = pd.Timestamp(year=end_ts.year, month=end_ts.month, day=1, tz="UTC")
    end_month_last_day = (end_month + pd.offsets.MonthEnd(1)).normalize()
    can_use_end_monthly = end_ts == end_month_last_day

    monthly_months: list[pd.Timestamp] = []
    if start_month < end_month:
        monthly_end = end_month if can_use_end_monthly else end_month - pd.offsets.MonthBegin(1)
        if monthly_end >= start_month:
            monthly_months = list(pd.date_range(start=start_month, end=monthly_end, freq="MS", tz="UTC"))

    if monthly_months:
        daily_start = monthly_months[-1] + pd.offsets.MonthBegin(1)
        if can_use_end_monthly:
            daily_dates: list[pd.Timestamp] = []
        else:
            daily_dates = iter_tick_dates(str(daily_start.date()), str(end_ts.date()))
    else:
        daily_dates = iter_tick_dates(str(start_ts.date()), str(end_ts.date()))

    plan: list[dict] = []
    for symbol in resolved["symbols"]:
        for month_start in monthly_months:
            plan.append(
                {
                    "symbol": symbol,
                    "granularity": "monthly",
                    "target": month_start,
                }
            )
        for date in daily_dates:
            plan.append(
                {
                    "symbol": symbol,
                    "granularity": "daily",
                    "target": date,
                }
            )
    return resolved, plan


def execute_backfill_plan(data_cfg: dict, max_workers: int) -> pd.DataFrame:
    resolved, plan = build_backfill_plan(data_cfg)
    rows: list[dict] = []
    if not plan:
        return pd.DataFrame()

    LOGGER.info("Starting tick archive backfill with %d tasks.", len(plan))
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        future_map = {executor.submit(_run_task, task, resolved): task for task in plan}
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append(
                    {
                        "symbol": task["symbol"],
                        "granularity": task["granularity"],
                        "target": _format_target(task["target"], task["granularity"]),
                        "status": "error",
                        "source_used": "error",
                        "archive_path": "",
                        "archive_size_bytes": 0,
                        "error": str(exc),
                    }
                )
    return pd.DataFrame(rows).sort_values(["symbol", "granularity", "target"]).reset_index(drop=True)


def build_backfill_report(
    resolved_cfg: dict,
    summary: pd.DataFrame,
    local_data_dir: str,
) -> str:
    lines = ["# Tick Archive Backfill", ""]
    lines.append("## 설정")
    lines.append(f"- 종목: {', '.join(resolved_cfg['symbols'])}")
    lines.append(f"- trade_kind: {resolved_cfg.get('trade_kind', 'aggTrades')}")
    lines.append(f"- 요청 구간: {resolved_cfg['resolved_start_utc']} ~ {resolved_cfg['resolved_end_utc']}")
    lines.append(f"- 로컬 저장 경로: {local_data_dir}")
    lines.append("")

    if summary.empty:
        lines.append("## 결과")
        lines.append("- 다운로드 작업이 없습니다.")
        lines.append("")
        return "\n".join(lines)

    total_tasks = int(len(summary))
    ok = summary[summary["status"] == "ok"].copy()
    cached = ok[ok["source_used"] == "cache"]
    downloaded = ok[ok["source_used"] == "download"]
    errors = summary[summary["status"] == "error"].copy()
    total_bytes = int(ok["archive_size_bytes"].sum())

    lines.append("## 요약")
    lines.append(f"- 전체 작업 수: {total_tasks}")
    lines.append(f"- 성공: {int(len(ok))}")
    lines.append(f"- 새 다운로드: {int(len(downloaded))}")
    lines.append(f"- 캐시 재사용: {int(len(cached))}")
    lines.append(f"- 실패: {int(len(errors))}")
    lines.append(f"- 확인된 총 아카이브 용량: {total_bytes / 1024 / 1024 / 1024:.2f} GB")
    lines.append("")

    lines.append("## 분해")
    grouped = (
        ok.groupby(["granularity", "source_used"], dropna=False)
        .agg(task_count=("target", "count"), total_bytes=("archive_size_bytes", "sum"))
        .reset_index()
    )
    for _, row in grouped.iterrows():
        lines.append(
            f"- {row['granularity']} / {row['source_used']}: {int(row['task_count'])}건, "
            f"{float(row['total_bytes']) / 1024 / 1024 / 1024:.2f} GB"
        )
    lines.append("")

    if not errors.empty:
        lines.append("## 실패 작업")
        for _, row in errors.head(20).iterrows():
            lines.append(f"- {row['symbol']} {row['granularity']} {row['target']}: {row['error']}")
        lines.append("")

    return "\n".join(lines)


def _run_task(task: dict, data_cfg: dict) -> dict:
    symbol = task["symbol"]
    granularity = task["granularity"]
    target = task["target"]

    if granularity == "monthly":
        archive_path, source_used = ensure_tick_month_archive(symbol=symbol, month_start=target, data_cfg=data_cfg)
    else:
        archive_path, source_used = ensure_tick_archive(symbol=symbol, date=target, data_cfg=data_cfg)

    return {
        "symbol": symbol,
        "granularity": granularity,
        "target": _format_target(target, granularity),
        "status": "ok",
        "source_used": source_used,
        "archive_path": str(archive_path),
        "archive_size_bytes": int(Path(archive_path).stat().st_size),
        "error": "",
    }


def _format_target(target: pd.Timestamp, granularity: str) -> str:
    return target.strftime("%Y-%m" if granularity == "monthly" else "%Y-%m-%d")

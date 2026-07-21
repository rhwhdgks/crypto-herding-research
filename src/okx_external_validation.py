from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crypto-herding-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from binance_external_validation import build_binance_panels


LOGGER = logging.getLogger(__name__)
HISTORY_COLUMNS = [
    "date",
    "instrument_id",
    "symbol",
    "open_usdt",
    "high_usdt",
    "low_usdt",
    "close_usdt",
    "base_volume",
    "quote_volume_usdt",
    "confirmed",
]


def build_chunk_windows(source_cfg: Mapping) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = _utc_day(source_cfg["source_start"])
    end = _utc_day(source_cfg["source_end"])
    chunk_days = int(source_cfg["chunk_days"])
    if start > end or chunk_days < 1 or chunk_days > int(source_cfg["limit"]):
        raise ValueError("OKX source range or chunk_days is invalid")
    windows = []
    cursor = start
    while cursor <= end:
        window_end = min(cursor + pd.Timedelta(days=chunk_days - 1), end)
        windows.append((cursor, window_end))
        cursor = window_end + pd.Timedelta(days=1)
    return windows


def checkpoint_path(
    cache_dir: str | Path,
    instrument_id: str,
    start: object,
    end: object,
) -> Path:
    safe_id = str(instrument_id).replace("/", "-")
    return (
        Path(cache_dir)
        / "raw"
        / f"instrument={safe_id}"
        / f"{pd.Timestamp(start):%Y-%m-%d}_{pd.Timestamp(end):%Y-%m-%d}.json.gz"
    )


def parse_okx_candles_payload(
    payload: Mapping,
    instrument: Mapping,
    start: object,
    end: object,
) -> pd.DataFrame:
    if str(payload.get("code")) != "0":
        raise ValueError(f"OKX payload error: {payload.get('msg', 'unknown')}")
    data = payload.get("data", [])
    if not isinstance(data, list):
        raise ValueError("OKX candle payload data must be a list")
    lower = _utc_day(start)
    upper = _utc_day(end)
    rows = []
    for item in data:
        if not isinstance(item, list) or len(item) < 8:
            raise ValueError("OKX candle row has an unsupported schema")
        date = pd.to_datetime(int(item[0]), unit="ms", utc=True).normalize()
        if not lower <= date <= upper:
            continue
        confirmed = str(item[-1])
        if confirmed != "1":
            raise ValueError("OKX historical window contains an unconfirmed candle")
        quote_volume = item[7] if len(item) >= 9 else item[6]
        rows.append(
            {
                "date": date,
                "instrument_id": str(instrument["instrument_id"]),
                "symbol": str(instrument["research_symbol"]),
                "open_usdt": _number(item[1]),
                "high_usdt": _number(item[2]),
                "low_usdt": _number(item[3]),
                "close_usdt": _number(item[4]),
                "base_volume": _number(item[5]),
                "quote_volume_usdt": _number(quote_volume),
                "confirmed": True,
            }
        )
    frame = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    if not frame.empty and frame.duplicated("date").any():
        raise ValueError("OKX checkpoint contains duplicate dates")
    return frame.sort_values("date").reset_index(drop=True)


def parse_okx_instruments_payload(
    payload: Mapping,
    instruments: Sequence[Mapping],
) -> pd.DataFrame:
    if str(payload.get("code")) != "0":
        raise ValueError(f"OKX instruments error: {payload.get('msg', 'unknown')}")
    data = payload.get("data", [])
    by_id = {str(row.get("instId")): row for row in data}
    rows = []
    for instrument in instruments:
        instrument_id = str(instrument["instrument_id"])
        if instrument_id not in by_id:
            raise ValueError(f"OKX current instrument metadata is missing {instrument_id}")
        row = by_id[instrument_id]
        if str(row.get("instType")) != "SPOT" or str(row.get("state")) != "live":
            raise ValueError(f"OKX instrument is not a live spot pair: {instrument_id}")
        list_time_ms = int(row["listTime"])
        rows.append(
            {
                "instrument_id": instrument_id,
                "symbol": str(instrument["research_symbol"]),
                "base_currency": str(row.get("baseCcy", "")),
                "quote_currency": str(row.get("quoteCcy", "")),
                "state": str(row.get("state", "")),
                "list_time_ms": list_time_ms,
                "list_time": pd.to_datetime(list_time_ms, unit="ms", utc=True),
            }
        )
    result = pd.DataFrame(rows)
    if result["instrument_id"].duplicated().any() or result["symbol"].duplicated().any():
        raise ValueError("OKX instrument metadata contains duplicate research identifiers")
    return result.sort_values("instrument_id").reset_index(drop=True)


def collect_okx_history(source_cfg: Mapping) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    instruments = list(source_cfg["instruments"])
    _validate_instruments(instruments)
    metadata = _ensure_instruments_snapshot(source_cfg, instruments)
    windows = build_chunk_windows(source_cfg)
    tasks = [
        (instrument, start, end)
        for instrument in instruments
        for start, end in windows
    ]
    cached = []
    missing = []
    for task in tasks:
        instrument, start, end = task
        path = checkpoint_path(source_cfg["cache_dir"], instrument["instrument_id"], start, end)
        if _valid_checkpoint(path, instrument, start, end):
            cached.append(task)
        else:
            missing.append(task)
    LOGGER.info(
        "OKX collection: expected=%d cached=%d missing=%d",
        len(tasks),
        len(cached),
        len(missing),
    )
    _write_collection_state(source_cfg, tasks, cached, [], "running")
    failures = []
    for count, (instrument, start, end) in enumerate(missing, start=1):
        try:
            _download_checkpoint(instrument, start, end, source_cfg)
            cached.append((instrument, start, end))
        except Exception as exc:  # noqa: BLE001 - persisted for resumability
            failures.append(
                {
                    "instrument_id": instrument["instrument_id"],
                    "start": f"{start:%Y-%m-%d}",
                    "end": f"{end:%Y-%m-%d}",
                    "error": str(exc),
                }
            )
            LOGGER.error(
                "OKX checkpoint failed: %s %s~%s: %s",
                instrument["instrument_id"],
                start.date(),
                end.date(),
                exc,
            )
        if count % 20 == 0 or count == len(missing):
            LOGGER.info(
                "OKX progress: downloaded=%d/%d failures=%d",
                count,
                len(missing),
                len(failures),
            )
            _write_collection_state(
                source_cfg,
                tasks,
                cached,
                failures,
                "running" if count < len(missing) else "validating",
            )
    if failures:
        _write_collection_state(source_cfg, tasks, cached, failures, "failed")
        raise RuntimeError(f"OKX collection failed for {len(failures)} checkpoints")
    manifest, history = build_okx_history_manifest(source_cfg)
    _write_collection_state(source_cfg, tasks, cached, [], "complete")
    return metadata, manifest, history


def build_okx_history_manifest(
    source_cfg: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_rows = []
    frames = []
    for instrument in source_cfg["instruments"]:
        for start, end in build_chunk_windows(source_cfg):
            path = checkpoint_path(
                source_cfg["cache_dir"], instrument["instrument_id"], start, end
            )
            if not _valid_checkpoint(path, instrument, start, end):
                raise ValueError(f"Missing or invalid OKX checkpoint: {path}")
            frame = parse_okx_candles_payload(
                _read_gzip_json(path), instrument, start, end
            )
            frames.append(frame)
            manifest_rows.append(
                {
                    "instrument_id": instrument["instrument_id"],
                    "symbol": instrument["research_symbol"],
                    "window_start": start,
                    "window_end": end,
                    "observations": len(frame),
                    "first_observation": frame["date"].min() if not frame.empty else pd.NaT,
                    "last_observation": frame["date"].max() if not frame.empty else pd.NaT,
                    "path": path.as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    history = pd.concat(frames, ignore_index=True)
    # Empty pre-listing checkpoints can coerce timezone-aware dates to object.
    history["date"] = pd.to_datetime(history["date"], utc=True)
    numeric_columns = [
        "open_usdt",
        "high_usdt",
        "low_usdt",
        "close_usdt",
        "base_volume",
        "quote_volume_usdt",
    ]
    history[numeric_columns] = history[numeric_columns].apply(pd.to_numeric, errors="coerce")
    history = history.sort_values(["date", "instrument_id"]).reset_index(drop=True)
    if history.duplicated(["date", "instrument_id"]).any():
        raise ValueError("Merged OKX history contains duplicate date-instrument rows")
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = Path(source_cfg["manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    normalized = Path(source_cfg["normalized_path"])
    normalized.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(normalized, index=False)
    return manifest, history


def load_okx_cached_history(
    source_cfg: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = _read_instruments_snapshot(source_cfg)
    manifest, history = build_okx_history_manifest(source_cfg)
    return metadata, manifest, history


def collection_status(source_cfg: Mapping) -> dict:
    instruments = list(source_cfg["instruments"])
    windows = build_chunk_windows(source_cfg)
    expected = len(instruments) * len(windows)
    valid = 0
    for instrument in instruments:
        for start, end in windows:
            path = checkpoint_path(
                source_cfg["cache_dir"], instrument["instrument_id"], start, end
            )
            valid += int(_valid_checkpoint(path, instrument, start, end))
    metadata_valid = False
    try:
        _read_instruments_snapshot(source_cfg)
        metadata_valid = True
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return {
        "metadata_complete": metadata_valid,
        "expected_checkpoints": expected,
        "valid_checkpoints": valid,
        "completion_share": valid / expected if expected else 0.0,
        "complete": metadata_valid and valid == expected,
    }


def build_okx_panels(
    history: pd.DataFrame,
    variant_cfg: Mapping,
    analysis_cfg: Mapping,
) -> dict[str, pd.DataFrame | pd.Series]:
    working = history.rename(columns={"quote_volume_usdt": "turnover_proxy_usdt"}).copy()
    working["date"] = pd.to_datetime(working["date"], utc=True)
    working["close_usdt"] = pd.to_numeric(working["close_usdt"], errors="coerce")
    working["turnover_proxy_usdt"] = pd.to_numeric(
        working["turnover_proxy_usdt"], errors="coerce"
    )
    adapted = dict(variant_cfg)
    if adapted["market_weighting"] == "lagged_quote_volume":
        adapted["market_weighting"] = "lagged_turnover"
    return build_binance_panels(working, adapted, analysis_cfg)


def build_asset_coverage(
    history: pd.DataFrame,
    metadata: pd.DataFrame,
    source_cfg: Mapping,
) -> pd.DataFrame:
    source_start = _utc_day(source_cfg["source_start"])
    source_end = _utc_day(source_cfg["source_end"])
    rows = []
    for item in metadata.itertuples():
        listing_day = pd.Timestamp(item.list_time).tz_convert("UTC").normalize()
        if pd.Timestamp(item.list_time) > listing_day:
            listing_day += pd.Timedelta(days=1)
        expected_start = max(source_start, listing_day)
        expected_days = max(len(pd.date_range(expected_start, source_end, freq="D")), 0)
        subset = history.loc[history["instrument_id"].eq(item.instrument_id)]
        rows.append(
            {
                "instrument_id": item.instrument_id,
                "symbol": item.symbol,
                "list_time": item.list_time,
                "expected_start": expected_start,
                "expected_days": expected_days,
                "observations": len(subset),
                "first_observation": subset["date"].min() if not subset.empty else pd.NaT,
                "last_observation": subset["date"].max() if not subset.empty else pd.NaT,
                "coverage_share": len(subset) / expected_days if expected_days else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["coverage_share", "instrument_id"]).reset_index(drop=True)


def validate_okx_quality(
    metadata: pd.DataFrame,
    manifest: pd.DataFrame,
    history: pd.DataFrame,
    asset_coverage: pd.DataFrame,
    panels_by_variant: Mapping[str, Mapping],
    source_cfg: Mapping,
    analysis_cfg: Mapping,
) -> pd.DataFrame:
    expected_checkpoints = len(source_cfg["instruments"]) * len(build_chunk_windows(source_cfg))
    checks = [
        ("metadata_completion", len(metadata) / len(source_cfg["instruments"]), 1.0),
        ("checkpoint_completion", len(manifest) / expected_checkpoints, 1.0),
        ("confirmed_candle_share", float(history["confirmed"].mean()), 1.0),
        (
            "positive_close_share",
            float(history["close_usdt"].gt(0).mean()),
            float(analysis_cfg["minimum_positive_price_share"]),
        ),
        (
            "minimum_asset_coverage",
            float(asset_coverage["coverage_share"].min()),
            float(analysis_cfg["minimum_asset_coverage"]),
        ),
        (
            "unique_date_instrument",
            float(not history.duplicated(["date", "instrument_id"]).any()),
            1.0,
        ),
    ]
    for variant, panels in panels_by_variant.items():
        coverage = panels["daily_coverage"]
        checks.extend(
            [
                (
                    f"eligible_daily_share:{variant}",
                    float(coverage["eligible"].mean()),
                    float(analysis_cfg["minimum_eligible_day_share"]),
                ),
                (
                    f"minimum_active_assets:{variant}",
                    float(coverage.loc[coverage["eligible"], "active_assets"].min()),
                    float(analysis_cfg["minimum_active_assets"]),
                ),
            ]
        )
    quality = pd.DataFrame(checks, columns=["check", "observed", "required"])
    quality["passes"] = quality["observed"].ge(quality["required"])
    if not bool(quality["passes"].all()):
        failed = quality.loc[~quality["passes"], "check"].tolist()
        raise ValueError(f"OKX external-validation quality gates failed: {', '.join(failed)}")
    return quality


def compare_external_sources(
    okx_targets: pd.DataFrame,
    cmc_historical: pd.DataFrame,
    cmc_holdout: pd.DataFrame,
    binance_targets: pd.DataFrame,
    decision_cfg: Mapping,
) -> pd.DataFrame:
    models = list(decision_cfg["required_models"])
    frequencies = list(decision_cfg["required_frequencies"])

    def select(
        frame: pd.DataFrame,
        label: str,
        variant: str,
        period: str,
    ) -> pd.DataFrame:
        result = frame.loc[
            frame["variant"].eq(variant)
            & frame["period"].eq(period)
            & frame["model"].isin(models)
            & frame["frequency"].isin(frequencies),
            [
                "frequency",
                "model",
                "coefficient",
                "standardized_target_coefficient",
                "t_stat",
                "q_value_bh_fdr",
                "supports_herding",
            ],
        ].copy()
        result.insert(0, "source_period", label)
        return result

    frames = [
        select(cmc_historical, "cmc_2018_2024", "replication_primary", "full_sample"),
        select(cmc_holdout, "cmc_2024_2026", "replication_primary", "holdout_full"),
        select(binance_targets, "binance_2021_2026", "equal_weight_primary", "full_5y"),
        select(
            okx_targets,
            "okx_2021_2026",
            str(decision_cfg["primary_variant"]),
            str(decision_cfg["decision_period"]),
        ),
    ]
    comparison = pd.concat(frames, ignore_index=True)
    expected = 4 * len(models) * len(frequencies)
    if len(comparison) != expected:
        raise ValueError(f"External comparison has {len(comparison)} rows, expected {expected}")
    return comparison


def build_okx_validation_report(
    config: Mapping,
    metadata: pd.DataFrame,
    asset_coverage: pd.DataFrame,
    quality: pd.DataFrame,
    panels_by_variant: Mapping[str, Mapping],
    targets: pd.DataFrame,
    decision_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    plot_paths: Sequence[str],
) -> str:
    decision_period = str(config["decision"]["decision_period"])
    full = targets.loc[targets["period"].eq(decision_period)]
    summary = decision_summary.set_index("variant")
    primary = str(config["decision"]["primary_variant"])
    sensitivity = str(config["decision"]["sensitivity_variant"])
    required = full.loc[full["model"].isin(config["decision"]["required_models"])]
    failed = required.loc[~required["supports_herding"]]
    strict_pass = bool(summary.loc[primary, "all_required_cells_pass"])
    lines = [
        "# OKX 14종목 상장 인지형 외부검증",
        "",
        "## 결론",
        "",
        f"- 동일가중 primary corrected 4개 셀: {int(summary.loc[primary, 'passing_cells'])}/4 통과",
        f"- 전기 quote-volume sensitivity corrected 4개 셀: {int(summary.loc[sensitivity, 'passing_cells'])}/4 통과",
        f"- 사전 고정 primary strict criterion: {'통과' if strict_pass else '미통과'}",
        "- OKX 동일가중 표준화 계수는 Binance와 매우 근접해 weekly SCSAD 미통과가 두 거래소에서 반복됐습니다.",
        "- Quote-volume sensitivity는 full sample 1/4이지만 late half 4/4여서 가중법·기간 불안정성이 큽니다.",
        "- 미관찰 거래소 회귀 결과이지만 corrected CSAD는 미래수익률 alpha나 intentional imitation의 직접 검정이 아닙니다.",
        "",
        "## 표본과 품질",
        "",
        f"- 분석 기간: {config['analysis']['start']}~{config['analysis']['end']}",
        f"- candidate universe: {len(metadata)}개 OKX USDT 현물, listing-aware active panel",
        f"- 자산별 최저 coverage: {asset_coverage['coverage_share'].min():.2%}",
        f"- quality gate: {int(quality['passes'].sum())}/{len(quality)} 통과",
    ]
    for variant, panels in panels_by_variant.items():
        lines.append(
            f"- {variant}: daily {len(panels['daily_market']):,}개, weekly {len(panels['weekly_market']):,}개, "
            f"활성 종목 {int(panels['daily_coverage']['active_assets'].min())}~{int(panels['daily_coverage']['active_assets'].max())}개"
        )
    lines.extend(["", "## Strict criterion 미통과 셀", ""])
    if failed.empty:
        lines.append("- 없음")
    else:
        for row in failed.itertuples():
            lines.append(
                f"- {row.variant} / {row.frequency} / {row.model}: "
                f"계수 {row.coefficient:.4f}, t={row.t_stat:.2f}, q={row.q_value_bh_fdr:.3g}"
            )
    lines.extend(["", "## 5년 전체표본 회귀", ""])
    for row in full.itertuples():
        lines.append(
            f"- {row.variant} / {row.frequency} / {row.model}: "
            f"계수 {row.coefficient:.4f}, 표준화 {row.standardized_target_coefficient:.3f}, "
            f"t={row.t_stat:.2f}, q={row.q_value_bh_fdr:.3g}, 통과={bool(row.supports_herding)}"
        )
    lines.extend(["", "## 기간 진단", ""])
    diagnostic = targets.loc[
        targets["period"].ne(decision_period)
        & targets["model"].isin(config["decision"]["required_models"])
    ]
    for row in diagnostic.itertuples():
        lines.append(
            f"- {row.period} / {row.variant} / {row.frequency} / {row.model}: "
            f"표준화 {row.standardized_target_coefficient:.3f}, t={row.t_stat:.2f}, "
            f"q={row.q_value_bh_fdr:.3g}, 통과={bool(row.supports_herding)}"
        )
    lines.extend(["", "## 공급자별 표준화 계수", ""])
    for row in comparison.itertuples():
        lines.append(
            f"- {row.source_period} / {row.frequency} / {row.model}: "
            f"표준화 {row.standardized_target_coefficient:.3f}, t={row.t_stat:.2f}, "
            f"q={row.q_value_bh_fdr:.3g}"
        )
    lines.extend(
        [
            "",
            "## 해석 제한",
            "",
            "- 현재 live instrument에서 고른 후보군이므로 상장폐지 자산 survivor bias가 남습니다.",
            "- Listing-aware 편입은 시가총액 기반 point-in-time 동적 universe와 동일하지 않습니다.",
            "- OKX 단일 거래소·USDT 현물과 선택한 14개 후보에 조건부인 결과입니다.",
            "- 전기 quote volume은 시점 안전하지만 시장수익률의 유일한 경제적 가중법은 아닙니다.",
            "- 동시적 CSAD 관계를 미래수익률 alpha, 인과효과 또는 의도적 모방으로 확대하지 않습니다.",
            "",
            "## 그림",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in plot_paths)
    return "\n".join(lines) + "\n"


def plot_external_comparison(comparison: pd.DataFrame, path: str | Path) -> None:
    frame = comparison.copy()
    frame["cell"] = frame["frequency"] + " / " + frame["model"]
    cells = frame["cell"].drop_duplicates().tolist()
    sources = frame["source_period"].drop_duplicates().tolist()
    x = np.arange(len(cells))
    width = 0.19
    fig, axis = plt.subplots(figsize=(12, 5.5))
    offsets = np.arange(len(sources)) - (len(sources) - 1) / 2
    for offset, source in zip(offsets, sources, strict=True):
        values = frame.loc[frame["source_period"].eq(source)].set_index("cell")
        axis.bar(
            x + offset * width,
            values.reindex(cells)["standardized_target_coefficient"],
            width=width,
            label=source,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, cells, rotation=20, ha="right")
    axis.set_ylabel("Standardized target coefficient")
    axis.set_title("Corrected CSAD across CMC, Binance, and OKX")
    axis.legend()
    fig.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160)
    plt.close(fig)


def _ensure_instruments_snapshot(source_cfg: Mapping, instruments: Sequence[Mapping]) -> pd.DataFrame:
    path = Path(source_cfg["instruments_path"])
    if path.is_file():
        try:
            return parse_okx_instruments_payload(_read_gzip_json(path), instruments)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            path.unlink(missing_ok=True)
    params = {"instType": "SPOT"}
    payload = _request_json(source_cfg["instruments_endpoint"], params, source_cfg)
    metadata = parse_okx_instruments_payload(payload, instruments)
    _write_gzip_json(path, payload)
    return metadata


def _read_instruments_snapshot(source_cfg: Mapping) -> pd.DataFrame:
    path = Path(source_cfg["instruments_path"])
    return parse_okx_instruments_payload(
        _read_gzip_json(path), source_cfg["instruments"]
    )


def _download_checkpoint(
    instrument: Mapping,
    start: pd.Timestamp,
    end: pd.Timestamp,
    source_cfg: Mapping,
) -> None:
    after = int((_utc_day(end) + pd.Timedelta(days=1)).timestamp() * 1000)
    params = {
        "instId": instrument["instrument_id"],
        "bar": source_cfg["bar"],
        "after": str(after),
        "limit": str(source_cfg["limit"]),
    }
    payload = _request_json(source_cfg["endpoint"], params, source_cfg)
    parse_okx_candles_payload(payload, instrument, start, end)
    path = checkpoint_path(
        source_cfg["cache_dir"], instrument["instrument_id"], start, end
    )
    _write_gzip_json(path, payload)
    time.sleep(float(source_cfg["request_interval_seconds"]))


def _request_json(url: str, params: Mapping, source_cfg: Mapping) -> dict:
    headers = {
        "User-Agent": str(source_cfg["user_agent"]),
        "Accept": "application/json",
    }
    attempts = int(source_cfg["maximum_attempts"])
    error = None
    for attempt in range(attempts):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=float(source_cfg["request_timeout_seconds"]),
            )
            response.raise_for_status()
            payload = response.json()
            if str(payload.get("code")) != "0":
                raise ValueError(f"OKX API error: {payload.get('msg', 'unknown')}")
            return payload
        except Exception as exc:  # noqa: BLE001 - retries preserve final error
            error = exc
            if attempt + 1 < attempts:
                time.sleep(float(source_cfg["backoff_seconds"]) * (2**attempt))
    raise RuntimeError(f"OKX request failed after {attempts} attempts: {error}")


def _valid_checkpoint(
    path: Path,
    instrument: Mapping,
    start: object,
    end: object,
) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        parse_okx_candles_payload(_read_gzip_json(path), instrument, start, end)
        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _write_collection_state(
    source_cfg: Mapping,
    tasks: Sequence,
    completed: Sequence,
    failures: Sequence,
    status: str,
) -> None:
    path = Path(source_cfg["state_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "expected_checkpoints": len(tasks),
        "completed_checkpoints": len(completed),
        "failure_count": len(failures),
        "failures": list(failures),
        "updated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _write_gzip_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"))
    os.replace(temporary, path)


def _read_gzip_json(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_instruments(instruments: Sequence[Mapping]) -> None:
    ids = [str(item["instrument_id"]) for item in instruments]
    symbols = [str(item["research_symbol"]) for item in instruments]
    if len(instruments) != 14:
        raise ValueError(f"OKX candidate universe must contain 14 instruments, got {len(instruments)}")
    if len(ids) != len(set(ids)) or len(symbols) != len(set(symbols)):
        raise ValueError("OKX candidate universe contains duplicates")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _utc_day(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC").normalize()
    return timestamp.tz_convert("UTC").normalize()

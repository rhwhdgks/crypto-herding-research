from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crypto-herding-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from cmc_dynamic_universe import MODEL_SPECS
from frequency_sensitivity import benjamini_hochberg


LOGGER = logging.getLogger(__name__)
HISTORY_COLUMNS = [
    "date",
    "cmc_id",
    "research_symbol",
    "provider_symbol",
    "name",
    "open_usd",
    "high_usd",
    "low_usd",
    "close_usd",
    "volume_usd",
    "market_cap_usd",
    "circulating_supply",
]


def build_chunk_windows(source_cfg: Mapping) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(source_cfg["source_start"]).normalize()
    end = pd.Timestamp(source_cfg["source_end"]).normalize()
    chunk_days = int(source_cfg["chunk_days"])
    if start > end or chunk_days < 1 or chunk_days > 400:
        raise ValueError("CMC fixed-universe source range or chunk_days is invalid")
    windows = []
    cursor = start
    while cursor <= end:
        window_end = min(cursor + pd.Timedelta(days=chunk_days - 1), end)
        windows.append((cursor, window_end))
        cursor = window_end + pd.Timedelta(days=1)
    return windows


def checkpoint_path(
    cache_dir: str | Path,
    cmc_id: int,
    start: object,
    end: object,
) -> Path:
    return (
        Path(cache_dir)
        / "raw"
        / f"cmc_id={int(cmc_id)}"
        / f"{pd.Timestamp(start):%Y-%m-%d}_{pd.Timestamp(end):%Y-%m-%d}.json.gz"
    )


def parse_individual_history_payload(
    payload: Mapping,
    research_symbol: str,
    expected_cmc_id: int,
    start: object,
    end: object,
    usd_convert_id: int = 2781,
) -> pd.DataFrame:
    status = payload.get("status", {})
    if str(status.get("error_code", "0")) != "0":
        raise ValueError(f"CMC payload error: {status.get('error_message', 'unknown')}")
    data = payload.get("data", {})
    if int(data.get("id", -1)) != int(expected_cmc_id):
        raise ValueError("CMC payload ID does not match the requested legacy CMC ID")
    quotes = data.get("quotes", [])
    if not isinstance(quotes, list):
        raise ValueError("CMC individual history quotes must be a list")

    quote_name = str(usd_convert_id)
    rows = []
    for item in quotes:
        quote = item.get("quote", {}) or {}
        if str(quote.get("name")) != quote_name:
            continue
        rows.append(
            {
                "date": pd.to_datetime(item.get("timeOpen"), utc=True).normalize(),
                "cmc_id": int(expected_cmc_id),
                "research_symbol": str(research_symbol),
                "provider_symbol": str(data.get("symbol", "")),
                "name": str(data.get("name", "")),
                "open_usd": _number(quote.get("open")),
                "high_usd": _number(quote.get("high")),
                "low_usd": _number(quote.get("low")),
                "close_usd": _number(quote.get("close")),
                "volume_usd": _number(quote.get("volume")),
                "market_cap_usd": _number(quote.get("marketCap")),
                "circulating_supply": _number(quote.get("circulatingSupply")),
            }
        )
    frame = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    if frame.empty:
        return frame
    lower = _utc_day(start)
    upper = _utc_day(end)
    frame = frame.loc[frame["date"].between(lower, upper)].copy()
    if frame.duplicated(["date", "cmc_id"]).any():
        raise ValueError("CMC individual history contains duplicate date/ID rows")
    return frame.sort_values("date").reset_index(drop=True)


def collect_fixed_universe_history(
    source_cfg: Mapping,
    universe: Sequence[Mapping],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_universe(universe)
    windows = build_chunk_windows(source_cfg)
    tasks = [
        (asset, start, end)
        for asset in universe
        for start, end in windows
    ]
    cached = []
    missing = []
    for task in tasks:
        asset, start, end = task
        path = checkpoint_path(source_cfg["cache_dir"], asset["cmc_id"], start, end)
        if _valid_checkpoint(path, asset, start, end, source_cfg):
            cached.append(task)
        else:
            missing.append(task)
    LOGGER.info(
        "CMC fixed-62 collection: expected=%d cached=%d missing=%d",
        len(tasks),
        len(cached),
        len(missing),
    )
    _write_collection_state(source_cfg, tasks, cached, [], "running")

    failures = []
    if missing:
        with ThreadPoolExecutor(max_workers=int(source_cfg["max_workers"])) as executor:
            futures = {
                executor.submit(_download_checkpoint, asset, start, end, source_cfg): (
                    asset,
                    start,
                    end,
                )
                for asset, start, end in missing
            }
            for count, future in enumerate(as_completed(futures), start=1):
                asset, start, end = futures[future]
                try:
                    future.result()
                    cached.append((asset, start, end))
                except Exception as exc:  # noqa: BLE001 - failure is persisted
                    failures.append(
                        {
                            "cmc_id": int(asset["cmc_id"]),
                            "symbol": asset["symbol"],
                            "start": f"{start:%Y-%m-%d}",
                            "end": f"{end:%Y-%m-%d}",
                            "error": str(exc),
                        }
                    )
                    LOGGER.error(
                        "CMC fixed checkpoint failed: %s %s~%s: %s",
                        asset["symbol"],
                        start.date(),
                        end.date(),
                        exc,
                    )
                if count % 25 == 0 or count == len(missing):
                    LOGGER.info(
                        "CMC fixed-62 progress: downloaded=%d/%d failures=%d",
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
        raise RuntimeError(f"CMC fixed-62 collection failed for {len(failures)} checkpoints")

    manifest, history = build_fixed_history_manifest(source_cfg, universe)
    _write_collection_state(source_cfg, tasks, cached, [], "complete")
    return manifest, history


def build_fixed_history_manifest(
    source_cfg: Mapping,
    universe: Sequence[Mapping],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_rows = []
    history_frames = []
    for asset in universe:
        for start, end in build_chunk_windows(source_cfg):
            path = checkpoint_path(source_cfg["cache_dir"], asset["cmc_id"], start, end)
            if not _valid_checkpoint(path, asset, start, end, source_cfg):
                raise ValueError(f"Missing or invalid CMC fixed checkpoint: {path}")
            payload = _read_gzip_json(path)
            frame = parse_individual_history_payload(
                payload,
                asset["symbol"],
                int(asset["cmc_id"]),
                start,
                end,
                int(source_cfg["usd_convert_id"]),
            )
            history_frames.append(frame)
            manifest_rows.append(
                {
                    "cmc_id": int(asset["cmc_id"]),
                    "symbol": asset["symbol"],
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
    history = pd.concat(history_frames, ignore_index=True)
    history = history.sort_values(["date", "cmc_id"]).reset_index(drop=True)
    if history.duplicated(["date", "cmc_id"]).any():
        raise ValueError("Merged CMC fixed history contains duplicate date/ID rows")
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = Path(source_cfg["manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    normalized = Path(source_cfg["normalized_path"])
    normalized.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(normalized, index=False)
    return manifest, history


def collection_status(source_cfg: Mapping, universe: Sequence[Mapping]) -> dict:
    windows = build_chunk_windows(source_cfg)
    expected = len(universe) * len(windows)
    valid = 0
    for asset in universe:
        for start, end in windows:
            path = checkpoint_path(source_cfg["cache_dir"], asset["cmc_id"], start, end)
            valid += int(_valid_checkpoint(path, asset, start, end, source_cfg))
    return {
        "expected_checkpoints": expected,
        "valid_checkpoints": valid,
        "completion_share": valid / expected if expected else 0.0,
        "complete": valid == expected,
    }


def build_fixed_panels(
    history: pd.DataFrame,
    variant_cfg: Mapping,
    analysis_cfg: Mapping,
) -> dict[str, pd.DataFrame | pd.Series]:
    daily_rows = _build_period_rows(history, "daily", variant_cfg, analysis_cfg)
    weekly_rows = _build_period_rows(history, "weekly", variant_cfg, analysis_cfg)
    return {
        "daily_rows": daily_rows,
        "daily_panel": daily_rows.pivot(index="period", columns="cmc_id", values="asset_return"),
        "daily_market": _series_from_rows(daily_rows, "market_return"),
        "daily_csad": _series_from_rows(daily_rows, "csad"),
        "daily_coverage": _coverage_from_rows(daily_rows, "day"),
        "weekly_rows": weekly_rows,
        "weekly_panel": weekly_rows.pivot(index="period", columns="cmc_id", values="asset_return"),
        "weekly_market": _series_from_rows(weekly_rows, "market_return"),
        "weekly_csad": _series_from_rows(weekly_rows, "csad"),
        "weekly_coverage": _coverage_from_rows(weekly_rows, "week"),
    }


def build_asset_coverage(
    history: pd.DataFrame,
    universe: Sequence[Mapping],
    source_cfg: Mapping,
) -> pd.DataFrame:
    expected_days = len(
        pd.date_range(source_cfg["source_start"], source_cfg["source_end"], freq="D")
    )
    stats = history.groupby(["cmc_id", "research_symbol"]).agg(
        observations=("date", "size"),
        first_observation=("date", "min"),
        last_observation=("date", "max"),
        positive_close_share=("close_usd", lambda values: float(values.gt(0).mean())),
        positive_market_cap_share=("market_cap_usd", lambda values: float(values.gt(0).mean())),
    ).reset_index()
    expected = pd.DataFrame(universe).rename(columns={"symbol": "research_symbol"})
    result = expected.merge(stats, on=["cmc_id", "research_symbol"], how="left")
    result["observations"] = result["observations"].fillna(0).astype(int)
    result["expected_days"] = expected_days
    result["coverage_share"] = result["observations"] / expected_days
    return result.sort_values(["coverage_share", "research_symbol"]).reset_index(drop=True)


def build_method_audit_summary(
    method_targets: pd.DataFrame,
    benchmark_cfg: Mapping,
) -> pd.DataFrame:
    paper = pd.DataFrame(benchmark_cfg["values"]).rename(
        columns={"coefficient": "paper_coefficient", "t_stat": "paper_t_stat", "nobs": "paper_nobs"}
    )
    full = method_targets.loc[method_targets["period"].eq("full_sample")].copy()
    comparison = full.merge(paper, on=["period", "frequency", "model"], how="inner", validate="many_to_one")
    comparison["coefficient_difference"] = comparison["coefficient"] - comparison["paper_coefficient"]
    comparison["absolute_coefficient_difference"] = comparison["coefficient_difference"].abs()
    comparison["absolute_t_stat_difference"] = (comparison["t_stat"] - comparison["paper_t_stat"]).abs()
    summary = comparison.groupby("variant", as_index=False).agg(
        benchmark_cells=("model", "size"),
        mean_absolute_coefficient_difference=("absolute_coefficient_difference", "mean"),
        median_absolute_coefficient_difference=("absolute_coefficient_difference", "median"),
        mean_absolute_t_stat_difference=("absolute_t_stat_difference", "mean"),
        sign_match_share=("coefficient_difference", lambda values: float(
            np.sign(comparison.loc[values.index, "coefficient"]).eq(
                np.sign(comparison.loc[values.index, "paper_coefficient"])
            ).mean()
        )),
    )
    return summary.sort_values(
        ["mean_absolute_coefficient_difference", "mean_absolute_t_stat_difference"]
    ).reset_index(drop=True)


def validate_fixed_quality(
    manifest: pd.DataFrame,
    history: pd.DataFrame,
    asset_coverage: pd.DataFrame,
    panels_by_variant: Mapping[str, Mapping],
    source_cfg: Mapping,
    analysis_cfg: Mapping,
    universe: Sequence[Mapping],
) -> pd.DataFrame:
    expected_checkpoints = len(universe) * len(build_chunk_windows(source_cfg))
    expected_asset_days = len(universe) * len(
        pd.date_range(source_cfg["source_start"], source_cfg["source_end"], freq="D")
    )
    positive_share = float(history["close_usd"].gt(0).mean())
    asset_day_share = len(history) / expected_asset_days
    checks = [
        ("checkpoint_completion", len(manifest) / expected_checkpoints, 1.0),
        ("positive_close_share", positive_share, float(analysis_cfg["minimum_positive_price_share"])),
        ("asset_day_coverage", asset_day_share, float(analysis_cfg["minimum_asset_day_coverage"])),
        ("unique_date_cmc_id", float(not history.duplicated(["date", "cmc_id"]).any()), 1.0),
    ]
    for variant, panels in panels_by_variant.items():
        coverage = panels["daily_coverage"]
        checks.append(
            (
                f"eligible_daily_share:{variant}",
                float(coverage["eligible"].mean()),
                float(analysis_cfg["minimum_eligible_day_share"]),
            )
        )
    quality = pd.DataFrame(checks, columns=["check", "observed", "required"])
    quality["passes"] = quality["observed"].ge(quality["required"])
    if not bool(quality["passes"].all()):
        failures = quality.loc[~quality["passes"], "check"].tolist()
        raise ValueError(f"CMC fixed-62 quality gates failed: {', '.join(failures)}")
    return quality


def run_fixed_regressions(
    panels_by_variant: Mapping[str, Mapping],
    analysis_cfg: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regression_cfg = analysis_cfg["regression"]
    target_rows = []
    coefficient_frames = []
    diagnostic_frames = []
    for variant_name, panels in panels_by_variant.items():
        series = {
            "daily": (panels["daily_csad"], panels["daily_market"]),
            "weekly": (panels["weekly_csad"], panels["weekly_market"]),
        }
        for period_cfg in analysis_cfg["subperiods"]:
            start = _utc_day(period_cfg["start"])
            end = _utc_day(period_cfg["end"]) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
            for frequency in analysis_cfg["frequencies"]:
                csad, market = series[frequency]
                csad = csad.loc[csad.index.to_series().between(start, end)]
                market = market.loc[market.index.to_series().between(start, end)]
                for model_name in regression_cfg["models"]:
                    model_fn, target_term = MODEL_SPECS[model_name]
                    coefficients, diagnostics, _, model, _ = model_fn(
                        csad,
                        market,
                        cov_type=regression_cfg["cov_type"],
                        hac_maxlags=regression_cfg["hac_maxlags"],
                    )
                    coefficient_frame = coefficients.reset_index()
                    coefficient_frame["variant"] = variant_name
                    coefficient_frame["period"] = period_cfg["name"]
                    coefficient_frame["frequency"] = frequency
                    coefficient_frame["model"] = model_name
                    coefficient_frames.append(coefficient_frame)
                    diagnostic_frame = diagnostics.copy()
                    diagnostic_frame["variant"] = variant_name
                    diagnostic_frame["period"] = period_cfg["name"]
                    diagnostic_frame["frequency"] = frequency
                    diagnostic_frame["model"] = model_name
                    diagnostic_frames.append(diagnostic_frame)
                    ci = model.conf_int().loc[target_term]
                    standardized_target = _standardized_target_coefficient(
                        model.params[target_term],
                        csad,
                        market,
                        target_term,
                    )
                    target_rows.append(
                        {
                            "variant": variant_name,
                            "period": period_cfg["name"],
                            "frequency": frequency,
                            "model": model_name,
                            "target_term": target_term,
                            "coefficient": float(model.params[target_term]),
                            "standardized_target_coefficient": standardized_target,
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
    expected = int(regression_cfg["family_size_per_period_variant"])
    for _, indices in targets.groupby(["variant", "period"]).groups.items():
        if len(indices) != expected:
            raise ValueError(f"Fixed-62 regression family has {len(indices)} tests, expected {expected}")
        targets.loc[indices, "q_value_bh_fdr"] = benjamini_hochberg(
            targets.loc[indices, "p_value"]
        )
    targets["supports_herding"] = (
        targets["coefficient"].lt(0)
        & targets["q_value_bh_fdr"].le(float(regression_cfg["fdr_alpha"]))
    )
    return (
        targets,
        pd.concat(coefficient_frames, ignore_index=True),
        pd.concat(diagnostic_frames, ignore_index=True),
    )


def build_benchmark_comparison(targets: pd.DataFrame, benchmark_cfg: Mapping) -> pd.DataFrame:
    paper = pd.DataFrame(benchmark_cfg["values"]).rename(
        columns={"coefficient": "paper_coefficient", "t_stat": "paper_t_stat", "nobs": "paper_nobs"}
    )
    ours = targets.loc[targets["variant"].eq("replication_primary")].rename(
        columns={"coefficient": "our_coefficient", "t_stat": "our_t_stat", "nobs": "our_nobs"}
    )
    columns = [
        "period", "frequency", "model", "our_coefficient", "our_t_stat", "our_nobs",
        "q_value_bh_fdr", "supports_herding",
    ]
    comparison = paper.merge(ours[columns], on=["period", "frequency", "model"], how="left", validate="one_to_one")
    comparison["coefficient_difference"] = comparison["our_coefficient"] - comparison["paper_coefficient"]
    comparison["absolute_coefficient_difference"] = comparison["coefficient_difference"].abs()
    comparison["coefficient_sign_matches"] = np.sign(comparison["our_coefficient"]).eq(np.sign(comparison["paper_coefficient"]))
    comparison["nobs_difference"] = comparison["our_nobs"] - comparison["paper_nobs"]
    return comparison


def build_fixed_report(
    config: Mapping,
    asset_coverage: pd.DataFrame,
    quality: pd.DataFrame,
    panels_by_variant: Mapping[str, Mapping],
    targets: pd.DataFrame,
    comparison: pd.DataFrame,
    method_audit_summary: pd.DataFrame,
    plot_paths: Sequence[str],
) -> str:
    primary = targets.loc[(targets["variant"] == "replication_primary") & (targets["period"] == "full_sample")]
    sensitivity = targets.loc[(targets["variant"] == "no_lookahead_sensitivity") & (targets["period"] == "full_sample")]
    lines = [
        "# CMC 고정 62종목 선행논문 재현",
        "",
        "## 결론",
        "",
    ]
    primary_passes = int(primary["supports_herding"].sum())
    if primary_passes == 6:
        lines.append("- primary daily·weekly 6개 사양이 모두 통과해 모형 간 강한 일치를 보였습니다.")
    elif primary_passes:
        lines.append(f"- primary 6개 사양 중 {primary_passes}개가 통과해 결과는 모형 민감적입니다.")
    else:
        lines.append("- primary 6개 사양 중 통과한 결과가 없어 fixed-62에서도 herding을 지지하지 않습니다.")
    lines.extend(
        [
            f"- no-look-ahead sensitivity는 6개 중 {int(sensitivity['supports_herding'].sum())}개가 통과했습니다.",
            "- 이 결과는 동시적 횡단면 수렴 관계이며 미래수익률 alpha가 아닙니다.",
            "",
            "## 데이터 품질",
            "",
            f"- 고정 universe: {len(asset_coverage)}개 CMC legacy ID",
            f"- 자산별 최저 coverage: {asset_coverage['coverage_share'].min():.2%}",
            f"- quality gate: {int(quality['passes'].sum())}/{len(quality)} 통과",
        ]
    )
    for variant, panels in panels_by_variant.items():
        lines.append(
            f"- {variant}: daily {len(panels['daily_market']):,}개, weekly {len(panels['weekly_market']):,}개"
        )
    lines.extend(["", "## 전체표본 Target", ""])
    for row in pd.concat([primary, sensitivity]).itertuples():
        lines.append(
            f"- {row.variant} / {row.frequency} / {row.model}: "
            f"계수 {row.coefficient:.4f}, t={row.t_stat:.2f}, q={row.q_value_bh_fdr:.3g}, "
            f"n={row.nobs}, 통과={bool(row.supports_herding)}"
        )
    lines.extend(["", "## 논문 Benchmark 비교", ""])
    for row in comparison.itertuples():
        lines.append(
            f"- {row.period} / {row.frequency} / {row.model}: 논문 {row.paper_coefficient:.3f}, "
            f"우리 {row.our_coefficient:.3f}, 차이 {row.coefficient_difference:+.3f}, "
            f"관측 수 차이 {int(row.nobs_difference):+d}"
        )
    lines.extend(["", "## 2×2 방법 감사", ""])
    for row in method_audit_summary.itertuples():
        lines.append(
            f"- {row.variant}: benchmark {row.benchmark_cells}개 평균 절대 계수차 "
            f"{row.mean_absolute_coefficient_difference:.4f}, 평균 절대 t-stat 차이 "
            f"{row.mean_absolute_t_stat_difference:.3f}"
        )
    lines.append(
        "- Table 1 기술통계와 위 방법 감사를 함께 근거로 log_contemporaneous를 direct-replication primary로 교정했습니다."
    )
    lines.extend(
        [
            "",
            "## 해석 제한",
            "",
            "- 논문의 정확한 시점별 $100M 선정 규칙과 weekly 307개 생성 규칙은 공개되지 않았습니다.",
            "- primary 당일 시총가중은 논문 직접 재현용이며 예측 가능한 가중치가 아닙니다.",
            "- 고정 62종목은 사후 생존·선정 편향 가능성이 있어 투자 universe로 사용하지 않습니다.",
            "- CMC 무료 웹 endpoint는 공식 Pro API 계약 endpoint가 아니므로 raw checkpoint와 hash를 보존했습니다.",
            "",
            "## 그림",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in plot_paths)
    return "\n".join(lines) + "\n"


def plot_fixed_results(
    panels_by_variant: Mapping[str, Mapping],
    targets: pd.DataFrame,
    coverage_path: str | Path,
    coefficient_path: str | Path,
) -> None:
    primary_coverage = panels_by_variant["replication_primary"]["daily_coverage"]
    fig, axis = plt.subplots(figsize=(12, 4.5))
    axis.plot(primary_coverage["period"], primary_coverage["active_assets"], linewidth=0.8)
    axis.axhline(primary_coverage["active_assets"].max(), color="black", linestyle="--", linewidth=0.8)
    axis.set_title("CMC fixed-62 daily active assets")
    axis.set_ylabel("Assets")
    fig.tight_layout()
    Path(coverage_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(coverage_path, dpi=160)
    plt.close(fig)

    full = targets.loc[targets["period"].eq("full_sample")].copy()
    full["label"] = full["frequency"] + " / " + full["model"]
    labels = full["label"].drop_duplicates().tolist()
    variants = full["variant"].drop_duplicates().tolist()
    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(12, 5.5))
    for index, variant in enumerate(variants):
        subset = full.loc[full["variant"].eq(variant)].set_index("label").reindex(labels)
        axis.bar(x + (index - 0.5) * width, subset["coefficient"], width=width, label=variant)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_title("CMC fixed-62 target coefficients")
    axis.legend()
    fig.tight_layout()
    Path(coefficient_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(coefficient_path, dpi=160)
    plt.close(fig)


def _build_period_rows(
    history: pd.DataFrame,
    frequency: str,
    variant_cfg: Mapping,
    analysis_cfg: Mapping,
) -> pd.DataFrame:
    working = history.sort_values(["cmc_id", "date"]).copy()
    if frequency == "weekly":
        working["period"] = working["date"] - pd.to_timedelta(working["date"].dt.weekday, unit="D")
        working = working.groupby(["cmc_id", "period"], as_index=False).agg(
            research_symbol=("research_symbol", "last"),
            period_date=("date", "last"),
            close_usd=("close_usd", "last"),
            market_cap_usd=("market_cap_usd", "last"),
            source_observations=("date", "count"),
        )
    elif frequency == "daily":
        working = working.rename(columns={"date": "period"})
        working["period_date"] = working["period"]
        working["source_observations"] = 1
    else:
        raise ValueError(f"Unsupported fixed panel frequency: {frequency}")

    grouped = working.groupby("cmc_id", sort=False)
    working["previous_period"] = grouped["period"].shift(1)
    working["previous_close_usd"] = grouped["close_usd"].shift(1)
    working["previous_market_cap_usd"] = grouped["market_cap_usd"].shift(1)
    expected_delta = pd.Timedelta(days=1 if frequency == "daily" else 7)
    exact_previous = working["period"].sub(working["previous_period"]).eq(expected_delta)
    positive = working["close_usd"].gt(0) & working["previous_close_usd"].gt(0)
    ratio = working["close_usd"].where(positive) / working["previous_close_usd"].where(positive)
    if variant_cfg["return_method"] == "simple":
        working["asset_return"] = (ratio - 1.0).where(exact_previous)
    elif variant_cfg["return_method"] == "log":
        working["asset_return"] = np.log(ratio).where(exact_previous)
    else:
        raise ValueError("return_method must be simple or log")
    if variant_cfg["market_cap_weighting"] == "contemporaneous":
        working["return_weight"] = working["market_cap_usd"]
    elif variant_cfg["market_cap_weighting"] == "lagged":
        working["return_weight"] = working["previous_market_cap_usd"].where(exact_previous)
    else:
        raise ValueError("market_cap_weighting must be contemporaneous or lagged")

    start = _utc_day(analysis_cfg["start"])
    end = _utc_day(analysis_cfg["end"])
    working = working.loc[working["period_date"].between(start, end)].copy()
    working["valid_return_weight"] = working["asset_return"].notna() & working["return_weight"].gt(0)
    active = working.groupby("period")["valid_return_weight"].sum().rename("active_assets")
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
    working = working.merge(active, left_on="period", right_index=True, validate="many_to_one")
    return working.sort_values(["period", "cmc_id"]).reset_index(drop=True)


def _coverage_from_rows(rows: pd.DataFrame, label: str) -> pd.DataFrame:
    coverage = rows.groupby("period", as_index=False).agg(
        active_assets=("active_assets", "first"),
        observed_assets=("cmc_id", "nunique"),
        eligible=("eligible", "first"),
        source_observations=("source_observations", "sum"),
    )
    coverage["frequency"] = label
    return coverage


def _series_from_rows(rows: pd.DataFrame, column: str) -> pd.Series:
    frame = rows.loc[rows["eligible"], ["period", column]].drop_duplicates("period").dropna()
    return frame.set_index("period")[column].sort_index().rename(column)


def _standardized_target_coefficient(
    coefficient: float,
    csad: pd.Series,
    market_return: pd.Series,
    target_term: str,
) -> float:
    frame = pd.concat(
        [csad.rename("csad"), market_return.rename("market_return")],
        axis=1,
        join="inner",
    ).dropna()
    if target_term == "market_return_sq":
        dependent = frame["csad"]
        target = frame["market_return"].pow(2)
    elif target_term == "market_return_cu":
        dependent = frame["csad"].where(frame["market_return"].ge(0), -frame["csad"])
        target = frame["market_return"].pow(3)
    else:
        raise ValueError(f"Unsupported standardized target term: {target_term}")
    dependent_scale = float(dependent.std(ddof=1))
    target_scale = float(target.std(ddof=1))
    if dependent_scale <= 0 or target_scale <= 0:
        return float("nan")
    return float(coefficient) * target_scale / dependent_scale


def _download_checkpoint(asset: Mapping, start: pd.Timestamp, end: pd.Timestamp, source_cfg: Mapping) -> None:
    path = checkpoint_path(source_cfg["cache_dir"], asset["cmc_id"], start, end)
    params = {
        "id": int(asset["cmc_id"]),
        "timeStart": int(_utc_day(start).timestamp()),
        "timeEnd": int((_utc_day(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).timestamp()),
        "interval": "1d",
        "convertId": int(source_cfg["usd_convert_id"]),
    }
    headers = {"User-Agent": str(source_cfg["user_agent"]), "Accept": "application/json"}
    attempts = int(source_cfg["maximum_attempts"])
    error = None
    for attempt in range(attempts):
        try:
            response = requests.get(
                source_cfg["endpoint"],
                params=params,
                headers=headers,
                timeout=float(source_cfg["request_timeout_seconds"]),
            )
            response.raise_for_status()
            payload = response.json()
            parse_individual_history_payload(
                payload,
                asset["symbol"],
                int(asset["cmc_id"]),
                start,
                end,
                int(source_cfg["usd_convert_id"]),
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            with gzip.open(temporary, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"))
            os.replace(temporary, path)
            return
        except Exception as exc:  # noqa: BLE001 - retry preserves final error
            error = exc
            if attempt + 1 < attempts:
                time.sleep(float(source_cfg["backoff_seconds"]) * (2**attempt))
    raise RuntimeError(f"CMC request failed after {attempts} attempts: {error}")


def _valid_checkpoint(path: Path, asset: Mapping, start: object, end: object, source_cfg: Mapping) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        parse_individual_history_payload(
            _read_gzip_json(path),
            asset["symbol"],
            int(asset["cmc_id"]),
            start,
            end,
            int(source_cfg["usd_convert_id"]),
        )
        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _write_collection_state(source_cfg: Mapping, tasks: Sequence, completed: Sequence, failures: Sequence, status: str) -> None:
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


def _validate_universe(universe: Sequence[Mapping]) -> None:
    ids = [int(asset["cmc_id"]) for asset in universe]
    symbols = [str(asset["symbol"]) for asset in universe]
    if len(universe) != 62:
        raise ValueError(f"Fixed universe must contain exactly 62 assets, received {len(universe)}")
    if len(ids) != len(set(ids)) or len(symbols) != len(set(symbols)):
        raise ValueError("Fixed universe contains duplicate CMC IDs or symbols")


def _read_gzip_json(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


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

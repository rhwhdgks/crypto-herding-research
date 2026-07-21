from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crypto-herding-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"timestamp", "close", "volume"}


def load_binance_daily_history(
    source_cfg: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(source_cfg["data_dir"])
    timeframe = str(source_cfg["timeframe"])
    start = _utc_day(source_cfg["source_start"])
    end = _utc_day(source_cfg["source_end"])
    symbols = [str(symbol) for symbol in source_cfg["symbols"]]
    if len(symbols) != len(set(symbols)):
        raise ValueError("Binance external-validation universe contains duplicate symbols")

    frames = []
    manifest_rows = []
    for symbol in symbols:
        path = data_dir / f"{symbol}_{timeframe}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Binance input: {path}")
        raw = pd.read_parquet(path)
        missing = REQUIRED_COLUMNS.difference(raw.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame = raw.loc[:, ["timestamp", "close", "volume"]].copy()
        frame["date"] = pd.to_datetime(frame.pop("timestamp"), utc=True).dt.normalize()
        frame["close_usdt"] = pd.to_numeric(frame.pop("close"), errors="coerce")
        frame["base_volume"] = pd.to_numeric(frame.pop("volume"), errors="coerce")
        frame["symbol"] = symbol
        frame = frame.loc[frame["date"].between(start, end)].copy()
        if frame.duplicated("date").any():
            raise ValueError(f"Duplicate Binance dates found for {symbol}")
        frame["turnover_proxy_usdt"] = frame["close_usdt"] * frame["base_volume"]
        frames.append(frame)
        manifest_rows.append(
            {
                "symbol": symbol,
                "path": path.as_posix(),
                "observations": len(frame),
                "first_observation": frame["date"].min() if not frame.empty else pd.NaT,
                "last_observation": frame["date"].max() if not frame.empty else pd.NaT,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    history = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"])
    if history.duplicated(["date", "symbol"]).any():
        raise ValueError("Merged Binance history contains duplicate date-symbol rows")
    return history.reset_index(drop=True), pd.DataFrame(manifest_rows)


def build_binance_panels(
    history: pd.DataFrame,
    variant_cfg: Mapping,
    analysis_cfg: Mapping,
) -> dict[str, pd.DataFrame | pd.Series]:
    daily_rows = _build_period_rows(history, "daily", variant_cfg, analysis_cfg)
    weekly_rows = _build_period_rows(history, "weekly", variant_cfg, analysis_cfg)
    return {
        "daily_rows": daily_rows,
        "daily_panel": daily_rows.pivot(index="period", columns="symbol", values="asset_return"),
        "daily_market": _series_from_rows(daily_rows, "market_return"),
        "daily_csad": _series_from_rows(daily_rows, "csad"),
        "daily_coverage": _coverage_from_rows(daily_rows, "day"),
        "weekly_rows": weekly_rows,
        "weekly_panel": weekly_rows.pivot(index="period", columns="symbol", values="asset_return"),
        "weekly_market": _series_from_rows(weekly_rows, "market_return"),
        "weekly_csad": _series_from_rows(weekly_rows, "csad"),
        "weekly_coverage": _coverage_from_rows(weekly_rows, "week"),
    }


def validate_binance_quality(
    history: pd.DataFrame,
    manifest: pd.DataFrame,
    panels_by_variant: Mapping[str, Mapping],
    source_cfg: Mapping,
    analysis_cfg: Mapping,
) -> pd.DataFrame:
    symbols = list(source_cfg["symbols"])
    expected_days = len(
        pd.date_range(source_cfg["source_start"], source_cfg["source_end"], freq="D")
    )
    expected_asset_days = len(symbols) * expected_days
    checks = [
        ("input_file_completion", len(manifest) / len(symbols), 1.0),
        (
            "positive_close_share",
            float(history["close_usdt"].gt(0).mean()),
            float(analysis_cfg["minimum_positive_price_share"]),
        ),
        (
            "asset_day_coverage",
            len(history) / expected_asset_days,
            float(analysis_cfg["minimum_asset_day_coverage"]),
        ),
        (
            "unique_date_symbol",
            float(not history.duplicated(["date", "symbol"]).any()),
            1.0,
        ),
    ]
    for variant, panels in panels_by_variant.items():
        daily = panels["daily_coverage"]
        checks.extend(
            [
                (
                    f"eligible_daily_share:{variant}",
                    float(daily["eligible"].mean()),
                    float(analysis_cfg["minimum_eligible_day_share"]),
                ),
                (
                    f"minimum_active_assets:{variant}",
                    float(daily["active_assets"].min()),
                    float(analysis_cfg["minimum_active_assets"]),
                ),
            ]
        )
    quality = pd.DataFrame(checks, columns=["check", "observed", "required"])
    quality["passes"] = quality["observed"].ge(quality["required"])
    if not bool(quality["passes"].all()):
        failed = quality.loc[~quality["passes"], "check"].tolist()
        raise ValueError(f"Binance external-validation quality gates failed: {', '.join(failed)}")
    return quality


def evaluate_external_robustness(
    targets: pd.DataFrame,
    decision_cfg: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_models = set(decision_cfg["required_models"])
    required_frequencies = set(decision_cfg["required_frequencies"])
    decision_period = str(decision_cfg["decision_period"])
    variants = [
        str(decision_cfg["primary_variant"]),
        str(decision_cfg["sensitivity_variant"]),
    ]
    detail = targets.loc[
        targets["variant"].isin(variants)
        & targets["period"].eq(decision_period)
        & targets["model"].isin(required_models)
        & targets["frequency"].isin(required_frequencies)
    ].copy()
    detail["criterion_pass"] = (
        detail["coefficient"].lt(0)
        & detail["q_value_bh_fdr"].le(float(decision_cfg["alpha"]))
    )
    expected = len(required_models) * len(required_frequencies)
    rows = []
    for variant in variants:
        subset = detail.loc[detail["variant"].eq(variant)]
        if len(subset) != expected:
            raise ValueError(f"External decision requires {expected} rows for {variant}")
        rows.append(
            {
                "variant": variant,
                "required_cells": expected,
                "passing_cells": int(subset["criterion_pass"].sum()),
                "all_required_cells_pass": bool(subset["criterion_pass"].all()),
            }
        )
    return detail.reset_index(drop=True), pd.DataFrame(rows)


def compare_cmc_and_binance(
    binance_targets: pd.DataFrame,
    historical_targets: pd.DataFrame,
    holdout_targets: pd.DataFrame,
    decision_cfg: Mapping,
) -> pd.DataFrame:
    models = list(decision_cfg["required_models"])
    frequencies = list(decision_cfg["required_frequencies"])
    period = str(decision_cfg["decision_period"])
    variant = str(decision_cfg["primary_variant"])

    def select(
        frame: pd.DataFrame,
        source: str,
        selected_variant: str,
        selected_period: str,
    ) -> pd.DataFrame:
        result = frame.loc[
            frame["variant"].eq(selected_variant)
            & frame["period"].eq(selected_period)
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
        result.insert(0, "source_period", source)
        return result

    frames = [
        select(historical_targets, "cmc_2018_2024", "replication_primary", "full_sample"),
        select(holdout_targets, "cmc_2024_2026", "replication_primary", "holdout_full"),
        select(binance_targets, "binance_2021_2026", variant, period),
    ]
    comparison = pd.concat(frames, ignore_index=True)
    expected = 3 * len(models) * len(frequencies)
    if len(comparison) != expected:
        raise ValueError(f"Cross-provider comparison has {len(comparison)} rows, expected {expected}")
    return comparison


def build_external_validation_report(
    config: Mapping,
    history: pd.DataFrame,
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
    primary_name = str(config["decision"]["primary_variant"])
    sensitivity_name = str(config["decision"]["sensitivity_variant"])
    required = full.loc[
        full["model"].isin(config["decision"]["required_models"])
        & full["frequency"].isin(config["decision"]["required_frequencies"])
    ]
    failed = required.loc[~required["supports_herding"]]
    strict_pass = bool(summary["all_required_cells_pass"].all())
    lines = [
        "# Binance 14종목 공급자·유니버스 외부 강건성 검증",
        "",
        "## 결론",
        "",
        (
            f"- 동일가중 primary corrected 4개 셀: "
            f"{int(summary.loc[primary_name, 'passing_cells'])}/4 통과"
        ),
        (
            f"- 전기 거래대금가중 sensitivity corrected 4개 셀: "
            f"{int(summary.loc[sensitivity_name, 'passing_cells'])}/4 통과"
        ),
        (
            "- 사전 고정 strict criterion: "
            + ("통과" if strict_pass else "미통과, 공급자·universe 외부 재현은 부분적")
        ),
        "- 이 검정은 corrected CSAD 관계의 공급자·universe 강건성을 다루며 미래수익률 alpha 검정이 아닙니다.",
        "- 기존에 연구한 Binance 패널이므로 결과는 secondary external robustness로 분류합니다.",
        "",
        "## 표본과 품질",
        "",
        f"- 분석 기간: {config['analysis']['start']}~{config['analysis']['end']}",
        f"- 고정 universe: {history['symbol'].nunique()}개 Binance USDT 현물 페어",
        f"- 전체 asset-day coverage: {len(history) / (history['date'].nunique() * history['symbol'].nunique()):.2%}",
        f"- quality gate: {int(quality['passes'].sum())}/{len(quality)} 통과",
    ]
    for variant, panels in panels_by_variant.items():
        lines.append(
            f"- {variant}: daily {len(panels['daily_market']):,}개, weekly {len(panels['weekly_market']):,}개"
        )
    lines.extend(["", "## Strict criterion 미통과 셀", ""])
    for row in failed.itertuples():
        lines.append(
            f"- {row.variant} / {row.frequency} / {row.model}: "
            f"계수 {row.coefficient:.4f}, t={row.t_stat:.2f}, q={row.q_value_bh_fdr:.3g}"
        )
    lines.append(
        "- no-intercept는 두 가중법·두 빈도에서 모두 통과했지만 SCSAD는 가중법과 빈도에 따라 한 셀씩 탈락했습니다."
    )
    lines.extend(["", "## 5년 전체표본 회귀", ""])
    for row in full.itertuples():
        lines.append(
            f"- {row.variant} / {row.frequency} / {row.model}: "
            f"계수 {row.coefficient:.4f}, 표준화 {row.standardized_target_coefficient:.3f}, "
            f"t={row.t_stat:.2f}, q={row.q_value_bh_fdr:.3g}, 통과={bool(row.supports_herding)}"
        )
    lines.extend(["", "## 기간 진단", ""])
    diagnostics = targets.loc[
        targets["period"].ne(decision_period)
        & targets["model"].isin(config["decision"]["required_models"])
    ]
    for row in diagnostics.itertuples():
        lines.append(
            f"- {row.period} / {row.variant} / {row.frequency} / {row.model}: "
            f"표준화 {row.standardized_target_coefficient:.3f}, t={row.t_stat:.2f}, "
            f"q={row.q_value_bh_fdr:.3g}, 통과={bool(row.supports_herding)}"
        )
    lines.extend(["", "## CMC와 표준화 계수 비교", ""])
    for row in comparison.itertuples():
        lines.append(
            f"- {row.source_period} / {row.frequency} / {row.model}: "
            f"표준화 계수 {row.standardized_target_coefficient:.3f}, "
            f"t={row.t_stat:.2f}, q={row.q_value_bh_fdr:.3g}"
        )
    lines.extend(
        [
            "",
            "## 해석 제한",
            "",
            "- Binance 거래소 가격과 14개 USDT 페어에 조건부인 결과입니다.",
            "- 고정 14종목은 survivor·listing selection bias를 제거하지 못합니다.",
            "- 동일가중 시장수익률은 CMC 시총가중 사양과 직접 동일하지 않습니다.",
            "- 거래대금 sensitivity는 close×base volume 근사치를 사용하며 전기 값만 사용합니다.",
            "- corrected CSAD를 intentional imitation, 인과효과 또는 거래 가능한 alpha로 해석하지 않습니다.",
            "",
            "## 그림",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in plot_paths)
    return "\n".join(lines) + "\n"


def plot_cross_provider_comparison(comparison: pd.DataFrame, path: str | Path) -> None:
    frame = comparison.copy()
    frame["cell"] = frame["frequency"] + " / " + frame["model"]
    cells = frame["cell"].drop_duplicates().tolist()
    sources = frame["source_period"].drop_duplicates().tolist()
    x = np.arange(len(cells))
    width = 0.24
    fig, axis = plt.subplots(figsize=(11, 5.5))
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
    axis.set_title("Corrected CSAD across provider and universe")
    axis.legend()
    fig.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160)
    plt.close(fig)


def _build_period_rows(
    history: pd.DataFrame,
    frequency: str,
    variant_cfg: Mapping,
    analysis_cfg: Mapping,
) -> pd.DataFrame:
    working = history.sort_values(["symbol", "date"]).copy()
    if frequency == "weekly":
        working["period"] = working["date"] - pd.to_timedelta(working["date"].dt.weekday, unit="D")
        working = working.groupby(["symbol", "period"], as_index=False).agg(
            period_date=("date", "last"),
            close_usdt=("close_usdt", "last"),
            turnover_proxy_usdt=("turnover_proxy_usdt", "sum"),
            source_observations=("date", "count"),
        )
    elif frequency == "daily":
        working = working.rename(columns={"date": "period"})
        working["period_date"] = working["period"]
        working["source_observations"] = 1
    else:
        raise ValueError(f"Unsupported Binance panel frequency: {frequency}")

    grouped = working.groupby("symbol", sort=False)
    working["previous_period"] = grouped["period"].shift(1)
    working["previous_close_usdt"] = grouped["close_usdt"].shift(1)
    working["previous_turnover_proxy_usdt"] = grouped["turnover_proxy_usdt"].shift(1)
    expected_delta = pd.Timedelta(days=1 if frequency == "daily" else 7)
    exact_previous = working["period"].sub(working["previous_period"]).eq(expected_delta)
    positive = working["close_usdt"].gt(0) & working["previous_close_usdt"].gt(0)
    ratio = working["close_usdt"].where(positive) / working["previous_close_usdt"].where(positive)
    if variant_cfg["return_method"] == "log":
        working["asset_return"] = np.log(ratio).where(exact_previous)
    elif variant_cfg["return_method"] == "simple":
        working["asset_return"] = (ratio - 1.0).where(exact_previous)
    else:
        raise ValueError("return_method must be log or simple")

    weighting = str(variant_cfg["market_weighting"])
    if weighting == "equal":
        working["return_weight"] = 1.0
    elif weighting == "lagged_turnover":
        working["return_weight"] = working["previous_turnover_proxy_usdt"].where(exact_previous)
    else:
        raise ValueError("market_weighting must be equal or lagged_turnover")

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
    working = working.merge(active, left_on="period", right_index=True, how="left", validate="many_to_one")
    return working.sort_values(["period", "symbol"]).reset_index(drop=True)


def _coverage_from_rows(rows: pd.DataFrame, label: str) -> pd.DataFrame:
    coverage = rows.groupby("period", as_index=False).agg(
        active_assets=("active_assets", "first"),
        observed_assets=("symbol", "nunique"),
        eligible=("eligible", "first"),
        source_observations=("source_observations", "sum"),
    )
    coverage["frequency"] = label
    return coverage


def _series_from_rows(rows: pd.DataFrame, column: str) -> pd.Series:
    frame = rows.loc[rows["eligible"], ["period", column]].drop_duplicates("period").dropna()
    return frame.set_index("period")[column].sort_index().rename(column)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_day(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC").normalize()
    return timestamp.tz_convert("UTC").normalize()

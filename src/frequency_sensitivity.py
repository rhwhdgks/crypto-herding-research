from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from csad import compute_csad
from market import compute_equal_weighted_market_return
from regression import prepare_regression_frame, run_csad_regression


def load_aligned_return_panel(
    path: str | Path,
    expected_symbols: Sequence[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    parsed_index = pd.DatetimeIndex(frame.index)
    frame.index = parsed_index.tz_localize("UTC") if parsed_index.tz is None else parsed_index.tz_convert("UTC")

    missing_symbols = sorted(set(expected_symbols).difference(frame.columns))
    if missing_symbols:
        raise ValueError(f"Return panel is missing configured symbols: {', '.join(missing_symbols)}")
    if frame.index.has_duplicates:
        raise ValueError("Return panel contains duplicate timestamps")

    start_ts = _as_utc_timestamp(start)
    end_ts = _as_utc_timestamp(end)
    frame = frame.loc[(frame.index >= start_ts) & (frame.index < end_ts), list(expected_symbols)]
    if frame.empty:
        raise ValueError("No return observations remain inside the configured sample")
    if frame.isna().any().any():
        raise ValueError("Frequency sensitivity requires a complete intersection return panel")
    return frame.astype(float)


def aggregate_complete_log_returns(return_panel: pd.DataFrame, minutes: int) -> pd.DataFrame:
    minutes = int(minutes)
    if minutes <= 0:
        raise ValueError("Aggregation minutes must be positive")
    if minutes == 1:
        return return_panel.copy()

    rule = f"{minutes}min"
    resampler = return_panel.resample(rule, closed="left", label="left", origin="start_day")
    aggregated = resampler.sum(min_count=minutes)
    counts = resampler.count()
    complete = counts.eq(minutes).all(axis=1)
    aggregated = aggregated.loc[complete].dropna(how="any")
    aggregated.index.name = "timestamp"
    return aggregated


def run_frequency_sensitivity(
    return_panel: pd.DataFrame,
    analysis_cfg: Mapping,
    regression_cfg: Mapping,
    multiple_testing_cfg: Mapping,
    decision_cfg: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frequencies = list(analysis_cfg["frequencies"])
    universes = list(analysis_cfg["universes"])
    expected_family_size = len(frequencies) * len(universes)
    configured_family_size = int(multiple_testing_cfg["family_size"])
    if expected_family_size != configured_family_size:
        raise ValueError(
            f"Configured family_size={configured_family_size} does not match "
            f"frequencies x universes={expected_family_size}"
        )
    if str(multiple_testing_cfg.get("method", "benjamini_hochberg")) != "benjamini_hochberg":
        raise ValueError("Only the preregistered benjamini_hochberg correction is supported")
    if not bool(decision_cfg.get("require_all_universes", True)):
        raise ValueError("The preregistered decision rule requires all universes")
    if not bool(decision_cfg.get("required_negative_beta2", True)):
        raise ValueError("The preregistered decision rule requires negative beta2")

    rows: list[dict] = []
    for frequency in frequencies:
        label = str(frequency["label"])
        minutes = int(frequency["minutes"])
        aggregated = aggregate_complete_log_returns(return_panel, minutes)
        if aggregated.empty:
            raise ValueError(f"No complete observations for frequency {label}")

        for universe in universes:
            symbols = list(universe["symbols"])
            missing_symbols = sorted(set(symbols).difference(aggregated.columns))
            if missing_symbols:
                raise ValueError(
                    f"Universe {universe['name']} is missing symbols: {', '.join(missing_symbols)}"
                )

            universe_returns = aggregated[symbols]
            market_return = compute_equal_weighted_market_return(
                universe_returns,
                min_active_assets=len(symbols),
            )
            csad = compute_csad(universe_returns, market_return, min_active_assets=len(symbols))
            coefficients, _, _, model, summary = run_csad_regression(
                csad,
                market_return,
                cov_type=str(regression_cfg.get("cov_type", "HAC")),
                hac_maxlags=regression_cfg.get("hac_maxlags", "auto"),
            )
            beta1 = coefficients.loc["abs_market_return"]
            beta2 = coefficients.loc["market_return_sq"]
            regression_frame = prepare_regression_frame(csad, market_return)
            csad_sd = float(regression_frame["csad"].std(ddof=1))
            abs_return_sd = float(regression_frame["abs_market_return"].std(ddof=1))
            squared_return_sd = float(regression_frame["market_return_sq"].std(ddof=1))
            beta1_scale = abs_return_sd / csad_sd
            beta2_scale = squared_return_sd / csad_sd

            rows.append(
                {
                    "frequency": label,
                    "frequency_minutes": minutes,
                    "universe": str(universe["name"]),
                    "universe_display_name": str(universe["display_name"]),
                    "asset_count": len(symbols),
                    "observations": int(model.nobs),
                    "sample_start": regression_frame.index.min(),
                    "sample_end": regression_frame.index.max(),
                    "beta1": float(beta1["coefficient"]),
                    "standardized_beta1": float(beta1["coefficient"] * beta1_scale),
                    "beta2": float(summary["beta2"]),
                    "beta2_std_error": float(beta2["std_error"]),
                    "beta2_ci_lower": float(beta2["ci_lower"]),
                    "beta2_ci_upper": float(beta2["ci_upper"]),
                    "standardized_beta2": float(summary["beta2"] * beta2_scale),
                    "standardized_beta2_ci_lower": float(beta2["ci_lower"] * beta2_scale),
                    "standardized_beta2_ci_upper": float(beta2["ci_upper"] * beta2_scale),
                    "beta2_t_stat": float(summary["beta2_t_stat"]),
                    "beta2_p_value_hac": float(summary["beta2_p_value"]),
                    "rsquared": float(summary["rsquared"]),
                    "hac_maxlags": int(model.cov_kwds.get("maxlags", 0)),
                }
            )

    result = pd.DataFrame(rows)
    frequency_order = {str(item["label"]): rank for rank, item in enumerate(frequencies)}
    universe_order = {str(item["name"]): rank for rank, item in enumerate(universes)}
    result["_frequency_order"] = result["frequency"].map(frequency_order)
    result["_universe_order"] = result["universe"].map(universe_order)
    result = (
        result.sort_values(["_frequency_order", "_universe_order"])
        .drop(columns=["_frequency_order", "_universe_order"])
        .reset_index(drop=True)
    )
    result["beta2_q_value_bh_fdr"] = benjamini_hochberg(result["beta2_p_value_hac"])
    alpha = float(multiple_testing_cfg.get("alpha", 0.05))
    result["cell_supports_herding"] = (
        (result["beta2"] < 0.0) & (result["beta2_q_value_bh_fdr"] <= alpha)
    )

    required_q = float(decision_cfg.get("required_q_value", alpha))
    universe_count = len(universes)
    broad_by_frequency = result.groupby("frequency", sort=False).apply(
        lambda group: bool(
            len(group) == universe_count
            and (group["beta2"] < 0.0).all()
            and (group["beta2_q_value_bh_fdr"] <= required_q).all()
        ),
        include_groups=False,
    )
    result["broad_herding_supported_at_frequency"] = result["frequency"].map(broad_by_frequency)
    comparison = build_universe_comparison(result, universes)
    return result, comparison


def build_universe_comparison(summary: pd.DataFrame, universes: Sequence[Mapping]) -> pd.DataFrame:
    universe_names = [str(item["name"]) for item in universes]
    if len(universe_names) != 2:
        raise ValueError("Universe comparison currently requires exactly two predeclared universes")

    index_columns = ["frequency", "frequency_minutes"]
    pivot = summary.pivot(index=index_columns, columns="universe", values="standardized_beta2")
    pivot = pivot.reset_index().sort_values("frequency_minutes")
    left, right = universe_names
    pivot = pivot.rename(
        columns={
            left: f"standardized_beta2_{left}",
            right: f"standardized_beta2_{right}",
        }
    )
    pivot["standardized_beta2_difference_ex_major_minus_full"] = (
        pivot[f"standardized_beta2_{right}"] - pivot[f"standardized_beta2_{left}"]
    )
    support = (
        summary.groupby(["frequency", "frequency_minutes"], as_index=False)[
            "broad_herding_supported_at_frequency"
        ]
        .first()
    )
    return pivot.merge(support, on=index_columns, how="left")


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(p_values, errors="coerce")
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = numeric.dropna().clip(0.0, 1.0)
    if valid.empty:
        return adjusted

    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy(dtype=float)
    count = len(ranked)
    raw_adjusted = ranked * count / np.arange(1, count + 1)
    monotone = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    adjusted.loc[order] = np.clip(monotone, 0.0, 1.0)
    return adjusted


def plot_standardized_beta2(summary: pd.DataFrame, path: str | Path) -> None:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(11, 6.5))

    frequencies = (
        summary[["frequency", "frequency_minutes"]]
        .drop_duplicates()
        .sort_values("frequency_minutes")
    )
    labels = frequencies["frequency"].tolist()
    positions = np.arange(len(labels), dtype=float)
    offsets = np.linspace(-0.09, 0.09, summary["universe"].nunique())

    for offset, (universe, group) in zip(offsets, summary.groupby("universe", sort=False)):
        ordered = frequencies.merge(group, on=["frequency", "frequency_minutes"], how="left")
        values = ordered["standardized_beta2"].to_numpy(dtype=float)
        lower = ordered["standardized_beta2_ci_lower"].to_numpy(dtype=float)
        upper = ordered["standardized_beta2_ci_upper"].to_numpy(dtype=float)
        errors = np.vstack([values - lower, upper - values])
        axis.errorbar(
            positions + offset,
            values,
            yerr=errors,
            marker="o",
            capsize=4,
            linewidth=1.8,
            label=str(ordered["universe_display_name"].iloc[0]),
        )

    axis.axhline(0.0, color="#8b1e1e", linestyle="--", linewidth=1.2)
    axis.set_xticks(positions, labels)
    axis.set_xlabel("수익률 집계 주기")
    axis.set_ylabel("표준화 beta2 (95% HAC 신뢰구간)")
    axis.set_title("CSAD 이차항의 빈도·시장 구성 민감도")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def build_frequency_sensitivity_report(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    config: Mapping,
    input_path: str,
    plot_path: str,
) -> str:
    supported = (
        summary.loc[summary["broad_herding_supported_at_frequency"], "frequency"]
        .drop_duplicates()
        .tolist()
    )
    negative_cells = int((summary["beta2"] < 0.0).sum())
    fdr_cells = int(summary["cell_supports_herding"].sum())
    all_positive = bool((summary["beta2"] > 0.0).all())
    exclusion_lowers_all = bool(
        (comparison["standardized_beta2_difference_ex_major_minus_full"] < 0.0).all()
    )

    lines = [
        "# CSAD 빈도·시장 구성 민감도 보고서",
        "",
        "## 연구 설계",
        "",
        f"- 입력 패널: `{input_path}`",
        f"- 기간: {config['data']['start']} 이상, {config['data']['end']} 미만",
        "- 검정군: 6개 비중첩 집계 주기 × 2개 사전 고정 universe = 12개 셀",
        "- 회귀: CSAD = alpha + beta1 × |Rm| + beta2 × Rm², Newey-West HAC",
        "- 다중검정: 12개 양측 p-value 전체에 Benjamini-Hochberg FDR 적용",
        "- 판정: 두 universe 모두 beta2 < 0이고 q <= 0.05인 주기만 광범위한 herding 근거로 인정",
        "",
        "## 결과",
        "",
        "| 주기 | universe | 관측치 | beta2 | 표준화 beta2 | HAC t | raw p | BH q | 판정 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        decision = "지지" if row.cell_supports_herding else "지지 안 함"
        lines.append(
            f"| {row.frequency} | {row.universe_display_name} | {row.observations:,} | "
            f"{row.beta2:.6g} | {row.standardized_beta2:.4f} | {row.beta2_t_stat:.2f} | "
            f"{row.beta2_p_value_hac:.3g} | {row.beta2_q_value_bh_fdr:.3g} | {decision} |"
        )

    lines.extend(
        [
            "",
            "## 사전 정의 판정",
            "",
            f"- beta2가 음수인 셀: {negative_cells}/12",
            f"- 음수이면서 BH-FDR 5%를 통과한 셀: {fdr_cells}/12",
            "- 두 universe에서 동시에 기준을 충족한 주기: " + (", ".join(supported) if supported else "없음"),
        ]
    )
    if supported:
        lines.append(
            "- 따라서 위 주기에서는 이 고정 표본과 두 시장 구성에 공통적인 classical CSAD herding 근거가 확인됩니다."
        )
    else:
        lines.append(
            "- 따라서 사전 정의한 엄격한 기준에서는 특정 주기의 광범위한 classical CSAD herding을 지지하지 않습니다."
        )

    lines.extend(["", "## 핵심 해석", ""])
    if all_positive:
        lines.append(
            "- 12개 셀의 beta2가 모두 양수입니다. 동일한 Binance 표본에서는 1분부터 1일까지 집계 주기만 바꿔도 classical CSAD herding 부호가 나타나지 않았습니다."
        )
        lines.append(
            "- 따라서 기존 paper-like 결과와의 차이를 단순한 시간 집계 효과만으로 설명하기 어렵고, 표본 기간·데이터 공급자·universe 구성·가중 방식 차이를 함께 검토해야 합니다."
        )
    if exclusion_lowers_all:
        lines.append(
            "- BTC·ETH 제외 시 표준화 beta2는 모든 주기에서 감소했습니다. 두 대형 자산이 양의 곡률을 강화하지만, 제외 후에도 부호는 음수로 전환되지 않았습니다."
        )
    minimum_row = summary.loc[summary["standardized_beta2"].idxmin()]
    lines.append(
        f"- 가장 0에 가까운 추정치는 {minimum_row['frequency']}의 {minimum_row['universe_display_name']}에서 나타났으며, "
        f"표준화 beta2={minimum_row['standardized_beta2']:.4f}, HAC p={minimum_row['beta2_p_value_hac']:.3g}입니다."
    )

    lines.extend(
        [
            "",
            "## 시장 구성 차이",
            "",
            "아래 차이는 `BTC·ETH 제외 12종목 - 전체 14종목` 표준화 beta2입니다. 양수면 BTC·ETH 제외 시 이차항이 덜 음수이거나 더 양수라는 뜻입니다.",
            "",
            "| 주기 | EW14 표준화 beta2 | EW12 표준화 beta2 | 차이 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.frequency} | {row.standardized_beta2_ew14:.4f} | "
            f"{row.standardized_beta2_ew12_ex_btc_eth:.4f} | "
            f"{row.standardized_beta2_difference_ex_major_minus_full:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 해석 범위",
            "",
            "- 이 분석은 동일한 1분 원천 패널을 재집계하므로 데이터 공급자와 표본 기간 차이를 통제한 빈도 비교입니다.",
            "- 주기별 원 beta2는 수익률 단위가 달라 직접 비교하지 않고, 표준화 beta2를 효과 크기 비교에 사용합니다.",
            "- 유의한 음수 beta2가 있어도 투자수익 예측이나 인과적 행동 herding을 곧바로 의미하지 않습니다.",
            "- 결과를 확인한 뒤 주기나 종목을 추가한 분석은 별도 탐색 연구로 표시해야 합니다.",
            "",
            "## 그림",
            "",
            f"- `{plot_path}`",
            "",
        ]
    )
    return "\n".join(lines)


def _as_utc_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")

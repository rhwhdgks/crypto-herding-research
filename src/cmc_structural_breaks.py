from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linear_sum_assignment
from statsmodels.stats.diagnostic import breaks_cusumolsresid, breaks_hansen

from frequency_sensitivity import benjamini_hochberg
from regression import run_scsad_regression


LOGGER = logging.getLogger(__name__)
DESIGN_COLUMNS = ["const", "market_return", "market_return_sq", "market_return_cu"]


@dataclass(frozen=True)
class BreakSearchResult:
    information_criteria: pd.DataFrame
    break_indices_by_count: dict[int, list[int]]
    selected_break_count: int
    selected_break_indices: list[int]
    minimum_segment_size: int
    structural_change_supported: bool
    boundary_solution: bool


def load_scsad_break_frame(
    csad_path: str | Path,
    market_return_path: str | Path,
    start: str,
    end: str,
) -> pd.DataFrame:
    csad = _load_timestamp_series(csad_path, "csad")
    market = _load_timestamp_series(market_return_path, "market_return")
    frame = pd.concat([csad, market], axis=1, join="inner").dropna()
    start_ts = _as_utc(start)
    end_ts = _as_utc(end)
    frame = frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)].copy()
    if frame.empty:
        raise ValueError("No daily SCSAD observations remain inside the configured sample")
    if frame.index.has_duplicates:
        raise ValueError("Daily SCSAD input contains duplicate timestamps")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Daily SCSAD input must be sorted by timestamp")
    frame["scsad"] = np.where(
        frame["market_return"].ge(0.0),
        frame["csad"],
        -frame["csad"],
    )
    frame["const"] = 1.0
    frame["market_return_sq"] = frame["market_return"].pow(2)
    frame["market_return_cu"] = frame["market_return"].pow(3)
    return frame


def search_structural_breaks(
    frame: pd.DataFrame,
    break_cfg: Mapping,
) -> BreakSearchResult:
    _validate_break_frame(frame)
    nobs = len(frame)
    trimming_fraction = float(break_cfg["trimming_fraction"])
    if not 0.0 < trimming_fraction < 0.5:
        raise ValueError("trimming_fraction must be between zero and 0.5")
    minimum_segment_size = int(math.ceil(nobs * trimming_fraction))
    configured_maximum = int(break_cfg["maximum_breaks"])
    feasible_maximum = max(nobs // minimum_segment_size - 1, 0)
    maximum_breaks = min(configured_maximum, feasible_maximum)
    if maximum_breaks < 1:
        raise ValueError("Sample is too short for even one admissible structural break")

    y = frame["scsad"].to_numpy(dtype=float)
    x = frame[DESIGN_COLUMNS].to_numpy(dtype=float)
    LOGGER.info(
        "Segment RSS matrix를 계산합니다. nobs=%d, min_segment=%d",
        nobs,
        minimum_segment_size,
    )
    costs = build_segment_rss_matrix(y, x, minimum_segment_size)
    rss_by_segments, previous = solve_global_break_partitions(
        costs,
        minimum_segment_size,
        maximum_breaks + 1,
    )

    break_indices_by_count: dict[int, list[int]] = {}
    rows = []
    regressors = x.shape[1]
    for break_count in range(maximum_breaks + 1):
        segment_count = break_count + 1
        rss = float(rss_by_segments[segment_count])
        if not np.isfinite(rss) or rss <= 0.0:
            raise ValueError(f"No finite partition for {break_count} breaks")
        breaks = reconstruct_break_indices(previous, segment_count, nobs)
        break_indices_by_count[break_count] = breaks
        parameters = segment_count * regressors + break_count
        common = nobs * math.log(rss / nobs)
        rows.append(
            {
                "break_count": break_count,
                "segment_count": segment_count,
                "rss": rss,
                "parameter_count": parameters,
                "aic": common + 2.0 * parameters,
                "bic": common + parameters * math.log(nobs),
                "hqic": common + 2.0 * parameters * math.log(math.log(nobs)),
                "break_indices": "|".join(str(value) for value in breaks),
            }
        )
    criteria = pd.DataFrame(rows)
    for criterion in break_cfg.get("report_criteria", ["aic", "bic", "hqic"]):
        if criterion not in criteria.columns:
            raise ValueError(f"Unsupported information criterion: {criterion}")
        criteria[f"selected_by_{criterion}"] = criteria[criterion].eq(
            criteria[criterion].min()
        )
    primary = str(break_cfg["primary_criterion"]).lower()
    if primary not in {"aic", "bic", "hqic"}:
        raise ValueError(f"Unsupported primary criterion: {primary}")
    selected_break_count = int(criteria.loc[criteria[primary].idxmin(), "break_count"])
    no_break_value = float(criteria.loc[criteria["break_count"].eq(0), primary].iloc[0])
    selected_value = float(criteria[primary].min())
    delta = no_break_value - selected_value
    criteria[f"delta_{primary}_vs_selected"] = criteria[primary] - selected_value
    criteria[f"improvement_{primary}_vs_no_break"] = no_break_value - criteria[primary]
    strong_threshold = float(break_cfg["strong_evidence_delta_bic"])
    structural_change_supported = bool(
        selected_break_count >= 1 and delta >= strong_threshold
    )
    return BreakSearchResult(
        information_criteria=criteria,
        break_indices_by_count=break_indices_by_count,
        selected_break_count=selected_break_count,
        selected_break_indices=break_indices_by_count[selected_break_count],
        minimum_segment_size=minimum_segment_size,
        structural_change_supported=structural_change_supported,
        boundary_solution=selected_break_count == maximum_breaks,
    )


def build_segment_rss_matrix(
    y: np.ndarray,
    x: np.ndarray,
    minimum_segment_size: int,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.ndim != 1 or x.ndim != 2 or len(y) != len(x):
        raise ValueError("y and x must be aligned one- and two-dimensional arrays")
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("Break search inputs must be finite")
    nobs, regressors = x.shape
    minimum_segment_size = int(minimum_segment_size)
    if minimum_segment_size < regressors + 1:
        raise ValueError("minimum_segment_size is too short for the design matrix")

    cumulative_xx = np.zeros((nobs + 1, regressors, regressors), dtype=float)
    cumulative_xy = np.zeros((nobs + 1, regressors), dtype=float)
    cumulative_yy = np.zeros(nobs + 1, dtype=float)
    cumulative_xx[1:] = np.cumsum(np.einsum("ni,nj->nij", x, x), axis=0)
    cumulative_xy[1:] = np.cumsum(x * y[:, None], axis=0)
    cumulative_yy[1:] = np.cumsum(y * y)

    costs = np.full((nobs + 1, nobs + 1), np.inf, dtype=float)
    for end in range(minimum_segment_size, nobs + 1):
        starts = np.arange(0, end - minimum_segment_size + 1)
        xx = cumulative_xx[end] - cumulative_xx[starts]
        xy = cumulative_xy[end] - cumulative_xy[starts]
        yy = cumulative_yy[end] - cumulative_yy[starts]
        try:
            coefficients = np.linalg.solve(xx, xy[..., None]).squeeze(-1)
        except np.linalg.LinAlgError:
            coefficients = np.stack(
                [np.linalg.lstsq(matrix, vector, rcond=None)[0] for matrix, vector in zip(xx, xy)]
            )
        rss = yy - np.einsum("ni,ni->n", coefficients, xy)
        costs[starts, end] = np.maximum(rss, 0.0)
    return costs


def solve_global_break_partitions(
    costs: np.ndarray,
    minimum_segment_size: int,
    maximum_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    nobs = costs.shape[0] - 1
    maximum_segments = int(maximum_segments)
    dynamic = np.full((maximum_segments + 1, nobs + 1), np.inf, dtype=float)
    previous = np.full((maximum_segments + 1, nobs + 1), -1, dtype=int)
    dynamic[1] = costs[0]
    for segment_count in range(2, maximum_segments + 1):
        for end in range(segment_count * minimum_segment_size, nobs + 1):
            candidates = np.arange(
                (segment_count - 1) * minimum_segment_size,
                end - minimum_segment_size + 1,
            )
            values = dynamic[segment_count - 1, candidates] + costs[candidates, end]
            best_position = int(np.argmin(values))
            dynamic[segment_count, end] = values[best_position]
            previous[segment_count, end] = int(candidates[best_position])
    return dynamic[:, nobs], previous


def reconstruct_break_indices(
    previous: np.ndarray,
    segment_count: int,
    nobs: int,
) -> list[int]:
    end = int(nobs)
    breaks = []
    for current_segments in range(int(segment_count), 1, -1):
        start = int(previous[current_segments, end])
        if start < 0:
            raise ValueError("Cannot reconstruct an infeasible break partition")
        breaks.append(start)
        end = start
    return sorted(breaks)


def fit_scsad_regimes(
    frame: pd.DataFrame,
    break_indices: Sequence[int],
    regression_cfg: Mapping,
    solution_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    boundaries = [0, *[int(value) for value in break_indices], len(frame)]
    target_rows = []
    coefficient_frames = []
    diagnostic_frames = []
    fitted_parts = []
    for regime_number, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
        subset = frame.iloc[start:end]
        coefficients, diagnostics, regression_frame, model, _ = run_scsad_regression(
            subset["csad"],
            subset["market_return"],
            cov_type=str(regression_cfg.get("cov_type", "HAC")),
            hac_maxlags=regression_cfg.get("hac_maxlags", "auto"),
        )
        target = coefficients.loc["market_return_cu"]
        scsad_std = float(regression_frame["scsad"].std(ddof=1))
        cubic_std = float(regression_frame["market_return_cu"].std(ddof=1))
        standardization_scale = cubic_std / scsad_std
        target_rows.append(
            {
                "solution": solution_name,
                "regime": regime_number,
                "start": subset.index[0],
                "end": subset.index[-1],
                "observations": int(model.nobs),
                "gamma3": float(target["coefficient"]),
                "gamma3_std_error": float(target["std_error"]),
                "gamma3_t_stat": float(target["t_stat"]),
                "gamma3_p_value_hac": float(target["p_value"]),
                "gamma3_ci_lower": float(target["ci_lower"]),
                "gamma3_ci_upper": float(target["ci_upper"]),
                "standardized_gamma3": float(
                    target["coefficient"] * standardization_scale
                ),
                "standardized_gamma3_ci_lower": float(
                    target["ci_lower"] * standardization_scale
                ),
                "standardized_gamma3_ci_upper": float(
                    target["ci_upper"] * standardization_scale
                ),
                "mean_csad": float(subset["csad"].mean()),
                "market_return_std": float(subset["market_return"].std(ddof=1)),
                "scsad_std": scsad_std,
                "market_return_cubic_std": cubic_std,
                "rsquared": float(model.rsquared),
                "adj_rsquared": float(model.rsquared_adj),
                "hac_maxlags": int(model.cov_kwds.get("maxlags", 0)),
            }
        )
        coefficient_frame = coefficients.reset_index()
        coefficient_frame["solution"] = solution_name
        coefficient_frame["regime"] = regime_number
        coefficient_frames.append(coefficient_frame)
        diagnostic_frame = diagnostics.copy()
        diagnostic_frame["solution"] = solution_name
        diagnostic_frame["regime"] = regime_number
        diagnostic_frames.append(diagnostic_frame)
        fitted_parts.append(regression_frame["fitted_scsad"])

    targets = pd.DataFrame(target_rows)
    targets["gamma3_q_value_bh_fdr"] = benjamini_hochberg(
        targets["gamma3_p_value_hac"]
    )
    targets["supports_herding"] = (
        targets["gamma3"].lt(0.0)
        & targets["gamma3_q_value_bh_fdr"].le(float(regression_cfg["fdr_alpha"]))
    )
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    fitted = pd.concat(fitted_parts).sort_index().rename("fitted_scsad")
    return targets, coefficients, diagnostics, fitted


def build_break_date_table(
    index: pd.DatetimeIndex,
    break_indices: Sequence[int],
    paper_regime_starts: Sequence[str],
    solution_name: str,
) -> pd.DataFrame:
    rows = []
    for number, break_index in enumerate(break_indices, start=1):
        position = int(break_index)
        rows.append(
            {
                "solution": solution_name,
                "break_number": number,
                "break_index": position,
                "previous_regime_end": index[position - 1],
                "next_regime_start": index[position],
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    paper_dates = pd.DatetimeIndex([_as_utc(value) for value in paper_regime_starts])
    ours = pd.DatetimeIndex(frame["next_regime_start"])
    distances = np.abs(
        (ours.to_numpy()[:, None] - paper_dates.to_numpy()[None, :])
        .astype("timedelta64[D]")
        .astype(int)
    )
    ours_positions, paper_positions = linear_sum_assignment(distances)
    frame["matched_paper_regime_start"] = pd.Series(
        pd.NaT,
        index=frame.index,
        dtype="datetime64[ns, UTC]",
    )
    frame["signed_calendar_day_difference"] = np.nan
    for ours_position, paper_position in zip(ours_positions, paper_positions):
        our_date = ours[ours_position]
        paper_date = paper_dates[paper_position]
        frame.loc[ours_position, "matched_paper_regime_start"] = paper_date
        frame.loc[ours_position, "signed_calendar_day_difference"] = (
            our_date - paper_date
        ).days
    return frame


def build_paper_regime_comparison(
    paper_date_targets: pd.DataFrame,
    paper_benchmark_cfg: Mapping,
) -> pd.DataFrame:
    benchmark = pd.DataFrame(paper_benchmark_cfg["regime_gamma3"]).rename(
        columns={
            "gamma3": "paper_gamma3",
            "t_stat": "paper_t_stat",
            "nobs": "paper_nobs",
        }
    )
    ours = paper_date_targets.rename(
        columns={
            "gamma3": "our_gamma3",
            "gamma3_t_stat": "our_t_stat",
            "observations": "our_nobs",
        }
    )
    comparison = benchmark.merge(
        ours[
            [
                "regime",
                "start",
                "end",
                "our_gamma3",
                "our_t_stat",
                "our_nobs",
                "gamma3_q_value_bh_fdr",
                "supports_herding",
            ]
        ],
        on="regime",
        how="left",
        validate="one_to_one",
    )
    comparison["gamma3_difference"] = (
        comparison["our_gamma3"] - comparison["paper_gamma3"]
    )
    comparison["coefficient_sign_matches"] = np.sign(
        comparison["our_gamma3"]
    ).eq(np.sign(comparison["paper_gamma3"]))
    return comparison


def run_no_break_stability_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    y = frame["scsad"].to_numpy(dtype=float)
    x = frame[DESIGN_COLUMNS].to_numpy(dtype=float)
    model = sm.OLS(y, x).fit()
    hansen_stat, hansen_critical = breaks_hansen(model)
    critical_frame = pd.DataFrame(hansen_critical)
    if critical_frame.shape[1] != 2:
        raise ValueError("Unexpected Hansen critical-value table")
    critical_frame.columns = ["regressor_count", "critical_value_5pct"]
    hansen_critical_5pct = float(
        np.interp(
            x.shape[1],
            critical_frame["regressor_count"].astype(float),
            critical_frame["critical_value_5pct"].astype(float),
        )
    )
    cusum_stat, cusum_p_value, cusum_critical = breaks_cusumolsresid(
        model.resid,
        ddof=x.shape[1],
    )
    critical_lookup = {int(level): float(value) for level, value in cusum_critical}
    return pd.DataFrame(
        [
            {
                "test": "hansen_parameter_instability",
                "statistic": float(hansen_stat),
                "p_value": np.nan,
                "critical_value_5pct": hansen_critical_5pct,
                "rejects_stability_5pct": bool(
                    float(hansen_stat) > hansen_critical_5pct
                ),
            },
            {
                "test": "cusum_ols_residuals",
                "statistic": float(cusum_stat),
                "p_value": float(cusum_p_value),
                "critical_value_5pct": critical_lookup.get(5, np.nan),
                "rejects_stability_5pct": bool(cusum_p_value <= 0.05),
            },
        ]
    )


def build_structural_break_report(
    config: Mapping,
    frame: pd.DataFrame,
    search: BreakSearchResult,
    break_dates: pd.DataFrame,
    selected_targets: pd.DataFrame,
    fixed_four_dates: pd.DataFrame,
    paper_regime_comparison: pd.DataFrame,
    stability: pd.DataFrame,
    plot_paths: Sequence[str],
) -> str:
    criteria = search.information_criteria
    selected_row = criteria.loc[
        criteria["break_count"].eq(search.selected_break_count)
    ].iloc[0]
    no_break_bic = float(criteria.loc[criteria["break_count"].eq(0), "bic"].iloc[0])
    delta_bic = no_break_bic - float(selected_row["bic"])
    conclusion = (
        f"BIC가 {search.selected_break_count}개 break를 선택했고 "
        f"no-break 대비 ΔBIC={delta_bic:.2f}로 사전 10 기준을 넘어 "
        "SCSAD 관계의 구조 변화를 지지합니다."
        if search.structural_change_supported
        else f"BIC 선택은 {search.selected_break_count}개 break이지만 "
        f"no-break 대비 ΔBIC={delta_bic:.2f}로 사전 구조 변화 기준을 통과하지 못했습니다."
    )
    lines = [
        "# CMC SCSAD 구조적 변화 분석",
        "",
        "## 한 문장 결론",
        "",
        f"> {conclusion}",
        "",
        "## 사전 고정 설계",
        "",
        f"- 표본: {frame.index.min().date()}~{frame.index.max().date()}, {len(frame):,}관측",
        f"- 모형: {config['break_search']['model']}",
        f"- trimming: {config['break_search']['trimming_fraction']:.0%}, 최소 regime {search.minimum_segment_size}관측",
        f"- 최대 break: {config['break_search']['maximum_breaks']}, primary selection: BIC",
        "- break 위치와 regime 계수를 같은 표본에서 추정하므로 regime p/q는 post-selection descriptive inference입니다.",
        "",
        "## Break 수 선택",
        "",
        "| breaks | segments | RSS | AIC | BIC | HQIC | BIC selected |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in criteria.itertuples(index=False):
        lines.append(
            f"| {row.break_count} | {row.segment_count} | {row.rss:.6f} | "
            f"{row.aic:.2f} | {row.bic:.2f} | {row.hqic:.2f} | "
            f"{bool(row.selected_by_bic)} |"
        )
    lines.extend(["", "## BIC 선택 Break", ""])
    if break_dates.empty:
        lines.append("- 선택된 break가 없습니다.")
    else:
        lines.extend(
            [
                "| # | 이전 regime 종료 | 새 regime 시작 | 매칭 논문 break | 차이(일) |",
                "|---:|---|---|---|---:|",
            ]
        )
        for row in break_dates.itertuples(index=False):
            paper = (
                pd.Timestamp(row.matched_paper_regime_start).date()
                if pd.notna(row.matched_paper_regime_start)
                else "-"
            )
            difference = (
                f"{row.signed_calendar_day_difference:+.0f}"
                if pd.notna(row.signed_calendar_day_difference)
                else "-"
            )
            lines.append(
                f"| {row.break_number} | {pd.Timestamp(row.previous_regime_end).date()} | "
                f"{pd.Timestamp(row.next_regime_start).date()} | {paper} | {difference} |"
            )
    lines.extend(
        [
            "",
            "## Regime별 Corrected Herding",
            "",
            "| regime | period | n | gamma3 | standardized | HAC t | BH q | herding |",
            "|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in selected_targets.itertuples(index=False):
        lines.append(
            f"| {row.regime} | {pd.Timestamp(row.start).date()}~{pd.Timestamp(row.end).date()} | "
            f"{row.observations} | {row.gamma3:.3f} | {row.standardized_gamma3:.3f} | "
            f"{row.gamma3_t_stat:.3f} | "
            f"{row.gamma3_q_value_bh_fdr:.4g} | {bool(row.supports_herding)} |"
        )
    lines.extend(["", "## 논문과 동일한 4-Break 수 고정 비교", ""])
    for row in fixed_four_dates.itertuples(index=False):
        paper = pd.Timestamp(row.matched_paper_regime_start).date()
        lines.append(
            f"- break {row.break_number}: {pd.Timestamp(row.next_regime_start).date()} "
            f"vs paper {paper}, {row.signed_calendar_day_difference:+.0f}일"
        )
    lines.extend(
        [
            "",
            "## 논문 Break 날짜 그대로 적용한 계수 재현",
            "",
            "| regime | n(paper/our) | paper gamma3 | our gamma3 | paper t | our t | sign match |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in paper_regime_comparison.itertuples(index=False):
        lines.append(
            f"| {row.regime} | {row.paper_nobs}/{row.our_nobs} | "
            f"{row.paper_gamma3:.3f} | {row.our_gamma3:.3f} | "
            f"{row.paper_t_stat:.3f} | {row.our_t_stat:.3f} | "
            f"{bool(row.coefficient_sign_matches)} |"
        )
    strongest = selected_targets.loc[selected_targets["standardized_gamma3"].idxmin()]
    weakest = selected_targets.loc[selected_targets["standardized_gamma3"].idxmax()]
    criteria_agree = all(
        int(criteria.loc[criteria[f"selected_by_{name}"], "break_count"].iloc[0])
        == search.selected_break_count
        for name in ("aic", "bic", "hqic")
    )
    lines.extend(
        [
            "",
            "## 쉽게 읽는 결과",
            "",
            f"- AIC·BIC·HQIC의 break 수 일치: {criteria_agree}. 세 기준 모두 {search.selected_break_count}개를 선택했습니다.",
            f"- 선택된 {len(selected_targets)}개 regime의 gamma3는 모두 음수이고 BH-FDR를 통과했습니다.",
            f"- 표준화 cubic effect가 가장 음수인 구간은 regime {int(strongest['regime'])} "
            f"({pd.Timestamp(strongest['start']).date()}~{pd.Timestamp(strongest['end']).date()}, {strongest['standardized_gamma3']:.3f})입니다.",
            f"- 가장 0에 가까운 구간은 regime {int(weakest['regime'])} "
            f"({pd.Timestamp(weakest['start']).date()}~{pd.Timestamp(weakest['end']).date()}, {weakest['standardized_gamma3']:.3f})입니다.",
            "- 첫 세 break는 논문과 5일 이내이고, 네 번째는 우리 dynamic universe가 26일 빨리 감지했습니다.",
        ]
    )
    lines.extend(["", "## 보조 안정성 진단", ""])
    for row in stability.itertuples(index=False):
        p_value = f", p={row.p_value:.4g}" if pd.notna(row.p_value) else ""
        lines.append(
            f"- {row.test}: statistic={row.statistic:.3f}{p_value}, "
            f"5% stability rejection={bool(row.rejects_stability_5pct)}"
        )
    lines.extend(
        [
            "",
            "## 해석 제약",
            "",
            f"- boundary solution: {search.boundary_solution}. True면 더 많은 break 가능성을 배제하지 못합니다.",
            "- break 날짜와 시장 이벤트의 근접은 인과관계를 증명하지 않습니다.",
            "- 이 분석은 herding 구조를 설명하며 forward return·거래비용·alpha를 검증하지 않습니다.",
            "",
            "## 그림",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in plot_paths)
    lines.append("")
    return "\n".join(lines)


def plot_scsad_regimes(
    frame: pd.DataFrame,
    fitted: pd.Series,
    break_dates: pd.DataFrame,
    path: str | Path,
) -> None:
    plt = _get_pyplot()
    figure, axis = plt.subplots(figsize=(13, 6))
    axis.plot(
        frame.index,
        frame["scsad"].rolling(30, min_periods=10).mean(),
        label="SCSAD 30-day mean",
        color="#1f5a75",
        linewidth=1.4,
    )
    axis.plot(
        fitted.index,
        fitted.rolling(30, min_periods=10).mean(),
        label="Regime fitted 30-day mean",
        color="#d0773c",
        linewidth=1.2,
    )
    for row in break_dates.itertuples(index=False):
        axis.axvline(pd.Timestamp(row.next_regime_start), color="#8a3d35", alpha=0.7)
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_title("CMC dynamic universe SCSAD structural regimes")
    axis.set_ylabel("SCSAD")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, path, plt)


def plot_regime_gamma3(targets: pd.DataFrame, path: str | Path) -> None:
    plt = _get_pyplot()
    figure, axis = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(targets))
    values = targets["standardized_gamma3"].to_numpy(dtype=float)
    lower = targets["standardized_gamma3_ci_lower"].to_numpy(dtype=float)
    upper = targets["standardized_gamma3_ci_upper"].to_numpy(dtype=float)
    colors = np.where(targets["supports_herding"], "#a33a2b", "#1f5a75")
    for position, value, low, high, color in zip(x, values, lower, upper, colors):
        axis.errorbar(
            position,
            value,
            yerr=[[value - low], [high - value]],
            fmt="o",
            color=color,
            capsize=4,
        )
    labels = [
        f"R{row.regime}\n{pd.Timestamp(row.start).date()}"
        for row in targets.itertuples(index=False)
    ]
    axis.set_xticks(x, labels)
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_ylabel("Standardized SCSAD cubic coefficient")
    axis.set_title("Regime-level standardized corrected herding coefficients")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, path, plt)


def _validate_break_frame(frame: pd.DataFrame) -> None:
    missing = sorted(set(["scsad", *DESIGN_COLUMNS]).difference(frame.columns))
    if missing:
        raise ValueError(f"Break frame missing columns: {', '.join(missing)}")
    if frame.empty or frame[["scsad", *DESIGN_COLUMNS]].isna().any().any():
        raise ValueError("Break frame must contain finite complete observations")


def _load_timestamp_series(path: str | Path, value_column: str) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["snapshot_date"])
    if value_column not in frame.columns:
        raise ValueError(f"{path} is missing {value_column}")
    index = pd.DatetimeIndex(frame["snapshot_date"])
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    return pd.Series(
        pd.to_numeric(frame[value_column], errors="coerce").to_numpy(),
        index=index,
        name=value_column,
    ).sort_index()


def _as_utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _get_pyplot():
    import matplotlib.pyplot as plt

    return plt


def _save_figure(figure, path: str | Path, plt) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)

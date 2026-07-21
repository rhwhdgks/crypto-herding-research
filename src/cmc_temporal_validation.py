from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def evaluate_temporal_persistence(
    targets: pd.DataFrame,
    decision_cfg: Mapping,
    period: str = "holdout_full",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = list(decision_cfg["required_models"])
    frequencies = list(decision_cfg["required_frequencies"])
    expected = {(frequency, model) for frequency in frequencies for model in models}
    alpha = float(decision_cfg["alpha"])
    detail_frames = []
    summary_rows = []
    for variant, subset in targets.loc[targets["period"].eq(period)].groupby("variant"):
        detail = subset.loc[
            subset["frequency"].isin(frequencies) & subset["model"].isin(models)
        ].copy()
        observed = set(zip(detail["frequency"], detail["model"]))
        if observed != expected:
            missing = sorted(expected.difference(observed))
            raise ValueError(f"Temporal persistence cells are incomplete for {variant}: {missing}")
        detail["negative_target"] = detail["coefficient"].lt(0)
        detail["passes_q"] = detail["q_value_bh_fdr"].le(alpha)
        detail["passes_required_cell"] = detail["negative_target"] & detail["passes_q"]
        detail_frames.append(detail)
        summary_rows.append(
            {
                "variant": variant,
                "period": period,
                "required_cells": len(expected),
                "passed_cells": int(detail["passes_required_cell"].sum()),
                "all_required_cells_pass": bool(detail["passes_required_cell"].all()),
                "maximum_required_q_value": float(detail["q_value_bh_fdr"].max()),
            }
        )
    if not detail_frames:
        raise ValueError(f"No temporal validation targets found for period={period}")
    return pd.concat(detail_frames, ignore_index=True), pd.DataFrame(summary_rows)


def compare_historical_and_holdout(
    historical_targets: pd.DataFrame,
    holdout_targets: pd.DataFrame,
    comparison_cfg: Mapping,
) -> pd.DataFrame:
    historical = historical_targets.loc[
        historical_targets["variant"].eq(comparison_cfg["historical_variant"])
        & historical_targets["period"].eq(comparison_cfg["historical_period"])
    ].copy()
    holdout = holdout_targets.loc[
        holdout_targets["variant"].eq(comparison_cfg["holdout_variant"])
        & holdout_targets["period"].eq(comparison_cfg["holdout_period"])
    ].copy()
    keys = ["frequency", "model", "target_term"]
    historical = historical.rename(
        columns={
            "coefficient": "historical_coefficient",
            "standardized_target_coefficient": "historical_standardized_coefficient",
            "t_stat": "historical_t_stat",
            "q_value_bh_fdr": "historical_q_value",
            "supports_herding": "historical_supports_herding",
            "nobs": "historical_nobs",
        }
    )
    holdout = holdout.rename(
        columns={
            "coefficient": "holdout_coefficient",
            "standardized_target_coefficient": "holdout_standardized_coefficient",
            "t_stat": "holdout_t_stat",
            "q_value_bh_fdr": "holdout_q_value",
            "supports_herding": "holdout_supports_herding",
            "nobs": "holdout_nobs",
        }
    )
    columns = keys + [
        "historical_coefficient", "historical_standardized_coefficient", "historical_t_stat", "historical_q_value",
        "historical_supports_herding", "historical_nobs",
    ]
    other_columns = keys + [
        "holdout_coefficient", "holdout_standardized_coefficient", "holdout_t_stat", "holdout_q_value",
        "holdout_supports_herding", "holdout_nobs",
    ]
    comparison = historical[columns].merge(
        holdout[other_columns], on=keys, how="outer", validate="one_to_one"
    )
    if comparison[["historical_coefficient", "holdout_coefficient"]].isna().any().any():
        raise ValueError("Historical and holdout target cells do not align")
    comparison["coefficient_difference"] = (
        comparison["holdout_coefficient"] - comparison["historical_coefficient"]
    )
    comparison["absolute_magnitude_ratio"] = (
        comparison["holdout_coefficient"].abs()
        / comparison["historical_coefficient"].abs().replace(0, np.nan)
    )
    comparison["standardized_coefficient_difference"] = (
        comparison["holdout_standardized_coefficient"]
        - comparison["historical_standardized_coefficient"]
    )
    comparison["standardized_absolute_magnitude_ratio"] = (
        comparison["holdout_standardized_coefficient"].abs()
        / comparison["historical_standardized_coefficient"].abs().replace(0, np.nan)
    )
    comparison["coefficient_sign_matches"] = np.sign(
        comparison["holdout_coefficient"]
    ).eq(np.sign(comparison["historical_coefficient"]))
    comparison["support_decision_matches"] = comparison[
        "holdout_supports_herding"
    ].eq(comparison["historical_supports_herding"])
    return comparison.sort_values(["frequency", "model"]).reset_index(drop=True)


def build_temporal_validation_report(
    config: Mapping,
    asset_coverage: pd.DataFrame,
    quality: pd.DataFrame,
    panels_by_variant: Mapping[str, Mapping],
    targets: pd.DataFrame,
    decision_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    plot_paths: Sequence[str],
) -> str:
    primary = decision_summary.set_index("variant").loc["replication_primary"]
    sensitivity = decision_summary.set_index("variant").loc["no_lookahead_sensitivity"]
    primary_status = "재현" if bool(primary["all_required_cells_pass"]) else "미재현"
    timing_status = "통과" if bool(sensitivity["all_required_cells_pass"]) else "미통과"
    corrected_comparison = comparison.loc[
        comparison["model"].isin(config["decision"]["required_models"])
    ]
    lines = [
        "# CMC Fixed-62 시간 외부표본 검증",
        "",
        "## 결론",
        "",
        f"- direct-replication primary corrected 4개 셀: {int(primary['passed_cells'])}/4 통과, 시간 외부표본 {primary_status}",
        f"- no-look-ahead sensitivity corrected 4개 셀: {int(sensitivity['passed_cells'])}/4 통과, timing 강건성 {timing_status}",
        f"- corrected 표준화 절대크기는 과거 대비 {corrected_comparison['standardized_absolute_magnitude_ratio'].min():.2f}~{corrected_comparison['standardized_absolute_magnitude_ratio'].max():.2f}배입니다.",
        "- 이 검정은 corrected CSAD 관계의 시간 지속성을 다루며 미래수익률 alpha 검정이 아닙니다.",
        "",
        "## 표본과 품질",
        "",
        f"- 분석 기간: {config['analysis']['start']}~{config['analysis']['end']}",
        f"- 고정 universe: {len(asset_coverage)}개 legacy CMC ID, 구성종목 교체 없음",
        f"- 전체 asset-day coverage: {asset_coverage['observations'].sum() / (asset_coverage['expected_days'].sum()):.2%}",
        f"- 자산별 최저 coverage: {asset_coverage['coverage_share'].min():.2%}",
        f"- quality gate: {int(quality['passes'].sum())}/{len(quality)} 통과",
    ]
    for variant, panels in panels_by_variant.items():
        lines.append(
            f"- {variant}: daily {len(panels['daily_market']):,}개, weekly {len(panels['weekly_market']):,}개"
        )
    lines.extend(["", "## Full Holdout 회귀", ""])
    full = targets.loc[targets["period"].eq("holdout_full")]
    for row in full.itertuples():
        lines.append(
            f"- {row.variant} / {row.frequency} / {row.model}: 계수 {row.coefficient:.4f}, "
            f"t={row.t_stat:.2f}, q={row.q_value_bh_fdr:.3g}, 통과={bool(row.supports_herding)}"
        )
    lines.extend(["", "## 2018-2024 대비 Holdout", ""])
    for row in comparison.itertuples():
        lines.append(
            f"- {row.frequency} / {row.model}: 과거 {row.historical_coefficient:.4f}, "
            f"holdout {row.holdout_coefficient:.4f}, 차이 {row.coefficient_difference:+.4f}, "
            f"표준화 계수 {row.historical_standardized_coefficient:.3f}→"
            f"{row.holdout_standardized_coefficient:.3f}, 표준화 절대크기 비율 "
            f"{row.standardized_absolute_magnitude_ratio:.2f}"
        )
    lines.extend(["", "## 기간 진단", ""])
    diagnostic = targets.loc[
        targets["variant"].eq("replication_primary")
        & targets["period"].isin(["holdout_year1", "holdout_later"])
        & targets["model"].isin(config["decision"]["required_models"])
    ]
    for row in diagnostic.itertuples():
        lines.append(
            f"- {row.period} / {row.frequency} / {row.model}: 계수 {row.coefficient:.4f}, "
            f"t={row.t_stat:.2f}, q={row.q_value_bh_fdr:.3g}, 통과={bool(row.supports_herding)}"
        )
    lines.extend(
        [
            "",
            "## 해석 제한",
            "",
            "- 동일 CMC 공급자를 사용한 시간 외부검증이며 공급자 외부검증은 아닙니다.",
            "- 당일 시총가중 primary는 논문 재현용이며 예측 가능한 투자 가중치가 아닙니다.",
            "- corrected CSAD는 intentional imitation이나 거래 가능한 수익률을 직접 식별하지 않습니다.",
            "- 비선형 원계수는 수익률 변동성 단위에 민감하므로 시기별 강도 비교는 표준화 계수를 우선합니다.",
            "- 하위기간 결과는 진단이며 full holdout 판정을 대체하지 않습니다.",
            "",
            "## 그림",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in plot_paths)
    return "\n".join(lines) + "\n"


def plot_temporal_comparison(comparison: pd.DataFrame, destination: str | Path) -> None:
    labels = (comparison["frequency"] + " / " + comparison["model"]).tolist()
    x = np.arange(len(labels))
    width = 0.38
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.bar(x - width / 2, comparison["historical_standardized_coefficient"], width, label="2018-2024")
    axis.bar(x + width / 2, comparison["holdout_standardized_coefficient"], width, label="2024-2026 holdout")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_title("CMC fixed-62 standardized target coefficients")
    axis.legend()
    figure.tight_layout()
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from csad_specification_audit import ORIGINAL_MODELS


MODEL_LABELS = {
    "standard_csad": "Standard CSAD",
    "no_intercept_csad": "No-intercept CSAD",
    "intercept_restored": "Intercept-restored",
    "scsad": "SCSAD",
}

plt.rcParams["axes.unicode_minus"] = False


def build_structural_robustness_decisions(
    model_diagnostics: pd.DataFrame,
    regime_results: pd.DataFrame,
    false_positive_summary: pd.DataFrame,
    config: Mapping,
) -> pd.DataFrame:
    rows: list[dict] = []
    threshold = float(config["simulation"]["false_positive_inflated_threshold"])
    mapping = config["null_mapping"]
    empirical = model_diagnostics.loc[model_diagnostics["model"].isin(ORIGINAL_MODELS)]
    for dataset_id, group in empirical.groupby("dataset_id"):
        baseline = group.loc[group["benchmark"].eq("baseline")]
        loo = group.loc[group["benchmark"].eq("leave_one_out")]
        baseline_supported = int(baseline["supports_negative_relation"].sum())
        loo_supported = int(loo["supports_negative_relation"].sum())
        baseline_corrected = baseline.loc[
            baseline["model"].isin(("no_intercept_csad", "scsad"))
        ]
        loo_corrected = loo.loc[loo["model"].isin(("no_intercept_csad", "scsad"))]
        baseline_corrected_supported = int(
            baseline_corrected["supports_negative_relation"].sum()
        )
        loo_corrected_supported = int(loo_corrected["supports_negative_relation"].sum())
        null_rows = []
        for frequency in ("daily", "weekly"):
            scenario = str(mapping[dataset_id][frequency])
            null_rows.append(
                false_positive_summary.loc[
                    false_positive_summary["scenario"].eq(scenario)
                    & false_positive_summary["model"].isin(ORIGINAL_MODELS)
                ]
            )
        null_frame = pd.concat(null_rows, ignore_index=True)
        worst_fpr = float(null_frame["bh_false_positive_rate"].max())
        regime = regime_results.loc[regime_results["dataset_id"].eq(dataset_id)]
        opposite_count = int(regime["significant_opposite_positive"].sum())
        rows.append(
            {
                "dataset_id": dataset_id,
                "provider": baseline["provider"].iloc[0],
                "sample": baseline["sample"].iloc[0],
                "universe_type": baseline["universe_type"].iloc[0],
                "weighting_class": baseline["weighting_class"].iloc[0],
                "baseline_supported_cells": baseline_supported,
                "baseline_required_cells": 6,
                "baseline_all_six": baseline_supported == 6,
                "baseline_corrected_supported_cells": baseline_corrected_supported,
                "baseline_corrected_required_cells": 4,
                "baseline_corrected_all_four": baseline_corrected_supported == 4,
                "loo_supported_cells": loo_supported,
                "loo_required_cells": 6,
                "loo_all_six": loo_supported == 6,
                "loo_corrected_supported_cells": loo_corrected_supported,
                "loo_corrected_required_cells": 4,
                "loo_corrected_all_four": loo_corrected_supported == 4,
                "worst_matched_null_bh_false_positive_rate": worst_fpr,
                "null_fpr_within_tolerance": worst_fpr <= threshold,
                "significant_opposite_regime_cells": opposite_count,
                "regime_consistent": opposite_count == 0,
                "structurally_robust": bool(
                    baseline_supported == 6
                    and loo_supported == 6
                    and worst_fpr <= threshold
                    and opposite_count == 0
                ),
                "interpretation": "nonlinear_convergence_relation_only_not_imitation_or_alpha",
            }
        )
    return pd.DataFrame(rows)


def build_intercept_verdict(
    empirical_mechanics: pd.DataFrame,
    null_mechanics: pd.DataFrame,
    false_positive_summary: pd.DataFrame,
    config: Mapping,
) -> pd.DataFrame:
    decision_cfg = config["decision"]
    empirical = empirical_mechanics.loc[
        empirical_mechanics["benchmark"].eq("baseline")
        & empirical_mechanics["no_intercept_supports_negative_relation"]
    ]
    empirical_share = float(empirical["mechanical_change"].mean()) if len(empirical) else np.nan
    no_intercept_fpr = false_positive_summary.loc[
        false_positive_summary["model"].eq("no_intercept_csad"),
        ["dgp", "scenario", "bh_false_positive_rate"],
    ]
    null = null_mechanics.merge(
        no_intercept_fpr, on=["dgp", "scenario"], how="left", validate="one_to_one"
    )
    threshold = float(config["simulation"]["false_positive_inflated_threshold"])
    null["null_mechanical_cell"] = (
        null["bh_false_positive_rate"].gt(threshold)
        & null["mechanical_change_rate"].ge(
            float(decision_cfg["intercept_effect_required_share"])
        )
    )
    null_share = float(null["null_mechanical_cell"].mean()) if len(null) else np.nan
    empirical_pass = bool(
        np.isfinite(empirical_share)
        and empirical_share >= float(decision_cfg["intercept_effect_required_share"])
    )
    null_pass = bool(
        np.isfinite(null_share)
        and null_share >= float(decision_cfg["null_effect_required_share"])
    )
    return pd.DataFrame(
        [
            {
                "empirical_supported_no_intercept_cells": len(empirical),
                "empirical_mechanical_change_cells": int(empirical["mechanical_change"].sum()),
                "empirical_mechanical_change_share": empirical_share,
                "empirical_axis_passes": empirical_pass,
                "null_dgp_scenario_cells": len(null),
                "null_mechanical_cells": int(null["null_mechanical_cell"].sum()),
                "null_mechanical_share": null_share,
                "null_axis_passes": null_pass,
                "substantive_intercept_mechanism_evidence": empirical_pass and null_pass,
            }
        ]
    )


def plot_false_positive_heatmaps(summary: pd.DataFrame, output_path: str | Path) -> None:
    data = summary.loc[summary["model"].isin(ORIGINAL_MODELS)]
    scenarios = data["scenario"].drop_duplicates().tolist()
    dgps = data["dgp"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(ORIGINAL_MODELS), 1, figsize=(16, 10), constrained_layout=True)
    for axis, model_name in zip(axes, ORIGINAL_MODELS, strict=True):
        pivot = (
            data.loc[data["model"].eq(model_name)]
            .pivot(index="dgp", columns="scenario", values="bh_false_positive_rate")
            .reindex(index=dgps, columns=scenarios)
        )
        image = axis.imshow(pivot.to_numpy(), vmin=0.0, vmax=max(0.15, float(data["bh_false_positive_rate"].max())), cmap="YlOrRd", aspect="auto")
        axis.set_title(f"{MODEL_LABELS[model_name]}: BH false-positive rate")
        axis.set_yticks(np.arange(len(dgps)), dgps)
        axis.set_xticks(np.arange(len(scenarios)), scenarios, rotation=45, ha="right")
        for row in range(len(dgps)):
            for column in range(len(scenarios)):
                value = pivot.iloc[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.1%}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value >= 0.55 else "black",
                )
        fig.colorbar(image, ax=axis, fraction=0.02, pad=0.01)
    _save_figure(fig, output_path)


def plot_intercept_mechanics(comparison: pd.DataFrame, output_path: str | Path) -> None:
    frame = comparison.loc[comparison["benchmark"].eq("baseline")]
    x = frame["no_intercept_target_standardized_coefficient"]
    y = frame["restored_target_standardized_coefficient"]
    bounds = [float(min(x.min(), y.min())), float(max(x.max(), y.max()))]
    fig, axis = plt.subplots(figsize=(8, 7))
    axis.scatter(x, y, c=frame["frequency"].map({"daily": "#C14924", "weekly": "#2A6F97"}), s=55, alpha=0.85)
    axis.plot(bounds, bounds, color="black", linestyle="--", linewidth=1, label="No change")
    axis.axhline(0.0, color="grey", linewidth=0.8)
    axis.axvline(0.0, color="grey", linewidth=0.8)
    for row in frame.itertuples():
        axis.annotate(
            row.dataset_id.replace("_", " "),
            (row.no_intercept_target_standardized_coefficient, row.restored_target_standardized_coefficient),
            fontsize=6,
            alpha=0.8,
        )
    axis.set_xlabel("No-intercept standardized target")
    axis.set_ylabel("Intercept-restored standardized target")
    axis.set_title("Target coefficient before and after restoring the intercept")
    axis.legend()
    fig.tight_layout()
    _save_figure(fig, output_path)


def plot_empirical_null_comparison(comparison: pd.DataFrame, output_path: str | Path) -> None:
    aggregated = comparison.groupby(
        ["dataset_id", "frequency", "model"], as_index=False
    ).agg(
        empirical=("empirical_standardized_coefficient", "first"),
        null_lower=("null_ci_lower", "min"),
        null_upper=("null_ci_upper", "max"),
    )
    labels = (
        aggregated[["dataset_id", "frequency"]]
        .drop_duplicates()
        .assign(label=lambda frame: frame["dataset_id"] + " / " + frame["frequency"])
    )
    ordered_labels = labels["label"].tolist()
    fig, axes = plt.subplots(1, len(ORIGINAL_MODELS), figsize=(18, 9), sharey=True, constrained_layout=True)
    for axis, model_name in zip(axes, ORIGINAL_MODELS, strict=True):
        frame = aggregated.loc[aggregated["model"].eq(model_name)].copy()
        frame["label"] = frame["dataset_id"] + " / " + frame["frequency"]
        frame = frame.set_index("label").reindex(ordered_labels)
        positions = np.arange(len(frame))
        axis.hlines(
            positions,
            frame["null_lower"],
            frame["null_upper"],
            color="#95D5B2",
            linewidth=2.0,
            label="Widest DGP null 95% interval" if axis is axes[0] else None,
        )
        axis.scatter(
            frame["empirical"],
            positions,
            marker="D",
            color="#1B4332",
            s=28,
            zorder=3,
            label="Empirical coefficient" if axis is axes[0] else None,
        )
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_title(MODEL_LABELS[model_name])
        axis.set_xlabel("Empirical coefficient and widest DGP null 95% interval")
        axis.set_yticks(positions)
        axis.invert_yaxis()
    axes[0].set_yticklabels(ordered_labels)
    for axis in axes[1:]:
        axis.tick_params(axis="y", labelleft=False)
    axes[0].legend(loc="lower left", fontsize=8)
    _save_figure(fig, output_path)


def plot_random_effects(summary: pd.DataFrame, output_path: str | Path) -> None:
    frame = summary.copy()
    frame["label"] = frame["model"].map(MODEL_LABELS) + " / " + frame["frequency"]
    positions = np.arange(len(frame))
    estimate = frame["random_effect_standardized_coefficient"]
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.errorbar(
        estimate,
        positions,
        xerr=np.vstack([estimate - frame["ci_lower"], frame["ci_upper"] - estimate]),
        fmt="o",
        color="#264653",
        ecolor="#2A9D8F",
        capsize=4,
    )
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_yticks(positions, frame["label"])
    axis.invert_yaxis()
    axis.set_xlabel("Descriptive random-effects standardized coefficient")
    axis.set_title("Descriptive heterogeneity across provider, universe and weighting")
    fig.tight_layout()
    _save_figure(fig, output_path)


def build_korean_audit_report(
    tables: Mapping[str, pd.DataFrame],
    plot_paths: Sequence[str],
    protocol_hash: str,
    config_hash: str,
) -> str:
    diagnostics = tables["model_diagnostics"]
    decisions = tables["structural_robustness_decisions"]
    intercept = tables["intercept_verdict"].iloc[0]
    fpr = tables["false_positive_summary"]
    comparison = tables["empirical_vs_null"]
    conditional = tables["conditional_concentration_results"]
    regimes = tables["volatility_regime_results"]
    random_effects = tables["descriptive_random_effects"]
    meta_diagnostics = tables["descriptive_meta_regression_diagnostics"]
    integrity = tables["input_integrity"]

    robust = decisions.loc[decisions["structurally_robust"]]
    inflated = fpr.loc[
        fpr["model"].isin(ORIGINAL_MODELS)
        & fpr["fpr_classification"].ne("within_preregistered_tolerance")
    ]
    empirical_null_support = int(comparison["more_negative_than_null_bh"].sum())
    lines = [
        "# Corrected CSAD Specification and Mechanism Audit v1",
        "",
        "## 한눈에 보는 결론",
        "",
        f"- 사전등록한 구조적 강건성 기준을 모두 통과한 10개 empirical 사양은 **{len(robust)}개**입니다.",
        f"- 절편 제약의 실질적 기계성 판정은 **{bool(intercept.substantive_intercept_mechanism_evidence)}**입니다. Empirical 변화 비율은 {intercept.empirical_mechanical_change_share:.1%}, null 변화 비율은 {intercept.null_mechanical_share:.1%}입니다.",
        f"- 4개 비허딩 DGP × 고정 시나리오에서 사전 허용치 7.5%를 넘은 기존 모형 false-positive 셀은 **{len(inflated)}개/{len(fpr.loc[fpr['model'].isin(ORIGINAL_MODELS)])}개**입니다.",
        f"- Empirical 계수가 대응 simulation null보다 더 음수라고 BH-FDR로 판정된 비교는 **{empirical_null_support}개/{len(comparison)}개**입니다.",
        "- 이 결과는 동시적 CSAD형 비선형 관계의 사양 민감도 감사이며, intentional imitation이나 미래수익률 alpha의 증거가 아닙니다.",
        "",
        "## 무엇을 감사했나",
        "",
        "- CMC fixed-62 historical·holdout, Binance fixed-14, OKX listing-aware 14, Binance archive point-in-time Top-50의 기존 중간 패널을 통합했습니다.",
        "- 각 표본의 equal/current/lagged weighting과 daily·weekly를 동일한 코드로 재계산했습니다.",
        "- Standard, no-intercept, SCSAD와 함께 no-intercept의 설명변수에 절편만 복원한 기계성 control을 적합했습니다.",
        "- 자기 자신을 시장수익률에서 제외한 leave-one-out, 과거정보 전용 volatility regime, N·HHI·BTC weight 교호항을 검사했습니다.",
        "- 독립 Gaussian, 공통요인, 이분산 공통요인, fat-tail 상관 DGP에는 자산 간 모방 규칙을 넣지 않았습니다.",
        "",
        "## 입력 무결성",
        "",
        f"- 읽은 패널: {len(integrity)}개, 저장 market/CSAD와 재계산값이 허용오차 내 일치한 패널: {int(integrity['rebuild_matches_stored'].sum())}/{len(integrity)}개",
        f"- protocol SHA-256: `{protocol_hash}`",
        f"- config SHA-256: `{config_hash}`",
        "",
        "## 절편 제거의 영향",
        "",
        f"- Baseline에서 음의 BH 지지를 보인 no-intercept 셀은 {int(intercept.empirical_supported_no_intercept_cells)}개였고, 이 중 지지가 사라지거나 표준화 절대크기가 50% 이상 줄어든 셀은 {int(intercept.empirical_mechanical_change_cells)}개였습니다.",
        f"- 비허딩 null의 DGP·시나리오 {int(intercept.null_dgp_scenario_cells)}개 중 사전 기계성 조건을 만족한 셀은 {int(intercept.null_mechanical_cells)}개였습니다.",
        "- No-intercept의 음의 계수 전체를 절편 하나로 설명할 수 있는지와, 일부 사양에서 절편 제약이 계수를 증폭하는지는 구분해서 봐야 합니다.",
        "",
        "## 구조적 강건성 판정",
        "",
    ]
    for row in decisions.itertuples():
        lines.append(
            f"- `{row.dataset_id}`: 전체 3모형 baseline {row.baseline_supported_cells}/6, LOO {row.loo_supported_cells}/6; "
            f"기존 corrected 2모형 baseline {row.baseline_corrected_supported_cells}/4, LOO {row.loo_corrected_supported_cells}/4; "
            f"최악 null FPR {row.worst_matched_null_bh_false_positive_rate:.1%}, 반대 regime {row.significant_opposite_regime_cells}개, "
            f"최종 통과={bool(row.structurally_robust)}"
        )
    lines.extend(
        [
            "",
            "## Synthetic null false positive",
            "",
        ]
    )
    for model_name in ORIGINAL_MODELS:
        model_fpr = fpr.loc[fpr["model"].eq(model_name)]
        maximum = model_fpr.loc[model_fpr["bh_false_positive_rate"].idxmax()]
        lines.append(
            f"- {MODEL_LABELS[model_name]}: 최댓값 {maximum.bh_false_positive_rate:.1%} "
            f"({maximum.dgp}, {maximum.scenario}), 허용치 초과 {int(model_fpr['bh_false_positive_rate'].gt(0.075).sum())}/{len(model_fpr)}개"
        )
    lines.extend(
        [
            "",
            "## 조건부 분석",
            "",
            f"- N·HHI·BTC weight 교호항 중 식별 가능: {int(conditional['status'].eq('estimated').sum())}개, BH-FDR 통과: {int(conditional['interaction_supported'].sum())}개",
            f"- 사전 고정 volatility regime 회귀 중 추정 가능: {int(regimes['status'].eq('estimated').sum())}개, 유의한 반대 양의 비선형항: {int(regimes['significant_opposite_positive'].sum())}개",
            "- 고정 14·62 universe에서 N 변화가 거의 없는 경우는 억지로 추정하지 않고 `not_identified`로 남겼습니다.",
            "",
            "## 이질성",
            "",
        ]
    )
    for row in random_effects.itertuples():
        lines.append(
            f"- {MODEL_LABELS[row.model]} / {row.frequency}: 표준화 random-effects {row.random_effect_standardized_coefficient:.3f}, "
            f"I-squared {row.i_squared_percent:.1f}%"
        )
    lines.extend(
        [
            "- 표본이 중첩되고 provider·universe·가중법이 서로 교락되어 있으므로 위 수치는 독립 연구 메타분석이나 인과효과가 아닙니다.",
            f"- 전체 moderator 동시 meta-regression은 {int(meta_diagnostics['rank_deficient'].sum())}/{len(meta_diagnostics)}개 모형에서 rank deficient였습니다. 따라서 계수 해석은 하지 않고 moderator별 HC3 단변량·BH-FDR 보조표를 함께 제공합니다.",
            "",
            "## 해석 한계",
            "",
            "- Null DGP는 알려진 네 가지 기계성을 분리하는 반사실적이며 실제 암호화폐 시장 전체를 완전하게 생성하지 않습니다.",
            "- CMC contemporaneous cap weighting은 논문 재현용이고 미래 시점에 실행 가능한 weighting이 아닙니다.",
            "- Survivor bias는 fixed와 point-in-time 표본 차이에 섞여 있으나 provider 차이도 동시에 존재해 단독 인과효과를 식별하지 못합니다.",
            "- 음의 계수가 null보다 극단적이어도 누락 공통요인, 가격결정 구조와 microstructure를 배제하지 못합니다.",
            "- 이 감사에는 미래수익률, 비용, portfolio backtest가 없으므로 자동매매 근거로 사용할 수 없습니다.",
            "",
            "## 재현 산출물",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in plot_paths)
    lines.extend(
        [
            "- 주요 CSV와 simulation parquet, input manifest, config/protocol snapshot은 같은 출력 폴더에 보존했습니다.",
            "",
            "## 최종 연구 판단",
            "",
            "이 감사의 판정은 사전등록 기준을 기계적으로 적용한 것입니다. 통과 셀이 있더라도 표현은 `CSAD형 비선형 수렴 관계`로 제한하며, 통과하지 못한 경우에는 같은 표본에서 threshold나 종목을 다시 최적화하지 않습니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def _save_figure(fig: plt.Figure, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)

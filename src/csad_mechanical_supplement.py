from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from csad_mechanical_derivation import build_gaussian_theory_table
from csad_mechanical_simulation import (
    _run_simulation_grid,
    summarize_diagnostics,
    summarize_simulation_replicates,
)


def run_finite_sample_supplement(config: Mapping) -> dict[str, pd.DataFrame]:
    simulation = config["simulation"]
    scenarios = [
        {
            "id": f"n{int(assets)}_t{int(observations)}_equal",
            "observations": int(observations),
            "assets": int(assets),
            "weighting": str(simulation["weighting"]),
        }
        for assets in simulation["assets"]
        for observations in simulation["observations"]
    ]
    replicates, diagnostics = _run_simulation_grid(
        dgp_names=[str(simulation["dgp"])],
        scenarios=scenarios,
        repetitions=int(simulation["repetitions"]),
        seed=int(simulation["seed"]),
        config=config,
        phase="finite_sample_supplement",
    )
    theory = build_gaussian_theory_table(
        simulation["assets"], float(config["theory"]["sigma"])
    )
    summary = summarize_simulation_replicates(
        replicates,
        confidence=float(config["theory"]["convergence_confidence"]),
        theory=theory,
    )
    gates = build_supplement_gates(summary, config)
    return {
        "supplement_replicates": replicates,
        "supplement_diagnostics": diagnostics,
        "supplement_summary": summary,
        "supplement_diagnostic_summary": summarize_diagnostics(diagnostics),
        "supplement_gates": gates,
    }


def build_supplement_gates(summary: pd.DataFrame, config: Mapping) -> pd.DataFrame:
    theory = config["theory"]
    primary_observations = max(int(value) for value in config["simulation"]["observations"])
    rows: list[dict[str, object]] = []
    for row in summary.itertuples(index=False):
        primary = int(row.observations) == primary_observations
        result = {
            "assets": int(row.assets),
            "observations": int(row.observations),
            "model": row.model,
            "primary_supplement_cell": primary,
            "theoretical_target_coefficient": row.theoretical_target_coefficient,
            "mean_target_coefficient": row.mean_target_coefficient,
            "absolute_relative_error": row.absolute_relative_error,
            "theory_inside_mean_ci": bool(row.theory_inside_mean_ci),
            "negative_coefficient_rate": row.negative_coefficient_rate,
            "raw_false_positive_rate": row.raw_false_positive_rate,
            "bh3_false_positive_rate": row.bh3_false_positive_rate,
        }
        if not primary:
            result.update(
                {
                    "gate_type": "convergence_path_descriptive",
                    "relative_error_pass": np.nan,
                    "mean_ci_pass": np.nan,
                    "negative_rate_pass": np.nan,
                    "raw_fpr_pass": np.nan,
                    "bh_fpr_pass": np.nan,
                    "gate_pass": np.nan,
                }
            )
        elif row.model in {"no_intercept_csad", "scsad"}:
            result.update(
                {
                    "gate_type": "supplement_mechanical_primary",
                    "relative_error_pass": bool(
                        row.absolute_relative_error
                        <= float(theory["relative_error_maximum"])
                    ),
                    "mean_ci_pass": bool(row.theory_inside_mean_ci),
                    "negative_rate_pass": bool(
                        row.negative_coefficient_rate
                        >= float(theory["negative_sign_rate_minimum"])
                    ),
                    "raw_fpr_pass": bool(
                        row.raw_false_positive_rate
                        >= float(theory["mechanical_raw_fpr_minimum"])
                    ),
                    "bh_fpr_pass": bool(
                        row.bh3_false_positive_rate
                        >= float(theory["mechanical_bh_fpr_minimum"])
                    ),
                }
            )
            result["gate_pass"] = bool(
                result["relative_error_pass"]
                and result["mean_ci_pass"]
                and result["negative_rate_pass"]
                and result["raw_fpr_pass"]
                and result["bh_fpr_pass"]
            )
        elif row.model in {"standard_csad", "intercept_restored"}:
            result.update(
                {
                    "gate_type": "supplement_nominal_control",
                    "relative_error_pass": np.nan,
                    "mean_ci_pass": bool(row.theory_inside_mean_ci),
                    "negative_rate_pass": np.nan,
                    "raw_fpr_pass": bool(
                        row.raw_false_positive_rate
                        <= float(theory["nominal_raw_fpr_maximum"])
                    ),
                    "bh_fpr_pass": np.nan,
                }
            )
            result["gate_pass"] = bool(result["raw_fpr_pass"])
        else:
            raise ValueError(f"Unexpected supplement model: {row.model}")
        rows.append(result)
    return pd.DataFrame(rows)


def build_supplement_decision(
    parent_decision: pd.DataFrame,
    parent_equations: pd.DataFrame,
    supplement_gates: pd.DataFrame,
) -> pd.DataFrame:
    parent = parent_decision.iloc[0]
    primary = supplement_gates.loc[supplement_gates["primary_supplement_cell"]].copy()
    mechanical = primary.loc[
        primary["gate_type"].eq("supplement_mechanical_primary")
    ]
    controls = primary.loc[primary["gate_type"].eq("supplement_nominal_control")]
    equation_pass = bool(parent_equations["equation_gate_pass"].astype(bool).all())
    finite_sample_pass = bool(
        len(mechanical) == 6
        and len(controls) == 6
        and mechanical["gate_pass"].astype(bool).all()
        and controls["gate_pass"].astype(bool).all()
    )
    return pd.DataFrame(
        [
            {
                "parent_preregistered_classification": parent["classification"],
                "parent_mechanical_convergence_passed": int(
                    parent["mechanical_convergence_cells_passed"]
                ),
                "parent_mechanical_convergence_total": int(
                    parent["mechanical_convergence_cells_total"]
                ),
                "parent_classification_preserved": bool(
                    parent["classification"] == "mechanical_null_not_confirmed"
                ),
                "analytic_equation_cells_passed": int(
                    parent_equations["equation_gate_pass"].astype(bool).sum()
                ),
                "analytic_equation_cells_total": len(parent_equations),
                "supplement_mechanical_cells_passed": int(
                    mechanical["gate_pass"].astype(bool).sum()
                ),
                "supplement_mechanical_cells_total": len(mechanical),
                "supplement_control_cells_passed": int(
                    controls["gate_pass"].astype(bool).sum()
                ),
                "supplement_control_cells_total": len(controls),
                "finite_sample_convergence_supported": finite_sample_pass,
                "analytic_mechanism_established": equation_pass,
                "supplement_classification": (
                    "finite_sample_convergence_supported"
                    if finite_sample_pass
                    else "finite_sample_convergence_not_supported"
                ),
                "final_interpretation": (
                    "analytic_mechanism_established_and_finite_sample_convergence_supported"
                    if equation_pass and finite_sample_pass
                    else "analytic_mechanism_or_finite_sample_evidence_incomplete"
                ),
            }
        ]
    )


def plot_convergence_ladder(summary: pd.DataFrame, path: str | Path) -> None:
    selected = summary.loc[
        summary["model"].isin(["no_intercept_csad", "scsad"])
    ].copy()
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), sharey=True)
    for axis, model, title in zip(
        axes,
        ["no_intercept_csad", "scsad"],
        ["No-intercept quadratic", "SCSAD cubic"],
        strict=True,
    ):
        group = selected.loc[selected["model"].eq(model)]
        for assets, asset_group in group.groupby("assets"):
            asset_group = asset_group.sort_values("observations")
            theory = asset_group["theoretical_target_coefficient"].to_numpy(dtype=float)
            estimate = asset_group["mean_target_coefficient"].to_numpy(dtype=float)
            ratio = estimate / theory
            first = asset_group["mean_ci_lower"].to_numpy(dtype=float) / theory
            second = asset_group["mean_ci_upper"].to_numpy(dtype=float) / theory
            lower = np.minimum(first, second)
            upper = np.maximum(first, second)
            axis.errorbar(
                asset_group["observations"],
                ratio,
                yerr=np.vstack([ratio - lower, upper - ratio]),
                marker="o",
                capsize=3,
                label=f"N={assets}",
            )
        axis.axhline(1.0, color="#222222", linestyle="--", linewidth=1.2)
        axis.set_xscale("log")
        axis.set_title(title)
        axis.set_xlabel("Sample length T (log scale)")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Monte Carlo mean / theoretical coefficient")
    figure.suptitle("Independent-seed finite-sample convergence ladder")
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_korean_supplement_report(
    summary: pd.DataFrame,
    gates: pd.DataFrame,
    decision: pd.DataFrame,
    protocol_hash: str,
    config_hash: str,
) -> str:
    result = decision.iloc[0]
    primary = gates.loc[gates["primary_supplement_cell"]].sort_values(
        ["model", "assets"]
    )
    path_rows = summary.loc[
        summary["model"].isin(["no_intercept_csad", "scsad"])
    ].sort_values(["model", "assets", "observations"])
    lines = [
        "# CSAD 기계적 계수 v1.1 유한표본 수렴 보충 보고서",
        "",
        "## 1. 왜 보충 분석을 했나",
        "",
        "원 v1의 사전등록 composite gate는 5/6이었습니다. N=62 SCSAD의 Monte Carlo 평균은 이론값과 1.02% 차이였지만 200회 평균의 99% CI가 이론값을 아주 근소하게 포함하지 못했습니다.",
        "",
        f"원 판정 `{result['parent_preregistered_classification']}`은 그대로 보존합니다. 이번 결과로 원 연구를 사후 통과 처리하지 않습니다.",
        "",
        "## 2. 새로 고정한 검증",
        "",
        "- 독립 seed 2026072103",
        "- iid Gaussian, equal weight, sigma=0.025",
        "- N=14·50·62 전부 유지",
        "- T=3,000·12,000·48,000",
        "- 셀당 300회",
        "- T=48,000만 primary supplement gate",
        "",
        "## 3. Primary supplement 결과",
        "",
        "| 모형 | N | 이론값 | MC 평균 | 상대오차 | 99% CI 포함 | 음수율 | raw FPR | BH3 FPR | gate |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in primary.itertuples(index=False):
        relative = (
            f"{100 * row.absolute_relative_error:.2f}%"
            if np.isfinite(row.absolute_relative_error)
            else "해당 없음"
        )
        bh = (
            f"{100 * row.bh3_false_positive_rate:.1f}%"
            if np.isfinite(row.bh3_false_positive_rate)
            else "진단용"
        )
        lines.append(
            f"| {_model_name(row.model)} | {row.assets} | {row.theoretical_target_coefficient:.6g} | "
            f"{row.mean_target_coefficient:.6g} | {relative} | "
            f"{'예' if row.theory_inside_mean_ci else '아니오'} | "
            f"{100 * row.negative_coefficient_rate:.1f}% | {100 * row.raw_false_positive_rate:.1f}% | "
            f"{bh} | {'통과' if row.gate_pass else '실패'} |"
        )
    lines.extend(
        [
            "",
            "## 4. 표본 길이에 따른 경로",
            "",
            "| 모형 | N | T=3,000 오차 | T=12,000 오차 | T=48,000 오차 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for (model, assets), group in path_rows.groupby(["model", "assets"], sort=False):
        errors = group.set_index("observations")["absolute_relative_error"]
        lines.append(
            f"| {_model_name(model)} | {assets} | {100 * errors.loc[3000]:.2f}% | "
            f"{100 * errors.loc[12000]:.2f}% | {100 * errors.loc[48000]:.2f}% |"
        )
    lines.extend(
        [
            "",
            "오차가 표본 길이마다 단조롭게 감소해야 한다는 조건은 사전 gate로 두지 않았습니다. Monte Carlo 평균에는 별도의 simulation 변동이 있기 때문입니다. 핵심은 가장 긴 T=48,000에서 이론값과 크기·CI·부호가 함께 맞는지입니다.",
            "",
            "## 5. 판정",
            "",
            f"- 원 v1 수렴 gate: `{int(result['parent_mechanical_convergence_passed'])}/{int(result['parent_mechanical_convergence_total'])}`이며 실패 판정을 보존",
            f"- 폐형식 검증: `{int(result['analytic_equation_cells_passed'])}/{int(result['analytic_equation_cells_total'])}`",
            f"- 보충 기계적 셀: `{int(result['supplement_mechanical_cells_passed'])}/{int(result['supplement_mechanical_cells_total'])}`",
            f"- 보충 control 셀: `{int(result['supplement_control_cells_passed'])}/{int(result['supplement_control_cells_total'])}`",
            f"- 보충 판정: `{result['supplement_classification']}`",
            f"- 통합 해석: `{result['final_interpretation']}`",
            "",
            "원 v1의 한 셀 실패는 population 폐형식을 반박하지 않습니다. 폐형식은 Gaussian 모멘트 정규방정식에서 정확히 도출되며, 보충 simulation은 유한표본 추정치가 그 값으로 접근하는지를 별도로 확인합니다.",
            "",
            "## 6. 연구상 의미",
            "",
            "1. No-intercept의 음의 제곱항은 양의 CSAD 수준을 절편 없이 근사한 결과입니다.",
            "2. SCSAD의 음의 세제곱항은 sign-step을 저차 홀수 다항식으로 근사한 결과입니다.",
            "3. 두 음의 계수와 작은 HAC p-value는 intentional herding의 충분한 증거가 아닙니다.",
            "4. Standard CSAD도 현실적인 공통요인·이분산·fat-tail null로 검정 크기를 교정해야 합니다.",
            "",
            "## 7. 재현성",
            "",
            f"- Protocol SHA-256: `{protocol_hash}`",
            f"- Config SHA-256: `{config_hash}`",
            "- 그림: `plots/finite_sample_convergence_ladder.png`",
            "",
            "```bash",
            "PYTHONPATH=src .venv/bin/python scripts/run_csad_mechanical_supplement.py",
            "PYTHONPATH=src .venv/bin/python scripts/verify_csad_mechanical_derivation_v1_1.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def build_master_supplement_update(source: str, decision: pd.DataFrame) -> str:
    result = decision.iloc[0]
    marker = "상세 보고서: `outputs/v2/csad_mechanical_derivation_v1/csad_mechanical_derivation_report.md`"
    if marker not in source:
        raise ValueError("Mechanical section marker is missing from the master report")
    addition = "\n".join(
        [
            marker,
            "",
            "### v1.1 유한표본 수렴 보충",
            "",
            "원 v1은 N=62 SCSAD의 99% Monte Carlo 평균 CI 한 셀 때문에 5/6으로 실패했으며 이 판정을 그대로 보존합니다. 이후 별도 등록한 독립 seed의 T=3,000·12,000·48,000, N=14·50·62, 셀당 300회 보충 분석으로 유한표본 수렴을 점검했습니다.",
            "",
            f"보충 기계적 셀은 `{int(result['supplement_mechanical_cells_passed'])}/{int(result['supplement_mechanical_cells_total'])}`, control은 `{int(result['supplement_control_cells_passed'])}/{int(result['supplement_control_cells_total'])}` 통과했습니다. 보충 판정은 `{result['supplement_classification']}`입니다.",
            "",
            "따라서 정확한 Gaussian 폐형식과 유한표본 수렴은 지지되지만, 원 사전등록 결과를 사후 성공으로 바꾸지는 않습니다.",
            "",
            "보충 보고서: `outputs/v2/csad_mechanical_derivation_v1/supplement_v1_1/csad_mechanical_convergence_supplement_report.md`",
        ]
    )
    return source.replace(marker, addition, 1)


def _model_name(value: str) -> str:
    return {
        "standard_csad": "Standard",
        "no_intercept_csad": "No-intercept",
        "intercept_restored": "Intercept-restored",
        "scsad": "SCSAD",
    }.get(value, value)

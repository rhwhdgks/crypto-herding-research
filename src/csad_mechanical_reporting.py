from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from csad_mechanical_derivation import gaussian_closed_form_coefficients


def plot_theory_convergence(summary: pd.DataFrame, path: str | Path) -> None:
    models = ["no_intercept_csad", "scsad"]
    labels = {"no_intercept_csad": "No-intercept quadratic", "scsad": "SCSAD cubic"}
    colors = {"no_intercept_csad": "#C84B31", "scsad": "#1F6E8C"}
    figure, axis = plt.subplots(figsize=(9.5, 5.8))
    for model in models:
        group = summary.loc[summary["model"].eq(model)].sort_values("assets")
        theory = group["theoretical_target_coefficient"].to_numpy(dtype=float)
        estimate = group["mean_target_coefficient"].to_numpy(dtype=float)
        ratio = estimate / theory
        first = group["mean_ci_lower"].to_numpy(dtype=float) / theory
        second = group["mean_ci_upper"].to_numpy(dtype=float) / theory
        lower = np.minimum(first, second)
        upper = np.maximum(first, second)
        axis.errorbar(
            group["assets"],
            ratio,
            yerr=np.vstack([ratio - lower, upper - ratio]),
            marker="o",
            linewidth=2.0,
            capsize=4,
            color=colors[model],
            label=labels[model],
        )
    axis.axhline(1.0, color="#2D2D2D", linestyle="--", linewidth=1.3, label="Exact theory")
    axis.set_xlabel("Number of assets")
    axis.set_ylabel("Monte Carlo mean / theoretical coefficient")
    axis.set_title("Large-sample convergence to Gaussian pseudo-true coefficients")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, path)


def plot_false_positive_contrast(summary: pd.DataFrame, path: str | Path) -> None:
    model_order = [
        "standard_csad",
        "intercept_restored",
        "no_intercept_csad",
        "scsad",
    ]
    labels = ["Standard", "Restored", "No intercept", "SCSAD"]
    pivot = (
        summary.groupby(["dgp", "model"], as_index=False)["raw_false_positive_rate"]
        .mean()
        .pivot(index="dgp", columns="model", values="raw_false_positive_rate")
        .reindex(columns=model_order)
    )
    figure, axis = plt.subplots(figsize=(9.5, 7.0))
    image = axis.imshow(pivot.to_numpy(dtype=float), vmin=0.0, vmax=1.0, cmap="YlOrRd")
    axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=20, ha="right")
    axis.set_yticks(np.arange(len(pivot.index)), labels=[_pretty_dgp(x) for x in pivot.index])
    axis.set_title("Raw negative-sign false-positive rate under non-herding DGPs")
    for row in range(len(pivot.index)):
        for column in range(len(labels)):
            value = float(pivot.iloc[row, column])
            axis.text(
                column,
                row,
                f"{100 * value:.1f}%",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "#222222",
                fontsize=9,
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.03)
    colorbar.set_label("False-positive rate")
    figure.tight_layout()
    _save_figure(figure, path)


def plot_symmetric_sign_robustness(summary: pd.DataFrame, path: str | Path) -> None:
    selected = summary.loc[summary["model"].isin(["no_intercept_csad", "scsad"])].copy()
    dgps = list(dict.fromkeys(selected["dgp"]))
    positions = np.arange(len(dgps), dtype=float)
    figure, axis = plt.subplots(figsize=(11.0, 6.0))
    for offset, (model, label, color) in enumerate(
        [
            ("no_intercept_csad", "No-intercept quadratic", "#C84B31"),
            ("scsad", "SCSAD cubic", "#1F6E8C"),
        ]
    ):
        means = []
        lows = []
        highs = []
        for dgp in dgps:
            values = selected.loc[
                selected["dgp"].eq(dgp) & selected["model"].eq(model),
                "negative_coefficient_rate",
            ].to_numpy(dtype=float)
            means.append(float(np.mean(values)))
            lows.append(float(np.min(values)))
            highs.append(float(np.max(values)))
        x = positions + (-0.11 if offset == 0 else 0.11)
        means_array = np.asarray(means)
        axis.errorbar(
            x,
            means_array,
            yerr=np.vstack([means_array - lows, np.asarray(highs) - means_array]),
            fmt="o",
            capsize=3,
            color=color,
            label=label,
        )
    axis.axhline(0.8, color="#2D2D2D", linestyle="--", linewidth=1.2, label="Preregistered gate")
    axis.set_xticks(positions, labels=[_pretty_dgp(x) for x in dgps], rotation=28, ha="right")
    axis.set_ylim(-0.03, 1.03)
    axis.set_ylabel("Negative coefficient rate across simulations")
    axis.set_title("Mechanical negative sign across non-herding processes")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=3, loc="lower left")
    figure.tight_layout()
    _save_figure(figure, path)


def plot_projection_mechanism(
    asset_scale: float,
    assets: int,
    path: str | Path,
) -> None:
    values = gaussian_closed_form_coefficients(asset_scale, assets)
    market_scale = values["market_scale"]
    expected_csad = values["expected_csad"]
    positive = np.linspace(0.0, 3.5 * market_scale, 400)
    no_intercept_fit = (
        values["no_intercept_abs"] * positive
        + values["no_intercept_target"] * positive**2
    )
    signed = np.linspace(-3.5 * market_scale, 3.5 * market_scale, 801)
    scsad_step = np.sign(signed) * expected_csad
    scsad_fit = values["scsad_linear"] * signed + values["scsad_target"] * signed**3

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    axes[0].plot(positive / market_scale, np.full_like(positive, expected_csad), label="True E[CSAD|M]", color="#2D2D2D")
    axes[0].plot(positive / market_scale, no_intercept_fit, label="No-intercept projection", color="#C84B31", linewidth=2.2)
    axes[0].set_title("Positive CSAD level forced through zero")
    axes[0].set_xlabel("|M| / SD(M)")
    axes[0].set_ylabel("CSAD")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    axes[1].plot(signed / market_scale, scsad_step, label="True sign(M) x E[CSAD]", color="#2D2D2D")
    axes[1].plot(signed / market_scale, scsad_fit, label="Linear + cubic projection", color="#1F6E8C", linewidth=2.2)
    axes[1].set_title("A sign step approximated by an odd cubic")
    axes[1].set_xlabel("M / SD(M)")
    axes[1].set_ylabel("SCSAD")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    figure.suptitle(f"Mechanical curvature under iid Gaussian returns (N={assets})", y=1.02)
    figure.tight_layout()
    _save_figure(figure, path)


def build_korean_mechanical_report(
    tables: Mapping[str, pd.DataFrame],
    protocol_hash: str,
    config_hash: str,
    plot_paths: list[str],
) -> str:
    decision = tables["final_mechanical_decision"].iloc[0]
    convergence = tables["convergence_summary"]
    gates = tables["convergence_gates"]
    robustness = tables["robustness_summary"]
    robust_decision = tables["symmetric_robustness_decision"].iloc[0]
    diagnostics = tables["robustness_diagnostic_summary"]

    gaussian_rows = convergence.loc[
        convergence["model"].isin(["no_intercept_csad", "scsad"])
    ]
    control_rows = convergence.loc[
        convergence["model"].isin(["standard_csad", "intercept_restored"])
    ]
    model_fpr = (
        robustness.groupby("model", as_index=False)["raw_false_positive_rate"]
        .agg(["min", "median", "max"])
        .reset_index()
    )
    equation = tables["equation_verification"]

    lines = [
        "# CSAD 기계적 음의 계수 연구 보고서",
        "",
        "## 1. 한 문장 결론",
        "",
        (
            "No-intercept CSAD의 음의 제곱항과 SCSAD의 음의 세제곱항은 독립 Gaussian 자산에 "
            "투자자 모방이 전혀 없어도 회귀식의 투영 구조만으로 발생합니다."
            if bool(decision["mechanical_null_confirmed"])
            else "사전 고정한 모든 조건을 통과하지 못해 기계적 null을 확정하지 못했습니다."
        ),
        "",
        "이 결과는 기존 empirical 계수가 틀렸다는 뜻이 아니라, 그 계수만으로 행동적 herding을 식별할 수 없다는 뜻입니다.",
        "",
        "## 2. 왜 음수가 되는가",
        "",
        "독립 Gaussian 수익률에서는 시장평균 `M`과 각 코인의 평균 이탈분이 독립입니다. 따라서 시장이 어느 방향으로 얼마나 움직였는지와 무관하게 평균 CSAD는 양의 상수입니다.",
        "",
        "No-intercept 식은 이 양의 상수를 표현할 절편이 없습니다. 그래서 `|M|` 항이 원점에서 빠르게 올라가고, `M^2` 항이 바깥쪽을 다시 아래로 굽혀 상수에 가깝게 만듭니다. 이때 제곱항은 수학적으로 반드시 음수가 됩니다.",
        "",
        "SCSAD는 양의 시장수익률에서 `+CSAD`, 음의 시장수익률에서 `-CSAD`이므로 원점에서 부호가 바뀌는 계단 모양입니다. 이를 선형항과 세제곱항으로 근사하면 선형항이 계단을 따라 올라가고 세제곱항이 양 끝의 과도한 증가를 누르므로 세제곱항이 음수가 됩니다.",
        "",
        "폐형식은 다음과 같습니다. `s=sigma/sqrt(N)`, `c=E[CSAD]`, `u=sqrt(2/pi)`입니다.",
        "",
        "```text",
        "No-intercept delta2* = (c/s^2) * (1 - 4/pi) / (3 - 8/pi) < 0",
        "SCSAD gamma3*        = -c*u / (6*s^3) < 0",
        "Standard beta2*      = 0",
        "Intercept-restored   = 0",
        "```",
        "",
        "## 3. 수학식 독립 검증",
        "",
        f"자산 수 14·50·62의 세 셀에서 폐형식과 Gaussian 절대모멘트 정규방정식의 최대 차이는 `{equation['maximum_absolute_difference'].max():.3e}`였습니다.",
        f"사전 허용오차 이내 통과는 `{int(equation['equation_gate_pass'].sum())}/{len(equation)}`입니다.",
        "",
        "## 4. 대규모 Monte Carlo 수렴",
        "",
        "각 자산 수에서 12,000개 시점과 200회 반복을 사용했습니다. 아래 값은 Monte Carlo 평균과 이론값의 차이입니다.",
        "",
        "| 모형 | N | 이론 계수 | MC 평균 | 상대오차 | 음수 비율 | raw FPR | BH3 FPR | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    gate_lookup = gates.set_index(["assets", "model"])
    for row in gaussian_rows.sort_values(["model", "assets"]).itertuples(index=False):
        gate = gate_lookup.loc[(row.assets, row.model), "gate_pass"]
        lines.append(
            f"| {_model_name(row.model)} | {row.assets} | {row.theoretical_target_coefficient:.6g} | "
            f"{row.mean_target_coefficient:.6g} | {100 * row.absolute_relative_error:.2f}% | "
            f"{100 * row.negative_coefficient_rate:.1f}% | {100 * row.raw_false_positive_rate:.1f}% | "
            f"{100 * row.bh3_false_positive_rate:.1f}% | {'통과' if gate else '실패'} |"
        )
    lines.extend(
        [
            "",
            "절편이 있는 control의 raw 음의 거짓양성률은 다음과 같습니다.",
            "",
            "| 모형 | raw FPR 최소 | 중앙 | 최대 | 7.5% gate 통과 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model, group in control_rows.groupby("model"):
        model_gates = gates.loc[gates["model"].eq(model)]
        lines.append(
            f"| {_model_name(model)} | {100 * group['raw_false_positive_rate'].min():.1f}% | "
            f"{100 * group['raw_false_positive_rate'].median():.1f}% | "
            f"{100 * group['raw_false_positive_rate'].max():.1f}% | "
            f"{int(model_gates['gate_pass'].sum())}/{len(model_gates)} |"
        )

    lines.extend(
        [
            "",
            "## 5. 더 현실적인 비허딩 과정",
            "",
            "정규성만의 우연인지 확인하기 위해 공통요인, 확률변동성, Student-t fat-tail, 양의 왜도, jump, 시변 상관, 음의 충격 비대칭을 포함했습니다. 어느 과정에도 다른 자산의 행동을 보고 따라가는 규칙은 없습니다.",
            "",
            "| 모형 | raw FPR 최소 | 중앙 | 최대 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in model_fpr.itertuples(index=False):
        lines.append(
            f"| {_model_name(row.model)} | {100 * row.min:.1f}% | {100 * row.median:.1f}% | {100 * row.max:.1f}% |"
        )
    lines.extend(
        [
            "",
            f"사전 정의한 대칭 DGP에서 두 기계적 모형의 음수 비율이 모두 80% 이상인 셀은 `{int(robust_decision['passing_cells'])}/{int(robust_decision['evaluated_cells'])}`였습니다. "
            f"통과 비중은 `{100 * robust_decision['passing_cell_share']:.1f}%`이고 75% gate는 `{'통과' if robust_decision['distributional_robustness_pass'] else '실패'}`했습니다.",
            "",
            "Standard CSAD도 이분산성·fat-tail·비대칭처럼 `E[CSAD|M]`가 상수가 아닌 과정에서는 명목 오탐률을 벗어날 수 있습니다. 따라서 절편을 넣는 것만으로 행동적 herding이 자동 식별되는 것은 아닙니다.",
            "",
            "## 6. 네 가지 최종 판정",
            "",
            "### 1. 기존 음의 CSAD 계수가 실제 herding 증거인가?",
            "",
            "아닙니다. 음의 계수는 관찰된 수치 관계이지만 intentional imitation의 충분조건이 아닙니다. 특히 no-intercept와 SCSAD target은 비허딩 Gaussian null에서도 population 값 자체가 음수입니다.",
            "",
            "### 2. 모형 구조가 만든 통계적 착시인가?",
            "",
            f"Gaussian null에 대해서는 `{'예' if decision['mechanical_null_confirmed'] else '확정 불가'}`입니다. 최종 사전등록 판정은 `{decision['classification']}`입니다. 절편 없는 양의 수준 근사와 sign-step의 cubic 근사가 음의 곡률을 강제합니다.",
            "",
            "### 3. 어떤 CSAD 사양까지 연구에 사용할 수 있는가?",
            "",
            "Standard CSAD는 절편을 포함하고 해당 universe·가중·분포에 맞춘 simulation 또는 bootstrap null로 크기를 교정할 때 기술적 조건부 수렴 검정에 사용할 수 있습니다. No-intercept와 현재 SCSAD의 일반 HAC p-value는 herding 식별 검정으로 사용하지 않습니다. 두 모형은 선행논문 수치 재현과 방법론 진단용으로만 보존합니다.",
            "",
            "### 4. 후속 실증연구의 식별 기준은 무엇인가?",
            "",
            "1. 절편을 임의로 제거하지 않는다.",
            "2. 선택한 universe, 가중법, 표본 길이와 동일한 비허딩 null에서 검정 크기를 먼저 교정한다.",
            "3. 공통요인·시변변동성·fat-tail·jump를 포함한 반사실적보다 empirical 계수가 더 극단적인지 확인한다.",
            "4. provider, 기간, point-in-time universe가 다른 외부 표본에서 방향과 효과크기를 재현한다.",
            "5. 음의 계수를 투자자 의도나 alpha로 바로 번역하지 않는다.",
            "",
            "## 7. 가정과 한계",
            "",
            "- 정확한 폐형식은 독립·동분산 Gaussian 자산과 동일가중 시장에 한정됩니다.",
            "- 강건성 simulation은 사전 고정한 여덟 DGP를 다루지만 가능한 모든 시장구조의 증명은 아닙니다.",
            "- HAC 유의성과 BH-FDR은 모형이 잘못 지정됐을 때 경제적 null을 복구하지 못합니다.",
            "- 이 연구는 뉴스, sentiment, 주문흐름, 미래수익률 또는 거래비용을 검정하지 않았습니다.",
            "",
            "## 8. 재현성",
            "",
            f"- Protocol SHA-256: `{protocol_hash}`",
            f"- Config SHA-256: `{config_hash}`",
            f"- 최종 판정: `{decision['classification']}`",
            f"- Gaussian equation gate: `{int(decision['equation_cells_passed'])}/{int(decision['equation_cells_total'])}`",
            f"- Mechanical convergence gate: `{int(decision['mechanical_convergence_cells_passed'])}/{int(decision['mechanical_convergence_cells_total'])}`",
            f"- Nominal control gate: `{int(decision['nominal_control_cells_passed'])}/{int(decision['nominal_control_cells_total'])}`",
            "",
            "실행:",
            "",
            "```bash",
            "PYTHONPATH=src .venv/bin/python scripts/run_csad_mechanical_derivation.py",
            "PYTHONPATH=src .venv/bin/python scripts/verify_csad_mechanical_derivation.py",
            "```",
            "",
            "## 9. 그림",
            "",
        ]
    )
    lines.extend(f"- `{plot_path}`" for plot_path in plot_paths)
    lines.append("")
    diagnostics_note = diagnostics.loc[
        diagnostics["dgp"].eq("independent_gaussian")
        & diagnostics["weighting"].eq("equal")
    ]
    if not diagnostics_note.empty:
        lines.extend(
            [
                "## 10. 구현 sanity check",
                "",
                f"독립 Gaussian 동일가중 셀의 평균 `corr(CSAD, M)` 절대값 중앙은 `{diagnostics_note['corr_csad_market'].abs().median():.4f}`, `corr(CSAD, |M|)` 절대값 중앙은 `{diagnostics_note['corr_csad_abs_market'].abs().median():.4f}`였습니다.",
                "",
            ]
        )
    return "\n".join(lines)


def build_master_report_update(
    source_text: str,
    decision: pd.DataFrame,
    convergence_summary: pd.DataFrame,
    robustness_decision: pd.DataFrame,
) -> str:
    result = decision.iloc[0]
    robust = robustness_decision.iloc[0]
    mechanical = convergence_summary.loc[
        convergence_summary["model"].isin(["no_intercept_csad", "scsad"])
    ]
    section = "\n".join(
        [
            "## 7-1. 음의 계수가 생기는 수학적 원인",
            "",
            "2026-07-21에 별도 사전등록 연구로 기존 null 감사의 97~100% 오탐 원인을 폐형식과 Monte Carlo로 검증했습니다.",
            "",
            "- 독립 Gaussian 동일가중에서 `M`과 CSAD는 독립이며 `E[CSAD|M]`는 양의 상수입니다.",
            "- No-intercept는 이 상수를 `|M|`와 `M^2`로 원점부터 근사하므로 population 제곱항이 반드시 음수입니다.",
            "- SCSAD는 `sign(M)` 계단을 선형·세제곱 다항식으로 근사하므로 population 세제곱항이 반드시 음수입니다.",
            f"- 12,000시점 × 200회 × N=14·50·62 수렴 검증에서 기계적 모형 gate는 `{int(result['mechanical_convergence_cells_passed'])}/{int(result['mechanical_convergence_cells_total'])}`, 절편 control은 `{int(result['nominal_control_cells_passed'])}/{int(result['nominal_control_cells_total'])}` 통과했습니다.",
            f"- 기계적 두 모형의 Gaussian raw FPR 범위는 `{100 * mechanical['raw_false_positive_rate'].min():.1f}%~{100 * mechanical['raw_false_positive_rate'].max():.1f}%`였습니다.",
            f"- 대칭 비허딩 DGP 강건성 셀은 `{int(robust['passing_cells'])}/{int(robust['evaluated_cells'])}` 통과했습니다.",
            f"- 최종 판정은 `{result['classification']}`입니다.",
            "",
            "따라서 no-intercept와 현재 SCSAD는 선행논문 수치 재현·모형 진단용으로만 보존하며 intentional herding 검정으로 사용하지 않습니다. Standard CSAD도 공통요인과 이분산성이 있는 현실적 null로 검정 크기를 먼저 교정해야 합니다.",
            "",
            "상세 보고서: `outputs/v2/csad_mechanical_derivation_v1/csad_mechanical_derivation_report.md`",
            "",
        ]
    )
    updated = source_text.replace("최종 갱신: 2026-07-20", "최종 갱신: 2026-07-21", 1)
    marker = "## 8. Tick 연구와 미래수익률"
    if marker not in updated:
        raise ValueError(f"Master report insertion marker not found: {marker}")
    return updated.replace(marker, section + marker, 1)


def _save_figure(figure: plt.Figure, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _pretty_dgp(value: str) -> str:
    return value.replace("_", " ").title()


def _model_name(value: str) -> str:
    return {
        "standard_csad": "Standard",
        "intercept_restored": "Intercept-restored",
        "no_intercept_csad": "No-intercept",
        "scsad": "SCSAD",
    }.get(value, value)

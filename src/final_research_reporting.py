from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


OUTCOME_LABELS = {
    "log_amihud_illiquidity": "Amihud형 비유동성",
    "zero_tick_share": "무가격변화 tick 비중",
    "log_mean_intertrade_ms": "평균 거래 간격",
    "log_quote_volume": "USDT 거래대금",
    "abs_aggressor_imbalance": "절대 aggressor 불균형",
}
OUTCOME_PLOT_LABELS = {
    "log_amihud_illiquidity": "Amihud illiquidity",
    "zero_tick_share": "Zero-tick share",
    "log_mean_intertrade_ms": "Mean trade interval",
    "log_quote_volume": "Quote volume",
    "abs_aggressor_imbalance": "Absolute aggressor imbalance",
}
ALLOWED_HYPOTHESIS_STATUSES = {
    "supported",
    "partially supported",
    "descriptive only",
    "falsified",
    "methodologically invalid",
    "requires new external data",
}


def plot_null_fpr(group_summary: pd.DataFrame, path: str | Path) -> None:
    labels = [f"{row.dimension}:{row.group}" for row in group_summary.itertuples()]
    x = np.arange(len(group_summary))
    figure, axis = plt.subplots(figsize=(14, 6))
    axis.errorbar(
        x,
        group_summary["null_fpr_mean"],
        yerr=np.vstack(
            [
                group_summary["null_fpr_mean"] - group_summary["null_fpr_q025"],
                group_summary["null_fpr_q975"] - group_summary["null_fpr_mean"],
            ]
        ),
        fmt="o",
        color="#176b87",
        ecolor="#9fbfc9",
        capsize=2,
    )
    axis.axhspan(0.015, 0.035, color="#dbe8d1", alpha=0.8, label="Pooled calibration band")
    axis.axhline(0.05, color="#b33f40", linestyle="--", label="Group maximum")
    axis.set_xticks(x, labels, rotation=75, ha="right", fontsize=8)
    axis.set_ylabel("Conditional-null false-positive rate")
    axis.set_title("Zero-run cutoff calibration across 23 frozen groups")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_empirical_vs_null(group_summary: pd.DataFrame, path: str | Path) -> None:
    selected = group_summary.loc[group_summary["dimension"].isin(["pooled", "asset"])].copy()
    labels = selected["group"].tolist()
    x = np.arange(len(selected))
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    specifications = [
        (
            "empirical_mean_intensity",
            "null_mean_intensity",
            "null_intensity_q025",
            "null_intensity_q975",
            "Mean zero-run intensity",
        ),
        (
            "empirical_clustering_share",
            "null_fpr_mean",
            "null_fpr_q025",
            "null_fpr_q975",
            "Clustering share at z <= -1.96",
        ),
    ]
    for axis, (actual, null, lower, upper, title) in zip(axes, specifications, strict=True):
        axis.errorbar(
            x,
            selected[null],
            yerr=np.vstack([selected[null] - selected[lower], selected[upper] - selected[null]]),
            fmt="o",
            color="#176b87",
            capsize=3,
            label="Conditional null",
        )
        axis.scatter(x, selected[actual], marker="D", color="#c55332", label="Observed")
        axis.set_xticks(x, labels, rotation=45, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_mechanism_null(mechanism_summary: pd.DataFrame, path: str | Path) -> None:
    ordered = mechanism_summary.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(ordered))
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.errorbar(
        ordered["null_mean_coefficient"],
        y,
        xerr=np.vstack(
            [
                ordered["null_mean_coefficient"] - ordered["null_q025"],
                ordered["null_q975"] - ordered["null_mean_coefficient"],
            ]
        ),
        fmt="o",
        color="#176b87",
        capsize=4,
        label="Count-conditioned null 95% interval",
    )
    axis.scatter(
        ordered["audit_sample_coefficient"],
        y,
        marker="D",
        color="#c55332",
        label="Observed audit coefficient",
    )
    axis.axvline(0.0, color="#444444", linewidth=0.8)
    axis.set_yticks(y, [OUTCOME_PLOT_LABELS[value] for value in ordered["outcome"]])
    axis.set_xlabel("Weighted standardized coefficient")
    axis.set_title("Observed microstructure associations versus exact conditional null")
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
    )
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_mechanical_null_report(
    group_summary: pd.DataFrame,
    mechanism_summary: pd.DataFrame,
    decisions: pd.DataFrame,
    integrity: pd.DataFrame,
    strata_summary: pd.DataFrame,
    diagnostics: Mapping,
    config: Mapping,
    plot_paths: Sequence[str],
) -> str:
    final = decisions.loc[decisions["decision"].eq("final_mechanical_null_audit")].iloc[0]
    calibration = decisions.loc[decisions["decision"].eq("conditional_cutoff_calibration")].iloc[0]
    excess = decisions.loc[decisions["decision"].eq("empirical_excess_clustering")].iloc[0]
    mechanism = decisions.loc[decisions["decision"].eq("mechanism_beyond_conditional_null")].iloc[0]
    pooled = group_summary.loc[group_summary["dimension"].eq("pooled")].iloc[0]
    assets = group_summary.loc[group_summary["dimension"].eq("asset")]
    supporting_assets = int((assets["supports_excess_intensity"] & assets["supports_excess_tail"]).sum())
    passed_mechanisms = mechanism_summary.loc[mechanism_summary["exceeds_count_conditioned_null"], "outcome"].tolist()
    lines = [
        "# Zero-Run 조건부 배열 Null 기계성 감사",
        "",
        "## 한눈에 보는 결론",
        "",
        f"최종 사전등록 판정은 **{final['classification']}**이며 전체 감사 기준은 **{'통과' if bool(final['passed']) else '미통과'}**입니다.",
        "이 판정은 거래 수와 zero-tick 수를 그대로 보존했을 때도 관찰된 run 배열과 동시점 미시구조 관계가 남는지를 묻습니다. 미래 가격 방향, 수익률 alpha, 의도적 모방 또는 인과관계를 검증하지 않습니다.",
        "",
        f"- 조건부 cutoff calibration: **{'통과' if bool(calibration['passed']) else '미통과'}**",
        f"- 실제 excess clustering: **{'통과' if bool(excess['passed']) else '미통과'}** ({supporting_assets}/7 자산)",
        f"- count-conditioned null을 넘는 mechanism family: **{'통과' if bool(mechanism['passed']) else '미통과'}** ({len(passed_mechanisms)}/5)",
        "- 기존 미래 절대수익률·실현변동성 family 0/2 판정은 변경하지 않습니다.",
        "",
        "## 사전등록과 표본",
        "",
        "결과를 보기 전에 protocol, config, seed, 999회 반복, 23개 그룹, 5개 outcome과 모든 판정 기준을 SHA-256으로 봉인했습니다.",
        "",
        f"- OOS 적격 모집단: {int(integrity.iloc[0]['oos_eligible_rows']):,}개 15분 bucket",
        f"- 층화 감사 표본: {int(strata_summary['sample_rows'].sum()):,}개, {len(strata_summary):,}개 자산×거래수×zero-share 층",
        f"- 조건부 null: 각 row의 `n=transaction_count`, `k=zero_ticks` 고정, 정확한 조합분포에서 {int(diagnostics['repetitions']):,}회 추출",
        f"- 원본 run-z 재구성 최대 오차: {float(integrity.iloc[0]['maximum_recomputed_run_z_difference']):.3g}",
        f"- PMF 합 최대 오차: {float(diagnostics['maximum_pmf_sum_error']):.3g}",
        "",
        "## 1. Cutoff calibration과 실제 배열",
        "",
        f"Pooled null FPR은 {pooled['null_fpr_mean']:.3%}이며 사전 허용범위 1.5%~3.5%와 비교했습니다. 실제 clustering share는 {pooled['empirical_clustering_share']:.3%}입니다.",
        "",
        "| 그룹 | 실제 평균 intensity | null 평균 | intensity BH q | 실제 tail | null FPR | tail BH q | 두 지표 지지 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in group_summary.loc[group_summary["dimension"].isin(["pooled", "asset"])].itertuples(index=False):
        both = bool(row.supports_excess_intensity and row.supports_excess_tail)
        lines.append(
            f"| {row.group} | {row.empirical_mean_intensity:.4f} | {row.null_mean_intensity:.4f} | "
            f"{row.intensity_q_value_bh_fdr_23:.4g} | {row.empirical_clustering_share:.3%} | "
            f"{row.null_fpr_mean:.3%} | {row.tail_q_value_bh_fdr_23:.4g} | {'예' if both else '아니오'} |"
        )
    lines.extend(
        [
            "",
            "## 2. 동시점 미시구조 관계",
            "",
            "실제 감사 표본 계수와 같은 row의 count-conditioned null 계수 999개를 비교했습니다. 계수는 기존 개발구간 scaler를 재사용한 가중 표준화 계수입니다.",
            "",
            "| Outcome | Full OOS | 감사 표본 | null 95% | BH q | 크기비율 | 최종 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in mechanism_summary.itertuples(index=False):
        lines.append(
            f"| {OUTCOME_LABELS[row.outcome]} | {row.full_oos_coefficient:.4f} | "
            f"{row.audit_sample_coefficient:.4f} | [{row.null_q025:.4f}, {row.null_q975:.4f}] | "
            f"{row.q_value_bh_fdr_5:.4g} | {row.audit_to_full_absolute_magnitude_ratio:.2f} | "
            f"{'통과' if row.exceeds_count_conditioned_null else '미통과'} |"
        )
    lines.extend(
        [
            "",
            "## 3. 해석 경계",
            "",
            "- 이 감사가 통과하면 zero/non-zero 배열 순서와 동시점 거래구조의 관계가 단순한 거래 수와 zero-tick 수 이상의 정보를 가진다는 뜻입니다.",
            "- 이 감사가 미통과하면 기존 5/5 동시점 관계의 일부 또는 전부가 run-z 산식과 구성변수의 기계적 연결로 설명될 수 있다는 뜻입니다.",
            "- 어느 경우에도 같은 15분 안의 동시성만으로 투자자의 의도적 따라하기나 인과를 식별할 수 없습니다.",
            "- AggTrades에는 호가 spread와 depth가 없으므로 quote/order-book 자료가 있어야 다음 메커니즘을 직접 검증할 수 있습니다.",
            "- 기존 OOS 미래 family 0/2, 방향성 alpha 미검정, tracker 비활성 상태는 그대로 유지합니다.",
            "",
            "## 그림",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in plot_paths)
    lines.extend(
        [
            "",
            "## 재현",
            "",
            "```bash",
            "python scripts/run_zero_run_mechanical_null_audit.py",
            "python scripts/verify_final_research_completion.py",
            "```",
            "",
            "프로토콜과 설정은 null 결과를 보기 전에 봉인됐으며, 산출물의 해시와 판정식은 읽기 전용 verifier에서 다시 계산합니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_hypothesis_closure_status() -> pd.DataFrame:
    rows = [
        ("H01", "CSAD", "Binance 14종목 2년 standard CSAD의 beta2는 음수다.", "falsified", "beta2=+4.534759, HAC t=11.25", "outputs/baseline/regression_results.csv", True, "새 거래소·새 기간의 사전등록 표본"),
        ("H02", "CSAD", "Binance standard CSAD의 음의 곡률은 빈도와 대형자산 제외에 강건하다.", "falsified", "1분~1일, EW14/EW12 12개 셀 모두 미지지", "outputs/v2/frequency_sensitivity/frequency_sensitivity_summary.csv", True, "현재 표본에서는 종료"),
        ("H03", "Replication", "논문 fixed-62의 corrected CSAD 수치는 공개 자료로 재현된다.", "supported", "daily SCSAD -2.902 (t=-4.621), 논문 -2.924 (t=-4.471)", "outputs/v2/cmc_fixed_62/replication_v1/cmc_fixed_62_replication_report.md", True, "weekly 307주 생성 규칙 공개 시 정밀 보완"),
        ("H04", "Replication", "Fixed-62 corrected 수치는 논문 이후 시간 holdout에서도 유지된다.", "supported", "primary corrected 4/4, lagged sensitivity 4/4", "outputs/v2/cmc_fixed_62/temporal_validation_v1/persistence_decision_summary.csv", True, "독립 provider와 point-in-time universe"),
        ("H05", "Identification", "No-intercept와 SCSAD의 음의 계수는 behavioral herding을 식별한다.", "methodologically invalid", "비허딩 null FPR 97~100%; empirical-vs-null 0/240", "outputs/v2/csad_specification_audit_v1/csad_specification_audit_report_v1_1.md", True, "식별력이 검증된 새 통계량과 외부표본"),
        ("H06", "External validity", "Corrected CSAD 관계는 provider, universe, weighting에 보편적이다.", "falsified", "Binance 3/4·3/4, OKX 3/4·1/4, PIT Top-50 2/4·0/4", "outputs/v2/csad_specification_audit_v1/structural_robustness_decisions_v1_1.csv", True, "다른 point-in-time 시장의 사전등록 검증"),
        ("H07", "Dynamics", "Corrected SCSAD 계수는 시간에 따라 안정적이다.", "falsified", "4개 break, Hansen·CUSUM 안정성 기각", "outputs/v2/cmc_dynamic_universe/structural_break_v1/cmc_scsad_structural_break_report.md", True, "새 표본에서 고정 break의 외부검증"),
        ("H08", "Convergence", "시장요인 조정 뒤에도 극단시장 초과 수렴이 남는다.", "partially supported", "365일 delta2=6.772, q=2.91e-05; 730일 전체 미통과·regime 4 반대", "outputs/v2/cmc_dynamic_universe/factor_adjusted_convergence_v1/cmc_factor_adjusted_convergence_report.md", True, "완전한 가격결정모형과 외부표본"),
        ("H09", "Convergence", "SIZE·LIQ·MOM 및 empirical residual 뒤에도 초과 수렴이 남는다.", "partially supported", "normal 6.600, empirical 6.885; regime 4는 모두 반대", "outputs/v2/cmc_dynamic_universe/multifactor_convergence_v1/cmc_multifactor_convergence_report.md", True, "독립 요인자료와 사전등록 holdout"),
        ("H10", "Tick semantics", "Run-clustering winner side는 가격 방향의 proxy다.", "falsified", "방향 일치율 49.23%", "outputs/v2/tick/semantic_validation/confirmatory_2y/tick_raw_confirmatory_report.md", True, "현재 label 폐기"),
        ("H11", "Tick semantics", "Run-clustering winner side는 aggressor 방향의 proxy다.", "falsified", "aggressor 방향 일치율 48.13%", "outputs/v2/tick/semantic_validation/confirmatory_2y/tick_raw_confirmatory_report.md", True, "현재 label 폐기"),
        ("H12", "Prediction", "Winner event는 30분 시장중립 방향 수익률을 예측한다.", "falsified", "up/down/zero 모두 BH q=0.9336", "outputs/v2/tick/semantic_validation/confirmatory_2y/market_neutral_predictive_coefficients.csv", True, "동일 표본 재최적화 금지"),
        ("H13", "Prediction", "연속형 up/down/zero run-z는 사전 경제성 기준을 충족한다.", "falsified", "OOS 사전 통계·효과크기 기준 0/9", "outputs/v2/tick/continuous_run_z_oos/continuous_run_z_oos_report.md", True, "새 외부 tick 표본만 허용"),
        ("H14", "Microstructure", "Zero-run 배열은 n과 zero-tick 수를 고정한 무작위 배열보다 더 clustered다.", "supported", "실제 tail 87.347%, null FPR 2.537%, 7/7 자산", "outputs/v2/final_research_completion_v1/empirical_null_group_comparison.csv", True, "다른 거래소·기간의 raw tick 외부검증"),
        ("H15", "Microstructure", "Zero-run과 5개 동시점 시장구조 관계는 count-conditioned null을 넘는다.", "descriptive only", "원표본 5/5였으나 exact-null 감사 0/5", "outputs/v2/final_research_completion_v1/mechanism_null_comparison.csv", True, "quote/order-book가 있는 새 외부자료"),
        ("H16", "Prediction", "Zero-run은 5·15·30분 미래 절대수익률 또는 실현변동성을 개선한다.", "falsified", "미래 family 0/2, RMSE 개선 -0.0073%~+0.0004%", "outputs/v2/zero_run_microstructure_v1/family_decisions.csv", True, "동일 표본 horizon 최적화 금지"),
        ("H17", "Behavior", "관찰된 횡단면 수렴과 run clustering은 투자자의 의도적 모방이다.", "requires new external data", "현재 자료는 행동 의도·정보흐름을 직접 관찰하지 않음", "outputs/v2/final_research_completion_v1/mechanical_null_audit_report.md", False, "계정·주문·정보노출을 연결한 식별 설계"),
        ("H18", "Mechanism", "Zero-run은 spread, depth, price recovery로 설명된다.", "requires new external data", "aggTrades에 bid·ask quote와 depth가 없음", "outputs/v2/final_research_completion_v1/mechanical_null_audit_report.md", False, "L2 order-book 또는 quote 자료"),
        ("H19", "Sentiment", "뉴스·Reddit sentiment가 검증된 herding event의 품질을 개선한다.", "requires new external data", "point-in-time first_seen archive와 검증된 base event가 부족", "outputs/research_master_report_2026-07-20.md", False, "누수 없는 first-seen sentiment archive와 새 event 표본"),
    ]
    columns = [
        "hypothesis_id",
        "domain",
        "hypothesis",
        "status",
        "key_evidence",
        "evidence_path",
        "closed_with_current_data",
        "reopening_requirement",
    ]
    result = pd.DataFrame(rows, columns=columns)
    if not set(result["status"]).issubset(ALLOWED_HYPOTHESIS_STATUSES):
        raise ValueError("Hypothesis table contains an unsupported closure status")
    return result


def plot_hypothesis_status(status: pd.DataFrame, path: str | Path) -> None:
    order = [
        "supported",
        "partially supported",
        "descriptive only",
        "falsified",
        "methodologically invalid",
        "requires new external data",
    ]
    counts = status["status"].value_counts().reindex(order, fill_value=0)
    colors = ["#287271", "#5b8e7d", "#d4a373", "#c55332", "#8d3b72", "#64748b"]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.barh(np.arange(len(order)), counts.to_numpy(), color=colors)
    axis.set_yticks(np.arange(len(order)), order)
    axis.invert_yaxis()
    axis.set_xlabel("Number of preregistered or audited hypotheses")
    axis.set_title("Final hypothesis closure map")
    axis.bar_label(bars, padding=4)
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_final_manuscript(status: pd.DataFrame) -> str:
    status_counts = status["status"].value_counts()
    lines = [
        "# From Replication to Falsification: Specification Mechanics, Universe Dependence, and Tick-Level Clustering in Cryptocurrency Herding Research",
        "",
        "## Abstract",
        "",
        "This project evaluates cryptocurrency herding across classical cross-sectional absolute deviation (CSAD) regressions, paper-aligned CoinMarketCap replications, exchange and universe external validations, factor-adjusted convergence designs, and two years of raw Binance aggregate trades. The published fixed-62 CoinMarketCap corrected-CSAD coefficients were closely reproduced and persisted in a non-overlapping temporal holdout. However, corrected relations weakened across Binance, OKX, and point-in-time universes. A preregistered synthetic-null audit showed that no-intercept CSAD and SCSAD generated negative significance in 97-100% of non-herding simulations, while no empirical coefficient was more negative than its matched null after false-discovery control. Tick-level winner labels failed semantic validation and had no economically meaningful out-of-sample directional response. Zero-tick runs were far more clustered than an exact count-conditioned random arrangement, but their five contemporaneous microstructure associations did not survive the corresponding mechanical-null family test, and they did not improve 5-30 minute absolute-return or realized-volatility forecasts. The evidence therefore supports numerical replication and descriptive market-state clustering, not intentional imitation, universal herding, or short-horizon return predictability.",
        "",
        "## 한국어 제목",
        "",
        "# 복제에서 반증까지: 암호화폐 herding 통계의 사양 기계성, universe 의존성, tick 배열 clustering",
        "",
        "## 초록",
        "",
        "본 연구는 암호화폐 herding을 한 개 회귀식의 유의성으로 판단하지 않고, 수치 복제·외부타당성·비허딩 귀무모형·변수 의미·시간 외 표본·미시구조 기계성을 순차적으로 감사했다. 선행논문의 CoinMarketCap fixed-62 corrected CSAD 수치는 거의 정확히 복제됐고 같은 공급자의 비중첩 시간 holdout에서도 유지됐다. 그러나 Binance, OKX, 상장폐지를 포함한 point-in-time universe로 이동하면 엄격한 재현 기준을 통과하지 못했다. 더 중요하게, no-intercept와 SCSAD는 herding 규칙이 없는 synthetic data에서도 97~100%의 거짓양성을 만들었고 empirical 계수는 대응 null보다 더 극단적이지 않았다. Raw tick 연구에서는 방향 label을 폐기했고, zero-run 배열 자체는 거래 수와 zero-tick 수를 보존한 exact null보다 강하게 clustered됐지만 기존 동시점 미시구조 5개 관계는 null 대비 0/5였다. 미래 절대수익률과 실현변동성 예측 family도 0/2였다. 따라서 이 프로젝트는 논문 수치의 복제와 시장상태의 기술적 clustering은 지지하지만, 의도적 모방·보편적 herding·단기 alpha는 지지하지 않는다.",
        "",
        "## 1. 연구 질문과 판정 원칙",
        "",
        "연구 질문은 세 층으로 분리했다. 첫째, 기존 논문의 숫자를 재현할 수 있는가. 둘째, 그 통계가 herding이 없는 반사실적과 구분되는가. 셋째, 그 변수가 미래 움직임에 증분 정보를 주는가. 첫째가 성공해도 둘째와 셋째를 자동으로 의미하지 않는다.",
        "",
        "모든 핵심 연구는 가능한 경우 결과 확인 전 protocol, universe, 기간, family, 다중검정, 효과크기를 고정했다. 결과가 실패하면 같은 표본에서 threshold·자산·기간을 다시 골라 살리지 않았다. 현재 자동매매, tracker, 방향성 alpha는 연구 범위에서 제외한다.",
        "",
        "## 2. 데이터와 연구 축",
        "",
        "| 축 | 자료 | 기간·크기 | 핵심 역할 |",
        "|---|---|---|---|",
        "| Binance baseline | 14개 USDT 1분봉 | 2024-04-08~2026-04-08, 1,051,199분 | Classical CSAD |",
        "| CMC dynamic | 월별 point-in-time Top-200 | 2018-01-01~2024-04-09 | 논문형 확장·구조변화 |",
        "| CMC fixed-62 | 논문 Table 1 legacy ID | historical 2,291일 + holdout 830일 | 직접 복제·시간 외 검증 |",
        "| Binance·OKX | 14종목 5년 | daily 1,826일, weekly 261주 | 공급자 외부검증 |",
        "| Binance archive | PIT Top-50 | 636,506 asset-day, 584 listing episode | survivor bias 완화 |",
        "| Raw aggTrades | 7개 자산 | 490,560개 15분 bucket | tick 의미·zero-run |",
        "| Exact mechanical null | OOS 층화표본 | 9,044개, 999회 | n·zero-count 조건부 배열 감사 |",
        "",
        "## 3. Classical CSAD: Binance에서는 음의 곡률이 없었다",
        "",
        "기본식은 `CSAD_t = alpha + beta1 |Rm,t| + beta2 Rm,t^2 + error_t`이다. 고정 14종목 2년 1분봉에서 `beta2=+4.534759`, HAC `t=11.25`였다. Classical herding의 필요조건으로 쓰이는 음의 곡률과 반대다. 1분·5분·15분·1시간·4시간·1일과 EW14·BTC/ETH 제외 EW12를 결합한 12개 셀도 모두 음의 herding 판정을 지지하지 않았다.",
        "",
        "이 결과는 암호화폐 전체에 herding이 없다는 뜻이 아니다. 특정 거래소, 고정 survivor universe, 특정 기간에서 standard CSAD 필요조건이 관찰되지 않았다는 뜻이다.",
        "",
        "## 4. 선행논문 복제: 숫자는 재현됐다",
        "",
        "CMC fixed-62 historical 표본에서 daily SCSAD는 `-2.902 (t=-4.621)`로 논문의 `-2.924 (t=-4.471)`와 근접했고, no-intercept·SCSAD daily·weekly 4개 셀이 모두 통과했다. Weekly 계수도 가까웠지만 공개 규칙으로는 논문 307주가 아니라 달력상 328주가 만들어져 21주 차이를 숨기지 않았다.",
        "",
        "논문 이후 2024-04-10~2026-07-18의 비중첩 holdout에서도 current와 lagged-size corrected가 각각 4/4였다. 따라서 '같은 CMC 공급자와 같은 fixed-62에서 corrected 수치가 시간적으로 반복된다'는 명제는 지지된다. 그러나 fixed survivor 목록과 동일 provider이므로 완전한 외부타당성은 아니다.",
        "",
        "## 5. 외부타당성: 현실적인 universe로 갈수록 약해졌다",
        "",
        "| 표본 | Equal/current | Lagged liquidity | 엄격 판정 |",
        "|---|---:|---:|---|",
        "| CMC fixed-62 historical | 4/4 | 4/4 | 수치 복제 |",
        "| CMC fixed-62 temporal holdout | 4/4 | 4/4 | 동일 provider 시간 재현 |",
        "| Binance fixed-14 | 3/4 | 3/4 | 미통과 |",
        "| OKX listing-aware 14 | 3/4 | 1/4 | 미통과 |",
        "| Binance archive PIT Top-50 | 2/4 | 0/4 | 미통과 |",
        "",
        "Binance archive는 21,479개 monthly ZIP과 636,506 asset-day를 열거해 종료 episode 156개를 포함했다. 동일가중도 weekly에서 약해졌고, 전기 quote-volume 가중은 0/4였다. 통합 이질성 `I²=69.3%~96.9%`는 pooled 음수보다 provider·universe·가중법 차이가 핵심임을 보여준다.",
        "",
        "## 6. Specification null audit: corrected 계수의 행동 해석은 무효다",
        "",
        "4개 비허딩 DGP, 16개 N·기간·가중 시나리오, 시나리오당 300회에서 standard CSAD의 BH 거짓양성 중앙값은 약 1%였지만 no-intercept는 99~100%, SCSAD는 97~100%였다. No-intercept 설명변수에 절편만 복원하자 기존 음의 지지 16/16이 사라지거나 표준화 절대크기가 절반 이상 감소했다.",
        "",
        "Empirical 10개 사양×2빈도×3모형×4DGP의 240개 직접 비교에서 대응 null보다 더 음수인 셀은 BH-FDR 후 `0/240`이었다. 구조적 강건성 기준도 `0/10`이었다. 따라서 논문 숫자의 복제 성공과 behavioral herding 식별 실패를 동시에 기록해야 한다.",
        "",
        "## 7. Factor-adjusted convergence: 조건부 기술 증거",
        "",
        "365일 leave-one-out 단일시장요인 반사실적에서 평균 초과 수렴은 28.22%, 극단시장 `delta2=6.772 (t=4.42, q=2.91e-05)`였다. MKT·SIZE·LIQ·MOM 다요인 normal은 `6.600`, empirical residual은 `6.885`로 전체표본에서 유지됐다.",
        "",
        "그러나 730일 창 전체는 미통과였고 regime 4는 단일·다요인·empirical 모두 반대였다. 이 결과는 알려진 공통요인 이상으로 수익률이 모이는 시점이 있다는 기술적 근거지만, 누락 요인과 의도적 모방을 분리하지 못한다.",
        "",
        "## 8. Tick 의미 감사: 방향 winner를 폐기했다",
        "",
        "구형 `run_clustering_side`는 조건부 run-z가 가장 낮은 category일 뿐 가격 방향이 아니었다. 2년 raw 표본에서 가격 방향 일치율은 49.23%, buyer-maker aggressor 방향 일치율은 48.13%였다. Winner event의 91.82%가 zero-run이었고, up/down/zero 30분 시장중립 계수는 모두 BH `q=0.9336`이었다.",
        "",
        "Winner를 없애고 up/down/zero intensity를 동시에 넣은 개발·OOS 회귀에서도 사전 기준 통과는 0/9였다. 미래 방향 계수의 신뢰구간은 ±5bp 실용 경계 안에 있었다. 따라서 과거 winner 기반 전략·tracker는 연구 증거에서 제외한다.",
        "",
        "## 9. Zero-run: 배열 clustering은 실재하지만 경제 해석은 제한된다",
        "",
        "초기 사전등록 OOS 분석에서 zero-run intensity는 Amihud 비유동성 `-0.207 SD`, 평균 거래간격 `-0.537 SD`, 거래대금 `+0.503 SD`, 절대 aggressor 불균형 `-0.120 SD`와 연결돼 5/5였다. 동시에 미래 절대수익률과 실현변동성 family는 0/2였고 개발→OOS RMSE 개선은 `-0.0073%~+0.0004%`였다.",
        "",
        "후속 기계성 감사는 결과 확인 전 n과 zero-tick 수를 고정한 exact binary-arrangement null을 봉인했다. 233,201개 OOS 모집단에서 141개 층, 9,044개 표본, 999회 조합분포를 사용했다. Null FPR은 2.537%로 보정됐지만 실제 `z<=-1.96` share는 87.347%였고 7/7 자산이 두 excess 기준을 통과했다. 즉 zero/non-zero 순서는 단순 무작위 배열이 아니다.",
        "",
        "반면 같은 null draw로 5개 동시점 계수 분포를 만들자 BH 기준 통과는 0/5였다. 최종 분류는 `clustering_beyond_counts_but_mechanism_not_distinct`다. 배열의 비무작위성은 남지만, aggTrades만으로 기존 유동성·균형 상태 해석을 독립적으로 확정할 수 없다.",
        "",
        "## 10. 통합 결론",
        "",
        "1. **수치 복제:** 성공했다. CMC fixed-62 corrected 계수와 시간 holdout은 재현된다.",
        "2. **행동적 herding 식별:** 실패했다. 핵심 corrected 통계가 비허딩 null을 구분하지 못한다.",
        "3. **외부 보편성:** 실패했다. 거래소·point-in-time universe·가중법에 따라 엄격 기준이 무너진다.",
        "4. **Tick 방향 의미:** 반증됐다. Winner side는 가격·aggressor 방향이 아니다.",
        "5. **배열 구조:** 지지된다. Zero-run 순서는 count-conditioned 무작위 배열보다 훨씬 clustered다.",
        "6. **동시점 경제 메커니즘:** 기술적 관계만 남는다. Exact-null family는 0/5다.",
        "7. **미래 예측:** 지지되지 않는다. 방향성, 절대수익률, 실현변동성 모두 사전 기준을 통과하지 못했다.",
        "",
        "따라서 이 프로젝트의 최종 기여는 alpha 발견이 아니라, 논문 수치를 재현하면서도 그 행동 해석을 반증하고, tick-level 비무작위 배열과 예측 불가능성을 분리한 재현 가능한 방법론 감사다.",
        "",
        "## 11. 가설 종료 현황",
        "",
        f"총 {len(status)}개 질문 중 supported {int(status_counts.get('supported', 0))}개, partially supported {int(status_counts.get('partially supported', 0))}개, descriptive only {int(status_counts.get('descriptive only', 0))}개, falsified {int(status_counts.get('falsified', 0))}개, methodologically invalid {int(status_counts.get('methodologically invalid', 0))}개, 새 외부자료 필요 {int(status_counts.get('requires new external data', 0))}개다.",
        "",
        "상세 근거·파일·재개 조건은 `hypothesis_closure_status.csv`에 있다. 현재 자료로 닫힌 가설은 같은 표본에서 재최적화하지 않는다.",
        "",
        "## 12. 한계와 다음 연구의 최소 조건",
        "",
        "- CSAD는 동시적 횡단면 분산이므로 관찰만으로 의도적 모방을 식별하지 못한다.",
        "- Fixed-62와 현재 상장목록은 survivor bias가 있으며, CMC·provider·universe 효과는 완전히 분리되지 않는다.",
        "- Break와 regime 일부는 같은 표본에서 선택된 기술적 결과다.",
        "- AggTrades는 taker order의 fill 집계이며 당시 bid·ask spread, depth, queue를 제공하지 않는다.",
        "- Zero-run exact null은 n과 zero count만 보존하고 duration, size, aggressor sequence는 보존하지 않는다.",
        "- News·Reddit은 `first_seen_at_utc`가 있는 충분한 과거 archive와 검증된 base event 없이는 confirmatory feature가 아니다.",
        "- 다음 연구는 완전히 새로운 거래소·기간의 quote/order-book 자료, 결과 전 protocol, 고정 family, 외부표본을 필요로 한다.",
        "",
        "## 13. 재현 자료",
        "",
        "최종 패키지의 `research_evidence_manifest.csv`는 본문 근거 파일의 크기와 SHA-256을 보존한다. `artifact_manifest.csv`는 이 폴더 전체 산출물을 봉인하고, `scripts/verify_final_research_completion.py`는 이를 읽기 전용으로 재검증한다. 실행 순서는 `REPRODUCIBILITY.md`에 정리했다.",
    ]
    return "\n".join(lines) + "\n"


def build_reproducibility_guide() -> str:
    return """# 최종 연구 재현 안내

## 빠른 무결성 검증

기존 결과를 변경하지 않고 해시, 행 수, FDR, 판정식, protocol seal, 보고서 필수 섹션을 확인합니다.

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_final_research_completion.py
```

## 기계성 감사 재실행

새 복제 작업공간에서 입력 산출물을 먼저 준비한 뒤 실행합니다. 관찰된 결과를 덮어쓰지 않도록 runner는 기존 판정 파일이 있으면 중단합니다.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_zero_run_microstructure.py
PYTHONPATH=src .venv/bin/python scripts/verify_zero_run_microstructure.py
PYTHONPATH=src .venv/bin/python scripts/run_zero_run_mechanical_null_audit.py
PYTHONPATH=src .venv/bin/python scripts/build_final_research_completion.py
PYTHONPATH=src .venv/bin/python scripts/verify_final_research_completion.py
```

## 전체 테스트

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

## 재현 보장 범위

- Protocol과 config의 사전 봉인 SHA-256을 검사합니다.
- 원시·분석 frame의 key, row count, run-z 재구성 오차를 검사합니다.
- Exact conditional run PMF는 작은 binary sequence 전수열거 테스트와 비교합니다.
- 23개 그룹의 두 BH family, 5개 mechanism BH family, 최종 판정식을 독립 재계산합니다.
- 최종 evidence와 artifact manifest의 파일 크기·SHA-256을 전수 검사합니다.

이 패키지는 방향성 alpha, tracker 또는 자동매매를 재현 대상으로 선언하지 않습니다.
"""

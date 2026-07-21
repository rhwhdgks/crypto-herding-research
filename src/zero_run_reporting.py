from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


FAMILY_LABELS = {
    "mechanism": "동시점 미시구조",
    "future_absolute_return": "미래 절대수익률",
    "future_realized_volatility": "미래 실현변동성",
}
OUTCOME_LABELS = {
    "log_amihud_illiquidity": "Amihud형 비유동성",
    "zero_tick_share": "무가격변화 tick 비중",
    "log_mean_intertrade_ms": "평균 aggregate-trade 간격",
    "log_quote_volume": "USDT 거래대금",
    "abs_aggressor_imbalance": "절대 aggressor 불균형",
}


def build_korean_report(
    tick_coverage: pd.DataFrame,
    ohlcv_quality: pd.DataFrame,
    sample_coverage: pd.DataFrame,
    coefficients: pd.DataFrame,
    mechanism_decisions: pd.DataFrame,
    future_decisions: pd.DataFrame,
    family_decisions: pd.DataFrame,
    prediction_metrics: pd.DataFrame,
    loao_summary: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    placebo: pd.DataFrame,
    diagnostics: pd.DataFrame,
    config: Mapping,
    plot_paths: Sequence[str],
) -> str:
    mechanism_success = bool(
        family_decisions.loc[
            family_decisions["family"].eq("mechanism"), "family_success"
        ].iloc[0]
    )
    absolute_success = bool(
        family_decisions.loc[
            family_decisions["family"].eq("future_absolute_return"),
            "family_success",
        ].iloc[0]
    )
    volatility_success = bool(
        family_decisions.loc[
            family_decisions["family"].eq("future_realized_volatility"),
            "family_success",
        ].iloc[0]
    )
    lines = [
        "# Zero-Run 시장 미시구조 연구 보고서",
        "",
        "## 한눈에 보는 결론",
        "",
        "이 연구는 가격이 오를지 내릴지를 예측하지 않습니다. 15분 동안 같은 가격에서 체결이 반복되는 정도가 시장의 거래 구조와 관련되는지, 그리고 그 뒤 가격이 얼마나 크게 움직이는지만 검증했습니다.",
        "",
        f"- 미시구조 연관성 종합 기준: **{'통과' if mechanism_success else '미통과'}**",
        f"- 미래 절대수익률 family: **{'통과' if absolute_success else '미통과'}**",
        f"- 미래 실현변동성 family: **{'통과' if volatility_success else '미통과'}**",
        "- 방향성 수익률, 거래비용 후 수익, 매수·매도 규칙은 분석하지 않았습니다.",
        "- 따라서 어떤 통과 결과도 자동매매 alpha로 해석할 수 없습니다.",
        "",
        "## 연구 질문",
        "",
        "`zero_run_intensity = -run_z_zero`가 클수록 15분 bucket 안에서 가격 변화가 없는 연속 체결이 조건부 기대보다 강하게 몰려 있습니다. 이 값이 비유동성, 가격 이산성, 거래 간격, 거래대금, order-flow 집중과 연결되는지 먼저 확인하고, 이후 5·15·30분 절대수익률과 실현변동성에 증분 정보를 주는지 검정했습니다.",
        "",
        "## 데이터와 시간 순서",
        "",
        f"- 자산: {', '.join(config['data']['symbols'])}",
        f"- 전체 기간: {config['data']['expected_start']} ~ {config['split']['oos_end_exclusive']} 미만",
        f"- 개발: {config['split']['development_start']} ~ {config['split']['oos_start']} 미만",
        f"- OOS: {config['split']['oos_start']} ~ {config['split']['oos_end_exclusive']} 미만",
        "- 신호는 15분 bucket이 완전히 끝난 뒤에만 사용할 수 있습니다.",
        "- 미래 outcome은 신호 이후 1분봉만 사용하고, split 경계에 닿는 outcome도 제외했습니다.",
        "- winsorization과 표준화 경계는 개발 구간에서만 적합해 OOS에 고정했습니다.",
        "",
        "### 표본 범위",
        "",
        "| 구간 | 전체 행 | 최소 거래수 충족 | UTC day | 5분 exact | 15분 exact | 30분 exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sample_coverage.itertuples(index=False):
        lines.append(
            f"| {row.split} | {row.rows:,} | {row.minimum_trade_count_rows:,} "
            f"({row.minimum_trade_count_share:.1%}) | {row.utc_days:,} | "
            f"{row.exact_5m_rows:,} | {row.exact_15m_rows:,} | {row.exact_30m_rows:,} |"
        )
    lines.extend(
        [
            "",
            f"- tick complete-grid 검증: {int(tick_coverage['complete_15m_grid'].sum())}/7 자산 통과",
            f"- 1분봉 exact-horizon 평균 가용률: {ohlcv_quality['all_horizons_exact_share'].mean():.4%}",
            f"- tick 마지막 가격과 직전 1분봉 종가의 최대 차이: {ohlcv_quality['maximum_tick_vs_1m_close_difference_bps'].max():.4f}bp",
            "",
            "## 1. 동시점 미시구조 연관성",
            "",
            "계수는 zero-run intensity가 1표준편차 커질 때 outcome이 몇 표준편차 변하는지를 뜻합니다. 이는 같은 15분 안의 동시점 연관성이므로 인과나 미래 예측이 아닙니다.",
            "",
            "| 미시구조 대리변수 | 개발 계수 | OOS 계수 | OOS 95% CI | OOS BH q | 0.05 SD 기준 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for outcome in mechanism_decisions["outcome"]:
        development = coefficients.loc[
            coefficients["split"].eq("development")
            & coefficients["family"].eq("mechanism")
            & coefficients["outcome"].eq(outcome)
        ].iloc[0]
        oos = mechanism_decisions.loc[
            mechanism_decisions["outcome"].eq(outcome)
        ].iloc[0]
        lines.append(
            f"| {OUTCOME_LABELS[outcome]} | {development['coefficient']:.4f} | "
            f"{oos['coefficient']:.4f} | [{oos['ci_lower']:.4f}, {oos['ci_upper']:.4f}] | "
            f"{oos['q_value_bh_fdr']:.4g} | {'통과' if oos['passes_predeclared_criteria'] else '미통과'} |"
        )
    passed_mechanisms = mechanism_decisions.loc[
        mechanism_decisions["passes_predeclared_criteria"], "outcome"
    ].tolist()
    lines.extend(
        [
            "",
            f"- 개별 기준 통과: {len(passed_mechanisms)}/5 ({', '.join(OUTCOME_LABELS[value] for value in passed_mechanisms) if passed_mechanisms else '없음'})",
            f"- 3개 이상이면서 Amihud형 비유동성 또는 가격 이산성을 포함해야 하는 종합 기준은 {'통과했습니다.' if mechanism_success else '통과하지 못했습니다.'}",
            "- `zero_tick_share`는 zero-run 통계와 구성상 직접 연결되므로, 유의하더라도 독립적인 경제 메커니즘 증거로 단독 해석하지 않습니다.",
            "",
            "## 2. 미래 움직임 크기",
            "",
            "아래 계수는 현재 정보와 자산·UTC hour·weekday 고정효과를 통제한 OOS 회귀 결과이며, zero-run intensity 1표준편차 변화당 basis point입니다.",
            "",
            "| 결과 | 기간 | OOS 계수(bp) | 95% CI | clustered BH q | permutation BH q | 경제성 | 최종 |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in future_decisions.sort_values(["family", "horizon_minutes"]).itertuples(index=False):
        lines.append(
            f"| {FAMILY_LABELS[row.family]} | {int(row.horizon_minutes)}분 | "
            f"{row.coefficient:.4f} | [{row.ci_lower:.4f}, {row.ci_upper:.4f}] | "
            f"{row.q_value_bh_fdr:.4g} | {row.q_value_bh_fdr_permutation:.4g} | "
            f"{row.effect_threshold_bps:.3f}bp 이상 | "
            f"{'통과' if row.passes_predeclared_criteria else '미통과'} |"
        )
    lines.extend(["", "### 왜 최종 기준이 엄격한가", ""])
    for row in future_decisions.sort_values(["family", "horizon_minutes"]).itertuples(index=False):
        failed = []
        if not row.passes_cluster_q:
            failed.append("clustered q")
        if not row.passes_permutation_q:
            failed.append("permutation")
        if not row.passes_effect_size:
            failed.append("경제 크기")
        if not row.passes_loao:
            failed.append("LOAO")
        if not row.passes_oos_prediction:
            failed.append("개발→OOS RMSE")
        if row.placebo_veto:
            failed.append("미래-lead placebo veto")
        lines.append(
            f"- {FAMILY_LABELS[row.family]} {int(row.horizon_minutes)}분: "
            f"{'모든 기준 통과' if not failed else '미통과 항목 = ' + ', '.join(failed)}"
        )
    lines.extend(
        [
            "",
            "## 3. 개발→OOS 실제 예측 성능",
            "",
            "개발 기간에서 계수를 학습한 뒤 OOS에 고정 적용했습니다. 양수 개선률은 zero-run을 추가한 모형의 오차가 작아졌다는 뜻입니다.",
            "",
            "| 결과 | 기간 | baseline RMSE | augmented RMSE | RMSE 개선 | MAE 개선 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in prediction_metrics.sort_values(["family", "horizon_minutes"]).itertuples(index=False):
        lines.append(
            f"| {FAMILY_LABELS[row.family]} | {int(row.horizon_minutes)}분 | "
            f"{row.baseline_rmse:.4f} | {row.augmented_rmse:.4f} | "
            f"{row.rmse_improvement_percent:.4f}% | {row.mae_improvement_percent:.4f}% |"
        )
    lines.extend(
        [
            "",
            "## 4. 강건성 검증",
            "",
            "### 자산 하나씩 제외(LOAO)",
            "",
            "| 결과 | 기간 | 같은 부호 | 중앙 절대크기 비율 | 범위(bp) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in loao_summary.sort_values(["family", "horizon_minutes"]).itertuples(index=False):
        lines.append(
            f"| {FAMILY_LABELS[row.family]} | {int(row.horizon_minutes)}분 | "
            f"{row.same_sign_count}/{row.loao_runs} | {row.median_magnitude_ratio:.1%} | "
            f"{row.minimum_coefficient:.3f} ~ {row.maximum_coefficient:.3f} |"
        )
    lines.extend(
        [
            "",
            "### Circular-shift permutation",
            "",
            f"OOS zero-run 잔차를 최소 {config['analysis']['permutation_minimum_shift_days']}일 이동해 {config['analysis']['permutation_repetitions']}회 재검정했습니다.",
            "",
            "| 결과 | 기간 | 실제 잔차계수 | null 95% | empirical p | BH q |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in permutation_summary.sort_values(["family", "horizon_minutes"]).itertuples(index=False):
        lines.append(
            f"| {FAMILY_LABELS[row.family]} | {int(row.horizon_minutes)}분 | "
            f"{row.observed_residualized_coefficient:.4f} | "
            f"[{row.null_q025:.4f}, {row.null_q975:.4f}] | "
            f"{row.empirical_p_value:.4g} | {row.q_value_bh_fdr:.4g} |"
        )
    lines.extend(
        [
            "",
            "### 미래 정보 lead placebo",
            "",
            f"실거래에 쓸 수 없는 {config['analysis']['placebo_lead_days']}일 뒤 zero-run을 붙여 느린 regime 교란을 확인했습니다. 같은 부호·절반 이상 크기의 유의한 placebo는 실제 결과를 veto합니다.",
            "",
            "| 결과 | 기간 | placebo 계수 | BH q | veto |",
            "|---|---:|---:|---:|---|",
        ]
    )
    placebo_lookup = future_decisions.set_index(["family", "horizon_minutes"])
    for row in placebo.sort_values(["family", "horizon_minutes"]).itertuples(index=False):
        veto = bool(placebo_lookup.loc[(row.family, row.horizon_minutes), "placebo_veto"])
        lines.append(
            f"| {FAMILY_LABELS[row.family]} | {int(row.horizon_minutes)}분 | "
            f"{row.coefficient:.4f} | {row.q_value_bh_fdr:.4g} | {'예' if veto else '아니오'} |"
        )
    lines.extend(
        [
            "",
            "## 해석",
            "",
        ]
    )
    if absolute_success or volatility_success:
        lines.append(
            "- 적어도 한 미래 움직임 family가 사전 기준을 통과했습니다. 이는 zero-run이 가격 방향이 아니라 단기 위험 크기를 조절하는 보조 feature가 될 가능성과 일치합니다. 완전히 새로운 기간 또는 거래소의 외부 검증 전에는 운영에 사용하지 않습니다."
        )
    else:
        lines.append(
            "- 미래 움직임 family가 사전 기준을 통과하지 못했습니다. 동일 표본에서 threshold나 종목을 다시 고르지 않고, zero-run의 독립적인 단기 위험 예측력은 현재 근거로 지지되지 않는다고 결론냅니다."
        )
    if mechanism_success:
        lines.append(
            "- 여러 미시구조 대리변수에서 구조적 연관성이 관찰됐습니다. 다만 동시점 관계이며 zero-run 정의와 일부 기계적 중첩이 있으므로 원인이라고 부를 수 없습니다."
        )
    else:
        lines.append(
            "- 미시구조 종합 기준도 통과하지 못했습니다. 개별 유의 결과가 있더라도 넓은 시장 미시구조 상태를 대표한다고 확대하지 않습니다."
        )
    lines.extend(
        [
            "- 이 결과는 방향성 alpha의 실패나 성공을 재검정한 것이 아닙니다. 연구 질문 자체가 움직임의 크기와 거래 구조입니다.",
            "- tracker, paper-sim, 자동매매는 계속 비활성 상태로 둡니다.",
            "",
            "## 한계",
            "",
            "- Binance aggTrades에는 당시 bid·ask quote가 없으므로 quoted spread나 effective spread를 직접 계산하지 못했습니다.",
            "- 거래 간격은 개별 fill 사이 간격이 아니라 15분 길이를 aggregate trade 수로 나눈 대리변수입니다.",
            "- zero-tick 비중은 zero-run intensity와 수학적으로 연결되므로 완전히 독립적인 설명변수가 아닙니다.",
            "- 30분 절대움직임의 유사한 결과를 과거 연구에서 일부 확인했으므로 이번 전체 연구는 완전한 미관찰 confirmatory가 아닙니다.",
            "- 단일 거래소와 7개 survivor 자산의 2년 표본이므로 외부 타당성이 제한됩니다.",
            "",
            "## 재현과 산출물",
            "",
            "- 사전등록: `research_protocols/zero_run_microstructure_v1.md`",
            "- 설정: `configs/research/zero_run_microstructure_v1.yaml`",
            "- 실행: `.venv/bin/python scripts/run_zero_run_microstructure.py`",
            "- 검증: `.venv/bin/python scripts/verify_zero_run_microstructure.py`",
            "- 주요 원자료 파생 frame은 `analysis_frame.parquet`, 전체 판정은 `future_decisions.csv`와 `family_decisions.csv`에 저장했습니다.",
            "",
            "## 그림",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in plot_paths)
    lines.extend(
        [
            "",
            "## 모형 진단 요약",
            "",
            f"- 최소 모형 관측치: {int(diagnostics['observations'].min()):,}",
            f"- 최소 UTC-day cluster: {int(diagnostics['utc_day_clusters'].min()):,}",
            f"- 최대 condition number: {diagnostics['condition_number'].max():.1f}",
            "",
        ]
    )
    return "\n".join(lines)


def plot_mechanism_coefficients(decisions: pd.DataFrame, path: str | Path) -> None:
    plt = _get_pyplot()
    _configure_font(plt)
    local = decisions.set_index("outcome").reindex(list(OUTCOME_LABELS))
    values = local["coefficient"].to_numpy(dtype=float)
    lower = local["ci_lower"].to_numpy(dtype=float)
    upper = local["ci_upper"].to_numpy(dtype=float)
    colors = np.where(local["passes_predeclared_criteria"], "#b2472f", "#2f6d78")
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    positions = np.arange(len(local))
    ax.bar(positions, values, color=colors, alpha=0.88)
    ax.errorbar(
        positions,
        values,
        yerr=np.vstack([values - lower, upper - values]),
        fmt="none",
        color="#222222",
        capsize=4,
    )
    ax.axhline(0.0, color="#222222", linewidth=1)
    ax.axhline(0.05, color="#b2472f", linestyle="--", linewidth=1)
    ax.axhline(-0.05, color="#b2472f", linestyle="--", linewidth=1)
    ax.set_xticks(positions, [OUTCOME_LABELS[value] for value in local.index], rotation=18)
    ax.set_ylabel("표준화 계수")
    ax.set_title("Zero-run intensity와 동시점 미시구조 대리변수")
    ax.grid(axis="y", alpha=0.2)
    _save_figure(fig, path, plt)


def plot_future_coefficients(decisions: pd.DataFrame, path: str | Path) -> None:
    plt = _get_pyplot()
    _configure_font(plt)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for ax, family in zip(
        axes, ("future_absolute_return", "future_realized_volatility")
    ):
        local = decisions.loc[decisions["family"].eq(family)].sort_values(
            "horizon_minutes"
        )
        x = np.arange(len(local))
        values = local["coefficient"].to_numpy(dtype=float)
        lower = local["ci_lower"].to_numpy(dtype=float)
        upper = local["ci_upper"].to_numpy(dtype=float)
        colors = np.where(local["passes_predeclared_criteria"], "#b2472f", "#2f6d78")
        ax.scatter(x, values, color=colors, s=55, zorder=3)
        ax.errorbar(
            x,
            values,
            yerr=np.vstack([values - lower, upper - values]),
            fmt="none",
            color="#333333",
            capsize=4,
        )
        ax.axhline(0.0, color="#222222", linewidth=1)
        ax.set_xticks(x, [f"{int(value)}분" for value in local["horizon_minutes"]])
        ax.set_ylabel("OOS 계수 (bp / 1 SD)")
        ax.set_title(FAMILY_LABELS[family])
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Zero-run intensity와 미래 움직임 크기")
    _save_figure(fig, path, plt)


def plot_prediction_improvement(metrics: pd.DataFrame, path: str | Path) -> None:
    plt = _get_pyplot()
    _configure_font(plt)
    local = metrics.sort_values(["family", "horizon_minutes"]).copy()
    labels = [
        f"{'절대수익률' if row.family == 'future_absolute_return' else '실현변동성'}\n{int(row.horizon_minutes)}분"
        for row in local.itertuples(index=False)
    ]
    values = local["rmse_improvement_percent"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.bar(np.arange(len(local)), values, color=np.where(values >= 0.25, "#b2472f", "#2f6d78"))
    ax.axhline(0.0, color="#222222", linewidth=1)
    ax.axhline(0.25, color="#b2472f", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(local)), labels)
    ax.set_ylabel("개발→OOS RMSE 개선률 (%)")
    ax.set_title("Zero-run feature의 증분 OOS 예측 성능")
    ax.grid(axis="y", alpha=0.2)
    _save_figure(fig, path, plt)


def plot_permutation_comparison(summary: pd.DataFrame, path: str | Path) -> None:
    plt = _get_pyplot()
    _configure_font(plt)
    local = summary.sort_values(["family", "horizon_minutes"]).copy()
    labels = [
        f"{'절대수익률' if row.family == 'future_absolute_return' else '실현변동성'}\n{int(row.horizon_minutes)}분"
        for row in local.itertuples(index=False)
    ]
    x = np.arange(len(local))
    observed = local["observed_residualized_coefficient"].to_numpy(dtype=float)
    lower = local["null_q025"].to_numpy(dtype=float)
    upper = local["null_q975"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.vlines(x, lower, upper, color="#9a9a90", linewidth=8, alpha=0.7, label="permutation 95%")
    ax.scatter(x, observed, color="#b2472f", s=55, zorder=3, label="실제 계수")
    ax.axhline(0.0, color="#222222", linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("잔차화 계수 (bp)")
    ax.set_title("실제 시점 정렬과 circular-shift null 비교")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    _save_figure(fig, path, plt)


def _get_pyplot():
    import matplotlib.pyplot as plt

    return plt


def _configure_font(plt) -> None:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
    plt.rcParams["axes.unicode_minus"] = False


def _save_figure(fig, path: str | Path, plt) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)

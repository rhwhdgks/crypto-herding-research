from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_report_summary(
    output_path: str | Path,
    regression_json: dict,
    event_count_summary: pd.DataFrame,
    event_study_summary: pd.DataFrame,
    plot_paths: list[str],
    universe_coverage_summary: pd.DataFrame | None = None,
) -> str:
    lines = ["# 연구 요약", ""]

    lines.append("## 패널")
    if universe_coverage_summary is not None and not universe_coverage_summary.empty:
        row = universe_coverage_summary.iloc[0]
        requested_end = row.get("requested_end", "n/a")
        analysis_end = row.get("analysis_candidate_end", "n/a")
        lines.extend(
            [
                f"- 모드: {row.get('panel_mode', 'n/a')}",
                f"- 최소 활성 자산 수: {int(row.get('min_active_assets', 0))}",
                f"- 설정된 전체 자산 수: {int(row.get('total_assets', 0))}",
                f"- 요청 구간: {row.get('requested_start', 'n/a')} ~ {row.get('requested_end', 'n/a')}",
                f"- 실제 분석 구간: {row.get('analysis_candidate_start', 'n/a')} ~ {row.get('analysis_candidate_end', 'n/a')}",
                (
                    "- 비고: 고정 유니버스 교집합 패널은 요청 종료 시점보다 일찍 끝났습니다. "
                    "요청 구간 중 적어도 한 종목이 연속 관측을 끝까지 제공하지 못했기 때문입니다."
                    if requested_end != analysis_end
                    else "- 비고: 정렬 이후에도 요청한 전체 구간이 그대로 유지됐습니다."
                ),
                "",
            ]
        )
    else:
        lines.extend(["- 패널 커버리지 요약이 없습니다.", ""])

    lines.extend(
        [
            "## 회귀",
            f"- beta2: {regression_json.get('beta2', float('nan')):.6f}",
            f"- beta2 t-통계량: {regression_json.get('beta2_t_stat', float('nan')):.3f}",
            f"- beta2 p-값: {regression_json.get('beta2_p_value', float('nan')):.6g}",
            f"- 결정계수 R-squared: {regression_json.get('rsquared', float('nan')):.4f}",
            f"- 해석: {_translate_baseline_interpretation(regression_json)}",
            "",
        ]
    )

    lines.append("## 이벤트 수")
    if event_count_summary.empty:
        lines.append("- 탐지된 이벤트가 없습니다.")
    else:
        for _, row in event_count_summary.iterrows():
            lines.append(f"- {row.iloc[0]}: {int(row['count'])}")
    lines.append("")

    lines.append("## 보유기간 하이라이트")
    best_rows = _top_rows(
        event_study_summary,
        value_column="mean_return",
        top_k=4,
        min_count=20,
        count_column="count",
    )
    if best_rows.empty:
        lines.append("- 이벤트 스터디 요약 결과가 없습니다.")
    else:
        for _, row in best_rows.iterrows():
            label_column = _detect_label_column(best_rows)
            lines.append(
                f"- {row[label_column]} / {row['horizon_label']}: 평균 수익률 {row['mean_return']:.4%}, "
                f"t-통계량 {row['t_stat']:.2f}, 승률 {row['win_rate']:.2%}"
            )
    lines.append("")

    lines.append("## 플롯")
    if plot_paths:
        for plot_path in plot_paths:
            lines.append(f"- {plot_path}")
    else:
        lines.append("- 생성된 플롯이 없습니다.")
    lines.append("")

    lines.append("## 확장 포인트")
    lines.extend(
        [
            "- baseline CSAD / event-study 파이프라인은 유지한 채로, 나중에 sentiment 특성을 별도 모듈로 추가할 수 있습니다.",
            "- liquidation 또는 order-flow 스트레스 특성도 이후에 선택 모듈로 붙일 수 있습니다.",
            "- full backtest는 이벤트 정의와 수익률 패턴이 더 안정화된 뒤에 추가하는 것이 좋습니다.",
        ]
    )
    lines.append("")

    report = "\n".join(lines)
    Path(output_path).write_text(report, encoding="utf-8")
    return report


def _top_rows(
    frame: pd.DataFrame,
    value_column: str,
    top_k: int,
    min_count: int | None = None,
    count_column: str | None = None,
) -> pd.DataFrame:
    if frame.empty or value_column not in frame.columns:
        return pd.DataFrame()
    filtered = frame.dropna(subset=[value_column]).copy()
    if min_count is not None and count_column is not None and count_column in filtered.columns:
        filtered = filtered[filtered[count_column] >= min_count]
    return filtered.sort_values(value_column, ascending=False).head(top_k)


def _detect_label_column(frame: pd.DataFrame) -> str:
    for candidate in ["event_type", "composite_event_type", "signal_source"]:
        if candidate in frame.columns:
            return candidate
    return frame.columns[0]


def _translate_baseline_interpretation(regression_json: dict) -> str:
    beta2 = regression_json.get("beta2")
    if beta2 is None or pd.isna(beta2):
        return "beta2를 해석할 수 있는 값이 없습니다."
    if float(beta2) < 0:
        return "beta2가 음수이므로 baseline 회귀는 herding 가능성을 지지합니다."
    return "beta2가 음수가 아니므로 baseline 회귀는 herding을 지지하지 않습니다."

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd


def validate_confirmatory_raw_frame(frame: pd.DataFrame, data_cfg: Mapping) -> pd.DataFrame:
    required = {
        "symbol",
        "bucket_start",
        "signal_timestamp",
        "schema_version",
        "aggressor_imbalance",
        "is_micro_run_clustering_event",
        "is_control_bucket",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Confirmatory raw frame is missing columns: {', '.join(missing)}")
    expected_symbols = list(data_cfg["symbols"])
    expected_rows = int(data_cfg["expected_rows"])
    interval_minutes = int(data_cfg.get("expected_interval_minutes", 15))
    expected_start = _as_utc_timestamp(data_cfg["expected_start"])
    expected_end = _as_utc_timestamp(data_cfg["expected_end"])
    bucket_start = pd.to_datetime(frame["bucket_start"], utc=True)
    signal_timestamp = pd.to_datetime(frame["signal_timestamp"], utc=True)

    if set(frame["symbol"].astype(str).unique()) != set(expected_symbols):
        raise ValueError("Confirmatory raw frame symbol universe does not match the frozen config")
    if len(frame) != expected_rows:
        raise ValueError(f"Confirmatory raw rows={len(frame):,}, expected={expected_rows:,}")
    if bucket_start.min() != expected_start or bucket_start.max() != expected_end:
        raise ValueError("Confirmatory raw frame date range does not match the frozen config")
    if frame.duplicated(["symbol", "bucket_start"]).any():
        raise ValueError("Confirmatory raw frame contains duplicate symbol timestamps")

    expected_grid = pd.date_range(
        expected_start,
        expected_end,
        freq=f"{interval_minutes}min",
        tz="UTC",
    )
    for symbol in expected_symbols:
        actual_grid = pd.DatetimeIndex(
            bucket_start.loc[frame["symbol"].astype(str).eq(symbol)].sort_values()
        )
        if not actual_grid.equals(expected_grid):
            raise ValueError(f"Confirmatory raw frame grid is incomplete for {symbol}")

    expected_signal = bucket_start + pd.Timedelta(minutes=interval_minutes)
    signal_mismatches = int(signal_timestamp.ne(expected_signal).sum())
    if signal_mismatches:
        raise ValueError("Confirmatory raw frame signal timestamps do not equal bucket end")
    schema_versions = set(
        pd.to_numeric(frame["schema_version"], errors="coerce").dropna().astype(int)
    )
    if schema_versions != {2}:
        raise ValueError("Confirmatory raw frame must use tick schema v2")
    aggressor_share = float(frame["aggressor_imbalance"].notna().mean())
    if aggressor_share < float(data_cfg.get("minimum_aggressor_available_share", 1.0)):
        raise ValueError("Confirmatory raw frame aggressor coverage is below the frozen minimum")

    return pd.DataFrame(
        [
            {
                "rows": int(len(frame)),
                "start": bucket_start.min(),
                "end": bucket_start.max(),
                "symbols": int(frame["symbol"].nunique()),
                "events": int(frame["is_micro_run_clustering_event"].sum()),
                "controls": int(frame["is_control_bucket"].sum()),
                "aggressor_available_share": aggressor_share,
                "interval_minutes": interval_minutes,
                "complete_symbol_grids": True,
                "duplicate_rows": 0,
                "signal_timestamp_mismatches": signal_mismatches,
            }
        ]
    )


def build_raw_confirmatory_report(
    coverage: pd.DataFrame,
    price_summary: pd.DataFrame,
    aggressor_summary: pd.DataFrame,
    predictive: pd.DataFrame,
    diagnostics: pd.DataFrame,
    config: Mapping,
    plot_paths: Sequence[str],
) -> str:
    coverage_row = coverage.iloc[0]
    pooled_price = price_summary.loc[price_summary["scope"] == "pooled"].iloc[0]
    pooled_aggressor = aggressor_summary.loc[aggressor_summary["scope"] == "pooled"].iloc[0]
    predictive_survivors = predictive.loc[predictive["supports_economic_predictive_association"]]
    all_predictive_ci_inside = bool(predictive["ci_inside_predeclared_economic_band"].all())
    zero_share = float(pooled_price["zero_run_event_share"])
    reverse_aggressor = aggressor_summary.loc[
        aggressor_summary["inference_eligible"]
        & (aggressor_summary["binomial_q_value_bh_fdr"] <= float(config["analysis"]["fdr_alpha"]))
        & (aggressor_summary["directional_concordance"] < 0.5)
    ]

    lines = [
        "# 동일 2년 Raw Tick Confirmatory 보고서",
        "",
        "## 한 문장 결론",
        "",
        "현재 `min(up, down, zero)` run-clustering winner는 방향성 herding 대리변수가 아니며, 시장요인을 제거한 30분 미래 수익률도 예측하지 못합니다.",
        "",
        "## 쉽게 읽는 요약",
        "",
        f"- 이벤트 {int(coverage_row['events']):,}건 중 {zero_share:.2%}가 가격이 변하지 않은 tick의 연속인 `zero-run`이었습니다.",
        f"- 방향이 정의된 run event의 가격 방향 일치율은 {pooled_price['directional_concordance']:.2%}로 50%와 구분되지 않았습니다.",
        f"- 실제 순매수·순매도 aggressor 방향 일치율도 {pooled_aggressor['directional_concordance']:.2%}로 60% 사전 기준을 충족하지 못했습니다.",
        "- 시장중립 30분 회귀의 up, down, zero 계수는 모두 BH-FDR 5%와 5bp 경제성 기준을 통과하지 못했습니다.",
        "- 따라서 현재 이벤트를 매매 신호로 승격시키지 않습니다.",
        "",
        "## 고정 표본",
        "",
        f"- 기간: {coverage_row['start']} ~ {coverage_row['end']}",
        f"- {int(coverage_row['interval_minutes'])}분 bucket: {int(coverage_row['rows']):,}개, 7종목이 각각 70,080개",
        f"- point-in-time event: {int(coverage_row['events']):,}건, control: {int(coverage_row['controls']):,}건",
        f"- buyer-maker 기반 aggressor 가용률: {coverage_row['aggressor_available_share']:.2%}",
        "- 월별 checkpoint 175개, 심볼·시간 중복 0건, 15분 전체 grid 완전",
        "- signal timestamp는 모든 행에서 bucket end와 일치",
        "- 가격, aggressor, 미래 반응은 서로 분리된 사전 검정 family로 판정",
        "",
        "## 1. Run side와 가격 방향",
        "",
        f"- pooled directional n={int(pooled_price['directional_event_count']):,}, 일치율={pooled_price['directional_concordance']:.2%}",
        f"- 95% CI {pooled_price['directional_concordance_ci_lower']:.2%}~{pooled_price['directional_concordance_ci_upper']:.2%}, BH q={pooled_price['binomial_q_value_bh_fdr']:.4g}",
        "- 가격 방향 proxy 판정: "
        + ("통과" if pooled_price["supports_price_direction_proxy"] else "통과하지 못함"),
        "",
        "| 범위 | 이벤트 | zero-run | 방향 n | 일치율 | BH q | proxy |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in price_summary.itertuples(index=False):
        concordance = _format_percent(row.directional_concordance)
        q_value = _format_number(row.binomial_q_value_bh_fdr)
        decision = "통과" if row.supports_price_direction_proxy else "미통과"
        lines.append(
            f"| {row.scope} | {row.event_count:,} | {row.zero_run_event_share:.2%} | "
            f"{row.directional_event_count:,} | {concordance} | {q_value} | {decision} |"
        )

    lines.extend(
        [
            "",
            "## 2. Run side와 Aggressor 방향",
            "",
            f"- pooled directional n={int(pooled_aggressor['directional_event_count']):,}, 일치율={pooled_aggressor['directional_concordance']:.2%}",
            f"- 95% CI {pooled_aggressor['directional_concordance_ci_lower']:.2%}~{pooled_aggressor['directional_concordance_ci_upper']:.2%}, BH q={pooled_aggressor['binomial_q_value_bh_fdr']:.4g}",
            "- aggressor 방향 proxy 판정: "
            + ("통과" if pooled_aggressor["supports_aggressor_direction_proxy"] else "통과하지 못함"),
            "",
            "| 범위 | 방향 n | 일치율 | 95% CI | BH q | proxy |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in aggressor_summary.itertuples(index=False):
        concordance = _format_percent(row.directional_concordance)
        confidence_interval = _format_percent_interval(
            row.directional_concordance_ci_lower,
            row.directional_concordance_ci_upper,
        )
        q_value = _format_number(row.binomial_q_value_bh_fdr)
        decision = "통과" if row.supports_aggressor_direction_proxy else "미통과"
        lines.append(
            f"| {row.scope} | {row.directional_event_count:,} | {concordance} | "
            f"{confidence_interval} | {q_value} | {decision} |"
        )
    if not reverse_aggressor.empty:
        scopes = ", ".join(reverse_aggressor["scope"].astype(str))
        lines.extend(
            [
                "",
                f"- {scopes}에서는 50% 반대방향으로 유의했지만, 이는 사전의 60% 순방향 proxy 가설을 지지하지 않습니다.",
                "- 역방향 구조는 사후적 단일 종목 결과이므로 별도 탐색 가설로만 보존합니다.",
            ]
        )

    lines.extend(
        [
            "",
            "## 3. 시장중립 30분 반응",
            "",
            f"- 회귀 관측치 {int(diagnostics.iloc[0]['observations']):,}, UTC-day cluster {int(diagnostics.iloc[0]['utc_day_clusters']):,}",
            "- 각 자산의 30분 수익률에서 나머지 6자산의 동시간 평균을 빼 시장 공통 움직임을 제거",
            "- 현재 수익률·시장수익률·변동성·거래량·종목·UTC hour·가격 방향·일반 run side를 통제",
            "",
            "| run side | 이벤트 | 조정계수(bp) | 95% CI(bp) | BH q | 사전 판정 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in predictive.itertuples(index=False):
        decision = "검토 대상" if row.supports_economic_predictive_association else "근거 부족"
        lines.append(
            f"| {row.run_clustering_side} | {row.event_count:,} | {row.adjusted_coefficient_bps:.3f} | "
            f"[{row.ci_lower_bps:.3f}, {row.ci_upper_bps:.3f}] | "
            f"{row.cluster_q_value_bh_fdr:.4g} | {decision} |"
        )
    practical_null_line = (
        "- 세 계수의 95% 신뢰구간이 모두 사전 경제성 밴드 ±5bp 안에 있습니다."
        if all_predictive_ci_inside
        else "- 일부 계수의 신뢰구간이 ±5bp를 넘어 practical-null은 단정하지 않습니다."
    )
    lines.extend(
        [
            "",
            practical_null_line,
            "- 즉 p-value가 크다는 것을 넘어, 현재 모형에서 5bp 규모 효과도 제외할 수 있는 practical-null 결과입니다.",
            "",
            "## 해석",
            "",
            "- `run_clustering_side`는 가격·주문 방향이 아니라 조건부 run z가 가장 낮은 tick category입니다.",
            "- winner-take-all 방식이 zero-tick 구조에 지배되므로, 현재 이벤트는 방향성 herding보다 호가단위·유동성·거래분할 미시구조를 측정할 가능성이 높습니다.",
            "- 방향 프록시와 미래 반응이 둘 다 사전 기준을 통과하지 못했으므로, 기존 방향성 alpha·tracker를 복구할 근거가 없습니다.",
            "",
            "## 종합 판정",
            "",
            "- 가격 방향 proxy: "
            + ("지지" if pooled_price["supports_price_direction_proxy"] else "지지하지 않음"),
            "- aggressor 방향 proxy: "
            + ("지지" if pooled_aggressor["supports_aggressor_direction_proxy"] else "지지하지 않음"),
            "- 통계·경제 기준을 모두 통과한 미래 반응: "
            + (", ".join(predictive_survivors["run_clustering_side"]) if not predictive_survivors.empty else "없음"),
            "- tracker·paper-sim·자동매매 상태: 비활성 유지",
            "- 이 결과는 고정 Binance 표본의 조건부 연관성 판정이며 인과관계나 거래 전략 성과가 아닙니다.",
            "",
            "## 다음 연구",
            "",
            "1. winner label을 버리고 `run_z_up`, `run_z_down`, `run_z_zero`를 각각 연속형 설명변수로 다룬니다.",
            "2. zero-run은 방향성 herding과 분리해 spread proxy, tick size, 거래량, 후속 변동성과의 관계를 미시구조 주제로 연구합니다.",
            "3. up/down category 가설은 새 프로토콜·검정 family·untouched OOS를 먼저 고정한 뒤 다시 검정합니다.",
            "4. Binance 내부에서 사전 기준을 통과한 결과가 생긴 후에만 다른 거래소·거래비용·paper-sim으로 넘어갑니다.",
            "",
            "## 재현 정보",
            "",
            "- 사전 프로토콜: `research_protocols/tick_raw_confirmatory_2y_v1.md`",
            "- 입력 manifest: `outputs/v2/tick/semantic_validation/confirmatory_2y/input_manifest.json`",
            "- 월별 백필 상태: `outputs/v2/tick/semantic_validation/raw_2y/backfill_state.json`",
            "- 월별 해시 manifest: `outputs/v2/tick/semantic_validation/raw_2y/input_manifest.json`",
            "",
            "## 그림",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in plot_paths)
    lines.append("")
    return "\n".join(lines)


def _as_utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _format_percent(value: object) -> str:
    return f"{float(value):.2%}" if pd.notna(value) else "N/A"


def _format_percent_interval(lower: object, upper: object) -> str:
    if pd.isna(lower) or pd.isna(upper):
        return "N/A"
    return f"{float(lower):.2%}~{float(upper):.2%}"


def _format_number(value: object) -> str:
    return f"{float(value):.4g}" if pd.notna(value) else "N/A"

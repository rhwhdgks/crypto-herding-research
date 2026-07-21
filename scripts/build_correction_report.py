from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "v2" / "CORRECTION_REPORT_2026-07-15.md"


def _load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _matrix_stats(path: str) -> dict:
    frame = pd.read_csv(ROOT / path)
    return {
        "rows": len(frame),
        "eligible": int(frame["inference_eligible"].fillna(False).sum()),
        "raw_p05": int((frame["p_value_block"] <= 0.05).sum()),
        "fdr": int(frame["reject_block_bh_fdr_05"].fillna(False).sum()),
        "min_q": float(frame["p_value_block_bh_fdr"].min()),
        "frame": frame,
    }


def _corrected_label_row(path: str, symbol: str, run_side: str) -> dict:
    frame = pd.read_csv(
        ROOT / path,
        usecols=[
            "symbol",
            "is_micro_run_clustering_event",
            "run_clustering_side",
            "price_direction",
            "bucket_return",
        ],
    )
    event = frame["is_micro_run_clustering_event"].astype(str).str.lower().eq("true")
    subset = frame.loc[event & frame["symbol"].eq(symbol) & frame["run_clustering_side"].eq(run_side)]
    return {
        "count": len(subset),
        "up_share": float(subset["price_direction"].eq("up").mean()) if len(subset) else float("nan"),
        "down_share": float(subset["price_direction"].eq("down").mean()) if len(subset) else float("nan"),
        "mean": float(subset["bucket_return"].mean()) if len(subset) else float("nan"),
    }


def _pct(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.2%}"


def main() -> int:
    one_year = _matrix_stats(
        "outputs/v2/tick/multi_asset_365d/lead_lag_matrix/lead_lag_matrix_summary.csv"
    )
    five_year = _matrix_stats(
        "outputs/v2/tick/multi_asset_5y/lead_lag_matrix/lead_lag_matrix_summary.csv"
    )
    period = pd.read_csv(
        ROOT / "outputs/v2/tick/multi_asset_5y/lead_lag_robustness/period_stability_matrix.csv"
    )
    regime = pd.read_csv(
        ROOT / "outputs/v2/tick/multi_asset_5y/lead_lag_robustness/volatility_regime_matrix.csv"
    )
    diagnostics = pd.read_csv(
        ROOT / "outputs/v2/tick/corrected_candidate/state_diagnostics/corrected_state_diagnostics.csv"
    )
    candidate = _load_json(
        "outputs/v2/tick/corrected_candidate/validation/corrected_candidate_summary.json"
    )
    permutation = _load_json(
        "outputs/v2/tick/corrected_candidate/validation/selection_aware_permutation.json"
    )
    regression = _load_json("outputs/baseline/regression_results.json")

    xrp_5y = _corrected_label_row(
        "outputs/v2/tick/multi_asset_5y/lead_lag_matrix/intermediate/tick_micro_frame_15m.csv",
        "XRPUSDT",
        "up",
    )
    xrp_1y = _corrected_label_row(
        "outputs/v2/tick/multi_asset_365d/lead_lag_matrix/intermediate/tick_micro_frame_15m.csv",
        "XRPUSDT",
        "up",
    )
    doge_1y = _corrected_label_row(
        "outputs/v2/tick/multi_asset_365d/lead_lag_matrix/intermediate/tick_micro_frame_15m.csv",
        "DOGEUSDT",
        "down",
    )
    avax_1y = _corrected_label_row(
        "outputs/v2/tick/multi_asset_365d/lead_lag_matrix/intermediate/tick_micro_frame_15m.csv",
        "AVAXUSDT",
        "down",
    )
    doge_edges = five_year["frame"].loc[
        five_year["frame"]["leader"].eq("DOGEUSDT")
        & five_year["frame"]["target"].isin(["ADAUSDT", "AVAXUSDT"])
        & five_year["frame"]["event_filter"].eq(
            "price_direction=down__run_clustering_side=down"
        )
    ].sort_values("target")
    smoke = pd.read_csv(
        ROOT / "outputs/v2/tick/raw_schema_smoke_2026-04-06/intermediate/tick_micro_frame_15m.csv"
    )

    beta2 = regression["coefficients"]["market_return_sq"]
    cov_type = regression.get("cov_type", "HAC")
    selection_p = permutation.get("p_value_add_one")
    selection_p_text = "N/A" if pd.isna(selection_p) else f"{float(selection_p):.4f}"
    lines = [
        "# Crypto Herding 교정 보고서",
        "",
        "- 기준일: 2026-07-15",
        "- 상태: corrected research, 거래 후보 미확정",
        "- authoritative output namespace: `outputs/v2/`",
        "",
        "## 1. Root cause",
        "",
        "구형 `micro_herding_up/down`은 가격 방향이 아니라 legacy run statistic에서 선택된 tick category였습니다. 이 값을 가격 상승·하락으로 해석하면서 DOGE/XRP 방향성부터 toxicity, funding/OI, permutation, execution, tracker까지 경제적 의미가 연쇄적으로 잘못 연결됐습니다.",
        "",
        "## 2. Changed files",
        "",
        "- 이벤트 의미·시간: `src/tick_herding.py`, `src/tick_short_horizon.py`, `src/tick_event_schema.py`",
        "- 추론·선택·실행: `src/tick_lead_lag.py`, `src/research_validation.py`, corrected candidate/diagnostic modules",
        "- runner/config: `scripts/run_tick_*`, `scripts/run_corrected_*`, `configs/tick/**`",
        "- baseline/문서/릴리스: `src/regression.py`, `src/event_detection.py`, `README.md`, `review_package_docs/**`, release build/verify scripts",
        "",
        "## 3. Event schema before/after",
        "",
        "| 구분 | legacy | schema v2 |",
        "|---|---|---|",
        "| run statistic | 고정 `p=1/3` score | category count 조건부 run z |",
        "| 방향 | `micro_herding_up/down` 한 필드 | run, price, aggressor 방향 별도 필드 |",
        "| event 시각 | bucket start로 오해 가능 | `signal_timestamp = bucket_end` |",
        "| forward return | row offset 가능 | exact clock-time join |",
        "| config | 모호한 `direction` | 명시적 `event_filter` |",
        "",
        "### 감사 사례와 corrected 분포",
        "",
        "| 표본 | legacy count·실제 방향 | corrected run-side count·실제 방향 |",
        "|---|---|---|",
        f"| XRP 5y, run side up | 2,937; 하락 96.46%; 평균 -0.5105% | {xrp_5y['count']:,}; 상승 {_pct(xrp_5y['up_share'])}, 하락 {_pct(xrp_5y['down_share'])}; 평균 {_pct(xrp_5y['mean'])} |",
        f"| XRP 365d, run side up | 599; 하락 96.83%; 평균 -0.5379% | {xrp_1y['count']:,}; 상승 {_pct(xrp_1y['up_share'])}, 하락 {_pct(xrp_1y['down_share'])}; 평균 {_pct(xrp_1y['mean'])} |",
        f"| DOGE 365d, run side down | 154; 상승 75.97%; 평균 +0.3072% | {doge_1y['count']:,}; 상승 {_pct(doge_1y['up_share'])}, 하락 {_pct(doge_1y['down_share'])}; 평균 {_pct(doge_1y['mean'])} |",
        f"| AVAX 365d, run side down | 86; 상승 88.37%; 평균 +0.4802% | {avax_1y['count']:,}; 방향 분포 N/A |",
        "",
        "## 4. Statistical-method changes",
        "",
        f"- Baseline: `beta2={beta2:.6f}`, covariance `{cov_type}`. 계수 부호가 양수이므로 broad classical CSAD herding을 지지하지 않습니다.",
        "- Tick: UTC-day cluster bootstrap, exact 30분 horizon, non-overlap event count를 사용합니다.",
        "- 추론 적격 기준: non-overlap event 30개 이상, event/control UTC day 각각 20개 이상입니다.",
        "- primary family: 252개 셀 전체에 BH-FDR을 적용합니다.",
        f"- 1년: 적격 {one_year['eligible']}/{one_year['rows']}, raw p<=0.05 {one_year['raw_p05']}개, FDR {one_year['fdr']}개.",
        f"- 5년: 적격 {five_year['eligible']}/{five_year['rows']}, raw p<=0.05 {five_year['raw_p05']}개, FDR {five_year['fdr']}개.",
        f"- period split: 적격 {int(period['inference_eligible'].sum())}/{len(period)}, FDR {int(period['reject_block_bh_fdr_05'].fillna(False).sum())}개.",
        f"- volatility split: 적격 {int(regime['inference_eligible'].sum())}/{len(regime)}, FDR {int(regime['reject_block_bh_fdr_05'].fillna(False).sum())}개.",
        f"- state diagnostic family: {len(diagnostics)}개 중 raw p<=0.05 {int((diagnostics['p_value_block'] <= .05).sum())}개, FDR {int(diagnostics['reject_block_bh_fdr_05'].fillna(False).sum())}개.",
        "",
        "## 5. Invalidated outputs",
        "",
        "`DOGE down -> AVAX/ADA`, low-toxicity, crowded funding × no OI-flush, pointwise permutation `p=0.041`, maker return, arithmetic equity/max drawdown, legacy tracker 결과는 모두 audit-only입니다. 산술 오타가 아니라 event economic-label mismatch이므로 숫자를 고쳐 재사용하지 않습니다.",
        "",
        "## 6. Regenerated outputs",
        "",
        "- `outputs/v2/tick/multi_asset_365d/lead_lag_matrix/`",
        "- `outputs/v2/tick/multi_asset_5y/lead_lag_matrix/`",
        "- `outputs/v2/tick/multi_asset_5y/lead_lag_robustness/`",
        "- `outputs/v2/tick/corrected_candidate/state_diagnostics/`",
        "- `outputs/v2/tick/corrected_candidate/validation/`",
        "- `outputs/v2/tick/raw_schema_smoke_2026-04-06/`",
        f"- raw smoke: {len(smoke)} buckets, aggressor missing {int(smoke['aggressor_imbalance'].isna().sum())}, directions {smoke['aggressor_direction'].value_counts().to_dict()}.",
        "",
        "### 기존 DOGE 후보의 corrected 비교",
        "",
        "| target | raw event | non-overlap | block 95% CI | block p | BH-FDR q |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in doge_edges.iterrows():
        lines.append(
            f"| {row['target']} | {int(row['event_count'])} | {int(row['n_nonoverlapping_events'])} | "
            f"[{row['confidence_interval_lower']:.4%}, {row['confidence_interval_upper']:.4%}] | "
            f"{row['p_value_block']:.4f} | {row['p_value_block_bh_fdr']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 7. Test commands and results",
            "",
            "최종 검증 명령은 `python -m pytest -q`, `python -m compileall -q src experiments scripts tests`, `git diff --check`입니다. 실제 최종 pass count와 release 검증값은 작업 완료 메시지에 기록합니다.",
            "",
            "## 8. Clean-room reproduction result",
            "",
            "Release archive를 새 임시 디렉터리에 풀고 새 venv에서 editable install, 전체 test, baseline을 실행합니다. 자기 자신의 archive hash는 archive 내부 문서에 포함할 수 없으므로 최종 SHA-256과 clean-room 결과는 외부 완료 기록에 남깁니다.",
            "",
            "## 9. Release verification",
            "",
            "`scripts/build_release.py`와 `scripts/verify_release.py`가 required files, duplicate entry, central directory/CRC, size와 SHA-256 manifest를 검사합니다. 하나라도 불일치하면 non-zero로 종료합니다.",
            "",
            "## 10. Remaining limitations",
            "",
            "- corrected 5년 장기 frame은 legacy aggregate cache migration이므로 aggressor direction이 unavailable입니다.",
            "- raw smoke는 parser와 schema 검증용 하루 표본이며 5년 통계 대체물이 아닙니다.",
            "- corrected candidate가 없어 selection-aware p-value, net portfolio return, max drawdown은 N/A입니다.",
            "- tracker는 disabled이며 corrected OOS signal count는 0입니다.",
            "- sentiment는 `first_seen_at_utc`가 없는 historical snapshot을 point-in-time alpha로 쓰지 않습니다.",
            "- Binance 단일 거래소 결과를 전체 암호화폐 시장으로 일반화할 수 없습니다.",
            "",
            "### 필수 수치 요약",
            "",
            "| 항목 | corrected 결과 |",
            "|---|---|",
            "| candidate family evaluable count | 0 |",
            f"| selection-aware permutation p-value | {selection_p_text} |",
            f"| complete-basket trade count | {candidate['execution']['trade_count']} |",
            "| net portfolio return | N/A |",
            "| max drawdown | N/A |",
            "| OOS signal count | 0 |",
        ]
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

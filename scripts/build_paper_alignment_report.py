from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "paper_like"
DAILY_DIR = OUTPUTS_DIR / "daily"
WEEKLY_DIR = OUTPUTS_DIR / "weekly"


EXPECTED_CLAIMS = [
    {
        "claim_id": "daily_standard_full",
        "paper_reference": "본문 기준: standard CSAD는 전체 표본에서 herding을 보이지 않아야 합니다.",
        "result_path": DAILY_DIR / "paper_regression_summary.csv",
        "model_name": "standard_csad",
        "period_name": "full_sample",
        "expected_significant_negative": False,
    },
    {
        "claim_id": "daily_scsad_full_main",
        "paper_reference": "본문 기준: daily SCSAD는 전체 표본에서 herding을 지지해야 합니다.",
        "result_path": DAILY_DIR / "paper_regression_summary.csv",
        "model_name": "scsad",
        "period_name": "full_sample",
        "expected_significant_negative": True,
    },
    {
        "claim_id": "daily_scsad_full_appendix",
        "paper_reference": "부록 A.6 기준: daily SCSAD는 확장 robustness 표본 전체에서는 herding이 약할 수 있습니다.",
        "result_path": DAILY_DIR / "paper_regression_summary.csv",
        "model_name": "scsad",
        "period_name": "full_sample",
        "expected_significant_negative": False,
    },
    {
        "claim_id": "weekly_standard_full",
        "paper_reference": "weekly standard CSAD는 음의 herding 계수를 보여야 합니다.",
        "result_path": WEEKLY_DIR / "paper_regression_summary.csv",
        "model_name": "standard_csad",
        "period_name": "full_sample",
        "expected_significant_negative": True,
    },
    {
        "claim_id": "weekly_scsad_full",
        "paper_reference": "weekly SCSAD는 음수이면서 유의한 cubic term을 보여야 합니다.",
        "result_path": WEEKLY_DIR / "paper_regression_summary.csv",
        "model_name": "scsad",
        "period_name": "full_sample",
        "expected_significant_negative": True,
    },
    {
        "claim_id": "daily_no_intercept_full",
        "paper_reference": "절편 없는 모형은 전체 표본에서 herding을 지지해야 합니다.",
        "result_path": DAILY_DIR / "paper_regression_summary.csv",
        "model_name": "no_intercept_csad",
        "period_name": "full_sample",
        "expected_significant_negative": True,
    },
    {
        "claim_id": "weekly_no_intercept_full",
        "paper_reference": "weekly 절편 제거 모형은 herding을 강하게 지지해야 합니다.",
        "result_path": WEEKLY_DIR / "paper_regression_summary.csv",
        "model_name": "no_intercept_csad",
        "period_name": "full_sample",
        "expected_significant_negative": True,
    },
]


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(DAILY_DIR / "paper_regression_summary.csv")
    weekly = pd.read_csv(WEEKLY_DIR / "paper_regression_summary.csv")

    alignment_rows = build_alignment_rows()
    alignment_frame = pd.DataFrame(alignment_rows)
    alignment_frame.to_csv(OUTPUTS_DIR / "paper_alignment_table.csv", index=False)

    period_ranking_frame = build_period_ranking_frame(daily=daily, weekly=weekly)
    period_ranking_frame.to_csv(OUTPUTS_DIR / "paper_subperiod_alignment.csv", index=False)

    report = build_alignment_report(
        alignment_frame=alignment_frame,
        period_ranking_frame=period_ranking_frame,
        daily=daily,
        weekly=weekly,
    )
    (OUTPUTS_DIR / "paper_alignment_report.md").write_text(report, encoding="utf-8")
    print(OUTPUTS_DIR / "paper_alignment_report.md")


def build_alignment_rows() -> list[dict]:
    rows = []
    for claim in EXPECTED_CLAIMS:
        frame = pd.read_csv(claim["result_path"])
        row = frame[(frame["model_name"] == claim["model_name"]) & (frame["period_name"] == claim["period_name"])].iloc[0]
        actual_negative = float(row["target_value"]) < 0
        actual_significant = float(row["target_p_value"]) < 0.05
        actual_supports_herding = actual_negative and actual_significant
        expected_supports_herding = bool(claim["expected_significant_negative"])
        rows.append(
            {
                "claim_id": claim["claim_id"],
                "paper_reference": claim["paper_reference"],
                "model_name": claim["model_name"],
                "period_name": claim["period_name"],
                "target_value": float(row["target_value"]),
                "target_t_stat": float(row["target_t_stat"]),
                "target_p_value": float(row["target_p_value"]),
                "expected_supports_herding": expected_supports_herding,
                "actual_supports_herding": actual_supports_herding,
                "alignment": "일치" if expected_supports_herding == actual_supports_herding else "불일치",
            }
        )
    return rows


def build_period_ranking_frame(daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for frequency, frame in [("daily", daily), ("weekly", weekly)]:
        for model_name in ["standard_csad", "no_intercept_csad", "scsad"]:
            subset = frame[(frame["model_name"] == model_name) & (frame["period_name"].isin(["pre_covid", "covid", "post_covid"]))].copy()
            if subset.empty:
                continue
            subset["abs_target_value"] = subset["target_value"].abs()
            ranking = subset.sort_values("abs_target_value", ascending=False)["period_name"].tolist()
            strongest_period = ranking[0]
            rows.append(
                {
                    "frequency": frequency,
                    "model_name": model_name,
                    "paper_expected_order": "post_covid > pre_covid > covid",
                    "observed_order": " > ".join(ranking),
                    "strongest_period": strongest_period,
                    "matches_paper_top_period": strongest_period == "post_covid",
                }
            )
    return pd.DataFrame(rows)


def build_alignment_report(
    alignment_frame: pd.DataFrame,
    period_ranking_frame: pd.DataFrame,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
) -> str:
    lines = ["# 논문 정렬 점검 보고서", ""]
    lines.append("## 전체 표본 정렬 여부")
    for _, row in alignment_frame.iterrows():
        lines.append(
            f"- {row['claim_id']}: {row['alignment']} | model={row['model_name']} | "
            f"value={row['target_value']:.6f} | t={row['target_t_stat']:.3f} | p={row['target_p_value']:.6g}"
        )
        lines.append(f"  논문 기준: {row['paper_reference']}")
    lines.append("")

    lines.append("## Daily vs Weekly 비교")
    daily_scsad = _pick_row(daily, "scsad", "full_sample")
    weekly_scsad = _pick_row(weekly, "scsad", "full_sample")
    daily_no_intercept = _pick_row(daily, "no_intercept_csad", "full_sample")
    weekly_no_intercept = _pick_row(weekly, "no_intercept_csad", "full_sample")
    lines.extend(
        [
            f"- daily 전체표본 SCSAD: value={daily_scsad['target_value']:.6f}, t={daily_scsad['target_t_stat']:.3f}, p={daily_scsad['target_p_value']:.6g}",
            f"- weekly 전체표본 SCSAD: value={weekly_scsad['target_value']:.6f}, t={weekly_scsad['target_t_stat']:.3f}, p={weekly_scsad['target_p_value']:.6g}",
            f"- daily 전체표본 no-intercept: value={daily_no_intercept['target_value']:.6f}, t={daily_no_intercept['target_t_stat']:.3f}",
            f"- weekly 전체표본 no-intercept: value={weekly_no_intercept['target_value']:.6f}, t={weekly_no_intercept['target_t_stat']:.3f}",
            "- 해석: weekly는 standard와 no-intercept가 더 안정적으로 음수를 유지하고, SCSAD도 방향은 음수라서 daily보다 논문 방향에 조금 더 가깝습니다.",
            "",
        ]
    )

    lines.append("## 하위 구간 강도 순위")
    for _, row in period_ranking_frame.iterrows():
        lines.append(
            f"- {row['frequency']} / {row['model_name']}: 관측 순서 {row['observed_order']} "
            f"| 논문 최상위 구간 일치 여부={bool(row['matches_paper_top_period'])}"
        )
    lines.append("")

    mismatch_count = int((alignment_frame["alignment"] == "불일치").sum())
    lines.append("## 요약")
    if mismatch_count == 0:
        lines.append("- 현재 paper-like 근사 결과는 추적한 모든 주장과 방향성 측면에서 일치합니다.")
    else:
        lines.append(
            f"- 현재 paper-like 근사 결과는 추적한 주장 {len(alignment_frame)}개 중 {int((alignment_frame['alignment'] == '일치').sum())}개와 일치합니다."
        )
        lines.append("- 주요 차이는 Binance OHLCV, 14개 종목, equal-weighted 시장수익률을 사용한다는 점에서 생깁니다. 논문은 더 넓은 유니버스와 market-cap weighted 설정에 가깝습니다.")
    lines.append("")
    return "\n".join(lines)


def _pick_row(frame: pd.DataFrame, model_name: str, period_name: str) -> pd.Series:
    return frame[(frame["model_name"] == model_name) & (frame["period_name"] == period_name)].iloc[0]


if __name__ == "__main__":
    raise SystemExit(main())

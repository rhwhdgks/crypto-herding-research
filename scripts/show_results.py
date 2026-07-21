from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "baseline"


def main() -> None:
    regression = _read_csv("regression_results.csv")
    events = _read_csv("event_count_summary.csv")
    event_study = _read_csv("event_study_summary.csv")
    data_load = _read_csv("data_load_summary.csv")
    universe = _read_csv("universe_coverage_summary.csv")

    print("=== 데이터 소스 ===")
    if not data_load.empty:
        display = data_load[["symbol", "source_used", "rows_loaded"]]
        print(display.to_string(index=False))
    else:
        print("데이터 로드 요약이 없습니다.")

    print("\n=== 패널 ===")
    if not universe.empty:
        row = universe.iloc[0]
        print(f"모드             : {row.get('panel_mode', 'n/a')}")
        print(f"최소 활성 자산 수 : {int(row.get('min_active_assets', 0))}")
        print(f"분석 시작         : {row.get('analysis_candidate_start', 'n/a')}")
        print(f"분석 종료         : {row.get('analysis_candidate_end', 'n/a')}")
    else:
        print("유니버스 커버리지 요약이 없습니다.")

    print("\n=== 회귀 ===")
    if not regression.empty and "term" in regression.columns:
        beta2 = regression[regression["term"] == "market_return_sq"]
        if not beta2.empty:
            row = beta2.iloc[0]
            print(f"beta2        : {row['coefficient']:.6f}")
            print(f"beta2 t-통계량: {row['t_stat']:.3f}")
            print(f"beta2 p-값   : {row['p_value']:.6g}")
        else:
            print("beta2 행을 찾지 못했습니다.")
    else:
        print("회귀 결과가 없습니다.")

    print("\n=== 이벤트 수 ===")
    if not events.empty:
        print(events.to_string(index=False))
    else:
        print("이벤트 집계 결과가 없습니다.")

    print("\n=== 보유기간 상위 결과 ===")
    if not event_study.empty:
        filtered = event_study[event_study["count"] >= 20] if "count" in event_study.columns else event_study
        if filtered.empty:
            filtered = event_study
        inference_columns = [
            column
            for column in [
                "p_value_block",
                "confidence_interval_lower",
                "confidence_interval_upper",
                "n_nonoverlapping_events",
            ]
            if column in filtered.columns
        ]
        best_rows = filtered.sort_values("mean_return", ascending=False)[
            ["event_type", "horizon_label", "count", "mean_return", *inference_columns, "win_rate"]
        ].head(8)
        print(best_rows.to_string(index=False))
    else:
        print("이벤트 스터디 요약이 없습니다.")

    print("\n=== 파일 ===")
    print(f"리포트: {OUTPUTS_DIR / 'report_summary.md'}")
    print(f"회귀 결과: {OUTPUTS_DIR / 'regression_results.csv'}")
    print(f"이벤트 스터디: {OUTPUTS_DIR / 'event_study_summary.csv'}")
    print(f"이벤트 라벨: {OUTPUTS_DIR / 'event_labels.csv'}")


def _read_csv(name: str) -> pd.DataFrame:
    path = OUTPUTS_DIR / name
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    return pd.read_csv(path)


if __name__ == "__main__":
    raise SystemExit(main())

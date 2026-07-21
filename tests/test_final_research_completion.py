import pandas as pd

from final_research_reporting import (
    ALLOWED_HYPOTHESIS_STATUSES,
    build_final_manuscript,
    build_hypothesis_closure_status,
)


def test_hypothesis_closure_table_is_complete_and_unambiguous() -> None:
    status = build_hypothesis_closure_status()
    assert len(status) == 19
    assert status["hypothesis_id"].is_unique
    assert set(status["status"]).issubset(ALLOWED_HYPOTHESIS_STATUSES)
    assert status["evidence_path"].str.len().gt(0).all()
    assert not status["hypothesis"].str.contains("자동매매").any()
    assert set(status.loc[status["status"].eq("requires new external data"), "hypothesis_id"]) == {
        "H17",
        "H18",
        "H19",
    }


def test_final_manuscript_contains_all_interpretation_layers() -> None:
    report = build_final_manuscript(build_hypothesis_closure_status())
    required = [
        "## Abstract",
        "## 초록",
        "Classical CSAD",
        "선행논문 복제",
        "외부타당성",
        "Specification null audit",
        "Factor-adjusted convergence",
        "Tick 의미 감사",
        "Zero-run",
        "통합 결론",
        "가설 종료 현황",
        "tracker",
    ]
    for marker in required:
        assert marker in report
    assert "행동적 herding 식별" in report


def test_status_counts_match_reported_total() -> None:
    status = build_hypothesis_closure_status()
    counts = status["status"].value_counts()
    assert int(counts.sum()) == 19
    assert int(status["closed_with_current_data"].sum()) == 16
    assert isinstance(status, pd.DataFrame)

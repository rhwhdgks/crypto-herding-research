from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs/v2/final_research_completion_v1"
CONFIG_PATH = PROJECT_ROOT / "configs/research/zero_run_mechanical_null_audit_v1.yaml"
PROTOCOL_PATH = PROJECT_ROOT / "research_protocols/zero_run_mechanical_null_audit_v1.md"
SEAL_PATH = PROJECT_ROOT / "research_protocols/zero_run_mechanical_null_audit_v1.seal.json"

REQUIRED_FILES = {
    "REPRODUCIBILITY.md",
    "artifact_manifest.csv",
    "empirical_null_group_comparison.csv",
    "final_research_manuscript.md",
    "final_run_summary.json",
    "final_verification_summary.json",
    "hypothesis_closure_status.csv",
    "hypothesis_status.csv",
    "input_integrity.csv",
    "mechanical_null_audit_report.md",
    "mechanical_null_config_snapshot.yaml",
    "mechanical_null_decisions.csv",
    "mechanical_null_input_manifest.json",
    "mechanical_null_protocol_snapshot.md",
    "mechanical_null_provenance.json",
    "mechanical_null_run_summary.json",
    "mechanism_model_diagnostics.csv",
    "mechanism_null_coefficients.parquet",
    "mechanism_null_comparison.csv",
    "null_group_replicates.parquet",
    "null_sampling_diagnostics.json",
    "plots/conditional_null_fpr.png",
    "plots/empirical_vs_conditional_null.png",
    "plots/hypothesis_closure_map.png",
    "plots/mechanism_vs_conditional_null.png",
    "preregistration_seal_verification.json",
    "requirement_audit.csv",
    "research_axis_summary.csv",
    "research_evidence_manifest.csv",
    "sampling_strata_summary.csv",
    "strata_boundaries.csv",
    "stratified_sample_keys.csv",
}
ALLOWED_STATUSES = {
    "supported",
    "partially supported",
    "descriptive only",
    "falsified",
    "methodologically invalid",
    "requires new external data",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benjamini_hochberg(values: pd.Series) -> np.ndarray:
    raw = values.to_numpy(dtype=float)
    order = np.argsort(raw)
    ranked = raw[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def verify_exact_distribution_identity() -> None:
    n, k = 8, 3
    counts: dict[int, int] = {}
    for positions in combinations(range(n), k):
        bits = [int(index in positions) for index in range(n)]
        runs = sum(value == 1 and (index == 0 or bits[index - 1] == 0) for index, value in enumerate(bits))
        counts[runs] = counts.get(runs, 0) + 1
    denominator = math.comb(n, k)
    for runs, count in counts.items():
        expected = math.comb(k - 1, runs - 1) * math.comb(n - k + 1, runs) / denominator
        assert expected == count / denominator
    assert sum(counts.values()) == denominator


def verify_artifact_manifest() -> None:
    manifest = pd.read_csv(OUTPUT_DIR / "artifact_manifest.csv")
    actual = {
        path.relative_to(OUTPUT_DIR).as_posix()
        for path in OUTPUT_DIR.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.csv"
    }
    assert set(manifest["path"]) == actual
    assert len(manifest) == len(actual)
    for row in manifest.itertuples(index=False):
        path = OUTPUT_DIR / row.path
        assert path.stat().st_size == int(row.size), row.path
        assert sha256(path) == row.sha256, row.path


def verify_preregistration() -> None:
    seal = json.loads(SEAL_PATH.read_text(encoding="utf-8"))
    assert seal["results_observed_at_seal"] is False
    assert sha256(PROTOCOL_PATH) == seal["protocol_sha256"]
    assert sha256(CONFIG_PATH) == seal["config_sha256"]
    verification = json.loads(
        (OUTPUT_DIR / "preregistration_seal_verification.json").read_text(encoding="utf-8")
    )
    assert verification["verified"] is True
    assert verification["protocol_sha256"] == seal["protocol_sha256"]
    assert verification["config_sha256"] == seal["config_sha256"]
    assert (OUTPUT_DIR / "mechanical_null_protocol_snapshot.md").read_bytes() == PROTOCOL_PATH.read_bytes()


def verify_input_and_evidence_manifests() -> None:
    inputs = json.loads(
        (OUTPUT_DIR / "mechanical_null_input_manifest.json").read_text(encoding="utf-8")
    )["files"]
    assert len(inputs) == 7
    for entry in inputs:
        path = PROJECT_ROOT / entry["path"]
        assert path.is_file(), entry["path"]
        assert path.stat().st_size == int(entry["size"]), entry["path"]
        assert sha256(path) == entry["sha256"], entry["path"]

    evidence = pd.read_csv(OUTPUT_DIR / "research_evidence_manifest.csv")
    assert len(evidence) >= 30
    assert evidence["path"].is_unique
    for row in evidence.itertuples(index=False):
        path = PROJECT_ROOT / row.path
        assert path.is_file(), row.path
        assert path.stat().st_size == int(row.size), row.path
        assert sha256(path) == row.sha256, row.path


def upper_p(values: np.ndarray, observed: float) -> float:
    return float((1 + np.sum(values >= observed)) / (len(values) + 1))


def two_sided_p(values: np.ndarray, observed: float) -> float:
    lower = (1 + np.sum(values <= observed)) / (len(values) + 1)
    upper = (1 + np.sum(values >= observed)) / (len(values) + 1)
    return float(min(1.0, 2 * min(lower, upper)))


def verify_mechanical_null_tables() -> None:
    integrity = pd.read_csv(OUTPUT_DIR / "input_integrity.csv")
    assert len(integrity) == 1
    assert int(integrity.loc[0, "raw_rows"]) == 490_560
    assert int(integrity.loc[0, "oos_eligible_rows"]) == 233_201
    assert bool(integrity.loc[0, "run_z_exact_match"])
    assert float(integrity.loc[0, "maximum_recomputed_run_z_difference"]) <= 1e-12

    sample = pd.read_csv(OUTPUT_DIR / "stratified_sample_keys.csv")
    strata = pd.read_csv(OUTPUT_DIR / "sampling_strata_summary.csv")
    diagnostics = json.loads((OUTPUT_DIR / "null_sampling_diagnostics.json").read_text())
    assert len(sample) == 9_044
    assert len(strata) == 141
    verify_close(float(sample["sampling_weight"].sum()), 233_201.0, 1e-6)
    assert diagnostics["rows"] == 9_044
    assert diagnostics["repetitions"] == 999
    assert diagnostics["maximum_pmf_sum_error"] <= 1e-12

    groups = pd.read_csv(OUTPUT_DIR / "empirical_null_group_comparison.csv")
    replicates = pd.read_parquet(OUTPUT_DIR / "null_group_replicates.parquet")
    assert len(groups) == 23
    assert len(replicates) == 23 * 999
    assert replicates.groupby(["dimension", "group"]).size().eq(999).all()
    for row in groups.itertuples(index=False):
        local = replicates.loc[
            replicates["dimension"].eq(row.dimension) & replicates["group"].eq(row.group)
        ]
        intensity = local["null_mean_intensity"].to_numpy(dtype=float)
        fpr = local["null_fpr"].to_numpy(dtype=float)
        verify_close(row.null_mean_intensity, float(intensity.mean()), 1e-12)
        verify_close(row.null_fpr_mean, float(fpr.mean()), 1e-12)
        verify_close(
            row.intensity_empirical_p_value,
            upper_p(intensity, row.empirical_mean_intensity),
            1e-12,
        )
        verify_close(
            row.tail_empirical_p_value,
            upper_p(fpr, row.empirical_clustering_share),
            1e-12,
        )
    assert np.allclose(
        groups["intensity_q_value_bh_fdr_23"],
        benjamini_hochberg(groups["intensity_empirical_p_value"]),
        atol=1e-12,
    )
    assert np.allclose(
        groups["tail_q_value_bh_fdr_23"],
        benjamini_hochberg(groups["tail_empirical_p_value"]),
        atol=1e-12,
    )

    mechanism = pd.read_csv(OUTPUT_DIR / "mechanism_null_comparison.csv")
    mechanism_draws = pd.read_parquet(OUTPUT_DIR / "mechanism_null_coefficients.parquet")
    assert len(mechanism) == 5
    assert len(mechanism_draws) == 5 * 999
    for row in mechanism.itertuples(index=False):
        values = mechanism_draws.loc[
            mechanism_draws["outcome"].eq(row.outcome), "null_coefficient"
        ].to_numpy(dtype=float)
        assert len(values) == 999
        verify_close(
            row.empirical_p_value,
            two_sided_p(values, row.audit_sample_coefficient),
            1e-12,
        )
    assert np.allclose(
        mechanism["q_value_bh_fdr_5"],
        benjamini_hochberg(mechanism["empirical_p_value"]),
        atol=1e-12,
    )
    assert not mechanism["exceeds_count_conditioned_null"].any()

    decisions = pd.read_csv(OUTPUT_DIR / "mechanical_null_decisions.csv")
    assert len(decisions) == 4
    expected = {
        "conditional_cutoff_calibration": True,
        "empirical_excess_clustering": True,
        "mechanism_beyond_conditional_null": False,
        "final_mechanical_null_audit": False,
    }
    assert decisions.set_index("decision")["passed"].to_dict() == expected
    final = decisions.loc[decisions["decision"].eq("final_mechanical_null_audit")].iloc[0]
    assert final["classification"] == "clustering_beyond_counts_but_mechanism_not_distinct"


def verify_close(actual: float, expected: float, tolerance: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{actual} != {expected} within {tolerance}")


def verify_hypotheses_and_report() -> None:
    status = pd.read_csv(OUTPUT_DIR / "hypothesis_closure_status.csv")
    status_alias = pd.read_csv(OUTPUT_DIR / "hypothesis_status.csv")
    pd.testing.assert_frame_equal(status, status_alias)
    assert len(status) == 19
    assert status["hypothesis_id"].is_unique
    assert set(status["status"]).issubset(ALLOWED_STATUSES)
    assert int(status["closed_with_current_data"].sum()) == 16
    assert set(status.loc[status["status"].eq("requires new external data"), "hypothesis_id"]) == {
        "H17",
        "H18",
        "H19",
    }
    for path in status["evidence_path"]:
        assert (PROJECT_ROOT / path).is_file(), path

    report = (OUTPUT_DIR / "final_research_manuscript.md").read_text(encoding="utf-8")
    required_markers = [
        "# From Replication to Falsification",
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
        "재현 자료",
    ]
    for marker in required_markers:
        assert marker in report, marker
    assert "alpha 발견이 아니라" in report
    assert "자동매매, tracker, 방향성 alpha는 연구 범위에서 제외" in report

    axes = pd.read_csv(OUTPUT_DIR / "research_axis_summary.csv")
    assert len(axes) == 11
    requirements = pd.read_csv(OUTPUT_DIR / "requirement_audit.csv")
    assert len(requirements) == 8
    implemented = requirements.loc[
        requirements["requirement"].ne("directional_alpha_or_tracker"), "implemented"
    ]
    assert implemented.all()
    assert not bool(
        requirements.loc[
            requirements["requirement"].eq("directional_alpha_or_tracker"), "implemented"
        ].iloc[0]
    )

    summary = json.loads((OUTPUT_DIR / "final_run_summary.json").read_text())
    assert summary["hypotheses"] == 19
    assert summary["closed_with_current_data"] == 16
    assert summary["behavioral_herding_identified"] is False
    assert summary["short_horizon_predictive_family_passed"] is False
    assert summary["tracker_activation_allowed"] is False
    assert summary["verifier_mode"] == "read_only"
    verification = json.loads(
        (OUTPUT_DIR / "final_verification_summary.json").read_text()
    )
    assert verification["status"] == "pass"
    assert verification["group_bh_family_sizes"] == [23, 23]
    assert verification["mechanism_bh_family_size"] == 5
    assert verification["mechanical_null_decision_recomputed"] is True
    assert verification["hypothesis_rows"] == 19
    assert verification["directional_alpha_active"] is False
    assert verification["tracker_active"] is False


def main() -> None:
    missing = sorted(value for value in REQUIRED_FILES if not (OUTPUT_DIR / value).is_file())
    if missing:
        raise AssertionError(f"Missing final research artifacts: {', '.join(missing)}")
    verify_exact_distribution_identity()
    verify_artifact_manifest()
    verify_preregistration()
    verify_input_and_evidence_manifests()
    verify_mechanical_null_tables()
    verify_hypotheses_and_report()
    print(
        "verified final research completion v1: 19 hypotheses, 11 research axes, "
        "23x999 group null draws, 5x999 mechanism null draws, read-only verification"
    )


if __name__ == "__main__":
    main()

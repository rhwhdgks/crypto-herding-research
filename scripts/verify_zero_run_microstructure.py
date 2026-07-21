from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs/v2/zero_run_microstructure_v1"
CONFIG_PATH = PROJECT_ROOT / "configs/research/zero_run_microstructure_v1.yaml"
PROTOCOL_PATH = PROJECT_ROOT / "research_protocols/zero_run_microstructure_v1.md"


REQUIRED_FILES = {
    "analysis_frame.parquet",
    "tick_input_coverage.csv",
    "ohlcv_input_quality.csv",
    "sample_coverage.csv",
    "clustered_coefficients.csv",
    "model_diagnostics.csv",
    "oos_prediction_metrics.csv",
    "loao_coefficients.csv",
    "loao_summary.csv",
    "permutation_draws.csv",
    "permutation_summary.csv",
    "future_lead_placebo.csv",
    "mechanism_decisions.csv",
    "future_decisions.csv",
    "family_decisions.csv",
    "scaling_artifact.json",
    "zero_run_microstructure_report.md",
    "config_snapshot.yaml",
    "protocol_snapshot.md",
    "input_manifest.json",
    "provenance.json",
    "run_summary.json",
    "model_specification.json",
    "artifact_manifest.csv",
    "plots/mechanism_coefficients.png",
    "plots/future_magnitude_coefficients.png",
    "plots/oos_prediction_improvement.png",
    "plots/permutation_null_comparison.png",
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


def verify_manifest() -> None:
    manifest = pd.read_csv(OUTPUT_DIR / "artifact_manifest.csv")
    assert len(manifest) == len(REQUIRED_FILES) - 1
    assert set(manifest["path"]) == REQUIRED_FILES - {"artifact_manifest.csv"}
    for row in manifest.itertuples(index=False):
        path = OUTPUT_DIR / row.path
        assert path.is_file(), row.path
        assert path.stat().st_size == int(row.size), row.path
        assert sha256(path) == row.sha256, row.path


def verify_inputs() -> None:
    current_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    snapshot_config = yaml.safe_load(
        (OUTPUT_DIR / "config_snapshot.yaml").read_text(encoding="utf-8")
    )
    assert current_config == snapshot_config
    assert (OUTPUT_DIR / "protocol_snapshot.md").read_bytes() == PROTOCOL_PATH.read_bytes()
    manifest = json.loads((OUTPUT_DIR / "input_manifest.json").read_text())
    assert len(manifest["files"]) == 10
    for entry in manifest["files"]:
        path = PROJECT_ROOT / entry["path"]
        assert path.is_file(), entry["path"]
        assert path.stat().st_size == int(entry["size"]), entry["path"]
        assert sha256(path) == entry["sha256"], entry["path"]


def verify_tables() -> None:
    frame = pd.read_parquet(
        OUTPUT_DIR / "analysis_frame.parquet",
        columns=[
            "symbol",
            "bucket_start",
            "signal_timestamp",
            "sample_split",
            "zero_run_intensity",
            "abs_forward_return_5m_bps",
            "abs_forward_return_15m_bps",
            "abs_forward_return_30m_bps",
            "realized_volatility_5m_bps",
            "realized_volatility_15m_bps",
            "realized_volatility_30m_bps",
        ],
    )
    assert len(frame) == 490_560
    assert frame["symbol"].nunique() == 7
    assert not frame.duplicated(["symbol", "bucket_start"]).any()
    assert set(frame["sample_split"]) == {"development", "oos"}
    assert frame["zero_run_intensity"].notna().all()
    assert not any("signed_forward" in column for column in frame.columns)

    tick = pd.read_csv(OUTPUT_DIR / "tick_input_coverage.csv")
    assert len(tick) == 7
    assert tick["complete_15m_grid"].all()
    assert tick["rows"].eq(70_080).all()
    ohlcv = pd.read_csv(OUTPUT_DIR / "ohlcv_input_quality.csv")
    assert len(ohlcv) == 7
    assert ohlcv["all_horizons_exact_share"].min() > 0.999

    coverage = pd.read_csv(OUTPUT_DIR / "sample_coverage.csv")
    assert coverage["rows"].tolist() == [245_280, 245_280]
    assert coverage["symbols"].eq(7).all()
    assert coverage["utc_days"].ge(365).all()

    coefficients = pd.read_csv(OUTPUT_DIR / "clustered_coefficients.csv")
    assert len(coefficients) == 22
    expected_sizes = {
        ("development", "mechanism"): 5,
        ("development", "future_absolute_return"): 3,
        ("development", "future_realized_volatility"): 3,
        ("oos", "mechanism"): 5,
        ("oos", "future_absolute_return"): 3,
        ("oos", "future_realized_volatility"): 3,
    }
    assert coefficients.groupby(["split", "family"]).size().to_dict() == expected_sizes
    assert coefficients["q_value_bh_fdr"].between(0, 1).all()
    for _, subset in coefficients.groupby(["split", "family"], sort=False):
        expected_q = benjamini_hochberg(subset["cluster_p_value"])
        assert np.allclose(subset["q_value_bh_fdr"], expected_q, atol=1e-12)

    loao = pd.read_csv(OUTPUT_DIR / "loao_coefficients.csv")
    loao_summary = pd.read_csv(OUTPUT_DIR / "loao_summary.csv")
    assert len(loao) == 42
    assert len(loao_summary) == 6
    assert loao.groupby(["family", "horizon_minutes"]).size().eq(7).all()
    assert loao["q_value_bh_fdr_21"].between(0, 1).all()

    permutation = pd.read_csv(OUTPUT_DIR / "permutation_draws.csv")
    permutation_summary = pd.read_csv(OUTPUT_DIR / "permutation_summary.csv")
    assert len(permutation) == 6 * 199
    assert len(permutation_summary) == 6
    assert permutation.groupby(["family", "horizon_minutes"]).size().eq(199).all()
    assert permutation["shift_days"].ge(7).all()
    assert permutation_summary["q_value_bh_fdr"].between(0, 1).all()
    for _, subset in permutation_summary.groupby("family", sort=False):
        expected_q = benjamini_hochberg(subset["empirical_p_value"])
        assert np.allclose(subset["q_value_bh_fdr"], expected_q, atol=1e-12)

    placebo = pd.read_csv(OUTPUT_DIR / "future_lead_placebo.csv")
    mechanism = pd.read_csv(OUTPUT_DIR / "mechanism_decisions.csv")
    future = pd.read_csv(OUTPUT_DIR / "future_decisions.csv")
    families = pd.read_csv(OUTPUT_DIR / "family_decisions.csv")
    assert len(placebo) == 6
    for _, subset in placebo.groupby("family", sort=False):
        expected_q = benjamini_hochberg(subset["cluster_p_value"])
        assert np.allclose(subset["q_value_bh_fdr"], expected_q, atol=1e-12)
    assert len(mechanism) == 5
    assert len(future) == 6
    assert len(families) == 3
    assert set(future["horizon_minutes"]) == {5, 15, 30}
    required_gates = [
        "passes_cluster_q",
        "passes_permutation_q",
        "passes_effect_size",
        "passes_loao",
        "passes_oos_prediction",
    ]
    recomputed = future[required_gates].all(axis=1) & ~future["placebo_veto"]
    assert recomputed.equals(future["passes_predeclared_criteria"])
    for row in families.loc[families["family"].ne("mechanism")].itertuples(index=False):
        subset = future.loc[future["family"].eq(row.family)]
        expected = bool(
            subset.loc[
                subset["horizon_minutes"].eq(15), "passes_predeclared_criteria"
            ].iloc[0]
            and subset["passes_predeclared_criteria"].sum() >= 2
        )
        assert bool(row.family_success) == expected

    summary = json.loads((OUTPUT_DIR / "run_summary.json").read_text())
    assert summary["rows"] == 490_560
    assert summary["symbols"] == 7
    assert summary["directional_alpha_tested"] is False
    assert summary["tracker_activation_allowed"] is False
    specification = json.loads(
        (OUTPUT_DIR / "model_specification.json").read_text()
    )
    assert specification["fixed_effects"] == ["symbol", "hour_utc", "weekday_utc"]
    assert specification["covariance"].startswith("UTC-day clustered")
    assert specification["signed_future_return_outcome"] is None
    assert set(specification["future_outcome_families"]) == {
        "absolute_return_bps",
        "realized_volatility_bps",
    }
    report = (OUTPUT_DIR / "zero_run_microstructure_report.md").read_text(
        encoding="utf-8"
    )
    assert "방향성 수익률, 거래비용 후 수익, 매수·매도 규칙은 분석하지 않았습니다" in report
    assert "tracker, paper-sim, 자동매매는 계속 비활성" in report


def main() -> None:
    missing = sorted(
        value for value in REQUIRED_FILES if not (OUTPUT_DIR / value).is_file()
    )
    if missing:
        raise AssertionError(f"Missing zero-run artifacts: {', '.join(missing)}")
    verify_manifest()
    verify_inputs()
    verify_tables()
    print(
        "verified zero-run microstructure v1: "
        "490,560 rows, 7 assets, 22 clustered models, 42 LOAO models, "
        "1,194 permutation draws, 6 placebo models"
    )


if __name__ == "__main__":
    main()

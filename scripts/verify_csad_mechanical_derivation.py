from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import load_config  # noqa: E402


CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "research" / "csad_mechanical_derivation_v1.yaml"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v2" / "csad_mechanical_derivation_v1"
MASTER_REPORT = PROJECT_ROOT / "outputs" / "research_master_report_2026-07-21.md"


def main() -> None:
    config = load_config(CONFIG_PATH)
    required = [
        "protocol_snapshot.md",
        "config_snapshot.yaml",
        "provenance.json",
        "input_manifest.csv",
        "code_manifest.csv",
        "artifact_manifest.csv",
        "theory_identities.csv",
        "gaussian_theory_coefficients.csv",
        "equation_verification.csv",
        "convergence_replicates.parquet",
        "convergence_diagnostics.parquet",
        "convergence_summary.csv",
        "convergence_gates.csv",
        "robustness_replicates.parquet",
        "robustness_diagnostics.parquet",
        "robustness_summary.csv",
        "robustness_diagnostic_summary.csv",
        "symmetric_robustness_cells.csv",
        "symmetric_robustness_decision.csv",
        "final_mechanical_decision.csv",
        "csad_mechanical_derivation_report.md",
        "plots/gaussian_theory_convergence.png",
        "plots/nonherding_false_positive_contrast.png",
        "plots/mechanical_negative_sign_robustness.png",
        "plots/projection_mechanism.png",
    ]
    missing = [relative for relative in required if not (OUTPUT_DIR / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing mechanical-derivation artifacts: {missing}")
    if not MASTER_REPORT.is_file():
        raise FileNotFoundError(f"Updated master report is missing: {MASTER_REPORT}")

    provenance = json.loads((OUTPUT_DIR / "provenance.json").read_text(encoding="utf-8"))
    protocol_path = PROJECT_ROOT / str(config["protocol"]["path"])
    _assert_equal(provenance["config_sha256"], _sha256(CONFIG_PATH), "config hash")
    _assert_equal(provenance["protocol_sha256"], _sha256(protocol_path), "protocol hash")
    _assert_equal(_sha256(OUTPUT_DIR / "config_snapshot.yaml"), _sha256(CONFIG_PATH), "config snapshot")
    _assert_equal(
        _sha256(OUTPUT_DIR / "protocol_snapshot.md"),
        _sha256(protocol_path),
        "protocol snapshot",
    )
    _verify_manifest(OUTPUT_DIR / "input_manifest.csv", PROJECT_ROOT)
    _verify_manifest(OUTPUT_DIR / "code_manifest.csv", PROJECT_ROOT)
    _verify_manifest(OUTPUT_DIR / "artifact_manifest.csv", OUTPUT_DIR)

    model_count = len(config["models"]["names"])
    convergence_repetitions = int(config["convergence_simulation"]["repetitions"])
    convergence_assets = len(config["convergence_simulation"]["assets"])
    convergence = pd.read_parquet(OUTPUT_DIR / "convergence_replicates.parquet")
    expected_convergence = convergence_repetitions * convergence_assets * model_count
    _assert_equal(len(convergence), expected_convergence, "convergence replicate rows")
    _verify_replicate_frame(
        convergence,
        expected_cells=convergence_assets * model_count,
        repetitions=convergence_repetitions,
    )
    convergence_diagnostics = pd.read_parquet(
        OUTPUT_DIR / "convergence_diagnostics.parquet"
    )
    _assert_equal(
        len(convergence_diagnostics),
        convergence_repetitions * convergence_assets,
        "convergence diagnostic rows",
    )

    robustness_repetitions = int(config["robustness_simulation"]["repetitions"])
    dgp_count = len(config["dgp"])
    scenario_count = len(config["robustness_simulation"]["scenarios"])
    robustness = pd.read_parquet(OUTPUT_DIR / "robustness_replicates.parquet")
    expected_robustness = robustness_repetitions * dgp_count * scenario_count * model_count
    _assert_equal(len(robustness), expected_robustness, "robustness replicate rows")
    _verify_replicate_frame(
        robustness,
        expected_cells=dgp_count * scenario_count * model_count,
        repetitions=robustness_repetitions,
    )
    _assert_equal(robustness["dgp"].nunique(), dgp_count, "robustness DGP count")
    _assert_equal(
        robustness["scenario"].nunique(), scenario_count, "robustness scenario count"
    )
    robustness_diagnostics = pd.read_parquet(
        OUTPUT_DIR / "robustness_diagnostics.parquet"
    )
    _assert_equal(
        len(robustness_diagnostics),
        robustness_repetitions * dgp_count * scenario_count,
        "robustness diagnostic rows",
    )

    theory = pd.read_csv(OUTPUT_DIR / "gaussian_theory_coefficients.csv")
    _assert_equal(len(theory), convergence_assets * model_count, "theory coefficient rows")
    equations = pd.read_csv(OUTPUT_DIR / "equation_verification.csv")
    _assert_equal(len(equations), convergence_assets, "equation rows")
    if not equations["equation_gate_pass"].astype(bool).all():
        raise ValueError("At least one closed-form equation gate failed")
    if not equations["maximum_absolute_difference"].le(
        float(config["theory"]["equation_tolerance"])
    ).all():
        raise ValueError("Closed-form and moment solutions exceed the frozen tolerance")

    convergence_summary = pd.read_csv(OUTPUT_DIR / "convergence_summary.csv")
    convergence_gates = pd.read_csv(OUTPUT_DIR / "convergence_gates.csv")
    _assert_equal(len(convergence_summary), convergence_assets * model_count, "convergence summary rows")
    _assert_equal(len(convergence_gates), convergence_assets * model_count, "convergence gate rows")
    if not convergence_gates["gate_pass"].astype(bool).all():
        failed = convergence_gates.loc[~convergence_gates["gate_pass"].astype(bool)]
        raise ValueError(f"Preregistered convergence gates failed: {failed.to_dict('records')}")

    robustness_summary = pd.read_csv(OUTPUT_DIR / "robustness_summary.csv")
    _assert_equal(
        len(robustness_summary), dgp_count * scenario_count * model_count, "robustness summary rows"
    )
    symmetric_cells = pd.read_csv(OUTPUT_DIR / "symmetric_robustness_cells.csv")
    expected_symmetric_cells = (
        len(config["robustness_simulation"]["symmetric_dgps"]) * scenario_count
    )
    _assert_equal(len(symmetric_cells), expected_symmetric_cells, "symmetric robustness cells")
    symmetric_decision = pd.read_csv(OUTPUT_DIR / "symmetric_robustness_decision.csv")
    _assert_equal(len(symmetric_decision), 1, "symmetric decision rows")

    final_decision = pd.read_csv(OUTPUT_DIR / "final_mechanical_decision.csv")
    _assert_equal(len(final_decision), 1, "final decision rows")
    if not bool(final_decision.iloc[0]["mechanical_null_confirmed"]):
        raise ValueError("Final preregistered mechanical-null decision is not confirmed")
    _assert_equal(
        str(final_decision.iloc[0]["classification"]),
        "mechanical_null_confirmed",
        "final classification",
    )

    report = (OUTPUT_DIR / "csad_mechanical_derivation_report.md").read_text(
        encoding="utf-8"
    )
    for phrase in (
        "No-intercept delta2*",
        "SCSAD gamma3*",
        "mechanical_null_confirmed",
        "후속 실증연구의 식별 기준",
    ):
        if phrase not in report:
            raise ValueError(f"Required report content is missing: {phrase}")
    master = MASTER_REPORT.read_text(encoding="utf-8")
    if "## 7-1. 음의 계수가 생기는 수학적 원인" not in master:
        raise ValueError("The latest master report lacks the mechanical-derivation section")

    result = {
        "status": "verified",
        "protocol_sha256": provenance["protocol_sha256"],
        "config_sha256": provenance["config_sha256"],
        "artifact_manifest_rows": len(
            pd.read_csv(OUTPUT_DIR / "artifact_manifest.csv")
        ),
        "convergence_replicate_rows": len(convergence),
        "robustness_replicate_rows": len(robustness),
        "equation_gates": f"{int(equations['equation_gate_pass'].sum())}/{len(equations)}",
        "convergence_gates": f"{int(convergence_gates['gate_pass'].sum())}/{len(convergence_gates)}",
        "final_classification": final_decision.iloc[0]["classification"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _verify_replicate_frame(
    frame: pd.DataFrame,
    expected_cells: int,
    repetitions: int,
) -> None:
    required_numeric = [
        "target_coefficient",
        "target_std_error",
        "target_p_value",
        "q_value_bh4",
    ]
    if frame[required_numeric].isna().any().any():
        raise ValueError("Simulation estimates contain missing required numeric values")
    for column in ("target_p_value", "q_value_bh4"):
        if not frame[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} values are outside [0, 1]")
    cells = frame.groupby(["dgp", "scenario", "model"])["repetition"].nunique()
    _assert_equal(len(cells), expected_cells, "simulation cell count")
    if not cells.eq(repetitions).all():
        raise ValueError("At least one simulation cell lacks preregistered repetitions")


def _verify_manifest(manifest_path: Path, base_dir: Path) -> None:
    manifest = pd.read_csv(manifest_path)
    if manifest["path"].duplicated().any():
        raise ValueError(f"Duplicate manifest paths: {manifest_path}")
    for row in manifest.itertuples(index=False):
        path = base_dir / str(row.path)
        if not path.is_file():
            raise FileNotFoundError(f"Manifest file missing: {path}")
        _assert_equal(path.stat().st_size, int(row.size_bytes), f"size for {row.path}")
        _assert_equal(_sha256(path), str(row.sha256), f"hash for {row.path}")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_equal(observed, expected, label: str) -> None:
    if observed != expected:
        raise ValueError(f"{label} mismatch: observed={observed!r}, expected={expected!r}")


if __name__ == "__main__":
    main()


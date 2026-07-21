from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import load_config  # noqa: E402


CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "research"
    / "csad_mechanical_convergence_supplement_v1_1.yaml"
)
PARENT_DIR = PROJECT_ROOT / "outputs" / "v2" / "csad_mechanical_derivation_v1"
OUTPUT_DIR = PARENT_DIR / "supplement_v1_1"
MASTER_REPORT = PROJECT_ROOT / "outputs" / "research_master_report_2026-07-21.md"
MASTER_REPORT_BACKUP = (
    PROJECT_ROOT / "outputs" / "research_master_report_2026-07-21_pre_supplement.md"
)


def main() -> None:
    config = load_config(CONFIG_PATH)
    required = [
        "protocol_snapshot.md",
        "config_snapshot.yaml",
        "input_manifest.csv",
        "code_manifest.csv",
        "artifact_manifest.csv",
        "provenance.json",
        "supplement_replicates.parquet",
        "supplement_diagnostics.parquet",
        "supplement_summary.csv",
        "supplement_diagnostic_summary.csv",
        "supplement_gates.csv",
        "supplement_decision.csv",
        "csad_mechanical_convergence_supplement_report.md",
        "master_report_update.md",
        "plots/finite_sample_convergence_ladder.png",
    ]
    missing = [relative for relative in required if not (OUTPUT_DIR / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing supplement artifacts: {missing}")
    if not MASTER_REPORT.is_file() or not MASTER_REPORT_BACKUP.is_file():
        raise FileNotFoundError("Current and preserved pre-supplement master reports are required")

    provenance = json.loads((OUTPUT_DIR / "provenance.json").read_text(encoding="utf-8"))
    protocol_path = PROJECT_ROOT / str(config["protocol"]["path"])
    _assert_equal(provenance["protocol_sha256"], _sha256(protocol_path), "protocol hash")
    _assert_equal(provenance["config_sha256"], _sha256(CONFIG_PATH), "config hash")
    _assert_equal(_sha256(OUTPUT_DIR / "protocol_snapshot.md"), _sha256(protocol_path), "protocol snapshot")
    _assert_equal(_sha256(OUTPUT_DIR / "config_snapshot.yaml"), _sha256(CONFIG_PATH), "config snapshot")
    _verify_manifest(OUTPUT_DIR / "input_manifest.csv", PROJECT_ROOT)
    _verify_manifest(OUTPUT_DIR / "code_manifest.csv", PROJECT_ROOT)
    _verify_manifest(OUTPUT_DIR / "artifact_manifest.csv", OUTPUT_DIR)

    parent_decision = pd.read_csv(PARENT_DIR / "final_mechanical_decision.csv")
    _assert_equal(
        str(parent_decision.iloc[0]["classification"]),
        "mechanical_null_not_confirmed",
        "preserved parent classification",
    )
    _assert_equal(
        int(parent_decision.iloc[0]["mechanical_convergence_cells_passed"]),
        5,
        "preserved parent convergence passes",
    )
    parent_equations = pd.read_csv(PARENT_DIR / "equation_verification.csv")
    if not parent_equations["equation_gate_pass"].astype(bool).all():
        raise ValueError("Parent analytic equation verification is not 3/3")

    repetitions = int(config["simulation"]["repetitions"])
    scenario_count = len(config["simulation"]["assets"]) * len(
        config["simulation"]["observations"]
    )
    model_count = len(config["models"]["names"])
    replicates = pd.read_parquet(OUTPUT_DIR / "supplement_replicates.parquet")
    expected_rows = repetitions * scenario_count * model_count
    _assert_equal(len(replicates), expected_rows, "supplement replicate rows")
    cells = replicates.groupby(["scenario", "model"])["repetition"].nunique()
    _assert_equal(len(cells), scenario_count * model_count, "supplement cell count")
    if not cells.eq(repetitions).all():
        raise ValueError("A supplement cell lacks preregistered repetitions")
    if replicates[["target_coefficient", "target_p_value", "q_value_bh4"]].isna().any().any():
        raise ValueError("Supplement replicate estimates contain missing values")
    diagnostics = pd.read_parquet(OUTPUT_DIR / "supplement_diagnostics.parquet")
    _assert_equal(len(diagnostics), repetitions * scenario_count, "supplement diagnostic rows")

    summary = pd.read_csv(OUTPUT_DIR / "supplement_summary.csv")
    gates = pd.read_csv(OUTPUT_DIR / "supplement_gates.csv")
    _assert_equal(len(summary), scenario_count * model_count, "supplement summary rows")
    _assert_equal(len(gates), scenario_count * model_count, "supplement gate rows")
    primary = gates.loc[gates["primary_supplement_cell"].astype(bool)]
    _assert_equal(len(primary), len(config["simulation"]["assets"]) * model_count, "primary gate rows")
    if not primary["gate_pass"].astype(bool).all():
        failed = primary.loc[~primary["gate_pass"].astype(bool)]
        raise ValueError(f"Preregistered supplement gates failed: {failed.to_dict('records')}")

    decision = pd.read_csv(OUTPUT_DIR / "supplement_decision.csv")
    _assert_equal(len(decision), 1, "supplement decision rows")
    row = decision.iloc[0]
    if not bool(row["parent_classification_preserved"]):
        raise ValueError("The parent preregistered failure was not preserved")
    if not bool(row["analytic_mechanism_established"]):
        raise ValueError("The exact analytic equations are not established")
    if not bool(row["finite_sample_convergence_supported"]):
        raise ValueError("The frozen finite-sample supplement did not pass")
    _assert_equal(
        str(row["supplement_classification"]),
        "finite_sample_convergence_supported",
        "supplement classification",
    )

    _assert_equal(
        _sha256(MASTER_REPORT),
        provenance["master_report_update_sha256"],
        "published master report hash",
    )
    _assert_equal(
        _sha256(OUTPUT_DIR / "master_report_update.md"),
        provenance["master_report_update_sha256"],
        "staged master report hash",
    )
    master = MASTER_REPORT.read_text(encoding="utf-8")
    if "### v1.1 유한표본 수렴 보충" not in master:
        raise ValueError("The latest master report lacks the v1.1 supplement")
    report = (OUTPUT_DIR / "csad_mechanical_convergence_supplement_report.md").read_text(
        encoding="utf-8"
    )
    for phrase in (
        "원 판정 `mechanical_null_not_confirmed`은 그대로 보존",
        "finite_sample_convergence_supported",
        "intentional herding의 충분한 증거가 아닙니다",
    ):
        if phrase not in report:
            raise ValueError(f"Required supplement report content is missing: {phrase}")

    result = {
        "status": "verified",
        "parent_classification": row["parent_preregistered_classification"],
        "parent_convergence_gate": (
            f"{int(row['parent_mechanical_convergence_passed'])}/"
            f"{int(row['parent_mechanical_convergence_total'])}"
        ),
        "analytic_equation_gate": (
            f"{int(row['analytic_equation_cells_passed'])}/"
            f"{int(row['analytic_equation_cells_total'])}"
        ),
        "supplement_primary_gates": f"{int(primary['gate_pass'].sum())}/{len(primary)}",
        "supplement_replicate_rows": len(replicates),
        "supplement_classification": row["supplement_classification"],
        "artifact_manifest_rows": len(pd.read_csv(OUTPUT_DIR / "artifact_manifest.csv")),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _verify_manifest(manifest_path: Path, base_dir: Path) -> None:
    manifest = pd.read_csv(manifest_path)
    if manifest["path"].duplicated().any():
        raise ValueError(f"Duplicate manifest paths: {manifest_path}")
    for item in manifest.itertuples(index=False):
        path = base_dir / str(item.path)
        if not path.is_file():
            raise FileNotFoundError(f"Manifest file missing: {path}")
        _assert_equal(path.stat().st_size, int(item.size_bytes), f"size for {item.path}")
        _assert_equal(_sha256(path), str(item.sha256), f"hash for {item.path}")


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


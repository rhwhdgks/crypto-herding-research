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
PRESERVED_MASTER = (
    PROJECT_ROOT / "outputs" / "research_master_report_2026-07-21_pre_supplement.md"
)


def main() -> None:
    config = load_config(CONFIG_PATH)
    required = [
        "protocol_snapshot.md",
        "config_snapshot.yaml",
        "input_manifest.csv",
        "input_manifest_v1_1.csv",
        "code_manifest.csv",
        "code_manifest_v1_1.csv",
        "artifact_manifest.csv",
        "artifact_manifest_v1_1.csv",
        "manifest_amendment_v1_1.json",
        "manifest_amendment_v1_1.md",
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
        raise FileNotFoundError(f"Missing amended supplement artifacts: {missing}")

    provenance = json.loads((OUTPUT_DIR / "provenance.json").read_text(encoding="utf-8"))
    protocol_path = PROJECT_ROOT / str(config["protocol"]["path"])
    _assert_equal(provenance["protocol_sha256"], _sha256(protocol_path), "protocol hash")
    _assert_equal(provenance["config_sha256"], _sha256(CONFIG_PATH), "config hash")
    _assert_equal(_sha256(OUTPUT_DIR / "protocol_snapshot.md"), _sha256(protocol_path), "protocol snapshot")
    _assert_equal(_sha256(OUTPUT_DIR / "config_snapshot.yaml"), _sha256(CONFIG_PATH), "config snapshot")
    _verify_manifest(OUTPUT_DIR / "input_manifest_v1_1.csv", PROJECT_ROOT)
    _verify_manifest(OUTPUT_DIR / "code_manifest_v1_1.csv", PROJECT_ROOT)
    _verify_manifest(OUTPUT_DIR / "artifact_manifest_v1_1.csv", OUTPUT_DIR)

    amendment = json.loads(
        (OUTPUT_DIR / "manifest_amendment_v1_1.json").read_text(encoding="utf-8")
    )
    if not bool(amendment["preserved_input_exact_match"]):
        raise ValueError("The preserved pre-supplement input was not an exact match")
    if any(
        bool(amendment[key])
        for key in ("result_tables_changed", "simulation_changed", "decision_changed")
    ):
        raise ValueError("The manifest-only amendment unexpectedly changed research results")
    original_manifest = pd.read_csv(OUTPUT_DIR / "input_manifest.csv")
    original_master = original_manifest.loc[
        original_manifest["path"].eq("outputs/research_master_report_2026-07-21.md")
    ].iloc[0]
    _assert_equal(PRESERVED_MASTER.stat().st_size, int(original_master["size_bytes"]), "preserved master size")
    _assert_equal(_sha256(PRESERVED_MASTER), str(original_master["sha256"]), "preserved master hash")

    parent = pd.read_csv(PARENT_DIR / "final_mechanical_decision.csv").iloc[0]
    _assert_equal(parent["classification"], "mechanical_null_not_confirmed", "parent classification")
    _assert_equal(int(parent["mechanical_convergence_cells_passed"]), 5, "parent passes")
    parent_equations = pd.read_csv(PARENT_DIR / "equation_verification.csv")
    if not parent_equations["equation_gate_pass"].astype(bool).all():
        raise ValueError("Parent equation gate is not 3/3")

    repetitions = int(config["simulation"]["repetitions"])
    scenario_count = len(config["simulation"]["assets"]) * len(config["simulation"]["observations"])
    model_count = len(config["models"]["names"])
    replicates = pd.read_parquet(OUTPUT_DIR / "supplement_replicates.parquet")
    _assert_equal(len(replicates), repetitions * scenario_count * model_count, "replicate rows")
    cells = replicates.groupby(["scenario", "model"])["repetition"].nunique()
    _assert_equal(len(cells), scenario_count * model_count, "replicate cells")
    if not cells.eq(repetitions).all():
        raise ValueError("At least one replicate cell is incomplete")
    diagnostics = pd.read_parquet(OUTPUT_DIR / "supplement_diagnostics.parquet")
    _assert_equal(len(diagnostics), repetitions * scenario_count, "diagnostic rows")

    gates = pd.read_csv(OUTPUT_DIR / "supplement_gates.csv")
    primary = gates.loc[gates["primary_supplement_cell"].astype(bool)]
    _assert_equal(len(primary), len(config["simulation"]["assets"]) * model_count, "primary gates")
    if not primary["gate_pass"].astype(bool).all():
        raise ValueError("At least one frozen supplement primary gate failed")
    decision = pd.read_csv(OUTPUT_DIR / "supplement_decision.csv").iloc[0]
    if not bool(decision["parent_classification_preserved"]):
        raise ValueError("Parent failure is not preserved")
    if not bool(decision["analytic_mechanism_established"]):
        raise ValueError("Analytic mechanism is not established")
    if not bool(decision["finite_sample_convergence_supported"]):
        raise ValueError("Finite-sample convergence is not supported")

    _assert_equal(_sha256(MASTER_REPORT), provenance["master_report_update_sha256"], "master report hash")
    _assert_equal(
        _sha256(OUTPUT_DIR / "master_report_update.md"),
        provenance["master_report_update_sha256"],
        "staged master report hash",
    )
    master = MASTER_REPORT.read_text(encoding="utf-8")
    if "### v1.1 유한표본 수렴 보충" not in master:
        raise ValueError("Published master report lacks the supplement amendment")

    result = {
        "status": "verified",
        "manifest_amendment": "path_only; research results unchanged",
        "parent_classification": decision["parent_preregistered_classification"],
        "parent_convergence_gate": (
            f"{int(decision['parent_mechanical_convergence_passed'])}/"
            f"{int(decision['parent_mechanical_convergence_total'])}"
        ),
        "analytic_equation_gate": (
            f"{int(decision['analytic_equation_cells_passed'])}/"
            f"{int(decision['analytic_equation_cells_total'])}"
        ),
        "supplement_primary_gate": f"{int(primary['gate_pass'].sum())}/{len(primary)}",
        "supplement_replicate_rows": len(replicates),
        "supplement_classification": decision["supplement_classification"],
        "artifact_manifest_rows": len(pd.read_csv(OUTPUT_DIR / "artifact_manifest_v1_1.csv")),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


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

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


CONFIG_PATH = PROJECT_ROOT / "configs" / "research" / "csad_specification_audit_v1.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v2" / "csad_specification_audit_v1"


def main() -> None:
    config = load_config(CONFIG_PATH)
    required = [
        "protocol_snapshot.md",
        "config_snapshot.yaml",
        "provenance.json",
        "input_manifest.csv",
        "artifact_manifest_v1_2.csv",
        "model_diagnostics.csv",
        "intercept_mechanical_comparison.csv",
        "conditional_concentration_results.csv",
        "volatility_regime_results.csv",
        "false_positive_summary.csv",
        "simulation_replicates.parquet",
        "simulation_diagnostics.parquet",
        "empirical_vs_null.csv",
        "heterogeneity_group_summary.csv",
        "descriptive_random_effects.csv",
        "descriptive_univariate_meta_coefficients.csv",
        "structural_robustness_decisions_v1_1.csv",
        "csad_specification_audit_report_v1_1.md",
        "plots/false_positive_rates.png",
        "plots/false_positive_rates_contrast_fixed.png",
        "plots/intercept_mechanics_font_fixed.png",
        "plots/empirical_vs_null.png",
        "plots/empirical_vs_null_labels_fixed.png",
        "plots/descriptive_random_effects_font_fixed.png",
    ]
    missing = [relative for relative in required if not (OUTPUT_DIR / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing audit artifacts: {missing}")

    provenance = json.loads((OUTPUT_DIR / "provenance.json").read_text(encoding="utf-8"))
    protocol_path = PROJECT_ROOT / str(config["protocol"]["path"])
    _assert_equal(provenance["config_sha256"], _sha256(CONFIG_PATH), "config hash")
    _assert_equal(provenance["protocol_sha256"], _sha256(protocol_path), "protocol hash")
    _assert_equal(_sha256(OUTPUT_DIR / "config_snapshot.yaml"), _sha256(CONFIG_PATH), "config snapshot")
    _assert_equal(
        _sha256(OUTPUT_DIR / "protocol_snapshot.md"), _sha256(protocol_path), "protocol snapshot"
    )
    _verify_manifest(OUTPUT_DIR / "input_manifest.csv", PROJECT_ROOT)
    _verify_manifest(OUTPUT_DIR / "artifact_manifest_v1_2.csv", OUTPUT_DIR)

    repetitions = int(config["simulation"]["repetitions"])
    dgp_count = len(config["simulation"]["dgp"])
    scenario_count = len(config["simulation"]["scenarios"])
    model_count = 4
    simulation = pd.read_parquet(OUTPUT_DIR / "simulation_replicates.parquet")
    expected_simulation_rows = dgp_count * scenario_count * repetitions * model_count
    _assert_equal(len(simulation), expected_simulation_rows, "simulation replicate rows")
    _assert_equal(simulation["dgp"].nunique(), dgp_count, "simulation DGP count")
    _assert_equal(simulation["scenario"].nunique(), scenario_count, "simulation scenario count")
    _assert_equal(simulation["model"].nunique(), model_count, "simulation model count")
    if simulation[["target_coefficient", "target_p_value"]].isna().any().any():
        raise ValueError("Simulation target estimates contain missing values")
    cell_sizes = simulation.groupby(["dgp", "scenario", "model"])["repetition"].nunique()
    if not cell_sizes.eq(repetitions).all():
        raise ValueError("Simulation cells do not contain every preregistered repetition")

    simulation_diagnostics = pd.read_parquet(OUTPUT_DIR / "simulation_diagnostics.parquet")
    _assert_equal(
        len(simulation_diagnostics),
        dgp_count * scenario_count * repetitions,
        "simulation diagnostic rows",
    )
    false_positive = pd.read_csv(OUTPUT_DIR / "false_positive_summary.csv")
    _assert_equal(
        len(false_positive), dgp_count * scenario_count * model_count, "false-positive rows"
    )
    if not false_positive["raw_false_positive_rate"].between(0.0, 1.0).all():
        raise ValueError("False-positive rates are outside [0, 1]")

    integrity = pd.read_csv(OUTPUT_DIR / "input_integrity.csv")
    _assert_equal(len(integrity), 20, "empirical panel count")
    if not integrity["rebuild_matches_stored"].astype(bool).all():
        raise ValueError("At least one empirical panel failed independent reconstruction")
    model_diagnostics = pd.read_csv(OUTPUT_DIR / "model_diagnostics.csv")
    _assert_equal(len(model_diagnostics), 160, "empirical model diagnostic rows")
    decisions = pd.read_csv(OUTPUT_DIR / "structural_robustness_decisions_v1_1.csv")
    _assert_equal(len(decisions), 10, "structural decision rows")
    empirical_null = pd.read_csv(OUTPUT_DIR / "empirical_vs_null.csv")
    _assert_equal(len(empirical_null), 240, "empirical-vs-null rows")

    result = {
        "status": "verified",
        "protocol_sha256": provenance["protocol_sha256"],
        "config_sha256": provenance["config_sha256"],
        "input_manifest_rows": len(pd.read_csv(OUTPUT_DIR / "input_manifest.csv")),
        "artifact_manifest_rows": len(
            pd.read_csv(OUTPUT_DIR / "artifact_manifest_v1_2.csv")
        ),
        "empirical_panels": len(integrity),
        "simulation_replicate_rows": len(simulation),
        "structural_decisions": len(decisions),
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

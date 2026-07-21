from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from csad_mechanical_derivation import (  # noqa: E402
    build_gaussian_theory_table,
    build_theory_identities,
    verify_gaussian_equations,
)
from csad_mechanical_reporting import (  # noqa: E402
    build_korean_mechanical_report,
    build_master_report_update,
    plot_false_positive_contrast,
    plot_projection_mechanism,
    plot_symmetric_sign_robustness,
    plot_theory_convergence,
)
from csad_mechanical_simulation import (  # noqa: E402
    build_final_mechanical_decision,
    run_convergence_simulation,
    run_robustness_simulation,
)
from utils import load_config  # noqa: E402


DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "research" / "csad_mechanical_derivation_v1.yaml"
)
MASTER_REPORT_SOURCE = PROJECT_ROOT / "outputs" / "research_master_report_2026-07-20.md"
MASTER_REPORT_OUTPUT = PROJECT_ROOT / "outputs" / "research_master_report_2026-07-21.md"
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive and simulate mechanical negative coefficients in CSAD models."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    logging.basicConfig(
        level=getattr(logging, str(config["logging"]["level"]).upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    output_dir = (PROJECT_ROOT / str(config["output"]["base_dir"])).resolve()
    if bool(config["output"].get("overwrite", False)):
        raise ValueError("This preregistered study forbids overwrite")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists and cannot be overwritten: {output_dir}")
    if MASTER_REPORT_OUTPUT.exists():
        raise FileExistsError(
            f"Updated master report already exists and cannot be overwritten: {MASTER_REPORT_OUTPUT}"
        )

    staging_dir = output_dir.with_name(f".{output_dir.name}.staging-{os.getpid()}")
    if staging_dir.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_dir}")
    staging_dir.mkdir(parents=True)
    try:
        _run(config, config_path, staging_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir.rename(output_dir)
        _write_master_report(output_dir)
    except Exception:
        LOGGER.exception("Mechanical derivation failed; staging artifacts preserved at %s", staging_dir)
        raise
    LOGGER.info("CSAD mechanical derivation complete: %s", output_dir)


def _run(config: dict, config_path: Path, output_dir: Path) -> None:
    protocol_path = (PROJECT_ROOT / str(config["protocol"]["path"])).resolve()
    protocol_hash = _sha256(protocol_path)
    config_hash = _sha256(config_path)
    shutil.copy2(protocol_path, output_dir / "protocol_snapshot.md")
    shutil.copy2(config_path, output_dir / "config_snapshot.yaml")
    _write_input_manifest(config_path, protocol_path, output_dir / "input_manifest.csv")
    _write_code_manifest(output_dir / "code_manifest.csv")

    asset_counts = [int(value) for value in config["convergence_simulation"]["assets"]]
    sigma = float(config["theory"]["sigma"])
    equation_verification = verify_gaussian_equations(
        asset_counts,
        sigma,
        float(config["theory"]["equation_tolerance"]),
    )
    theory_tables = {
        "theory_identities": build_theory_identities(),
        "gaussian_theory_coefficients": build_gaussian_theory_table(asset_counts, sigma),
        "equation_verification": equation_verification,
    }

    LOGGER.info("Running preregistered large-sample Gaussian convergence simulation")
    convergence = run_convergence_simulation(config)
    LOGGER.info("Running preregistered eight-DGP robustness simulation")
    robustness = run_robustness_simulation(config)
    final_decision = build_final_mechanical_decision(
        equation_verification,
        convergence["convergence_gates"],
        robustness["symmetric_robustness_decision"],
    )

    tables = {
        **theory_tables,
        "convergence_summary": convergence["convergence_summary"],
        "convergence_gates": convergence["convergence_gates"],
        "robustness_summary": robustness["robustness_summary"],
        "robustness_diagnostic_summary": robustness[
            "robustness_diagnostic_summary"
        ],
        "symmetric_robustness_cells": robustness["symmetric_robustness_cells"],
        "symmetric_robustness_decision": robustness[
            "symmetric_robustness_decision"
        ],
        "final_mechanical_decision": final_decision,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    convergence["convergence_replicates"].to_parquet(
        output_dir / "convergence_replicates.parquet", index=False
    )
    convergence["convergence_diagnostics"].to_parquet(
        output_dir / "convergence_diagnostics.parquet", index=False
    )
    robustness["robustness_replicates"].to_parquet(
        output_dir / "robustness_replicates.parquet", index=False
    )
    robustness["robustness_diagnostics"].to_parquet(
        output_dir / "robustness_diagnostics.parquet", index=False
    )

    plot_paths = [
        "plots/gaussian_theory_convergence.png",
        "plots/nonherding_false_positive_contrast.png",
        "plots/mechanical_negative_sign_robustness.png",
        "plots/projection_mechanism.png",
    ]
    plot_theory_convergence(tables["convergence_summary"], output_dir / plot_paths[0])
    plot_false_positive_contrast(tables["robustness_summary"], output_dir / plot_paths[1])
    plot_symmetric_sign_robustness(
        tables["robustness_summary"], output_dir / plot_paths[2]
    )
    plot_projection_mechanism(sigma, asset_counts[0], output_dir / plot_paths[3])
    report = build_korean_mechanical_report(
        tables,
        protocol_hash=protocol_hash,
        config_hash=config_hash,
        plot_paths=plot_paths,
    )
    (output_dir / "csad_mechanical_derivation_report.md").write_text(
        report, encoding="utf-8"
    )
    provenance = {
        "pipeline_version": "csad-mechanical-derivation-v1",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_hash,
        "config_sha256": config_hash,
        "convergence_seed": int(config["convergence_simulation"]["seed"]),
        "convergence_repetitions": int(
            config["convergence_simulation"]["repetitions"]
        ),
        "robustness_seed": int(config["robustness_simulation"]["seed"]),
        "robustness_repetitions": int(
            config["robustness_simulation"]["repetitions"]
        ),
        "dgp_count": len(config["dgp"]),
        "scenario_count": len(config["robustness_simulation"]["scenarios"]),
        "output_overwrite_permitted": False,
        "interpretation_scope": "statistical_mechanism_not_imitation_or_alpha",
        "final_classification": final_decision.iloc[0]["classification"],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_artifact_manifest(output_dir)


def _write_master_report(output_dir: Path) -> None:
    source = MASTER_REPORT_SOURCE.read_text(encoding="utf-8")
    decision = pd.read_csv(output_dir / "final_mechanical_decision.csv")
    convergence = pd.read_csv(output_dir / "convergence_summary.csv")
    robustness = pd.read_csv(output_dir / "symmetric_robustness_decision.csv")
    updated = build_master_report_update(source, decision, convergence, robustness)
    temporary = MASTER_REPORT_OUTPUT.with_suffix(".md.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary master report already exists: {temporary}")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(MASTER_REPORT_OUTPUT)


def _write_input_manifest(
    config_path: Path,
    protocol_path: Path,
    destination: Path,
) -> None:
    rows = []
    for path in sorted([config_path, protocol_path]):
        rows.append(_manifest_row(path, PROJECT_ROOT))
    pd.DataFrame(rows).to_csv(destination, index=False)


def _write_code_manifest(destination: Path) -> None:
    relative_paths = [
        "src/csad_mechanical_derivation.py",
        "src/csad_mechanical_simulation.py",
        "src/csad_mechanical_reporting.py",
        "src/csad_null_simulation.py",
        "src/frequency_sensitivity.py",
        "scripts/run_csad_mechanical_derivation.py",
        "scripts/verify_csad_mechanical_derivation.py",
        "tests/test_csad_mechanical_derivation.py",
    ]
    rows = [_manifest_row(PROJECT_ROOT / relative, PROJECT_ROOT) for relative in relative_paths]
    pd.DataFrame(rows).to_csv(destination, index=False)


def _write_artifact_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.csv":
            continue
        rows.append(_manifest_row(path, output_dir))
    pd.DataFrame(rows).to_csv(output_dir / "artifact_manifest.csv", index=False)


def _manifest_row(path: Path, base: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve().relative_to(base.resolve())),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()

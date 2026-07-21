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

from csad_mechanical_supplement import (  # noqa: E402
    build_korean_supplement_report,
    build_master_supplement_update,
    build_supplement_decision,
    plot_convergence_ladder,
    run_finite_sample_supplement,
)
from utils import load_config  # noqa: E402


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "research"
    / "csad_mechanical_convergence_supplement_v1_1.yaml"
)
PARENT_DIR = PROJECT_ROOT / "outputs" / "v2" / "csad_mechanical_derivation_v1"
MASTER_REPORT = PROJECT_ROOT / "outputs" / "research_master_report_2026-07-21.md"
MASTER_REPORT_BACKUP = (
    PROJECT_ROOT / "outputs" / "research_master_report_2026-07-21_pre_supplement.md"
)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered v1.1 finite-sample convergence supplement."
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
        raise ValueError("This supplement forbids overwrite")
    if output_dir.exists():
        raise FileExistsError(f"Supplement output already exists: {output_dir}")
    if not PARENT_DIR.is_dir() or not MASTER_REPORT.is_file():
        raise FileNotFoundError("The immutable parent study and master report are required")

    staging_dir = output_dir.with_name(f".{output_dir.name}.staging-{os.getpid()}")
    if staging_dir.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_dir}")
    staging_dir.mkdir(parents=True)
    try:
        _run(config, config_path, staging_dir)
        staging_dir.rename(output_dir)
        _publish_master_report(output_dir / "master_report_update.md")
    except Exception:
        LOGGER.exception("Supplement failed; staging artifacts preserved at %s", staging_dir)
        raise
    LOGGER.info("CSAD finite-sample supplement complete: %s", output_dir)


def _run(config: dict, config_path: Path, output_dir: Path) -> None:
    protocol_path = (PROJECT_ROOT / str(config["protocol"]["path"])).resolve()
    protocol_hash = _sha256(protocol_path)
    config_hash = _sha256(config_path)
    shutil.copy2(protocol_path, output_dir / "protocol_snapshot.md")
    shutil.copy2(config_path, output_dir / "config_snapshot.yaml")
    _write_input_manifest(config_path, protocol_path, output_dir / "input_manifest.csv")
    _write_code_manifest(output_dir / "code_manifest.csv")

    parent_decision = pd.read_csv(PARENT_DIR / "final_mechanical_decision.csv")
    parent_equations = pd.read_csv(PARENT_DIR / "equation_verification.csv")
    LOGGER.info("Running independent-seed finite-sample convergence ladder")
    tables = run_finite_sample_supplement(config)
    supplement_decision = build_supplement_decision(
        parent_decision, parent_equations, tables["supplement_gates"]
    )
    csv_tables = {
        "supplement_summary": tables["supplement_summary"],
        "supplement_diagnostic_summary": tables["supplement_diagnostic_summary"],
        "supplement_gates": tables["supplement_gates"],
        "supplement_decision": supplement_decision,
    }
    for name, table in csv_tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    tables["supplement_replicates"].to_parquet(
        output_dir / "supplement_replicates.parquet", index=False
    )
    tables["supplement_diagnostics"].to_parquet(
        output_dir / "supplement_diagnostics.parquet", index=False
    )
    plot_convergence_ladder(
        tables["supplement_summary"],
        output_dir / "plots" / "finite_sample_convergence_ladder.png",
    )
    report = build_korean_supplement_report(
        tables["supplement_summary"],
        tables["supplement_gates"],
        supplement_decision,
        protocol_hash,
        config_hash,
    )
    (output_dir / "csad_mechanical_convergence_supplement_report.md").write_text(
        report, encoding="utf-8"
    )
    updated_master = build_master_supplement_update(
        MASTER_REPORT.read_text(encoding="utf-8"), supplement_decision
    )
    (output_dir / "master_report_update.md").write_text(updated_master, encoding="utf-8")
    provenance = {
        "pipeline_version": "csad-mechanical-convergence-supplement-v1.1",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_hash,
        "config_sha256": config_hash,
        "parent_provenance_sha256": _sha256(PARENT_DIR / "provenance.json"),
        "simulation_seed": int(config["simulation"]["seed"]),
        "simulation_repetitions": int(config["simulation"]["repetitions"]),
        "parent_classification_preserved": str(
            supplement_decision.iloc[0]["parent_preregistered_classification"]
        ),
        "supplement_classification": str(
            supplement_decision.iloc[0]["supplement_classification"]
        ),
        "master_report_update_sha256": _sha256(
            output_dir / "master_report_update.md"
        ),
        "output_overwrite_permitted": False,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_artifact_manifest(output_dir)


def _publish_master_report(source: Path) -> None:
    if not MASTER_REPORT_BACKUP.exists():
        shutil.copy2(MASTER_REPORT, MASTER_REPORT_BACKUP)
    temporary = MASTER_REPORT.with_suffix(".md.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary master report exists: {temporary}")
    shutil.copy2(source, temporary)
    temporary.replace(MASTER_REPORT)


def _write_input_manifest(config_path: Path, protocol_path: Path, destination: Path) -> None:
    paths = [
        config_path,
        protocol_path,
        PROJECT_ROOT / "research_protocols" / "csad_mechanical_derivation_v1.md",
        PARENT_DIR / "provenance.json",
        PARENT_DIR / "final_mechanical_decision.csv",
        PARENT_DIR / "equation_verification.csv",
        PARENT_DIR / "convergence_summary.csv",
        PARENT_DIR / "convergence_gates.csv",
        MASTER_REPORT,
    ]
    pd.DataFrame([_manifest_row(path, PROJECT_ROOT) for path in paths]).to_csv(
        destination, index=False
    )


def _write_code_manifest(destination: Path) -> None:
    relative_paths = [
        "src/csad_mechanical_derivation.py",
        "src/csad_mechanical_simulation.py",
        "src/csad_mechanical_supplement.py",
        "src/csad_null_simulation.py",
        "src/frequency_sensitivity.py",
        "scripts/run_csad_mechanical_supplement.py",
        "scripts/verify_csad_mechanical_derivation_v1_1.py",
        "tests/test_csad_mechanical_supplement.py",
    ]
    pd.DataFrame(
        [_manifest_row(PROJECT_ROOT / relative, PROJECT_ROOT) for relative in relative_paths]
    ).to_csv(destination, index=False)


def _write_artifact_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.csv":
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


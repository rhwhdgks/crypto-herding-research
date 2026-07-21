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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/crypto-herding-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from csad_audit_reporting import (  # noqa: E402
    build_intercept_verdict,
    build_korean_audit_report,
    build_structural_robustness_decisions,
    plot_empirical_null_comparison,
    plot_false_positive_heatmaps,
    plot_intercept_mechanics,
    plot_random_effects,
)
from csad_null_simulation import compare_empirical_to_null, run_null_monte_carlo  # noqa: E402
from csad_specification_audit import (  # noqa: E402
    build_empirical_heterogeneity,
    load_empirical_panels,
    run_conditional_concentration_audit,
    run_empirical_model_audit,
    run_volatility_regime_audit,
    validate_audit_config,
)
from utils import load_config, setup_logging  # noqa: E402


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrected CSAD specification and mechanism audit v1을 실행합니다."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "research" / "csad_specification_audit_v1.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    validate_audit_config(config, PROJECT_ROOT)
    output_dir = PROJECT_ROOT / str(config["output"]["base_dir"])
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite preregistered audit output: {output_dir}"
        )
    staging_dir = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    if staging_dir.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_dir}")
    staging_dir.mkdir(parents=True)
    try:
        _run(config, config_path, staging_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir.rename(output_dir)
    except Exception:
        LOGGER.exception("Audit failed; staging artifacts are preserved at %s", staging_dir)
        raise
    LOGGER.info("CSAD specification audit complete: %s", output_dir)


def _run(config: dict, config_path: Path, output_dir: Path) -> None:
    protocol_path = PROJECT_ROOT / str(config["protocol"]["path"])
    protocol_hash = _sha256(protocol_path)
    config_hash = _sha256(config_path)
    shutil.copy2(config_path, output_dir / "config_snapshot.yaml")
    shutil.copy2(protocol_path, output_dir / "protocol_snapshot.md")
    input_manifest = _build_input_manifest(config, config_path, protocol_path)
    input_manifest.to_csv(output_dir / "input_manifest.csv", index=False)

    LOGGER.info("Loading and rebuilding 20 empirical frequency panels")
    panels = load_empirical_panels(config, PROJECT_ROOT)
    empirical_tables = run_empirical_model_audit(panels, config)
    conditional = run_conditional_concentration_audit(panels, config)
    regimes, regime_assignments = run_volatility_regime_audit(panels, config)
    heterogeneity = build_empirical_heterogeneity(
        empirical_tables["model_diagnostics"], empirical_tables["panel_metrics"]
    )

    LOGGER.info(
        "Running frozen null Monte Carlo: dgps=%d scenarios=%d repetitions=%d",
        len(config["simulation"]["dgp"]),
        len(config["simulation"]["scenarios"]),
        int(config["simulation"]["repetitions"]),
    )
    simulation_tables = run_null_monte_carlo(config)
    empirical_vs_null = compare_empirical_to_null(
        empirical_tables["model_diagnostics"],
        simulation_tables["simulation_replicates"],
        config,
    )
    decisions = build_structural_robustness_decisions(
        empirical_tables["model_diagnostics"],
        regimes,
        simulation_tables["false_positive_summary"],
        config,
    )
    intercept_verdict = build_intercept_verdict(
        empirical_tables["intercept_mechanical_comparison"],
        simulation_tables["null_intercept_mechanical_summary"],
        simulation_tables["false_positive_summary"],
        config,
    )

    tables = {
        **empirical_tables,
        **heterogeneity,
        "conditional_concentration_results": conditional,
        "volatility_regime_results": regimes,
        "volatility_regime_assignments": regime_assignments,
        "empirical_vs_null": empirical_vs_null,
        "structural_robustness_decisions": decisions,
        "intercept_verdict": intercept_verdict,
        "false_positive_summary": simulation_tables["false_positive_summary"],
        "simulation_diagnostic_summary": simulation_tables["simulation_diagnostic_summary"],
        "null_intercept_mechanical_summary": simulation_tables[
            "null_intercept_mechanical_summary"
        ],
    }
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    simulation_tables["simulation_replicates"].to_parquet(
        output_dir / "simulation_replicates.parquet", index=False
    )
    simulation_tables["simulation_diagnostics"].to_parquet(
        output_dir / "simulation_diagnostics.parquet", index=False
    )

    plots_dir = output_dir / "plots"
    plot_paths = [
        "plots/false_positive_rates.png",
        "plots/intercept_mechanics.png",
        "plots/empirical_vs_null.png",
        "plots/descriptive_random_effects.png",
    ]
    plot_false_positive_heatmaps(
        simulation_tables["false_positive_summary"], output_dir / plot_paths[0]
    )
    plot_intercept_mechanics(
        empirical_tables["intercept_mechanical_comparison"], output_dir / plot_paths[1]
    )
    plot_empirical_null_comparison(empirical_vs_null, output_dir / plot_paths[2])
    plot_random_effects(
        heterogeneity["descriptive_random_effects"], output_dir / plot_paths[3]
    )
    report = build_korean_audit_report(
        tables,
        plot_paths,
        protocol_hash=protocol_hash,
        config_hash=config_hash,
    )
    (output_dir / "csad_specification_audit_report.md").write_text(report, encoding="utf-8")
    provenance = {
        "pipeline_version": "csad-specification-audit-v1",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_hash,
        "config_sha256": config_hash,
        "simulation_seed": int(config["simulation"]["seed"]),
        "simulation_repetitions": int(config["simulation"]["repetitions"]),
        "empirical_dataset_count": len(config["empirical"]["datasets"]),
        "empirical_panel_count": len(panels),
        "output_overwrite_permitted": False,
        "interpretation_scope": "specification_and_mechanism_not_imitation_or_alpha",
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_artifact_manifest(output_dir)


def _build_input_manifest(
    config: dict,
    config_path: Path,
    protocol_path: Path,
) -> pd.DataFrame:
    paths = {config_path, protocol_path}
    for dataset in config["empirical"]["datasets"]:
        paths.add(PROJECT_ROOT / str(dataset["targets_path"]))
        base = PROJECT_ROOT / str(dataset["intermediate_dir"])
        for frequency in config["empirical"]["frequencies"]:
            paths.add(base / f"{frequency}_member_rows.parquet")
            return_panel = base / f"{frequency}_return_panel.parquet"
            if return_panel.exists():
                paths.add(return_panel)
    rows = []
    for path in sorted(paths):
        resolved = path.resolve()
        rows.append(
            {
                "path": str(resolved.relative_to(PROJECT_ROOT.resolve())),
                "size_bytes": resolved.stat().st_size,
                "sha256": _sha256(resolved),
            }
        )
    return pd.DataFrame(rows)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_artifact_manifest(
    output_dir: Path,
    manifest_name: str = "artifact_manifest.csv",
) -> None:
    manifest_path = output_dir / manifest_name
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("artifact_manifest"):
            continue
        rows.append(
            {
                "path": str(path.relative_to(output_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    pd.DataFrame(rows).to_csv(manifest_path, index=False)


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "v2"
    / "csad_mechanical_derivation_v1"
    / "supplement_v1_1"
)
CURRENT_MASTER = PROJECT_ROOT / "outputs" / "research_master_report_2026-07-21.md"
PRESERVED_MASTER = (
    PROJECT_ROOT / "outputs" / "research_master_report_2026-07-21_pre_supplement.md"
)


def main() -> None:
    destinations = [
        OUTPUT_DIR / "input_manifest_v1_1.csv",
        OUTPUT_DIR / "code_manifest_v1_1.csv",
        OUTPUT_DIR / "manifest_amendment_v1_1.json",
        OUTPUT_DIR / "manifest_amendment_v1_1.md",
        OUTPUT_DIR / "artifact_manifest_v1_1.csv",
    ]
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(f"Manifest amendment outputs already exist: {existing}")

    original = pd.read_csv(OUTPUT_DIR / "input_manifest.csv")
    old_relative = "outputs/research_master_report_2026-07-21.md"
    new_relative = "outputs/research_master_report_2026-07-21_pre_supplement.md"
    matching = original["path"].eq(old_relative)
    if int(matching.sum()) != 1:
        raise ValueError("Original input manifest does not contain exactly one master report row")
    original_row = original.loc[matching].iloc[0]
    if (
        int(original_row["size_bytes"]) != PRESERVED_MASTER.stat().st_size
        or str(original_row["sha256"]) != _sha256(PRESERVED_MASTER)
    ):
        raise ValueError("Preserved pre-supplement report does not match the recorded input")
    corrected = original.copy()
    corrected.loc[matching, "path"] = new_relative
    corrected.to_csv(OUTPUT_DIR / "input_manifest_v1_1.csv", index=False)

    amendment = {
        "amendment_version": "1.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "The original input manifest recorded the pre-supplement master report hash under a path that the runner subsequently updated.",
        "original_manifest_preserved": "input_manifest.csv",
        "corrected_manifest": "input_manifest_v1_1.csv",
        "original_path": old_relative,
        "corrected_path": new_relative,
        "recorded_size_bytes": int(original_row["size_bytes"]),
        "recorded_sha256": str(original_row["sha256"]),
        "preserved_input_exact_match": True,
        "current_master_sha256": _sha256(CURRENT_MASTER),
        "result_tables_changed": False,
        "simulation_changed": False,
        "decision_changed": False,
    }
    (OUTPUT_DIR / "manifest_amendment_v1_1.json").write_text(
        json.dumps(amendment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "manifest_amendment_v1_1.md").write_text(
        "\n".join(
            [
                "# Manifest amendment v1.1",
                "",
                "원 보충 실행은 갱신 전 master report를 input으로 읽은 뒤 같은 경로에 amendment를 발행했습니다.",
                "그 결과 원 `input_manifest.csv`의 크기와 SHA-256은 정확하지만, 검증 시점의 현재 경로 내용은 갱신본이 되어 경로 검증이 실패했습니다.",
                "",
                "- 원 manifest와 실패 verifier는 보존합니다.",
                "- 기록된 input과 byte-for-byte 일치하는 `research_master_report_2026-07-21_pre_supplement.md` 백업이 존재합니다.",
                "- `input_manifest_v1_1.csv`는 그 보존 백업 경로만 교정합니다.",
                "- simulation, summary, gate, decision, report 내용은 변경하지 않았습니다.",
                f"- 보존 input SHA-256: `{original_row['sha256']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_code_manifest(OUTPUT_DIR / "code_manifest_v1_1.csv")
    _write_artifact_manifest(OUTPUT_DIR / "artifact_manifest_v1_1.csv")
    print(json.dumps(amendment, ensure_ascii=False, indent=2))


def _write_code_manifest(destination: Path) -> None:
    relative_paths = [
        "src/csad_mechanical_derivation.py",
        "src/csad_mechanical_simulation.py",
        "src/csad_mechanical_supplement.py",
        "src/csad_null_simulation.py",
        "src/frequency_sensitivity.py",
        "scripts/run_csad_mechanical_supplement.py",
        "scripts/build_csad_mechanical_manifest_amendment.py",
        "scripts/verify_csad_mechanical_derivation_v1_1_amended.py",
        "tests/test_csad_mechanical_derivation.py",
        "tests/test_csad_mechanical_supplement.py",
    ]
    rows = [_manifest_row(PROJECT_ROOT / relative, PROJECT_ROOT) for relative in relative_paths]
    pd.DataFrame(rows).to_csv(destination, index=False)


def _write_artifact_manifest(destination: Path) -> None:
    rows = []
    for path in sorted(OUTPUT_DIR.rglob("*")):
        if not path.is_file() or path.name.startswith("artifact_manifest"):
            continue
        rows.append(_manifest_row(path, OUTPUT_DIR))
    pd.DataFrame(rows).to_csv(destination, index=False)


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


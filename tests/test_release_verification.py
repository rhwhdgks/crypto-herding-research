from __future__ import annotations

import importlib.util
import hashlib
import json
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_release", ROOT / "scripts" / "verify_release.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_release_verifier_rejects_archive_without_manifest(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("README.md", b"incomplete")
    result = MODULE.verify(archive_path)
    assert result["status"] == "failed"
    assert "RELEASE_MANIFEST.json" in result["missing_files"]


def _write_manifest_archive(path: Path, payload: bytes, declared_hash: str | None = None) -> None:
    manifest = {
        "required_files": ["README.md"],
        "files": [
            {
                "path": "README.md",
                "size": len(payload),
                "sha256": declared_hash or hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("README.md", payload)
        archive.writestr("RELEASE_MANIFEST.json", json.dumps(manifest))


def test_release_verifier_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    archive_path = tmp_path / "hash-mismatch.zip"
    _write_manifest_archive(archive_path, b"payload", declared_hash="0" * 64)
    result = MODULE.verify(archive_path)
    assert result["status"] == "failed"
    assert result["hash_mismatches"] == ["README.md"]


def test_release_verifier_rejects_missing_required_payload(tmp_path: Path) -> None:
    archive_path = tmp_path / "missing.zip"
    manifest = {"required_files": ["README.md"], "files": []}
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("RELEASE_MANIFEST.json", json.dumps(manifest))
    result = MODULE.verify(archive_path)
    assert result["status"] == "failed"
    assert "README.md" in result["missing_files"]


def test_truncated_archive_fails_central_directory_read(tmp_path: Path) -> None:
    archive_path = tmp_path / "truncated.zip"
    _write_manifest_archive(archive_path, b"payload")
    archive_path.write_bytes(archive_path.read_bytes()[:-20])
    try:
        MODULE.verify(archive_path)
    except zipfile.BadZipFile:
        pass
    else:
        raise AssertionError("truncated archive unexpectedly passed central-directory parsing")


def test_release_verifier_rejects_duplicate_entries(tmp_path: Path) -> None:
    archive_path = tmp_path / "duplicate.zip"
    payload = b"payload"
    manifest = {
        "required_files": ["README.md"],
        "files": [
            {
                "path": "README.md",
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("README.md", payload)
            archive.writestr("README.md", payload)
            archive.writestr("RELEASE_MANIFEST.json", json.dumps(manifest))
    result = MODULE.verify(archive_path)
    assert result["status"] == "failed"
    assert result["duplicate_entries"] == ["README.md"]

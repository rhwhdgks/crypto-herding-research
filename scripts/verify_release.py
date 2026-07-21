from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path


MANIFEST_NAME = "RELEASE_MANIFEST.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_entry_digest(archive: zipfile.ZipFile, name: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with archive.open(name) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def verify(path: Path) -> dict:
    result = {
        "archive_sha256": _sha256_file(path),
        "file_count": 0,
        "required_file_count": 0,
        "missing_files": [],
        "size_mismatches": [],
        "hash_mismatches": [],
        "crc_failures": [],
        "unexpected_files": [],
        "duplicate_entries": [],
        "status": "failed",
    }
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            result["crc_failures"].append(bad)
        archive_names = archive.namelist()
        names = set(archive_names)
        result["duplicate_entries"] = sorted(name for name, count in Counter(archive_names).items() if count > 1)
        if MANIFEST_NAME not in names:
            result["missing_files"].append(MANIFEST_NAME)
            return result
        manifest = json.loads(archive.read(MANIFEST_NAME))
        expected = {entry["path"]: entry for entry in manifest["files"]}
        required = set(manifest["required_files"])
        result["file_count"] = len(expected)
        result["required_file_count"] = len(required)
        result["missing_files"].extend(sorted((set(expected) | required).difference(names)))
        result["unexpected_files"] = sorted(names.difference(set(expected) | {MANIFEST_NAME}))
        for name, entry in expected.items():
            if name not in names:
                continue
            size, digest = _archive_entry_digest(archive, name)
            if size != int(entry["size"]):
                result["size_mismatches"].append(name)
            if digest != entry["sha256"]:
                result["hash_mismatches"].append(name)
    failure_fields = (
        "missing_files",
        "size_mismatches",
        "hash_mismatches",
        "crc_failures",
        "unexpected_files",
        "duplicate_entries",
    )
    if not any(result[field] for field in failure_fields):
        result["status"] = "ok"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify every entry in a crypto-herding review archive.")
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.archive)
    except (OSError, zipfile.BadZipFile, KeyError, ValueError, json.JSONDecodeError) as exc:
        result = {"status": "failed", "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

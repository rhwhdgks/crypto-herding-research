from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRS = ("src", "scripts", "configs", "tests", "review_package_docs")
TOP_LEVEL_FILES = (
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "agents.md",
    "claude.md",
    "CODEX_HANDOFF_2026-07-15.md",
    "LICENSE",
)
BASELINE_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "SOLUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "LTCUSDT",
    "TRXUSDT",
    "ATOMUSDT",
    "NEARUSDT",
)
CURATED_RESULTS = (
    "outputs/legacy_invalidated_2026-07-15/INVALIDATED.md",
    "outputs/baseline/report_summary.md",
    "outputs/baseline/regression_results.csv",
    "outputs/baseline/event_study_summary.csv",
    "outputs/baseline/robustness/baseline_robustness_report.md",
    "outputs/paper_like/paper_alignment_report.md",
    "outputs/paper_like/weekly/paper_like_summary.md",
)
REQUIRED_CODE_FILES = (
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "src/tick_herding.py",
    "src/tick_short_horizon.py",
    "src/tick_event_schema.py",
    "scripts/run_pipeline.py",
    "scripts/run_corrected_candidate_validation.py",
    "scripts/run_corrected_state_diagnostics.py",
    "scripts/build_release.py",
    "scripts/verify_release.py",
    "configs/baseline/config.yaml",
    "tests/test_tick_schema_v2.py",
    "outputs/legacy_invalidated_2026-07-15/INVALIDATED.md",
    "outputs/v2/CORRECTION_REPORT_2026-07-15.md",
    "review_package_docs/START_HERE.md",
    "review_package_docs/DATA_SCOPE.md",
)
MANIFEST_NAME = "RELEASE_MANIFEST.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_tick_paths() -> list[Path]:
    selection = ROOT / "review_package_docs" / "RAW_TICK_SELECTION.csv"
    if not selection.is_file():
        raise FileNotFoundError(selection)
    with selection.open(encoding="utf-8", newline="") as handle:
        paths = [ROOT / row["path"] for row in csv.DictReader(handle)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Selected tick files are missing: " + ", ".join(str(path) for path in missing))
    return paths


def _data_files() -> list[Path]:
    files: list[Path] = []
    for symbol in BASELINE_SYMBOLS:
        for timeframe in ("1m", "1d", "1w"):
            path = ROOT / "data" / f"{symbol}_{timeframe}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"Required baseline data is missing: {path}")
            files.append(path)
    for directory in (ROOT / "data" / "news", ROOT / "data" / "reddit", ROOT / "data" / "futures_archive"):
        if not directory.is_dir():
            raise FileNotFoundError(f"Required raw-data directory is missing: {directory}")
        files.extend(path for path in directory.rglob("*") if path.is_file())
    files.extend(_selected_tick_paths())
    references = ROOT / "references"
    if references.is_dir():
        files.extend(path for path in references.glob("*.pdf") if path.is_file())
    return files


def collect_files(output: Path, include_data: bool) -> list[Path]:
    files = [ROOT / name for name in TOP_LEVEL_FILES if (ROOT / name).is_file()]
    for directory in INCLUDE_DIRS:
        base = ROOT / directory
        if base.exists():
            files.extend(path for path in base.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    for experiment in (ROOT / "experiments").rglob("*"):
        if experiment.is_file() and "__pycache__" not in experiment.parts and experiment.suffix in {".py", ".md"}:
            files.append(experiment)
    files.extend(ROOT / name for name in CURATED_RESULTS if (ROOT / name).is_file())
    v2_root = ROOT / "outputs" / "v2"
    if v2_root.exists():
        files.extend(
            path
            for path in v2_root.rglob("*")
            if path.is_file() and "intermediate" not in path.parts
        )
    if include_data:
        files.extend(_data_files())
    resolved_output = output.resolve()
    return sorted(
        {path for path in files if path.resolve() != resolved_output},
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a self-verifying crypto-herding review archive.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--code-only", action="store_true", help="Exclude all raw data for a lightweight CI artifact.")
    parser.add_argument("--max-bytes", type=int, default=500_000_000)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    include_data = not args.code_only
    files = collect_files(output, include_data=include_data)
    relative_names = {path.relative_to(ROOT).as_posix() for path in files}
    required = set(REQUIRED_CODE_FILES)
    if include_data:
        required.update(f"data/{symbol}_1m.parquet" for symbol in BASELINE_SYMBOLS)
        required.update({"data/news/news_headlines.csv", "data/reddit/reddit_posts.csv"})
    missing = sorted(required.difference(relative_names))
    if missing:
        raise SystemExit(f"Required release files are missing: {', '.join(missing)}")

    entries = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in files
    ]
    manifest = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_market_data_included": include_data,
        "raw_data_profile": "baseline-plus-research-samples" if include_data else "none",
        "required_files": sorted(required),
        "files": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True) as archive:
        for path, entry in zip(files, entries, strict=True):
            archive.write(path, entry["path"])
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    if output.stat().st_size > int(args.max_bytes):
        actual_size = output.stat().st_size
        output.unlink(missing_ok=True)
        raise SystemExit(f"Release exceeds size limit: {actual_size} > {args.max_bytes}")
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad is not None:
            output.unlink(missing_ok=True)
            raise SystemExit(f"Archive CRC verification failed: {bad}")
    print(
        json.dumps(
            {
                "output": str(output),
                "file_count": len(entries),
                "archive_bytes": output.stat().st_size,
                "archive_sha256": _sha256_file(output),
                "raw_market_data_included": include_data,
                "status": "ok",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

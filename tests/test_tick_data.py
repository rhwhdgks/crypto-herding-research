import zipfile

from tick_data import _is_readable_zip


def test_readable_zip_requires_a_csv_member(tmp_path) -> None:
    archive_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("README.txt", "not tick data")

    assert not _is_readable_zip(archive_path)


def test_readable_zip_accepts_csv_archive(tmp_path) -> None:
    archive_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("BTCUSDT-aggTrades-2024-01.csv", "1,2,3\n")

    assert _is_readable_zip(archive_path)


def test_readable_zip_rejects_truncated_file(tmp_path) -> None:
    archive_path = tmp_path / "archive.zip"
    archive_path.write_bytes(b"PK\x03\x04truncated")

    assert not _is_readable_zip(archive_path)

"""Tests for generic imaging archive extraction helpers."""

import shutil
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from pyhealth.datasets.imaging.io import (
    _download_archives,
    _extract_zip,
    _resolve_metadata_dir,
)


class TestImagingDownload(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_download_tar(self):
        archive = self.root / "Task04_Hippocampus.tar"
        with tarfile.open(archive, "w") as handle:
            source = self.root / "Task04_Hippocampus"
            source.mkdir()
            (source / "dataset.json").write_text("{}")
            handle.add(source, arcname="Task04_Hippocampus")
        destination = self.root / "extracted" / "Task04_Hippocampus"
        _download_archives(
            [archive.as_uri()],
            destination,
            archive_type="tar",
            expected=("dataset.json",),
            extract_to_parent=True,
        )
        self.assertTrue((destination / "dataset.json").exists())

    def test_download_chaos_zips(self):
        train_zip = self.root / "train.zip"
        test_zip = self.root / "test.zip"
        with zipfile.ZipFile(train_zip, "w") as handle:
            handle.writestr("Train_Sets/CT/1/DICOM_anon/a.dcm", b"")
        with zipfile.ZipFile(test_zip, "w") as handle:
            handle.writestr("Test_Sets/CT/2/DICOM_anon/a.dcm", b"")
        destination = self.root / "CHAOS"
        _download_archives(
            [train_zip.as_uri(), test_zip.as_uri()],
            destination,
            archive_type="zip",
            expected=("Train_Sets", "Test_Sets"),
        )
        self.assertTrue(
            (destination / "Train_Sets/CT/1/DICOM_anon/a.dcm").exists()
        )
        self.assertTrue(
            (destination / "Test_Sets/CT/2/DICOM_anon/a.dcm").exists()
        )

    def test_resolve_metadata_dir(self):
        self.assertEqual(
            _resolve_metadata_dir(self.root, "demo"),
            self.root / "metadata" / "demo",
        )

    def test_unsupported_archive_type_is_rejected(self):
        with self.assertRaises(ValueError):
            _download_archives(
                [],
                self.root / "output",
                archive_type="rar",
                expected=(),
            )

    def test_unsafe_zip_member_is_rejected(self):
        archive = self.root / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("../outside.txt", "bad")
        with self.assertRaises(ValueError):
            _extract_zip(archive, self.root / "extract")


if __name__ == "__main__":
    unittest.main()

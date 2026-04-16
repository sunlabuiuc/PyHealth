"""
Unit tests for PH2Dataset and PH2MelanomaClassification.

Fixture layout (test-resources/core/ph2/):
    PH2_dataset.csv         — 5 images, original format: 2 common_nevus, 2 atypical_nevus, 1 melanoma
    PH2_Dataset_images/     — fake BMPs generated in setUpClass, deleted in tearDownClass
    PH2_simple_dataset.csv  — same 5 images, GitHub mirror format

Image IDs: IMD001–IMD005
BMP structure:
    PH2_Dataset_images/IMDXXX/IMDXXX_Dermoscopic_Image/IMDXXX.bmp
JPEG structure (mirror):
    images/IMDXXX.jpg
"""

import io
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from pyhealth.datasets import PH2Dataset
from pyhealth.tasks import PH2MelanomaClassification

_RESOURCES = Path(__file__).parent.parent.parent / "test-resources" / "core" / "ph2"

_IMAGE_IDS = ["IMD003", "IMD009", "IMD002", "IMD004", "IMD058"]
_DIAGNOSES = {
    "IMD003": "common_nevus",
    "IMD009": "common_nevus",
    "IMD002": "atypical_nevus",
    "IMD004": "atypical_nevus",
    "IMD058": "melanoma",
}


def _make_fake_bmp_images(root: Path) -> None:
    """Create nested BMP structure (original PH2 format)."""
    for img_id in _IMAGE_IDS:
        img_dir = root / "PH2_Dataset_images" / img_id / f"{img_id}_Dermoscopic_Image"
        img_dir.mkdir(parents=True, exist_ok=True)
        arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"{img_id}.bmp")


def _make_fake_jpg_images(root: Path) -> None:
    """Create flat JPEG structure (GitHub mirror format)."""
    (root / "images").mkdir(exist_ok=True)
    for img_id in _IMAGE_IDS:
        arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(root / "images" / f"{img_id}.jpg")


def _delete_bmp_images(root: Path) -> None:
    img_root = root / "PH2_Dataset_images"
    if img_root.exists():
        shutil.rmtree(img_root)


def _delete_jpg_images(root: Path) -> None:
    img_root = root / "images"
    if img_root.exists():
        shutil.rmtree(img_root)


def _make_mirror_zip(root: Path) -> str:
    """Build a fake GitHub mirror zip archive for download tests."""
    buf = io.BytesIO()
    prefix = "PH2-dataset-master/"
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(prefix, "")  # top-level dir entry
        zf.writestr(prefix + "images/", "")

        # Write simple CSV
        csv_lines = "image_name,diagnosis\n"
        for img_id, diag in _DIAGNOSES.items():
            label = {
                "common_nevus": "Common Nevus",
                "atypical_nevus": "Atypical Nevus",
                "melanoma": "Melanoma",
            }[diag]
            csv_lines += f"{img_id},{label}\n"
        zf.writestr(prefix + "PH2_simple_dataset.csv", csv_lines)

        # Write fake JPEGs
        for img_id in _IMAGE_IDS:
            arr = np.zeros((8, 8, 3), dtype=np.uint8)
            img_buf = io.BytesIO()
            Image.fromarray(arr).save(img_buf, format="JPEG")
            zf.writestr(prefix + f"images/{img_id}.jpg", img_buf.getvalue())

    zip_path = str(root / "ph2_mirror.zip")
    with open(zip_path, "wb") as f:
        f.write(buf.getvalue())
    return zip_path


class TestPH2Dataset(unittest.TestCase):
    """Tests for PH2Dataset loading and indexing (original BMP format)."""

    @classmethod
    def setUpClass(cls):
        cls.root = _RESOURCES
        _make_fake_bmp_images(cls.root)
        (cls.root / "ph2_metadata_pyhealth.csv").unlink(missing_ok=True)
        cls.cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.dataset = PH2Dataset(root=str(cls.root), cache_dir=cls.cache_dir.name)

    @classmethod
    def tearDownClass(cls):
        _delete_bmp_images(cls.root)
        (cls.root / "ph2_metadata_pyhealth.csv").unlink(missing_ok=True)
        (cls.root / "ph2-config-pyhealth.yaml").unlink(missing_ok=True)
        try:
            cls.cache_dir.cleanup()
        except Exception:
            pass

    def test_num_patients(self):
        self.assertEqual(len(self.dataset.unique_patient_ids), 5)

    def test_metadata_csv_created(self):
        self.assertTrue((self.root / "ph2_metadata_pyhealth.csv").exists())

    def test_default_task_is_classification(self):
        self.assertIsInstance(self.dataset.default_task, PH2MelanomaClassification)

    def test_event_has_path(self):
        pid = next(iter(self.dataset.unique_patient_ids))
        event = self.dataset.get_patient(pid).get_events(event_type="ph2")[0]
        self.assertTrue(os.path.isfile(event["path"]))

    def test_event_has_diagnosis(self):
        for img_id, expected in _DIAGNOSES.items():
            event = self.dataset.get_patient(img_id).get_events(event_type="ph2")[0]
            self.assertEqual(event["diagnosis"], expected)

    def test_image_paths_exist(self):
        for pid in self.dataset.unique_patient_ids:
            event = self.dataset.get_patient(pid).get_events(event_type="ph2")[0]
            self.assertTrue(os.path.isfile(event["path"]))

    def test_missing_root_raises(self):
        with self.assertRaises(FileNotFoundError):
            PH2Dataset(root="/nonexistent/path/xyz")

    def test_missing_source_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "images").mkdir()
            with self.assertRaises(FileNotFoundError):
                PH2Dataset(root=tmpdir)

    def test_missing_images_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.copy(_RESOURCES / "PH2_dataset.csv", tmpdir)
            with self.assertRaises(FileNotFoundError):
                PH2Dataset(root=tmpdir)


class TestPH2PrepareMetadata(unittest.TestCase):
    """Tests for PH2Dataset._prepare_metadata from raw CSV sources."""

    def test_prepare_metadata_from_original_csv(self):
        """_prepare_metadata reads PH2_dataset.csv (original format)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shutil.copy(_RESOURCES / "PH2_dataset.csv", root / "PH2_dataset.csv")
            _make_fake_bmp_images(root)

            obj = PH2Dataset.__new__(PH2Dataset)
            obj._prepare_metadata(str(root))

            out_csv = root / "ph2_metadata_pyhealth.csv"
            self.assertTrue(out_csv.exists())

            df = __import__("pandas").read_csv(out_csv)
            self.assertEqual(len(df), 5)
            self.assertIn("image_id", df.columns)
            self.assertIn("path", df.columns)
            self.assertIn("diagnosis", df.columns)
            self.assertCountEqual(
                df["diagnosis"].tolist(),
                ["common_nevus", "common_nevus", "atypical_nevus", "atypical_nevus", "melanoma"],
            )
            _delete_bmp_images(root)

    def test_prepare_metadata_from_simple_csv(self):
        """_prepare_metadata reads PH2_simple_dataset.csv (GitHub mirror format)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shutil.copy(_RESOURCES / "PH2_simple_dataset.csv", root / "PH2_simple_dataset.csv")
            _make_fake_jpg_images(root)

            obj = PH2Dataset.__new__(PH2Dataset)
            obj._prepare_metadata(str(root))

            out_csv = root / "ph2_metadata_pyhealth.csv"
            df = __import__("pandas").read_csv(out_csv)
            self.assertEqual(len(df), 5)
            self.assertCountEqual(
                df["diagnosis"].tolist(),
                ["common_nevus", "common_nevus", "atypical_nevus", "atypical_nevus", "melanoma"],
            )
            _delete_jpg_images(root)

    def test_missing_source_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            obj = PH2Dataset.__new__(PH2Dataset)
            with self.assertRaises(FileNotFoundError):
                obj._prepare_metadata(tmpdir)


class TestPH2Download(unittest.TestCase):
    """Tests for PH2Dataset download functionality."""

    def test_download_skipped_when_images_present(self):
        """_download should not call _download_file if images/ already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "images").mkdir()
            obj = PH2Dataset.__new__(PH2Dataset)
            with patch("pyhealth.datasets.ph2._download_file") as mock_dl:
                obj._download(tmpdir)
            mock_dl.assert_not_called()

    def test_download_fetches_and_extracts(self):
        """_download downloads zip and extracts images/ + PH2_simple_dataset.csv."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zip_path = _make_mirror_zip(root)

            obj = PH2Dataset.__new__(PH2Dataset)
            with patch("pyhealth.datasets.ph2._download_file",
                       side_effect=lambda url, dest, **kw: shutil.copy(zip_path, dest)):
                obj._download(tmpdir)

            self.assertTrue((root / "images").is_dir())
            self.assertTrue((root / "PH2_simple_dataset.csv").exists())
            # All 5 fake images extracted
            jpgs = list((root / "images").glob("*.jpg"))
            self.assertEqual(len(jpgs), 5)

    def test_download_true_loads_dataset(self):
        """download=True followed by dataset loading works end-to-end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zip_path = _make_mirror_zip(root)

            with patch("pyhealth.datasets.ph2._download_file",
                       side_effect=lambda url, dest, **kw: shutil.copy(zip_path, dest)):
                ds = PH2Dataset(root=tmpdir, download=True)

            self.assertEqual(len(ds.unique_patient_ids), 5)


class TestPH2MelanomaClassification(unittest.TestCase):
    """Tests for PH2MelanomaClassification task logic."""

    @classmethod
    def setUpClass(cls):
        _make_fake_bmp_images(_RESOURCES)
        (_RESOURCES / "ph2_metadata_pyhealth.csv").unlink(missing_ok=True)
        cls.cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.dataset = PH2Dataset(root=str(_RESOURCES), cache_dir=cls.cache_dir.name)
        cls.task = PH2MelanomaClassification()

    @classmethod
    def tearDownClass(cls):
        _delete_bmp_images(_RESOURCES)
        (_RESOURCES / "ph2_metadata_pyhealth.csv").unlink(missing_ok=True)
        (_RESOURCES / "ph2-config-pyhealth.yaml").unlink(missing_ok=True)
        try:
            cls.cache_dir.cleanup()
        except Exception:
            pass

    def test_task_name(self):
        self.assertEqual(self.task.task_name, "PH2MelanomaClassification")

    def test_output_schema_is_multiclass(self):
        self.assertEqual(self.task.output_schema["label"], "multiclass")

    def test_call_returns_sample_per_image(self):
        pid = next(iter(self.dataset.unique_patient_ids))
        patient = self.dataset.get_patient(pid)
        samples = self.task(patient)
        self.assertEqual(len(samples), 1)
        self.assertIn("image", samples[0])
        self.assertIn("label", samples[0])

    def test_correct_labels(self):
        for img_id, expected_diag in _DIAGNOSES.items():
            patient = self.dataset.get_patient(img_id)
            samples = self.task(patient)
            self.assertEqual(samples[0]["label"], expected_diag)

    def test_melanoma_count(self):
        mel_count = sum(
            1
            for pid in self.dataset.unique_patient_ids
            for s in self.task(self.dataset.get_patient(pid))
            if s["label"] == "melanoma"
        )
        self.assertEqual(mel_count, 1)


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for the ISIC2018Dataset and ISIC2018Classification classes.
"""
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

import requests

from pyhealth.datasets import ISIC2018Dataset
from pyhealth.tasks import ISIC2018Classification


class TestISIC2018Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = (
            Path(__file__).parent.parent.parent
            / "test-resources"
            / "core"
            / "isic2018"
        )
        cls.generate_fake_images()
        cls.cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.dataset = ISIC2018Dataset(cls.root, cache_dir=cls.cache_dir.name)
        from pyhealth.processors import ImageProcessor
        cls.samples = cls.dataset.set_task(
            input_processors={"image": ImageProcessor(mode="RGB")}
        )

    @classmethod
    def tearDownClass(cls):
        cls.samples.close()
        (cls.root / "isic2018-metadata-pyhealth.csv").unlink(missing_ok=True)
        cls.delete_fake_images()
        try:
            cls.cache_dir.cleanup()
        except Exception:
            pass
        cls.cache_dir = None

    @classmethod
    def generate_fake_images(cls):
        images_dir = cls.root / "ISIC2018_Task3_Training_Input"
        with open(cls.root / "ISIC2018_Task3_Training_GroundTruth.csv", "r") as f:
            lines = f.readlines()

        for line in lines[1:]:  # Skip header row
            image_id = line.split(",")[0].strip()
            img = Image.fromarray(
                np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            )
            img.save(images_dir / f"{image_id}.jpg")

    @classmethod
    def delete_fake_images(cls):
        for jpg in (cls.root / "ISIC2018_Task3_Training_Input").glob("*.jpg"):
            jpg.unlink()

    def test_stats(self):
        self.dataset.stats()

    def test_num_patients(self):
        # Each ISIC image is its own patient
        self.assertEqual(len(self.dataset.unique_patient_ids), 10)

    def test_default_task(self):
        self.assertIsInstance(self.dataset.default_task, ISIC2018Classification)

    def test_metadata_csv_created(self):
        self.assertTrue(
            (self.root / "isic2018-metadata-pyhealth.csv").exists()
        )

    def test_event_fields(self):
        # Patient ID equals image ID for ISIC images
        patient = self.dataset.get_patient("ISIC_0024307")
        events = patient.get_events()

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["image_id"], "ISIC_0024307")
        self.assertEqual(events[0]["mel"], "0.0")
        self.assertEqual(events[0]["nv"], "1.0")
        self.assertEqual(events[0]["bcc"], "0.0")
        self.assertEqual(events[0]["akiec"], "0.0")
        self.assertEqual(events[0]["bkl"], "0.0")
        self.assertEqual(events[0]["df"], "0.0")
        self.assertEqual(events[0]["vasc"], "0.0")

    def test_event_fields_mel(self):
        patient = self.dataset.get_patient("ISIC_0024310")
        events = patient.get_events()

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["mel"], "1.0")
        self.assertEqual(events[0]["nv"], "0.0")

    def test_all_label_columns_present(self):
        # All 7 class columns must be accessible on every event
        for cls in ISIC2018Dataset.classes:
            for pid in self.dataset.unique_patient_ids:
                event = self.dataset.get_patient(pid).get_events()[0]
                self.assertIn(cls, event)

    def test_image_paths_exist(self):
        for pid in self.dataset.unique_patient_ids:
            event = self.dataset.get_patient(pid).get_events()[0]
            self.assertTrue(os.path.isfile(event["path"]))

    def test_rgb_processor_produces_tensor(self):
        # RGB images should produce 3-channel tensors (C, H, W)
        sample = self.samples[0]
        self.assertEqual(sample["image"].shape[0], 3)

    def test_num_samples(self):
        self.assertEqual(len(self.samples), 10)

    def test_sample_labels(self):
        actual_labels = [sample["label"].item() for sample in self.samples]

        # Only 3 classes appear in fixture, sorted alphabetically: bkl=0, mel=1, nv=2
        # Real labels: NV, NV, NV, NV, MEL, NV, BKL, MEL, NV, MEL
        expected_labels = [2, 2, 2, 2, 1, 2, 0, 1, 2, 1]
        self.assertCountEqual(actual_labels, expected_labels)

    def test_mel_count(self):
        mel_samples = [s for s in self.samples if s["label"].item() == 1]
        self.assertEqual(len(mel_samples), 3)

    def test_nv_count(self):
        nv_samples = [s for s in self.samples if s["label"].item() == 2]
        self.assertEqual(len(nv_samples), 6)

    def test_verify_data_missing_root(self):
        with self.assertRaises(FileNotFoundError):
            ISIC2018Dataset(root="/nonexistent/path")

    def test_verify_data_missing_csv(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                ISIC2018Dataset(root=tmpdir)


class TestISIC2018Task12Dataset(unittest.TestCase):
    """Tests for ISIC2018Dataset with task='task1_2'."""

    @classmethod
    def setUpClass(cls):
        cls.root = (
            Path(__file__).parent.parent.parent
            / "test-resources"
            / "core"
            / "isic2018"
        )
        cls.images_dir = cls.root / "ISIC2018_Task1-2_Training_Input"
        cls.masks_dir = cls.root / "ISIC2018_Task1_Training_GroundTruth"
        cls._generated_images = []
        cls._generated_masks = []
        cls._generate_fake_data()
        cls.cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.dataset = ISIC2018Dataset(
            str(cls.root), task="task1_2", cache_dir=cls.cache_dir.name
        )

    @classmethod
    def tearDownClass(cls):
        for p in cls._generated_images + cls._generated_masks:
            p.unlink(missing_ok=True)
        for f in ["isic2018-task12-metadata-pyhealth.csv",
                  "isic2018-task12-config-pyhealth.yaml"]:
            (cls.root / f).unlink(missing_ok=True)
        try:
            cls.cache_dir.cleanup()
        except Exception:
            pass

    @classmethod
    def _generate_fake_data(cls):
        image_ids = [f"ISIC_002430{i}" for i in range(5)]
        for img_id in image_ids:
            img_path = cls.images_dir / f"{img_id}.jpg"
            Image.fromarray(
                np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            ).save(img_path)
            cls._generated_images.append(img_path)

            # Create a matching segmentation mask for every other image
            mask_path = cls.masks_dir / f"{img_id}_segmentation.png"
            Image.fromarray(
                np.random.randint(0, 2, (64, 64), dtype=np.uint8) * 255
            ).save(mask_path)
            cls._generated_masks.append(mask_path)

    # ── Construction ──────────────────────────────────────────

    def test_invalid_task_raises(self):
        with self.assertRaises(ValueError):
            ISIC2018Dataset(str(self.root), task="task99")

    def test_num_patients(self):
        self.assertEqual(len(self.dataset.unique_patient_ids), 5)

    def test_default_task_is_none(self):
        self.assertIsNone(self.dataset.default_task)

    # ── Config / metadata files ───────────────────────────────

    def test_metadata_csv_created(self):
        self.assertTrue((self.root / "isic2018-task12-metadata-pyhealth.csv").exists())

    def test_config_yaml_created(self):
        self.assertTrue((self.root / "isic2018-task12-config-pyhealth.yaml").exists())

    # ── Event attributes ──────────────────────────────────────

    def test_event_has_image_id(self):
        pid = sorted(self.dataset.unique_patient_ids)[0]
        event = self.dataset.get_patient(pid).get_events()[0]
        self.assertEqual(event["image_id"], pid)

    def test_event_has_path(self):
        for pid in self.dataset.unique_patient_ids:
            event = self.dataset.get_patient(pid).get_events()[0]
            self.assertTrue(os.path.isfile(event["path"]))

    def test_event_has_mask_path(self):
        # All 5 images have masks in our fixture
        for pid in self.dataset.unique_patient_ids:
            event = self.dataset.get_patient(pid).get_events()[0]
            self.assertIn("mask_path", event)
            self.assertIsNotNone(event["mask_path"])

    def test_event_missing_mask_path_is_none(self):
        # Temporarily remove one mask to simulate absence, re-index, check
        pid = sorted(self.dataset.unique_patient_ids)[0]
        mask_path = self.masks_dir / f"{pid}_segmentation.png"
        tmp_path = self.masks_dir / f"{pid}_segmentation.png.bak"
        mask_path.rename(tmp_path)
        try:
            with tempfile.TemporaryDirectory() as cache:
                ds = ISIC2018Dataset(str(self.root), task="task1_2", cache_dir=cache)
            event = ds.get_patient(pid).get_events()[0]
            self.assertIsNone(event["mask_path"])
        finally:
            tmp_path.rename(mask_path)

    # ── Validation ────────────────────────────────────────────

    def test_missing_mask_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy only images, no masks dir
            img_dir = Path(tmpdir) / "ISIC2018_Task1-2_Training_Input"
            img_dir.mkdir()
            Image.fromarray(
                np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            ).save(img_dir / "ISIC_0000001.jpg")
            with self.assertRaises(FileNotFoundError):
                ISIC2018Dataset(tmpdir, task="task1_2")


class TestISIC2018VerifyDataEdgeCases(unittest.TestCase):
    """Covers _verify_data raise paths not hit by the main test classes."""

    def test_task3_missing_csv_raises(self):
        """Image dir present but CSV absent → FileNotFoundError (line 267)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "ISIC2018_Task3_Training_Input").mkdir()
            with self.assertRaises(FileNotFoundError):
                ISIC2018Dataset(tmpdir, task="task3")

    def test_task3_empty_image_dir_raises(self):
        """Image dir present but no .jpg files → ValueError (line 273)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "ISIC2018_Task3_Training_Input").mkdir()
            (root / "ISIC2018_Task3_Training_GroundTruth.csv").write_text(
                "image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n"
            )
            with self.assertRaises(ValueError):
                ISIC2018Dataset(tmpdir, task="task3")

    def test_task12_empty_image_dir_raises(self):
        """Image dir present but no .jpg files → ValueError (line 322)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "ISIC2018_Task1-2_Training_Input").mkdir()
            (root / "ISIC2018_Task1_Training_GroundTruth").mkdir()
            with self.assertRaises(ValueError):
                ISIC2018Dataset(tmpdir, task="task1_2")


class TestISIC2018Download(unittest.TestCase):
    """Covers _download_file, _extract_zip, and the _download dispatch method."""

    # ── _download_file ────────────────────────────────────────

    def test_download_file_writes_content(self):
        from pyhealth.datasets.isic2018 import _download_file

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.headers = {"content-length": "5"}
        mock_resp.iter_content.return_value = [b"hello"]

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = str(Path(tmpdir) / "out.bin")
            with patch(
                "pyhealth.datasets.isic2018.requests.get", return_value=mock_resp
            ):
                _download_file("http://example.com/file", dest)
            self.assertEqual(Path(dest).read_bytes(), b"hello")

    def test_download_file_verifies_md5_checksum(self):
        from pyhealth.datasets.isic2018 import _download_file
        import hashlib

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.headers = {"content-length": "5"}
        mock_resp.iter_content.return_value = [b"hello"]

        # Correct MD5 for "hello"
        correct_md5 = hashlib.md5(b"hello").hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = str(Path(tmpdir) / "out.bin")
            with patch(
                "pyhealth.datasets.isic2018.requests.get", return_value=mock_resp
            ):
                _download_file(
                    "http://example.com/file", dest, expected_md5=correct_md5
                )
            self.assertEqual(Path(dest).read_bytes(), b"hello")

    def test_download_file_raises_on_md5_mismatch(self):
        from pyhealth.datasets.isic2018 import _download_file

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.headers = {"content-length": "5"}
        mock_resp.iter_content.return_value = [b"hello"]

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = str(Path(tmpdir) / "out.bin")
            with patch(
                "pyhealth.datasets.isic2018.requests.get", return_value=mock_resp
            ):
                with self.assertRaises(ValueError) as ctx:
                    _download_file(
                        "http://example.com/file",
                        dest,
                        expected_md5="wronghash123",
                    )
            self.assertIn("MD5 checksum mismatch", str(ctx.exception))
            self.assertFalse(Path(dest).exists())  # File should be removed

    def test_download_file_propagates_http_error(self):
        from pyhealth.datasets.isic2018 import _download_file

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: mock_resp
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.raise_for_status.side_effect = requests.HTTPError("404")

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "pyhealth.datasets.isic2018.requests.get", return_value=mock_resp
            ):
                with self.assertRaises(requests.HTTPError):
                    _download_file("http://example.com/bad", str(Path(tmpdir) / "f"))

    # ── _extract_zip ──────────────────────────────────────────

    def test_extract_zip_normal(self):
        from pyhealth.datasets.isic2018 import _extract_zip

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = str(Path(tmpdir) / "good.zip")
            with zipfile.ZipFile(zip_path, "w") as z:
                z.writestr("subdir/file.txt", "hello")
            _extract_zip(zip_path, tmpdir)
            self.assertTrue((Path(tmpdir) / "subdir" / "file.txt").exists())

    def test_extract_zip_path_traversal_raises(self):
        from pyhealth.datasets.isic2018 import _extract_zip

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = str(Path(tmpdir) / "evil.zip")
            with zipfile.ZipFile(zip_path, "w") as z:
                z.writestr("../evil.txt", "bad")
            with self.assertRaises(ValueError):
                _extract_zip(zip_path, tmpdir)

    # ── _download (skip when already present) ─────────────────

    def test_download_task3_skipped_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "ISIC2018_Task3_Training_Input").mkdir()
            (root / "ISIC2018_Task3_Training_GroundTruth.csv").write_text("x")
            ds = ISIC2018Dataset.__new__(ISIC2018Dataset)
            ds.task = "task3"
            ds._image_dir = str(root / "ISIC2018_Task3_Training_Input")
            ds._label_path = str(root / "ISIC2018_Task3_Training_GroundTruth.csv")
            with patch("pyhealth.datasets.isic2018._download_file") as mock_dl:
                ds._download(str(root))
            mock_dl.assert_not_called()

    def test_download_task3_skipped_when_zip_present(self):
        """If ZIP exists but not extracted, skip download and proceed to extract."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Create fake ZIPs but don't extract
            (root / "ISIC2018_Task3_Training_GroundTruth.zip").write_text("fake")
            (root / "ISIC2018_Task3_Training_Input.zip").write_text("fake")

            ds = ISIC2018Dataset.__new__(ISIC2018Dataset)
            ds.task = "task3"
            ds._image_dir = str(root / "ISIC2018_Task3_Training_Input")
            ds._label_path = str(root / "ISIC2018_Task3_Training_GroundTruth.csv")

            with patch("pyhealth.datasets.isic2018._download_file") as mock_dl, \
                 patch("pyhealth.datasets.isic2018._extract_zip"):
                ds._download(str(root))
            # Should not call download if ZIP already exists
            mock_dl.assert_not_called()

    def test_download_task12_skipped_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "ISIC2018_Task1-2_Training_Input").mkdir()
            (root / "ISIC2018_Task1_Training_GroundTruth").mkdir()
            ds = ISIC2018Dataset.__new__(ISIC2018Dataset)
            ds.task = "task1_2"
            ds._image_dir = str(root / "ISIC2018_Task1-2_Training_Input")
            ds._mask_dir = str(root / "ISIC2018_Task1_Training_GroundTruth")
            with patch("pyhealth.datasets.isic2018._download_file") as mock_dl:
                ds._download(str(root))
            mock_dl.assert_not_called()

    # ── _download (fetch when missing) ────────────────────────

    def test_download_task3_fetches_both_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ds = ISIC2018Dataset.__new__(ISIC2018Dataset)
            ds.task = "task3"
            ds._image_dir = str(root / "ISIC2018_Task3_Training_Input")
            ds._label_path = str(root / "ISIC2018_Task3_Training_GroundTruth.csv")
            with patch("pyhealth.datasets.isic2018._download_file") as mock_dl, \
                 patch("pyhealth.datasets.isic2018._extract_zip"), \
                 patch("pyhealth.datasets.isic2018.os.remove"):
                ds._download(str(root))
            self.assertEqual(mock_dl.call_count, 2)  # labels zip + images zip

    def test_download_task3_skips_labels_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "ISIC2018_Task3_Training_GroundTruth.csv").write_text("x")
            ds = ISIC2018Dataset.__new__(ISIC2018Dataset)
            ds.task = "task3"
            ds._image_dir = str(root / "ISIC2018_Task3_Training_Input")
            ds._label_path = str(root / "ISIC2018_Task3_Training_GroundTruth.csv")
            with patch("pyhealth.datasets.isic2018._download_file") as mock_dl, \
                 patch("pyhealth.datasets.isic2018._extract_zip"), \
                 patch("pyhealth.datasets.isic2018.os.remove"):
                ds._download(str(root))
            self.assertEqual(mock_dl.call_count, 1)  # only images zip

    def test_download_task12_fetches_both_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ds = ISIC2018Dataset.__new__(ISIC2018Dataset)
            ds.task = "task1_2"
            ds._image_dir = str(root / "ISIC2018_Task1-2_Training_Input")
            ds._mask_dir = str(root / "ISIC2018_Task1_Training_GroundTruth")
            with patch("pyhealth.datasets.isic2018._download_file") as mock_dl, \
                 patch("pyhealth.datasets.isic2018._extract_zip"), \
                 patch("pyhealth.datasets.isic2018.os.remove"):
                ds._download(str(root))
            self.assertEqual(mock_dl.call_count, 2)  # images zip + masks zip

    def test_download_task12_skips_images_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "ISIC2018_Task1-2_Training_Input").mkdir()
            ds = ISIC2018Dataset.__new__(ISIC2018Dataset)
            ds.task = "task1_2"
            ds._image_dir = str(root / "ISIC2018_Task1-2_Training_Input")
            ds._mask_dir = str(root / "ISIC2018_Task1_Training_GroundTruth")
            with patch("pyhealth.datasets.isic2018._download_file") as mock_dl, \
                 patch("pyhealth.datasets.isic2018._extract_zip"), \
                 patch("pyhealth.datasets.isic2018.os.remove"):
                ds._download(str(root))
            self.assertEqual(mock_dl.call_count, 1)  # only masks zip


if __name__ == "__main__":
    unittest.main()

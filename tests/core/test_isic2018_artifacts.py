"""
Unit tests for ISIC2018ArtifactsDataset and associated task classes.

Fixture layout (test-resources/core/isic2018_artifacts/):
    isic_bias.csv  — 8 images, 4 melanoma / 4 non-melanoma, semicolon-delimited
    images/        — fake PNGs generated in setUpClass, deleted in tearDownClass
    masks/         — fake segmentation PNGs generated in setUpClass

Fixture summary
---------------
image             label  ruler  hair  dark_corner  gel_border  gel_bubble  ink  patches
ISIC_0024306.png    0      1     0        0            0           0        0       0
ISIC_0024307.png    1      0     1        0            0           0        0       0
ISIC_0024308.png    0      0     0        1            0           0        0       0
ISIC_0024309.png    1      1     0        0            0           1        1       0
ISIC_0024310.png    0      0     0        0            0           0        1       0
ISIC_0024311.png    1      0     0        0            0           0        0       0
ISIC_0024312.png    0      0     0        0            1           0        0       0
ISIC_0024313.png    1      0     0        0            0           0        0       1
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from pyhealth.datasets import ISIC2018ArtifactsDataset
from pyhealth.datasets.isic2018_artifacts import ARTIFACT_LABELS
from pyhealth.tasks import (
    ISICArtifactClassification,
    ISICMelanomaClassification,
    ISICTrapSetMelanomaClassification,
)

_RESOURCES = (
    Path(__file__).parent.parent.parent
    / "test-resources"
    / "core"
    / "isic2018_artifacts"
)

_IMAGE_NAMES = [f"ISIC_{24306 + i:07d}.png" for i in range(8)]


def _make_fake_images(image_dir: Path) -> None:
    image_dir.mkdir(exist_ok=True)
    for name in _IMAGE_NAMES:
        arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(image_dir / name)


def _make_fake_masks(mask_dir: Path) -> None:
    mask_dir.mkdir(exist_ok=True)
    for name in _IMAGE_NAMES:
        stem = Path(name).stem
        arr = np.zeros((64, 64), dtype=np.uint8)
        arr[16:48, 16:48] = 255
        Image.fromarray(arr).save(mask_dir / f"{stem}_segmentation.png")


def _delete_fake_images(image_dir: Path) -> None:
    for png in image_dir.glob("*.png"):
        png.unlink()


def _delete_fake_masks(mask_dir: Path) -> None:
    for png in mask_dir.glob("*.png"):
        png.unlink()


class TestISIC2018ArtifactsDataset(unittest.TestCase):
    """Tests for ISIC2018ArtifactsDataset loading, indexing, and validation."""

    @classmethod
    def setUpClass(cls):
        cls.root = _RESOURCES
        _make_fake_images(cls.root / "images")
        _make_fake_masks(cls.root / "masks")
        cls.cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.dataset = ISIC2018ArtifactsDataset(
            root=str(cls.root),
            image_dir="images",
            mask_dir="masks",
            mode="whole",
            cache_dir=cls.cache_dir.name,
        )

    @classmethod
    def tearDownClass(cls):
        _delete_fake_images(cls.root / "images")
        _delete_fake_masks(cls.root / "masks")
        (cls.root / "isic-artifact-metadata-pyhealth.csv").unlink(missing_ok=True)
        (cls.root / "isic-artifact-config-pyhealth.yaml").unlink(missing_ok=True)
        try:
            cls.cache_dir.cleanup()
        except Exception:
            pass

    # ── Dataset-level ────────────────────────────────────────────────────────

    def test_stats(self):
        self.dataset.stats()

    def test_num_patients(self):
        # One patient per image
        self.assertEqual(len(self.dataset.unique_patient_ids), 8)

    def test_default_task_is_none(self):
        self.assertIsNone(self.dataset.default_task)

    def test_metadata_csv_created(self):
        self.assertTrue((self.root / "isic-artifact-metadata-pyhealth.csv").exists())

    def test_config_yaml_created(self):
        self.assertTrue((self.root / "isic-artifact-config-pyhealth.yaml").exists())

    def test_artifact_labels_class_attribute(self):
        self.assertEqual(ISIC2018ArtifactsDataset.artifact_labels, ARTIFACT_LABELS)

    # ── Event field access ───────────────────────────────────────────────────

    def test_event_fields_ruler_image(self):
        events = self.dataset.get_patient("ISIC_0024306").get_events(
            event_type="isic_artifacts"
        )
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["image_id"], "ISIC_0024306")
        self.assertEqual(event["ruler"], "1")
        self.assertEqual(event["label"], "0")

    def test_event_fields_test_split(self):
        event = self.dataset.get_patient("ISIC_0024311").get_events()[0]
        self.assertEqual(event["label"], "1")

    def test_all_artifact_columns_present_on_every_patient(self):
        for pid in self.dataset.unique_patient_ids:
            event = self.dataset.get_patient(pid).get_events()[0]
            for col in ARTIFACT_LABELS:
                self.assertIn(col, event, f"Missing '{col}' for patient {pid}")

    def test_image_paths_exist(self):
        import os
        for pid in self.dataset.unique_patient_ids:
            event = self.dataset.get_patient(pid).get_events()[0]
            self.assertTrue(os.path.isfile(event["path"]))

    # ── Validation errors ────────────────────────────────────────────────────

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            ISIC2018ArtifactsDataset(root=str(self.root), mode="not_a_mode")
        with self.assertRaises(FileNotFoundError):
            ISIC2018ArtifactsDataset(root="/nonexistent/path/xyz")

    def test_missing_csv_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "images").mkdir()
            Path(tmpdir, "masks").mkdir()
            with self.assertRaises(FileNotFoundError):
                ISIC2018ArtifactsDataset(root=tmpdir)

    def test_missing_image_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import shutil
            shutil.copy(self.root / "isic_bias.csv", tmpdir)
            Path(tmpdir, "masks").mkdir()
            with self.assertRaises(FileNotFoundError):
                ISIC2018ArtifactsDataset(root=tmpdir, image_dir="nonexistent")

    def test_missing_mask_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import shutil
            shutil.copy(self.root / "isic_bias.csv", tmpdir)
            Path(tmpdir, "images").mkdir()
            with self.assertRaises(FileNotFoundError):
                ISIC2018ArtifactsDataset(root=tmpdir, mask_dir="nonexistent")

    def test_download_raises_for_custom_csv(self):
        with self.assertRaises(ValueError):
            ISIC2018ArtifactsDataset(
                root=str(self.root),
                annotations_csv="custom.csv",
                download=True,
            )

    # ── Download behaviour ───────────────────────────────────────────────────

    def test_download_skipped_when_csv_already_present(self):
        """_download_bias_csv should not call requests.get if CSV exists."""
        with patch("pyhealth.datasets.isic2018_artifacts.requests.get") as mock_get:
            self.dataset._download_bias_csv(str(self.root))
            mock_get.assert_not_called()

    def test_download_fetches_csv_when_absent(self):
        """_download_bias_csv should write the CSV when it is missing."""
        from pyhealth.datasets.isic2018_artifacts import _BIAS_CSV

        csv_bytes = (self.root / "isic_bias.csv").read_bytes()
        mock_resp = MagicMock()
        mock_resp.content = csv_bytes
        mock_resp.raise_for_status = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Instantiate without calling __init__ to test the method in isolation
            obj = ISIC2018ArtifactsDataset.__new__(ISIC2018ArtifactsDataset)
            obj.annotations_csv = _BIAS_CSV
            obj._bias_csv_path = str(Path(tmpdir) / "isic_bias.csv")

            with patch(
                "pyhealth.datasets.isic2018_artifacts.requests.get",
                return_value=mock_resp,
            ):
                obj._download_bias_csv(tmpdir)

            mock_resp.raise_for_status.assert_called_once()
            self.assertTrue(Path(tmpdir, "isic_bias.csv").exists())


    def test_download_images_skipped_when_dirs_exist(self):
        """_download_images should not call requests.get if extracted dirs exist."""
        from pyhealth.datasets.isic2018_artifacts import _IMAGES_DIR, _MASKS_DIR

        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, _IMAGES_DIR))
            os.makedirs(os.path.join(tmpdir, _MASKS_DIR))
            obj = ISIC2018ArtifactsDataset.__new__(ISIC2018ArtifactsDataset)
            with patch("pyhealth.datasets.isic2018_artifacts._download_file") as mock_dl:
                obj._download_images(tmpdir)
                mock_dl.assert_not_called()


class TestISICArtifactTasks(unittest.TestCase):
    """Tests for ISICMelanomaClassification, ISICArtifactClassification,
    and ISICTrapSetMelanomaClassification against the fixture dataset."""

    @classmethod
    def setUpClass(cls):
        _make_fake_images(_RESOURCES / "images")
        _make_fake_masks(_RESOURCES / "masks")
        cls.cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.dataset = ISIC2018ArtifactsDataset(
            root=str(_RESOURCES),
            image_dir="images",
            mask_dir="masks",
            mode="whole",
            cache_dir=cls.cache_dir.name,
        )
        cls.mel_all = cls.dataset.set_task(ISICMelanomaClassification())
        cls.artifact_ruler = cls.dataset.set_task(
            ISICArtifactClassification(artifact="ruler")
        )
        cls.artifact_hair = cls.dataset.set_task(
            ISICArtifactClassification(artifact="hair")
        )

    @classmethod
    def tearDownClass(cls):
        _delete_fake_images(_RESOURCES / "images")
        _delete_fake_masks(_RESOURCES / "masks")
        (_RESOURCES / "isic-artifact-metadata-pyhealth.csv").unlink(missing_ok=True)
        (_RESOURCES / "isic-artifact-config-pyhealth.yaml").unlink(missing_ok=True)
        for ds in [
            cls.mel_all,
            cls.artifact_ruler, cls.artifact_hair,
        ]:
            try:
                ds.close()
            except Exception:
                pass
        try:
            cls.cache_dir.cleanup()
        except Exception:
            pass

    # ── ISICMelanomaClassification ───────────────────────────────────────────

    def test_mel_all_count(self):
        self.assertEqual(len(self.mel_all), 8)

    def test_mel_all_melanoma_count(self):
        mel_count = sum(s["label"].item() for s in self.mel_all)
        self.assertEqual(mel_count, 4)

    def test_mel_sample_has_image_tensor(self):
        sample = self.mel_all[0]
        self.assertIn("image", sample)
        self.assertEqual(sample["image"].shape[0], 3)  # (C, H, W)

    def test_mel_invalid_split_raises(self):
        with self.assertRaises(ValueError):
            ISICMelanomaClassification(split=6)

    def test_mel_invalid_split_subset_raises(self):
        with self.assertRaises(ValueError):
            ISICMelanomaClassification(split_subset="val")

    # ── ISICArtifactClassification ───────────────────────────────────────────

    def test_artifact_ruler_total_count(self):
        self.assertEqual(len(self.artifact_ruler), 8)

    def test_artifact_ruler_positive_count(self):
        # ISIC_0024306 and ISIC_0024309 have ruler=1
        positives = sum(s["label"].item() for s in self.artifact_ruler)
        self.assertEqual(positives, 2)

    def test_artifact_hair_positive_count(self):
        # Only ISIC_0024307 has hair=1
        positives = sum(s["label"].item() for s in self.artifact_hair)
        self.assertEqual(positives, 1)

    def test_artifact_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            ISICArtifactClassification(artifact="stethoscope")

    def test_artifact_invalid_split_raises(self):
        with self.assertRaises(ValueError):
            ISICArtifactClassification(split=0)

    def test_artifact_invalid_split_subset_raises(self):
        with self.assertRaises(ValueError):
            ISICArtifactClassification(split_subset="val")

    # ── ISICTrapSetMelanomaClassification ────────────────────────────────────

    def test_trapset_invalid_artifact_raises(self):
        with self.assertRaises(ValueError):
            ISICTrapSetMelanomaClassification(artifact="hair")

    def test_trapset_invalid_prevalence_raises(self):
        with self.assertRaises(ValueError):
            ISICTrapSetMelanomaClassification(prevalence=10)

    def test_trapset_invalid_split_subset_raises(self):
        with self.assertRaises(ValueError):
            ISICTrapSetMelanomaClassification(split_subset="val")

    def test_trapset_repr(self):
        task = ISICTrapSetMelanomaClassification(
            artifact="ruler", prevalence=25, split_subset="train"
        )
        r = repr(task)
        self.assertIn("ruler", r)
        self.assertIn("25", r)
        self.assertIn("train", r)


class TestISIC2018ArtifactsDownload(unittest.TestCase):
    """Covers _download_bias_csv fetch, _download_images branches, and
    the no-matching-images ValueError — the remaining coverage gaps."""

    # ── _download_bias_csv fetch ──────────────────────────────────────────────

    def test_download_bias_csv_fetches_when_absent(self):
        from pyhealth.datasets.isic2018_artifacts import _BIAS_CSV

        csv_bytes = (_RESOURCES / "isic_bias.csv").read_bytes()
        mock_resp = MagicMock()
        mock_resp.content = csv_bytes
        mock_resp.raise_for_status = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            obj = ISIC2018ArtifactsDataset.__new__(ISIC2018ArtifactsDataset)
            obj.annotations_csv = _BIAS_CSV
            obj._bias_csv_path = str(Path(tmpdir) / _BIAS_CSV)

            with patch("pyhealth.datasets.isic2018_artifacts.requests.get",
                       return_value=mock_resp):
                obj._download_bias_csv(tmpdir)

            mock_resp.raise_for_status.assert_called_once()
            self.assertTrue(Path(tmpdir, _BIAS_CSV).exists())

    # ── _download_images branches ─────────────────────────────────────────────

    def test_download_images_fetches_when_dirs_absent(self):
        """Both image and mask dirs missing → _download_file called twice."""

        with tempfile.TemporaryDirectory() as tmpdir:
            obj = ISIC2018ArtifactsDataset.__new__(ISIC2018ArtifactsDataset)
            with patch("pyhealth.datasets.isic2018_artifacts._download_file") as mock_dl, \
                 patch("pyhealth.datasets.isic2018_artifacts._extract_zip"), \
                 patch("pyhealth.datasets.isic2018_artifacts.os.remove"):
                obj._download_images(tmpdir)
            self.assertEqual(mock_dl.call_count, 2)

    def test_download_images_skips_images_when_present(self):
        """Image dir already present → only mask zip fetched."""
        from pyhealth.datasets.isic2018_artifacts import _IMAGES_DIR

        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, _IMAGES_DIR))
            obj = ISIC2018ArtifactsDataset.__new__(ISIC2018ArtifactsDataset)
            with patch("pyhealth.datasets.isic2018_artifacts._download_file") as mock_dl, \
                 patch("pyhealth.datasets.isic2018_artifacts._extract_zip"), \
                 patch("pyhealth.datasets.isic2018_artifacts.os.remove"):
                obj._download_images(tmpdir)
            self.assertEqual(mock_dl.call_count, 1)

    def test_download_images_skips_masks_when_present(self):
        """Mask dir already present → only image zip fetched."""
        from pyhealth.datasets.isic2018_artifacts import _MASKS_DIR

        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, _MASKS_DIR))
            obj = ISIC2018ArtifactsDataset.__new__(ISIC2018ArtifactsDataset)
            with patch("pyhealth.datasets.isic2018_artifacts._download_file") as mock_dl, \
                 patch("pyhealth.datasets.isic2018_artifacts._extract_zip"), \
                 patch("pyhealth.datasets.isic2018_artifacts.os.remove"):
                obj._download_images(tmpdir)
            self.assertEqual(mock_dl.call_count, 1)

    # ── no-matching-images ValueError ─────────────────────────────────────────

    def test_no_matching_images_raises(self):
        """CSV present but no image files match → ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            # CSV references images that don't exist in the images dir
            (_RESOURCES / "isic_bias.csv").read_bytes()
            import shutil
            shutil.copy(_RESOURCES / "isic_bias.csv", root / "isic_bias.csv")
            with self.assertRaises(ValueError):
                ISIC2018ArtifactsDataset(
                    root=str(root),
                    image_dir="images",
                    mask_dir="masks",
                )

    # ── download=True constructor path ────────────────────────────────────────

    def test_constructor_download_true_calls_both_downloads(self):
        """download=True triggers both _download_bias_csv and _download_images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(ISIC2018ArtifactsDataset, "_download_bias_csv") as mock_csv, \
                 patch.object(ISIC2018ArtifactsDataset, "_download_images") as mock_img, \
                 patch.object(ISIC2018ArtifactsDataset, "_verify_data"), \
                 patch.object(ISIC2018ArtifactsDataset, "_index_data", return_value=None), \
                 patch("pyhealth.datasets.base_dataset.BaseDataset.__init__"):
                obj = ISIC2018ArtifactsDataset.__new__(ISIC2018ArtifactsDataset)
                obj.mode = "whole"
                obj.annotations_csv = "isic_bias.csv"
                obj._image_dir = tmpdir
                obj._mask_dir = tmpdir
                obj._bias_csv_path = str(Path(tmpdir) / "isic_bias.csv")
                # Call __init__ manually with download=True
                ISIC2018ArtifactsDataset.__init__(obj, root=tmpdir, download=True)
            mock_csv.assert_called_once_with(tmpdir)
            mock_img.assert_called_once_with(tmpdir)


if __name__ == "__main__":
    unittest.main()

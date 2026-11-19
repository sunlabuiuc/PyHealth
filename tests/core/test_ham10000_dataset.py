"""Tests for HAM10000Dataset using a tiny synthetic ISIC 2018 layout."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from PIL import Image

from pyhealth.datasets import HAM10000Dataset

_CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

# image, diagnosis, lesion
_TRAIN = [
    ("ISIC_0000001", "MEL", "HAM_1"),
    ("ISIC_0000002", "MEL", "HAM_1"),
    ("ISIC_0000003", "NV", "HAM_2"),
    ("ISIC_0000004", "BCC", "HAM_3"),
    ("ISIC_0000005", "BKL", "HAM_4"),
    ("ISIC_0000006", "DF", "HAM_5"),
    ("ISIC_0000007", "VASC", "HAM_6"),
    ("ISIC_0000008", "AKIEC", "HAM_7"),
    ("ISIC_0000009", "NV", "HAM_8"),  # no JPEG on disk
]
_VAL = [("ISIC_0000101", "NV"), ("ISIC_0000102", "MEL")]
_TEST = [("ISIC_0000201", "BCC")]


def _write_split(root: Path, name: str, rows) -> None:
    image_dir = root / f"ISIC2018_Task3_{name}_Input"
    gt_dir = root / f"ISIC2018_Task3_{name}_GroundTruth"
    image_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    records = []
    for i, row in enumerate(rows):
        image, dx = row[0], row[1]
        records.append({"image": image, **{c: float(c == dx) for c in _CLASSES}})
        if image != "ISIC_0000009":
            Image.new("RGB", (12, 9), (i * 25, 90, 150)).save(
                image_dir / f"{image}.jpg"
            )
    pd.DataFrame(records).to_csv(gt_dir / f"{gt_dir.name}.csv", index=False)


def make_ham10000_root(root: Path) -> None:
    """Write tiny training, validation and test splits plus lesion ids."""
    _write_split(root, "Training", _TRAIN)
    _write_split(root, "Validation", _VAL)
    _write_split(root, "Test", _TEST)
    pd.DataFrame(
        {
            "image": [r[0] for r in _TRAIN],
            "lesion_id": [r[2] for r in _TRAIN],
            "diagnosis_confirm_type": "histopathology",
        }
    ).to_csv(root / "ISIC2018_Task3_Training_LesionGroupings.csv", index=False)


class TestHAM10000Dataset(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "isic2018"
        make_ham10000_root(self.root)
        self.cache = self.tmp / "cache"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _dataset(self, **kwargs):
        return HAM10000Dataset(root=str(self.root), cache_dir=str(self.cache), **kwargs)

    def test_training_metadata_grouped_by_lesion(self):
        dataset = self._dataset()
        self.assertFalse((self.root / "ham10000-metadata-pyhealth.csv").exists())
        meta = pd.read_csv(dataset.metadata_path)
        self.assertEqual(len(meta), 8)  # ISIC_0000009 has no JPEG
        self.assertEqual(set(meta["split"]), {"train"})
        self.assertEqual(len(dataset.unique_patient_ids), 7)
        events = dataset.get_patient("HAM_1").get_events(event_type="ham10000")
        self.assertEqual(sorted(e.diagnosis for e in events), ["MEL", "MEL"])

    def test_official_splits(self):
        dataset = self._dataset(splits=("train", "val", "test"))
        meta = pd.read_csv(dataset.metadata_path)
        self.assertEqual(
            meta["split"].value_counts().to_dict(), {"train": 8, "val": 2, "test": 1}
        )
        # holdout images have no lesion id, so each is its own patient
        holdout = meta[meta["split"] != "train"]
        self.assertEqual(list(holdout["patient_id"]), list(holdout["image_id"]))

    def test_dataverse_metadata_fallback(self):
        groupings = self.root / "ISIC2018_Task3_Training_LesionGroupings.csv"
        pd.read_csv(groupings).rename(
            columns={"image": "image_id", "diagnosis_confirm_type": "dx_type"}
        ).to_csv(self.root / "HAM10000_metadata.csv", index=False)
        groupings.unlink()
        dataset = self._dataset()
        self.assertIn("HAM_1", dataset.unique_patient_ids)

    def test_missing_lesion_ids_raises(self):
        (self.root / "ISIC2018_Task3_Training_LesionGroupings.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            self._dataset()

    def test_bad_split_raises(self):
        with self.assertRaises(ValueError):
            self._dataset(splits=("holdout",))

    def test_rebuilds_metadata_when_splits_change(self):
        metadata_dir = str(self.tmp / "meta")
        first = self._dataset(metadata_dir=metadata_dir)
        self.assertEqual(set(pd.read_csv(first.metadata_path)["split"]), {"train"})
        second = self._dataset(metadata_dir=metadata_dir, splits=("train", "val"))
        meta = pd.read_csv(second.metadata_path)
        self.assertEqual(meta["split"].value_counts().to_dict(), {"train": 8, "val": 2})
        self.assertIn("ISIC_0000101", second.unique_patient_ids)

    def test_rebuilds_metadata_when_root_changes(self):
        metadata_dir = str(self.tmp / "meta")
        self._dataset(metadata_dir=metadata_dir)
        other = self.tmp / "other"
        shutil.copytree(self.root, other)
        (other / "ISIC2018_Task3_Training_Input" / "ISIC_0000001.jpg").unlink()
        dataset = HAM10000Dataset(
            root=str(other), cache_dir=str(self.cache), metadata_dir=metadata_dir
        )
        self.assertEqual(len(pd.read_csv(dataset.metadata_path)), 7)

    def test_refresh_metadata_rewrites_csv(self):
        metadata_dir = str(self.tmp / "meta")
        dataset = self._dataset(metadata_dir=metadata_dir)
        path = dataset.metadata_path
        pd.read_csv(path).head(1).to_csv(path, index=False)
        self.assertEqual(
            len(self._dataset(metadata_dir=metadata_dir).unique_patient_ids), 1
        )
        dataset = self._dataset(metadata_dir=metadata_dir, refresh_metadata=True)
        self.assertEqual(len(pd.read_csv(path)), 8)


class TestHAM10000Download(unittest.TestCase):
    """download=True, with the network calls replaced by local copies."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.source = self.tmp / "source"
        make_ham10000_root(self.source)
        self.root = self.tmp / "isic2018"
        self.cache = self.tmp / "cache"
        self.urls = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fake_download_file(self, url, path):
        self.urls.append(url)
        shutil.copy(self.source / url.rsplit("/", 1)[1], path)

    def _fake_download_archives(self, urls, destination, **kwargs):
        self.urls.extend(urls)
        self.assertTrue(kwargs["extract_to_parent"])
        shutil.copytree(self.source / Path(destination).name, destination)

    def test_download_fetches_missing_files_into_root(self):
        with (
            mock.patch(
                "pyhealth.datasets.ham10000.download_file",
                side_effect=self._fake_download_file,
            ),
            mock.patch(
                "pyhealth.datasets.ham10000._download_archives",
                side_effect=self._fake_download_archives,
            ),
        ):
            dataset = HAM10000Dataset(
                root=str(self.root),
                splits=("train", "test"),
                download=True,
                cache_dir=str(self.cache),
            )
        base = "https://isic-archive.s3.amazonaws.com/challenges/2018/"
        self.assertEqual(
            sorted(self.urls),
            sorted(
                [base + "ISIC2018_Task3_Training_LesionGroupings.csv"]
                + [
                    f"{base}ISIC2018_Task3_{split}_{kind}.zip"
                    for split in ("Training", "Test")
                    for kind in ("Input", "GroundTruth")
                ]
            ),
        )
        self.assertTrue(
            (self.root / "ISIC2018_Task3_Training_LesionGroupings.csv").exists()
        )
        self.assertEqual(len(pd.read_csv(dataset.metadata_path)), 9)

    def test_html_groupings_response_is_rejected(self):
        def _html(url, path):
            Path(path).write_text("<!DOCTYPE html><html>moved</html>\n")

        with (
            mock.patch(
                "pyhealth.datasets.ham10000._download_archives",
                side_effect=self._fake_download_archives,
            ),
            mock.patch("pyhealth.datasets.ham10000.download_file", side_effect=_html),
            self.assertRaisesRegex(RuntimeError, "not the expected CSV"),
        ):
            HAM10000Dataset(
                root=str(self.root), download=True, cache_dir=str(self.cache)
            )
        self.assertFalse(
            (self.root / "ISIC2018_Task3_Training_LesionGroupings.csv").exists()
        )


if __name__ == "__main__":
    unittest.main()

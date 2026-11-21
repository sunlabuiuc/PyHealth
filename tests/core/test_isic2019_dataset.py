"""Tests for ISIC2019Dataset using a tiny synthetic ISIC 2019 layout."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from PIL import Image

from pyhealth.datasets import ISIC2019Dataset

_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

# image, diagnosis, lesion id (None: no lesion id)
_TRAIN = [
    ("ISIC_0000001", "MEL", "BCN_1"),
    ("ISIC_0000002", "MEL", "BCN_1"),
    ("ISIC_0000003", "NV", "HAM_1"),
    ("ISIC_0000004", "BCC", "HAM_2"),
    ("ISIC_0000005", "AK", "MSK4_1"),
    ("ISIC_0000006_downsampled", "BKL", None),
    ("ISIC_0000007", "DF", "BCN_2"),
    ("ISIC_0000008", "VASC", "BCN_3"),
    ("ISIC_0000009", "SCC", "BCN_4"),
    ("ISIC_0000010", "NV", "BCN_5"),  # no JPEG on disk
]
# image, diagnosis; ISIC_0034321..ISIC_0036064 are ISIC 2018 holdout images
_TEST = [
    ("ISIC_0034400", "NV"),
    ("ISIC_0036064", "MEL"),
    ("ISIC_0060000", "SCC"),
    ("ISIC_0060001", "UNK"),
]


def _write_split(root: Path, name: str, rows, with_lesions: bool) -> None:
    image_dir = root / f"ISIC_2019_{name}_Input"
    image_dir.mkdir(parents=True)
    gt, meta = [], []
    for i, row in enumerate(rows):
        image, dx = row[0], row[1]
        gt.append({"image": image, **{c: float(c == dx) for c in _CLASSES}})
        record = {
            "image": image,
            "age_approx": 50,
            "anatom_site_general": "anterior torso",
            "sex": "male",
        }
        if with_lesions:
            record["lesion_id"] = row[2]
        meta.append(record)
        if image != "ISIC_0000010":
            Image.new("RGB", (10, 10), (i * 20, 60, 200)).save(
                image_dir / f"{image}.jpg"
            )
    gt = pd.DataFrame(gt)
    if name == "Test":
        gt["score_weight"] = 1.0
    gt.to_csv(root / f"ISIC_2019_{name}_GroundTruth.csv", index=False)
    pd.DataFrame(meta).to_csv(root / f"ISIC_2019_{name}_Metadata.csv", index=False)


def make_isic2019_root(root: Path) -> None:
    """Write tiny training and test splits with CSVs and JPEGs."""
    _write_split(root, "Training", _TRAIN, with_lesions=True)
    _write_split(root, "Test", _TEST, with_lesions=False)


class TestISIC2019Dataset(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "isic2019"
        make_isic2019_root(self.root)
        self.cache = self.tmp / "cache"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _dataset(self, **kwargs):
        return ISIC2019Dataset(root=str(self.root), cache_dir=str(self.cache), **kwargs)

    def test_training_metadata(self):
        dataset = self._dataset()
        self.assertFalse((self.root / "isic2019-metadata-pyhealth.csv").exists())
        meta = pd.read_csv(dataset.metadata_path)
        self.assertEqual(len(meta), 9)  # ISIC_0000010 has no JPEG
        # lesion id where known, otherwise the image id
        self.assertEqual(
            sorted(set(meta["patient_id"])),
            sorted(
                [
                    "BCN_1",
                    "HAM_1",
                    "HAM_2",
                    "MSK4_1",
                    "ISIC_0000006_downsampled",
                    "BCN_2",
                    "BCN_3",
                    "BCN_4",
                ]
            ),
        )
        ham = set(meta.loc[meta["is_ham10000"] == 1, "image_id"])
        self.assertEqual(ham, {"ISIC_0000003", "ISIC_0000004"})

    def test_test_split_flags_isic2018_holdout(self):
        dataset = self._dataset(splits=("train", "test"))
        meta = pd.read_csv(dataset.metadata_path)
        test = meta[meta["split"] == "test"].set_index("image_id")
        self.assertEqual(len(test), 4)
        self.assertEqual(
            test["is_ham10000"].to_dict(),
            {
                "ISIC_0034400": 1,
                "ISIC_0036064": 1,
                "ISIC_0060000": 0,
                "ISIC_0060001": 0,
            },
        )
        self.assertEqual(test.loc["ISIC_0060001", "diagnosis"], "UNK")

    def test_missing_ground_truth_raises(self):
        (self.root / "ISIC_2019_Training_GroundTruth.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            self._dataset()

    def test_bad_split_raises(self):
        with self.assertRaises(ValueError):
            self._dataset(splits=("val",))

    def test_missing_training_metadata_raises(self):
        # Without lesion ids, lesion grouping and the HAM10000 flag break.
        (self.root / "ISIC_2019_Training_Metadata.csv").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "lesion ids"):
            self._dataset()

    def test_training_metadata_without_lesion_ids_raises(self):
        path = self.root / "ISIC_2019_Training_Metadata.csv"
        pd.read_csv(path).drop(columns=["lesion_id"]).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "lesion_id"):
            self._dataset()

    def test_missing_test_metadata_still_loads(self):
        (self.root / "ISIC_2019_Test_Metadata.csv").unlink()
        dataset = self._dataset(splits=("test",))
        meta = pd.read_csv(dataset.metadata_path)
        self.assertEqual(len(meta), 4)
        self.assertTrue(meta["age_approx"].isna().all())

    def test_rebuilds_metadata_when_splits_change(self):
        metadata_dir = str(self.tmp / "meta")
        first = self._dataset(metadata_dir=metadata_dir)
        self.assertEqual(set(pd.read_csv(first.metadata_path)["split"]), {"train"})
        second = self._dataset(metadata_dir=metadata_dir, splits=("train", "test"))
        self.assertEqual(
            set(pd.read_csv(second.metadata_path)["split"]), {"train", "test"}
        )


    def test_rebuilds_metadata_when_missing_images_are_restored(self):
        metadata_dir = str(self.tmp / "meta")
        first = self._dataset(metadata_dir=metadata_dir)
        self.assertNotIn(
            "ISIC_0000010", set(pd.read_csv(first.metadata_path)["image_id"])
        )
        Image.new("RGB", (10, 10)).save(
            self.root / "ISIC_2019_Training_Input" / "ISIC_0000010.jpg"
        )
        second = self._dataset(metadata_dir=metadata_dir)
        self.assertIn(
            "ISIC_0000010", set(pd.read_csv(second.metadata_path)["image_id"])
        )


class TestISIC2019Download(unittest.TestCase):
    """download=True, with the network calls replaced by local copies."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.source = self.tmp / "source"
        make_isic2019_root(self.source)
        # a complete release: every ground-truth row has its JPEG
        Image.new("RGB", (10, 10)).save(
            self.source / "ISIC_2019_Training_Input" / "ISIC_0000010.jpg"
        )
        self.root = self.tmp / "isic2019"
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
                "pyhealth.datasets.isic2019.download_file",
                side_effect=self._fake_download_file,
            ),
            mock.patch(
                "pyhealth.datasets.isic2019._download_archives",
                side_effect=self._fake_download_archives,
            ),
        ):
            dataset = ISIC2019Dataset(
                root=str(self.root),
                splits=("train", "test"),
                download=True,
                cache_dir=str(self.cache),
            )
        base = "https://isic-archive.s3.amazonaws.com/challenges/2019/"
        self.assertEqual(
            sorted(self.urls),
            sorted(
                base + f"ISIC_2019_{split}_{kind}"
                for split in ("Training", "Test")
                for kind in ("GroundTruth.csv", "Metadata.csv", "Input.zip")
            ),
        )
        self.assertTrue((self.root / "ISIC_2019_Training_Metadata.csv").exists())
        self.assertEqual(len(pd.read_csv(dataset.metadata_path)), 14)

        # A second download=True finds every file and fetches nothing.
        self.urls.clear()
        with mock.patch(
            "pyhealth.datasets.isic2019.download_file",
            side_effect=self._fake_download_file,
        ):
            ISIC2019Dataset(
                root=str(self.root),
                splits=("train", "test"),
                download=True,
                cache_dir=str(self.cache),
            )
        self.assertEqual(self.urls, [])

    def test_incomplete_image_folder_is_downloaded_again(self):
        self.root.mkdir()
        for name in ("GroundTruth", "Metadata"):
            shutil.copy(self.source / f"ISIC_2019_Training_{name}.csv", self.root)
        partial = self.root / "ISIC_2019_Training_Input"
        partial.mkdir()
        shutil.copy(self.source / "ISIC_2019_Training_Input" / "ISIC_0000001.jpg",
                    partial)

        def _extract(urls, destination, **kwargs):
            self.urls.extend(urls)
            shutil.copytree(self.source / Path(destination).name, destination,
                            dirs_exist_ok=True)

        with (
            mock.patch("pyhealth.datasets.isic2019.download_file",
                       side_effect=self._fake_download_file),
            mock.patch("pyhealth.datasets.isic2019._download_archives",
                       side_effect=_extract),
            self.assertLogs("pyhealth.datasets.isic2019", "WARNING"),
        ):
            dataset = ISIC2019Dataset(root=str(self.root), download=True,
                                      cache_dir=str(self.cache))
        self.assertEqual([u.rsplit("/", 1)[1] for u in self.urls],
                         ["ISIC_2019_Training_Input.zip"])
        self.assertEqual(len(pd.read_csv(dataset.metadata_path)), 10)

    def test_html_response_is_rejected(self):
        def _html(url, path):
            Path(path).write_text("<!DOCTYPE html><html>moved</html>\n")

        with (
            mock.patch("pyhealth.datasets.isic2019.download_file", side_effect=_html),
            self.assertRaisesRegex(RuntimeError, "not the expected CSV"),
        ):
            ISIC2019Dataset(
                root=str(self.root), download=True, cache_dir=str(self.cache)
            )
        self.assertFalse((self.root / "ISIC_2019_Training_GroundTruth.csv").exists())


if __name__ == "__main__":
    unittest.main()

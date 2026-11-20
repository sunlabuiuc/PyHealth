"""Tests for ISIC2020Dataset using a tiny synthetic layout."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from PIL import Image

from pyhealth.datasets import ISIC2020Dataset
from pyhealth.datasets import isic2020 as isic2020_module

_ROWS = [
    # image_name, patient_id, target
    ("ISIC_0000001", "IP_1", 0),
    ("ISIC_0000002", "IP_1", 1),
    ("ISIC_0000003", "IP_2", 0),
    ("ISIC_0000004", "IP_2", 0),  # duplicate of ISIC_0000003
    ("ISIC_0000005", "IP_3", 1),
    ("ISIC_0000006", "IP_3", 0),  # no JPEG on disk
    ("ISIC_0000007", "IP_2", 1),
    ("ISIC_0000008", "IP_3", 0),
]


def make_isic2020_root(root: Path) -> None:
    """Write a ground-truth CSV, a duplicates CSV and seven tiny JPEGs."""
    (root / "train").mkdir(parents=True)
    rows = []
    for i, (name, patient, target) in enumerate(_ROWS):
        rows.append(
            {
                "image_name": name,
                "patient_id": patient,
                "lesion_id": f"IL_{i}",
                "sex": "female" if i % 2 else "male",
                "age_approx": 40 + i,
                "anatom_site_general_challenge": "torso",
                "diagnosis": "melanoma" if target else "unknown",
                "benign_malignant": "malignant" if target else "benign",
                "target": target,
            }
        )
        if name != "ISIC_0000006":
            Image.new("RGB", (12, 8), (i * 40, 80, 120)).save(
                root / "train" / f"{name}.jpg"
            )
    pd.DataFrame(rows).to_csv(
        root / "ISIC_2020_Training_GroundTruth_v2.csv", index=False
    )
    pd.DataFrame(
        {"image_name_1": ["ISIC_0000003"], "image_name_2": ["ISIC_0000004"]}
    ).to_csv(root / "ISIC_2020_Training_Duplicates.csv", index=False)


class TestISIC2020Dataset(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "isic2020"
        make_isic2020_root(self.root)
        self.cache = self.tmp / "cache"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _dataset(self, **kwargs):
        return ISIC2020Dataset(root=str(self.root), cache_dir=str(self.cache), **kwargs)

    def test_metadata_written_outside_root(self):
        dataset = self._dataset()
        self.assertTrue(dataset.metadata_path.exists())
        self.assertFalse((self.root / "isic2020-metadata-pyhealth.csv").exists())
        meta = pd.read_csv(dataset.metadata_path)
        # duplicate ISIC_0000004 and missing ISIC_0000006 are dropped
        self.assertEqual(
            sorted(meta["image_id"]),
            [
                "ISIC_0000001",
                "ISIC_0000002",
                "ISIC_0000003",
                "ISIC_0000005",
                "ISIC_0000007",
                "ISIC_0000008",
            ],
        )
        self.assertTrue(all(Path(p).is_absolute() for p in meta["path"]))

    def test_patients_are_isic_patients(self):
        dataset = self._dataset()
        self.assertEqual(sorted(dataset.unique_patient_ids), ["IP_1", "IP_2", "IP_3"])
        events = dataset.get_patient("IP_1").get_events(event_type="isic2020")
        self.assertEqual(len(events), 2)

    def test_keep_duplicates_regenerates_metadata(self):
        self._dataset()
        dataset = self._dataset(drop_duplicates=False)
        meta = pd.read_csv(dataset.metadata_path)
        self.assertIn("ISIC_0000004", set(meta["image_id"]))

    def test_rebuilds_metadata_when_missing_images_are_restored(self):
        metadata_dir = str(self.tmp / "meta")
        first = self._dataset(metadata_dir=metadata_dir)
        self.assertNotIn(
            "ISIC_0000006", set(pd.read_csv(first.metadata_path)["image_id"])
        )
        Image.new("RGB", (12, 8)).save(self.root / "train" / "ISIC_0000006.jpg")
        second = self._dataset(metadata_dir=metadata_dir)
        self.assertIn(
            "ISIC_0000006", set(pd.read_csv(second.metadata_path)["image_id"])
        )

    def test_v1_ground_truth_is_used_when_v2_is_absent(self):
        v2 = self.root / "ISIC_2020_Training_GroundTruth_v2.csv"
        gt = pd.read_csv(v2).drop(columns=["lesion_id"])
        gt.loc[gt["image_name"] == "ISIC_0000001", "target"] = 1
        gt.to_csv(self.root / "ISIC_2020_Training_GroundTruth.csv", index=False)
        v2.unlink()
        meta = pd.read_csv(self._dataset().metadata_path).set_index("image_id")
        self.assertEqual(meta.loc["ISIC_0000001", "target"], 1)
        self.assertTrue(meta["lesion_id"].isna().all())

    def test_missing_lesion_id_column_is_left_blank(self):
        v2 = self.root / "ISIC_2020_Training_GroundTruth_v2.csv"
        pd.read_csv(v2).drop(columns=["lesion_id"]).to_csv(v2, index=False)
        dataset = self._dataset()
        meta = pd.read_csv(dataset.metadata_path)
        self.assertEqual(len(meta), 6)
        self.assertTrue(meta["lesion_id"].isna().all())
        self.assertEqual(len(dataset.unique_patient_ids), 3)

    def test_missing_ground_truth_raises(self):
        (self.root / "ISIC_2020_Training_GroundTruth_v2.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            self._dataset()

    def test_download_rejects_html_redirect_page(self):
        empty = self.tmp / "empty"

        def fake_download(url, path):
            self.assertTrue(url.startswith(isic2020_module.ISIC2020_BASE_URL))
            Path(path).write_text("<html><meta http-equiv='refresh'></html>")
            return path

        with (
            mock.patch.object(isic2020_module, "download_file", fake_download),
            self.assertRaises(RuntimeError),
        ):
            ISIC2020Dataset(root=str(empty), cache_dir=str(self.cache), download=True)
        self.assertFalse((empty / "ISIC_2020_Training_GroundTruth_v2.csv").exists())

    def test_download_fetches_missing_files_into_root(self):
        target = self.tmp / "downloaded"
        urls = []

        def fake_download(url, path):
            urls.append(url)
            shutil.copy(self.root / url.rsplit("/", 1)[1], path)

        def fake_archives(archive_urls, destination, **kwargs):
            urls.extend(archive_urls)
            self.assertTrue(kwargs["extract_to_parent"])
            shutil.copytree(self.root / "train", destination)

        with (
            mock.patch.object(isic2020_module, "download_file", fake_download),
            mock.patch.object(isic2020_module, "_download_archives", fake_archives),
        ):
            dataset = ISIC2020Dataset(
                root=str(target), cache_dir=str(self.cache), download=True
            )
        base = isic2020_module.ISIC2020_BASE_URL
        self.assertEqual(
            sorted(urls),
            sorted(
                base + name
                for name in (
                    "ISIC_2020_Training_GroundTruth_v2.csv",
                    "ISIC_2020_Training_Duplicates.csv",
                    "ISIC_2020_Training_JPEG.zip",
                )
            ),
        )
        self.assertTrue((target / "ISIC_2020_Training_Duplicates.csv").exists())
        self.assertEqual(len(pd.read_csv(dataset.metadata_path)), 6)

    def test_download_refetches_incomplete_image_folder(self):
        target = self.tmp / "partial"
        (target / "train").mkdir(parents=True)
        for name in (
            "ISIC_2020_Training_GroundTruth_v2.csv",
            "ISIC_2020_Training_Duplicates.csv",
        ):
            shutil.copy(self.root / name, target)
        shutil.copy(self.root / "train" / "ISIC_0000001.jpg", target / "train")
        urls = []

        def fake_download(url, path):
            raise AssertionError(f"CSV {url} should not be downloaded again")

        def fake_archives(archive_urls, destination, **kwargs):
            urls.extend(archive_urls)
            shutil.copytree(self.root / "train", destination, dirs_exist_ok=True)

        with (
            mock.patch.object(isic2020_module, "download_file", fake_download),
            mock.patch.object(isic2020_module, "_download_archives", fake_archives),
            self.assertLogs("pyhealth.datasets.isic2020", "WARNING") as logs,
        ):
            dataset = ISIC2020Dataset(
                root=str(target), cache_dir=str(self.cache), download=True
            )
        self.assertEqual(
            urls, [isic2020_module.ISIC2020_BASE_URL + "ISIC_2020_Training_JPEG.zip"]
        )
        self.assertTrue(any("1 of 8 images" in line for line in logs.output))
        self.assertEqual(len(pd.read_csv(dataset.metadata_path)), 6)

    def test_download_skips_complete_image_folder(self):
        target = self.tmp / "complete"
        shutil.copytree(self.root, target)
        Image.new("RGB", (12, 8)).save(target / "train" / "ISIC_0000006.jpg")

        def fail(*args, **kwargs):
            raise AssertionError("nothing should be downloaded")

        with (
            mock.patch.object(isic2020_module, "download_file", fail),
            mock.patch.object(isic2020_module, "_download_archives", fail),
        ):
            dataset = ISIC2020Dataset(
                root=str(target), cache_dir=str(self.cache), download=True
            )
        self.assertEqual(len(pd.read_csv(dataset.metadata_path)), 7)


if __name__ == "__main__":
    unittest.main()

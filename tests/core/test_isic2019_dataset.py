"""Tests for ISIC2019Dataset using a tiny synthetic ISIC 2019 layout."""

import shutil
import tempfile
import unittest
from pathlib import Path

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
        return ISIC2019Dataset(
            root=str(self.root), cache_dir=str(self.cache), **kwargs
        )

    def test_training_metadata(self):
        dataset = self._dataset()
        self.assertFalse((self.root / "isic2019-metadata-pyhealth.csv").exists())
        meta = pd.read_csv(dataset.metadata_path)
        self.assertEqual(len(meta), 9)  # ISIC_0000010 has no JPEG
        # lesion id where known, otherwise the image id
        self.assertEqual(
            sorted(set(meta["patient_id"])),
            sorted(
                ["BCN_1", "HAM_1", "HAM_2", "MSK4_1", "ISIC_0000006_downsampled",
                 "BCN_2", "BCN_3", "BCN_4"]
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
            {"ISIC_0034400": 1, "ISIC_0036064": 1, "ISIC_0060000": 0,
             "ISIC_0060001": 0},
        )
        self.assertEqual(test.loc["ISIC_0060001", "diagnosis"], "UNK")

    def test_missing_ground_truth_raises(self):
        (self.root / "ISIC_2019_Training_GroundTruth.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            self._dataset()

    def test_bad_split_raises(self):
        with self.assertRaises(ValueError):
            self._dataset(splits=("val",))


if __name__ == "__main__":
    unittest.main()

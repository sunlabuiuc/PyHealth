"""Tests for the Jichi DR fundus dataset loader (synthetic data only)."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from pyhealth.datasets import JichiDRDataset
from pyhealth.datasets.jichi import parse_jichi_filename

# (image, one-figure grade, concatenated grade)
ROWS = [
    ("1_1_R.jpg", "ndr", "ndr"),
    ("1_2_L.jpg", "sdr", "ndr"),
    ("2_1_R.jpg", "ppdr", "ppdr"),
    ("3_1_L.jpg", "pdr", "pdr"),
    ("3_2_R.jpg", "ndr", "sdr"),
]


def write_fixture(folder: Path, rows=ROWS, missing=()) -> None:
    """Write a tiny ``dmr/`` folder: list.csv plus small JPEGs."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, _, _ in rows:
        if name in missing:
            continue
        pixels = np.full((10, 10, 3), 90, dtype=np.uint8)
        Image.fromarray(pixels).save(folder / name)
    pd.DataFrame(
        rows,
        columns=[
            "Image",
            "Davis_grading_of_one_figure",
            "Davis_grading_of_concatenated_figures",
        ],
    ).to_csv(folder / "list.csv", index=False)


class TestJichiDRDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cache = self.tmp / "cache"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_parse_filename(self):
        self.assertEqual(parse_jichi_filename("12_3_r.jpg"), ("12", 3, "R"))
        self.assertIsNone(parse_jichi_filename("12_3.jpg"))

    def test_loads_patients_from_filename_prefix(self):
        write_fixture(self.tmp / "dmr")
        ds = JichiDRDataset(root=str(self.tmp / "dmr"), cache_dir=str(self.cache))
        self.assertEqual(sorted(ds.unique_patient_ids), ["1", "2", "3"])
        events = ds.get_patient("1").get_events(event_type="jichi")
        self.assertEqual(sorted(e.eye for e in events), ["L", "R"])
        self.assertEqual(sorted(e.davis_one for e in events), ["ndr", "sdr"])
        self.assertEqual({e.davis_concatenated for e in events}, {"ndr"})

    def test_parent_folder_and_no_writes_to_root(self):
        write_fixture(self.tmp / "data" / "dmr")
        ds = JichiDRDataset(root=str(self.tmp / "data"), cache_dir=str(self.cache))
        self.assertEqual(len(ds.unique_patient_ids), 3)
        self.assertTrue(ds.metadata_path.is_file())
        self.assertFalse(
            any(p.name.endswith("pyhealth.csv") for p in (self.tmp / "data").rglob("*"))
        )

    def test_missing_images_dropped(self):
        write_fixture(self.tmp / "dmr", missing=("2_1_R.jpg",))
        ds = JichiDRDataset(root=str(self.tmp / "dmr"), cache_dir=str(self.cache))
        self.assertEqual(sorted(ds.unique_patient_ids), ["1", "3"])

    def test_missing_list_raises(self):
        (self.tmp / "dmr").mkdir()
        with self.assertRaises(FileNotFoundError):
            JichiDRDataset(root=str(self.tmp / "dmr"), cache_dir=str(self.cache))

    def test_missing_label_column_raises(self):
        write_fixture(self.tmp / "dmr")
        table = pd.read_csv(self.tmp / "dmr" / "list.csv")
        table.drop(columns=["Davis_grading_of_one_figure"]).to_csv(
            self.tmp / "dmr" / "list.csv", index=False
        )
        with self.assertRaisesRegex(ValueError, "Davis_grading_of_one_figure"):
            JichiDRDataset(root=str(self.tmp / "dmr"), cache_dir=str(self.cache))

    def test_no_images_raises(self):
        write_fixture(self.tmp / "dmr", missing=tuple(r[0] for r in ROWS))
        with self.assertRaises(FileNotFoundError):
            JichiDRDataset(root=str(self.tmp / "dmr"), cache_dir=str(self.cache))

    def test_explicit_metadata_dir_rebuilt_when_data_changes(self):
        meta = self.tmp / "meta"
        write_fixture(self.tmp / "a" / "dmr")
        ds = JichiDRDataset(
            root=str(self.tmp / "a" / "dmr"),
            cache_dir=str(self.cache / "a"),
            metadata_dir=str(meta),
        )
        self.assertEqual(ds.metadata_path, meta / "jichi-metadata-pyhealth.csv")
        write_fixture(self.tmp / "b" / "dmr", rows=ROWS[:2])
        ds = JichiDRDataset(
            root=str(self.tmp / "b" / "dmr"),
            cache_dir=str(self.cache / "b"),
            metadata_dir=str(meta),
        )
        table = pd.read_csv(ds.metadata_path, dtype=str)
        self.assertEqual(sorted(table["patient_id"].unique()), ["1"])
        self.assertTrue(all("/b/dmr/" in p for p in table["path"]))


if __name__ == "__main__":
    unittest.main()

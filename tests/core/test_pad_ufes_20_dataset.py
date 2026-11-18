"""Tests for the PAD-UFES-20 dataset loader (synthetic data only)."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from pyhealth.datasets import PADUFES20Dataset

# (patient_id, lesion_id, img_id, diagnostic, part folder)
ROWS = [
    ("PAT_1", 10, "PAT_1_10_1.png", "BCC", "imgs_part_1"),
    ("PAT_1", 11, "PAT_1_11_2.png", "SCC", "imgs_part_1"),
    ("PAT_2", 20, "PAT_2_20_3.png", "MEL", "imgs_part_2"),
    ("PAT_3", 30, "PAT_3_30_4.png", "SEK", "imgs_part_3"),
]


def write_fixture(root: Path, rows=ROWS, missing=()) -> None:
    """Write a tiny PAD-UFES-20 layout with RGBA PNGs under imgs_part_*."""
    records = []
    for pid, lesion, img, diag, part in rows:
        records.append(
            {
                "patient_id": pid,
                "lesion_id": lesion,
                "age": 50,
                "gender": "FEMALE",
                "fitspatrick": 2.0,
                "region": "ARM",
                "diameter_1": 5.0,
                "diameter_2": 4.0,
                "diagnostic": diag,
                "itch": True,
                "grew": False,
                "hurt": False,
                "changed": False,
                "bleed": False,
                "elevation": True,
                "img_id": img,
                "biopsed": True,
            }
        )
        if img in missing:
            continue
        folder = root / "images" / part
        folder.mkdir(parents=True, exist_ok=True)
        pixels = np.full((12, 16, 4), 128, dtype=np.uint8)
        Image.fromarray(pixels, mode="RGBA").save(folder / img)
    pd.DataFrame(records).to_csv(root / "metadata.csv", index=False)


class TestPADUFES20Dataset(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "pad"
        self.root.mkdir()
        self.cache = self.tmp / "cache"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_loads_patients_and_events(self):
        write_fixture(self.root)
        ds = PADUFES20Dataset(root=str(self.root), cache_dir=str(self.cache))
        self.assertEqual(sorted(ds.unique_patient_ids), ["PAT_1", "PAT_2", "PAT_3"])
        events = ds.get_patient("PAT_1").get_events(event_type="pad_ufes_20")
        self.assertEqual(len(events), 2)
        self.assertEqual({e.diagnostic for e in events}, {"BCC", "SCC"})
        self.assertEqual({e.lesion_id for e in events}, {"10", "11"})
        for e in events:
            self.assertTrue(Path(e.path).is_file())

    def test_metadata_written_outside_root(self):
        write_fixture(self.root)
        ds = PADUFES20Dataset(root=str(self.root), cache_dir=str(self.cache))
        self.assertTrue(ds.metadata_path.is_file())
        self.assertIn(str(self.cache), str(ds.metadata_path))
        self.assertEqual(
            sorted(p.name for p in self.root.iterdir()), ["images", "metadata.csv"]
        )

    def test_missing_images_are_dropped(self):
        write_fixture(self.root, missing=("PAT_3_30_4.png",))
        ds = PADUFES20Dataset(root=str(self.root), cache_dir=str(self.cache))
        self.assertEqual(sorted(ds.unique_patient_ids), ["PAT_1", "PAT_2"])

    def test_missing_metadata_raises(self):
        (self.root / "images").mkdir()
        with self.assertRaises(FileNotFoundError):
            PADUFES20Dataset(root=str(self.root), cache_dir=str(self.cache))

    def test_no_images_raises(self):
        write_fixture(self.root, missing=tuple(r[2] for r in ROWS))
        (self.root / "images").mkdir(exist_ok=True)
        with self.assertRaises(FileNotFoundError):
            PADUFES20Dataset(root=str(self.root), cache_dir=str(self.cache))


if __name__ == "__main__":
    unittest.main()

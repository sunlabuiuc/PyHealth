"""Tests for the PAD-UFES-20 dataset loader (synthetic data only)."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from pyhealth.datasets import PADUFES20Dataset
from pyhealth.datasets.pad_ufes_20 import PAD_UFES_20_CLINICAL_COLUMNS

# (patient_id, lesion_id, img_id, diagnostic, part folder)
ROWS = [
    ("PAT_1", 10, "PAT_1_10_1.png", "BCC", "imgs_part_1"),
    ("PAT_1", 11, "PAT_1_11_2.png", "SCC", "imgs_part_1"),
    ("PAT_2", 20, "PAT_2_20_3.png", "MEL", "imgs_part_2"),
    ("PAT_3", 30, "PAT_3_30_4.png", "SEK", "imgs_part_3"),
]


def write_fixture(
    root: Path, rows=ROWS, missing=(), flat=False, drop_columns=()
) -> None:
    """Write a tiny PAD-UFES-20 layout with RGBA PNGs.

    PNGs go under ``images/imgs_part_*`` (the official layout), or directly
    under ``images/`` when ``flat`` is True. ``drop_columns`` are left out of
    ``metadata.csv``.
    """
    records = []
    for pid, lesion, img, diag, part in rows:
        records.append(
            {
                "patient_id": pid,
                "lesion_id": lesion,
                "smoke": False,
                "drink": True,
                "background_father": "POMERANIA",
                "background_mother": "ITALIA",
                "age": 50,
                "pesticide": False,
                "skin_cancer_history": True,
                "cancer_history": False,
                "has_piped_water": True,
                "has_sewage_system": True,
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
        folder = root / "images" if flat else root / "images" / part
        folder.mkdir(parents=True, exist_ok=True)
        pixels = np.full((12, 16, 4), 128, dtype=np.uint8)
        Image.fromarray(pixels, mode="RGBA").save(folder / img)
    frame = pd.DataFrame(records).drop(columns=list(drop_columns))
    frame.to_csv(root / "metadata.csv", index=False)


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

    def test_all_clinical_columns_are_attributes(self):
        write_fixture(self.root)
        ds = PADUFES20Dataset(root=str(self.root), cache_dir=str(self.cache))
        event = ds.get_patient("PAT_2").get_events(event_type="pad_ufes_20")[0]
        for column in PAD_UFES_20_CLINICAL_COLUMNS:
            self.assertIn(column, event.attr_dict)
        self.assertEqual(event.background_father, "POMERANIA")
        self.assertEqual(event.skin_cancer_history, "True")

    def test_absent_clinical_column_is_empty(self):
        write_fixture(self.root, drop_columns=("smoke", "pesticide"))
        ds = PADUFES20Dataset(root=str(self.root), cache_dir=str(self.cache))
        event = ds.get_patient("PAT_2").get_events(event_type="pad_ufes_20")[0]
        self.assertIsNone(event.smoke)
        self.assertEqual(event.background_mother, "ITALIA")

    def test_flat_images_layout(self):
        write_fixture(self.root, flat=True)
        self.assertFalse(any(p.is_dir() for p in (self.root / "images").iterdir()))
        ds = PADUFES20Dataset(root=str(self.root), cache_dir=str(self.cache))
        self.assertEqual(sorted(ds.unique_patient_ids), ["PAT_1", "PAT_2", "PAT_3"])

    def test_missing_required_column_raises(self):
        write_fixture(self.root, drop_columns=("diagnostic",))
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            PADUFES20Dataset(root=str(self.root), cache_dir=str(self.cache))

    def test_refresh_metadata_rewrites_csv(self):
        write_fixture(self.root)
        kwargs = {
            "root": str(self.root),
            "cache_dir": str(self.cache),
            "metadata_dir": str(self.tmp / "meta"),
        }
        ds = PADUFES20Dataset(**kwargs)
        # Truncate the derived CSV to one row: it is reused as is ...
        pd.read_csv(ds.metadata_path).head(1).to_csv(ds.metadata_path, index=False)
        self.assertEqual(len(PADUFES20Dataset(**kwargs).unique_patient_ids), 1)
        # ... until refresh_metadata=True rebuilds it from metadata.csv.
        ds = PADUFES20Dataset(**kwargs, refresh_metadata=True)
        self.assertEqual(len(pd.read_csv(ds.metadata_path)), len(ROWS))
        self.assertEqual(len(ds.unique_patient_ids), 3)


if __name__ == "__main__":
    unittest.main()

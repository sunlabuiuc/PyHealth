"""Tests for the KiTS23 dataset loader, on a synthetic KiTS23 tree."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - the test env installs nibabel
    nib = None

from pyhealth.datasets import KiTS23Dataset
from pyhealth.tasks import KiTS23MalignancyClassification

SHAPE_XYZ = (16, 16, 8)

CASES = [
    {"case_id": "case_00000", "malignant": True,
     "tumor_histologic_subtype": "clear_cell_rcc", "tumor_isup_grade": "2"},
    {"case_id": "case_00001", "malignant": False,
     "tumor_histologic_subtype": "oncocytoma", "tumor_isup_grade": None},
    {"case_id": "case_00002", "malignant": True,
     "tumor_histologic_subtype": "papillary", "tumor_isup_grade": "3"},
    {"case_id": "case_00003", "malignant": None,
     "tumor_histologic_subtype": None, "tumor_isup_grade": None},
]


def write_case(folder: Path, tumor_slice: int, seed: int) -> None:
    """Write a CT and a segmentation with a tumour on ``tumor_slice``."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ct = rng.integers(-1000, 1000, size=SHAPE_XYZ).astype(np.int16)
    seg = np.zeros(SHAPE_XYZ, dtype=np.uint8)
    seg[4:12, 4:12, :] = 1
    seg[6:10, 6:10, tumor_slice] = 2
    nib.save(nib.Nifti1Image(ct, np.eye(4)), folder / "imaging.nii.gz")
    nib.save(nib.Nifti1Image(seg, np.eye(4)), folder / "segmentation.nii.gz")


def build_kits23_root(root: Path, with_imaging=("case_00000", "case_00001",
                                                 "case_00003")) -> Path:
    """A KiTS23 ``dataset/`` folder; case_00002 has no imaging."""
    dataset = root / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "kits23.json").write_text(json.dumps(CASES))
    for index, case in enumerate(CASES):
        if case["case_id"] in with_imaging:
            write_case(dataset / case["case_id"], tumor_slice=2 + index, seed=index)
    return dataset


@unittest.skipUnless(nib is not None, "nibabel is not installed")
class TestKiTS23Dataset(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.dataset_dir = build_kits23_root(self.tmp / "kits23")
        self.cache = self.tmp / "cache"

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_metadata_rows_and_labels(self):
        dataset = KiTS23Dataset(root=str(self.dataset_dir), cache_dir=str(self.cache))
        meta = pd.read_csv(dataset.metadata_path).set_index("record_id")
        self.assertEqual(len(meta), 4)
        self.assertEqual(meta.loc["case_00000", "malignant"], 1)
        self.assertEqual(meta.loc["case_00001", "malignant"], 0)
        self.assertEqual(meta.loc["case_00003", "malignant"], -1)
        self.assertEqual(meta.loc["case_00002", "has_imaging"], 0)
        self.assertEqual(meta.loc["case_00001", "tumor_histologic_subtype"], "oncocytoma")
        self.assertTrue(str(dataset.metadata_path).startswith(str(self.cache)))
        self.assertFalse(any(self.dataset_dir.rglob("*pyhealth*")))
        self.assertEqual(len(dataset.unique_patient_ids), 4)

    def test_repository_root_is_accepted(self):
        dataset = KiTS23Dataset(root=str(self.tmp / "kits23"), cache_dir=str(self.cache))
        self.assertEqual(dataset.source_root, self.dataset_dir.resolve())

    def test_metadata_rebuilds_when_imaging_appears(self):
        KiTS23Dataset(root=str(self.dataset_dir), cache_dir=str(self.cache))
        write_case(self.dataset_dir / "case_00002", tumor_slice=3, seed=9)
        dataset = KiTS23Dataset(root=str(self.dataset_dir), cache_dir=str(self.cache))
        meta = pd.read_csv(dataset.metadata_path).set_index("record_id")
        self.assertEqual(meta.loc["case_00002", "has_imaging"], 1)

    def test_roots_sharing_cache_dir_keep_separate_metadata(self):
        first = KiTS23Dataset(root=str(self.dataset_dir), cache_dir=str(self.cache))
        before = pd.read_csv(first.metadata_path)
        other_cases = [{"case_id": "case_00100", "malignant": False},
                       {"case_id": "case_00101", "malignant": True}]
        other_dir = self.tmp / "other" / "dataset"
        other_dir.mkdir(parents=True)
        (other_dir / "kits23.json").write_text(json.dumps(other_cases))
        for index, case in enumerate(other_cases):
            write_case(other_dir / case["case_id"], tumor_slice=3, seed=20 + index)
        second = KiTS23Dataset(root=str(other_dir), cache_dir=str(self.cache))

        self.assertNotEqual(first.metadata_dir, second.metadata_dir)
        self.assertTrue(first.metadata_dir.name.startswith("kits23-"))
        pd.testing.assert_frame_equal(pd.read_csv(first.metadata_path), before)
        self.assertEqual(
            sorted(pd.read_csv(second.metadata_path)["record_id"]),
            ["case_00100", "case_00101"],
        )
        self.assertEqual(
            sorted(first.unique_patient_ids),
            ["case_00000", "case_00001", "case_00002", "case_00003"],
        )
        task = KiTS23MalignancyClassification(target_shape=(4, 8, 8))
        samples = first.set_task(task, num_workers=1)
        try:
            self.assertEqual(
                sorted(s["record_id"] for s in samples),
                ["case_00000", "case_00001"],
            )
        finally:
            samples.close()

    def test_explicit_metadata_dir_is_used(self):
        target = self.tmp / "explicit_meta"
        dataset = KiTS23Dataset(
            root=str(self.dataset_dir), cache_dir=str(self.cache),
            metadata_dir=str(target),
        )
        self.assertEqual(dataset.metadata_dir, target)
        self.assertTrue(dataset.metadata_path.exists())

    def test_missing_json_raises(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError):
            KiTS23Dataset(root=str(empty), cache_dir=str(self.cache))

    def test_default_task(self):
        dataset = KiTS23Dataset(root=str(self.dataset_dir), cache_dir=str(self.cache))
        self.assertEqual(dataset.default_task.task_name, "KiTS23MalignancyClassification")


if __name__ == "__main__":
    unittest.main()

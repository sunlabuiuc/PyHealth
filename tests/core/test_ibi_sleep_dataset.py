"""Tests for IBISleepDataset and SleepStagingIBI integration."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pyhealth.datasets import IBISleepDataset
from pyhealth.tasks import SleepStagingIBI

requires_py312 = unittest.skipIf(
    sys.version_info < (3, 12),
    reason="BaseDataset.set_task uses itertools.batched (Python 3.12+)",
)


def _write_npz(
    directory: str,
    name: str,
    n_epochs: int = 3,
    ahi: float = 5.0,
    fs: int = 25,
    include_ahi_key: bool = True,
) -> str:
    n = n_epochs * 750
    rng = np.random.default_rng(abs(hash(name)) % (2**31))
    data = rng.random(n).astype(np.float32)
    stages = rng.integers(0, 5, size=n).astype(np.int32)
    path = os.path.join(directory, f"{name}.npz")
    kwargs = dict(data=data, stages=stages, fs=np.int64(fs))
    if include_ahi_key:
        kwargs["ahi"] = np.float32(ahi)
    np.savez(path, **kwargs)
    return path


def _make_dataset(
    tmp_dir: str, n: int = 3, source: str = "dreamt", **kwargs
) -> IBISleepDataset:
    for i in range(n):
        _write_npz(tmp_dir, f"S{i:03d}")
    return IBISleepDataset(root=tmp_dir, source=source, **kwargs)


def _read_meta(directory: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(directory, "ibi_sleep-metadata.csv"))


class TestIBISleepDataset(unittest.TestCase):
    """Tests for IBISleepDataset.

    Tests that need BaseDataset's lazy patient-loading (unique_patient_ids,
    get_patient) share a single dataset instance via setUpClass so the ~2s
    load cost is paid once. Tests that only verify metadata CSV output call
    prepare_metadata() indirectly through the constructor and read the CSV
    directly, avoiding the patient-load entirely.
    """

    @classmethod
    def setUpClass(cls):
        cls._cls_tmpdir = tempfile.TemporaryDirectory()
        cls._cls_path = cls._cls_tmpdir.name
        for i in range(3):
            _write_npz(cls._cls_path, f"S{i:03d}")
        cls._ds = IBISleepDataset(root=cls._cls_path, source="dreamt")
        cls._pids = list(cls._ds.unique_patient_ids)

    @classmethod
    def tearDownClass(cls):
        cls._cls_tmpdir.cleanup()

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_load_dreamt_source(self):
        self.assertEqual(self._ds.source, "dreamt")
        self.assertTrue((Path(self._cls_path) / "ibi_sleep-metadata.csv").exists())

    def test_load_shhs_source(self):
        _write_npz(str(self.tmp_path), "shhs1-200001")
        _write_npz(str(self.tmp_path), "shhs2-300001")
        ds = IBISleepDataset(root=str(self.tmp_path), source="shhs")
        self.assertEqual(ds.source, "shhs")
        self.assertEqual(len(_read_meta(str(self.tmp_path))), 2)

    def test_load_mesa_source(self):
        for i in range(4):
            _write_npz(str(self.tmp_path), f"mesa-sleep-{i:05d}")
        ds = IBISleepDataset(root=str(self.tmp_path), source="mesa")
        self.assertEqual(ds.source, "mesa")
        self.assertEqual(len(_read_meta(str(self.tmp_path))), 4)

    def test_patient_ids(self):
        names = ["Alpha", "Beta", "Gamma"]
        for name in names:
            _write_npz(str(self.tmp_path), name)
        IBISleepDataset(root=str(self.tmp_path), source="dreamt")
        self.assertEqual(set(_read_meta(str(self.tmp_path))["patient_id"]), set(names))

    def test_getitem_keys(self):
        patient = self._ds.get_patient(self._pids[0])
        events = patient.get_events()
        self.assertGreaterEqual(len(events), 1)
        event = events[0]
        self.assertTrue(hasattr(event, "npz_path"))
        self.assertTrue(hasattr(event, "ahi"))

    def test_ahi_nan_passes_through(self):
        _write_npz(str(self.tmp_path), "nan_subject", ahi=float("nan"))
        IBISleepDataset(root=str(self.tmp_path), source="dreamt")
        df = _read_meta(str(self.tmp_path))
        self.assertTrue(df.loc[df["patient_id"] == "nan_subject", "ahi"].isna().all())

    def test_dev_mode(self):
        ds = _make_dataset(str(self.tmp_path), n=5, dev=True)
        self.assertTrue(ds.dev)

    def test_empty_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            IBISleepDataset(root=str(self.tmp_path), source="dreamt")

    def test_missing_dir_raises(self):
        with self.assertRaises((FileNotFoundError, OSError)):
            IBISleepDataset(root="/nonexistent/path/xyz", source="dreamt")

    def test_corrupt_npz_skipped(self):
        _write_npz(str(self.tmp_path), "good_subject")
        Path(os.path.join(str(self.tmp_path), "corrupt_subject.npz")).write_bytes(
            b"not a valid npz file"
        )
        IBISleepDataset(root=str(self.tmp_path), source="dreamt")
        df = _read_meta(str(self.tmp_path))
        self.assertEqual(len(df), 1)
        self.assertIn("good_subject", df["patient_id"].values)

    def test_missing_ahi_key_stores_nan(self):
        _write_npz(str(self.tmp_path), "no_ahi", include_ahi_key=False)
        IBISleepDataset(root=str(self.tmp_path), source="dreamt")
        df = _read_meta(str(self.tmp_path))
        self.assertTrue(df.loc[df["patient_id"] == "no_ahi", "ahi"].isna().all())

    def test_default_task(self):
        self.assertIsInstance(self._ds.default_task, SleepStagingIBI)


class TestIBISleepDatasetSetTask(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    @requires_py312
    def test_set_task_3class(self):
        ds = _make_dataset(str(self.tmp_path), n=2)
        sample_ds = ds.set_task()
        labels = [sample_ds[i]["label"] for i in range(len(sample_ds))]
        self.assertTrue(all(lbl in {0, 1, 2} for lbl in labels))

    @requires_py312
    def test_set_task_5class(self):
        ds = _make_dataset(str(self.tmp_path), n=2)
        sample_ds = ds.set_task(SleepStagingIBI(num_classes=5))
        labels = [sample_ds[i]["label"] for i in range(len(sample_ds))]
        self.assertTrue(all(lbl in {0, 1, 2, 3, 4} for lbl in labels))

    @requires_py312
    def test_set_task_signal_key(self):
        ds = _make_dataset(str(self.tmp_path), n=2)
        sample_ds = ds.set_task()
        self.assertGreater(len(sample_ds), 0)
        sample = sample_ds[0]
        self.assertIn("signal", sample)
        self.assertEqual(sample["signal"].shape, (750,))


if __name__ == "__main__":
    unittest.main()

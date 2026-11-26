"""Tests for the ABIDE autism connectivity classification task."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pyhealth.datasets import ABIDEDataset
from pyhealth.datasets.abide import ABIDE_PHENOTYPIC_CSV, abide_roi_relpath
from pyhealth.tasks import ABIDEAutismClassification
from pyhealth.tasks.abide_classification import abide_connectivity


class TestConnectivity(unittest.TestCase):
    def test_matches_numpy_and_handles_flat_roi(self):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=(30, 4))
        ts[:, 3] = 2.0  # zero-variance ROI
        corr = abide_connectivity(ts)
        np.testing.assert_allclose(corr[:3, :3], np.corrcoef(ts[:, :3].T), atol=1e-5)
        self.assertTrue(np.isfinite(corr).all())
        self.assertEqual(corr[3, 3], 1.0)
        self.assertEqual(corr[3, 0], 0.0)

    def test_fisher_and_vectorize(self):
        ts = np.random.default_rng(1).normal(size=(30, 4))
        z = abide_connectivity(ts, fisher_z=True)
        self.assertEqual(z[0, 0], 0.0)
        np.testing.assert_allclose(
            z[0, 1], np.arctanh(abide_connectivity(ts)[0, 1]), atol=1e-5
        )
        self.assertEqual(abide_connectivity(ts, vectorize=True).shape, (6,))


class TestABIDEAutismClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        root = cls.tmp / "abide"
        root.mkdir()
        pd.DataFrame(
            {
                "SUB_ID": [1, 2, 3],
                "SITE_ID": ["A", "B", "B"],
                "FILE_ID": ["A_1", "B_2", "B_3"],
                "DX_GROUP": [1, 2, 2],
                "AGE_AT_SCAN": [10.0, 11.0, 12.0],
                "SEX": [1, 2, 1],
            }
        ).to_csv(root / ABIDE_PHENOTYPIC_CSV, index=False)
        rng = np.random.default_rng(0)
        for file_id, n_time in (("A_1", 25), ("B_2", 25), ("B_3", 5)):
            path = root / abide_roi_relpath(file_id, "cpac", "filt_noglobal", "cc200")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(path, rng.normal(size=(n_time, 6)), header="#1 #2 #3 #4 #5 #6", comments="")
        cls.dataset = ABIDEDataset(
            root=str(root),
            metadata_cache_dir=str(cls.tmp / "meta"),
            cache_dir=str(cls.tmp / "cache"),
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_samples(self):
        samples = self.dataset.set_task(ABIDEAutismClassification(), num_workers=1)
        self.assertEqual(len(samples), 2)  # B_3 has too few time points
        by_id = {samples[i]["record_id"]: samples[i] for i in range(len(samples))}
        self.assertEqual(tuple(by_id["A_1"]["connectivity"].shape), (6, 6))
        self.assertEqual(int(by_id["A_1"]["autism"]), 1)
        self.assertEqual(int(by_id["B_2"]["autism"]), 0)
        self.assertEqual(by_id["B_2"]["site"], "B")

    def test_vectorized(self):
        samples = self.dataset.set_task(
            ABIDEAutismClassification(vectorize=True), num_workers=1
        )
        self.assertEqual(tuple(samples[0]["connectivity"].shape), (15,))


if __name__ == "__main__":
    unittest.main()

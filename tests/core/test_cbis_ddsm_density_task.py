"""Tests for CBISDDSMDensityClassification on a synthetic CBIS-DDSM root."""

import shutil
import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import CBISDDSMDataset
from pyhealth.tasks import CBISDDSMDensityClassification
from tests.core.cbis_ddsm_test_utils import build_cbis_root, pydicom

# P_00004 gets density 0 (not a BI-RADS category), so its images are skipped.
_DENSITY = {"P_00001": 1, "P_00002": 3, "P_00003": 4, "P_00004": 0}


@unittest.skipUnless(pydicom is not None, "pydicom is not installed")
class TestCBISDDSMDensityClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        root = cls.temp_dir / "cbis"
        root.mkdir()
        build_cbis_root(root, density=_DENSITY)
        cls.dataset = CBISDDSMDataset(
            root=str(root), cache_dir=str(cls.temp_dir / "cache")
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_density_labels(self):
        samples = self.dataset.set_task(CBISDDSMDensityClassification(), num_workers=1)
        self.assertEqual(samples.output_processors["density"].size(), 4)
        # class index = density - 1
        labels = {s["record_id"]: int(s["density"].item()) for s in samples}
        self.assertEqual(
            labels,
            {
                "P_00001_LEFT_CC_train": 0,
                "P_00002_RIGHT_MLO_train": 2,
                "P_00003_LEFT_MLO_test": 3,
            },
        )


if __name__ == "__main__":
    unittest.main()

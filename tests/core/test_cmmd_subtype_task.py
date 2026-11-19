"""Tests for the CMMD molecular subtype classification task."""

import shutil
import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import CMMDDataset, get_dataloader
from pyhealth.tasks import CMMDSubtypeClassification
from pyhealth.tasks.cmmd_subtype_classification import CMMD_SUBTYPES
from tests.core.cmmd_test_utils import CLINICAL, build_cmmd_root, pydicom


@unittest.skipUnless(pydicom is not None, "pydicom is not installed")
class TestCMMDSubtypeClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        root = cls.temp_dir / "cmmd"
        root.mkdir()
        clinical = CLINICAL.copy()
        # D1-0002's right breast gets a subtype too (with stray whitespace);
        # D2-0001's left breast keeps "Luminal B".
        clinical.loc[2, "subtype"] = " HER2-enriched "
        build_cmmd_root(root, clinical=clinical)
        cls.dataset = CMMDDataset(root=str(root), cache_dir=str(cls.temp_dir / "c"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_only_breasts_with_a_subtype(self):
        samples = self.dataset.set_task(CMMDSubtypeClassification(), num_workers=1)
        self.assertEqual(samples.output_processors["subtype"].size(), 4)
        labels = {
            (s["patient_id"], s["laterality"], s["view"]): int(s["subtype"].item())
            for s in samples
        }
        self.assertEqual(
            labels,
            {
                ("D1-0002", "R", "CC"): CMMD_SUBTYPES.index("HER2-enriched"),
                ("D2-0001", "L", "CC"): CMMD_SUBTYPES.index("Luminal B"),
                ("D2-0001", "L", "MLO"): CMMD_SUBTYPES.index("Luminal B"),
            },
        )
        batch = next(iter(get_dataloader(samples, batch_size=3)))
        self.assertEqual(tuple(batch["image"].shape), (3, 1, 512, 512))


if __name__ == "__main__":
    unittest.main()

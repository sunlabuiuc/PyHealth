"""Tests for the CMMD molecular subtype classification task."""

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pyhealth.datasets import CMMDDataset, get_dataloader
from pyhealth.tasks import CMMDSubtypeClassification, cmmd_subtype_classification
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
        # D1-0002's malignant left breast gets a subtype (with stray
        # whitespace); its benign right breast gets one too, which must be
        # ignored. D2-0001's left breast keeps "Luminal B".
        clinical.loc[1, "subtype"] = " HER2-enriched "
        clinical.loc[2, "subtype"] = "Luminal A"
        build_cmmd_root(root, clinical=clinical)
        cls.dataset = CMMDDataset(root=str(root), cache_dir=str(cls.temp_dir / "c"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_only_malignant_breasts_with_a_subtype(self):
        samples = self.dataset.set_task(CMMDSubtypeClassification(), num_workers=1)
        self.assertEqual(samples.output_processors["subtype"].size(), 4)
        labels = {
            (s["patient_id"], s["laterality"], s["view"]): int(s["subtype"].item())
            for s in samples
        }
        self.assertEqual(
            labels,
            {
                ("D1-0002", "L", "CC"): CMMD_SUBTYPES.index("HER2-enriched"),
                ("D2-0001", "L", "CC"): CMMD_SUBTYPES.index("Luminal B"),
                ("D2-0001", "L", "MLO"): CMMD_SUBTYPES.index("Luminal B"),
            },
        )
        batch = next(iter(get_dataloader(samples, batch_size=3)))
        self.assertEqual(tuple(batch["image"].shape), (3, 1, 512, 512))

    def test_unrecognised_subtype_warns_once_and_is_skipped(self):
        cmmd_subtype_classification._WARNED_SUBTYPES.clear()
        task = CMMDSubtypeClassification()
        state_before = dict(vars(task))
        event = SimpleNamespace(
            has_label="1",
            label="1.0",
            subtype="Luminal C",
            record_id="r",
            image_path="p",
            laterality="L",
            view="CC",
            cohort="D2",
        )
        benign = SimpleNamespace(**{**vars(event), "label": "0.0"})
        patient = mock.Mock(patient_id="D2-0009")
        patient.get_events.return_value = [event, event, benign]
        with self.assertLogs(
            "pyhealth.tasks.cmmd_subtype_classification", level="WARNING"
        ) as logs:
            self.assertEqual(task(patient), [])
            self.assertEqual(task(patient), [])
        self.assertEqual(len(logs.output), 1)
        self.assertIn("Luminal C", logs.output[0])
        # The task cache key hashes vars(task); warning must not change it.
        self.assertEqual(vars(task), state_before)

    def test_benign_breast_with_subtype_is_skipped(self):
        event = SimpleNamespace(
            has_label="1",
            label="0.0",
            subtype="Luminal A",
            record_id="r",
            image_path="p",
            laterality="R",
            view="CC",
            cohort="D2",
        )
        patient = mock.Mock(patient_id="D2-0009")
        patient.get_events.return_value = [event]
        self.assertEqual(CMMDSubtypeClassification()(patient), [])


if __name__ == "__main__":
    unittest.main()

"""DiseaseProgressionMIMIC4 end to end on small fabricated files.

Builds tiny MIMIC-IV, MIMIC-CXR-JPG and Chest ImaGenome trees with the
example's writers, then runs the real MIMIC4ImaGenomeDataset join, set_task
and GoldTestSplit, with no mocks.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pyhealth.datasets import GoldTestSplit, MIMIC4ImaGenomeDataset
from pyhealth.tasks import DiseaseProgressionMIMIC4

EXAMPLE = (Path(__file__).parents[2] / "examples" / "cxr"
           / "disease_progression_mimic4.py")


def _load_example():
    spec = importlib.util.spec_from_file_location("dp_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDiseaseProgressionEndToEnd(unittest.TestCase):
    N_PATIENTS = 6

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        example = _load_example()
        rng = np.random.default_rng(0)
        example.write_ehr(root / "mimic-iv", cls.N_PATIENTS, rng)
        example.write_images(root / "mimic-cxr", cls.N_PATIENTS, rng)
        example.write_imagenome(root / "chest-imagenome", cls.N_PATIENTS, rng)
        cls.dataset = MIMIC4ImaGenomeDataset(
            imagenome_root=str(root / "chest-imagenome"),
            imagenome_tables=["silver_images", "silver_comparisons",
                              "gold_comparisons"],
            ehr_root=str(root / "mimic-iv"),
            cxr_root=str(root / "mimic-cxr"),
            ehr_tables=["chartevents", "labevents"],
            cxr_tables=["metadata"],
            cache_dir=str(root / "cache"),
        )
        cls.task = DiseaseProgressionMIMIC4(image_size=16)
        cls.gold = cls.task.gold_subjects(cls.dataset)
        cls.splits = cls.dataset.set_task(cls.task, split=GoldTestSplit(cls.gold))

    @classmethod
    def tearDownClass(cls):
        for split in cls.splits:
            split.close()
        cls.tmp.cleanup()

    def test_one_patient_has_gold_labels(self):
        # write_imagenome gives gold comparisons to the last patient only
        self.assertEqual(self.gold, [str(10000000 + self.N_PATIENTS - 1)])

    def test_gold_pairs_are_in_test_and_labelled(self):
        train, val, test = self.splits
        self.assertGreater(len(train), 0)
        test_samples = [test[i] for i in range(len(test))]
        # the gold patient's two consecutive pairs are in test, labelled from
        # gold; test may also hold other patients' (silver) pairs
        gold_pairs = [s for s in test_samples if int(s["from_gold"]) == 1]
        self.assertEqual(len(gold_pairs), 2)
        for sample in gold_pairs:
            self.assertEqual(str(sample["patient_id"]), self.gold[0])
            self.assertEqual(tuple(sample["progression"].shape), (21,))
            self.assertEqual(int(sample["progression"].sum()), 1)
        for split in (train, val):
            for i in range(len(split)):
                self.assertNotEqual(str(split[i]["patient_id"]), self.gold[0])
                self.assertEqual(int(split[i]["from_gold"]), 0)

    def test_samples_join_ehr_and_images(self):
        sample = self.splits[0][0]
        images, hours, _ = sample["cxr"]
        self.assertEqual(tuple(images.shape)[1:], (3, 16, 16))
        self.assertEqual(sample["ehr"].dim(), 2)
        self.assertGreater(len(hours), 1)


if __name__ == "__main__":
    unittest.main()

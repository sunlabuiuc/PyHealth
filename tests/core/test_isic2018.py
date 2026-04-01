"""
Unit tests for the ISIC2018Dataset and ISIC2018Classification classes.

Author:
    Fan Zhang (fanz6@illinois.edu)
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pyhealth.datasets import ISIC2018Dataset
from pyhealth.tasks import ISIC2018Classification


class TestISIC2018Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = (
            Path(__file__).parent.parent.parent
            / "test-resources"
            / "core"
            / "isic2018"
        )
        cls.generate_fake_images()
        cls.cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.dataset = ISIC2018Dataset(cls.root, cache_dir=cls.cache_dir.name)
        cls.samples = cls.dataset.set_task()

    @classmethod
    def tearDownClass(cls):
        cls.samples.close()
        (cls.root / "isic2018-metadata-pyhealth.csv").unlink(missing_ok=True)
        cls.delete_fake_images()
        try:
            cls.cache_dir.cleanup()
        except Exception:
            pass
        cls.cache_dir = None

    @classmethod
    def generate_fake_images(cls):
        images_dir = cls.root / "ISIC2018_Task3_Training_Input"
        with open(cls.root / "ISIC2018_Task3_Training_GroundTruth.csv", "r") as f:
            lines = f.readlines()

        for line in lines[1:]:  # Skip header row
            image_id = line.split(",")[0].strip()
            img = Image.fromarray(
                np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            )
            img.save(images_dir / f"{image_id}.jpg")

    @classmethod
    def delete_fake_images(cls):
        for jpg in (cls.root / "ISIC2018_Task3_Training_Input").glob("*.jpg"):
            jpg.unlink()

    def test_stats(self):
        self.dataset.stats()

    def test_num_patients(self):
        # Each ISIC image is its own patient
        self.assertEqual(len(self.dataset.unique_patient_ids), 10)

    def test_default_task(self):
        self.assertIsInstance(self.dataset.default_task, ISIC2018Classification)

    def test_metadata_csv_created(self):
        self.assertTrue(
            (self.root / "isic2018-metadata-pyhealth.csv").exists()
        )

    def test_event_fields(self):
        # Patient ID equals image ID for ISIC images
        patient = self.dataset.get_patient("ISIC_0024307")
        events = patient.get_events()

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["image_id"], "ISIC_0024307")
        self.assertEqual(events[0]["mel"], "0.0")
        self.assertEqual(events[0]["nv"], "1.0")
        self.assertEqual(events[0]["bcc"], "0.0")
        self.assertEqual(events[0]["akiec"], "0.0")
        self.assertEqual(events[0]["bkl"], "0.0")
        self.assertEqual(events[0]["df"], "0.0")
        self.assertEqual(events[0]["vasc"], "0.0")

    def test_event_fields_mel(self):
        patient = self.dataset.get_patient("ISIC_0024310")
        events = patient.get_events()

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["mel"], "1.0")
        self.assertEqual(events[0]["nv"], "0.0")

    def test_all_label_columns_present(self):
        # All 7 class columns must be accessible on every event
        for cls in ISIC2018Dataset.classes:
            for pid in self.dataset.unique_patient_ids:
                event = self.dataset.get_patient(pid).get_events()[0]
                self.assertIn(cls, event)

    def test_image_paths_exist(self):
        import os
        for pid in self.dataset.unique_patient_ids:
            event = self.dataset.get_patient(pid).get_events()[0]
            self.assertTrue(os.path.isfile(event["path"]))

    def test_set_task_injects_rgb_processor(self):
        # RGB images should produce 3-channel tensors (C, H, W)
        sample = self.samples[0]
        self.assertEqual(sample["image"].shape[0], 3)

    def test_num_samples(self):
        self.assertEqual(len(self.samples), 10)

    def test_sample_labels(self):
        actual_labels = [sample["label"].item() for sample in self.samples]

        # Only 3 classes appear in fixture, sorted alphabetically: bkl=0, mel=1, nv=2
        # Real labels: NV, NV, NV, NV, MEL, NV, BKL, MEL, NV, MEL
        expected_labels = [2, 2, 2, 2, 1, 2, 0, 1, 2, 1]
        self.assertCountEqual(actual_labels, expected_labels)

    def test_mel_count(self):
        mel_samples = [s for s in self.samples if s["label"].item() == 1]
        self.assertEqual(len(mel_samples), 3)

    def test_nv_count(self):
        nv_samples = [s for s in self.samples if s["label"].item() == 2]
        self.assertEqual(len(nv_samples), 6)

    def test_verify_data_missing_root(self):
        with self.assertRaises(FileNotFoundError):
            ISIC2018Dataset(root="/nonexistent/path")

    def test_verify_data_missing_csv(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                ISIC2018Dataset(root=tmpdir)


if __name__ == "__main__":
    unittest.main()

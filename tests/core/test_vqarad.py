import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl
import torch
from PIL import Image

from pyhealth.data import Patient
from pyhealth.datasets import VQARADDataset
from pyhealth.processors import ImageProcessor
from pyhealth.tasks import MedicalVQATask


class TestMedicalVQATask(unittest.TestCase):
    def test_generates_samples_from_vqarad_events(self):
        task = MedicalVQATask()
        patient = Patient(
            patient_id="patient-1",
            data_source=pl.DataFrame(
                {
                    "patient_id": ["patient-1", "patient-1"],
                    "event_type": ["vqarad", "vqarad"],
                    "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                    "vqarad/image_path": ["/tmp/img1.png", "/tmp/img2.png"],
                    "vqarad/question": ["What is shown?", "Is there a fracture?"],
                    "vqarad/answer": ["lung", "no"],
                }
            ),
        )

        samples = task(patient)

        self.assertEqual(task.input_schema, {"image": "image", "question": "text"})
        self.assertEqual(task.output_schema, {"answer": "multiclass"})
        self.assertEqual(len(samples), 2)
        self.assertEqual(
            samples[0],
            {
                "patient_id": "patient-1",
                "image": "/tmp/img1.png",
                "question": "What is shown?",
                "answer": "lung",
            },
        )
        self.assertEqual(samples[1]["patient_id"], "patient-1")
        self.assertEqual(samples[1]["answer"], "no")


class TestVQARADDataset(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        (self.root / "images").mkdir()
        self.cache_dir = tempfile.TemporaryDirectory()

        self.entries = [
            {
                "IMAGE_PATH": "img1.png",
                "QUESTION": "What organ is shown?",
                "ANSWER": "chest",
                "ANSWER_TYPE": "open",
                "QUESTION_TYPE": "organ",
                "IMAGE_ORGAN": "chest",
            },
            {
                "IMAGES_PATH": "img2.png",
                "QUESTION": "Is there a fracture?",
                "ANSWER": "no",
                "ANSWER_TYPE": "closed",
                "QUESTION_TYPE": "abnormality",
                "IMAGE_ORGAN": "arm",
            },
        ]

        with (self.root / "VQA_RAD Dataset Public.json").open("w", encoding="utf-8") as f:
            json.dump(self.entries, f)

        for image_name in ("img1.png", "img2.png"):
            Image.new("RGB", (16, 16), color=(255, 0, 0)).save(
                self.root / "images" / image_name
            )

        self.sample_dataset = None

    def tearDown(self):
        if self.sample_dataset is not None:
            self.sample_dataset.close()
        self.cache_dir.cleanup()
        self.tmpdir.cleanup()

    def test_prepare_metadata_creates_expected_csv(self):
        dataset = VQARADDataset.__new__(VQARADDataset)
        dataset.prepare_metadata(str(self.root))

        metadata_path = self.root / "vqarad-metadata-pyhealth.csv"
        self.assertTrue(metadata_path.exists())

        df = pd.read_csv(metadata_path)
        self.assertEqual(
            list(df.columns),
            [
                "image_path",
                "question",
                "answer",
                "answer_type",
                "question_type",
                "image_organ",
            ],
        )
        self.assertEqual(df.loc[0, "image_path"], str(self.root / "images" / "img1.png"))
        self.assertEqual(df.loc[1, "image_path"], str(self.root / "images" / "img2.png"))
        self.assertEqual(df.loc[1, "answer"], "no")

    def test_set_task_builds_samples_and_uses_image_processor(self):
        dataset = VQARADDataset(
            root=str(self.root),
            cache_dir=self.cache_dir.name,
        )

        self.assertIsInstance(dataset.default_task, MedicalVQATask)

        self.sample_dataset = dataset.set_task()

        self.assertEqual(len(self.sample_dataset), 2)
        self.assertIn("image", self.sample_dataset.input_processors)
        self.assertIsInstance(
            self.sample_dataset.input_processors["image"],
            ImageProcessor,
        )
        self.assertIn("answer", self.sample_dataset.output_processors)
        self.assertEqual(self.sample_dataset.output_processors["answer"].size(), 2)

        sample = self.sample_dataset[0]
        self.assertIn("patient_id", sample)
        self.assertIsInstance(sample["image"], torch.Tensor)
        self.assertEqual(tuple(sample["image"].shape), (3, 224, 224))
        self.assertIsInstance(sample["question"], str)
        self.assertIsInstance(sample["answer"], torch.Tensor)


if __name__ == "__main__":
    unittest.main()

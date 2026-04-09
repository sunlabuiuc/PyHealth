import json
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path

import torch
from PIL import Image

from pyhealth.datasets import VQARADDataset
from pyhealth.processors import ImageProcessor
from pyhealth.tasks import MedicalVQATask

warnings.filterwarnings(
    "ignore",
    message=r"A newer version of litdata is available .*",
    category=UserWarning,
)


class TestVQARADDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root_dir = tempfile.mkdtemp()
        cls.cache_dir = tempfile.mkdtemp()
        cls.root = Path(cls.root_dir)
        cls.image_dir = cls.root / "VQA_RAD Image Folder"
        cls.image_dir.mkdir(parents=True, exist_ok=True)

        entries = []
        for idx, (question, answer, organ) in enumerate(
            [
                ("is there a fracture", "yes", "chest"),
                ("is the study normal", "no", "head"),
                ("is there edema", "yes", "abdomen"),
            ]
        ):
            image_name = f"study_{idx}.png"
            image = Image.fromarray(
                torch.randint(0, 255, (12, 12, 3), dtype=torch.uint8).numpy(),
                mode="RGB",
            )
            image.save(cls.image_dir / image_name)
            entries.append(
                {
                    "image_name": image_name,
                    "question": question,
                    "answer": answer,
                    "answer_type": "closed",
                    "question_type": "presence",
                    "image_organ": organ,
                }
            )

        with (cls.root / "VQA_RAD Dataset Public.json").open("w", encoding="utf-8") as f:
            json.dump(entries, f)

        cls.dataset = VQARADDataset(
            root=str(cls.root),
            cache_dir=cls.cache_dir,
            num_workers=1,
        )
        cls.samples = cls.dataset.set_task(
            num_workers=1,
            image_processor=ImageProcessor(mode="RGB", image_size=16),
        )

    @classmethod
    def tearDownClass(cls):
        cls.samples.close()
        shutil.rmtree(cls.root_dir)
        shutil.rmtree(cls.cache_dir)

    def test_prepare_metadata_creates_expected_csv(self):
        metadata_path = self.root / "vqarad-metadata-pyhealth.csv"
        self.assertTrue(metadata_path.exists())

        with metadata_path.open("r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")

        self.assertEqual(
            header,
            [
                "image_path",
                "question",
                "answer",
                "answer_type",
                "question_type",
                "image_organ",
            ],
        )

    def test_dataset_initialization(self):
        self.assertEqual(self.dataset.dataset_name, "vqarad")
        self.assertEqual(self.dataset.root, str(self.root))
        self.assertEqual(len(self.dataset.unique_patient_ids), 3)

    def test_get_patient_and_event_parsing(self):
        patient = self.dataset.get_patient("0")
        events = patient.get_events(event_type="vqarad")

        self.assertEqual(patient.patient_id, "0")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].question, "is there a fracture")
        self.assertEqual(events[0].answer, "yes")
        self.assertEqual(events[0].answer_type, "closed")
        self.assertEqual(events[0].question_type, "presence")
        self.assertEqual(events[0].image_organ, "chest")
        self.assertTrue(events[0].image_path.endswith("study_0.png"))

    def test_default_task(self):
        self.assertIsInstance(self.dataset.default_task, MedicalVQATask)

    def test_set_task_returns_processed_samples(self):
        self.assertEqual(len(self.samples), 3)

        sample = self.samples[0]
        self.assertEqual(sample["question"], "is there a fracture")
        self.assertEqual(sample["patient_id"], "0")
        self.assertIsInstance(sample["answer"], torch.Tensor)
        self.assertEqual(sample["answer"].ndim, 0)
        self.assertEqual(tuple(sample["image"].shape), (3, 16, 16))


if __name__ == "__main__":
    unittest.main()

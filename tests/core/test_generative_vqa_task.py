import json
import tempfile
import unittest
import warnings
from pathlib import Path

from PIL import Image

from pyhealth.datasets import VQARADDataset
from pyhealth.tasks import GenerativeMedicalVQA


class TestGenerativeMedicalVQATask(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        images = self.root / "images"
        images.mkdir(parents=True, exist_ok=True)

        Image.new("RGB", (8, 8), color=(10, 20, 30)).save(images / "a.png")
        Image.new("RGB", (8, 8), color=(30, 20, 10)).save(images / "b.png")

        annotations = [
            {
                "image": "a.png",
                "question": "what modality?",
                "answer": "xray",
                "split": "train",
                "question_id": "gq1",
                "image_id": "gi1",
            },
            {
                "image": "b.png",
                "question": "is lesion present?",
                "answer": "no",
                "split": "test",
                "question_id": "gq2",
                "image_id": "gi2",
            },
        ]
        with open(self.root / "annotations.json", "w", encoding="utf-8") as file:
            json.dump(annotations, file)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_schema_correctness(self):
        self.assertEqual(GenerativeMedicalVQA.task_name, "GenerativeMedicalVQA")
        self.assertEqual(
            GenerativeMedicalVQA.input_schema,
            {
                "image_path": "raw",
                "question": "raw",
                "split": "raw",
                "question_id": "raw",
                "image_id": "raw",
                "dataset": "raw",
            },
        )
        self.assertEqual(GenerativeMedicalVQA.output_schema, {"answer": "raw"})

    def test_sample_keys_and_record_id(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = VQARADDataset(
                root=str(self.root),
                images_dir="images",
                cache_dir=str(self.root / "cache"),
            )

        with dataset.set_task(task=GenerativeMedicalVQA()) as sample_dataset:
            sample = sample_dataset[0]
            self.assertIn("patient_id", sample)
            self.assertIn("record_id", sample)
            self.assertIn("image_path", sample)
            self.assertIn("question", sample)
            self.assertIn("answer", sample)
            self.assertIn("split", sample)
            self.assertIn("question_id", sample)
            self.assertIn("image_id", sample)
            self.assertIn("dataset", sample)

            self.assertEqual(sample["record_id"], sample["question_id"])

    def test_split_filtering(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = VQARADDataset(
                root=str(self.root),
                images_dir="images",
                cache_dir=str(self.root / "cache"),
            )

        with dataset.set_task(task=GenerativeMedicalVQA(split="test")) as sample_dataset:
            self.assertEqual(len(sample_dataset), 1)
            sample = sample_dataset[0]
            self.assertEqual(sample["split"], "test")
            self.assertEqual(sample["record_id"], "gq2")


if __name__ == "__main__":
    unittest.main()

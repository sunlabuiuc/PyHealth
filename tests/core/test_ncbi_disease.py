"""Unit tests for the NCBI Disease dataset and task."""

import shutil
import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import NCBIDiseaseDataset
from pyhealth.tasks import (
    NCBIDiseaseConceptClassification,
    NCBIDiseaseRecognition,
)


class TestNCBIDiseaseDataset(unittest.TestCase):
    """Tests for loading the synthetic NCBI Disease corpus."""

    @classmethod
    def setUpClass(cls):
        cls.test_resources = (
            Path(__file__).parent.parent.parent / "test-resources" / "ncbi_disease"
        )
        cls.root_dir = tempfile.TemporaryDirectory()
        cls.cache_dir = tempfile.TemporaryDirectory()

        for path in cls.test_resources.glob("NCBI*_corpus.txt"):
            shutil.copy(path, Path(cls.root_dir.name) / path.name)

        cls.dataset = NCBIDiseaseDataset(
            root=cls.root_dir.name,
            cache_dir=cls.cache_dir.name,
            num_workers=1,
        )
        cls.recognition_samples = cls.dataset.set_task(NCBIDiseaseRecognition())
        cls.abstract_samples = cls.dataset.set_task(
            NCBIDiseaseRecognition(text_source="abstract")
        )
        cls.concept_train_samples = cls.dataset.set_task(
            NCBIDiseaseConceptClassification(split="train")
        )
        cls.concept_dev_samples = cls.dataset.set_task(
            NCBIDiseaseConceptClassification(split="dev")
        )
        cls.concept_abstract_samples = cls.dataset.set_task(
            NCBIDiseaseConceptClassification(text_source="abstract")
        )

    @classmethod
    def tearDownClass(cls):
        cls.recognition_samples.close()
        cls.abstract_samples.close()
        cls.concept_train_samples.close()
        cls.concept_dev_samples.close()
        cls.concept_abstract_samples.close()
        cls.cache_dir.cleanup()
        cls.root_dir.cleanup()

    def test_dataset_initialization(self):
        self.assertEqual(self.dataset.dataset_name, "ncbi_disease")

    def test_stats(self):
        self.dataset.stats()

    def test_duplicate_pmids_keep_unique_records(self):
        self.assertEqual(len(self.dataset.unique_patient_ids), 5)

    def test_default_task(self):
        self.assertIsInstance(self.dataset.default_task, NCBIDiseaseRecognition)

    def test_task_output_format(self):
        self.assertGreater(len(self.recognition_samples), 0)

        sample = self.recognition_samples[0]
        self.assertIn("text", sample)
        self.assertIn("entities", sample)
        self.assertIn("concept_ids", sample)
        self.assertIn("split", sample)

    def test_abstract_only_offsets_are_shifted(self):
        sample = next(
            item for item in self.abstract_samples if item["document_id"] == "1001"
        )

        self.assertEqual(sample["text"], "Asthma worsens quickly.")
        self.assertEqual(sample["entities"][0]["start"], 0)
        self.assertEqual(sample["entities"][0]["text"], "Asthma")

    def test_concept_classification_train_split(self):
        self.assertEqual(len(self.concept_train_samples), 3)

    def test_concept_classification_labels(self):
        sample = next(
            item for item in self.concept_dev_samples if item["document_id"] == "2001"
        )

        processor = self.concept_dev_samples.output_processors["concept_ids"]
        labels_by_index = {
            index: label for label, index in processor.label_vocab.items()
        }
        decoded = [
            labels_by_index[index]
            for index, value in enumerate(sample["concept_ids"].tolist())
            if value == 1.0
        ]

        self.assertEqual(sample["text"], "Asthma Type 2 diabetes is common.")
        self.assertEqual(decoded, ["D001249", "D003924"])

    def test_concept_classification_abstract_filters_title_labels(self):
        sample = next(
            item
            for item in self.concept_abstract_samples
            if item["document_id"] == "2001"
        )

        processor = self.concept_abstract_samples.output_processors["concept_ids"]
        labels_by_index = {
            index: label for label, index in processor.label_vocab.items()
        }
        decoded = [
            labels_by_index[index]
            for index, value in enumerate(sample["concept_ids"].tolist())
            if value == 1.0
        ]

        self.assertEqual(sample["text"], "Type 2 diabetes is common.")
        self.assertEqual(decoded, ["D003924"])


class TestNCBIDiseaseRecognition(unittest.TestCase):
    """Tests for task utilities."""

    def test_entities_to_bio_tags(self):
        text = "Asthma worsens quickly."
        entities = [
            {
                "text": "Asthma",
                "type": "SpecificDisease",
                "concept_id": "D001249",
                "start": 0,
                "end": 6,
            }
        ]

        tokens, tags = NCBIDiseaseRecognition.entities_to_bio_tags(text, entities)

        self.assertEqual(tokens, ["Asthma", "worsens", "quickly."])
        self.assertEqual(tags, ["B-Disease", "O", "O"])


if __name__ == "__main__":
    unittest.main()

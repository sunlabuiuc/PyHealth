"""
Optimized Unit Tests for SleepQA.
Fixed for PyHealth 2.x Polars backend (tables as a list).
"""
import json
import unittest
import shutil
import gc
import time
import os
from pathlib import Path
import torch

from pyhealth.datasets.sleepqa import SleepQADataset
from pyhealth.tasks.sleepqa_extractive_qa import SleepQAExtractiveQA
from pyhealth.models.sleepqa_biobert import SleepQABioBERT

class TestSleepQAPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1. Setup local test directory
        cls.root = Path("./ph_test_tmp")
        if cls.root.exists():
            shutil.rmtree(cls.root, ignore_errors=True)
        cls.root.mkdir(parents=True, exist_ok=True)

        # 2. Synthetic Data
        data = {"data": [{"passage_id": "p1", "text": "Sleep is vital.",
                "qas": [{"id": "q1", "question": "What is vital?", 
                         "answers": [{"text": "Sleep", "answer_start": 0}]}]}]}
        
        with open(cls.root / "sleepqa.json", "w", encoding="utf-8") as f:
            json.dump(data, f)

        # 3. Initialize Dataset
        cls.dataset = SleepQADataset(
            root=str(cls.root), 
            cache_dir=str(cls.root / "cache")
        )

    @classmethod
    def tearDownClass(cls):
        """Aggressive cleanup for Windows file locks."""
        if hasattr(cls, 'dataset'):
            # Close handle if the version supports it
            if hasattr(cls.dataset, 'close'): cls.dataset.close()
            del cls.dataset
        gc.collect()
        time.sleep(0.5) 
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_dataset_integrity(self):
        """Verifies the dataset registry loaded the sleepqa table."""
        # FIX: Check the list membership, not dictionary indexing
        self.assertIn("sleepqa", self.dataset.tables, "Table 'sleepqa' not registered.")
        
        # Verify the CSV was actually created in the root
        expected_csv = Path(self.dataset.root) / "sleepqa-metadata-pyhealth.csv"
        self.assertTrue(expected_csv.exists(), "Metadata CSV was not generated.")

    def test_task_extraction(self):
        """Verifies that the task can pull samples from the backend."""
        # Success here proves the data in the Polars backend is valid
        qa_dataset = self.dataset.set_task(SleepQAExtractiveQA())
        self.assertGreater(len(qa_dataset), 0)
        self.assertEqual(qa_dataset[0]["answer_text"], "Sleep")

    def test_model_forward(self):
        """Verifies the model forward pass with tiny weights."""
        qa_dataset = self.dataset.set_task(SleepQAExtractiveQA())
        model = SleepQABioBERT(
            dataset=qa_dataset, 
            model_name="sshleifer/tiny-distilbert-base-cased-distilled-squad"
        )
        batch = {"passage": ["test"], "question": ["test"]}
        with torch.no_grad():
            outputs = model(**batch)
        self.assertIn("logit", outputs)

if __name__ == "__main__":
    unittest.main()
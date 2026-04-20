import os
import unittest
from pathlib import Path

from pyhealth.datasets import CaReSoundDataset
from pyhealth.tasks import CaReSoundAQA


class TestCaReSoundDataset(unittest.TestCase):
    """Test cases for the CaReSoundDataset."""

    @classmethod
    def setUpClass(cls):
        """Set up test resources path pointing to the PyHealthCS598/test-resources folder."""
        # This navigates up from tests/datasets/test_caresound.py to the project root
        cls.test_resources = Path(__file__).parent.parent.parent / "test-resources" / "caresound" / "datasets"
        
        # Ensure the directory actually exists to prevent confusing errors
        if not cls.test_resources.exists():
            raise FileNotFoundError(
                f"Test resources not found at {cls.test_resources}. "
                "Please ensure your sample audio and CaReSoundQA.csv are placed there."
            )

    def test_dataset_initialization(self):
        """Test that the dataset initializes correctly from the test-resources folder."""
        dataset = CaReSoundDataset(root=str(self.test_resources))
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.dataset_name, "caresound")
        
    def test_stats(self):
        """Test that stats() runs without error."""
        dataset = CaReSoundDataset(root=str(self.test_resources))
        import sys, io
        captured_output = io.StringIO()
        sys.stdout = captured_output
        dataset.stats()
        sys.stdout = sys.__stdout__
        
        # Updated to match the actual PyHealth output format!
        self.assertIn("Dataset: caresound", captured_output.getvalue())

    def test_default_task(self):
        """Test that the default task is properly assigned to CaReSoundAQA."""
        dataset = CaReSoundDataset(root=str(self.test_resources))
        self.assertIsInstance(dataset.default_task, CaReSoundAQA)

    def test_set_task(self):
        """Test applying the CaReSoundAQA task to the dataset."""
        dataset = CaReSoundDataset(root=str(self.test_resources))
        task = CaReSoundAQA()
        samples = dataset.set_task(task)

        # Ensure the task actually generated samples
        self.assertGreater(len(samples), 0)
        
        # Verify the schema of the first sample
        sample = samples[0]
        self.assertIn("patient_id", sample)
        self.assertIn("question", sample)
        self.assertIn("answer", sample)
        self.assertIn("audio_path", sample)


class TestCaReSoundAQA(unittest.TestCase):
    """Test cases for the CaReSoundAQA task schema and utilities."""

    def setUp(self):
        self.task = CaReSoundAQA()

    def test_task_attributes(self):
        """Test task class attributes."""
        self.assertEqual(self.task.task_name, "CaReSoundAQA")
        self.assertIn("question", self.task.input_schema)
        self.assertEqual(self.task.input_schema["question"], "text")
        self.assertIn("answer", self.task.output_schema)
        self.assertEqual(self.task.output_schema["answer"], "text")

    def test_safe_str(self):
        """Test the string safety utility."""
        self.assertEqual(self.task._safe_str("Hello"), "Hello")
        self.assertEqual(self.task._safe_str(None, default="N/A"), "N/A")
        self.assertEqual(self.task._safe_str("nan", default="missing"), "missing")


if __name__ == "__main__":
    unittest.main()
import unittest
import tempfile
import pandas as pd
from pathlib import Path
from pyhealth.datasets import CaReSoundDataset
from pyhealth.tasks import CaReSoundAQA


class TestCaReSoundDataset(unittest.TestCase):
    """Test cases for the CaReSoundDataset using synthetic data."""

    def setUp(self):
        """Create a temporary directory with synthetic data."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.test_dir.name)

        # 1. Create synthetic CSV (2 patients)
        self.df = pd.DataFrame(
            {
                "patient_id": ["101", "102"],
                "dataset": ["icbhi", "circor"],
                "question": ["Is this normal?", "Any abnormalities?"],
                "answer": ["Normal", "Abnormal"],
                "hf_split": ["train", "test"],
                "metadata/audio_path": ["icbhi_101.wav", "circor_102.wav"],
            }
        )
        self.df.to_csv(self.root / "caresound_metadata.csv", index=False)

        # 2. Create dummy audio files (just empty files are enough for path matching)
        (self.root / "icbhi_101.wav").touch()
        (self.root / "circor_102.wav").touch()

    def tearDown(self):
        """Cleanup temporary directory."""
        self.test_dir.cleanup()

    def test_dataset_initialization(self):
        dataset = CaReSoundDataset(root=str(self.root))
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.dataset_name, "caresound")

    def test_default_task(self):
        dataset = CaReSoundDataset(root=str(self.root))
        self.assertIsInstance(dataset.default_task, CaReSoundAQA)

    def test_set_task(self):
        dataset = CaReSoundDataset(root=str(self.root))
        task = CaReSoundAQA()
        samples = dataset.set_task(task)

        # We expect 2 samples since we created 2 patients
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["patient_id"], "101")


class TestCaReSoundAQA(unittest.TestCase):
    """Test cases for the CaReSoundAQA task schema."""

    def setUp(self):
        self.task = CaReSoundAQA()

    def test_task_attributes(self):
        self.assertEqual(self.task.task_name, "CaReSoundAQA")
        self.assertIn("question", self.task.input_schema)


if __name__ == "__main__":
    unittest.main()

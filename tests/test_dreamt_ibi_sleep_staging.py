import unittest
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

from pyhealth.datasets import DREAMTDataset
from pyhealth.tasks.dreamt_ibi_sleep_staging import DreamtIBISleepStagingTask

class TestDreamtIBISleepStaging(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        root = Path(cls.temp_dir)

        # Create DREAMT-like folder structure
        (root / "data_64Hz").mkdir()
        (root / "data_100Hz").mkdir()

        # Create minimal participant_info.csv
        df = pd.DataFrame({
            "SID": ["S001"],
            "AGE": [30],
            "GENDER": ["M"],
            "BMI": [22],
            "OAHI": [5],
            "AHI": [6],
            "Mean_SaO2": ["95%"],
            "Arousal Index": [12],
            "MEDICAL_HISTORY": ["None"],
            "Sleep_Disorders": ["None"],
        })
        df.to_csv(root / "participant_info.csv", index=False)

        # Create a simple 64Hz CSV file containing IBI + stage
        ibi = np.random.randint(500, 1000, size=120)
        stages = np.random.choice(["W", "N1", "N2", "N3", "R"], size=120)
        pd.DataFrame({
            "IBI": ibi,
            "Sleep_Stage": stages
        }).to_csv(root / "data_64Hz" / "S001_whole_df.csv", index=False)

        # Create dummy PSG 100Hz file (unused but required by DREAMT loader)
        pd.DataFrame({"TEMP": [1]}).to_csv(root / "data_100Hz" / "S001_PSG_df.csv")

        cls.dataset = DREAMTDataset(root=str(root))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_task_generates_samples(self):
        task = DreamtIBISleepStagingTask(window_size=30)
        patient = self.dataset.get_patient("S001")
        samples = task(patient)
        self.assertGreater(len(samples), 0)

    def test_sample_format(self):
        task = DreamtIBISleepStagingTask(window_size=30)
        patient = self.dataset.get_patient("S001")
        sample = task(patient)[0]

        self.assertIn("ibi_seq", sample)
        self.assertIn("stage", sample)
        self.assertEqual(len(sample["ibi_seq"]), 30)
        self.assertIsInstance(sample["stage"], int)


if __name__ == "__main__":
    unittest.main()
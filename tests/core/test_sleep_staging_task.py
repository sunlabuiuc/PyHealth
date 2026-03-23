import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pyhealth.datasets import DREAMTDataset
from pyhealth.tasks import SleepStagingDREAMT


def _create_task_test_root(root: Path) -> Path:
    dreamt_root = root / "dreamt"
    (dreamt_root / "data_64Hz").mkdir(parents=True)

    participant_info = pd.DataFrame(
        {
            "SID": ["S001"],
            "AGE": [29.0],
            "GENDER": ["F"],
            "BMI": [22.0],
            "OAHI": [1.0],
            "AHI": [1.0],
            "Mean_SaO2": ["98%"],
            "Arousal Index": [9.0],
            "MEDICAL_HISTORY": ["None"],
            "Sleep_Disorders": ["None"],
        }
    )
    participant_info.to_csv(dreamt_root / "participant_info.csv", index=False)

    stages = ["P"] * 10 + ["W"] * 10 + ["N1"] * 10 + ["R"] * 10
    timestamps = [index / 64.0 for index in range(len(stages))]
    frame = pd.DataFrame(
        {
            "TIMESTAMP": timestamps,
            "IBI": [0.8 + 0.01 * index for index in range(len(stages))],
            "HR": [60.0 + index for index in range(len(stages))],
            "BVP": [0.1] * len(stages),
            "EDA": [0.01] * len(stages),
            "TEMP": [33.2] * len(stages),
            "ACC_X": [0.0] * len(stages),
            "ACC_Y": [1.0] * len(stages),
            "ACC_Z": [2.0] * len(stages),
            "Sleep_Stage": stages,
        }
    )
    frame.to_csv(dreamt_root / "data_64Hz" / "S001_whole_df.csv", index=False)
    return dreamt_root


class TestSleepStagingDREAMT(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.root = _create_task_test_root(self.temp_dir)
        self.dataset = DREAMTDataset(
            root=str(self.root),
            cache_dir=self.temp_dir / "cache",
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_task_maps_labels_and_extracts_fixed_windows(self):
        task = SleepStagingDREAMT(
            window_size=10,
            stride=10,
            source_preference="wearable",
        )
        patient = self.dataset.get_patient("S001")

        samples = task(patient)

        self.assertEqual(len(samples), 3)
        self.assertEqual([sample["label"] for sample in samples], [0, 1, 2])
        self.assertEqual(samples[0]["signal"].shape, (10, 8))
        self.assertEqual(samples[0]["patient_id"], "S001")
        self.assertIn("record_id", samples[0])

    def test_set_task_builds_sample_dataset(self):
        task = SleepStagingDREAMT(
            window_size=10,
            stride=10,
            source_preference="wearable",
        )
        sample_dataset = self.dataset.set_task(task=task, num_workers=1)

        self.assertEqual(len(sample_dataset), 3)
        sample = sample_dataset[0]
        self.assertIn("signal", sample)
        self.assertIn("label", sample)
        self.assertEqual(tuple(sample["signal"].shape), (10, 8))


if __name__ == "__main__":
    unittest.main()

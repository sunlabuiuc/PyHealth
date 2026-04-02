import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pyhealth.datasets import DREAMTDataset
from pyhealth.tasks import SleepStagingDREAMT, SleepStagingDREAMTSeq


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


def _create_seq_task_test_root(root: Path) -> Path:
    dreamt_root = root / "dreamt_seq"
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

    stage_blocks = [
        ("P", 30),
        ("W", 30),
        ("N2", 30),
        ("R", 30),
    ]
    labels = [label for label, count in stage_blocks for _ in range(count)]
    timestamps = np.arange(len(labels), dtype=np.float32)
    frame = pd.DataFrame(
        {
            "TIMESTAMP": timestamps,
            "IBI": np.linspace(0.8, 1.3, len(labels)),
            "HR": np.linspace(60.0, 72.0, len(labels)),
            "BVP": np.linspace(0.1, 0.4, len(labels)),
            "EDA": np.linspace(0.01, 0.03, len(labels)),
            "TEMP": np.full(len(labels), 33.2),
            "ACC_X": np.zeros(len(labels)),
            "ACC_Y": np.ones(len(labels)),
            "ACC_Z": np.full(len(labels), 2.0),
            "Sleep_Stage": labels,
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

class TestSleepStagingDREAMTSeq(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.root = _create_seq_task_test_root(self.temp_dir)
        self.dataset = DREAMTDataset(
            root=str(self.root),
            cache_dir=self.temp_dir / "cache",
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_seq_task_builds_epoch_sequence(self):
        task = SleepStagingDREAMTSeq(
            epoch_seconds=30,
            sequence_length=5,
            source_preference="wearable",
        )
        patient = self.dataset.get_patient("S001")

        samples = task(patient)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["signal"].shape, (5, 1))
        self.assertEqual(samples[0]["mask"].tolist(), [1.0, 1.0, 1.0, 0.0, 0.0])
        self.assertEqual(samples[0]["label"][:3].tolist(), [0, 1, 2])

    def test_seq_task_outputs_mask_and_padded_labels(self):
        task = SleepStagingDREAMTSeq(
            epoch_seconds=30,
            sequence_length=5,
            source_preference="wearable",
            ignore_index=-100,
        )
        patient = self.dataset.get_patient("S001")
        sample = task(patient)[0]

        self.assertTrue(np.allclose(sample["signal"][3:], 0.0))
        self.assertEqual(sample["label"][3:].tolist(), [-100, -100])

    def test_seq_task_ibi_only_mode(self):
        task = SleepStagingDREAMTSeq(
            feature_columns=("IBI",),
            epoch_seconds=30,
            sequence_length=5,
            source_preference="wearable",
        )
        patient = self.dataset.get_patient("S001")
        sample = task(patient)[0]

        self.assertEqual(sample["signal"].shape[1], 1)

    def test_seq_set_task_builds_sample_dataset(self):
        task = SleepStagingDREAMTSeq(
            epoch_seconds=30,
            sequence_length=5,
            source_preference="wearable",
        )
        sample_dataset = self.dataset.set_task(task=task, num_workers=1)
        sample = sample_dataset[0]

        self.assertIn("signal", sample)
        self.assertIn("mask", sample)
        self.assertIn("label", sample)
        self.assertEqual(tuple(sample["signal"].shape), (5, 1))


if __name__ == "__main__":
    unittest.main()

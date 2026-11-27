import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pyhealth.datasets import DREAMTDataset
from pyhealth.datasets.dreamt import load_dreamt_frame, map_dreamt_stages
from pyhealth.tasks import SleepStagingIBIDREAMT


def participant_info(num_patients: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "SID": [f"S{i:03d}" for i in range(1, num_patients + 1)],
            "AGE": rng.uniform(25, 65, num_patients),
            "GENDER": rng.choice(["M", "F"], num_patients),
            "BMI": rng.integers(20, 50, num_patients),
            "OAHI": rng.integers(0, 50, num_patients),
            "AHI": rng.integers(0, 50, num_patients),
            "Mean_SaO2": [f"{v}%" for v in rng.integers(85, 99, num_patients)],
            "Arousal Index": rng.integers(1, 100, num_patients),
            "MEDICAL_HISTORY": ["Medical History"] * num_patients,
            "Sleep_Disorders": ["Sleep Disorder"] * num_patients,
        }
    )


def synthetic_bvp(n_samples: int, fs: int = 100, period: int = 100) -> np.ndarray:
    t = np.arange(n_samples) / fs
    waveform = np.sin(2 * np.pi * t / period) + 0.5 * np.sin(
        4 * np.pi * t / period
    )
    waveform[::period] += 2.0
    return waveform.astype(np.float64)


class TestDREAMTDatasetNewerVersions(unittest.TestCase):
    """Tests DREAMT 64 Hz and 100 Hz folder layouts with synthetic files."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
        (self.root / "data_64Hz").mkdir()
        (self.root / "data_100Hz").mkdir()

        self.num_patients = 5
        participant_info(self.num_patients).to_csv(
            self.root / "participant_info.csv", index=False
        )
        self._create_files()

    def _create_files(self):
        rng = np.random.default_rng(0)
        for i in range(1, self.num_patients + 1):
            sid = f"S{i:03d}"
            partial_data = {
                "TIMESTAMP": [rng.uniform(0, 100)],
                "BVP": [rng.uniform(1, 10)],
                "HR": [rng.integers(15, 100)],
                "EDA": [rng.uniform(0, 1)],
                "TEMP": [rng.uniform(20, 30)],
                "ACC_X": [rng.uniform(1, 50)],
                "ACC_Y": [rng.uniform(1, 50)],
                "ACC_Z": [rng.uniform(1, 50)],
                "IBI": [rng.uniform(0.6, 1.2)],
                "Sleep_Stage": [rng.choice(["W", "N1", "N2", "N3", "R"])],
            }
            frame = pd.DataFrame(partial_data)
            frame.to_csv(self.root / "data_64Hz" / f"{sid}_whole_df.csv", index=False)
            frame.to_csv(self.root / "data_100Hz" / f"{sid}_PSG_df.csv", index=False)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_dataset_initialization(self):
        dataset = DREAMTDataset(root=str(self.root))
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.dataset_name, "dreamt_sleep")
        self.assertEqual(dataset.root, str(self.root))
        self.assertIsInstance(dataset.default_task, SleepStagingIBIDREAMT)

    def test_metadata_file_created(self):
        DREAMTDataset(root=str(self.root))
        self.assertTrue((self.root / "dreamt-metadata.csv").exists())

    def test_patient_count(self):
        dataset = DREAMTDataset(root=str(self.root))
        self.assertEqual(len(dataset.unique_patient_ids), self.num_patients)

    def test_stats_method(self):
        dataset = DREAMTDataset(root=str(self.root))
        dataset.stats()

    def test_get_patient(self):
        dataset = DREAMTDataset(root=str(self.root))
        patient = dataset.get_patient("S001")
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, "S001")

    def test_get_patient_not_found(self):
        dataset = DREAMTDataset(root=str(self.root))
        with self.assertRaises(AssertionError):
            dataset.get_patient("S222")

    def test_missing_and_unexpected_file_folder(self):
        dataset = DREAMTDataset(root=str(self.root))
        self.assertIsNone(dataset.get_patient_file("S001", str(self.root), "nope"))
        self.assertIsNone(dataset.get_patient_file("S222", str(self.root), "data"))
        path = dataset.get_patient_file("S001", str(self.root), "data_100Hz")
        self.assertTrue(Path(path).is_file())

    def test_stage_map(self):
        self.assertEqual(
            map_dreamt_stages(["W", "N1", "N2", "N3", "R", "Missing", "P"]).tolist(),
            [0, 1, 2, 3, 4, -1, -1],
        )

    def test_load_dreamt_frame(self):
        path = self.root / "data_100Hz" / "S001_PSG_df.csv"
        frame = load_dreamt_frame(str(path), signal_cols=("BVP", "HR"))
        self.assertEqual(set(frame.columns), {"BVP", "HR", "Sleep_Stage"})
        self.assertEqual(frame["BVP"].dtype, np.float64)
        with self.assertRaises(ValueError):
            load_dreamt_frame(str(path), signal_cols=("missing",))
        with self.assertRaises(FileNotFoundError):
            load_dreamt_frame(str(self.root / "absent.csv"))


class TestDREAMTIBITask(unittest.TestCase):
    """Checks IBI emission on a deterministic synthetic DREAMT night."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
        (self.root / "data_100Hz").mkdir()
        participant_info(1).to_csv(self.root / "participant_info.csv", index=False)

        samples_per_epoch = 100 * 30
        stages = ["W", "N2", "Missing", "R", "W"]
        self.labels = [0, 1, -1, 2, 0]
        frame = pd.DataFrame(
            {
                "BVP": synthetic_bvp(samples_per_epoch * len(stages)),
                "Sleep_Stage": np.repeat(stages, samples_per_epoch),
            }
        )
        frame.to_csv(self.root / "data_100Hz" / "S001_PSG_df.csv", index=False)
        self.dataset = DREAMTDataset(root=str(self.root))

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_direct_task_call(self):
        task = SleepStagingIBIDREAMT(peak_backend="scipy", max_epochs=5)
        patient = self.dataset.get_patient("S001")
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["signal"].shape, (5, 750))
        self.assertEqual(sample["signal"].dtype, np.float32)
        self.assertEqual(sample["label"].tolist(), self.labels)
        self.assertEqual(sample["lengths"], 5)
        self.assertEqual(sample["record_id"], "S001_PSG_df")
        self.assertAlmostEqual(float(np.median(sample["signal"])), 1.0, delta=0.01)

    def test_set_task_integration(self):
        import torch

        task = SleepStagingIBIDREAMT(peak_backend="scipy", max_epochs=5)
        samples = self.dataset.set_task(task, num_workers=1)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(tuple(sample["signal"].shape), (5, 750))
        self.assertEqual(sample["signal"].dtype, torch.float32)
        self.assertEqual(tuple(sample["label"].shape), (5,))
        self.assertEqual(sample["label"].dtype, torch.long)
        self.assertEqual(sample["lengths"], 5)


if __name__ == "__main__":
    unittest.main()

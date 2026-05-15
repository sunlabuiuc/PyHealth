import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pyhealth.datasets import DREAMTDataset


def _create_synthetic_dreamt_root(root: Path) -> Path:
    dreamt_root = root / "physionet.org" / "files" / "dreamt" / "2.1.0"
    (dreamt_root / "data_64Hz").mkdir(parents=True)
    (dreamt_root / "data_100Hz").mkdir(parents=True)

    participant_info = pd.DataFrame(
        {
            "SID": ["S001", "S002"],
            "AGE": [28.0, 35.0],
            "GENDER": ["F", "M"],
            "BMI": [21.0, 24.5],
            "OAHI": [1.0, 2.0],
            "AHI": [3.0, 4.0],
            "Mean_SaO2": ["97%", "96%"],
            "Arousal Index": [10.0, 12.0],
            "MEDICAL_HISTORY": ["None", "None"],
            "Sleep_Disorders": ["OSA", "None"],
        }
    )
    participant_info.to_csv(dreamt_root / "participant_info.csv", index=False)

    stage_pattern = ["P"] * 10 + ["W"] * 10 + ["N2"] * 10 + ["R"] * 10
    timestamps = np.arange(len(stage_pattern), dtype=np.float32) / 64.0

    for patient_id in ["S001", "S002"]:
        frame = pd.DataFrame(
            {
                "TIMESTAMP": timestamps,
                "IBI": np.linspace(0.8, 1.2, len(stage_pattern)),
                "HR": np.linspace(60.0, 68.0, len(stage_pattern)),
                "BVP": np.linspace(0.1, 0.5, len(stage_pattern)),
                "EDA": np.linspace(0.01, 0.04, len(stage_pattern)),
                "TEMP": np.linspace(33.0, 33.5, len(stage_pattern)),
                "ACC_X": np.zeros(len(stage_pattern)),
                "ACC_Y": np.ones(len(stage_pattern)),
                "ACC_Z": np.full(len(stage_pattern), 2.0),
                "Sleep_Stage": stage_pattern,
            }
        )
        frame.to_csv(
            dreamt_root / "data_64Hz" / f"{patient_id}_whole_df.csv",
            index=False,
        )
        frame.to_csv(
            dreamt_root / "data_100Hz" / f"{patient_id}_PSG_df_updated.csv",
            index=False,
        )

    return dreamt_root


class TestDREAMTDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.synthetic_root = _create_synthetic_dreamt_root(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_resolves_nested_root_and_creates_metadata(self):
        dataset = DREAMTDataset(
            root=str(self.temp_dir),
            cache_dir=self.temp_dir / "cache",
        )

        self.assertEqual(
            Path(dataset.root).resolve(),
            self.synthetic_root.resolve(),
        )
        self.assertEqual(dataset.dataset_name, "dreamt_sleep")
        self.assertTrue((self.synthetic_root / "dreamt-metadata.csv").exists())
        self.assertEqual(len(dataset.unique_patient_ids), 2)

    def test_patient_event_contains_signal_references(self):
        dataset = DREAMTDataset(
            root=str(self.synthetic_root),
            preferred_source="psg",
            cache_dir=self.temp_dir / "cache_psg",
        )

        patient = dataset.get_patient("S001")
        event = patient.get_events("dreamt_sleep")[0]

        self.assertTrue(event.signal_file.endswith("S001_PSG_df_updated.csv"))
        self.assertTrue(event.file_64hz.endswith("S001_whole_df.csv"))
        self.assertTrue(event.file_100hz.endswith("S001_PSG_df_updated.csv"))
        self.assertEqual(event.signal_source, "psg")
        self.assertEqual(float(event.sampling_rate_hz), 100.0)

    def test_missing_root_raises_helpful_error(self):
        with self.assertRaises(FileNotFoundError):
            DREAMTDataset(root=str(self.temp_dir / "missing"))

    def test_partial_download_drops_missing_subjects(self):
        missing_wearable_file = self.synthetic_root / "data_64Hz" / "S002_whole_df.csv"
        missing_psg_file = (
            self.synthetic_root / "data_100Hz" / "S002_PSG_df_updated.csv"
        )
        missing_wearable_file.unlink()
        missing_psg_file.unlink()
        metadata_file = self.synthetic_root / "dreamt-metadata.csv"
        if metadata_file.exists():
            metadata_file.unlink()

        dataset = DREAMTDataset(
            root=str(self.synthetic_root),
            cache_dir=self.temp_dir / "cache_partial",
        )

        self.assertEqual(len(dataset.unique_patient_ids), 1)
        self.assertEqual(dataset.unique_patient_ids[0], "S001")


if __name__ == "__main__":
    unittest.main()

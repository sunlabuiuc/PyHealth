"""Integration tests for the NHANES dataset and mortality task."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pyhealth.datasets import NHANESDataset
from pyhealth.tasks import NHANESMortalityPrediction


class TestNHANESIntegration(unittest.TestCase):
    """Tests NHANES dataset loading and task integration."""

    def test_dataset_and_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed/pyhealth"
            activity_dir = processed / "activity"

            activity_dir.mkdir(parents=True)

            metadata = []

            for pid, mortstat, followup in [
                ("1", 0, 120),
                ("2", 1, 30),
            ]:
                activity_path = activity_dir / f"{pid}.parquet"

                epochs = pd.DataFrame({
                    "epoch": np.arange(1, 2017),
                    "day": np.arange(2016) // 288 + 1,
                    "hour": np.arange(2016) % 288 // 12,
                    "raw_intensity": np.full(2016, 100.0),
                    "intensity": np.full(2016, 100.0),
                    "wear_minutes": np.full(2016, 5),
                    "wear_fraction": np.ones(2016),
                })

                epochs.to_parquet(activity_path, index=False)

                metadata.append({
                    "patient_id": pid,
                    "cycle": "2003-2004",
                    "activity_file": (
                        activity_path.relative_to(root).as_posix()
                    ),
                    "num_epochs": 2016,
                    "total_wear_minutes": 10080,
                    "age": 50.0,
                    "gender": 1.0,
                    "race_ethnicity": 3.0,
                    "weight": 70.0,
                    "height": 170.0,
                    "bmi": 24.2,
                    "systolic_bp": 120.0,
                    "diastolic_bp": 80.0,
                    "eligstat": 1,
                    "mortstat": mortstat,
                    "permth_int": followup,
                    "permth_exm": followup,
                })

            pd.DataFrame(metadata).to_parquet(
                processed / "nhanes-metadata.parquet",
                index=False,
            )

            dataset = NHANESDataset(
                root=str(root),
                cache_dir=str(root / "cache"),
                num_workers=1,
            )

            task = NHANESMortalityPrediction(root=root)

            # Validate dataset loading and patient identifiers.
            self.assertEqual(len(dataset.unique_patient_ids), 2)
            self.assertEqual(
                set(dataset.unique_patient_ids), {"1", "2"}
            )

            total_samples = 0

            for pid, expected_label in [("1", 0), ("2", 1)]:
                with self.subTest(patient_id=pid):
                    # Validate patient and event parsing.
                    patient = dataset.get_patient(pid)
                    self.assertEqual(str(patient.patient_id), pid)

                    events = patient.get_events(
                        event_type="participants"
                    )

                    self.assertEqual(len(events), 1)

                    event = events[0]

                    self.assertEqual(event.age, 50.0)
                    self.assertEqual(event.num_epochs, 2016)
                    self.assertEqual(event.total_wear_minutes, 10080)
                    self.assertEqual(event.mortstat, expected_label)

                    activity_path = root / event.activity_file
                    self.assertTrue(activity_path.is_file())

                    # Validate activity data integrity.
                    activity = pd.read_parquet(activity_path)

                    self.assertEqual(activity.shape, (2016, 7))
                    self.assertEqual(
                        activity["wear_minutes"].sum(), 10080
                    )

                    # Validate task processing with a real Patient.
                    samples = task(patient)

                    self.assertEqual(len(samples), 2)
                    self.assertEqual(
                        [sample["window_id"] for sample in samples],
                        [1, 2],
                    )

                    for sample in samples:
                        self.assertEqual(sample["patient_id"], pid)
                        self.assertEqual(sample["label"], expected_label)
                        self.assertEqual(sample["valid_fraction"], 1.0)
                        self.assertEqual(
                            sample["activity"].shape, (1008, 2)
                        )
                        self.assertEqual(
                            sample["activity"].dtype, np.float32
                        )

                        np.testing.assert_allclose(
                            sample["activity"][:, 0], 100.0
                        )
                        np.testing.assert_allclose(
                            sample["activity"][:, 1], 1.0
                        )

                    total_samples += len(samples)

            self.assertEqual(total_samples, 4)


if __name__ == "__main__":
    unittest.main()

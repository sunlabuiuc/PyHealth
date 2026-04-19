"""Unit tests for the Cardiology2Dataset and CardiologyMultilabelClassification."""

# TestCardiology2Dataset covers the dataset
# TestCardiologyMultilabelClassification covers the task
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pyhealth.datasets import Cardiology2Dataset
from pyhealth.tasks import CardiologyMultilabelClassification


class TestCardiology2Dataset(unittest.TestCase):
    def _write_recording(
        self,
        patient_dir: Path,
        record_name: str,
        dx: str,
        sex: str = "Male",
        age: str = "63",
        signal_length: int = 2500,
    ) -> None:
        patient_dir.mkdir(parents=True, exist_ok=True)
        (patient_dir / f"{record_name}.mat").write_bytes(b"")
        (patient_dir / f"{record_name}.hea").write_text(
            "\n".join(
                [
                    f"{record_name} 12 500 {signal_length} 16 0 0 0 0",
                    f"# Age: {age}",
                    f"# Sex: {sex}",
                    f"# Dx: {dx}",
                ]
            )
            + "\n"
        )

    def test_invalid_chosen_dataset_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                Cardiology2Dataset(root=tmp, chosen_dataset=[1, 0, 1])

    def test_dataset_indexes_metadata_and_default_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_recording(
                root / "cpsc_2018" / "patient_a",
                "A0001",
                dx="164889003,427172004",
                sex="Female",
                age="54",
            )
            self._write_recording(
                root / "cpsc_2018" / "patient_a",
                "A0002",
                dx="426627000",
                sex="Female",
                age="54",
                signal_length=1500,
            )
            self._write_recording(
                root / "georgia" / "patient_b",
                "E0001",
                dx="713427006",
                sex="Male",
                age="61",
            )

            cache_dir = root / "cache"
            dataset = Cardiology2Dataset(
                root=str(root),
                chosen_dataset=[1, 0, 1, 0, 0, 0],
                cache_dir=str(cache_dir),
            )

            metadata_path = root / "cardiology-metadata-pyhealth.csv"
            self.assertTrue(metadata_path.exists())

            with metadata_path.open(newline="") as f:
                metadata = list(csv.DictReader(f))
            self.assertEqual(len(metadata), 3)
            self.assertCountEqual(
                [row["chosen_dataset"] for row in metadata],
                ["cpsc_2018", "cpsc_2018", "georgia"],
            )
            self.assertIn("signal_path", metadata[0])
            self.assertIn("dx", metadata[0])
            self.assertIn("sex", metadata[0])
            self.assertIn("age", metadata[0])

            self.assertEqual(len(dataset.unique_patient_ids), 2)
            patient = dataset.get_patient("0_0")
            events = patient.get_events(event_type="cardiology")

            self.assertEqual(len(events), 2)
            self.assertEqual(events[0]["patient_id"], "0_0")
            self.assertEqual(events[0]["sex"], "Female")
            self.assertEqual(events[0]["age"], "54")
            self.assertEqual(events[0]["chosen_dataset"], "cpsc_2018")
            self.assertTrue(str(events[0]["signal_path"]).endswith(".mat"))

            self.assertIsInstance(
                dataset.default_task, CardiologyMultilabelClassification
            )


class TestCardiologyMultilabelClassification(unittest.TestCase):
    def _write_recording(
        self,
        patient_dir: Path,
        record_name: str,
        dx: str,
        signal_length: int,
    ) -> None:
        patient_dir.mkdir(parents=True, exist_ok=True)
        (patient_dir / f"{record_name}.mat").write_bytes(b"")
        (patient_dir / f"{record_name}.hea").write_text(
            "\n".join(
                [
                    f"{record_name} 12 500 {signal_length} 16 0 0 0 0",
                    "# Age: 63",
                    "# Sex: Male",
                    f"# Dx: {dx}",
                ]
            )
            + "\n"
        )

    def test_task_generates_windowed_samples_and_filters_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patient_dir = root / "cpsc_2018" / "patient_a"
            self._write_recording(
                patient_dir,
                "A0001",
                dx="164889003,427172004,999999999",
                signal_length=2500,
            )
            self._write_recording(
                patient_dir,
                "A0002",
                dx="164889003",
                signal_length=1000,
            )

            dataset = Cardiology2Dataset(
                root=str(root),
                chosen_dataset=[1, 0, 0, 0, 0, 0],
                cache_dir=str(root / "cache"),
            )
            patient = dataset.get_patient("0_0")
            task = CardiologyMultilabelClassification(
                epoch_sec=2.5,
                shift=1.25,
                leads=[0, 2, 4],
            )

            fake_signal = np.arange(12 * 2500, dtype=np.float32).reshape(12, 2500)
            with patch(
                "pyhealth.tasks.cardiology_multilabel_classification.loadmat",
                side_effect=[{"val": fake_signal}, {"val": fake_signal[:, :1000]}],
            ):
                samples = task(patient)

            self.assertEqual(len(samples), 3)
            for sample in samples:
                self.assertEqual(sample["patient_id"], "0_0")
                self.assertEqual(sample["visit_id"], "A0001")
                self.assertEqual(sample["signal"].shape, (3, 1250))
                self.assertEqual(sample["labels"], ["164889003", "427172004"])

    def test_task_schema_attributes(self):
        task = CardiologyMultilabelClassification(leads=[0])
        self.assertEqual(task.task_name, "CardiologyMultilabelClassification")
        self.assertEqual(task.input_schema, {"signal": "tensor"})
        self.assertEqual(task.output_schema, {"labels": "multilabel"})
        self.assertEqual(task.leads, [0])


if __name__ == "__main__":
    unittest.main()

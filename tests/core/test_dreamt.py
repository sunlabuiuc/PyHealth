import shutil
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from pyhealth.datasets import DREAMTDataset
from pyhealth.tasks.dreamt_sleep_classification import DREAMTSleepClassification


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


class TestDREAMTDatasetNewerVersions(unittest.TestCase):
    """Test DREAMT dataset containing 64Hz and 100Hz folders with local test data."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
        (self.root / "data_64Hz").mkdir()
        (self.root / "data_100Hz").mkdir()
        self.num_patients = 5
        patient_data = {
            'SID': [f"S{i:03d}" for i in range(1, self.num_patients + 1)],
            'AGE': np.random.uniform(25, 65, self.num_patients),
            'GENDER': np.random.choice(['M', 'F'], self.num_patients),
            'BMI': np.random.randint(20, 50, self.num_patients),
            'OAHI': np.random.randint(0, 50, self.num_patients),
            'AHI': np.random.randint(0, 50, self.num_patients),
            'Mean_SaO2': [f"{val}%" for val in np.random.randint(85, 99, self.num_patients)],
            'Arousal Index': np.random.randint(1, 100, self.num_patients),
            'MEDICAL_HISTORY': ['Medical History'] * self.num_patients,
            'Sleep_Disorders': ['Sleep Disorder'] * self.num_patients,
        }
        pd.DataFrame(patient_data).to_csv(self.root / "participant_info.csv", index=False)
        self._create_files()

    def _create_files(self):
        for i in range(1, self.num_patients + 1):
            sid = f"S{i:03d}"
            partial_data = {
                'TIMESTAMP': [np.random.uniform(0, 100)],
                'BVP': [np.random.uniform(1, 10)],
                'HR': [np.random.randint(15, 100)],
                'EDA': [np.random.uniform(0, 1)],
                'TEMP': [np.random.uniform(20, 30)],
                'ACC_X': [np.random.uniform(1, 50)],
                'ACC_Y': [np.random.uniform(1, 50)],
                'ACC_Z': [np.random.uniform(1, 50)],
                'IBI': [np.random.uniform(0.6, 1.2)],
                'Sleep_Stage': [np.random.choice(['W', 'N1', 'N2', 'N3', 'R'])],
            }
            pd.DataFrame(partial_data).to_csv(self.root / "data_64Hz" / f"{sid}_whole_df.csv")
            pd.DataFrame(partial_data).to_csv(self.root / "data_100Hz" / f"{sid}_PSG_df.csv")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_dataset_initialization(self):
        dataset = DREAMTDataset(root=str(self.root))
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.dataset_name, "dreamt_sleep")
        self.assertEqual(dataset.root, str(self.root))

    def test_metadata_file_created(self):
        dataset = DREAMTDataset(root=str(self.root))
        self.assertTrue((self.root / "dreamt-metadata.csv").exists())

    def test_patient_count(self):
        dataset = DREAMTDataset(root=str(self.root))
        self.assertEqual(len(dataset.unique_patient_ids), self.num_patients)

    def test_stats_method(self):
        DREAMTDataset(root=str(self.root)).stats()

    def test_get_patient(self):
        dataset = DREAMTDataset(root=str(self.root))
        patient = dataset.get_patient('S001')
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, 'S001')

    def test_get_patient_not_found(self):
        dataset = DREAMTDataset(root=str(self.root))
        with self.assertRaises(AssertionError):
            dataset.get_patient('S222')


# ---------------------------------------------------------------------------
# Task stubs
# ---------------------------------------------------------------------------


@dataclass
class _DummyEvent:
    file_64hz: Optional[str]


class _DummyPatient:
    def __init__(self, patient_id: str, events: List[_DummyEvent]) -> None:
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type: Optional[str] = None) -> List[_DummyEvent]:
        return self._events


def _make_dreamt_csv(
    path: Path,
    n_rows: int = 120,
    stage: str = "N2",
    ibi_every: int = 4,
    include_acc: bool = True,
) -> None:
    rng = np.random.default_rng(42)
    ibi = np.zeros(n_rows)
    beat_indices = np.arange(0, n_rows, ibi_every)
    ibi[beat_indices] = rng.uniform(0.7, 1.1, len(beat_indices))
    data: dict = {
        "TIMESTAMP": np.arange(n_rows) / 4.0,
        "BVP": rng.standard_normal(n_rows),
        "HR": rng.integers(50, 90, n_rows).astype(float),
        "EDA": rng.uniform(0.0, 1.0, n_rows),
        "TEMP": rng.uniform(33.0, 37.0, n_rows),
        "IBI": ibi,
        "Sleep_Stage": [stage] * n_rows,
    }
    if include_acc:
        data["ACC_X"] = rng.standard_normal(n_rows)
        data["ACC_Y"] = rng.standard_normal(n_rows)
        data["ACC_Z"] = rng.standard_normal(n_rows)
    pd.DataFrame(data).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Task init tests
# ---------------------------------------------------------------------------


class TestDREAMTSleepClassificationInit(unittest.TestCase):
    """Verify task attributes are set correctly at construction time."""

    def test_default_attributes(self):
        task = DREAMTSleepClassification()
        self.assertEqual(task.task_name, "DREAMTSleepClassification")
        self.assertEqual(task.epoch_seconds, 30)
        self.assertEqual(task.num_classes, 3)
        self.assertFalse(task.use_accelerometer)
        self.assertEqual(task.sample_rate, 64)

    def test_output_schema(self):
        self.assertEqual(DREAMTSleepClassification().output_schema, {"label": "multiclass"})

    def test_3class_input_schema(self):
        task = DREAMTSleepClassification(num_classes=3)
        self.assertEqual(task.input_schema, {"ibi_sequence": "tensor"})
        self.assertNotIn("accelerometer", task.input_schema)

    def test_accelerometer_input_schema(self):
        task = DREAMTSleepClassification(use_accelerometer=True)
        self.assertIn("ibi_sequence", task.input_schema)
        self.assertIn("accelerometer", task.input_schema)

    def test_4class_initialises(self):
        self.assertEqual(DREAMTSleepClassification(num_classes=4).num_classes, 4)

    def test_invalid_num_classes_raises(self):
        with self.assertRaises(ValueError):
            DREAMTSleepClassification(num_classes=5)

    def test_invalid_num_classes_1_raises(self):
        with self.assertRaises(ValueError):
            DREAMTSleepClassification(num_classes=1)


# ---------------------------------------------------------------------------
# Task __call__ tests
# ---------------------------------------------------------------------------


class TestDREAMTSleepClassificationCall(unittest.TestCase):
    """End-to-end tests of the task's __call__ method."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.root = Path(self.tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def _patient(self, n_rows=120, stage="N2", file_exists=True, include_acc=True):
        csv_path = self.root / "S001_whole_df.csv"
        if file_exists:
            _make_dreamt_csv(csv_path, n_rows=n_rows, stage=stage, include_acc=include_acc)
        return _DummyPatient(
            "S001",
            [_DummyEvent(file_64hz=str(csv_path) if file_exists else None)],
        )

    def _task(self, **kwargs):
        return DREAMTSleepClassification(sample_rate=4, **kwargs)

    def test_returns_one_epoch_for_exactly_one_window(self):
        self.assertEqual(len(self._task()(self._patient(n_rows=120))), 1)

    def test_returns_two_epochs_for_two_windows(self):
        self.assertEqual(len(self._task()(self._patient(n_rows=240))), 2)

    def test_insufficient_rows_returns_empty(self):
        self.assertEqual(self._task()(self._patient(n_rows=50)), [])

    def test_sample_has_required_keys(self):
        s = self._task()(self._patient(n_rows=120))[0]
        for key in ("patient_id", "epoch_idx", "ibi_sequence", "label"):
            self.assertIn(key, s)

    def test_patient_id_propagated(self):
        self.assertEqual(self._task()(self._patient(n_rows=120))[0]["patient_id"], "S001")

    def test_epoch_idx_zero_for_first_epoch(self):
        self.assertEqual(self._task()(self._patient(n_rows=120))[0]["epoch_idx"], 0)

    def test_epoch_indices_sequential(self):
        samples = self._task()(self._patient(n_rows=360))
        self.assertEqual([s["epoch_idx"] for s in samples], [0, 1, 2])

    def test_ibi_values_are_float32(self):
        self.assertEqual(self._task()(self._patient(n_rows=120))[0]["ibi_sequence"].dtype, np.float32)

    def test_ibi_values_positive(self):
        ibi = self._task()(self._patient(n_rows=120))[0]["ibi_sequence"]
        self.assertTrue(np.all(ibi > 0))

    def test_ibi_length_matches_beat_count(self):
        _make_dreamt_csv(self.root / "S002_whole_df.csv", n_rows=120, ibi_every=4)
        patient = _DummyPatient("S002", [_DummyEvent(str(self.root / "S002_whole_df.csv"))])
        self.assertEqual(len(self._task()(patient)[0]["ibi_sequence"]), 30)

    def test_3class_wake_label(self):
        self.assertEqual(self._task(num_classes=3)(self._patient(stage="W"))[0]["label"], 0)

    def test_3class_n1_maps_to_nrem(self):
        self.assertEqual(self._task(num_classes=3)(self._patient(stage="N1"))[0]["label"], 1)

    def test_3class_n2_maps_to_nrem(self):
        self.assertEqual(self._task(num_classes=3)(self._patient(stage="N2"))[0]["label"], 1)

    def test_3class_n3_maps_to_nrem(self):
        self.assertEqual(self._task(num_classes=3)(self._patient(stage="N3"))[0]["label"], 1)

    def test_3class_rem_label(self):
        self.assertEqual(self._task(num_classes=3)(self._patient(stage="R"))[0]["label"], 2)

    def test_4class_wake_label(self):
        self.assertEqual(self._task(num_classes=4)(self._patient(stage="W"))[0]["label"], 0)

    def test_4class_n1_label(self):
        self.assertEqual(self._task(num_classes=4)(self._patient(stage="N1"))[0]["label"], 1)

    def test_4class_n2_label(self):
        self.assertEqual(self._task(num_classes=4)(self._patient(stage="N2"))[0]["label"], 2)

    def test_4class_n3_label(self):
        self.assertEqual(self._task(num_classes=4)(self._patient(stage="N3"))[0]["label"], 3)

    def test_4class_rem_label(self):
        self.assertEqual(self._task(num_classes=4)(self._patient(stage="R"))[0]["label"], 4)

    def test_none_file_returns_empty(self):
        self.assertEqual(self._task()(self._patient(file_exists=False)), [])

    def test_unknown_stage_skipped(self):
        self.assertEqual(self._task()(self._patient(stage="X")), [])

    def test_no_ibi_values_skipped(self):
        csv_path = self.root / "S003_whole_df.csv"
        rng = np.random.default_rng(0)
        n = 120
        pd.DataFrame({
            "TIMESTAMP": np.arange(n) / 4.0,
            "BVP": rng.standard_normal(n),
            "HR": rng.integers(50, 90, n).astype(float),
            "EDA": rng.uniform(0.0, 1.0, n),
            "TEMP": rng.uniform(33.0, 37.0, n),
            "IBI": np.zeros(n),
            "Sleep_Stage": ["N2"] * n,
        }).to_csv(csv_path, index=False)
        patient = _DummyPatient("S003", [_DummyEvent(str(csv_path))])
        self.assertEqual(self._task()(patient), [])

    def test_multiple_events_aggregated(self):
        csv1, csv2 = self.root / "night1.csv", self.root / "night2.csv"
        _make_dreamt_csv(csv1, stage="N2")
        _make_dreamt_csv(csv2, stage="R")
        patient = _DummyPatient("S004", [_DummyEvent(str(csv1)), _DummyEvent(str(csv2))])
        samples = self._task()(patient)
        self.assertEqual(len(samples), 2)
        labels = {s["label"] for s in samples}
        self.assertIn(1, labels)
        self.assertIn(2, labels)

    def test_accelerometer_key_absent_by_default(self):
        self.assertNotIn("accelerometer", self._task()(self._patient(n_rows=120))[0])

    def test_accelerometer_present_when_enabled(self):
        samples = self._task(use_accelerometer=True)(self._patient(include_acc=True))
        self.assertIn("accelerometer", samples[0])

    def test_accelerometer_shape(self):
        samples = self._task(use_accelerometer=True)(self._patient(n_rows=120, include_acc=True))
        self.assertEqual(samples[0]["accelerometer"].shape, (120, 3))

    def test_accelerometer_dtype_float32(self):
        samples = self._task(use_accelerometer=True)(self._patient(n_rows=120, include_acc=True))
        self.assertEqual(samples[0]["accelerometer"].dtype, np.float32)

    def test_accelerometer_missing_columns_skips_epoch(self):
        self.assertEqual(self._task(use_accelerometer=True)(self._patient(include_acc=False)), [])


if __name__ == "__main__":
    unittest.main()

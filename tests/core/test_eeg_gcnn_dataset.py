"""Tests for EEGGCNNDataset and EEGGCNNClassification.

Uses synthetic/pseudo data only — no real EEG recordings required.
All tests complete in milliseconds.
"""

import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from joblib import dump

from pyhealth.datasets.eeg_gcnn import EEGGCNNDataset
from pyhealth.tasks.eeg_gcnn_classification import (
    BAND_NAMES,
    EEGGCNNClassification,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

# Minimal 10-10 coordinate TSV — only the 8 reference electrodes needed.
_COORDS_TSV = (
    "label\tx\ty\tz\n"
    "F5\t-0.55\t0.67\t0.50\n"
    "F6\t 0.55\t0.67\t0.50\n"
    "C5\t-0.71\t0.00\t0.71\n"
    "C6\t 0.71\t0.00\t0.71\n"
    "P5\t-0.55\t-0.67\t0.50\n"
    "P6\t 0.55\t-0.67\t0.50\n"
    "O1\t-0.30\t-0.95\t0.10\n"
    "O2\t 0.30\t-0.95\t0.10\n"
)

_N_WINDOWS = 4   # 4 windows across 2 patients
_N_PATIENTS = 2


def _write_synthetic_root(root: str) -> None:
    """Write all five required raw files to *root* using synthetic data."""
    rng = np.random.default_rng(0)

    # psd_features_data_X  — shape (N, 48)
    X = rng.random((_N_WINDOWS, 48)).astype(np.float32)
    dump(X, os.path.join(root, "psd_features_data_X"))

    # labels_y  — alternating 0/1 string labels
    y = np.array(["diseased", "healthy", "diseased", "healthy"])
    dump(y, os.path.join(root, "labels_y"))

    # master_metadata_index.csv  — 2 patients, 2 windows each
    pd.DataFrame({
        "patient_ID": ["p001", "p001", "p002", "p002"],
    }).to_csv(os.path.join(root, "master_metadata_index.csv"), index=False)

    # spec_coh_values.npy  — shape (N, 64)
    coh = rng.random((_N_WINDOWS, 64)).astype(np.float32)
    np.save(os.path.join(root, "spec_coh_values.npy"), coh)

    # standard_1010.tsv.txt
    Path(os.path.join(root, "standard_1010.tsv.txt")).write_text(_COORDS_TSV)


@dataclass
class _DummyEvent:
    node_features_path: str
    adj_matrix_path: str
    window_idx: int
    label: int


class _DummyPatient:
    def __init__(self, patient_id: str, events: List[_DummyEvent]) -> None:
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type=None) -> List[_DummyEvent]:
        if event_type == "eeg_windows":
            return self._events
        return []


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestEEGGCNNDatasetPrepareMetadata(unittest.TestCase):

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = self._tmpdir.name
        _write_synthetic_root(self.root)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_prepare_metadata_creates_csv(self) -> None:
        EEGGCNNDataset.prepare_metadata(self.root, alpha=0.5)
        csv_path = os.path.join(self.root, "eeg_gcnn_windows_alpha0.50.csv")
        self.assertTrue(os.path.exists(csv_path))

    def test_prepare_metadata_csv_row_count(self) -> None:
        EEGGCNNDataset.prepare_metadata(self.root, alpha=0.5)
        df = pd.read_csv(
            os.path.join(self.root, "eeg_gcnn_windows_alpha0.50.csv")
        )
        self.assertEqual(len(df), _N_WINDOWS)

    def test_prepare_metadata_csv_columns(self) -> None:
        EEGGCNNDataset.prepare_metadata(self.root, alpha=0.5)
        df = pd.read_csv(
            os.path.join(self.root, "eeg_gcnn_windows_alpha0.50.csv")
        )
        for col in ("patient_id", "window_idx", "label",
                    "node_features_path", "adj_matrix_path"):
            self.assertIn(col, df.columns)

    def test_prepare_metadata_npy_shapes(self) -> None:
        EEGGCNNDataset.prepare_metadata(self.root, alpha=0.5)
        df = pd.read_csv(
            os.path.join(self.root, "eeg_gcnn_windows_alpha0.50.csv")
        )
        for _, row in df.iterrows():
            nf = np.load(row["node_features_path"])
            am = np.load(row["adj_matrix_path"])
            self.assertEqual(nf.shape, (8, 6))
            self.assertEqual(am.shape, (8, 8))

    def test_prepare_metadata_idempotent(self) -> None:
        EEGGCNNDataset.prepare_metadata(self.root, alpha=0.5)
        # Second call must not raise
        EEGGCNNDataset.prepare_metadata(self.root, alpha=0.5)

    def test_prepare_metadata_different_alpha_creates_separate_csv(self) -> None:
        EEGGCNNDataset.prepare_metadata(self.root, alpha=0.0)
        EEGGCNNDataset.prepare_metadata(self.root, alpha=1.0)
        self.assertTrue(
            os.path.exists(
                os.path.join(self.root, "eeg_gcnn_windows_alpha0.00.csv")
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(self.root, "eeg_gcnn_windows_alpha1.00.csv")
            )
        )

    def test_missing_required_file_raises(self) -> None:
        os.remove(os.path.join(self.root, "labels_y"))
        with self.assertRaises(FileNotFoundError):
            EEGGCNNDataset.prepare_metadata(self.root, alpha=0.5)

    def test_default_task_returns_classification_instance(self) -> None:
        ds = EEGGCNNDataset.__new__(EEGGCNNDataset)
        self.assertIsInstance(ds.default_task, EEGGCNNClassification)


# ---------------------------------------------------------------------------
# Task tests
# ---------------------------------------------------------------------------

class TestEEGGCNNClassification(unittest.TestCase):

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp = self._tmpdir.name

        rng = np.random.default_rng(1)

        # Write two windows to disk as .npy files
        self._events: List[_DummyEvent] = []
        for i in range(2):
            nf = rng.random((8, 6)).astype(np.float32)
            am = rng.random((8, 8)).astype(np.float32)
            nf_path = os.path.join(tmp, f"{i}_nf.npy")
            am_path = os.path.join(tmp, f"{i}_am.npy")
            np.save(nf_path, nf)
            np.save(am_path, am)
            self._events.append(
                _DummyEvent(
                    node_features_path=nf_path,
                    adj_matrix_path=am_path,
                    window_idx=i,
                    label=i % 2,
                )
            )

        self._patient = _DummyPatient("p001", self._events)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_schema_attributes(self) -> None:
        task = EEGGCNNClassification()
        self.assertEqual(task.task_name, "EEGGCNNClassification")
        self.assertEqual(task.input_schema["node_features"], "tensor")
        self.assertEqual(task.input_schema["adj_matrix"], "tensor")
        self.assertEqual(task.output_schema["label"], "binary")

    def test_call_returns_one_sample_per_window(self) -> None:
        task = EEGGCNNClassification()
        samples = task(self._patient)
        self.assertEqual(len(samples), 2)

    def test_call_sample_keys(self) -> None:
        task = EEGGCNNClassification()
        sample = task(self._patient)[0]
        for key in ("patient_id", "window_idx", "node_features",
                    "adj_matrix", "label"):
            self.assertIn(key, sample)

    def test_call_sample_shapes(self) -> None:
        task = EEGGCNNClassification()
        for sample in task(self._patient):
            self.assertEqual(sample["node_features"].shape, (8, 6))
            self.assertEqual(sample["adj_matrix"].shape, (8, 8))

    def test_call_patient_id_propagated(self) -> None:
        task = EEGGCNNClassification()
        for sample in task(self._patient):
            self.assertEqual(sample["patient_id"], "p001")

    def test_excluded_band_zeros_correct_column(self) -> None:
        band = "delta"          # index 0 in BAND_NAMES
        task = EEGGCNNClassification(excluded_bands=[band])
        for sample in task(self._patient):
            np.testing.assert_array_equal(
                sample["node_features"][:, 0],
                np.zeros(8, dtype=np.float32),
            )
            # Other columns should not be zeroed
            self.assertFalse(np.all(sample["node_features"][:, 1] == 0))

    def test_multiple_excluded_bands(self) -> None:
        task = EEGGCNNClassification(excluded_bands=["delta", "theta"])
        for sample in task(self._patient):
            np.testing.assert_array_equal(
                sample["node_features"][:, 0], np.zeros(8, dtype=np.float32)
            )
            np.testing.assert_array_equal(
                sample["node_features"][:, 1], np.zeros(8, dtype=np.float32)
            )

    def test_invalid_band_raises(self) -> None:
        with self.assertRaises(ValueError):
            EEGGCNNClassification(excluded_bands=["not_a_band"])

    def test_empty_patient_returns_empty_list(self) -> None:
        task = EEGGCNNClassification()
        empty_patient = _DummyPatient("p_empty", [])
        self.assertEqual(task(empty_patient), [])

    def test_all_band_names_are_valid(self) -> None:
        # Should not raise for any individual band
        for band in BAND_NAMES:
            EEGGCNNClassification(excluded_bands=[band])


if __name__ == "__main__":
    unittest.main()

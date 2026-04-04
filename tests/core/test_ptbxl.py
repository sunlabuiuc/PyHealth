"""
Unit tests for PTBXLDataset, PTBXLDiagnosis, and PTBXLMulticlassDiagnosis.

Tests use only synthetic/pseudo data — no real PTB-XL files are required.
All tests complete in milliseconds.

Author:
    Ankita Jain (ankitaj3@illinois.edu), Manish Singh (manishs4@illinois.edu)
"""

import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pandas as pd

from pyhealth.datasets.ptbxl import PTBXLDataset
from pyhealth.tasks.ptbxl_diagnosis import (
    PTBXLDiagnosis,
    PTBXLMulticlassDiagnosis,
    _scp_to_superclasses,
)


# ---------------------------------------------------------------------------
# Minimal stubs so we can test task logic without a real BaseDataset
# ---------------------------------------------------------------------------

@dataclass
class _FakeEvent:
    """Minimal event stub."""
    record_id: int
    signal_file: str
    scp_codes: str
    sampling_rate: int = 100
    num_leads: int = 12

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class _FakePatient:
    """Minimal patient stub."""

    def __init__(self, patient_id: str, events: List[_FakeEvent]) -> None:
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type: str = "ptbxl") -> List[_FakeEvent]:
        return self._events


# ---------------------------------------------------------------------------
# Helper: build a minimal ptbxl_database.csv in a temp directory
# ---------------------------------------------------------------------------

def _make_fake_db(tmp: Path, sampling_rate: int = 100) -> None:
    """Write a minimal ptbxl_database.csv with 3 synthetic patients."""
    rows = [
        {
            "ecg_id": 1,
            "patient_id": 101,
            "filename_lr": "records100/00000/00001_lr",
            "filename_hr": "records500/00000/00001_hr",
            "scp_codes": "{'NORM': 100.0}",
        },
        {
            "ecg_id": 2,
            "patient_id": 102,
            "filename_lr": "records100/00000/00002_lr",
            "filename_hr": "records500/00000/00002_hr",
            "scp_codes": "{'IMI': 80.0, 'CLBBB': 20.0}",
        },
        {
            "ecg_id": 3,
            "patient_id": 103,
            "filename_lr": "records100/00000/00003_lr",
            "filename_hr": "records500/00000/00003_hr",
            "scp_codes": "{'UNKNOWN_CODE': 50.0}",
        },
    ]
    pd.DataFrame(rows).to_csv(tmp / "ptbxl_database.csv", index=False)


# ---------------------------------------------------------------------------
# Tests: _scp_to_superclasses helper
# ---------------------------------------------------------------------------

class TestSCPToSuperclasses(unittest.TestCase):
    def test_norm(self):
        self.assertEqual(_scp_to_superclasses("{'NORM': 100.0}"), ["NORM"])

    def test_mi_and_cd(self):
        result = _scp_to_superclasses("{'IMI': 80.0, 'CLBBB': 20.0}")
        self.assertIn("MI", result)
        self.assertIn("CD", result)

    def test_zero_likelihood_excluded(self):
        result = _scp_to_superclasses("{'NORM': 0.0, 'IMI': 50.0}")
        self.assertNotIn("NORM", result)
        self.assertIn("MI", result)

    def test_unknown_code_returns_empty(self):
        self.assertEqual(_scp_to_superclasses("{'UNKNOWN_CODE': 100.0}"), [])

    def test_malformed_string_returns_empty(self):
        self.assertEqual(_scp_to_superclasses("not_a_dict"), [])

    def test_empty_dict(self):
        self.assertEqual(_scp_to_superclasses("{}"), [])


# ---------------------------------------------------------------------------
# Tests: PTBXLDataset.prepare_metadata
# ---------------------------------------------------------------------------

class TestPTBXLDatasetPrepareMetadata(unittest.TestCase):
    def test_prepare_metadata_creates_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fake_db(root)

            ds = PTBXLDataset.__new__(PTBXLDataset)
            ds.sampling_rate = 100
            ds.root = str(root)
            ds.prepare_metadata(str(root))

            out_csv = root / "ptbxl-metadata-pyhealth.csv"
            self.assertTrue(out_csv.exists(), "Metadata CSV was not created.")

            df = pd.read_csv(out_csv)
            self.assertEqual(len(df), 3)
            self.assertIn("patient_id", df.columns)
            self.assertIn("record_id", df.columns)
            self.assertIn("signal_file", df.columns)
            self.assertIn("scp_codes", df.columns)

    def test_prepare_metadata_500hz(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fake_db(root)

            ds = PTBXLDataset.__new__(PTBXLDataset)
            ds.sampling_rate = 500
            ds.root = str(root)
            ds.prepare_metadata(str(root))

            df = pd.read_csv(root / "ptbxl-metadata-pyhealth.csv")
            # 500 Hz paths contain "records500"
            self.assertTrue(df["signal_file"].str.contains("records500").all())

    def test_prepare_metadata_missing_db_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = PTBXLDataset.__new__(PTBXLDataset)
            ds.sampling_rate = 100
            ds.root = tmp
            with self.assertRaises(FileNotFoundError):
                ds.prepare_metadata(tmp)

    def test_patient_ids_are_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fake_db(root)

            ds = PTBXLDataset.__new__(PTBXLDataset)
            ds.sampling_rate = 100
            ds.root = str(root)
            ds.prepare_metadata(str(root))

            df = pd.read_csv(root / "ptbxl-metadata-pyhealth.csv")
            # patient_id column must exist and have no nulls
            self.assertIn("patient_id", df.columns)
            self.assertFalse(df["patient_id"].isnull().any())


# ---------------------------------------------------------------------------
# Tests: PTBXLDiagnosis task
# ---------------------------------------------------------------------------

class TestPTBXLDiagnosis(unittest.TestCase):
    def _make_patient(self, scp_codes_list: List[str]) -> _FakePatient:
        events = [
            _FakeEvent(
                record_id=i + 1,
                signal_file=f"records100/00000/0000{i + 1}_lr",
                scp_codes=scp,
            )
            for i, scp in enumerate(scp_codes_list)
        ]
        return _FakePatient(patient_id="p001", events=events)

    def test_schema(self):
        task = PTBXLDiagnosis()
        self.assertEqual(task.task_name, "PTBXLDiagnosis")
        self.assertIn("signal_file", task.input_schema)
        self.assertIn("labels", task.output_schema)
        self.assertEqual(task.output_schema["labels"], "multilabel")

    def test_normal_ecg(self):
        task = PTBXLDiagnosis()
        patient = self._make_patient(["{'NORM': 100.0}"])
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["labels"], ["NORM"])

    def test_multilabel_ecg(self):
        task = PTBXLDiagnosis()
        patient = self._make_patient(["{'IMI': 80.0, 'CLBBB': 20.0}"])
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        self.assertIn("MI", samples[0]["labels"])
        self.assertIn("CD", samples[0]["labels"])

    def test_unknown_code_skipped(self):
        task = PTBXLDiagnosis()
        patient = self._make_patient(["{'UNKNOWN': 100.0}"])
        samples = task(patient)
        self.assertEqual(len(samples), 0)

    def test_multiple_events(self):
        task = PTBXLDiagnosis()
        patient = self._make_patient(["{'NORM': 100.0}", "{'IMI': 90.0}"])
        samples = task(patient)
        self.assertEqual(len(samples), 2)

    def test_sample_keys(self):
        task = PTBXLDiagnosis()
        patient = self._make_patient(["{'NORM': 100.0}"])
        sample = task(patient)[0]
        for key in ("patient_id", "record_id", "signal_file", "labels"):
            self.assertIn(key, sample)

    def test_empty_patient(self):
        task = PTBXLDiagnosis()
        patient = _FakePatient(patient_id="empty", events=[])
        self.assertEqual(task(patient), [])


# ---------------------------------------------------------------------------
# Tests: PTBXLMulticlassDiagnosis task
# ---------------------------------------------------------------------------

class TestPTBXLMulticlassDiagnosis(unittest.TestCase):
    def _make_patient(self, scp_codes_list: List[str]) -> _FakePatient:
        events = [
            _FakeEvent(
                record_id=i + 1,
                signal_file=f"records100/00000/0000{i + 1}_lr",
                scp_codes=scp,
            )
            for i, scp in enumerate(scp_codes_list)
        ]
        return _FakePatient(patient_id="p002", events=events)

    def test_schema(self):
        task = PTBXLMulticlassDiagnosis()
        self.assertEqual(task.task_name, "PTBXLMulticlassDiagnosis")
        self.assertEqual(task.output_schema["label"], "multiclass")

    def test_dominant_class_selected(self):
        task = PTBXLMulticlassDiagnosis()
        # IMI (MI) has higher likelihood than CLBBB (CD)
        patient = self._make_patient(["{'IMI': 80.0, 'CLBBB': 20.0}"])
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["label"], "MI")

    def test_norm_ecg(self):
        task = PTBXLMulticlassDiagnosis()
        patient = self._make_patient(["{'NORM': 100.0}"])
        samples = task(patient)
        self.assertEqual(samples[0]["label"], "NORM")

    def test_unknown_code_skipped(self):
        task = PTBXLMulticlassDiagnosis()
        patient = self._make_patient(["{'UNKNOWN': 100.0}"])
        self.assertEqual(task(patient), [])

    def test_sample_keys(self):
        task = PTBXLMulticlassDiagnosis()
        patient = self._make_patient(["{'NORM': 100.0}"])
        sample = task(patient)[0]
        for key in ("patient_id", "record_id", "signal_file", "label"):
            self.assertIn(key, sample)

    def test_empty_patient(self):
        task = PTBXLMulticlassDiagnosis()
        patient = _FakePatient(patient_id="empty", events=[])
        self.assertEqual(task(patient), [])


if __name__ == "__main__":
    unittest.main()

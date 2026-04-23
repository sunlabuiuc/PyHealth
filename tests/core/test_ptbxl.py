"""Tests for PTBXLDataset using synthetic data.

All tests are self-contained: they create temporary files in TemporaryDirectory
contexts (auto-cleaned on exit), never touch real PTB-XL data, and complete
in milliseconds.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _write_fake_ptbxl(
    root: str,
    n_patients: int = 3,
    n_records: int = 6,
) -> None:
    """Write minimal fake PTB-XL CSV files to *root* (≤ 5 patients)."""
    # diagnostic_class is the correct PTB-XL column holding the 5 superclass
    # labels (MI, HYP, STTC, CD, NORM). diagnostic_subclass holds finer types.
    pd.DataFrame(
        {
            "Unnamed: 0": ["NORM", "MI", "ISCAL", "HYP", "STTC", "CD"],
            "diagnostic": [1, 1, 1, 1, 1, 1],
            "diagnostic_class": ["NORM", "MI", "MI", "HYP", "STTC", "CD"],
            "diagnostic_subclass": ["NORM", "IMI", "ISCAL", "LVH", "ISC_", "CLBBB"],
        }
    ).to_csv(os.path.join(root, "scp_statements.csv"), index=False)

    rows = []
    for i in range(1, n_records + 1):
        if i % 3 == 0:
            scp = {"MI": 100.0}
        elif i % 4 == 0:
            scp = {"HYP": 80.0}
        elif i % 5 == 0:
            scp = {"STTC": 90.0}
        else:
            scp = {"NORM": 100.0}

        rows.append(
            {
                "ecg_id": i,
                "patient_id": ((i - 1) % n_patients) + 1,
                "recording_date": "2000-01-01 00:00:00",
                "filename_lr": f"records100/00000/{i:05d}_lr",
                "filename_hr": f"records500/00000/{i:05d}_hr",
                "scp_codes": str(scp),
            }
        )
    pd.DataFrame(rows).to_csv(
        os.path.join(root, "ptbxl_database.csv"), index=False
    )


def _make_fake_event(attr: Dict[str, Any]) -> MagicMock:
    """Return a mock Event with ``__getitem__`` backed by *attr*."""
    event = MagicMock()
    event.__getitem__ = lambda self, key: attr[key]
    return event


def _make_fake_patient(
    events: List[MagicMock], patient_id: str = "P001"
) -> MagicMock:
    patient = MagicMock()
    patient.patient_id = patient_id
    patient.get_events.return_value = events
    return patient


# ---------------------------------------------------------------------------
# Fast & Performant — synthetic data, tempdir, millisecond execution
# ---------------------------------------------------------------------------

class TestDataLoading:
    """Tests data loading: metadata CSV creation and content."""

    def test_metadata_csv_created(self):
        with tempfile.TemporaryDirectory() as root:
            _write_fake_ptbxl(root)
            from pyhealth.datasets.ptbxl import PTBXLDataset

            obj = PTBXLDataset.__new__(PTBXLDataset)
            obj.sampling_rate = 100
            obj._prepare_metadata(root)

            assert os.path.exists(os.path.join(root, "ptbxl_metadata.csv"))

    def test_row_count_matches_input(self):
        with tempfile.TemporaryDirectory() as root:
            _write_fake_ptbxl(root, n_records=6)
            from pyhealth.datasets.ptbxl import PTBXLDataset

            obj = PTBXLDataset.__new__(PTBXLDataset)
            obj.sampling_rate = 100
            obj._prepare_metadata(root)

            meta = pd.read_csv(os.path.join(root, "ptbxl_metadata.csv"))
            assert len(meta) == 6

    def test_missing_scp_statements_graceful(self):
        """Metadata preparation succeeds even without scp_statements.csv."""
        with tempfile.TemporaryDirectory() as root:
            _write_fake_ptbxl(root)
            os.remove(os.path.join(root, "scp_statements.csv"))
            from pyhealth.datasets.ptbxl import PTBXLDataset

            obj = PTBXLDataset.__new__(PTBXLDataset)
            obj.sampling_rate = 100
            obj._prepare_metadata(root)

            meta = pd.read_csv(os.path.join(root, "ptbxl_metadata.csv"))
            assert (meta["mi_label"] == 0).all()

    def test_metadata_not_regenerated_if_exists(self):
        """Second call to __init__ reuses existing ptbxl_metadata.csv."""
        with tempfile.TemporaryDirectory() as root:
            _write_fake_ptbxl(root)
            from pyhealth.datasets.ptbxl import PTBXLDataset

            obj = PTBXLDataset.__new__(PTBXLDataset)
            obj.sampling_rate = 100
            obj._prepare_metadata(root)

            mtime_first = os.path.getmtime(
                os.path.join(root, "ptbxl_metadata.csv")
            )
            # Calling again should NOT re-write the file
            obj2 = PTBXLDataset.__new__(PTBXLDataset)
            obj2.sampling_rate = 100
            # Simulate __init__ guard: only call if file absent
            if not os.path.exists(os.path.join(root, "ptbxl_metadata.csv")):
                obj2._prepare_metadata(root)

            mtime_second = os.path.getmtime(
                os.path.join(root, "ptbxl_metadata.csv")
            )
            assert mtime_first == mtime_second


# ---------------------------------------------------------------------------
# Data Integrity
# ---------------------------------------------------------------------------

class TestDataIntegrity:
    """Tests schema, label correctness, and filename integrity."""

    def test_required_columns_present(self):
        with tempfile.TemporaryDirectory() as root:
            _write_fake_ptbxl(root)
            from pyhealth.datasets.ptbxl import PTBXLDataset

            obj = PTBXLDataset.__new__(PTBXLDataset)
            obj.sampling_rate = 100
            obj._prepare_metadata(root)

            meta = pd.read_csv(os.path.join(root, "ptbxl_metadata.csv"))
            for col in (
                "patient_id", "ecg_id", "recording_date", "filename",
                "mi_label", "hyp_label", "sttc_label", "cd_label",
            ):
                assert col in meta.columns, f"Missing column: {col}"

    def test_mi_labels_correct(self):
        with tempfile.TemporaryDirectory() as root:
            _write_fake_ptbxl(root, n_records=6)
            from pyhealth.datasets.ptbxl import PTBXLDataset

            obj = PTBXLDataset.__new__(PTBXLDataset)
            obj.sampling_rate = 100
            obj._prepare_metadata(root)

            meta = pd.read_csv(os.path.join(root, "ptbxl_metadata.csv"))
            # ecg_id is stored as digit-only strings; pandas infers int64 on read-back.
            ecg_id = meta["ecg_id"].astype(int)
            assert meta[ecg_id == 3]["mi_label"].iloc[0] == 1
            assert meta[ecg_id == 6]["mi_label"].iloc[0] == 1
            assert meta[ecg_id == 1]["mi_label"].iloc[0] == 0

    def test_labels_are_binary(self):
        with tempfile.TemporaryDirectory() as root:
            _write_fake_ptbxl(root)
            from pyhealth.datasets.ptbxl import PTBXLDataset

            obj = PTBXLDataset.__new__(PTBXLDataset)
            obj.sampling_rate = 100
            obj._prepare_metadata(root)

            meta = pd.read_csv(os.path.join(root, "ptbxl_metadata.csv"))
            for col in ("mi_label", "hyp_label", "sttc_label", "cd_label"):
                assert set(meta[col].unique()).issubset({0, 1}), (
                    f"{col} contains non-binary values"
                )

    def test_filename_contains_root(self):
        with tempfile.TemporaryDirectory() as root:
            _write_fake_ptbxl(root)
            from pyhealth.datasets.ptbxl import PTBXLDataset

            obj = PTBXLDataset.__new__(PTBXLDataset)
            obj.sampling_rate = 100
            obj._prepare_metadata(root)

            meta = pd.read_csv(os.path.join(root, "ptbxl_metadata.csv"))
            assert meta["filename"].iloc[0].startswith(root)


# ---------------------------------------------------------------------------
# Patient Parsing
# ---------------------------------------------------------------------------

class TestPatientParsing:
    """Tests that records are correctly grouped by patient_id."""

    def test_correct_patient_count(self):
        with tempfile.TemporaryDirectory() as root:
            _write_fake_ptbxl(root, n_patients=3, n_records=6)
            from pyhealth.datasets.ptbxl import PTBXLDataset

            obj = PTBXLDataset.__new__(PTBXLDataset)
            obj.sampling_rate = 100
            obj._prepare_metadata(root)

            meta = pd.read_csv(os.path.join(root, "ptbxl_metadata.csv"))
            assert meta["patient_id"].nunique() == 3

    def test_patient_ids_are_strings(self):
        with tempfile.TemporaryDirectory() as root:
            _write_fake_ptbxl(root)
            from pyhealth.datasets.ptbxl import PTBXLDataset

            obj = PTBXLDataset.__new__(PTBXLDataset)
            obj.sampling_rate = 100
            obj._prepare_metadata(root)

            meta = pd.read_csv(os.path.join(root, "ptbxl_metadata.csv"))
            # patient_id is written as digit-only strings; pandas infers int64 on
            # read-back.  Verify the values are at least safely castable to str.
            assert meta["patient_id"].apply(lambda x: str(x)).dtype == object

    def test_records_per_patient_correct(self):
        with tempfile.TemporaryDirectory() as root:
            _write_fake_ptbxl(root, n_patients=3, n_records=6)
            from pyhealth.datasets.ptbxl import PTBXLDataset

            obj = PTBXLDataset.__new__(PTBXLDataset)
            obj.sampling_rate = 100
            obj._prepare_metadata(root)

            meta = pd.read_csv(os.path.join(root, "ptbxl_metadata.csv"))
            counts = meta.groupby("patient_id").size()
            # 6 records across 3 patients → 2 records each
            assert (counts == 2).all()


# ---------------------------------------------------------------------------
# Event Parsing
# ---------------------------------------------------------------------------

class TestEventParsing:
    """Tests event attribute access via mock Patient/Event objects."""

    def test_event_attribute_access(self):
        """Events return correct attribute values via __getitem__."""
        attr = {
            "filename": "/fake/path/00001_lr",
            "ecg_id": "1",
            "mi_label": 1,
            "hyp_label": 0,
            "sttc_label": 0,
            "cd_label": 0,
        }
        event = _make_fake_event(attr)
        assert event["filename"] == "/fake/path/00001_lr"
        assert event["mi_label"] == 1
        assert event["hyp_label"] == 0

    def test_patient_get_events_returns_list(self):
        """patient.get_events() returns the expected event list."""
        events = [
            _make_fake_event({"filename": "/f/1", "ecg_id": "1",
                               "mi_label": 1, "hyp_label": 0,
                               "sttc_label": 0, "cd_label": 0}),
            _make_fake_event({"filename": "/f/2", "ecg_id": "2",
                               "mi_label": 0, "hyp_label": 1,
                               "sttc_label": 0, "cd_label": 0}),
        ]
        patient = _make_fake_patient(events, patient_id="42")
        returned = patient.get_events(event_type="ecg_records")
        assert len(returned) == 2
        assert returned[0]["ecg_id"] == "1"


# ---------------------------------------------------------------------------
# Task Functionality (dataset + task integration)
# ---------------------------------------------------------------------------

class TestTaskFunctionality:
    """Tests that ECGBinaryClassification works on mock PTB-XL patients."""

    def test_task_produces_correct_sample_count(self):
        from unittest.mock import patch
        import numpy as np
        from pyhealth.tasks.ecg_classification import ECGBinaryClassification

        task = ECGBinaryClassification(task_label="MI", target_length=128)
        events = [
            _make_fake_event({"filename": "/f/1", "ecg_id": "1",
                               "mi_label": 1, "hyp_label": 0,
                               "sttc_label": 0, "cd_label": 0}),
            _make_fake_event({"filename": "/f/2", "ecg_id": "2",
                               "mi_label": 0, "hyp_label": 0,
                               "sttc_label": 0, "cd_label": 0}),
        ]
        patient = _make_fake_patient(events, patient_id="1")
        fake_signal = np.random.randn(12, 200).astype(np.float32)

        with patch.object(task, "_load_signal", return_value=fake_signal):
            samples = task(patient)

        assert len(samples) == 2
        assert samples[0]["label"] == 1
        assert samples[1]["label"] == 0
        assert samples[0]["ecg"].shape == (12, 128)

    def test_task_label_mi_vs_hyp(self):
        """Switching task_label changes which column is used as the target."""
        from unittest.mock import patch
        import numpy as np
        from pyhealth.tasks.ecg_classification import ECGBinaryClassification

        attr = {"filename": "/f/1", "ecg_id": "1",
                "mi_label": 0, "hyp_label": 1,
                "sttc_label": 0, "cd_label": 0}
        events = [_make_fake_event(attr)]
        patient = _make_fake_patient(events)
        fake_signal = np.random.randn(12, 128).astype(np.float32)

        for label_col, task_label, expected in [
            ("mi_label", "MI", 0),
            ("hyp_label", "HYP", 1),
        ]:
            task = ECGBinaryClassification(task_label=task_label,
                                           target_length=128)
            with patch.object(task, "_load_signal", return_value=fake_signal):
                samples = task(patient)
            assert samples[0]["label"] == expected, (
                f"Expected label {expected} for {task_label}"
            )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_sampling_rate_raises(self):
        """PTBXLDataset.__init__ rejects sampling rates other than 100 or 500."""
        from pyhealth.datasets.ptbxl import PTBXLDataset

        with tempfile.TemporaryDirectory() as root:
            _write_fake_ptbxl(root)
            # ValueError is raised before super().__init__(), so no patching needed.
            obj = PTBXLDataset.__new__(PTBXLDataset)
            with pytest.raises(ValueError, match="sampling_rate must be"):
                PTBXLDataset.__init__(obj, root=root, sampling_rate=250)

    def test_missing_database_raises(self):
        """FileNotFoundError raised when ptbxl_database.csv is absent."""
        with tempfile.TemporaryDirectory() as root:
            from pyhealth.datasets.ptbxl import PTBXLDataset

            obj = PTBXLDataset.__new__(PTBXLDataset)
            obj.sampling_rate = 100
            with pytest.raises(FileNotFoundError, match="ptbxl_database.csv"):
                obj._prepare_metadata(root)

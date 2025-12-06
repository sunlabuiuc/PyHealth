import os
import tempfile
import pandas as pd
import pytest

from pyhealth.datasets import MIMIC4ECGDataset


@pytest.fixture
def mock_dataset_dir():
    """Create a temporary dataset directory with mock ECG CSVs in an `ecg/` subfolder."""
    tmpdir = tempfile.TemporaryDirectory()
    root = tmpdir.name
    ecg_dir = os.path.join(root, "ecg")
    os.makedirs(ecg_dir, exist_ok=True)

    # machine_measurements.csv
    pd.DataFrame([
        {
            "subject_id": 1,
            "study_id": 101,
            "cart_id": "C1",
            "ecg_time": "2025-01-01 10:00:00",
            "report_0": "Normal ECG",
            "bandwidth": "0.5-40Hz",
            "filtering": "None",
            "rr_interval": 800,
            "p_onset": 100,
            "p_end": 120,
            "qrs_onset": 130,
            "qrs_end": 160,
            "t_end": 300,
            "p_axis": 45,
            "qrs_axis": 60,
            "t_axis": 30,
        }
    ]).to_csv(os.path.join(ecg_dir, "machine_measurements.csv"), index=False)

    # record_list.csv
    pd.DataFrame([
        {
            "subject_id": 1,
            "study_id": 101,
            "file_name": "ecg_101.dat",
            "ecg_time": "2025-01-01 10:00:00",
            "path": "/waveforms/ecg_101.dat",
        }
    ]).to_csv(os.path.join(ecg_dir, "record_list.csv"), index=False)

    # waveform_note_links.csv
    pd.DataFrame([
        {
            "subject_id": 1,
            "study_id": 101,
            "waveform_path": "/waveforms/ecg_101.dat",
            "note_id": 5001,
            "note_seq": 1,
            "charttime": "2025-01-01 11:00:00",
        }
    ]).to_csv(os.path.join(ecg_dir, "waveform_note_links.csv"), index=False)

    yield root
    tmpdir.cleanup()


def test_dataset_initialization(mock_dataset_dir):
    ds = MIMIC4ECGDataset(root=mock_dataset_dir)
    assert "machine_measurements" in ds.tables
    assert "record_list" in ds.tables
    assert "waveform_note_links" in ds.tables


def test_dataset_len_and_getitem(mock_dataset_dir):
    ds = MIMIC4ECGDataset(root=mock_dataset_dir)
    assert len(ds) == 1
    sample = ds[0]
    assert sample["subject_id"] == 1
    assert sample["study_id"] == 101


def test_missing_table_raises_error(mock_dataset_dir):
    os.remove(os.path.join(mock_dataset_dir, "ecg", "record_list.csv"))
    with pytest.raises(FileNotFoundError):
        MIMIC4ECGDataset(root=mock_dataset_dir)
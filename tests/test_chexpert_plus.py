"""Tests for CheXpertPlusDataset.

Run with:
    pytest tests/test_chexpert_plus.py -v
"""

import os
import tempfile

import pandas as pd
import pytest

from pyhealth.datasets import CheXpertPlusDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chexpert_plus_root(tmp_path):
    """Create a minimal CheXpert Plus CSV in a temporary directory."""
    data = {
        "path_to_image": [
            "train/patient00001/study1/view1_frontal.jpg",
            "train/patient00002/study1/view1_frontal.jpg",
            "train/patient00003/study1/view1_frontal.jpg",
        ],
        "section_findings": [
            "No acute cardiopulmonary process.",
            "Mild cardiomegaly is noted. Lungs are clear.",
            "Small left pleural effusion. No pneumothorax.",
        ],
        "section_impression": [
            "Normal chest radiograph.",
            "Cardiomegaly.",
            "Left pleural effusion.",
        ],
        "section_history": ["Cough.", "Shortness of breath.", "Chest pain."],
        "section_comparison": ["None.", "Prior study 2021.", "None."],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "df_chexpert_plus_240401.csv"
    df.to_csv(csv_path, index=False)
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_init_raises_without_csv(tmp_path):
    """Dataset should raise FileNotFoundError when CSV is absent."""
    with pytest.raises(FileNotFoundError):
        CheXpertPlusDataset(root=str(tmp_path))


def test_init_succeeds_with_csv(chexpert_plus_root):
    """Dataset should initialise without errors when CSV is present."""
    dataset = CheXpertPlusDataset(root=chexpert_plus_root, dev=True)
    assert dataset.dataset_name == "CheXpertPlus"


def test_default_task_type(chexpert_plus_root):
    """default_task should return a RadiologyKGExtractionTask."""
    from pyhealth.tasks import RadiologyKGExtractionTask

    dataset = CheXpertPlusDataset(root=chexpert_plus_root, dev=True)
    task = dataset.default_task
    assert isinstance(task, RadiologyKGExtractionTask)


def test_patient_ids_populated(chexpert_plus_root):
    """There should be one patient per row (path_to_image as patient_id)."""
    dataset = CheXpertPlusDataset(root=chexpert_plus_root, dev=True)
    patient_ids = dataset.unique_patient_ids
    assert len(patient_ids) == 3


def test_stats_runs(chexpert_plus_root, capsys):
    """stats() should not raise and should print dataset information."""
    dataset = CheXpertPlusDataset(root=chexpert_plus_root, dev=True)
    dataset.stats()
    captured = capsys.readouterr()
    assert "CheXpertPlus" in captured.out

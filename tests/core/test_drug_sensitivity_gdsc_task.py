"""Unit tests for DrugSensitivityPredictionGDSC task.

Uses only in-memory synthetic data — no file I/O, no network access.
"""

import numpy as np
import pytest

from pyhealth.tasks.drug_sensitivity_gdsc import DrugSensitivityPredictionGDSC

N_GENES = 20
N_DRUGS = 10


@pytest.fixture
def task():
    return DrugSensitivityPredictionGDSC()


@pytest.fixture
def patient():
    rng = np.random.RandomState(42)
    drug_sens = rng.randint(0, 2, size=N_DRUGS).astype(float)
    drug_sens[rng.rand(N_DRUGS) < 0.2] = np.nan
    return {
        "patient_id": "COSMIC.42",
        "gene_expression": rng.randint(0, 2, size=N_GENES),
        "drug_sensitivity": drug_sens,
        "drug_pathway_ids": list(range(N_DRUGS)),
    }


def test_schema(task):
    assert task.task_name == "drug_sensitivity_prediction"
    assert "gene_indices" in task.input_schema
    assert "drug_pathway_ids" in task.input_schema
    assert "labels" in task.output_schema
    assert "mask" in task.output_schema


def test_call_output(task, patient):
    result = task(patient)
    assert len(result) == 1
    sample = result[0]
    assert set(sample.keys()) == {"patient_id", "visit_id", "gene_indices", "labels", "mask", "drug_pathway_ids"}
    assert sample["patient_id"] == patient["patient_id"]
    assert sample["visit_id"] == sample["patient_id"]
    assert 0 not in sample["gene_indices"]                          # 1-indexed
    assert len(sample["gene_indices"]) == int(patient["gene_expression"].sum())
    assert len(sample["labels"]) == N_DRUGS
    assert len(sample["mask"]) == N_DRUGS
    assert all(v in (0, 1) for v in sample["labels"])
    assert all(v in (0, 1) for v in sample["mask"])


def test_nan_handling(task):
    """NaN sensitivity → mask=0, label=0. Observed values → mask=1."""
    patient = {
        "patient_id": "COSMIC.0",
        "gene_expression": np.array([1, 0, 1, 0]),
        "drug_sensitivity": np.array([1.0, np.nan, 0.0]),
        "drug_pathway_ids": [0, 1, 2],
    }
    sample = task(patient)[0]
    assert sample["mask"] == [1, 0, 1]
    assert sample["labels"] == [1, 0, 0]

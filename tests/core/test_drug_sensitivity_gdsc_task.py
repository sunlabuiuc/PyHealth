"""Unit tests for DrugSensitivityPredictionGDSC task.

Uses only in-memory synthetic data — no file I/O, no network access.
"""

import numpy as np
import pytest

from pyhealth.tasks.drug_sensitivity_gdsc import DrugSensitivityPredictionGDSC

N_GENES = 20
N_DRUGS = 10


def _make_patient(seed: int = 0):
    rng = np.random.RandomState(seed)
    gene_expr = rng.randint(0, 2, size=N_GENES)
    drug_sens = rng.randint(0, 2, size=N_DRUGS).astype(float)
    drug_sens[rng.rand(N_DRUGS) < 0.2] = np.nan
    return {
        "patient_id": f"COSMIC.{seed}",
        "gene_expression": gene_expr,
        "drug_sensitivity": drug_sens,
        "drug_pathway_ids": list(range(N_DRUGS)),
    }


@pytest.fixture
def task():
    return DrugSensitivityPredictionGDSC()


@pytest.fixture
def patient():
    return _make_patient(seed=42)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_task_name(task):
    assert task.task_name == "drug_sensitivity_prediction"


def test_input_schema_exists(task):
    assert "gene_indices" in task.input_schema
    assert "drug_pathway_ids" in task.input_schema


def test_output_schema_exists(task):
    assert "labels" in task.output_schema
    assert "mask" in task.output_schema


def test_call_returns_list(task, patient):
    result = task(patient)
    assert isinstance(result, list)


def test_returns_one_sample_per_cell_line(task, patient):
    result = task(patient)
    assert len(result) == 1


def test_sample_keys(task, patient):
    sample = task(patient)[0]
    expected = {"patient_id", "visit_id", "gene_indices", "labels", "mask", "drug_pathway_ids"}
    assert expected == set(sample.keys())


def test_patient_id_preserved(task, patient):
    sample = task(patient)[0]
    assert sample["patient_id"] == patient["patient_id"]


def test_visit_id_equals_patient_id(task, patient):
    sample = task(patient)[0]
    assert sample["visit_id"] == sample["patient_id"]


def test_gene_indices_one_indexed(task, patient):
    """Index 0 is reserved for padding and must not appear."""
    sample = task(patient)[0]
    assert 0 not in sample["gene_indices"]


def test_gene_indices_match_active_genes(task, patient):
    gene_expr = patient["gene_expression"]
    expected_count = int(gene_expr.sum())
    sample = task(patient)[0]
    assert len(sample["gene_indices"]) == expected_count


def test_labels_length(task, patient):
    sample = task(patient)[0]
    assert len(sample["labels"]) == N_DRUGS


def test_mask_length(task, patient):
    sample = task(patient)[0]
    assert len(sample["mask"]) == N_DRUGS


def test_mask_binary(task, patient):
    sample = task(patient)[0]
    assert all(v in (0, 1) for v in sample["mask"])


def test_labels_binary(task, patient):
    sample = task(patient)[0]
    assert all(v in (0, 1) for v in sample["labels"])


def test_missing_drugs_masked_zero(task):
    """NaN sensitivity values should produce mask=0 and label=0."""
    patient = {
        "patient_id": "COSMIC.999",
        "gene_expression": np.array([1, 0, 1, 0]),
        "drug_sensitivity": np.array([np.nan, np.nan, np.nan]),
        "drug_pathway_ids": [0, 1, 2],
    }
    sample = task(patient)[0]
    assert sample["mask"] == [0, 0, 0]
    assert sample["labels"] == [0, 0, 0]


def test_all_tested_drugs_fully_masked(task):
    """Fully observed row: all mask entries should be 1."""
    patient = {
        "patient_id": "COSMIC.888",
        "gene_expression": np.array([1, 1, 0, 0]),
        "drug_sensitivity": np.array([1.0, 0.0, 1.0]),
        "drug_pathway_ids": [0, 1, 2],
    }
    sample = task(patient)[0]
    assert sample["mask"] == [1, 1, 1]


def test_drug_pathway_ids_preserved(task, patient):
    sample = task(patient)[0]
    assert sample["drug_pathway_ids"] == list(range(N_DRUGS))

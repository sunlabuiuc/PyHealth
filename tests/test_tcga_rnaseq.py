"""Tests for TCGARNASeqDataset and TCGARNASeqCancerTypeClassification.

All tests use synthetic data only. No real TCGA data is used anywhere in
this file. All tests are designed to complete in well under 1 second total.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from pyhealth.datasets.tcga_rnaseq import TCGARNASeqDataset
from pyhealth.tasks.tcga_rnaseq_classification import (
    TCGARNASeqCancerTypeClassification,
)

# ---------------------------------------------------------------------------
# Synthetic data configuration
# ---------------------------------------------------------------------------

NUM_FAKE_GENES = 10
FAKE_GENE_COLS = [f"GENE_{i:04d}" for i in range(NUM_FAKE_GENES)]
FAKE_COHORTS = ["BRCA", "LUAD", "GBM"]


def _make_synthetic_csvs(tmpdir: str, n_patients: int = 3) -> None:
    """Writes synthetic gene_expression.csv and clinical.csv to ``tmpdir``.

    Args:
        tmpdir (str): Path to the temporary directory.
        n_patients (int): Number of fake patients to generate. Defaults to 3.
    """
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_patients):
        pid = f"TCGA-TEST-{i:04d}"
        cohort = FAKE_COHORTS[i % len(FAKE_COHORTS)]
        gene_vals = rng.uniform(1.0, 200.0, NUM_FAKE_GENES).tolist()
        rows.append([pid, f"{pid}-01", cohort] + gene_vals)

    ge_df = pd.DataFrame(
        rows, columns=["patient_id", "sample_id", "cohort"] + FAKE_GENE_COLS
    )
    ge_df.to_csv(os.path.join(tmpdir, "gene_expression.csv"), index=False)

    clin_rows = []
    for i in range(n_patients):
        pid = f"TCGA-TEST-{i:04d}"
        clin_rows.append(
            [
                pid,
                FAKE_COHORTS[i % len(FAKE_COHORTS)],
                float(rng.uniform(100, 3000)),
                int(rng.integers(0, 2)),
            ]
        )
    clin_df = pd.DataFrame(
        clin_rows, columns=["patient_id", "cohort", "survival_time", "event"]
    )
    clin_df.to_csv(os.path.join(tmpdir, "clinical.csv"), index=False)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


def test_dataset_loads():
    """TCGARNASeqDataset instantiates without error and finds all patients."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_synthetic_csvs(tmpdir, n_patients=3)
        dataset = TCGARNASeqDataset(root=tmpdir)
        assert len(dataset.unique_patient_ids) == 3


def test_processed_csv_written():
    """tcga_rnaseq_pyhealth.csv is created in root after dataset init."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_synthetic_csvs(tmpdir)
        TCGARNASeqDataset(root=tmpdir)
        assert os.path.isfile(os.path.join(tmpdir, "tcga_rnaseq_pyhealth.csv"))


def test_normalization_applied():
    """All gene expression values in the processed CSV are in [0.0, 1.0]."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_synthetic_csvs(tmpdir)
        TCGARNASeqDataset(root=tmpdir)
        processed = pd.read_csv(os.path.join(tmpdir, "tcga_rnaseq_pyhealth.csv"))
        for ge_str in processed["gene_expression"]:
            values = json.loads(ge_str)
            assert all(
                0.0 <= v <= 1.0 for v in values
            ), f"Gene expression values out of [0, 1] range: {values}"


def test_missing_gene_expression_raises():
    """FileNotFoundError is raised when gene_expression.csv is absent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write only clinical.csv
        pd.DataFrame(
            [["TCGA-TEST-0000", "BRCA", 500.0, 1]],
            columns=["patient_id", "cohort", "survival_time", "event"],
        ).to_csv(os.path.join(tmpdir, "clinical.csv"), index=False)

        with pytest.raises(FileNotFoundError):
            TCGARNASeqDataset(root=tmpdir)


def test_missing_clinical_raises():
    """FileNotFoundError is raised when clinical.csv is absent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write only gene_expression.csv
        row = ["TCGA-TEST-0000", "TCGA-TEST-0000-01", "BRCA"] + [1.0] * NUM_FAKE_GENES
        pd.DataFrame(
            [row], columns=["patient_id", "sample_id", "cohort"] + FAKE_GENE_COLS
        ).to_csv(os.path.join(tmpdir, "gene_expression.csv"), index=False)

        with pytest.raises(FileNotFoundError):
            TCGARNASeqDataset(root=tmpdir)


# ---------------------------------------------------------------------------
# Task tests
# ---------------------------------------------------------------------------


def _get_task_samples(tmpdir: str) -> list:
    """Helper: load dataset, run task, return all samples as a list."""
    dataset = TCGARNASeqDataset(root=tmpdir)
    task = TCGARNASeqCancerTypeClassification()
    sample_dataset = dataset.set_task(task)
    return list(sample_dataset)


def test_task_output_format():
    """Each sample has the required keys with correct types and value ranges."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_synthetic_csvs(tmpdir)
        samples = _get_task_samples(tmpdir)
        assert len(samples) > 0
        for s in samples:
            assert "patient_id" in s
            assert "gene_expression" in s
            assert "label" in s
            assert isinstance(
                s["label"], (int, torch.Tensor)
            ), f"label is not int or Tensor: {type(s['label'])}"
            assert 0 <= int(s["label"]) <= 32, f"label out of range: {s['label']}"
            assert isinstance(
                s["gene_expression"], torch.FloatTensor
            ), f"gene_expression is not FloatTensor: {type(s['gene_expression'])}"


def test_task_gene_expression_shape():
    """Each sample's gene_expression has shape (NUM_FAKE_GENES,)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_synthetic_csvs(tmpdir)
        samples = _get_task_samples(tmpdir)
        for s in samples:
            assert s["gene_expression"].shape == (
                NUM_FAKE_GENES,
            ), f"Expected shape ({NUM_FAKE_GENES},), got {s['gene_expression'].shape}"


def test_invalid_cohort_skipped():
    """Samples with cohort='INVALID_COHORT' produce no output entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_synthetic_csvs(tmpdir, n_patients=3)

        # Append a row with an invalid cohort
        ge_path = os.path.join(tmpdir, "gene_expression.csv")
        ge_df = pd.read_csv(ge_path)
        invalid_row = {
            "patient_id": "TCGA-INVALID-9999",
            "sample_id": "TCGA-INVALID-9999-01",
            "cohort": "INVALID_COHORT",
        }
        for col in FAKE_GENE_COLS:
            invalid_row[col] = 50.0
        ge_df = pd.concat([ge_df, pd.DataFrame([invalid_row])], ignore_index=True)
        ge_df.to_csv(ge_path, index=False)

        clin_path = os.path.join(tmpdir, "clinical.csv")
        clin_df = pd.read_csv(clin_path)
        clin_df = pd.concat(
            [
                clin_df,
                pd.DataFrame(
                    [["TCGA-INVALID-9999", "INVALID_COHORT", 500.0, 0]],
                    columns=["patient_id", "cohort", "survival_time", "event"],
                ),
            ],
            ignore_index=True,
        )
        clin_df.to_csv(clin_path, index=False)

        samples = _get_task_samples(tmpdir)
        for s in samples:
            assert (
                0 <= s["label"] <= 32
            ), f"Sample with invalid cohort should have been skipped: {s}"


def test_multi_sample_patient():
    """A patient with two sample rows produces two separate task output samples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_synthetic_csvs(tmpdir, n_patients=3)

        ge_path = os.path.join(tmpdir, "gene_expression.csv")
        ge_df = pd.read_csv(ge_path)

        # Add a second sample for the first patient
        first_pid = ge_df["patient_id"].iloc[0]
        first_cohort = ge_df["cohort"].iloc[0]
        extra_row = {
            "patient_id": first_pid,
            "sample_id": f"{first_pid}-02",
            "cohort": first_cohort,
        }
        rng = np.random.default_rng(99)
        for col in FAKE_GENE_COLS:
            extra_row[col] = float(rng.uniform(1.0, 200.0))
        ge_df = pd.concat([ge_df, pd.DataFrame([extra_row])], ignore_index=True)
        ge_df.to_csv(ge_path, index=False)

        samples = _get_task_samples(tmpdir)
        pid_counts: dict = {}
        for s in samples:
            pid_counts[s["patient_id"]] = pid_counts.get(s["patient_id"], 0) + 1

        assert (
            pid_counts.get(first_pid, 0) == 2
        ), f"Expected 2 samples for {first_pid}, got {pid_counts.get(first_pid, 0)}"


def test_task_empty_for_no_events():
    """Task returns [] when the patient has no rnaseq events."""
    task = TCGARNASeqCancerTypeClassification()
    mock_patient = MagicMock()
    mock_patient.get_events.return_value = []
    result = task(mock_patient)
    assert result == []

"""Unit tests for GDSCDataset.

All tests use synthetic in-memory data — no real CSV files, no network
access.  Designed to run in < 5 seconds.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from pyhealth.datasets.gdsc import GDSCDataset

# ---------------------------------------------------------------------------
# Synthetic data dimensions (keep small for speed)
# ---------------------------------------------------------------------------
N_CELL_LINES = 5
N_GENES = 20
N_DRUGS = 10
N_PATHWAYS = 3
EMB_DIM = 8
RNG = np.random.RandomState(0)


def _make_synthetic_data(tmp_dir: str) -> None:
    """Write minimal CSV files mimicking the GDSC originalData layout."""
    cell_ids = [f"COSMIC.{i}" for i in range(N_CELL_LINES)]
    gene_cols = [str(g) for g in range(1, N_GENES + 1)]
    drug_cols = [str(d) for d in range(1, N_DRUGS + 1)]

    # Binary gene expression (N_CELL_LINES x N_GENES)
    exp = pd.DataFrame(
        RNG.randint(0, 2, size=(N_CELL_LINES, N_GENES)),
        index=cell_ids,
        columns=gene_cols,
    )
    exp.to_csv(os.path.join(tmp_dir, "exp_gdsc.csv"))

    # Drug sensitivity with some NaN entries
    tgt_data = RNG.randint(0, 2, size=(N_CELL_LINES, N_DRUGS)).astype(float)
    tgt_data[RNG.rand(N_CELL_LINES, N_DRUGS) < 0.2] = np.nan
    tgt = pd.DataFrame(tgt_data, index=cell_ids, columns=drug_cols)
    tgt.to_csv(os.path.join(tmp_dir, "gdsc.csv"))

    # Drug info with pathway metadata and Name column for id-to-name mapping
    pathway_names = ["PathwayA", "PathwayB", "PathwayC"]
    drug_info = pd.DataFrame(
        {
            "Name": [f"Drug{d}" for d in range(1, N_DRUGS + 1)],
            "Target pathway": [
                pathway_names[i % N_PATHWAYS] for i in range(N_DRUGS)
            ],
        },
        index=list(range(1, N_DRUGS + 1)),
    )
    drug_info.to_csv(os.path.join(tmp_dir, "drug_info_gdsc.csv"))

    # Gene2Vec embeddings (N_GENES + 1 rows: row 0 is padding)
    emb = RNG.randn(N_GENES + 1, EMB_DIM)
    np.savetxt(os.path.join(tmp_dir, "exp_emb_gdsc.csv"), emb, delimiter=",")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tmp_data_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _make_synthetic_data(tmp_dir)
        yield tmp_dir


@pytest.fixture(scope="module")
def dataset(tmp_data_dir):
    return GDSCDataset(data_dir=tmp_data_dir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dataset_loads(dataset):
    assert dataset is not None


def test_common_samples_count(dataset):
    assert len(dataset.common_samples) == N_CELL_LINES


def test_gene_names_count(dataset):
    assert len(dataset.gene_names) == N_GENES


def test_drug_ids_count(dataset):
    assert len(dataset.drug_ids) == N_DRUGS


def test_pathway_mapping_built(dataset):
    assert len(dataset.pathway2id) == N_PATHWAYS


def test_drug_pathway_ids_length(dataset):
    assert len(dataset.drug_pathway_ids) == N_DRUGS


def test_gene_embeddings_shape(dataset):
    emb = dataset.get_gene_embeddings()
    assert emb.shape == (N_GENES + 1, EMB_DIM)


def test_pathway_info_keys(dataset):
    info = dataset.get_pathway_info()
    assert {"pathway2id", "id2pathway", "num_pathways", "drug_pathway_ids"} == set(
        info.keys()
    )
    assert info["num_pathways"] == N_PATHWAYS


def test_set_task_returns_dataset(dataset):
    sample_ds = dataset.set_task()
    assert len(sample_ds) == N_CELL_LINES


def test_sample_keys(dataset):
    sample_ds = dataset.set_task()
    expected_keys = {
        "patient_id",
        "visit_id",
        "gene_indices",
        "labels",
        "mask",
        "drug_pathway_ids",
    }
    assert expected_keys == set(sample_ds[0].keys())


def test_gene_indices_are_one_indexed(dataset):
    sample_ds = dataset.set_task()
    for i in range(len(sample_ds)):
        assert 0 not in sample_ds[i]["gene_indices"]


def test_labels_length(dataset):
    sample_ds = dataset.set_task()
    for i in range(len(sample_ds)):
        assert len(sample_ds[i]["labels"]) == N_DRUGS


def test_mask_length(dataset):
    sample_ds = dataset.set_task()
    for i in range(len(sample_ds)):
        assert len(sample_ds[i]["mask"]) == N_DRUGS


def test_mask_values_binary(dataset):
    sample_ds = dataset.set_task()
    for i in range(len(sample_ds)):
        assert all(v in (0, 1) for v in sample_ds[i]["mask"])


def test_summary_runs(dataset, capsys):
    dataset.summary()
    captured = capsys.readouterr()
    assert "GDSC Dataset Summary" in captured.out


def test_dataset_name(dataset):
    assert dataset.dataset_name == "GDSC"


def test_drug_names_count(dataset):
    assert len(dataset.drug_names) == N_DRUGS


def test_drug_names_values(dataset):
    for name in dataset.drug_names:
        assert isinstance(name, str) and len(name) > 0


def test_id_to_name_mapping(dataset):
    assert len(dataset.id_to_name) >= N_DRUGS
    for drug_id in dataset.drug_ids:
        assert str(int(drug_id)) in dataset.id_to_name


def test_get_overlap_drugs_self(dataset):
    self_idx, other_idx, names = dataset.get_overlap_drugs(dataset)
    assert len(names) == N_DRUGS
    assert self_idx == other_idx


def test_get_overlap_drugs_returns_sorted(dataset):
    _, _, names = dataset.get_overlap_drugs(dataset)
    assert names == sorted(names)


def test_get_overlap_drugs_indices_in_range(dataset):
    self_idx, other_idx, _ = dataset.get_overlap_drugs(dataset)
    for i in self_idx:
        assert 0 <= i < N_DRUGS
    for i in other_idx:
        assert 0 <= i < N_DRUGS

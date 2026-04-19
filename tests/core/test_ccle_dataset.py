"""Unit tests for CCLEDataset and cross-dataset overlap with GDSCDataset.

All tests use synthetic in-memory data — no real CSV files, no network
access.  Designed to run in < 5 seconds.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from pyhealth.datasets.ccle import CCLEDataset
from pyhealth.datasets.gdsc import GDSCDataset

# ---------------------------------------------------------------------------
# Synthetic data dimensions
# ---------------------------------------------------------------------------
N_CELL_LINES = 4
N_GENES = 20
N_PATHWAYS = 3
EMB_DIM = 8
RNG = np.random.RandomState(42)

SHARED_DRUG_NAMES = ["DrugA", "DrugB", "DrugC"]
CCLE_ONLY_NAMES = ["DrugX", "DrugY"]
GDSC_ONLY_NAMES = ["DrugM", "DrugN", "DrugO", "DrugP", "DrugQ", "DrugR", "DrugS"]


def _make_ccle_data(tmp_dir: str) -> None:
    cell_ids = [f"CCLE.{i}" for i in range(N_CELL_LINES)]
    gene_cols = [str(g) for g in range(1, N_GENES + 1)]
    drug_names = SHARED_DRUG_NAMES + CCLE_ONLY_NAMES

    exp = pd.DataFrame(
        RNG.randint(0, 2, size=(N_CELL_LINES, N_GENES)),
        index=cell_ids, columns=gene_cols,
    )
    exp.to_csv(os.path.join(tmp_dir, "exp_ccle.csv"))

    tgt_data = RNG.randint(0, 2, size=(N_CELL_LINES, len(drug_names))).astype(float)
    tgt_data[RNG.rand(N_CELL_LINES, len(drug_names)) < 0.2] = np.nan
    pd.DataFrame(tgt_data, index=cell_ids, columns=drug_names).to_csv(
        os.path.join(tmp_dir, "ccle.csv")
    )

    pathway_names = ["PathwayA", "PathwayB", "PathwayC"]
    pd.DataFrame(
        {"Target pathway": [pathway_names[i % N_PATHWAYS] for i in range(len(drug_names))]},
        index=drug_names,
    ).to_csv(os.path.join(tmp_dir, "drug_info_ccle.csv"))

    np.savetxt(
        os.path.join(tmp_dir, "exp_emb_ccle.csv"),
        RNG.randn(N_GENES + 1, EMB_DIM), delimiter=","
    )


def _make_gdsc_data(tmp_dir: str) -> None:
    cell_ids = [f"COSMIC.{i}" for i in range(N_CELL_LINES)]
    gene_cols = [str(g) for g in range(1, N_GENES + 1)]
    all_names = SHARED_DRUG_NAMES + GDSC_ONLY_NAMES
    drug_ids = list(range(1, len(all_names) + 1))

    exp = pd.DataFrame(
        RNG.randint(0, 2, size=(N_CELL_LINES, N_GENES)),
        index=cell_ids, columns=gene_cols,
    )
    exp.to_csv(os.path.join(tmp_dir, "exp_gdsc.csv"))

    tgt_data = RNG.randint(0, 2, size=(N_CELL_LINES, len(drug_ids))).astype(float)
    tgt_data[RNG.rand(N_CELL_LINES, len(drug_ids)) < 0.2] = np.nan
    pd.DataFrame(tgt_data, index=cell_ids, columns=[str(d) for d in drug_ids]).to_csv(
        os.path.join(tmp_dir, "gdsc.csv")
    )

    pathway_names = ["PathwayA", "PathwayB", "PathwayC"]
    pd.DataFrame(
        {
            "Name": all_names,
            "Target pathway": [pathway_names[i % N_PATHWAYS] for i in range(len(all_names))],
        },
        index=drug_ids,
    ).to_csv(os.path.join(tmp_dir, "drug_info_gdsc.csv"))

    np.savetxt(
        os.path.join(tmp_dir, "exp_emb_gdsc.csv"),
        RNG.randn(N_GENES + 1, EMB_DIM), delimiter=","
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ccle_tmp_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _make_ccle_data(tmp_dir)
        yield tmp_dir


@pytest.fixture(scope="module")
def gdsc_tmp_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _make_gdsc_data(tmp_dir)
        yield tmp_dir


@pytest.fixture(scope="module")
def ccle_dataset(ccle_tmp_dir):
    return CCLEDataset(data_dir=ccle_tmp_dir)


@pytest.fixture(scope="module")
def gdsc_dataset(gdsc_tmp_dir):
    return GDSCDataset(data_dir=gdsc_tmp_dir)


# ---------------------------------------------------------------------------
# CCLEDataset tests
# ---------------------------------------------------------------------------


def test_ccle_loads(ccle_dataset):
    assert ccle_dataset is not None


def test_ccle_common_samples(ccle_dataset):
    assert len(ccle_dataset.common_samples) == N_CELL_LINES


def test_ccle_gene_names(ccle_dataset):
    assert len(ccle_dataset.gene_names) == N_GENES


def test_ccle_drug_ids(ccle_dataset):
    assert len(ccle_dataset.drug_ids) == len(SHARED_DRUG_NAMES) + len(CCLE_ONLY_NAMES)


def test_ccle_drug_names_equals_drug_ids(ccle_dataset):
    assert ccle_dataset.drug_names == ccle_dataset.drug_ids


def test_ccle_pathway_mapping_built(ccle_dataset):
    assert len(ccle_dataset.pathway2id) <= N_PATHWAYS
    assert len(ccle_dataset.drug_pathway_ids) == len(ccle_dataset.drug_ids)


def test_ccle_gene_embeddings_shape(ccle_dataset):
    assert ccle_dataset.get_gene_embeddings().shape == (N_GENES + 1, EMB_DIM)


def test_ccle_pathway_info_keys(ccle_dataset):
    info = ccle_dataset.get_pathway_info()
    assert {"pathway2id", "id2pathway", "num_pathways", "drug_pathway_ids"} == set(info.keys())


def test_ccle_set_task_returns_dataset(ccle_dataset):
    sample_ds = ccle_dataset.set_task()
    assert len(sample_ds) == N_CELL_LINES


def test_ccle_sample_keys(ccle_dataset):
    sample_ds = ccle_dataset.set_task()
    expected = {"patient_id", "visit_id", "gene_indices", "labels", "mask", "drug_pathway_ids"}
    assert expected == set(sample_ds[0].keys())


def test_ccle_gene_indices_one_indexed(ccle_dataset):
    sample_ds = ccle_dataset.set_task()
    for i in range(len(sample_ds)):
        assert 0 not in sample_ds[i]["gene_indices"]


def test_ccle_labels_length(ccle_dataset):
    n_drugs = len(ccle_dataset.drug_ids)
    sample_ds = ccle_dataset.set_task()
    for i in range(len(sample_ds)):
        assert len(sample_ds[i]["labels"]) == n_drugs


def test_ccle_mask_values_binary(ccle_dataset):
    sample_ds = ccle_dataset.set_task()
    for i in range(len(sample_ds)):
        assert all(v in (0, 1) for v in sample_ds[i]["mask"])


def test_ccle_summary_runs(ccle_dataset, capsys):
    ccle_dataset.summary()
    captured = capsys.readouterr()
    assert "CCLE Dataset Summary" in captured.out


def test_ccle_dataset_name(ccle_dataset):
    assert ccle_dataset.dataset_name == "CCLE"


def test_ccle_missing_data_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CCLEDataset(data_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Cross-dataset overlap tests (GDSC <-> CCLE)
# ---------------------------------------------------------------------------


def test_overlap_drugs_count(gdsc_dataset, ccle_dataset):
    _, _, names = gdsc_dataset.get_overlap_drugs(ccle_dataset)
    assert set(names) == set(SHARED_DRUG_NAMES)


def test_overlap_drugs_sorted(gdsc_dataset, ccle_dataset):
    _, _, names = gdsc_dataset.get_overlap_drugs(ccle_dataset)
    assert names == sorted(names)


def test_overlap_indices_valid_gdsc(gdsc_dataset, ccle_dataset):
    n = len(gdsc_dataset.drug_ids)
    self_idx, _, _ = gdsc_dataset.get_overlap_drugs(ccle_dataset)
    for i in self_idx:
        assert 0 <= i < n


def test_overlap_indices_valid_ccle(gdsc_dataset, ccle_dataset):
    n = len(ccle_dataset.drug_ids)
    _, other_idx, _ = gdsc_dataset.get_overlap_drugs(ccle_dataset)
    for i in other_idx:
        assert 0 <= i < n


def test_overlap_names_match_positions(gdsc_dataset, ccle_dataset):
    self_idx, other_idx, names = gdsc_dataset.get_overlap_drugs(ccle_dataset)
    for i, (gi, ci) in enumerate(zip(self_idx, other_idx)):
        assert gdsc_dataset.drug_names[gi] == names[i]
        assert ccle_dataset.drug_names[ci] == names[i]


def test_overlap_symmetric(gdsc_dataset, ccle_dataset):
    _, _, names_from_gdsc = gdsc_dataset.get_overlap_drugs(ccle_dataset)
    _, _, names_from_ccle = ccle_dataset.get_overlap_drugs(gdsc_dataset)
    assert set(names_from_gdsc) == set(names_from_ccle)

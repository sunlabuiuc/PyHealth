"""Unit tests for GDSCDataset.

Uses synthetic in-memory data — no real CSV files, no network access.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from pyhealth.datasets.gdsc import GDSCDataset

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


@pytest.fixture(scope="module")
def dataset():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _make_synthetic_data(tmp_dir)
        yield GDSCDataset(data_dir=tmp_dir)


def test_loads_and_shapes(dataset):
    assert len(dataset.common_samples) == N_CELL_LINES
    assert len(dataset.gene_names) == N_GENES
    assert len(dataset.drug_ids) == N_DRUGS
    assert len(dataset.drug_pathway_ids) == N_DRUGS
    assert len(dataset.pathway2id) == N_PATHWAYS
    assert dataset.get_gene_embeddings().shape == (N_GENES + 1, EMB_DIM)
    assert dataset.dataset_name == "GDSC"


def test_drug_names(dataset):
    assert len(dataset.drug_names) == N_DRUGS
    assert all(isinstance(n, str) and len(n) > 0 for n in dataset.drug_names)
    assert all(str(int(d)) in dataset.id_to_name for d in dataset.drug_ids)


def test_pathway_info(dataset):
    info = dataset.get_pathway_info()
    assert set(info.keys()) == {"pathway2id", "id2pathway", "num_pathways", "drug_pathway_ids"}
    assert info["num_pathways"] == N_PATHWAYS


def test_set_task(dataset):
    sample_ds = dataset.set_task()
    assert len(sample_ds) == N_CELL_LINES
    sample = sample_ds[0]
    assert set(sample.keys()) == {"patient_id", "visit_id", "gene_indices", "labels", "mask", "drug_pathway_ids"}
    assert 0 not in sample["gene_indices"]          # 1-indexed
    assert len(sample["labels"]) == N_DRUGS
    assert len(sample["mask"]) == N_DRUGS
    assert all(v in (0, 1) for v in sample["mask"])


def test_get_overlap_drugs(dataset):
    self_idx, other_idx, names = dataset.get_overlap_drugs(dataset)
    assert len(names) == N_DRUGS
    assert names == sorted(names)
    assert self_idx == other_idx
    assert all(0 <= i < N_DRUGS for i in self_idx)


def test_summary_runs(dataset, capsys):
    dataset.summary()
    assert "GDSC Dataset Summary" in capsys.readouterr().out

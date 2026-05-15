"""Unit tests for CCLEDataset and cross-dataset overlap with GDSCDataset.

Uses synthetic in-memory data — no real CSV files, no network access.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from pyhealth.datasets.ccle import CCLEDataset
from pyhealth.datasets.gdsc import GDSCDataset

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


@pytest.fixture(scope="module")
def ccle():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _make_ccle_data(tmp_dir)
        yield CCLEDataset(data_dir=tmp_dir)


@pytest.fixture(scope="module")
def gdsc():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _make_gdsc_data(tmp_dir)
        yield GDSCDataset(data_dir=tmp_dir)


def test_ccle_loads_and_shapes(ccle):
    assert len(ccle.common_samples) == N_CELL_LINES
    assert len(ccle.gene_names) == N_GENES
    assert len(ccle.drug_ids) == len(SHARED_DRUG_NAMES) + len(CCLE_ONLY_NAMES)
    assert ccle.drug_names == ccle.drug_ids          # CCLE uses names as column headers
    assert ccle.get_gene_embeddings().shape == (N_GENES + 1, EMB_DIM)
    assert ccle.dataset_name == "CCLE"


def test_ccle_set_task(ccle):
    sample_ds = ccle.set_task()
    assert len(sample_ds) == N_CELL_LINES
    sample = sample_ds[0]
    assert set(sample.keys()) == {"patient_id", "visit_id", "gene_indices", "labels", "mask", "drug_pathway_ids"}
    assert 0 not in sample["gene_indices"]
    assert len(sample["labels"]) == len(ccle.drug_ids)
    assert all(v in (0, 1) for v in sample["mask"])


def test_ccle_missing_data_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CCLEDataset(data_dir=str(tmp_path))


def test_cross_dataset_overlap(gdsc, ccle):
    gdsc_idx, ccle_idx, names = gdsc.get_overlap_drugs(ccle)

    # Correct drugs found
    assert set(names) == set(SHARED_DRUG_NAMES)
    assert names == sorted(names)

    # Indices point to the right names in each dataset
    for i, (gi, ci) in enumerate(zip(gdsc_idx, ccle_idx)):
        assert gdsc.drug_names[gi] == names[i]
        assert ccle.drug_names[ci] == names[i]

    # Symmetric: same result from either side
    _, _, names_rev = ccle.get_overlap_drugs(gdsc)
    assert set(names) == set(names_rev)


def test_ccle_summary_runs(ccle, capsys):
    ccle.summary()
    assert "CCLE Dataset Summary" in capsys.readouterr().out

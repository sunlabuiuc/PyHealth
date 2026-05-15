"""Unit tests for CADRE and CADREDotAttn models.

Uses tiny synthetic inputs — no real data files, no network access.
"""

import numpy as np
import pytest
import torch

from pyhealth.models.cadre import CADRE, collate_fn
from pyhealth.models.cadre_dot_attn import CADREDotAttn

N_GENES_EMB = 21   # 1 padding row + 20 genes
N_DRUGS = 10
N_PATHWAYS = 3
EMB_DIM = 16
BATCH = 3
MAX_GENES = 6


def _gene_emb():
    return np.random.RandomState(0).randn(N_GENES_EMB, EMB_DIM).astype(np.float32)


def _pathway_ids():
    return [i % N_PATHWAYS for i in range(N_DRUGS)]


def _gene_indices():
    return torch.randint(1, N_GENES_EMB, (BATCH, MAX_GENES))


@pytest.fixture
def cadre():
    return CADRE(
        gene_embeddings=_gene_emb(), num_drugs=N_DRUGS, num_pathways=N_PATHWAYS,
        drug_pathway_ids=_pathway_ids(), embedding_dim=EMB_DIM,
        attention_size=8, attention_head=2, dropout_rate=0.0,
    ).eval()


@pytest.fixture
def dot_attn():
    return CADREDotAttn(
        gene_embeddings=_gene_emb(), num_drugs=N_DRUGS,
        embedding_dim=EMB_DIM, num_heads=2, d_k=4, dropout_rate=0.0,
    ).eval()


def test_cadre_forward(cadre):
    gi = _gene_indices()
    labels = torch.randint(0, 2, (BATCH, N_DRUGS)).float()
    mask = torch.ones(BATCH, N_DRUGS)

    out = cadre(gi, labels=labels, mask=mask)
    assert out["probs"].shape == (BATCH, N_DRUGS)
    assert out["logits"].shape == (BATCH, N_DRUGS)
    assert 0.0 <= out["probs"].min() and out["probs"].max() <= 1.0
    assert out["loss"].shape == ()
    assert out["loss"].item() > 0.0
    assert torch.equal(out["y_true"], labels)
    assert cadre.get_attention_weights().shape == (BATCH, N_DRUGS, MAX_GENES)


def test_cadre_no_loss_without_labels(cadre):
    out = cadre(_gene_indices())
    assert "loss" not in out


def test_cadre_ablations():
    """Both use_attention=False and use_cntx_attn=False should still produce valid probs."""
    gi = _gene_indices()
    for kwargs in [{"use_attention": False}, {"use_cntx_attn": False}]:
        model = CADRE(
            gene_embeddings=_gene_emb(), num_drugs=N_DRUGS, num_pathways=N_PATHWAYS,
            drug_pathway_ids=_pathway_ids(), embedding_dim=EMB_DIM,
            attention_size=8, attention_head=2, dropout_rate=0.0, **kwargs,
        ).eval()
        assert model(gi)["probs"].shape == (BATCH, N_DRUGS)


def test_dot_attn_forward(dot_attn):
    gi = _gene_indices()
    out = dot_attn(gi)
    assert out["probs"].shape == (BATCH, N_DRUGS)
    assert 0.0 <= out["probs"].min() and out["probs"].max() <= 1.0
    attn = dot_attn.get_attention_weights()
    assert attn.shape == (BATCH, N_DRUGS, MAX_GENES)
    assert attn.min() >= 0.0


def test_collate_fn():
    batch = [
        {"gene_indices": [1, 2, 3], "labels": [0] * N_DRUGS, "mask": [1] * N_DRUGS, "patient_id": "A"},
        {"gene_indices": [4, 5],    "labels": [1] * N_DRUGS, "mask": [1] * N_DRUGS, "patient_id": "B"},
    ]
    out = collate_fn(batch)
    assert out["gene_indices"].shape == (2, 3)        # padded to longest
    assert out["gene_indices"][1, 2].item() == 0      # padding index
    assert out["labels"].shape == (2, N_DRUGS)

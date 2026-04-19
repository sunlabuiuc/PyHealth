"""Unit tests for CADRE and CADREDotAttn models.

All tests use tiny synthetic inputs — no real data files, no network
access.  Designed to run in < 10 seconds on CPU.
"""

import numpy as np
import pytest
import torch

from pyhealth.models.cadre import CADRE, collate_fn
from pyhealth.models.cadre_dot_attn import CADREDotAttn

# ---------------------------------------------------------------------------
# Tiny model configuration for fast tests
# ---------------------------------------------------------------------------
N_GENES_EMB = 21   # embedding rows: 1 padding + 20 genes
N_DRUGS = 10
N_PATHWAYS = 3
EMB_DIM = 16
ATTN_SIZE = 8
ATTN_HEAD = 2
D_K = 4
BATCH = 3
MAX_GENES = 6


def _gene_emb():
    rng = np.random.RandomState(0)
    return rng.randn(N_GENES_EMB, EMB_DIM).astype(np.float32)


def _pathway_ids():
    return [i % N_PATHWAYS for i in range(N_DRUGS)]


def _gene_indices():
    """Batch of gene index tensors (1-indexed, no zeros)."""
    return torch.randint(1, N_GENES_EMB, (BATCH, MAX_GENES))


def _labels():
    return torch.randint(0, 2, (BATCH, N_DRUGS)).float()


def _mask():
    m = torch.ones(BATCH, N_DRUGS)
    m[0, 0] = 0  # simulate one missing entry
    return m


# ---------------------------------------------------------------------------
# CADRE fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cadre_model():
    return CADRE(
        gene_embeddings=_gene_emb(),
        num_drugs=N_DRUGS,
        num_pathways=N_PATHWAYS,
        drug_pathway_ids=_pathway_ids(),
        embedding_dim=EMB_DIM,
        attention_size=ATTN_SIZE,
        attention_head=ATTN_HEAD,
        dropout_rate=0.0,
    ).eval()


@pytest.fixture
def cadre_no_attn():
    return CADRE(
        gene_embeddings=_gene_emb(),
        num_drugs=N_DRUGS,
        num_pathways=N_PATHWAYS,
        drug_pathway_ids=_pathway_ids(),
        embedding_dim=EMB_DIM,
        attention_size=ATTN_SIZE,
        attention_head=ATTN_HEAD,
        dropout_rate=0.0,
        use_attention=False,
    ).eval()


@pytest.fixture
def cadre_no_cntx():
    return CADRE(
        gene_embeddings=_gene_emb(),
        num_drugs=N_DRUGS,
        num_pathways=N_PATHWAYS,
        drug_pathway_ids=_pathway_ids(),
        embedding_dim=EMB_DIM,
        attention_size=ATTN_SIZE,
        attention_head=ATTN_HEAD,
        dropout_rate=0.0,
        use_cntx_attn=False,
    ).eval()


@pytest.fixture
def dot_model():
    return CADREDotAttn(
        gene_embeddings=_gene_emb(),
        num_drugs=N_DRUGS,
        embedding_dim=EMB_DIM,
        num_heads=ATTN_HEAD,
        d_k=D_K,
        dropout_rate=0.0,
    ).eval()


# ---------------------------------------------------------------------------
# CADRE forward pass tests
# ---------------------------------------------------------------------------


def test_cadre_probs_shape(cadre_model):
    out = cadre_model(_gene_indices())
    assert out["probs"].shape == (BATCH, N_DRUGS)


def test_cadre_logits_shape(cadre_model):
    out = cadre_model(_gene_indices())
    assert out["logits"].shape == (BATCH, N_DRUGS)


def test_cadre_probs_in_zero_one(cadre_model):
    out = cadre_model(_gene_indices())
    assert out["probs"].min() >= 0.0
    assert out["probs"].max() <= 1.0


def test_cadre_loss_scalar(cadre_model):
    out = cadre_model(_gene_indices(), labels=_labels(), mask=_mask())
    assert "loss" in out
    assert out["loss"].shape == ()


def test_cadre_loss_positive(cadre_model):
    out = cadre_model(_gene_indices(), labels=_labels(), mask=_mask())
    assert out["loss"].item() > 0.0


def test_cadre_y_true_in_output(cadre_model):
    labels = _labels()
    out = cadre_model(_gene_indices(), labels=labels, mask=_mask())
    assert "y_true" in out
    assert torch.equal(out["y_true"], labels)


def test_cadre_attention_weights_shape(cadre_model):
    cadre_model(_gene_indices())
    attn = cadre_model.get_attention_weights()
    assert attn is not None
    assert attn.shape == (BATCH, N_DRUGS, MAX_GENES)


def test_cadre_no_attention_probs_shape(cadre_no_attn):
    out = cadre_no_attn(_gene_indices())
    assert out["probs"].shape == (BATCH, N_DRUGS)


def test_cadre_no_cntx_probs_shape(cadre_no_cntx):
    out = cadre_no_cntx(_gene_indices())
    assert out["probs"].shape == (BATCH, N_DRUGS)


def test_cadre_no_loss_without_labels(cadre_model):
    out = cadre_model(_gene_indices())
    assert "loss" not in out


# ---------------------------------------------------------------------------
# CADREDotAttn forward pass tests
# ---------------------------------------------------------------------------


def test_dot_probs_shape(dot_model):
    out = dot_model(_gene_indices())
    assert out["probs"].shape == (BATCH, N_DRUGS)


def test_dot_logits_shape(dot_model):
    out = dot_model(_gene_indices())
    assert out["logits"].shape == (BATCH, N_DRUGS)


def test_dot_probs_in_zero_one(dot_model):
    out = dot_model(_gene_indices())
    assert out["probs"].min() >= 0.0
    assert out["probs"].max() <= 1.0


def test_dot_loss_scalar(dot_model):
    out = dot_model(_gene_indices(), labels=_labels(), mask=_mask())
    assert out["loss"].shape == ()


def test_dot_attention_weights_shape(dot_model):
    dot_model(_gene_indices())
    attn = dot_model.get_attention_weights()
    assert attn is not None
    assert attn.shape == (BATCH, N_DRUGS, MAX_GENES)


def test_dot_attention_non_negative(dot_model):
    dot_model(_gene_indices())
    attn = dot_model.get_attention_weights()
    assert attn.min() >= 0.0


# ---------------------------------------------------------------------------
# Padding behaviour
# ---------------------------------------------------------------------------


def test_cadre_handles_padded_genes(cadre_model):
    """Batch with a mix of gene counts — padded to longest."""
    indices = torch.zeros(BATCH, MAX_GENES, dtype=torch.long)
    for i in range(BATCH):
        n = i + 2  # 2, 3, 4 active genes
        indices[i, :n] = torch.randint(1, N_GENES_EMB, (n,))
    out = cadre_model(indices)
    assert out["probs"].shape == (BATCH, N_DRUGS)


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------


def test_collate_fn_pads_gene_indices():
    batch = [
        {"gene_indices": [1, 2, 3], "labels": [0] * N_DRUGS,
         "mask": [1] * N_DRUGS, "patient_id": "A"},
        {"gene_indices": [4, 5],    "labels": [1] * N_DRUGS,
         "mask": [1] * N_DRUGS, "patient_id": "B"},
    ]
    out = collate_fn(batch)
    assert out["gene_indices"].shape == (2, 3)   # padded to longest
    assert out["gene_indices"][1, 2].item() == 0  # padding index


def test_collate_fn_labels_shape():
    batch = [
        {"gene_indices": [1], "labels": [0] * N_DRUGS,
         "mask": [1] * N_DRUGS, "patient_id": "X"},
    ]
    out = collate_fn(batch)
    assert out["labels"].shape == (1, N_DRUGS)

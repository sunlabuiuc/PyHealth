"""Unit and integration tests for the DILA PyHealth implementation.

Run with:
    pytest tests/models/test_dila.py -v

Tests are intentionally lightweight (small tensors, CPU-only) so they run
quickly without GPU or real MIMIC data.
"""

import os
import tempfile

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# The implementation lives in Project/pyhealth.  Adjust sys.path so the tests
# can be run from the Project root without installing the package.
# ---------------------------------------------------------------------------
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pyhealth.models.dila_sparse_autoencoder import SparseAutoencoder
from pyhealth.models.dila_dict_label_attention import DictionaryLabelAttention
from pyhealth.models.dila import DILA, pretrain_sparse_autoencoder

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

INPUT_DIM = 16
DICT_SIZE = 32
NUM_LABELS = 5
BATCH = 4
SEQ_LEN = 8


# ===========================================================================
# 1. SparseAutoencoder unit tests
# ===========================================================================


class TestSparseAutoencoder:
    """Unit tests for SparseAutoencoder."""

    def _make_sae(self, lambda_l1=1e-4, lambda_l2=1e-5):
        return SparseAutoencoder(
            input_dim=INPUT_DIM,
            dict_size=DICT_SIZE,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
        )

    def test_output_shapes(self):
        """f and x_hat must match (batch, dict_size) and (batch, input_dim)."""
        sae = self._make_sae()
        x = torch.randn(BATCH, INPUT_DIM)
        f, x_hat, losses = sae(x)
        assert f.shape == (
            BATCH,
            DICT_SIZE,
        ), f"Expected ({BATCH}, {DICT_SIZE}), got {f.shape}"
        assert x_hat.shape == (
            BATCH,
            INPUT_DIM,
        ), f"Expected ({BATCH}, {INPUT_DIM}), got {x_hat.shape}"

    def test_loss_keys(self):
        """Returned dict must contain exactly loss_saenc, loss_recon, loss_l1."""
        sae = self._make_sae()
        x = torch.randn(BATCH, INPUT_DIM)
        _, _, losses = sae(x)
        assert set(losses.keys()) == {"loss_saenc", "loss_recon", "loss_l1"}

    def test_non_negativity(self):
        """All feature activations must be >= 0 (ReLU encoder)."""
        sae = self._make_sae()
        x = torch.randn(BATCH * 4, INPUT_DIM)
        f, _, _ = sae(x)
        assert (f >= 0).all(), "Sparse features contain negative values"

    def test_sparsity_under_high_l1(self):
        """After several gradient steps with high lambda_l1, f should be sparse."""
        torch.manual_seed(42)
        sae = self._make_sae(lambda_l1=1.0, lambda_l2=0.0)
        optimizer = torch.optim.AdamW(sae.parameters(), lr=1e-3)
        x = torch.randn(64, INPUT_DIM)

        for _ in range(50):
            f, _, losses = sae(x)
            optimizer.zero_grad()
            losses["loss_saenc"].backward()
            optimizer.step()
            sae.normalize_decoder()

        f, _, _ = sae(x)
        fraction_active = (f > 0).float().mean().item()
        assert (
            fraction_active < 0.9
        ), f"Expected sparse activations, but {fraction_active:.2%} of features are active"

    def test_loss_is_scalar_and_differentiable(self):
        """loss_saenc must be a scalar tensor with a gradient function."""
        sae = self._make_sae()
        x = torch.randn(BATCH, INPUT_DIM)
        _, _, losses = sae(x)
        loss = losses["loss_saenc"]
        assert loss.dim() == 0, "loss_saenc should be a scalar tensor"
        assert loss.grad_fn is not None, "loss_saenc should be differentiable"

    def test_normalize_decoder(self):
        """After normalize_decoder(), all decoder column norms should be ~1."""
        sae = self._make_sae()
        # Perturb decoder weights to have non-unit norms
        with torch.no_grad():
            sae.decoder.weight.data *= 5.0
        sae.normalize_decoder()
        col_norms = sae.decoder.weight.norm(dim=0)
        assert torch.allclose(
            col_norms, torch.ones_like(col_norms), atol=1e-5
        ), "Decoder column norms are not ~1 after normalize_decoder()"


# ===========================================================================
# 2. DictionaryLabelAttention unit tests
# ===========================================================================


class TestDictionaryLabelAttention:
    """Unit tests for DictionaryLabelAttention."""

    def _make_modules(self):
        sae = SparseAutoencoder(INPUT_DIM, DICT_SIZE)
        attn = DictionaryLabelAttention(sae, NUM_LABELS, INPUT_DIM)
        return sae, attn

    def test_output_shapes(self):
        """logits must be (batch, num_labels); aux_losses must have loss_saenc."""
        _, attn = self._make_modules()
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        logits, aux_losses = attn(x)
        assert logits.shape == (
            BATCH,
            NUM_LABELS,
        ), f"Expected ({BATCH}, {NUM_LABELS}), got {logits.shape}"
        assert "loss_saenc" in aux_losses

    def test_attention_sums_to_one_over_seq_len(self):
        """Softmax over seq_len: each label's attention weights across tokens must sum to 1."""
        sae, attn = self._make_modules()
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        x_flat = x.reshape(BATCH * SEQ_LEN, INPUT_DIM)
        f_flat = sae.encode(x_flat)
        f_note = f_flat.view(BATCH, SEQ_LEN, DICT_SIZE)
        attn_logits = f_note @ attn.icd_projection.t()  # (B, S, C)
        a_laat = F.softmax(attn_logits, dim=1)  # softmax over S
        col_sums = a_laat.sum(dim=1)  # (B, C)
        assert torch.allclose(
            col_sums, torch.ones_like(col_sums), atol=1e-5
        ), "Attention weights do not sum to 1 over seq_len"

    def test_initialize_from_icd_descriptions(self):
        """initialize_from_icd_descriptions must copy values into icd_projection."""
        _, attn = self._make_modules()
        init_matrix = torch.rand(NUM_LABELS, DICT_SIZE)
        attn.initialize_from_icd_descriptions(init_matrix)
        assert torch.allclose(
            attn.icd_projection.data, init_matrix
        ), "icd_projection was not updated by initialize_from_icd_descriptions()"

    def test_initialize_wrong_shape_raises(self):
        """initialize_from_icd_descriptions should raise on shape mismatch."""
        _, attn = self._make_modules()
        bad_matrix = torch.rand(NUM_LABELS + 1, DICT_SIZE)
        with pytest.raises(ValueError):
            attn.initialize_from_icd_descriptions(bad_matrix)

    def test_logits_are_differentiable(self):
        """Loss computed from logits must be backprop-able."""
        _, attn = self._make_modules()
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        logits, _ = attn(x)
        targets = torch.zeros(BATCH, NUM_LABELS)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()


# ===========================================================================
# 3. DILA model integration tests
# ===========================================================================


def _build_mock_dataset(
    embedding_dim=INPUT_DIM, seq_len=SEQ_LEN, n_labels=NUM_LABELS, n_samples=6
):
    """Return a minimal in-memory SampleDataset with tensor embeddings and multilabel targets."""
    # Import here to avoid hard dependency at module level during SAE-only tests
    from pyhealth.datasets import create_sample_dataset

    label_names = [f"label_{i}" for i in range(n_labels)]
    samples = []
    for idx in range(n_samples):
        # Deterministic embeddings so the test is reproducible
        emb = [
            [(float(idx + 1) * 0.01 * (t + 1) * (d + 1)) for d in range(embedding_dim)]
            for t in range(seq_len)
        ]
        # Assign two labels per sample in a round-robin fashion
        labels = [label_names[idx % n_labels], label_names[(idx + 1) % n_labels]]
        samples.append(
            {
                "patient_id": f"p{idx}",
                "visit_id": f"v{idx}",
                "embeddings": emb,
                "labels": labels,
            }
        )

    dataset = create_sample_dataset(
        samples=samples,
        input_schema={"embeddings": "tensor"},
        output_schema={"labels": "multilabel"},
        dataset_name="dila_test",
    )
    return dataset


class TestDILAIntegration:
    """Integration tests for the DILA BaseModel subclass."""

    def test_forward_output_keys_without_labels(self):
        """Forward without label key must return logit and y_prob only."""
        dataset = _build_mock_dataset()
        model = DILA(
            dataset,
            feature_key="embeddings",
            label_key="labels",
            embedding_dim=INPUT_DIM,
            dict_size=DICT_SIZE,
        )
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        out = model(embeddings=x)
        assert "logit" in out
        assert "y_prob" in out
        assert "loss" not in out

    def test_forward_output_keys_with_labels(self):
        """Forward with label key must include loss, loss_bce, loss_saenc, y_true."""
        dataset = _build_mock_dataset()
        model = DILA(
            dataset,
            feature_key="embeddings",
            label_key="labels",
            embedding_dim=INPUT_DIM,
            dict_size=DICT_SIZE,
        )
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        y = torch.zeros(BATCH, NUM_LABELS)
        out = model(embeddings=x, labels=y)
        for key in ("logit", "y_prob", "loss", "loss_bce", "loss_saenc", "y_true"):
            assert key in out, f"Missing key: {key}"

    def test_y_prob_in_unit_interval(self):
        """y_prob values must lie in [0, 1]."""
        dataset = _build_mock_dataset()
        model = DILA(
            dataset,
            feature_key="embeddings",
            label_key="labels",
            embedding_dim=INPUT_DIM,
            dict_size=DICT_SIZE,
        )
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        out = model(embeddings=x)
        y_prob = out["y_prob"]
        assert (y_prob >= 0).all() and (
            y_prob <= 1
        ).all(), f"y_prob out of [0, 1]: min={y_prob.min():.4f}, max={y_prob.max():.4f}"

    def test_loss_is_scalar_and_differentiable(self):
        """loss must be a scalar tensor that supports backpropagation."""
        dataset = _build_mock_dataset()
        model = DILA(
            dataset,
            feature_key="embeddings",
            label_key="labels",
            embedding_dim=INPUT_DIM,
            dict_size=DICT_SIZE,
        )
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        y = torch.zeros(BATCH, NUM_LABELS)
        out = model(embeddings=x, labels=y)
        loss = out["loss"]
        assert loss.dim() == 0, "loss should be a scalar tensor"
        assert loss.grad_fn is not None, "loss should be differentiable"
        loss.backward()

    def test_logit_shape(self):
        """logit must have shape (batch, num_labels)."""
        dataset = _build_mock_dataset()
        model = DILA(
            dataset,
            feature_key="embeddings",
            label_key="labels",
            embedding_dim=INPUT_DIM,
            dict_size=DICT_SIZE,
        )
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        out = model(embeddings=x)
        assert out["logit"].shape == (BATCH, NUM_LABELS)

    def test_dataloader_forward(self):
        """Full round-trip through create_sample_dataset → get_dataloader → model.forward."""
        from pyhealth.datasets import get_dataloader

        dataset = _build_mock_dataset()
        model = DILA(
            dataset,
            feature_key="embeddings",
            label_key="labels",
            embedding_dim=INPUT_DIM,
            dict_size=DICT_SIZE,
        )
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        out = model(**batch)
        assert "logit" in out
        assert "loss" in out


# ===========================================================================
# 4. Two-stage training smoke tests
# ===========================================================================


class TestTwoStageTraining:
    """Smoke tests for Stage-1 pretraining and Stage-2 loading."""

    def test_pretrain_loss_decreases(self):
        """SAE loss should decrease over pretraining epochs."""
        torch.manual_seed(0)
        sae = SparseAutoencoder(INPUT_DIM, DICT_SIZE, lambda_l1=1e-3, lambda_l2=0.0)
        embeddings = torch.randn(200, INPUT_DIM)

        # Record initial loss
        with torch.no_grad():
            _, _, losses_before = sae(embeddings)
        loss_before = losses_before["loss_saenc"].item()

        pretrain_sparse_autoencoder(
            sae,
            embeddings,
            epochs=5,
            lr=1e-3,
            batch_size=32,
            device="cpu",
        )

        with torch.no_grad():
            _, _, losses_after = sae(embeddings)
        loss_after = losses_after["loss_saenc"].item()

        assert (
            loss_after < loss_before
        ), f"SAE loss did not decrease: before={loss_before:.4f}, after={loss_after:.4f}"

    def test_save_and_load_pretrained_weights(self):
        """Pretrained SAE weights should be loadable into a DILA model."""
        from pyhealth.datasets import create_sample_dataset

        torch.manual_seed(1)
        sae = SparseAutoencoder(INPUT_DIM, DICT_SIZE)
        embeddings = torch.randn(64, INPUT_DIM)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "sae.pt")
            pretrain_sparse_autoencoder(
                sae,
                embeddings,
                epochs=2,
                lr=1e-3,
                batch_size=32,
                device="cpu",
                save_path=save_path,
            )

            # Build a minimal dataset so DILA can be constructed
            label_names = [f"label_{i}" for i in range(NUM_LABELS)]
            samples = [
                {
                    "patient_id": f"p{i}",
                    "visit_id": f"v{i}",
                    "embeddings": [[0.1] * INPUT_DIM] * SEQ_LEN,
                    "labels": [label_names[i % NUM_LABELS]],
                }
                for i in range(6)
            ]
            dataset = create_sample_dataset(
                samples=samples,
                input_schema={"embeddings": "tensor"},
                output_schema={"labels": "multilabel"},
            )

            model = DILA(
                dataset,
                feature_key="embeddings",
                label_key="labels",
                embedding_dim=INPUT_DIM,
                dict_size=DICT_SIZE,
                pretrained_autoencoder_path=save_path,
            )

        # Verify that the loaded weights match the pretrained SAE
        for (name, param), (_, loaded_param) in zip(
            sae.named_parameters(), model.autoencoder.named_parameters()
        ):
            assert torch.allclose(
                param, loaded_param
            ), f"Parameter '{name}' mismatch after loading pretrained weights"

        # Verify forward pass still works
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        out = model(embeddings=x)
        assert "logit" in out
        assert out["y_prob"].shape == (BATCH, NUM_LABELS)
        
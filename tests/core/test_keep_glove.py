"""Tests for KEEP regularized GloVe training (Stage 2).

Uses tiny synthetic co-occurrence matrices (3-5 codes) to verify
GloVe loss computation, regularization, and training convergence
without real patient data.
"""

import numpy as np
import torch
import pytest


class TestCooccurrenceDataset:
    """Tests for CooccurrenceDataset."""

    def test_enumerates_nonzero_entries(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.train_glove import (
            CooccurrenceDataset,
        )

        matrix = np.array([
            [0, 2, 0],
            [2, 0, 1],
            [0, 1, 0],
        ], dtype=np.float32)

        ds = CooccurrenceDataset(matrix)
        # 4 non-zero entries: (0,1), (1,0), (1,2), (2,1)
        assert len(ds) == 4

    def test_returns_correct_triples(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.train_glove import (
            CooccurrenceDataset,
        )

        matrix = np.array([
            [0, 5],
            [5, 0],
        ], dtype=np.float32)

        ds = CooccurrenceDataset(matrix)
        i, j, count = ds[0]
        assert count.item() == 5.0


class TestKeepGloVe:
    """Tests for KeepGloVe model."""

    def test_forward_returns_scalar_loss(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.train_glove import (
            KeepGloVe,
        )

        model = KeepGloVe(vocab_size=5, embedding_dim=4)
        row = torch.tensor([0, 1, 2])
        col = torch.tensor([1, 2, 0])
        counts = torch.tensor([3.0, 1.0, 2.0])

        glove_loss, reg_loss = model(row, col, counts)
        assert glove_loss.dim() == 0  # scalar
        assert glove_loss.item() > 0  # should be positive
        # No init_embeddings, so reg_loss should be 0
        assert reg_loss.item() == 0.0

    def test_regularization_increases_loss(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.train_glove import (
            KeepGloVe,
        )

        np.random.seed(42)
        init = np.random.randn(5, 4).astype(np.float32)

        # Without regularization
        model_noreg = KeepGloVe(5, 4, init_embeddings=init, lambd=0.0)
        # With regularization
        model_reg = KeepGloVe(5, 4, init_embeddings=init, lambd=1.0)

        row = torch.tensor([0, 1])
        col = torch.tensor([1, 2])
        counts = torch.tensor([3.0, 1.0])

        # Perturb weights so regularization kicks in
        with torch.no_grad():
            model_reg.emb_u.weight.add_(torch.randn_like(model_reg.emb_u.weight))

        glove_noreg, reg_noreg = model_noreg(row, col, counts)
        glove_reg, reg_reg = model_reg(row, col, counts)

        # With lambd=0, reg loss should be 0
        assert reg_noreg.item() == 0.0
        # With lambd=1 and perturbed weights, reg loss should be > 0
        assert reg_reg.item() > 0

    def test_get_embeddings_shape(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.train_glove import (
            KeepGloVe,
        )

        model = KeepGloVe(vocab_size=10, embedding_dim=8)
        emb = model.get_embeddings()
        assert emb.shape == (10, 8)

    def test_get_embeddings_averages_u_and_v(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.train_glove import (
            KeepGloVe,
        )

        model = KeepGloVe(vocab_size=3, embedding_dim=2)
        with torch.no_grad():
            model.emb_u.weight.fill_(2.0)
            model.emb_v.weight.fill_(4.0)

        emb = model.get_embeddings()
        # Average of 2.0 and 4.0 = 3.0
        np.testing.assert_allclose(emb, 3.0)


class TestTrainKeep:
    """Tests for train_keep() end-to-end training."""

    def test_returns_correct_shape(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.train_glove import (
            train_keep,
        )

        # Tiny matrix: 3 codes
        matrix = np.array([
            [2, 3, 0],
            [3, 2, 1],
            [0, 1, 1],
        ], dtype=np.float32)

        emb = train_keep(
            matrix, embedding_dim=4, epochs=5, batch_size=4, seed=42,
        )
        assert emb.shape == (3, 4)

    def test_with_init_embeddings(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.train_glove import (
            train_keep,
        )

        np.random.seed(42)
        matrix = np.array([
            [2, 5, 1],
            [5, 3, 2],
            [1, 2, 1],
        ], dtype=np.float32)
        init = np.random.randn(3, 4).astype(np.float32)

        emb = train_keep(
            matrix,
            init_embeddings=init,
            embedding_dim=4,
            epochs=10,
            batch_size=4,
            lambd=1e-3,
            seed=42,
        )
        assert emb.shape == (3, 4)
        # Embeddings should have changed from init
        assert not np.allclose(emb, init, atol=0.01)

    def test_loss_decreases(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.train_glove import (
            KeepGloVe, CooccurrenceDataset,
        )

        matrix = np.array([
            [5, 10, 1],
            [10, 5, 3],
            [1, 3, 2],
        ], dtype=np.float32)

        model = KeepGloVe(3, 4)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.05)
        ds = CooccurrenceDataset(matrix)

        losses = []
        for epoch in range(20):
            total = 0.0
            for i in range(len(ds)):
                r, c, cnt = ds[i]
                glove_loss, reg_loss = model(
                    r.unsqueeze(0), c.unsqueeze(0), cnt.unsqueeze(0),
                )
                loss = glove_loss + reg_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            losses.append(total)

        # Loss should decrease over 20 epochs
        assert losses[-1] < losses[0]

    def test_empty_matrix(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.train_glove import (
            train_keep,
        )

        matrix = np.zeros((0, 0), dtype=np.float32)
        emb = train_keep(matrix, embedding_dim=4, epochs=1)
        assert emb.shape == (0, 4)

    def test_deterministic_with_seed(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.train_glove import (
            train_keep,
        )

        matrix = np.array([
            [2, 3],
            [3, 2],
        ], dtype=np.float32)

        emb1 = train_keep(matrix, embedding_dim=4, epochs=5, seed=42)
        emb2 = train_keep(matrix, embedding_dim=4, epochs=5, seed=42)
        np.testing.assert_array_equal(emb1, emb2)

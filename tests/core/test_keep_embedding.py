import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# pyhealth.models.__init__ imports every model in the package, each of which
# pulls in its own optional deps (einops, litdata, polars, rdkit, …).
# Mock out everything except keep_embedding and base_model so Python never
# loads those files, keeping test imports fast and dep-free.
_datasets_mock = MagicMock()
_datasets_mock.SampleDataset = MagicMock
sys.modules.setdefault("pyhealth.datasets", _datasets_mock)
sys.modules.setdefault("pyhealth.processors", MagicMock())

for _mod in (
    "pyhealth.models.adacare",
    "pyhealth.models.agent",
    "pyhealth.models.biot",
    "pyhealth.models.cnn",
    "pyhealth.models.concare",
    "pyhealth.models.contrawr",
    "pyhealth.models.deepr",
    "pyhealth.models.embedding",
    "pyhealth.models.gamenet",
    "pyhealth.models.jamba_ehr",
    "pyhealth.models.logistic_regression",
    "pyhealth.models.gan",
    "pyhealth.models.gnn",
    "pyhealth.models.graph_torchvision_model",
    "pyhealth.models.graphcare",
    "pyhealth.models.grasp",
    "pyhealth.models.medlink",
    "pyhealth.models.micron",
    "pyhealth.models.mlp",
    "pyhealth.models.molerec",
    "pyhealth.models.retain",
    "pyhealth.models.rnn",
    "pyhealth.models.safedrug",
    "pyhealth.models.sparcnet",
    "pyhealth.models.stagenet",
    "pyhealth.models.stagenet_mha",
    "pyhealth.models.tcn",
    "pyhealth.models.tfm_tokenizer",
    "pyhealth.models.torchvision_model",
    "pyhealth.models.transformer",
    "pyhealth.models.transformers_model",
    "pyhealth.models.ehrmamba",
    "pyhealth.models.vae",
    "pyhealth.models.vision_embedding",
    "pyhealth.models.text_embedding",
    "pyhealth.models.sdoh",
    "pyhealth.models.unified_embedding",
    "pyhealth.models.transformer_deid",
    "pyhealth.models.califorest",
):
    sys.modules.setdefault(_mod, MagicMock())

from pyhealth.models.keep_embedding import N2V, KeepEmbedding


# Tiny sizes so every test finishes in milliseconds.
NUM_CONCEPTS = 8
EMBEDDING_DIM = 4


class TestN2VHelpers(unittest.TestCase):
    """Test N2V helper methods that require no CSV files or graph construction."""

    def setUp(self):
        """Set up a minimal N2V instance."""
        self.n2v = N2V(
            embedding_dim=EMBEDDING_DIM,
            walk_length=5,
            num_walks=5,
        )

    def test_build_index_mapping(self):
        """Test that concept string keys are mapped to integer indices."""
        wv = MagicMock()
        wv.index_to_key = ["100", "200", "300"]
        mapping = self.n2v._build_index_mapping(wv)
        self.assertEqual(mapping, {100: 0, 200: 1, 300: 2})

    def test_get_vector_iso_found(self):
        """Test that the correct embedding is returned for a known concept."""
        wv = MagicMock()
        wv.index_to_key = ["42"]
        vec = np.array([1.0, 2.0])
        wv.get_vector.return_value = vec
        result = self.n2v._get_vector_iso("42", wv, {42: 0}, np.zeros(2))
        np.testing.assert_array_equal(result, vec)

    def test_get_vector_iso_missing_returns_mean(self):
        """Test that the mean vector is returned for an unknown concept."""
        wv = MagicMock()
        wv.index_to_key = []
        mean_vec = np.array([0.5, 0.5])
        result = self.n2v._get_vector_iso("999", wv, {}, mean_vec)
        np.testing.assert_array_equal(result, mean_vec)


class TestKeepEmbeddingInit(unittest.TestCase):
    """Test KeepEmbedding initialization with a mocked N2V embedding matrix."""

    def setUp(self):
        """Set up a KeepEmbedding instance with N2V mocked out."""
        fake_matrix = np.random.randn(NUM_CONCEPTS, EMBEDDING_DIM).astype(np.float32)
        fake_keys = list(range(NUM_CONCEPTS))
        with patch.object(N2V, "generate_embeddings", return_value=(fake_matrix, fake_keys)):
            self.model = KeepEmbedding(
                dataset=None,
                graph=MagicMock(),
                embedding_dim=EMBEDDING_DIM,
                walk_length=5,
                num_walks=5,
                num_words=NUM_CONCEPTS,
                device="cpu",
            )

    def test_embedding_shapes(self):
        """Test that embedding layers are created with the correct dimensions."""
        self.assertEqual(self.model.embeddings_v.num_embeddings, NUM_CONCEPTS)
        self.assertEqual(self.model.embeddings_v.embedding_dim, EMBEDDING_DIM)
        self.assertEqual(self.model.initial_embeddings.shape, (NUM_CONCEPTS, EMBEDDING_DIM))

    def test_biases_initialized_to_zero(self):
        """Test that bias embeddings are initialized to zero."""
        self.assertTrue(torch.all(self.model.biases_v.weight.data == 0))
        self.assertTrue(torch.all(self.model.biases_u.weight.data == 0))


class TestKeepEmbeddingForward(unittest.TestCase):
    """Test KeepEmbedding forward pass across regularization configurations."""

    def _make_model(self, lambda_reg=1.0, reg_norm=None, log_scale=False):
        """Return a KeepEmbedding with N2V mocked out."""
        fake_matrix = np.random.randn(NUM_CONCEPTS, EMBEDDING_DIM).astype(np.float32)
        fake_keys = list(range(NUM_CONCEPTS))
        with patch.object(N2V, "generate_embeddings", return_value=(fake_matrix, fake_keys)):
            return KeepEmbedding(
                dataset=None,
                graph=MagicMock(),
                embedding_dim=EMBEDDING_DIM,
                walk_length=5,
                num_walks=5,
                num_words=NUM_CONCEPTS,
                lambda_reg=lambda_reg,
                reg_norm=reg_norm,
                log_scale=log_scale,
                device="cpu",
            )

    def _glove_batch(self, batch_size=3):
        """Return a minimal GloVe batch with random indices."""
        return {
            "i_indices": torch.randint(0, NUM_CONCEPTS, (batch_size,)),
            "j_indices": torch.randint(0, NUM_CONCEPTS, (batch_size,)),
            "counts": torch.rand(batch_size) * 10 + 1,
            "weights": torch.rand(batch_size),
        }

    def setUp(self):
        """Set up a default KeepEmbedding model."""
        self.model = self._make_model(lambda_reg=1.0)

    def test_output_keys(self):
        """Test that the forward pass returns all expected output keys."""
        ret = self.model(**self._glove_batch())
        for key in ("loss", "logit", "y_prob", "y_true", "reg_loss"):
            self.assertIn(key, ret)

    def test_loss_is_scalar(self):
        """Test that the total loss is a scalar tensor."""
        self.assertEqual(self.model(**self._glove_batch())["loss"].dim(), 0)

    def test_placeholder_shapes(self):
        """Test that placeholder output tensors match the batch size."""
        ret = self.model(**self._glove_batch(batch_size=3))
        self.assertEqual(ret["y_prob"].shape[0], 3)
        self.assertEqual(ret["y_true"].shape[0], 3)

    def test_no_inputs_returns_zero_loss(self):
        """Test that calling forward with no inputs returns zero loss."""
        self.assertEqual(self.model()["loss"].item(), 0.0)

    def test_no_regularization(self):
        """Test that lambda_reg=0 produces zero regularization loss."""
        ret = self._make_model(lambda_reg=0.0)(**self._glove_batch())
        self.assertEqual(ret["reg_loss"].item(), 0.0)

    def test_lp_norm_regularization(self):
        """Test that Lp norm regularization runs without error."""
        ret = self._make_model(lambda_reg=1.0, reg_norm=2)(**self._glove_batch())
        self.assertEqual(ret["loss"].dim(), 0)

    def test_log_scale_regularization(self):
        """Test that log-scale cosine regularization runs without error."""
        ret = self._make_model(lambda_reg=1.0, log_scale=True)(**self._glove_batch())
        self.assertEqual(ret["loss"].dim(), 0)


class TestKeepEmbeddingBackward(unittest.TestCase):
    """Test that gradients flow correctly through KeepEmbedding."""

    def _make_model(self, lambda_reg=1.0):
        """Return a KeepEmbedding with N2V mocked out."""
        fake_matrix = np.random.randn(NUM_CONCEPTS, EMBEDDING_DIM).astype(np.float32)
        fake_keys = list(range(NUM_CONCEPTS))
        with patch.object(N2V, "generate_embeddings", return_value=(fake_matrix, fake_keys)):
            return KeepEmbedding(
                dataset=None,
                graph=MagicMock(),
                embedding_dim=EMBEDDING_DIM,
                walk_length=5,
                num_walks=5,
                num_words=NUM_CONCEPTS,
                lambda_reg=lambda_reg,
                device="cpu",
            )

    def _glove_batch(self, batch_size=3):
        """Return a minimal GloVe batch with random indices."""
        return {
            "i_indices": torch.randint(0, NUM_CONCEPTS, (batch_size,)),
            "j_indices": torch.randint(0, NUM_CONCEPTS, (batch_size,)),
            "counts": torch.rand(batch_size) * 10 + 1,
            "weights": torch.rand(batch_size),
        }

    def _has_grads(self, model):
        return any(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_gradients_flow_with_regularization(self):
        """Test that gradients flow through the GloVe + regularization loss."""
        model = self._make_model(lambda_reg=1.0)
        model(**self._glove_batch())["loss"].backward()
        self.assertTrue(self._has_grads(model))

    def test_gradients_flow_without_regularization(self):
        """Test that gradients flow through the GloVe-only loss."""
        model = self._make_model(lambda_reg=0.0)
        model(**self._glove_batch())["loss"].backward()
        self.assertTrue(self._has_grads(model))


if __name__ == "__main__":
    unittest.main()

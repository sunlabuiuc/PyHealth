"""Synthetic data tests for CNNLSTMPredictor

These tests run in milliseconds and validate: instantiation, forward pass,
output shapes, gradient flow, loss computation, and hyperparameter configs.

Uses a small SampleBuilder + SampleDataset with synthetic codes (no real
data). All tests use 2-3 synthetic patients with temporary directories.
"""

import os
import pickle
import random
import shutil
import tempfile
import time
import unittest

import numpy as np
import torch

from pyhealth.models.cnn_lstm import CNNLSTMPredictor

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def make_synthetic_dataset(n_patients: int = 3):
    """Create a tiny synthetic dataset for testing CNNLSTMPredictor.

    Always includes at least 2 samples with both label values (0 and 1)
    to satisfy BinaryLabelProcessor.

    Args:
        n_patients: number of synthetic patients to generate.

    Returns:
        a SampleDataset backed by a temporary directory.
    """
    from pyhealth.datasets.sample_dataset import SampleBuilder, SampleDataset

    code_pool_c = [f"C{i:04d}" for i in range(20)]
    code_pool_p = [f"P{i:04d}" for i in range(15)]
    samples = []
    for i in range(max(n_patients, 2)):
        samples.append(
            {
                "patient_id": f"synth_{i}",
                "visit_id": f"visit_{i}",
                "conditions": random.sample(
                    code_pool_c, random.randint(1, 5)
                ),
                "procedures": random.sample(
                    code_pool_p, random.randint(1, 3)
                ),
                "mortality": i % 2,
            }
        )

    input_schema = {"conditions": "sequence", "procedures": "sequence"}
    output_schema = {"mortality": "binary"}

    builder = SampleBuilder(input_schema, output_schema)
    builder.fit(samples)

    tmpdir = tempfile.mkdtemp()
    builder.save(os.path.join(tmpdir, "schema.pkl"))

    import litdata

    litdata.optimize(
        fn=builder.transform,
        inputs=[{"sample": pickle.dumps(s)} for s in samples],
        output_dir=tmpdir,
        chunk_bytes="64MB",
        num_workers=0,
    )
    return SampleDataset(path=tmpdir), tmpdir


class TestCNNLSTMPredictor(unittest.TestCase):
    """Tests for the CNNLSTMPredictor model"""

    @classmethod
    def setUpClass(cls):
        """Create shared synthetic dataset and dataloader"""
        cls.dataset, cls._tmpdir = make_synthetic_dataset(n_patients=3)

        from pyhealth.datasets import get_dataloader

        cls.loader = get_dataloader(
            cls.dataset, batch_size=3, shuffle=False
        )
        cls.batch = next(iter(cls.loader))

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary directory created by setUpClass"""
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def _make_model(self, **kwargs):
        """Helper to instantiate a CNNLSTMPredictor with defaults"""
        defaults = {
            "embedding_dim": 32,
            "hidden_dim": 32,
            "num_cnn_layers": 1,
            "num_lstm_layers": 1,
            "dropout": 0.1,
        }
        defaults.update(kwargs)
        return CNNLSTMPredictor(dataset=self.dataset, **defaults)

    def test_instantiation(self):
        """Test model instantiation and attribute setup"""
        t0 = time.time()
        from pyhealth.models import BaseModel

        model = self._make_model()
        self.assertIsInstance(model, BaseModel)
        self.assertEqual(
            set(model.feature_keys), {"conditions", "procedures"}
        )
        self.assertEqual(model.label_keys, ["mortality"])
        self.assertEqual(model.mode, "binary")
        elapsed = time.time() - t0
        self.assertLess(elapsed, 5.0, "Instantiation took too long")

    def test_forward_pass_output_shapes(self):
        """Test forward pass returns correct keys and shapes"""
        model = self._make_model()
        model.eval()
        with torch.no_grad():
            output = model.forward(**self.batch)

        self.assertIn("loss", output)
        self.assertIn("y_prob", output)
        self.assertIn("y_true", output)
        self.assertIn("logit", output)

        batch_size = output["y_prob"].shape[0]
        self.assertEqual(output["y_prob"].shape, (batch_size, 1))
        self.assertEqual(output["logit"].shape, (batch_size, 1))

        # y_prob should be in [0, 1]
        self.assertTrue((output["y_prob"] >= 0).all())
        self.assertTrue((output["y_prob"] <= 1).all())

    def test_gradient_computation(self):
        """Test that gradients flow through the model"""
        model = self._make_model()
        model.train()
        output = model(**self.batch)
        output["loss"].backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        self.assertTrue(has_grad, "No gradients computed")

    def test_hyperparameter_configs(self):
        """Test various hyperparameter combinations including batch size"""
        from pyhealth.datasets import get_dataloader

        configs = [
            {
                "embedding_dim": 16,
                "hidden_dim": 16,
                "num_cnn_layers": 1,
                "num_lstm_layers": 1,
                "dropout": 0.0,
                "batch_size": 1,
            },
            {
                "embedding_dim": 64,
                "hidden_dim": 64,
                "num_cnn_layers": 3,
                "num_lstm_layers": 3,
                "dropout": 0.5,
                "batch_size": 2,
            },
            {
                "embedding_dim": 32,
                "hidden_dim": 128,
                "num_cnn_layers": 2,
                "num_lstm_layers": 1,
                "dropout": 0.2,
                "batch_size": 3,
            },
        ]
        for cfg in configs:
            bs = cfg.pop("batch_size")
            model = self._make_model(**cfg)
            model.eval()
            loader = get_dataloader(
                self.dataset, batch_size=bs, shuffle=False
            )
            batch = next(iter(loader))
            with torch.no_grad():
                out = model(**batch)
            actual_bs = out["y_prob"].shape[0]
            self.assertLessEqual(actual_bs, bs)
            self.assertEqual(out["y_prob"].shape, (actual_bs, 1))
            self.assertEqual(out["logit"].shape, (actual_bs, 1))
            self.assertEqual(out["loss"].dim(), 0)

    def test_single_sample_batch(self):
        """Test edge case with batch_size=1"""
        from pyhealth.datasets import get_dataloader

        loader_1 = get_dataloader(
            self.dataset, batch_size=1, shuffle=False
        )
        batch_1 = next(iter(loader_1))

        model = self._make_model()
        model.eval()
        with torch.no_grad():
            out = model(**batch_1)
        self.assertEqual(out["y_prob"].shape, (1, 1))

    def test_loss_is_scalar(self):
        """Test that loss is a scalar tensor"""
        model = self._make_model()
        model.eval()
        with torch.no_grad():
            output = model(**self.batch)
        self.assertEqual(output["loss"].dim(), 0)

    def test_embed_output(self):
        """Test that embed key is returned when requested"""
        model = self._make_model()
        model.eval()
        batch_with_embed = dict(self.batch)
        batch_with_embed["embed"] = True
        with torch.no_grad():
            output = model(**batch_with_embed)
        self.assertIn("embed", output)


if __name__ == "__main__":
    unittest.main()

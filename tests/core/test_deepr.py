# Author: Joshua Steier
# Description: Unit tests for Deepr model (PyHealth 2.0)
import unittest

import torch

from pyhealth.datasets import  get_dataloader
from pyhealth.datasets import create_sample_dataset
from pyhealth.models import Deepr


class TestDeepr(unittest.TestCase):
    """Test cases for the Deepr model."""

    def setUp(self):
        """Set up test data and model."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86", "cond-80", "cond-12"],
                "procedures": ["proc-1", "proc-2"],
                "label": 1,
            },
            {
                "patient_id": "patient-0",
                "visit_id": "visit-1",
                "conditions": ["cond-33", "cond-86", "cond-80"],
                "procedures": ["proc-2"],
                "label": 0,
            },
        ]

        self.input_schema = {
            "conditions": "sequence",
            "procedures": "sequence",
        }
        self.output_schema = {"label": "binary"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        self.model = Deepr(dataset=self.dataset)

    def test_model_initialization(self):
        """Test that the Deepr model initializes correctly."""
        self.assertIsInstance(self.model, Deepr)
        self.assertEqual(self.model.embedding_dim, 128)
        self.assertEqual(self.model.hidden_dim, 128)
        self.assertEqual(self.model.window, 1)
        self.assertEqual(len(self.model.feature_keys), 2)
        self.assertIn("conditions", self.model.feature_keys)
        self.assertIn("procedures", self.model.feature_keys)
        self.assertEqual(self.model.label_key, "label")

    def test_model_forward(self):
        """Test that the Deepr model forward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["y_true"].shape[0], 2)
        self.assertEqual(ret["logit"].shape[0], 2)
        self.assertEqual(ret["loss"].dim(), 0)

    def test_model_backward(self):
        """Test that the Deepr model backward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters have gradients after backward pass")

    def test_model_with_embedding(self):
        """Test that the Deepr model returns embeddings when requested."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        data_batch["embed"] = True

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape[0], 2)
        expected_embed_dim = len(self.model.feature_keys) * self.model.hidden_dim
        self.assertEqual(ret["embed"].shape[1], expected_embed_dim)

    def test_custom_hyperparameters(self):
        """Test Deepr model with custom hyperparameters."""
        model = Deepr(
            dataset=self.dataset,
            embedding_dim=64,
            hidden_dim=32,
            window=2,
        )

        self.assertEqual(model.embedding_dim, 64)
        self.assertEqual(model.hidden_dim, 32)
        self.assertEqual(model.window, 2)

        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
    def test_nested_sequence_input(self):
        """Test Deepr model with nested sequence inputs (4D tensors)."""
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": [["cond-33", "cond-86"], ["cond-80", "cond-12"]],
                "procedures": [["proc-1"], ["proc-2"]],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": [["cond-33"], ["cond-86", "cond-80"]],
                "procedures": [["proc-2"]],
                "label": 0,
            },
        ]

        dataset = create_sample_dataset(
            samples=samples,
            input_schema={
                "conditions": "nested_sequence",
                "procedures": "nested_sequence",
            },
            output_schema={"label": "binary"},
            dataset_name="test_nested",
        )

        model = Deepr(dataset=dataset, embedding_dim=64, hidden_dim=64)

        train_loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertEqual(ret["y_prob"].shape[0], 2)


class TestDeeprInputShapes(unittest.TestCase):
    """Deepr on inputs other than code sequences."""

    def _train_step(self, samples, schema):
        dataset = create_sample_dataset(
            samples=samples,
            input_schema=schema,
            output_schema={"label": "binary"},
            dataset_name="test_deepr_shapes",
        )
        torch.manual_seed(0)
        model = Deepr(dataset=dataset, embedding_dim=8, hidden_dim=8)
        batch = next(iter(get_dataloader(dataset, batch_size=len(samples))))
        ret = model(**batch)
        ret["loss"].backward()
        self.assertEqual(tuple(ret["y_prob"].shape), (len(samples), 1))
        self.assertTrue(torch.isfinite(ret["loss"]))
        return ret

    def test_one_vector_per_sample_inputs(self):
        """tensor and multi_hot fields give 2-D embeddings (one step)."""
        samples = [
            {
                "patient_id": f"p{i}",
                "vitals": [float(i), 1.0, 2.0],
                "flags": ["a", "b"][: 1 + i % 2],
                "label": i % 2,
            }
            for i in range(4)
        ]
        for schema in ({"vitals": "tensor"}, {"flags": "multi_hot"}):
            with self.subTest(schema=schema):
                self._train_step(samples, schema)

    def test_continuous_sequences(self):
        """Per-step feature vectors carry a feature dimension in the raw input."""
        samples = [
            {
                "patient_id": f"p{i}",
                "labs": [[float(i), 0.0], [1.0, 2.0], [0.0, 3.0]][: 1 + i % 3],
                "label": i % 2,
            }
            for i in range(4)
        ]
        self._train_step(samples, {"labs": "nested_sequence_floats"})

    def test_deep_nested_codes(self):
        """deep_nested_sequence fields give 5-D embeddings."""
        samples = [
            {
                "patient_id": f"p{i}",
                "label": i % 2,
                "visits": [[["a", "b"], ["c"]], [["d"]], [["a"], ["e"]]][
                    : 1 + i % 3
                ],
            }
            for i in range(4)
        ]
        self._train_step(samples, {"visits": "deep_nested_sequence"})

    def test_deep_nested_last_visit_is_used(self):
        """Patients that differ only in their last visit get different outputs."""
        samples = [
            {"patient_id": "p0", "label": 0, "visits": [[["a"], ["b"]], [["c"]]]},
            {"patient_id": "p1", "label": 1, "visits": [[["a"], ["b"]], [["d"]]]},
        ]
        ret = self._train_step(samples, {"visits": "deep_nested_sequence"})
        self.assertFalse(torch.allclose(ret["y_prob"][0], ret["y_prob"][1]))


if __name__ == "__main__":
    unittest.main()

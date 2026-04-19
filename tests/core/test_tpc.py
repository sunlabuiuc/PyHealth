"""Synthetic tests for the PyHealth TPC model contribution.

Contributor: Hasham Ul Haq (huhaq2)
Paper: Temporal Pointwise Convolutional Networks for Length of Stay Prediction
    in the Intensive Care Unit
Paper link: https://arxiv.org/abs/2007.09483
Description: Fast unit tests for the TPC model using only synthetic PyHealth
    sample datasets and tiny tensors.
"""

import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import TPC
from pyhealth.models.tpc import TPCLayer


def _make_binary_dataset():
    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "diagnoses": ["A", "B", "C"],
            "icd_codes": ([0.0, 24.0], [["A", "B"], ["C"]]),
            "labs": ([0.0, 12.0], [[1.0, None, 3.0], [2.0, 5.0, None]]),
            "label": 1,
        },
        {
            "patient_id": "patient-1",
            "visit_id": "visit-0",
            "diagnoses": ["D", "E"],
            "icd_codes": ([0.0, 18.0], [["D"], ["E", "F"]]),
            "labs": ([0.0, 6.0], [[4.0, 1.0, None], [5.0, None, 9.0]]),
            "label": 0,
        },
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={
            "diagnoses": "sequence",
            "icd_codes": "stagenet",
            "labs": "stagenet_tensor",
        },
        output_schema={"label": "binary"},
        dataset_name="test_tpc_binary",
    )


def _make_multiclass_dataset():
    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "icd_codes": ([0.0, 24.0], [["A", "B"], ["C"]]),
            "labs": ([0.0, 12.0], [[1.0, 2.0], [2.0, 3.0]]),
            "los": 0,
        },
        {
            "patient_id": "patient-1",
            "visit_id": "visit-0",
            "icd_codes": ([0.0, 12.0], [["D"], ["E"]]),
            "labs": ([0.0, 6.0], [[3.0, 4.0], [5.0, 6.0]]),
            "los": 1,
        },
        {
            "patient_id": "patient-2",
            "visit_id": "visit-0",
            "icd_codes": ([0.0, 30.0], [["F"], ["G", "H"]]),
            "labs": ([0.0, 8.0], [[7.0, 8.0], [9.0, 10.0]]),
            "los": 2,
        },
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"icd_codes": "stagenet", "labs": "stagenet_tensor"},
        output_schema={"los": "multiclass"},
        dataset_name="test_tpc_multiclass",
    )


class TestTPCLayer(unittest.TestCase):
    """Unit tests for the standalone TPC encoder layer."""

    def test_output_shapes(self):
        layer = TPCLayer(feature_dim=32, hidden_dim=64, num_layers=2, kernel_size=3)
        x = torch.randn(4, 7, 32)
        outputs, pooled = layer(x)
        self.assertEqual(outputs.shape, (4, 7, 32))
        self.assertEqual(pooled.shape, (4, 32))

    def test_masked_forward(self):
        layer = TPCLayer(feature_dim=16, hidden_dim=32, num_layers=1, kernel_size=2)
        x = torch.randn(3, 5, 16)
        mask = torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]],
            dtype=torch.bool,
        )
        outputs, pooled = layer(x, mask)
        self.assertEqual(outputs.shape, (3, 5, 16))
        self.assertEqual(pooled.shape, (3, 16))
        self.assertTrue(torch.allclose(outputs[0, 3:], torch.zeros_like(outputs[0, 3:])))


class TestTPC(unittest.TestCase):
    """Unit tests for the full TPC PyHealth model."""

    @classmethod
    def setUpClass(cls):
        cls.binary_dataset = _make_binary_dataset()
        cls.multiclass_dataset = _make_multiclass_dataset()

    def test_model_initialization(self):
        model = TPC(
            dataset=self.binary_dataset,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=3,
            kernel_size=5,
            dropout=0.2,
        )
        self.assertIsInstance(model, TPC)
        self.assertEqual(model.embedding_dim, 32)
        self.assertEqual(model.hidden_dim, 64)
        self.assertEqual(model.num_layers, 3)
        self.assertEqual(model.kernel_size, 5)
        self.assertEqual(model.label_key, "label")
        self.assertEqual(set(model.feature_keys), {"diagnoses", "icd_codes", "labs"})

    def test_binary_forward(self):
        model = TPC(dataset=self.binary_dataset, embedding_dim=32, hidden_dim=64)
        batch = next(iter(get_dataloader(self.binary_dataset, batch_size=2, shuffle=False)))

        with torch.no_grad():
            ret = model(**batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)
        self.assertEqual(ret["y_prob"].shape, (2, 1))
        self.assertEqual(ret["y_true"].shape, (2, 1))
        self.assertEqual(ret["logit"].shape, (2, 1))
        self.assertEqual(ret["loss"].dim(), 0)

    def test_multiclass_forward(self):
        model = TPC(dataset=self.multiclass_dataset, embedding_dim=24, hidden_dim=48)
        batch = next(
            iter(get_dataloader(self.multiclass_dataset, batch_size=3, shuffle=False))
        )

        with torch.no_grad():
            ret = model(**batch)

        self.assertEqual(ret["y_prob"].shape, (3, 3))
        self.assertEqual(ret["logit"].shape, (3, 3))
        self.assertEqual(ret["y_true"].shape[0], 3)

    def test_backward(self):
        model = TPC(dataset=self.binary_dataset, embedding_dim=16, hidden_dim=32)
        batch = next(iter(get_dataloader(self.binary_dataset, batch_size=2, shuffle=False)))

        ret = model(**batch)
        ret["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters have gradients after backward pass")

    def test_embed_output(self):
        model = TPC(dataset=self.binary_dataset, embedding_dim=20, hidden_dim=40)
        batch = next(iter(get_dataloader(self.binary_dataset, batch_size=2, shuffle=False)))
        batch["embed"] = True

        with torch.no_grad():
            ret = model(**batch)

        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape, (2, 60))


if __name__ == "__main__":
    unittest.main()

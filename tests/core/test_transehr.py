"""Tests for the TransEHR model."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import TransEHR


def _make_sequence_event_dataset():
    """Create a tiny synthetic dataset for TransEHR tests."""
    t0 = datetime(2020, 1, 1, 0, 0)
    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "multivariate": (
                [t0, t0 + timedelta(hours=1), t0 + timedelta(hours=3)],
                np.array(
                    [
                        [1.0, 0.1, 0.0],
                        [0.8, 0.2, 0.1],
                        [0.6, 0.4, 0.0],
                    ],
                    dtype=np.float32,
                ),
            ),
            "events": ["LAB_A", "MED_X", "LAB_B"],
            "static": [65.0, 1.0],
            "label": 0,
        },
        {
            "patient_id": "patient-1",
            "visit_id": "visit-0",
            "multivariate": (
                [t0, t0 + timedelta(hours=2), t0 + timedelta(hours=4)],
                np.array(
                    [
                        [0.2, 1.2, 0.0],
                        [0.3, 1.0, 0.4],
                        [0.1, 0.9, 0.2],
                    ],
                    dtype=np.float32,
                ),
            ),
            "events": ["MED_X", "PROC_Y"],
            "static": [72.0, 0.0],
            "label": 1,
        },
    ]

    input_schema = {
        "multivariate": "temporal_timeseries",
        "events": "sequence",
        "static": "tensor",
    }
    output_schema = {"label": "multiclass"}

    return create_sample_dataset(
        samples=samples,
        input_schema=input_schema,
        output_schema=output_schema,
        dataset_name="transehr_test_sequence",
    )


def _make_model(dataset):
    return TransEHR(
        dataset=dataset,
        feature_keys={
            "multivariate": "multivariate",
            "events": "events",
            "static": "static",
        },
        label_key="label",
        mode="multiclass",
        embedding_dim=16,
        hidden_dim=16,
        num_heads=4,
        dropout=0.1,
        num_encoder_layers=1,
        max_event_len=16,
        max_ts_len=16,
    )


class TestTransEHR(unittest.TestCase):
    """TransEHR tests (unittest so ``python -m unittest discover`` finds them)."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_sequence_event_dataset()
        cls.model = _make_model(cls.dataset)
        cls.model.eval()
        cls.loader = get_dataloader(cls.dataset, batch_size=2, shuffle=False)
        cls.batch = next(iter(cls.loader))

    def test_instantiation(self):
        self.assertIsInstance(self.model, TransEHR)

    def test_forward_output_keys_and_shapes(self):
        output = self.model(**self.batch)

        self.assertIn("loss", output)
        self.assertIn("logit", output)
        self.assertIn("y_prob", output)
        self.assertIn("y_true", output)

        self.assertEqual(output["logit"].shape[0], 2)
        self.assertEqual(output["y_prob"].shape[0], 2)
        self.assertEqual(output["y_true"].shape[0], 2)
        self.assertEqual(output["loss"].ndim, 0)

    def test_backward(self):
        # Use an isolated model to avoid cross-test gradient state.
        model = _make_model(self.dataset)
        output = model(**self.batch)
        output["loss"].backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None for g in grads))

    def test_embed_flag(self):
        batch = dict(self.batch)
        batch["embed"] = True
        output = self.model(**batch)
        self.assertIn("embed", output)
        self.assertEqual(output["embed"].shape[0], 2)

    def test_forward_from_embedding_matches_forward(self):
        model = _make_model(self.dataset)
        model.eval()
        a = model(**self.batch)
        b = model.forward_from_embedding(**self.batch)
        self.assertEqual(a.keys(), b.keys())
        for k in a:
            ta, tb = a[k], b[k]
            if not torch.is_tensor(ta):
                continue
            if ta.dtype in (torch.float32, torch.float64):
                self.assertTrue(torch.allclose(ta, tb))
            else:
                self.assertTrue(torch.equal(ta, tb))

    def test_get_embedding_model_is_none(self):
        self.assertIsNone(self.model.get_embedding_model())

    def test_use_event_stream_false(self):
        model = TransEHR(
            dataset=self.dataset,
            feature_keys={
                "multivariate": "multivariate",
                "events": "events",
                "static": "static",
            },
            label_key="label",
            mode="multiclass",
            embedding_dim=16,
            hidden_dim=16,
            num_heads=4,
            num_encoder_layers=1,
            max_event_len=16,
            max_ts_len=16,
            use_event_stream=False,
        )
        output = model(**self.batch)
        self.assertIn("loss", output)
        self.assertEqual(output["logit"].shape[0], 2)
        output["loss"].backward()


if __name__ == "__main__":
    unittest.main()

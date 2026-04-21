import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from pyhealth.models.dynamic_survival_rnn import DynamicSurvivalRNN


class DummyDataset:
    def __init__(self, samples):
        self.samples = samples
        self.input_schema = {
            "x": "timeseries",
            "hazard_y": "vector",
            "hazard_mask": "vector",
        }
        self.output_schema = {
            "event_within_h": "binary",
        }

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


class TestDynamicSurvivalRNN(unittest.TestCase):
    def setUp(self):
        self.horizon = 8
        self.samples = []
        for i in range(4):
            x = np.random.randn(6, 3).astype(np.float32)
            hazard_y = np.zeros(self.horizon, dtype=np.float32)
            hazard_mask = np.ones(self.horizon, dtype=np.float32)
            label = 0.0

            if i % 2 == 0:
                hazard_y[min(i + 1, self.horizon - 1)] = 1.0
                label = 1.0

            self.samples.append(
                {
                    "x": x,
                    "hazard_y": hazard_y,
                    "hazard_mask": hazard_mask,
                    "event_within_h": label,
                }
            )

        self.dataset = DummyDataset(self.samples)

    def _collate(self, batch):
        return {
            "x": torch.tensor(np.stack([b["x"] for b in batch]), dtype=torch.float32),
            "hazard_y": torch.tensor(
                np.stack([b["hazard_y"] for b in batch]), dtype=torch.float32
            ),
            "hazard_mask": torch.tensor(
                np.stack([b["hazard_mask"] for b in batch]), dtype=torch.float32
            ),
            "event_within_h": torch.tensor(
                [[b["event_within_h"]] for b in batch], dtype=torch.float32
            ),
        }

    def test_instantiation(self):
        model = DynamicSurvivalRNN(
            dataset=self.dataset,
            feature_key="x",
            label_key="event_within_h",
            hazard_label_key="hazard_y",
            hazard_mask_key="hazard_mask",
            hidden_dim=16,
            horizon=self.horizon,
        )
        self.assertIsNotNone(model)

    def test_forward_shapes(self):
        model = DynamicSurvivalRNN(
            dataset=self.dataset,
            feature_key="x",
            label_key="event_within_h",
            hazard_label_key="hazard_y",
            hazard_mask_key="hazard_mask",
            hidden_dim=16,
            horizon=self.horizon,
        )
        loader = DataLoader(self.dataset, batch_size=2, collate_fn=self._collate)
        batch = next(iter(loader))
        ret = model(**batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)
        self.assertIn("hazard", ret)
        self.assertIn("cdf", ret)

        self.assertEqual(ret["hazard"].shape, (2, self.horizon))
        self.assertEqual(ret["cdf"].shape, (2, self.horizon))
        self.assertEqual(ret["y_prob"].shape, (2, 1))

    def test_loss_is_finite(self):
        model = DynamicSurvivalRNN(
            dataset=self.dataset,
            feature_key="x",
            label_key="event_within_h",
            hazard_label_key="hazard_y",
            hazard_mask_key="hazard_mask",
            hidden_dim=16,
            horizon=self.horizon,
        )
        loader = DataLoader(self.dataset, batch_size=2, collate_fn=self._collate)
        batch = next(iter(loader))
        ret = model(**batch)
        self.assertTrue(torch.isfinite(ret["loss"]).item())

    def test_backward(self):
        model = DynamicSurvivalRNN(
            dataset=self.dataset,
            feature_key="x",
            label_key="event_within_h",
            hazard_label_key="hazard_y",
            hazard_mask_key="hazard_mask",
            hidden_dim=16,
            horizon=self.horizon,
        )
        loader = DataLoader(self.dataset, batch_size=2, collate_fn=self._collate)
        batch = next(iter(loader))
        ret = model(**batch)
        ret["loss"].backward()

        total_grad = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad += float(p.grad.abs().sum())
        self.assertGreater(total_grad, 0.0)


if __name__ == "__main__":
    unittest.main()

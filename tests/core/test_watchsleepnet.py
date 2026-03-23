import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import WatchSleepNet


class TestWatchSleepNet(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.samples = [
            {
                "patient_id": f"patient-{index}",
                "record_id": f"record-{index}",
                "signal": rng.normal(size=(24, 8)).astype(np.float32),
                "label": index % 3,
            }
            for index in range(6)
        ]
        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={"signal": "tensor"},
            output_schema={"label": "multiclass"},
            dataset_name="watchsleepnet_test",
        )
        self.model = WatchSleepNet(
            dataset=self.dataset,
            hidden_dim=16,
            conv_channels=16,
            conv_blocks=2,
            tcn_blocks=2,
            num_attention_heads=4,
            dropout=0.1,
        )

    def test_model_initialization(self):
        self.assertIsInstance(self.model, WatchSleepNet)
        self.assertEqual(self.model.feature_key, "signal")
        self.assertEqual(self.model.input_dim, 8)
        self.assertEqual(self.model.hidden_dim, 16)
        self.assertEqual(self.model.conv_channels, 16)

    def test_forward_shapes(self):
        loader = get_dataloader(self.dataset, batch_size=3, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            output = self.model(**batch)

        self.assertIn("loss", output)
        self.assertIn("y_prob", output)
        self.assertIn("y_true", output)
        self.assertIn("logit", output)
        self.assertEqual(tuple(output["logit"].shape), (3, 3))
        self.assertEqual(tuple(output["y_prob"].shape), (3, 3))
        self.assertEqual(tuple(output["y_true"].shape), (3,))

    def test_backward_pass(self):
        loader = get_dataloader(self.dataset, batch_size=3, shuffle=False)
        batch = next(iter(loader))

        output = self.model(**batch)
        output["loss"].backward()

        has_grad = any(
            parameter.requires_grad and parameter.grad is not None
            for parameter in self.model.parameters()
        )
        self.assertTrue(has_grad)

    def test_without_attention_and_tcn(self):
        model = WatchSleepNet(
            dataset=self.dataset,
            hidden_dim=12,
            conv_channels=12,
            num_attention_heads=3,
            use_attention=False,
            use_tcn=False,
        )
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            output = model(**batch)

        self.assertEqual(tuple(output["logit"].shape), (2, 3))


if __name__ == "__main__":
    unittest.main()

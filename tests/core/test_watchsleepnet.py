"""Unit tests for WatchSleepNet model."""

import unittest

import torch

from pyhealth.models.watchsleepnet import WatchSleepNet


class TestWatchSleepNet(unittest.TestCase):

    def test_instantiation_defaults(self):
        model = WatchSleepNet()
        self.assertIsNotNone(model)

    def test_instantiation_3class(self):
        model = WatchSleepNet(num_classes=3)
        signal = torch.randn(2, 750)
        out = model(signal=signal)
        self.assertEqual(out["y_prob"].shape, (2, 3))

    def test_forward_shape(self):
        model = WatchSleepNet(num_classes=5)
        signal = torch.randn(4, 750)
        out = model(signal=signal)
        self.assertEqual(out["y_prob"].shape, (4, 5))

    def test_forward_with_label(self):
        model = WatchSleepNet(num_classes=3)
        signal = torch.randn(4, 750)
        label = torch.randint(0, 3, (4,))
        out = model(signal=signal, label=label)
        self.assertEqual(out["loss"].ndim, 0)
        out["loss"].backward()

    def test_forward_without_label(self):
        model = WatchSleepNet(num_classes=3)
        signal = torch.randn(4, 750)
        out = model(signal=signal)
        self.assertEqual(float(out["loss"]), 0.0)
        self.assertIn("y_true", out)
        self.assertIsNone(out["y_true"])

    def test_wrong_input_length(self):
        model = WatchSleepNet()
        signal = torch.randn(4, 500)
        with self.assertRaisesRegex(ValueError, "750"):
            model(signal=signal)

    def test_invalid_lstm_hidden(self):
        with self.assertRaisesRegex(ValueError, "BiLSTM"):
            WatchSleepNet(hidden_dim=256, lstm_hidden=100)

    def test_gradients_flow(self):
        model = WatchSleepNet(num_classes=3)
        model.train()
        signal = torch.randn(4, 750)
        label = torch.randint(0, 3, (4,))
        out = model(signal=signal, label=label)
        out["loss"].backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.numel() > 0:
                self.assertIsNotNone(param.grad, f"No grad for {name}")

    def test_output_dict_keys(self):
        model = WatchSleepNet()
        signal = torch.randn(2, 750)
        out = model(signal=signal)
        self.assertEqual(set(out.keys()), {"loss", "y_prob", "y_true"})

    def test_3class_5class_num_params(self):
        model3 = WatchSleepNet(num_classes=3)
        model5 = WatchSleepNet(num_classes=5)
        params3 = sum(p.numel() for p in model3.parameters())
        params5 = sum(p.numel() for p in model5.parameters())
        fc_weight_diff = (5 - 3) * model3.hidden_dim
        fc_bias_diff = 5 - 3
        self.assertEqual(params5 - params3, fc_weight_diff + fc_bias_diff)


if __name__ == "__main__":
    unittest.main()

"""Tests for the EEGNet model, on small synthetic signals."""

import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import EEGNet
from pyhealth.trainer import Trainer


def _label(i, mode):
    """A label for sample ``i`` that cycles through every class."""
    if mode == "multiclass":
        return i % 3
    if mode == "multilabel":
        return [["a"], ["b"], ["a", "c"]][i % 3]
    return i % 2


def _dataset(n_channels, n_times, mode="multiclass", n_samples=4, extra=None):
    """Build a tiny SampleDataset of random signals.

    Args:
        n_channels: Channels per signal; ``None`` gives a 1D signal.
        n_times: Time samples per signal.
        mode: Output schema of the label.
        n_samples: Number of samples.
        extra: Optional extra input field name added to every sample.

    Returns:
        The SampleDataset.
    """
    rng = np.random.default_rng(42)
    shape = (n_times,) if n_channels is None else (n_channels, n_times)
    samples = []
    for i in range(n_samples):
        sample = {
            "patient_id": f"p{i}",
            "signal": rng.standard_normal(shape).astype(np.float32),
            "label": _label(i, mode),
        }
        if extra is not None:
            sample[extra] = rng.standard_normal((2, 4, 4)).astype(np.float32)
        samples.append(sample)
    input_schema = {"signal": "tensor"}
    if extra is not None:
        input_schema[extra] = "tensor"
    return create_sample_dataset(
        samples=samples,
        input_schema=input_schema,
        output_schema={"label": mode},
        dataset_name="test_eegnet",
    )


def _batch(dataset, batch_size=4):
    return next(iter(get_dataloader(dataset, batch_size=batch_size, shuffle=False)))


class TestEEGNet(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dataset = _dataset(n_channels=4, n_times=128)
        self.model = EEGNet(dataset=self.dataset, sampling_rate=128)

    def test_paper_defaults_at_128_hz(self):
        self.assertEqual(self.model.n_temporal_filters, 8)
        self.assertEqual(self.model.depth_multiplier, 2)
        self.assertEqual(self.model.n_pointwise_filters, 16)
        self.assertEqual(self.model.kernel_length, 64)
        self.assertEqual(self.model.separable_kernel_length, 16)
        self.assertEqual((self.model.n_channels, self.model.n_times), (4, 128))

    def test_kernel_lengths_follow_sampling_rate(self):
        model = EEGNet(dataset=_dataset(4, 200), sampling_rate=200)
        self.assertEqual(model.kernel_length, 100)
        self.assertEqual(model.separable_kernel_length, 25)
        model = EEGNet(
            dataset=self.dataset, kernel_length=32, separable_kernel_length=8
        )
        self.assertEqual((model.kernel_length, model.separable_kernel_length), (32, 8))

    def test_pointwise_filters_default_to_depth_times_temporal(self):
        model = EEGNet(dataset=self.dataset, n_temporal_filters=4, depth_multiplier=3)
        self.assertEqual(model.n_pointwise_filters, 12)
        model = EEGNet(dataset=self.dataset, n_pointwise_filters=5)
        self.assertEqual(model.n_pointwise_filters, 5)
        self.assertEqual(tuple(model(**_batch(self.dataset))["logit"].shape), (4, 3))

    def test_forward_multiclass_outputs(self):
        out = self.model(**_batch(self.dataset))
        self.assertEqual(set(out), {"loss", "y_prob", "y_true", "logit"})
        self.assertEqual(out["loss"].dim(), 0)
        self.assertEqual(tuple(out["logit"].shape), (4, 3))
        self.assertEqual(tuple(out["y_prob"].shape), (4, 3))
        torch.testing.assert_close(out["y_prob"].sum(dim=1), torch.ones(4))

    def test_forward_without_label_skips_loss(self):
        batch = _batch(self.dataset)
        del batch["label"]
        self.model.eval()
        out = self.model(**batch)
        self.assertEqual(set(out), {"y_prob", "logit"})
        self.assertEqual(tuple(out["y_prob"].shape), (4, 3))

    def test_forward_binary_outputs(self):
        dataset = _dataset(4, 128, mode="binary")
        out = EEGNet(dataset=dataset)(**_batch(dataset))
        self.assertEqual(tuple(out["y_prob"].shape), (4, 1))
        self.assertTrue(bool(((out["y_prob"] >= 0) & (out["y_prob"] <= 1)).all()))

    def test_forward_multilabel_outputs(self):
        dataset = _dataset(4, 128, mode="multilabel", n_samples=6)
        out = EEGNet(dataset=dataset)(**_batch(dataset, batch_size=6))
        n_labels = dataset.output_processors["label"].size()
        self.assertEqual(n_labels, 3)
        self.assertEqual(tuple(out["y_prob"].shape), (6, n_labels))
        self.assertEqual(tuple(out["y_true"].shape), (6, n_labels))
        self.assertEqual(out["loss"].dim(), 0)
        self.assertTrue(bool(((out["y_prob"] >= 0) & (out["y_prob"] <= 1)).all()))

    def test_backward_reaches_every_parameter(self):
        out = self.model(**_batch(self.dataset))
        out["loss"].backward()
        for name, param in self.model.named_parameters():
            if name == "_dummy_param":
                continue
            self.assertIsNotNone(param.grad, name)

    def test_several_input_sizes(self):
        for n_channels, n_times in ((None, 64), (1, 96), (3, 100), (22, 500)):
            with self.subTest(n_channels=n_channels, n_times=n_times):
                dataset = _dataset(n_channels, n_times)
                model = EEGNet(dataset=dataset)
                out = model(**_batch(dataset), embed=True)
                self.assertEqual(tuple(out["logit"].shape), (4, 3))
                expected = model.n_pointwise_filters * (n_times // 32)
                self.assertEqual(tuple(out["embed"].shape), (4, expected))

    def test_signal_shorter_than_pooling_raises(self):
        with self.assertRaises(ValueError):
            EEGNet(dataset=_dataset(4, 31))

    def test_several_label_keys_raise(self):
        dataset = create_sample_dataset(
            samples=[
                {
                    "patient_id": f"p{i}",
                    "signal": np.zeros((4, 128), np.float32),
                    "label": i % 3,
                    "other": i % 2,
                }
                for i in range(4)
            ],
            input_schema={"signal": "tensor"},
            output_schema={"label": "multiclass", "other": "binary"},
            dataset_name="test_eegnet",
        )
        with self.assertRaisesRegex(ValueError, "one label key"):
            EEGNet(dataset=dataset)

    def test_several_signal_keys_raise(self):
        dataset = _dataset(4, 128, extra="second_signal")
        with self.assertRaisesRegex(ValueError, "one signal feature key"):
            EEGNet(dataset=dataset)

    def test_signal_of_three_dimensions_raises(self):
        dataset = create_sample_dataset(
            samples=[
                {
                    "patient_id": f"p{i}",
                    "signal": np.zeros((2, 4, 128), np.float32),
                    "label": i % 3,
                }
                for i in range(4)
            ],
            input_schema={"signal": "tensor"},
            output_schema={"label": "multiclass"},
            dataset_name="test_eegnet",
        )
        with self.assertRaisesRegex(ValueError, r"expects a \(channels, time\)"):
            EEGNet(dataset=dataset)

    def test_stft_feature_is_ignored(self):
        dataset = _dataset(4, 128, extra="stft")
        model = EEGNet(dataset=dataset)
        self.assertEqual(model.feature_keys, ["signal"])
        self.assertEqual(tuple(model(**_batch(dataset))["logit"].shape), (4, 3))

    def test_parameter_count_matches_paper_formula(self):
        # Table 2 of the paper, with C channels, N classes and T samples.
        f1, d, f2, c, n, t = 8, 2, 16, 4, 3, 128
        expected = (
            64 * f1  # temporal convolution, no bias
            + 2 * f1  # batch norm
            + c * d * f1  # depthwise convolution, no bias
            + 2 * d * f1  # batch norm
            + 16 * d * f1
            + d * f1 * f2  # separable convolution, no bias
            + 2 * f2  # batch norm
            + (f2 * (t // 32)) * n
            + n  # classifier with bias
        )
        n_params = sum(
            p.numel()
            for name, p in self.model.named_parameters()
            if name != "_dummy_param"
        )
        self.assertEqual(n_params, expected)

    def _depthwise_and_classifier_norms(self):
        depthwise_conv = next(
            m for m in self.model.modules() if type(m).__name__ == "Conv2dWithMaxNorm"
        )
        depthwise = depthwise_conv.weight.flatten(1).norm(dim=1)
        classifier = self.model.fc.weight.norm(dim=1)
        return depthwise, classifier

    def _push_weights_past_max_norm(self, batch):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=10.0)
        for _ in range(3):
            optimizer.zero_grad()
            self.model(**batch)["loss"].backward()
            optimizer.step()
        depthwise, classifier = self._depthwise_and_classifier_norms()
        # The last optimizer step is not followed by a projection, so the test
        # below only means something if the norms are exceeded here.
        self.assertTrue(bool((depthwise > 1.0).any() or (classifier > 0.25).any()))

    def _assert_within_max_norm(self):
        depthwise, classifier = self._depthwise_and_classifier_norms()
        self.assertTrue(bool((depthwise <= 1.0 + 1e-5).all()))
        self.assertTrue(bool((classifier <= 0.25 + 1e-5).all()))

    def test_max_norm_holds_for_eval_forward_after_training_steps(self):
        batch = _batch(self.dataset)
        self._push_weights_past_max_norm(batch)
        self.model.eval()
        with torch.no_grad():
            logit = self.model(**batch)["logit"]
        # The weights used for that prediction, and any state dict saved after
        # it, are within the norm.
        self._assert_within_max_norm()
        clone = EEGNet(dataset=self.dataset)
        clone.load_state_dict(self.model.state_dict())
        clone.eval()
        with torch.no_grad():
            torch.testing.assert_close(clone(**batch)["logit"], logit)

    def test_max_norm_holds_under_inference_mode(self):
        batch = _batch(self.dataset)
        self._push_weights_past_max_norm(batch)
        self.model.eval()
        with torch.inference_mode():
            self.model(**batch)
        self._assert_within_max_norm()

    def test_two_forward_passes_before_one_backward(self):
        # The projection writes the weights in place; a second forward pass
        # before backward must not modify a weight saved by the first one.
        batch = _batch(self.dataset)
        self._push_weights_past_max_norm(batch)
        self.model.train()
        loss = self.model(**batch)["loss"] + self.model(**batch)["loss"]
        loss.backward()
        self._assert_within_max_norm()

    def test_trainer_runs_one_epoch(self):
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        trainer = Trainer(model=self.model, device="cpu", enable_logging=False)
        trainer.train(train_dataloader=loader, val_dataloader=loader, epochs=1)
        scores = trainer.evaluate(loader)
        self.assertTrue(np.isfinite(scores["loss"]))
        self._assert_within_max_norm()

    def test_eval_mode_is_deterministic(self):
        self.model.eval()
        batch = _batch(self.dataset)
        with torch.no_grad():
            first = self.model(**batch)["logit"]
            second = self.model(**batch)["logit"]
        torch.testing.assert_close(first, second)

    def test_state_dict_round_trip(self):
        clone = EEGNet(dataset=self.dataset)
        clone.load_state_dict(self.model.state_dict())
        self.model.eval()
        clone.eval()
        batch = _batch(self.dataset)
        with torch.no_grad():
            torch.testing.assert_close(
                self.model(**batch)["logit"], clone(**batch)["logit"]
            )


if __name__ == "__main__":
    unittest.main()

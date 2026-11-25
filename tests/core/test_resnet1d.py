"""Unit tests for the ResNet1D ECG classifier.

Synthetic samples only; small signal lengths keep every test in the
millisecond range. Covers binary/multiclass/multilabel modes, forward and
loss shapes, gradient flow, length-agnostic pooling, the 18/34/50 depth
layouts, head configuration, and argument errors.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import BasicBlock1D, Bottleneck1D, ResNet1D


def _signal_dataset(output_schema: dict, n: int = 6, t: int = 400,
                    labels_fn=None):
    """SampleDataset with (12, t) float32 tensor signals."""

    def labels_fn_default(i):
        return i % 3

    if labels_fn is None:
        labels_fn = labels_fn_default
    samples = [
        {
            "patient_id": f"patient-{i}",
            "signal": np.random.RandomState(i).randn(12, t).astype(
                np.float32
            ),
            **{key: labels_fn(i) for key in output_schema},
        }
        for i in range(n)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema=output_schema,
        dataset_name="test_resnet1d",
    )


def _batch(dataset, label_key: str):
    loader = get_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    x = batch["signal"].float()
    y = batch[label_key]
    return x, y


class TestResNet1DModes(unittest.TestCase):
    """mode / output size follow the dataset's label processor."""

    def test_binary_mode(self):
        dataset = _signal_dataset(
            {"label": "binary"}, labels_fn=lambda i: i % 2
        )
        model = ResNet1D(dataset=dataset, depth=18)
        self.assertEqual(model.mode, "binary")
        x, y = _batch(dataset, "label")
        out = model(signal=x, label=y)
        self.assertEqual(tuple(out["logit"].shape), (2, 1))
        self.assertEqual(tuple(out["y_prob"].shape), (2, 1))
        self.assertEqual(tuple(out["y_true"].shape), (2, 1))

    def test_multiclass_fixed_vocab_mode(self):
        schema = {
            "label": (
                "multiclass",
                {"labels": ["normal", "target", "other"]},
            )
        }
        dataset = _signal_dataset(
            schema, labels_fn=lambda i: ["normal", "target", "other"][i % 3]
        )
        model = ResNet1D(dataset=dataset)
        self.assertEqual(model.mode, "multiclass")
        x, y = _batch(dataset, "label")
        out = model(signal=x, label=y)
        self.assertEqual(tuple(out["logit"].shape), (2, 3))

    def test_multilabel_mode(self):
        dataset = _signal_dataset(
            {"labels": "multilabel"},
            labels_fn=lambda i: [["x"], ["y"], ["x", "y"]][i % 3],
        )
        model = ResNet1D(dataset=dataset)
        self.assertEqual(model.mode, "multilabel")
        x, y = _batch(dataset, "labels")
        out = model(signal=x, labels=y)
        self.assertEqual(tuple(out["logit"].shape), (2, 2))

    def test_forward_without_label_omits_loss(self):
        dataset = _signal_dataset({"label": "binary"}, labels_fn=lambda i: i % 2)
        model = ResNet1D(dataset=dataset)
        x, _ = _batch(dataset, "label")
        out = model(signal=x)
        self.assertEqual(sorted(out), ["logit", "y_prob"])

    def test_embed_flag_returns_backbone_features(self):
        dataset = _signal_dataset({"label": "binary"}, labels_fn=lambda i: i % 2)
        model = ResNet1D(dataset=dataset, hidden_dim=64)
        x, y = _batch(dataset, "label")
        out = model(signal=x, label=y, embed=True)
        self.assertEqual(tuple(out["embed"].shape), (2, 64))


class TestResNet1DTrainingStep(unittest.TestCase):
    """Loss + backward populate gradients across stem, stages and head."""

    def test_loss_backward_grad_flow(self):
        dataset = _signal_dataset({"label": "binary"}, labels_fn=lambda i: i % 2)
        model = ResNet1D(dataset=dataset, depth=18)
        x, y = _batch(dataset, "label")
        out = model(signal=x, label=y)
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()
        for name in ("conv1.weight", "layer2.0.conv1.weight",
                     "projection.weight", "head.0.weight", "head.4.weight"):
            grad = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(grad, msg=name)
            self.assertGreater(float(grad.abs().sum()), 0.0, msg=name)


class TestResNet1DShapeFlexibility(unittest.TestCase):
    def test_variable_signal_lengths(self):
        dataset = _signal_dataset(
            {"label": "binary"}, n=4, t=5000, labels_fn=lambda i: i % 2
        )
        model = ResNet1D(dataset=dataset, depth=18)
        model.eval()
        with torch.no_grad():
            for t in (625, 1000, 5000):
                x = torch.randn(1, 12, t)
                out = model(signal=x)
                self.assertEqual(tuple(out["logit"].shape), (1, 1))

    def test_in_channels_and_class_count_configurable(self):
        dataset = _signal_dataset(
            {"labels": "multilabel"},
            labels_fn=lambda i: [["a"], ["b"], ["a", "b"]][i % 3],
        )
        # 8-lead fixture with 2 classes; exercises the stem only.
        model = ResNet1D(dataset=dataset, in_channels=8)
        self.assertEqual(model.conv1.in_channels, 8)
        out = model(signal=torch.randn(2, 8, 400))
        self.assertEqual(tuple(out["logit"].shape), (2, 2))


class TestResNet1DDepths(unittest.TestCase):
    def _assert_layer_layout(self, model, blocks, bottleneck):
        block_cls = Bottleneck1D if bottleneck else BasicBlock1D
        for stage_name, n_blocks in zip(
            ("layer1", "layer2", "layer3", "layer4"), blocks
        ):
            stage = getattr(model, stage_name)
            self.assertEqual(len(stage), n_blocks, msg=stage_name)
            for block in stage:
                self.assertIsInstance(block, block_cls, msg=stage_name)
            if bottleneck:
                self.assertEqual(stage[0].expansion, 4, msg=stage_name)

    def test_depth_34_uses_basic_blocks(self):
        dataset = _signal_dataset({"label": "binary"}, labels_fn=lambda i: i % 2)
        model = ResNet1D(dataset=dataset, depth=34)
        self._assert_layer_layout(model, [3, 4, 6, 3], bottleneck=False)
        self.assertEqual(model.projection.in_features, 512)

    def test_depth_50_uses_bottleneck_blocks(self):
        dataset = _signal_dataset({"label": "binary"}, labels_fn=lambda i: i % 2)
        model = ResNet1D(dataset=dataset, depth=50)
        self._assert_layer_layout(model, [3, 4, 6, 3], bottleneck=True)
        self.assertEqual(model.projection.in_features, 2048)

    def test_depth_18_34_50_forward_smoke(self):
        dataset = _signal_dataset({"label": "binary"}, labels_fn=lambda i: i % 2)
        x, _ = _batch(dataset, "label")
        for depth in (18, 34, 50):
            with self.subTest(depth=depth):
                model = ResNet1D(dataset=dataset, depth=depth)
                model.eval()
                with torch.no_grad():
                    out = model(signal=x[:1])
                self.assertEqual(tuple(out["logit"].shape), (1, 1))


class TestResNet1DConfig(unittest.TestCase):
    def test_head_config_passthrough(self):
        dataset = _signal_dataset(
            {"labels": "multilabel"},
            labels_fn=lambda i: [["a"], ["b"]][i % 2],
        )
        model = ResNet1D(
            dataset=dataset,
            hidden_dim=32,
            head_hidden_dim=8,
            head_dropout=0.0,
        )
        self.assertEqual(model.projection.out_features, 32)
        self.assertEqual(model.head[0].out_features, 8)
        self.assertEqual(model.head[3].p, 0.0)
        self.assertEqual(model.head[4].out_features, 2)


class TestResNet1DCropWindow(unittest.TestCase):
    """Benchmark protocol: random training crop + sliding-window eval."""

    def _binary_model(self, **kwargs):
        dataset = _signal_dataset(
            {"label": "binary"}, n=4, labels_fn=lambda i: i % 2
        )
        return ResNet1D(dataset=dataset, head_dropout=0.0, **kwargs)

    def test_training_crop_shapes_offsets_and_seed(self):
        model = self._binary_model(training_crop_len=100)
        model.train()
        x = torch.randn(2, 12, 400)
        out = model(signal=x)
        self.assertEqual(tuple(out["logit"].shape), (2, 1))
        torch.manual_seed(7)
        crop_a = model._random_crop(x, 100)
        torch.manual_seed(7)
        crop_b = model._random_crop(x, 100)
        self.assertTrue(torch.equal(crop_a, crop_b))
        self.assertEqual(tuple(crop_a.shape), (2, 12, 100))
        torch.manual_seed(11)
        self.assertFalse(torch.equal(crop_a, model._random_crop(x, 100)))
        for sample in range(2):
            offsets = [
                s for s in range(400 - 100 + 1)
                if torch.equal(crop_a[sample], x[sample, :, s:s + 100])
            ]
            self.assertTrue(offsets, msg=f"sample {sample} is not a crop")

    def test_crop_skipped_when_signal_shorter(self):
        model = self._binary_model(training_crop_len=1000)
        model.train()
        x = torch.randn(2, 12, 400)
        torch.manual_seed(3)
        out_crop = model(signal=x)["logit"]
        torch.manual_seed(3)
        model.training_crop_len = None
        out_plain = model(signal=x)["logit"]
        torch.testing.assert_close(out_crop, out_plain)

    def test_eval_window_matches_manual_unfold(self):
        model = self._binary_model(
            eval_window_len=100, eval_window_stride=50
        )
        model.eval()
        x = torch.randn(2, 12, 400)
        with torch.no_grad():
            out = model(signal=x)
            # Mirror SubsampleEval: windows start at arange(0, T, stride);
            # the tail is zero-padded so the last window fits.
            n_windows = (400 - 1) // 50 + 1
            self.assertEqual(n_windows, 8)
            padded = F.pad(x, (0, (n_windows - 1) * 50 + 100 - 400))
            windows = padded.unfold(2, 100, 50)
            windows = windows.permute(0, 2, 1, 3).reshape(-1, 12, 100)
            logits = model.head(model.backbone(windows))
            manual = logits.view(2, n_windows, -1).max(dim=1).values
        torch.testing.assert_close(out["logit"], manual)

    def test_eval_window_default_stride_is_half_window(self):
        model = self._binary_model(eval_window_len=100)
        self.assertIsNone(model.eval_window_stride)
        model.eval()
        x = torch.randn(1, 12, 300)
        with torch.no_grad():
            out = model(signal=x)["logit"]
            n_windows = (300 - 1) // 50 + 1
            padded = F.pad(x, (0, (n_windows - 1) * 50 + 100 - 300))
            windows = padded.unfold(2, 100, 50)
            windows = windows.permute(0, 2, 1, 3).reshape(-1, 12, 100)
            manual = model.head(model.backbone(windows)).view(
                1, n_windows, -1
            ).max(dim=1).values
        torch.testing.assert_close(out, manual)

    def test_window_skipped_when_signal_shorter(self):
        model = self._binary_model(eval_window_len=1000)
        model.eval()
        x = torch.randn(2, 12, 400)
        with torch.no_grad():
            out = model(signal=x)["logit"]
            model.eval_window_len = None
            plain = model(signal=x)["logit"]
        torch.testing.assert_close(out, plain)

    def test_embed_returns_mean_window_features(self):
        model = self._binary_model(
            hidden_dim=32, eval_window_len=100, eval_window_stride=100
        )
        model.eval()
        x = torch.randn(2, 12, 400)
        with torch.no_grad():
            out = model(signal=x, embed=True)
        self.assertEqual(tuple(out["embed"].shape), (2, 32))

    def test_invalid_crop_or_window_raises(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self._binary_model(training_crop_len=0)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self._binary_model(eval_window_len=-5)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            self._binary_model(eval_window_len=100, eval_window_stride=150)


class TestResNet1DClassWeight(unittest.TestCase):
    def _binary_model(self, **kwargs):
        dataset = _signal_dataset(
            {"label": "binary"}, n=4, labels_fn=lambda i: i % 2
        )
        return ResNet1D(dataset=dataset, head_dropout=0.0, **kwargs)

    def test_class_weight_rejected_for_non_multiclass(self):
        with self.assertRaisesRegex(ValueError, "label mode is 'binary'"):
            self._binary_model(class_weight=[1.0, 2.0])

    def test_class_weight_length_must_match_classes(self):
        schema = {
            "label": (
                "multiclass",
                {"labels": ["normal", "target", "other"]},
            )
        }
        dataset = _signal_dataset(
            schema, n=6, labels_fn=lambda i: ["normal", "target", "other"][i % 3]
        )
        with self.assertRaisesRegex(ValueError, "one entry per class"):
            ResNet1D(dataset=dataset, class_weight=[1.0, 2.0])

    def test_weighted_loss_differs_from_unweighted(self):
        schema = {
            "label": (
                "multiclass",
                {"labels": ["normal", "target", "other"]},
            )
        }
        dataset = _signal_dataset(
            schema, n=6, labels_fn=lambda i: ["normal", "target", "other"][i % 3]
        )
        torch.manual_seed(0)
        model_w = ResNet1D(dataset=dataset, class_weight=[1.0, 0.5, 1.5])
        torch.manual_seed(0)
        model_0 = ResNet1D(dataset=dataset)
        self.assertEqual(model_w.class_weight.tolist(), [1.0, 0.5, 1.5])
        x, y = _batch(dataset, "label")
        model_w.train()
        model_0.train()
        loss_w = model_w(signal=x, label=y)["loss"]
        loss_0 = model_0(signal=x, label=y)["loss"]
        self.assertTrue(torch.isfinite(loss_w))
        self.assertFalse(torch.allclose(loss_w, loss_0))
        loss_w.backward()
        self.assertIsNotNone(model_w.projection.weight.grad)


class TestResNet1DErrors(unittest.TestCase):
    def setUp(self):
        self.dataset = _signal_dataset(
            {"label": "binary"}, labels_fn=lambda i: i % 2
        )

    def test_invalid_depth_raises(self):
        with self.assertRaisesRegex(ValueError, "depth must be one of"):
            ResNet1D(dataset=self.dataset, depth=101)

    def test_missing_signal_feature_raises(self):
        dataset = create_sample_dataset(
            samples=[
                {"patient_id": "p0", "conditions": ["a"], "label": 0},
                {"patient_id": "p1", "conditions": ["b"], "label": 1},
            ],
            input_schema={"conditions": "sequence"},
            output_schema={"label": "binary"},
            dataset_name="test_resnet1d_no_signal",
        )
        with self.assertRaisesRegex(ValueError, "expects a 'signal'"):
            ResNet1D(dataset=dataset)

    def test_forward_missing_signal_key_raises(self):
        model = ResNet1D(dataset=self.dataset)
        with self.assertRaisesRegex(KeyError, "signal"):
            model(not_signal=torch.randn(2, 12, 100))


if __name__ == "__main__":
    unittest.main()

"""Tests for SequenceTransformerNetwork (Oh et al., MLHC 2018)

Contributors:
    Matthew Nguyen (mnguyen358, mnguyen358@gatech.edu)
    Alireza Ghiam (aghiam3, aghiam3@gatech.edu)

Paper:
    Jeeheh Oh, Jiaxuan Wang and Jenna Wiens. "Learning to Exploit Invariances
    in Clinical Time-Series Data using Sequence Transformer Networks."
    Machine Learning for Healthcare (MLHC) 2018, PMLR 85.
    https://proceedings.mlr.press/v85/oh18a.html

All data are tiny synthetic tensors: 4 samples, 3-4 channels, 24 time steps
"""

import unittest
from typing import Any

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import (
    SequenceTransformerLayer,
    SequenceTransformerNetwork,
    temporal_sample,
)
from pyhealth.models.sequence_transformer import _leaky_clamp

STEPS = 24


def _tensor_dataset(steps: int = STEPS, label: str = "binary"):
    """4 samples with a 2-value static vector and a label"""
    rng = np.random.default_rng(0)
    samples = [
        {
            "patient_id": f"p{i}",
            "vitals": rng.normal(size=(steps, 3)).astype(np.float32),
            "demographics": rng.normal(size=2).astype(np.float32),
            "label": i % (2 if label == "binary" else 3),
        }
        for i in range(4)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"vitals": "tensor", "demographics": "tensor"},
        output_schema={"label": label},
    )


def _jagged_dataset(lengths=None):
    """Hourly series of different lengths with missing values, plus statics"""
    series_levels = [
        {
            "ordering": "series",
            "values": [2],
            "impute": [{"method": "forward_fill"}, {"method": "zero"}],
            "missing_indicator": True,
        }
    ]
    static_levels = [{"ordering": "single", "values": [1]}]
    if lengths is None:
        lengths = [5, 30, 1, 24]
    samples = []
    for i, length in enumerate(lengths):
        hours = [float(t) for t in range(length)]
        values = [[hour, float(i)] for hour in hours]
        if values:
            values[0][1] = float("nan")
        samples.append(
            {
                "patient_id": f"p{i}",
                "dynamic": {"time": hours, "values": values},
                "static": {"values": [[60.0 + i]]},
                "label": i % 2,
            }
        )
    return create_sample_dataset(
        samples=samples,
        input_schema={
            "dynamic": ("jagged", {"levels": series_levels}),
            "static": ("jagged", {"levels": static_levels}),
        },
        output_schema={"label": "binary"},
    )


def _batch(dataset) -> dict[str, Any]:
    return next(iter(get_dataloader(dataset, batch_size=len(dataset))))


class TestHelpers(unittest.TestCase):
    def test_temporal_sample_identity(self):
        x = torch.randn(2, 3, STEPS)
        theta = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        torch.testing.assert_close(temporal_sample(x, theta), x)

    def test_temporal_sample_shifts_and_pads_with_edge_value(self):
        x = torch.arange(5.0).view(1, 1, 5)
        # moves the read position one step later
        shifted = temporal_sample(x, torch.tensor([[1.0, 0.5]]))
        torch.testing.assert_close(shifted, torch.tensor([[[1.0, 2.0, 3.0, 4.0, 4.0]]]))

    def test_temporal_sample_interpolates_linearly(self):
        x = torch.tensor([[[0.0, 10.0, 20.0]]])
        # Half a step later
        out = temporal_sample(x, torch.tensor([[1.0, 0.5]]))
        torch.testing.assert_close(out, torch.tensor([[[5.0, 15.0, 20.0]]]))

    def test_temporal_sample_compresses(self):
        x = torch.arange(5.0).view(1, 1, 5)
        out = temporal_sample(x, torch.tensor([[2.0, 0.0]]))
        torch.testing.assert_close(out, torch.tensor([[[0.0, 0.0, 2.0, 4.0, 4.0]]]))

    def test_leaky_clamp(self):
        x = torch.tensor([-3.0, -1.0, 0.5, 2.0, 4.0])
        expected = torch.tensor([-2.01, -1.0, 0.5, 2.0, 2.02])
        torch.testing.assert_close(_leaky_clamp(x, 2.0), expected)
        x.requires_grad_(True)
        _leaky_clamp(x, 2.0).sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([0.01, 1.0, 1.0, 1.0, 0.01]))


class TestSequenceTransformerLayer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_starts_as_identity(self):
        layer = SequenceTransformerLayer(in_channels=3, seq_len=STEPS)
        x = torch.randn(2, 3, STEPS)
        out, theta, phi = layer(x)
        torch.testing.assert_close(out, x)
        torch.testing.assert_close(theta, torch.tensor([[1.0, 0.0]] * 2))
        torch.testing.assert_close(phi, torch.tensor([[1.0, 0.0]] * 2))

    def test_output_shapes(self):
        layer = SequenceTransformerLayer(in_channels=4, seq_len=30)
        out, theta, phi = layer(torch.randn(3, 4, 30))
        self.assertEqual(out.shape, (3, 4, 30))
        self.assertEqual(theta.shape, (3, 2))
        self.assertEqual(phi.shape, (3, 2))

    def test_disabled_transformations_stay_identity(self):
        x = torch.randn(2, 3, STEPS)
        for temporal, magnitude in ((True, False), (False, True)):
            with self.subTest(temporal=temporal, magnitude=magnitude):
                layer = SequenceTransformerLayer(3, STEPS, temporal, magnitude)
                with torch.no_grad():
                    layer.head[-1].bias.copy_(torch.tensor([1.5, 0.2, 0.5, 0.3]))
                _, theta, phi = layer(x)
                identity = torch.tensor([[1.0, 0.0]] * 2)
                if temporal:
                    torch.testing.assert_close(phi, identity)
                    torch.testing.assert_close(theta[0], torch.tensor([1.5, 0.2]))
                else:
                    torch.testing.assert_close(theta, identity)
                    torch.testing.assert_close(phi[0], torch.tensor([0.5, 0.3]))

    def test_magnitude_transformation_is_applied(self):
        layer = SequenceTransformerLayer(3, STEPS, temporal=False, magnitude=True)
        with torch.no_grad():
            layer.head[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.5, 0.3]))
        x = torch.randn(2, 3, STEPS)
        torch.testing.assert_close(layer(x)[0], 0.5 * x + 0.3)

    def test_rejects_short_sequences(self):
        with self.assertRaisesRegex(ValueError, "at least 18"):
            SequenceTransformerLayer(in_channels=3, seq_len=10)

    def test_rejects_a_bound_excluding_the_identity(self):
        with self.assertRaisesRegex(ValueError, "exceed 1"):
            SequenceTransformerLayer(in_channels=3, seq_len=STEPS, bound=0.5)


class TestSequenceTransformerNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = _tensor_dataset()
        cls.batch = _batch(cls.dataset)

    def setUp(self):
        torch.manual_seed(0)

    def test_initialization(self):
        model = SequenceTransformerNetwork(self.dataset, seq_len=STEPS)
        self.assertEqual(model.in_channels, 5)  # 5 = 3 series + 2 static channels
        self.assertEqual(model.label_key, "label")
        self.assertIsNotNone(model.transformer)
        baseline = SequenceTransformerNetwork(
            self.dataset, seq_len=STEPS, temporal=False, magnitude=False
        )
        self.assertIsNone(baseline.transformer)

    def test_custom_configuration(self):
        model = SequenceTransformerNetwork(
            self.dataset,
            seq_len=STEPS,
            conv_channels=(4, 6, 8),
            kernel_size=3,
            hidden_dim=7,
            dropout=0.5,
        )
        convs = [m for m in model.cnn.features if isinstance(m, torch.nn.Conv1d)]
        self.assertEqual([c.out_channels for c in convs], [4, 6, 8])
        self.assertEqual(model.cnn.head[0].out_features, 7)
        self.assertEqual(model(**self.batch)["logit"].shape, (4, 1))

    def test_forward_outputs(self):
        model = SequenceTransformerNetwork(self.dataset, seq_len=STEPS)
        out = model(**self.batch)
        self.assertEqual(set(out), {"logit", "y_prob", "loss", "y_true"})
        self.assertEqual(out["logit"].shape, (4, 1))
        self.assertEqual(out["y_prob"].shape, (4, 1))
        self.assertEqual(out["loss"].dim(), 0)
        self.assertTrue(((out["y_prob"] >= 0) & (out["y_prob"] <= 1)).all())

    def test_forward_without_labels(self):
        model = SequenceTransformerNetwork(self.dataset, seq_len=STEPS)
        batch = {k: v for k, v in self.batch.items() if k != "label"}
        self.assertEqual(set(model(**batch)), {"logit", "y_prob"})

    def test_all_parameters_get_gradients(self):
        for temporal, magnitude in ((True, True), (True, False), (False, True)):
            with self.subTest(temporal=temporal, magnitude=magnitude):
                model = SequenceTransformerNetwork(
                    self.dataset, STEPS, temporal, magnitude, dropout=0.0
                )
                with torch.no_grad():
                    model.transformer.head[-1].bias.add_(0.1)
                model(**self.batch)["loss"].backward()
                for name, param in model.named_parameters():
                    if name == "_dummy_param":
                        continue
                    self.assertIsNotNone(param.grad, name)
                head = model.transformer.head[-1]
                self.assertGreater(head.weight.grad.abs().sum().item(), 0.0)

    def test_transform_starts_as_identity(self):
        model = SequenceTransformerNetwork(self.dataset, seq_len=STEPS)
        out = model.transform(**self.batch)
        torch.testing.assert_close(out["x_transformed"], out["x"])
        self.assertEqual(out["x"].shape, (4, 5, STEPS))
        torch.testing.assert_close(out["theta"], torch.tensor([[1.0, 0.0]] * 4))

    def test_static_features_repeat_over_time(self):
        model = SequenceTransformerNetwork(self.dataset, seq_len=STEPS)
        x = model.series(**self.batch)
        static = x[:, 3:]
        torch.testing.assert_close(static, static[:, :, :1].expand_as(static))
        torch.testing.assert_close(x[:, :3], self.batch["vitals"].transpose(1, 2))

    def test_crops_long_and_pads_short_series(self):
        model = SequenceTransformerNetwork(self.dataset, seq_len=20)
        x = model.series(**self.batch)  # 24 steps cut to the first 20
        first_20 = self.batch["vitals"][:, :20].transpose(1, 2)
        torch.testing.assert_close(x[:, :3], first_20)
        model = SequenceTransformerNetwork(self.dataset, seq_len=30)
        x = model.series(**self.batch)  # the last one repeated
        last = self.batch["vitals"][:, -1].unsqueeze(-1).expand(-1, -1, 6)
        torch.testing.assert_close(x[:, :3, 24:], last)

    def test_all_empty_series_are_zero(self):
        model = SequenceTransformerNetwork(self.dataset, seq_len=STEPS)
        empty = torch.empty(4, 0, 3, dtype=torch.float64)
        fitted = model._fit_length(empty, torch.zeros(4, dtype=torch.long))
        self.assertEqual(fitted.shape, (4, 3, STEPS))
        self.assertEqual(fitted.dtype, empty.dtype)
        torch.testing.assert_close(fitted, empty.new_zeros(4, 3, STEPS))

        batch = dict(self.batch, vitals=empty.float())
        out = model(**batch)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertTrue(torch.isfinite(out["y_prob"]).all())

    def test_multiclass_label(self):
        dataset = _tensor_dataset(label="multiclass")
        model = SequenceTransformerNetwork(dataset, seq_len=STEPS)
        out = model(**_batch(dataset))
        self.assertEqual(out["logit"].shape, (4, 3))
        torch.testing.assert_close(out["y_prob"].sum(1), torch.ones(4))

    def test_state_dict_round_trip(self):
        model = SequenceTransformerNetwork(self.dataset, seq_len=STEPS)
        model.eval()
        copy = SequenceTransformerNetwork(self.dataset, seq_len=STEPS)
        copy.load_state_dict(model.state_dict())
        copy.eval()
        torch.testing.assert_close(
            copy(**self.batch)["logit"], model(**self.batch)["logit"]
        )

    def test_rejects_too_short_seq_len(self):
        with self.assertRaisesRegex(ValueError, "at least 18"):
            SequenceTransformerNetwork(self.dataset, seq_len=10)
        with self.assertRaisesRegex(ValueError, "too short"):
            SequenceTransformerNetwork(
                self.dataset, seq_len=10, temporal=False, magnitude=False
            )


class TestJaggedInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = _jagged_dataset()
        cls.batch = _batch(cls.dataset)

    def setUp(self):
        torch.manual_seed(0)

    def test_series_layout(self):
        model = SequenceTransformerNetwork(self.dataset, seq_len=STEPS)
        # 5 = 2 values + 2 missing-value indicators + 1 static value
        self.assertEqual(model.in_channels, 5)
        x = model.series(**self.batch)
        self.assertEqual(x.shape, (4, 5, STEPS))
        torch.testing.assert_close(x[0, 0, 4:], torch.full((STEPS - 4,), 4.0))
        torch.testing.assert_close(x[1, 0], torch.arange(float(STEPS)))
        torch.testing.assert_close(x[:, 3, 0], torch.ones(4))
        torch.testing.assert_close(x[[0, 1, 3], 3, 1:3].sum(), torch.tensor(0.0))
        torch.testing.assert_close(x[2, 4], torch.full((STEPS,), 62.0))

    def test_forward_and_backward(self):
        model = SequenceTransformerNetwork(self.dataset, seq_len=STEPS)
        out = model(**self.batch)
        self.assertEqual(out["y_prob"].shape, (4, 1))
        out["loss"].backward()
        self.assertIsNotNone(model.transformer.head[-1].weight.grad)

    def test_empty_first_sample_can_initialize_and_run(self):
        dataset = _jagged_dataset(lengths=[0, 3, 0, 2])
        model = SequenceTransformerNetwork(dataset, seq_len=STEPS)
        batch = _batch(dataset)
        x = model.series(**batch)
        self.assertEqual(model.in_channels, 5)
        torch.testing.assert_close(x[[0, 2], :4], torch.zeros(2, 4, STEPS))
        torch.testing.assert_close(x[0, 4], torch.full((STEPS,), 60.0))
        out = model(**batch)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertTrue(torch.isfinite(out["y_prob"]).all())


if __name__ == "__main__":
    unittest.main()

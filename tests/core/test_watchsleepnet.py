import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import WatchSleepNet
from pyhealth.tasks import SleepStagingDREAMTSequence


class TestWatchSleepNet(unittest.TestCase):
    """Tests for PyHealth's WatchSleepNet implementation."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.samples = [
            {
                "patient_id": "patient-0",
                "signal": rng.normal(size=(3, 750)).astype(np.float32),
                "label": {
                    "time": [0.0, 30.0, 60.0],
                    "values": [0, 1, 2],
                },
            },
            {
                "patient_id": "patient-1",
                "signal": rng.normal(size=(2, 750)).astype(np.float32),
                "label": {
                    "time": [0.0, 30.0],
                    "values": [1, 2],
                },
            },
        ]

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=SleepStagingDREAMTSequence.input_schema,
            output_schema=SleepStagingDREAMTSequence.output_schema,
            dataset_name="test_watchsleepnet",
        )

        self.batch = next(
            iter(get_dataloader(self.dataset, batch_size=2, shuffle=False))
        )

    def _make_model(self, **kwargs):
        return WatchSleepNet(
            dataset=self.dataset,
            num_channels=8,
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
            tcn_layers=1,
            **kwargs,
        )

    def test_model_instantiation(self):
        """The model initializes with the expected architecture."""
        model = self._make_model()

        self.assertIsInstance(model, WatchSleepNet)
        self.assertEqual(model.segment_length, 750)
        self.assertTrue(model.use_tcn)
        self.assertTrue(model.use_attention)
        self.assertEqual(model.classifier.out_features, 3)

    def test_variable_length_forward(self):
        """The model predicts once for each real epoch, excluding padding."""
        model = self._make_model()
        result = model(**self.batch)

        self.assertEqual(tuple(self.batch["signal"].shape), (2, 3, 750))
        self.assertEqual(self.batch["label"][2].tolist(), [0, 3, 5])
        self.assertEqual(tuple(result["logit"].shape), (5, 3))
        self.assertEqual(tuple(result["y_prob"].shape), (5, 3))
        self.assertTrue(torch.isfinite(result["loss"]).all())

    def test_backward(self):
        """The loss supports gradient backpropagation."""
        model = self._make_model()
        result = model(**self.batch)
        result["loss"].backward()

        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        )

    def test_without_tcn_or_attention(self):
        """The model supports the TCN and attention ablation settings."""
        model = self._make_model(use_tcn=False, use_attention=False)
        result = model(**self.batch)

        self.assertEqual(tuple(result["logit"].shape), (5, 3))
        self.assertTrue(torch.isfinite(result["loss"]).all())

    def test_class_weighted_loss(self):
        """Weighted loss matches PyTorch cross-entropy and supports gradients."""
        weights = [1.6577, 0.4697, 3.7320]
        model = self._make_model(class_weights=weights)
        result = model(**self.batch)

        expected_loss = torch.nn.functional.cross_entropy(
            result["logit"],
            self.batch["label"][0].to(result["logit"].device).long(),
            weight=torch.tensor(
                weights, dtype=torch.float32, device=result["logit"].device
            ),
        )

        self.assertTrue(torch.allclose(result["loss"], expected_loss, atol=1e-6))

        result["loss"].backward()

        self.assertTrue(
            any(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.requires_grad
            )
        )


if __name__ == "__main__":
    unittest.main()


def test_tcn_is_causal():
    """Changing future epochs must not change earlier TCN outputs."""
    import torch
    from pyhealth.models.watchsleepnet import DeepTCNBlock

    torch.manual_seed(42)
    tcn = DeepTCNBlock(
        in_channels=4,
        out_channels=4,
        kernel_size=3,
        num_layers=3,
    ).eval()

    original = torch.randn(1, 4, 12)
    changed = original.clone()
    changed[:, :, 8:] += 100.0

    with torch.no_grad():
        original_output = tcn(original)
        changed_output = tcn(changed)

    torch.testing.assert_close(
        original_output[:, :, :8],
        changed_output[:, :, :8],
        rtol=0,
        atol=1e-6,
    )


def test_tcn_is_causal_during_training():
    """Future epochs must not affect earlier TCN outputs in training mode."""
    import torch
    from pyhealth.models.watchsleepnet import DeepTCNBlock

    torch.manual_seed(42)
    tcn = DeepTCNBlock(
        in_channels=4,
        out_channels=4,
        kernel_size=3,
        num_layers=3,
    ).train()

    original = torch.randn(1, 4, 12)
    changed = original.clone()
    changed[:, :, 8:] += 100.0

    with torch.no_grad():
        torch.manual_seed(123)
        original_output = tcn(original)
        torch.manual_seed(123)
        changed_output = tcn(changed)

    torch.testing.assert_close(
        original_output[:, :, :8],
        changed_output[:, :, :8],
        rtol=0,
        atol=1e-6,
    )



# Residual attention extension tests use synthetic inputs only.

def _make_residual_attention_layer(enabled):
    from pyhealth.models.watchsleepnet import (
        LSTMWithMultiheadAttention,
    )

    return LSTMWithMultiheadAttention(
        input_size=8,
        hidden_size=8,
        num_heads=2,
        num_layers=1,
        residual_attention=enabled,
    )


def test_residual_attention_forward():
    import torch
    from torch.nn.utils.rnn import (
        pack_padded_sequence,
        pad_packed_sequence,
    )

    torch.manual_seed(42)

    original = _make_residual_attention_layer(False)
    residual = _make_residual_attention_layer(True)

    residual.load_state_dict(
        original.state_dict(), strict=True
    )

    original.eval()
    residual.eval()

    x = torch.randn(2, 4, 8)
    lengths = torch.tensor([4, 2])

    with torch.no_grad():
        packed = pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = original.lstm(packed)

        recurrent_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=x.shape[1],
        )

        expected = original(x, lengths) + recurrent_output
        actual = residual(x, lengths)

    torch.testing.assert_close(actual, expected)


def test_residual_attention_backward():
    import torch

    torch.manual_seed(42)

    layer = _make_residual_attention_layer(True)

    x = torch.randn(
        2, 4, 8, requires_grad=True
    )

    lengths = torch.tensor([4, 2])

    output = layer(x, lengths)

    loss = output.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0

    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_residual_attention_parameter_count():
    original = _make_residual_attention_layer(False)
    residual = _make_residual_attention_layer(True)

    assert sum(
        p.numel() for p in original.parameters()
    ) == sum(
        p.numel() for p in residual.parameters()
    )

    residual.load_state_dict(
        original.state_dict(), strict=True
    )


def test_residual_attention_default():
    import torch
    from pyhealth.models.watchsleepnet import (
        LSTMWithMultiheadAttention,
    )

    default = LSTMWithMultiheadAttention(
        input_size=8,
        hidden_size=8,
        num_heads=2,
        num_layers=1,
    )

    explicit = _make_residual_attention_layer(False)

    explicit.load_state_dict(
        default.state_dict(), strict=True
    )

    default.eval()
    explicit.eval()

    x = torch.randn(2, 4, 8)
    lengths = torch.tensor([4, 2])

    with torch.no_grad():
        torch.testing.assert_close(
            default(x, lengths),
            explicit(x, lengths),
        )

    assert default.residual_attention is False


def test_residual_attention_invalid_configuration():
    import pytest
    import numpy as np
    from pyhealth.datasets import create_sample_dataset
    from pyhealth.models import WatchSleepNet

    dataset = create_sample_dataset(
        samples=[
            {
                "patient_id": "synthetic",
                "signal": np.zeros(
                    (3, 750), dtype=np.float32
                ),
                "label": {
                    "time": [0, 30, 60],
                    "values": [0, 1, 2],
                },
            }
        ],
        input_schema={"signal": "tensor"},
        output_schema={
            "label": (
                "timeseries_multiclass",
                {
                    "labels": 3,
                    "timeline": ["signal"],
                },
            )
        },
    )

    with pytest.raises(
        ValueError, match="residual_attention"
    ):
        WatchSleepNet(
            dataset=dataset,
            use_attention=False,
            residual_attention=True,
        )


def test_dreamt_sequence_task_schema_integration():
    """Exercise the public task schema through dataset and model APIs."""
    rng = np.random.default_rng(42)
    task = SleepStagingDREAMTSequence()

    samples = [
        {
            "patient_id": "synthetic-0",
            "signal": rng.random((3, 750)).astype(np.float32),
            "label": {
                "time": [0.0, 30.0, 60.0],
                "values": [0, 1, 2],
            },
        },
        {
            "patient_id": "synthetic-1",
            "signal": rng.random((2, 750)).astype(np.float32),
            "label": {
                "time": [0.0, 30.0],
                "values": [2, 1],
            },
        },
    ]

    dataset = create_sample_dataset(
        samples=samples,
        input_schema=task.input_schema,
        output_schema=task.output_schema,
        dataset_name="dreamt_sequence_schema_integration",
    )

    processor = dataset.output_processors["label"]
    assert tuple(processor.timeline) == ("signal",)
    assert processor.output_size() == 3

    batch = next(
        iter(get_dataloader(dataset, batch_size=2, shuffle=False))
    )

    values, times, offsets = batch["label"]

    assert tuple(batch["signal"].shape) == (2, 3, 750)
    assert offsets.tolist() == [0, 3, 5]
    assert values.tolist() == [0, 1, 2, 2, 1]
    assert times.tolist() == [0.0, 30.0, 60.0, 0.0, 30.0]

    model = WatchSleepNet(
        dataset=dataset,
        num_channels=8,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        tcn_layers=1,
    )

    result = model(**batch)

    assert tuple(result["logit"].shape) == (5, 3)
    assert tuple(result["y_prob"].shape) == (5, 3)
    assert torch.isfinite(result["loss"])

    expected_loss = torch.nn.functional.cross_entropy(
        result["logit"],
        values.to(result["logit"].device).long(),
    )

    torch.testing.assert_close(result["loss"], expected_loss)

    result["loss"].backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert gradients
    assert all(torch.isfinite(grad).all() for grad in gradients)



class TestWatchSleepNetDefensiveValidation(unittest.TestCase):
    """Verify checkpoint persistence and invalid input handling."""

    def setUp(self):
        fixture = TestWatchSleepNet(
            methodName="test_model_instantiation"
        )
        fixture.setUp()
        self.dataset = fixture.dataset
        self.batch = fixture.batch
        self.config = {
            "num_channels": 8,
            "hidden_dim": 8,
            "num_heads": 2,
            "num_layers": 1,
            "tcn_layers": 1,
        }

    def _model(self, **kwargs):
        config = dict(self.config)
        config.update(kwargs)
        return WatchSleepNet(dataset=self.dataset, **config)

    def test_checkpoint_disk_round_trip(self):
        import tempfile
        from pathlib import Path

        model = self._model(
            residual_attention=True,
            class_weights=[1.0, 2.0, 3.0],
        )
        model.eval()

        with torch.no_grad():
            expected = model(**self.batch)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "watchsleepnet.pt"
            torch.save(model.state_dict(), checkpoint)

            restored = self._model(
                residual_attention=True,
                class_weights=[1.0, 2.0, 3.0],
            )
            restored.load_state_dict(
                torch.load(
                    checkpoint,
                    map_location="cpu",
                    weights_only=True,
                ),
                strict=True,
            )
            restored.eval()

            with torch.no_grad():
                actual = restored(**self.batch)

        for key in ("logit", "y_prob", "loss"):
            torch.testing.assert_close(
                actual[key], expected[key]
            )

        torch.testing.assert_close(
            restored.class_weights, model.class_weights
        )

    def test_invalid_model_dimensions(self):
        cases = [
            {"num_channels": 0},
            {"hidden_dim": 0},
            {"num_layers": 0},
            {"num_heads": 0},
            {"tcn_layers": 0},
        ]
        for config in cases:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    self._model(**config)

    def test_invalid_class_weights(self):
        for weights in ([1.0, 2.0], [1.0, 0.0, 2.0],
                        [1.0, -1.0, 2.0]):
            with self.subTest(weights=weights):
                with self.assertRaisesRegex(
                    ValueError, "class_weights"
                ):
                    self._model(class_weights=weights)

    def test_invalid_temporal_layer_configurations(self):
        from pyhealth.models.watchsleepnet import (
            DeepTCNBlock,
            LSTMWithMultiheadAttention,
            ResBlock,
        )

        with self.assertRaisesRegex(ValueError, "odd"):
            DeepTCNBlock(8, 8, kernel_size=4)

        with self.assertRaisesRegex(ValueError, "divisible"):
            LSTMWithMultiheadAttention(
                input_size=8,
                hidden_size=8,
                num_heads=3,
                num_layers=1,
            )

        block = ResBlock(8, 8, stride=1)
        block.eval()
        self.assertIsInstance(
            block.downsample, torch.nn.Identity
        )
        with torch.no_grad():
            output = block(torch.zeros(2, 8, 16))
        self.assertEqual(tuple(output.shape), (2, 8, 16))

    def test_invalid_signal_dimensions(self):
        model = self._model()
        for signal in (
            torch.zeros(2, 750),
            torch.zeros(2, 3, 749),
        ):
            with self.subTest(shape=tuple(signal.shape)):
                batch = dict(self.batch)
                batch["signal"] = signal
                with self.assertRaises(ValueError):
                    model(**batch)

    def test_invalid_packed_labels(self):
        model = self._model()
        model.eval()

        batch = dict(self.batch)
        values, times, offsets = batch["label"]

        missing = dict(batch)
        del missing["label"]

        with self.assertRaisesRegex(ValueError, "Missing"):
            model(**missing)

        cases = [
            (values, "packed time-series"),
            (
                (values, times, torch.tensor([0, 5])),
                "batch sizes",
            ),
            (
                (values, times, torch.tensor([0, 0, 5])),
                "at least one label",
            ),
            (
                (values, times, torch.tensor([0, 4, 5])),
                "longer than",
            ),
            (
                (values[:-1], times[:-1], offsets),
                "different lengths",
            ),
        ]

        for labels, message in cases:
            with self.subTest(message=message):
                invalid = dict(batch)
                invalid["label"] = labels
                with self.assertRaisesRegex(
                    ValueError, message
                ):
                    model(**invalid)

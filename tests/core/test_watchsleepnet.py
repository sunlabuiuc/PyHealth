"""Tests for the simplified WatchSleepNet model. All data is synthetic."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset
from pyhealth.models import WatchSleepNet
from pyhealth.trainer import Trainer

SEQ_LEN = 3
EPOCH_LEN = 750


def _dataset(n_samples=2, include_lengths=True, single_epoch=False, seed=0):
    rng = np.random.default_rng(seed)
    shape = (EPOCH_LEN,) if single_epoch else (SEQ_LEN, EPOCH_LEN)
    samples = []
    for i in range(n_samples):
        sample = {
            "patient_id": str(i),
            "signal": rng.standard_normal(shape).astype(np.float32),
            "label": rng.integers(0, 3, size=1 if single_epoch else SEQ_LEN).astype(
                np.int64
            ),
        }
        if include_lengths:
            sample["lengths"] = 1 if single_epoch else SEQ_LEN
        samples.append(sample)
    schema = {"signal": "tensor"}
    if include_lengths:
        schema["lengths"] = "raw"
    return create_sample_dataset(
        samples=samples,
        input_schema=schema,
        output_schema={"label": ("tensor", {"dtype": torch.long})},
    )


def _tiny_model(dataset, **kwargs):
    params = {
        "num_channels": 16,
        "kernel_size": 3,
        "hidden_dim": 32,
        "num_heads": 8,
        "num_layers": 1,
        "tcn_layers": 1,
    }
    params.update(kwargs)
    return WatchSleepNet(dataset, **params)


class TestWatchSleepNet(unittest.TestCase):
    def test_construction_and_label_processor(self):
        model = _tiny_model(_dataset(), n_classes=4)
        self.assertEqual(model.n_classes, 4)
        self.assertEqual(model.label_processors["label"].size(), 4)
        self.assertEqual(model.label_processor.size(), 4)
        self.assertIs(model.forward_from_embedding.__func__, model.forward.__func__)
        self.assertNotIn("_dummy_param", model.state_dict())

    def test_forward_shapes_gradients_and_masking(self):
        dataset = _dataset()
        model = _tiny_model(dataset)
        x = torch.randn(2, SEQ_LEN, EPOCH_LEN)
        y = torch.tensor([[0, 1, -1], [1, 2, -1]])
        out = model(signal=x, label=y, lengths=[2, 2])
        self.assertEqual(out["logit"].shape, (2, SEQ_LEN, 3))
        self.assertEqual(out["y_prob"].shape, (4, 3))
        self.assertEqual(out["y_true"].tolist(), [0, 1, 1, 2])
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertTrue(torch.allclose(out["y_prob"].sum(dim=1), torch.ones(4)))
        # Padded positions are zeroed in the returned logit grid.
        self.assertTrue(torch.equal(out["logit"][0, 2], torch.zeros(3)))
        out["loss"].backward()
        for name in (
            "feature_extractor.layers.2.conv1.weight",
            "tcn.layers.0.0.weight",
            "lstm_layer.lstm.weight_ih_l0",
            "lstm_layer.attention.in_proj_weight",
            "classifier.weight",
        ):
            param = dict(model.named_parameters())[name]
            self.assertIsNotNone(param.grad, name)

    def test_derived_lengths_and_single_epoch(self):
        dataset = _dataset(include_lengths=False)
        model = _tiny_model(dataset)
        x = torch.stack(
            [torch.randn(SEQ_LEN, EPOCH_LEN), torch.zeros(SEQ_LEN, EPOCH_LEN)]
        )
        x[0, 1:] = 0
        y = torch.tensor([[0, -1, 2], [1, 2, 0]])
        out = model(signal=x, label=y)
        self.assertEqual(out["y_true"].tolist(), [0, 1])
        self.assertTrue(torch.equal(out["logit"][1, 1:], torch.zeros(2, 3)))

        single = _tiny_model(_dataset(single_epoch=True))
        out = single(
            signal=torch.randn(2, EPOCH_LEN), label=torch.tensor([0, -1])
        )
        self.assertEqual(out["logit"].shape, (2, 1, 3))
        self.assertEqual(out["y_true"].tolist(), [0])

    def test_quantization_and_checkpoint_roundtrip(self):
        dataset = _dataset()
        fp32 = _tiny_model(dataset)
        bf16 = _tiny_model(dataset, quantization="bf16")
        self.assertEqual(bf16.classifier.weight.dtype, torch.bfloat16)
        args = {
            "signal": torch.randn(2, SEQ_LEN, EPOCH_LEN),
            "label": torch.tensor([[0, 1, 2], [2, 0, 1]]),
            "lengths": [2, 3],
        }
        fp32.eval()
        bf16.eval()
        for model in (fp32, bf16):
            out = model(**args)
            self.assertEqual(out["logit"].dtype, torch.float32)
            self.assertTrue(torch.isfinite(out["loss"]))
        try:
            fp16 = _tiny_model(dataset, quantization="fp16")
        except (RuntimeError, NotImplementedError) as exc:
            self.skipTest(f"PyTorch cannot run fp16 here: {exc}")
        fp16.eval()
        self.assertTrue(torch.isfinite(fp16(**args)["loss"]))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "watchsleepnet.pt"
            Trainer(model=fp32, device="cpu", enable_logging=False).save_ckpt(str(path))
            state = torch.load(path, weights_only=True)
            self.assertEqual(len(state), len(fp32.state_dict()))
            reloaded = _tiny_model(dataset, quantization="fp16")
            Trainer(model=reloaded, device="cpu", enable_logging=False).load_ckpt(
                str(path)
            )
            reloaded.eval()
            self.assertTrue(torch.isfinite(reloaded(**args)["loss"]))

    def test_invalid_arguments(self):
        dataset = _dataset()
        for kwargs in (
            {"n_classes": 1},
            {"num_layers": 0},
            {"hidden_dim": 30, "num_heads": 8},
            {"quantization": "fp8"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                _tiny_model(dataset, **kwargs)


if __name__ == "__main__":
    unittest.main()

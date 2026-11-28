"""Tests for WatchSleepNet on tiny synthetic IBI tensors."""

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import WatchSleepNet

IGNORE = -100
EPOCHS = 5  # 30 s epochs per sample (T)
LENGTH = 40  # IBI values per epoch (L); 750 in the paper
TINY = {
    "input_length": LENGTH,
    "res_channels": (4, 8),
    "res_strides": (2, 2),
    "kernel_size": 3,
    "tcn_layers": 1,
    "hidden_dim": 8,
    "lstm_layers": 1,
    "num_heads": 2,
    "dropout": 0.0,
}


def _make_samples() -> list[dict]:
    """Four windows of 5 epochs; one epoch is unscored (ignored)."""
    rng = np.random.default_rng(0)
    labels = [
        [0, 1, 1, 2, 0],
        [1, 1, 2, 2, IGNORE],
        [0, 0, 1, 2, 1],
        [2, 1, 0, 1, 1],
    ]
    return [
        {
            "patient_id": f"patient-{i}",
            "ibi": rng.uniform(0.5, 1.2, (EPOCHS, LENGTH)).astype(np.float32),
            "label": np.array(label, dtype=np.int64),
        }
        for i, label in enumerate(labels)
    ]


class TestWatchSleepNet(unittest.TestCase):
    """Test cases for the WatchSleepNet model."""

    @classmethod
    def setUpClass(cls) -> None:
        """Builds one tiny sample dataset shared by all tests."""
        cls.dataset = create_sample_dataset(
            samples=_make_samples(),
            input_schema={"ibi": "tensor"},
            output_schema={"label": "tensor"},
            dataset_name="test_watchsleepnet",
        )

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = WatchSleepNet(dataset=self.dataset, **TINY)
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        self.batch = next(iter(loader))

    def _model(self, **overrides) -> WatchSleepNet:
        return WatchSleepNet(dataset=self.dataset, **{**TINY, **overrides})

    def _live_model(self, **overrides) -> WatchSleepNet:
        """A model whose TCN passes signal regardless of the random init.

        The TCN ends in ``relu(out + x)`` and its convolutions are followed by
        ReLUs, so a tiny randomly initialised model can have every TCN unit
        dead and ignore its input. Forcing positive ``collapse`` outputs and
        positive TCN weights keeps the input-sensitivity tests deterministic
        and lets them see inside the TCN.
        """
        model = self._model(**overrides)
        with torch.no_grad():
            model.collapse.weight.abs_()
            model.collapse.bias.fill_(0.5)
            for block in model.tcn:
                for conv in (block.conv1, block.conv2):
                    conv.weight.abs_()
                    conv.bias.zero_()
        return model.eval()

    # ----------------------------------------------------------------- init
    def test_model_initialization(self) -> None:
        self.assertIsInstance(self.model, WatchSleepNet)
        self.assertEqual(self.model.feature_key, "ibi")
        self.assertEqual(self.model.label_key, "label")
        self.assertEqual(self.model.num_classes, 3)
        self.assertGreater(sum(p.numel() for p in self.model.parameters()), 0)
        self.assertIsNotNone(self.model.tcn)
        self.assertIsNotNone(self.model.attention)
        self.assertIsNone(self.model.class_weights)

    def test_optional_modules_can_be_disabled(self) -> None:
        model = self._model(use_tcn=False, use_attention=False)
        self.assertIsNone(model.tcn)
        self.assertIsNone(model.attention)
        out = model(**self.batch)
        self.assertEqual(out["logit"].shape, (4, EPOCHS, 3))

    def test_mismatched_resnet_config_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._model(res_channels=(4, 8), res_strides=(2,))

    # -------------------------------------------------------------- forward
    def test_model_forward(self) -> None:
        out = self.model(**self.batch)
        self.assertEqual(set(out), {"logit", "y_prob", "loss", "y_true"})
        self.assertEqual(out["logit"].shape, (4, EPOCHS, 3))
        self.assertEqual(out["y_prob"].shape, (4, EPOCHS, 3))
        self.assertEqual(out["y_true"].shape, (4, EPOCHS))
        self.assertEqual(out["loss"].dim(), 0)
        self.assertTrue(torch.isfinite(out["loss"]))
        sums = out["y_prob"].sum(-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_forward_without_label_has_no_loss(self) -> None:
        out = self.model(ibi=self.batch["ibi"])
        self.assertEqual(set(out), {"logit", "y_prob"})

    def test_num_classes_sets_output_width(self) -> None:
        model = self._model(num_classes=5)
        self.assertEqual(model(ibi=self.batch["ibi"])["logit"].shape[-1], 5)

    def test_extra_channel_input(self) -> None:
        model = self._model(num_input_channels=2)
        x = torch.rand(2, EPOCHS, 2, LENGTH)
        self.assertEqual(model.encode(x).shape, (2, EPOCHS, 3))
        with self.assertRaises(ValueError):  # a 3-D input has one channel
            model.encode(torch.rand(2, EPOCHS, LENGTH))

    # ----------------------------------------------------------------- loss
    def test_loss_ignores_unscored_epochs(self) -> None:
        out = self.model(**self.batch)
        flat_true = out["y_true"].reshape(-1)
        keep = flat_true != IGNORE
        self.assertEqual(int((~keep).sum()), 1)
        expected = F.cross_entropy(out["logit"].reshape(-1, 3)[keep], flat_true[keep])
        self.assertTrue(torch.allclose(out["loss"], expected, atol=1e-5))

    def test_all_ignored_labels_give_zero_loss(self) -> None:
        batch = dict(self.batch)
        batch["label"] = torch.full_like(batch["label"], IGNORE)
        out = self.model(**batch)
        self.assertEqual(float(out["loss"].detach()), 0.0)

    def test_class_weights_change_the_loss(self) -> None:
        logits = torch.randn(2, EPOCHS, 3)
        y = torch.tensor([[0, 1, 2, 1, 0], [2, 2, 1, 0, IGNORE]])
        weights = [1.0, 2.0, 4.0]
        weighted = self._model(class_weights=weights)
        plain = self._model()
        self.assertTrue(torch.equal(weighted.class_weights, torch.tensor(weights)))
        valid = int((y != IGNORE).sum())
        expected = (
            F.cross_entropy(
                logits.reshape(-1, 3),
                y.reshape(-1),
                weight=torch.tensor(weights),
                ignore_index=IGNORE,
                reduction="sum",
            )
            / valid
        )
        self.assertTrue(
            torch.allclose(weighted.compute_loss(logits, y), expected, atol=1e-5)
        )
        self.assertFalse(
            torch.allclose(
                weighted.compute_loss(logits, y), plain.compute_loss(logits, y)
            )
        )

    def test_focal_loss_downweights_easy_epochs(self) -> None:
        logits = torch.randn(2, EPOCHS, 3)
        y = torch.tensor([[0, 1, 2, 1, 0], [2, 2, 1, 0, 1]])
        ce = self._model(focal_gamma=0.0).compute_loss(logits, y)
        focal = self._model(focal_gamma=2.0).compute_loss(logits, y)
        self.assertGreater(float(focal), 0.0)
        self.assertLess(float(focal), float(ce))

    # ------------------------------------------------------------- backward
    def test_model_backward(self) -> None:
        model = self._live_model().train()
        out = model(**self.batch)
        out["loss"].backward()
        # BaseModel's `_dummy_param` only tracks the device and gets no gradient.
        params = {
            n: p
            for n, p in model.named_parameters()
            if p.requires_grad and n != "_dummy_param"
        }
        self.assertEqual([n for n, p in params.items() if p.grad is None], [])
        for name, p in params.items():
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        # The gradient reaches the very first layer, so the network is connected.
        first = model.resnet[0].conv1.weight.grad
        self.assertGreater(float(first.abs().sum()), 0.0)
        self.assertGreater(float(model.fc.weight.grad.abs().sum()), 0.0)

    def test_one_optimizer_step_changes_weights(self) -> None:
        opt = torch.optim.SGD(self.model.parameters(), lr=0.1)
        before = self.model.fc.weight.detach().clone()
        opt.zero_grad()
        self.model(**self.batch)["loss"].backward()
        opt.step()
        self.assertFalse(torch.equal(before, self.model.fc.weight))

    # --------------------------------------------------------------- causal
    def test_causal_model_ignores_the_future(self) -> None:
        causal = self._live_model(bidirectional=False)
        x = self.batch["ibi"].float()
        future = x.clone()
        future[:, 3:] += 1.0  # change only epochs 3 and 4
        with torch.no_grad():
            a, b = causal.encode(x), causal.encode(future)
        self.assertTrue(torch.allclose(a[:, :3], b[:, :3], atol=1e-5))
        self.assertFalse(torch.allclose(a[:, 3:], b[:, 3:], atol=1e-5))

    def test_bidirectional_model_uses_the_future(self) -> None:
        bi = self._live_model(bidirectional=True)
        x = self.batch["ibi"].float()
        future = x.clone()
        future[:, 3:] += 1.0
        with torch.no_grad():
            a, b = bi.encode(x), bi.encode(future)
        self.assertFalse(torch.allclose(a[:, :3], b[:, :3], atol=1e-5))


if __name__ == "__main__":
    unittest.main()

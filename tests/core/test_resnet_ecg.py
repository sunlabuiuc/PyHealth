"""Tests for 1-D ResNet-based ECG models.

Covers ResNet18ECG, SEResNet50ECG, and LambdaResNet18ECG, exercising:
  - model initialisation and attribute checks
  - forward pass output keys and shapes
  - backward pass (gradient flow)
  - embed flag
  - custom hyperparameter variants
  - all three output modes (multilabel, multiclass, binary)
  - forward_sliding_window evaluation helper
"""

import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.resnet        import ResNet18ECG
from pyhealth.models.se_resnet     import SEResNet50ECG
from pyhealth.models.lambda_resnet import LambdaResNet18ECG

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

_N_LEADS  = 12
_LENGTH   = 1250   # 2.5 s @ 500 Hz — matches the paper's window size
_N_LABELS = 5      # number of multilabel classes used in tests


def _make_samples(n: int, rng: np.random.RandomState,
                  label_mode: str = "multilabel") -> list:
    """Return ``n`` synthetic ECG samples for the given label mode.

    For multilabel, PyHealth's MultiLabelProcessor expects each label to be a
    list of *active class indices* (set-style encoding), not a fixed-length
    binary vector.  E.g. ``[1, 3]`` means classes 1 and 3 are active.  The
    processor builds its vocabulary from the union of all class indices seen
    across the dataset, so every class in ``range(_N_LABELS)`` must appear at
    least once to guarantee a full-size output vector.
    """
    samples = []
    for i in range(n):
        if label_mode == "multilabel":
            # Sample a random subset of class indices; ensure each class
            # appears in at least one sample by cycling through them.
            active = [j for j in range(_N_LABELS) if rng.randint(0, 2)]
            # Guarantee the i-th class (mod _N_LABELS) is always present so
            # the full vocabulary is established across the dataset.
            forced = i % _N_LABELS
            if forced not in active:
                active.append(forced)
            label = sorted(active)
        elif label_mode == "multiclass":
            label = int(rng.randint(0, 3))
        else:  # binary
            label = int(rng.randint(0, 2))
        samples.append({
            "patient_id": f"p{i}",
            "visit_id":   "v0",
            "signal":     rng.randn(_N_LEADS, _LENGTH).astype(np.float32),
            "label":      label,
        })
    return samples


def _make_dataset(samples: list, label_mode: str):
    return create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema={"label": label_mode},
        dataset_name=f"test_ecg_{label_mode}",
    )


def _assert_forward_output(tc: unittest.TestCase, ret: dict,
                           batch_size: int, n_classes: int) -> None:
    """Assert standard forward-output contract."""
    tc.assertIn("loss",   ret)
    tc.assertIn("y_prob", ret)
    tc.assertIn("y_true", ret)
    tc.assertIn("logit",  ret)
    tc.assertEqual(ret["loss"].dim(), 0)
    tc.assertEqual(ret["y_prob"].shape[0], batch_size)
    tc.assertEqual(ret["y_prob"].shape[1], n_classes)
    tc.assertEqual(ret["y_true"].shape[0], batch_size)
    tc.assertEqual(ret["logit"].shape[0],  batch_size)
    tc.assertEqual(ret["logit"].shape[1],  n_classes)
    tc.assertTrue(torch.isfinite(ret["loss"]))


# ---------------------------------------------------------------------------
# ResNet-18
# ---------------------------------------------------------------------------

class TestResNet18ECG(unittest.TestCase):
    """Tests for ResNet18ECG."""

    def setUp(self):
        rng = np.random.RandomState(0)
        samples = _make_samples(4, rng, "multilabel")
        self.dataset = _make_dataset(samples, "multilabel")
        self.model   = ResNet18ECG(dataset=self.dataset)
        self.batch   = next(iter(get_dataloader(self.dataset, batch_size=4, shuffle=False)))

    # -- initialisation -------------------------------------------------------

    def test_initialization(self):
        self.assertIsInstance(self.model, ResNet18ECG)
        self.assertEqual(len(self.model.feature_keys), 1)
        self.assertIn("signal", self.model.feature_keys)
        self.assertEqual(len(self.model.label_keys), 1)
        self.assertIn("label", self.model.label_keys)
        # backbone: 4 stages
        self.assertEqual(len(self.model.backbone.stages), 4)
        # head ends with a linear layer
        self.assertIsInstance(list(self.model.head.children())[-1], torch.nn.Linear)

    def test_backbone_output_dim(self):
        x = torch.randn(2, _N_LEADS, _LENGTH)
        with torch.no_grad():
            out = self.model.backbone(x)
        self.assertEqual(out.shape, (2, 256))

    # -- forward --------------------------------------------------------------

    def test_forward_multilabel(self):
        with torch.no_grad():
            ret = self.model(**self.batch)
        _assert_forward_output(self, ret, batch_size=4, n_classes=_N_LABELS)
        # multilabel y_prob in [0, 1]
        self.assertTrue(torch.all(ret["y_prob"] >= 0))
        self.assertTrue(torch.all(ret["y_prob"] <= 1))

    def test_forward_multiclass(self):
        rng = np.random.RandomState(1)
        n_classes = 4
        samples = _make_samples(4, rng, "multiclass")
        # ensure all classes represented so tokeniser has correct size
        for i, s in enumerate(samples):
            s["label"] = i % n_classes
        ds = _make_dataset(samples, "multiclass")
        model = ResNet18ECG(dataset=ds)
        batch = next(iter(get_dataloader(ds, batch_size=4, shuffle=False)))
        with torch.no_grad():
            ret = model(**batch)
        _assert_forward_output(self, ret, batch_size=4, n_classes=n_classes)
        # multiclass y_prob rows sum to ~1
        self.assertTrue(torch.allclose(ret["y_prob"].sum(dim=1),
                                       torch.ones(4), atol=1e-5))

    def test_forward_binary(self):
        rng = np.random.RandomState(2)
        samples = _make_samples(4, rng, "binary")
        for i, s in enumerate(samples):
            s["label"] = i % 2
        ds = _make_dataset(samples, "binary")
        model = ResNet18ECG(dataset=ds)
        batch = next(iter(get_dataloader(ds, batch_size=4, shuffle=False)))
        with torch.no_grad():
            ret = model(**batch)
        _assert_forward_output(self, ret, batch_size=4, n_classes=1)
        self.assertTrue(torch.all(ret["y_prob"] >= 0))
        self.assertTrue(torch.all(ret["y_prob"] <= 1))

    # -- backward -------------------------------------------------------------

    def test_backward(self):
        ret = self.model(**self.batch)
        ret["loss"].backward()
        has_grad = any(
            p.requires_grad and p.grad is not None
            for p in self.model.parameters()
        )
        self.assertTrue(has_grad, "No parameters received gradients")

    # -- embed ----------------------------------------------------------------

    def test_embed_flag(self):
        batch = dict(self.batch, embed=True)
        with torch.no_grad():
            ret = self.model(**batch)
        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape, (4, 256))

    # -- hyperparameters ------------------------------------------------------

    def test_custom_hyperparameters(self):
        model = ResNet18ECG(
            dataset=self.dataset,
            in_channels=_N_LEADS,
            base_channels=32,
            backbone_output_dim=128,
            dropout=0.1,
        )
        x = torch.randn(2, _N_LEADS, _LENGTH)
        with torch.no_grad():
            emb = model.backbone(x)
        self.assertEqual(emb.shape, (2, 128))
        batch = next(iter(get_dataloader(self.dataset, batch_size=2, shuffle=False)))
        with torch.no_grad():
            ret = model(**batch)
        self.assertIn("loss", ret)

    # -- sliding window -------------------------------------------------------

    def test_forward_sliding_window(self):
        # Full-length recording is 5 s @ 500 Hz = 2500 samples
        signal = torch.randn(2, _N_LEADS, 2500)
        self.model.eval()
        probs = self.model.forward_sliding_window(signal, window_size=_LENGTH)
        self.assertEqual(probs.shape, (2, _N_LABELS))
        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))

    def test_forward_sliding_window_short_signal(self):
        """Signal shorter than one window is zero-padded and processed."""
        signal = torch.randn(2, _N_LEADS, 500)
        self.model.eval()
        probs = self.model.forward_sliding_window(signal, window_size=_LENGTH)
        self.assertEqual(probs.shape, (2, _N_LABELS))

    def test_forward_sliding_window_custom_step(self):
        """Custom step size produces the same output shape."""
        signal = torch.randn(2, _N_LEADS, 2500)
        self.model.eval()
        probs = self.model.forward_sliding_window(
            signal, window_size=_LENGTH, step_size=250)
        self.assertEqual(probs.shape, (2, _N_LABELS))


# ---------------------------------------------------------------------------
# SE-ResNet-50
# ---------------------------------------------------------------------------

class TestSEResNet50ECG(unittest.TestCase):
    """Tests for SEResNet50ECG."""

    def setUp(self):
        rng = np.random.RandomState(3)
        samples = _make_samples(4, rng, "multilabel")
        self.dataset = _make_dataset(samples, "multilabel")
        self.model   = SEResNet50ECG(dataset=self.dataset)
        self.batch   = next(iter(get_dataloader(self.dataset, batch_size=4, shuffle=False)))

    # -- initialisation -------------------------------------------------------

    def test_initialization(self):
        self.assertIsInstance(self.model, SEResNet50ECG)
        self.assertIn("signal", self.model.feature_keys)
        self.assertIn("label",  self.model.label_keys)
        # SE-ResNet-50 has 4 stages
        self.assertEqual(len(self.model.backbone.stages), 4)

    def test_se_blocks_present(self):
        """Every bottleneck block in the backbone contains an SEModule1d."""
        from pyhealth.models.se_resnet import SEResNetBottleneck1d, SEModule1d
        for stage in self.model.backbone.stages:
            for block in stage.children():
                self.assertIsInstance(block, SEResNetBottleneck1d)
                self.assertIsInstance(block.se_module, SEModule1d)

    def test_backbone_output_dim(self):
        x = torch.randn(2, _N_LEADS, _LENGTH)
        with torch.no_grad():
            out = self.model.backbone(x)
        self.assertEqual(out.shape, (2, 256))

    # -- forward --------------------------------------------------------------

    def test_forward_multilabel(self):
        with torch.no_grad():
            ret = self.model(**self.batch)
        _assert_forward_output(self, ret, batch_size=4, n_classes=_N_LABELS)
        self.assertTrue(torch.all(ret["y_prob"] >= 0))
        self.assertTrue(torch.all(ret["y_prob"] <= 1))

    def test_forward_multiclass(self):
        rng = np.random.RandomState(4)
        n_classes = 3
        samples = _make_samples(4, rng, "multiclass")
        for i, s in enumerate(samples):
            s["label"] = i % n_classes
        ds = _make_dataset(samples, "multiclass")
        model = SEResNet50ECG(dataset=ds)
        batch = next(iter(get_dataloader(ds, batch_size=4, shuffle=False)))
        with torch.no_grad():
            ret = model(**batch)
        _assert_forward_output(self, ret, batch_size=4, n_classes=n_classes)
        self.assertTrue(torch.allclose(ret["y_prob"].sum(dim=1),
                                       torch.ones(4), atol=1e-5))

    def test_forward_binary(self):
        rng = np.random.RandomState(5)
        samples = _make_samples(4, rng, "binary")
        for i, s in enumerate(samples):
            s["label"] = i % 2
        ds = _make_dataset(samples, "binary")
        model = SEResNet50ECG(dataset=ds)
        batch = next(iter(get_dataloader(ds, batch_size=4, shuffle=False)))
        with torch.no_grad():
            ret = model(**batch)
        _assert_forward_output(self, ret, batch_size=4, n_classes=1)

    # -- backward -------------------------------------------------------------

    def test_backward(self):
        ret = self.model(**self.batch)
        ret["loss"].backward()
        has_grad = any(
            p.requires_grad and p.grad is not None
            for p in self.model.parameters()
        )
        self.assertTrue(has_grad, "No parameters received gradients")

    # -- embed ----------------------------------------------------------------

    def test_embed_flag(self):
        batch = dict(self.batch, embed=True)
        with torch.no_grad():
            ret = self.model(**batch)
        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape, (4, 256))

    # -- hyperparameters ------------------------------------------------------

    def test_custom_reduction_ratio(self):
        """SE reduction ratio is forwarded to every SEModule1d."""
        from pyhealth.models.se_resnet import SEModule1d
        model = SEResNet50ECG(dataset=self.dataset, reduction=8)
        for m in model.backbone.modules():
            if isinstance(m, SEModule1d):
                # fc1 maps C → C//8
                in_ch  = m.fc1.in_channels
                out_ch = m.fc1.out_channels
                self.assertEqual(out_ch, in_ch // 8)

    def test_custom_hyperparameters(self):
        model = SEResNet50ECG(
            dataset=self.dataset,
            backbone_output_dim=128,
            dropout=0.1,
            reduction=8,
        )
        batch = next(iter(get_dataloader(self.dataset, batch_size=2, shuffle=False)))
        with torch.no_grad():
            ret = model(**batch)
        self.assertIn("loss", ret)
        self.assertEqual(ret["y_prob"].shape[1], _N_LABELS)

    # -- sliding window -------------------------------------------------------

    def test_forward_sliding_window(self):
        signal = torch.randn(2, _N_LEADS, 2500)
        self.model.eval()
        probs = self.model.forward_sliding_window(signal, window_size=_LENGTH)
        self.assertEqual(probs.shape, (2, _N_LABELS))
        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))

    def test_forward_sliding_window_short_signal(self):
        signal = torch.randn(2, _N_LEADS, 500)
        self.model.eval()
        probs = self.model.forward_sliding_window(signal, window_size=_LENGTH)
        self.assertEqual(probs.shape, (2, _N_LABELS))


# ---------------------------------------------------------------------------
# Lambda-ResNet-18
# ---------------------------------------------------------------------------

class TestLambdaResNet18ECG(unittest.TestCase):
    """Tests for LambdaResNet18ECG."""

    def setUp(self):
        rng = np.random.RandomState(6)
        samples = _make_samples(4, rng, "multilabel")
        self.dataset = _make_dataset(samples, "multilabel")
        self.model   = LambdaResNet18ECG(dataset=self.dataset)
        self.batch   = next(iter(get_dataloader(self.dataset, batch_size=4, shuffle=False)))

    # -- initialisation -------------------------------------------------------

    def test_initialization(self):
        self.assertIsInstance(self.model, LambdaResNet18ECG)
        self.assertIn("signal", self.model.feature_keys)
        self.assertIn("label",  self.model.label_keys)
        # backbone has 4 stages
        self.assertIsNotNone(self.model.backbone.layer1)
        self.assertIsNotNone(self.model.backbone.layer4)

    def test_lambda_layers_present(self):
        """Every block in the backbone contains a LambdaConv1d."""
        from pyhealth.models.lambda_resnet import LambdaBottleneck1d, LambdaConv1d
        for layer in [self.model.backbone.layer1,
                      self.model.backbone.layer2,
                      self.model.backbone.layer3,
                      self.model.backbone.layer4]:
            for block in layer.children():
                self.assertIsInstance(block, LambdaBottleneck1d)
                # The lambda layer is embedded in block.conv2 (an nn.Sequential)
                has_lambda = any(
                    isinstance(m, LambdaConv1d)
                    for m in block.conv2.modules()
                )
                self.assertTrue(has_lambda,
                                "LambdaConv1d not found inside LambdaBottleneck1d")

    def test_backbone_output_dim(self):
        x = torch.randn(2, _N_LEADS, _LENGTH)
        with torch.no_grad():
            out = self.model.backbone(x)
        self.assertEqual(out.shape, (2, 256))

    # -- forward --------------------------------------------------------------

    def test_forward_multilabel(self):
        with torch.no_grad():
            ret = self.model(**self.batch)
        _assert_forward_output(self, ret, batch_size=4, n_classes=_N_LABELS)
        self.assertTrue(torch.all(ret["y_prob"] >= 0))
        self.assertTrue(torch.all(ret["y_prob"] <= 1))

    def test_forward_multiclass(self):
        rng = np.random.RandomState(7)
        n_classes = 3
        samples = _make_samples(4, rng, "multiclass")
        for i, s in enumerate(samples):
            s["label"] = i % n_classes
        ds = _make_dataset(samples, "multiclass")
        model = LambdaResNet18ECG(dataset=ds)
        batch = next(iter(get_dataloader(ds, batch_size=4, shuffle=False)))
        with torch.no_grad():
            ret = model(**batch)
        _assert_forward_output(self, ret, batch_size=4, n_classes=n_classes)
        self.assertTrue(torch.allclose(ret["y_prob"].sum(dim=1),
                                       torch.ones(4), atol=1e-5))

    def test_forward_binary(self):
        rng = np.random.RandomState(8)
        samples = _make_samples(4, rng, "binary")
        for i, s in enumerate(samples):
            s["label"] = i % 2
        ds = _make_dataset(samples, "binary")
        model = LambdaResNet18ECG(dataset=ds)
        batch = next(iter(get_dataloader(ds, batch_size=4, shuffle=False)))
        with torch.no_grad():
            ret = model(**batch)
        _assert_forward_output(self, ret, batch_size=4, n_classes=1)

    # -- backward -------------------------------------------------------------

    def test_backward(self):
        ret = self.model(**self.batch)
        ret["loss"].backward()
        has_grad = any(
            p.requires_grad and p.grad is not None
            for p in self.model.parameters()
        )
        self.assertTrue(has_grad, "No parameters received gradients")

    def test_lambda_layer_gradients(self):
        """Gradients flow back through LambdaConv1d (embedding parameter)."""
        from pyhealth.models.lambda_resnet import LambdaConv1d
        ret = self.model(**self.batch)
        ret["loss"].backward()
        lambda_layers = [
            m for m in self.model.backbone.modules()
            if isinstance(m, LambdaConv1d)
        ]
        self.assertGreater(len(lambda_layers), 0)
        for lm in lambda_layers:
            self.assertIsNotNone(lm.embedding.grad,
                                 "embedding parameter has no gradient")

    # -- embed ----------------------------------------------------------------

    def test_embed_flag(self):
        batch = dict(self.batch, embed=True)
        with torch.no_grad():
            ret = self.model(**batch)
        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape, (4, 256))

    # -- hyperparameters ------------------------------------------------------

    def test_custom_hyperparameters(self):
        model = LambdaResNet18ECG(
            dataset=self.dataset,
            backbone_output_dim=128,
            dropout=0.1,
        )
        batch = next(iter(get_dataloader(self.dataset, batch_size=2, shuffle=False)))
        with torch.no_grad():
            ret = model(**batch)
        self.assertIn("loss", ret)
        self.assertEqual(ret["y_prob"].shape[1], _N_LABELS)

    # -- sliding window -------------------------------------------------------

    def test_forward_sliding_window(self):
        signal = torch.randn(2, _N_LEADS, 2500)
        self.model.eval()
        probs = self.model.forward_sliding_window(signal, window_size=_LENGTH)
        self.assertEqual(probs.shape, (2, _N_LABELS))
        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))

    def test_forward_sliding_window_short_signal(self):
        signal = torch.randn(2, _N_LEADS, 500)
        self.model.eval()
        probs = self.model.forward_sliding_window(signal, window_size=_LENGTH)
        self.assertEqual(probs.shape, (2, _N_LABELS))

    def test_clamp_stability(self):
        """Extreme input values are clamped and do not produce NaN/Inf."""
        # The backbone clamps input and inter-stage activations to [-20, 20].
        # Use batch size 4 to match self.batch["label"].
        signal = torch.full((4, _N_LEADS, _LENGTH), fill_value=1e6)
        batch = {
            "signal": signal,
            "label":  self.batch["label"],
        }
        with torch.no_grad():
            ret = self.model(**batch)
        self.assertTrue(torch.isfinite(ret["loss"]),
                        "Loss is not finite for clamped extreme input")
        self.assertTrue(torch.all(torch.isfinite(ret["y_prob"])))


# ---------------------------------------------------------------------------
# Cross-model consistency tests
# ---------------------------------------------------------------------------

class TestECGResNetConsistency(unittest.TestCase):
    """Sanity checks that hold across all three model classes."""

    def _make_model_and_batch(self, model_cls, rng_seed=42):
        rng = np.random.RandomState(rng_seed)
        samples = _make_samples(4, rng, "multilabel")
        ds = _make_dataset(samples, "multilabel")
        model = model_cls(dataset=ds)
        batch = next(iter(get_dataloader(ds, batch_size=4, shuffle=False)))
        return model, batch, ds

    def test_all_models_share_head_architecture(self):
        """All three models use the same HeadModule architecture."""
        for cls in [ResNet18ECG, SEResNet50ECG, LambdaResNet18ECG]:
            model, _, _ = self._make_model_and_batch(cls)
            children = list(model.head.children())
            self.assertIsInstance(children[0], torch.nn.Linear,   f"{cls.__name__}: head[0]")
            self.assertIsInstance(children[1], torch.nn.ReLU,     f"{cls.__name__}: head[1]")
            self.assertIsInstance(children[2], torch.nn.BatchNorm1d, f"{cls.__name__}: head[2]")
            self.assertIsInstance(children[3], torch.nn.Dropout,  f"{cls.__name__}: head[3]")
            self.assertIsInstance(children[4], torch.nn.Linear,   f"{cls.__name__}: head[4]")
            # hidden size is 128 (matching HeadModule in reference code)
            self.assertEqual(children[0].out_features, 128,
                             f"{cls.__name__}: head hidden dim should be 128")

    def test_all_models_produce_finite_output(self):
        """Forward pass produces finite loss and probabilities for all models."""
        for cls in [ResNet18ECG, SEResNet50ECG, LambdaResNet18ECG]:
            model, batch, _ = self._make_model_and_batch(cls)
            with torch.no_grad():
                ret = model(**batch)
            self.assertTrue(torch.isfinite(ret["loss"]),
                            f"{cls.__name__} loss is not finite")
            self.assertTrue(torch.all(torch.isfinite(ret["y_prob"])),
                            f"{cls.__name__} y_prob contains non-finite values")

    def test_all_models_eval_train_switch(self):
        """train() / eval() mode switches do not break forward."""
        for cls in [ResNet18ECG, SEResNet50ECG, LambdaResNet18ECG]:
            model, batch, _ = self._make_model_and_batch(cls)
            model.train()
            ret_train = model(**batch)
            model.eval()
            with torch.no_grad():
                ret_eval = model(**batch)
            self.assertIn("loss", ret_train)
            self.assertIn("loss", ret_eval)

    def test_all_models_sliding_window_consistent(self):
        """forward_sliding_window output shape is consistent with forward."""
        signal = torch.randn(2, _N_LEADS, 2500)
        for cls in [ResNet18ECG, SEResNet50ECG, LambdaResNet18ECG]:
            model, batch, _ = self._make_model_and_batch(cls)
            model.eval()
            probs = model.forward_sliding_window(signal, window_size=_LENGTH)
            self.assertEqual(probs.shape[0], 2,
                             f"{cls.__name__}: sliding window batch dim wrong")
            self.assertEqual(probs.shape[1], _N_LABELS,
                             f"{cls.__name__}: sliding window class dim wrong")


if __name__ == "__main__":
    unittest.main()

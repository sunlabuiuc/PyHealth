"""Unit tests for MedTsLLM model and its preprocessing cache helper.

Author: Anton Barchukov

Tests cover:
    - Model instantiation with synthetic data (no LLM download)
    - Forward pass output keys and shapes
    - Backward pass (gradient flow)
    - Loss computation when labels provided
    - Parameter filtering (frozen params excluded from optimizer)
    - State dict save/load with frozen params
    - Internal layers (RevIN, PatchEmbedding, ReprogrammingLayer)
    - Different sequence lengths, feature counts, covariate modes
    - Edge cases: no labels, invalid inputs
    - Compatibility with PyHealth pipeline
    - ``_medtsllm_cache`` fingerprint + load_or_build helpers used by
      LUDB / MIT-BIH / BIDMC preprocessing
"""

import os
import tempfile
import unittest

import torch
import numpy as np


def _make_dataset(
    n_samples: int = 4,
    seq_len: int = 128,
    n_classes: int = 4,
):
    """Create a minimal SampleDataset with signal + label."""
    from pyhealth.datasets import create_sample_dataset

    samples = []
    for i in range(n_samples):
        signal = np.random.randn(seq_len).astype(np.float32)
        label = np.random.randint(0, n_classes, size=seq_len).astype(
            np.int64
        )
        samples.append({
            "patient_id": f"p{i}",
            "visit_id": "v0",
            "signal": signal,
            "label": label,
        })

    return create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema={"label": "tensor"},
        dataset_name="test_medtsllm",
    )


def _make_model(dataset, **kwargs):
    """Create a MedTsLLM with synthetic word embeddings (no LLM)."""
    from pyhealth.models.medtsllm import MedTsLLM

    defaults = dict(
        dataset=dataset,
        seq_len=128,
        n_features=1,
        n_classes=4,
        backbone=None,
        word_embeddings=torch.randn(100, 64),
        d_model=16,
        d_ff=32,
        n_heads=4,
        num_tokens=50,
        patch_len=16,
        stride=8,
        dropout=0.1,
    )
    defaults.update(kwargs)
    return MedTsLLM(**defaults)


def _make_batch(dataset, batch_size=2):
    """Get a single batch from the dataset."""
    from pyhealth.datasets import get_dataloader

    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)
    return next(iter(loader))


# ------------------------------------------------------------------ #
# Forward pass tests
# ------------------------------------------------------------------ #


class TestMedTsLLMForward(unittest.TestCase):
    """Tests for MedTsLLM forward pass."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_dataset()
        cls.model = _make_model(cls.dataset)
        cls.batch = _make_batch(cls.dataset)

    def test_forward_keys(self):
        """Output dict has expected keys when labels provided."""
        out = self.model(**self.batch)
        self.assertIn("logit", out)
        self.assertIn("y_prob", out)
        self.assertIn("loss", out)
        self.assertIn("y_true", out)

    def test_logit_shape(self):
        """Logit has shape (batch, seq_len, n_classes)."""
        out = self.model(**self.batch)
        bs = self.batch["label"].shape[0]
        self.assertEqual(out["logit"].shape, (bs, 128, 4))

    def test_y_prob_shape(self):
        """y_prob has same shape as logit."""
        out = self.model(**self.batch)
        self.assertEqual(out["y_prob"].shape, out["logit"].shape)

    def test_y_prob_sums_to_one(self):
        """Softmax probabilities sum to ~1 along class dim."""
        out = self.model(**self.batch)
        sums = out["y_prob"].sum(dim=-1)
        self.assertTrue(
            torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        )

    def test_y_prob_non_negative(self):
        """All probabilities are non-negative."""
        out = self.model(**self.batch)
        self.assertTrue((out["y_prob"] >= 0).all())

    def test_loss_is_scalar(self):
        """Loss is a scalar tensor."""
        out = self.model(**self.batch)
        self.assertEqual(out["loss"].shape, ())

    def test_loss_is_finite(self):
        """Loss is not NaN or Inf."""
        out = self.model(**self.batch)
        self.assertTrue(out["loss"].isfinite())

    def test_2d_signal_input(self):
        """Forward works with 2D signal (batch, seq_len) — auto-unsqueeze."""
        signal = torch.randn(2, 128)
        label = torch.randint(0, 4, (2, 128))
        out = self.model(signal=signal, label=label)
        self.assertEqual(out["logit"].shape, (2, 128, 4))

    def test_3d_signal_input(self):
        """Forward works with 3D signal (batch, seq_len, features)."""
        signal = torch.randn(2, 128, 1)
        label = torch.randint(0, 4, (2, 128))
        out = self.model(signal=signal, label=label)
        self.assertEqual(out["logit"].shape, (2, 128, 4))


# ------------------------------------------------------------------ #
# Backward pass tests
# ------------------------------------------------------------------ #


class TestMedTsLLMBackward(unittest.TestCase):
    """Tests for MedTsLLM backward pass."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_dataset()
        cls.model = _make_model(cls.dataset)
        cls.batch = _make_batch(cls.dataset)

    def test_backward(self):
        """Loss backward pass succeeds."""
        out = self.model(**self.batch)
        out["loss"].backward()

    def test_gradients_flow(self):
        """Trainable parameters receive gradients."""
        self.model.zero_grad()
        out = self.model(**self.batch)
        out["loss"].backward()

        has_grad = False
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    has_grad = True
                    break
        self.assertTrue(has_grad, "No gradients found in trainable params")

    def test_word_embeddings_frozen(self):
        """Word embeddings should not receive gradients."""
        self.assertFalse(self.model.word_embeddings.requires_grad)

    def test_parameters_only_trainable(self):
        """parameters() yields only trainable params (no frozen)."""
        for p in self.model.parameters():
            self.assertTrue(
                p.requires_grad,
                "parameters() yielded a frozen parameter",
            )

    def test_named_parameters_only_trainable(self):
        """named_parameters() excludes frozen params."""
        for name, p in self.model.named_parameters():
            self.assertTrue(
                p.requires_grad,
                f"named_parameters() yielded frozen param: {name}",
            )

    def test_frozen_params_excluded_from_count(self):
        """Frozen word_embeddings not in parameters() count."""
        param_names = {n for n, _ in self.model.named_parameters()}
        self.assertNotIn("word_embeddings", param_names)


# ------------------------------------------------------------------ #
# State dict tests
# ------------------------------------------------------------------ #


class TestMedTsLLMStateDict(unittest.TestCase):
    """Tests for checkpoint save/load with frozen params."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_dataset()
        cls.model = _make_model(cls.dataset)

    def test_state_dict_includes_all_params(self):
        """state_dict() includes frozen params (needed for full reload)."""
        sd = self.model.state_dict()
        self.assertIn("word_embeddings", sd)
        self.assertIn("patch_embedding.value_embedding.conv.weight", sd)

    def test_load_state_dict_roundtrip(self):
        """Save and reload state dict without errors."""
        sd = self.model.state_dict()
        model2 = _make_model(self.dataset)
        model2.load_state_dict(sd)

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(
            self.model.state_dict().items(),
            model2.state_dict().items(),
        ):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.equal(p1, p2), f"Mismatch in {n1}")


# ------------------------------------------------------------------ #
# No labels tests
# ------------------------------------------------------------------ #


class TestMedTsLLMNoLabels(unittest.TestCase):
    """Tests for forward pass without labels."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_dataset()
        cls.model = _make_model(cls.dataset)

    def test_no_loss_without_labels(self):
        """No loss key when labels not in kwargs."""
        signal = torch.randn(2, 128)
        out = self.model(signal=signal)
        self.assertNotIn("loss", out)
        self.assertNotIn("y_true", out)
        self.assertIn("logit", out)
        self.assertIn("y_prob", out)


# ------------------------------------------------------------------ #
# Internal layers tests
# ------------------------------------------------------------------ #


class TestRevIN(unittest.TestCase):
    """Tests for RevIN normalization layer."""

    def test_normalize_denormalize_roundtrip(self):
        """RevIN(x, 'norm') -> RevIN(result, 'denorm') ≈ x."""
        from pyhealth.models._medtsllm.layers import RevIN

        revin = RevIN(num_features=3)
        x = torch.randn(2, 64, 3) * 10 + 5  # non-zero mean, large std
        normed = revin(x, "norm")
        recovered = revin(normed, "denorm")
        self.assertTrue(
            torch.allclose(x, recovered, atol=1e-5),
            "RevIN roundtrip failed",
        )

    def test_normalize_zero_mean_unit_var(self):
        """After normalization, each feature has ~0 mean and ~1 std."""
        from pyhealth.models._medtsllm.layers import RevIN

        revin = RevIN(num_features=2)
        x = torch.randn(4, 100, 2) * 5 + 3
        normed = revin(x, "norm")
        # Check per-sample, per-feature
        means = normed.mean(dim=1)
        stds = normed.std(dim=1, unbiased=False)
        self.assertTrue(
            torch.allclose(means, torch.zeros_like(means), atol=1e-4)
        )
        self.assertTrue(
            torch.allclose(stds, torch.ones_like(stds), atol=0.1)
        )

    def test_denorm_before_norm_raises(self):
        """Calling denorm before norm raises RuntimeError."""
        from pyhealth.models._medtsllm.layers import RevIN

        revin = RevIN(num_features=1)
        x = torch.randn(1, 10, 1)
        with self.assertRaises(RuntimeError):
            revin(x, "denorm")

    def test_invalid_mode_raises(self):
        """Invalid mode raises ValueError."""
        from pyhealth.models._medtsllm.layers import RevIN

        revin = RevIN(num_features=1)
        x = torch.randn(1, 10, 1)
        with self.assertRaises(ValueError):
            revin(x, "invalid")


class TestPatchEmbedding(unittest.TestCase):
    """Tests for PatchEmbedding layer."""

    def test_output_shape(self):
        """Output has correct shape (batch*features, n_patches, d_model)."""
        from pyhealth.models._medtsllm.layers import PatchEmbedding

        pe = PatchEmbedding(d_model=32, patch_len=16, stride=8)
        x = torch.randn(2, 1, 128)  # (batch, features, seq_len)
        out, n_vars = pe(x)
        self.assertEqual(n_vars, 1)
        # n_patches = (128 - 16) // 8 + 2 = 16
        expected_patches = (128 - 16) // 8 + 2
        self.assertEqual(out.shape, (2, expected_patches, 32))

    def test_multivariate_output_shape(self):
        """Multivariate input merges batch and feature dims."""
        from pyhealth.models._medtsllm.layers import PatchEmbedding

        pe = PatchEmbedding(d_model=16, patch_len=8, stride=4)
        x = torch.randn(3, 5, 64)  # 3 batches, 5 features
        out, n_vars = pe(x)
        self.assertEqual(n_vars, 5)
        self.assertEqual(out.shape[0], 15)  # 3 * 5
        self.assertEqual(out.shape[2], 16)  # d_model

    def test_gradient_flow(self):
        """Gradients flow through patch embedding."""
        from pyhealth.models._medtsllm.layers import PatchEmbedding

        pe = PatchEmbedding(d_model=16, patch_len=8, stride=4)
        x = torch.randn(2, 1, 32, requires_grad=True)
        out, _ = pe(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)


class TestReprogrammingLayer(unittest.TestCase):
    """Tests for ReprogrammingLayer (cross-attention)."""

    def test_output_shape(self):
        """Output shape matches (batch, n_patches, d_llm)."""
        from pyhealth.models._medtsllm.layers import ReprogrammingLayer

        layer = ReprogrammingLayer(
            d_model=32, n_heads=4, d_keys=16, d_llm=64
        )
        target = torch.randn(2, 10, 32)  # patches
        source = torch.randn(50, 64)  # word prototypes
        out = layer(target, source, source)
        self.assertEqual(out.shape, (2, 10, 64))

    def test_gradient_flow(self):
        """Gradients flow through reprogramming layer."""
        from pyhealth.models._medtsllm.layers import ReprogrammingLayer

        layer = ReprogrammingLayer(
            d_model=16, n_heads=2, d_keys=8, d_llm=32
        )
        target = torch.randn(2, 5, 16, requires_grad=True)
        source = torch.randn(20, 32)
        out = layer(target, source, source)
        out.sum().backward()
        self.assertIsNotNone(target.grad)


class TestFlattenHead(unittest.TestCase):
    """Tests for FlattenHead output projection."""

    def test_output_shape(self):
        """Flattens and projects to correct output size."""
        from pyhealth.models._medtsllm.layers import FlattenHead

        head = FlattenHead(n_features_in=64, n_outputs=128)
        x = torch.randn(2, 8, 8)  # (batch, d_ff, n_patches)
        out = head(x)
        self.assertEqual(out.shape, (2, 128))


# ------------------------------------------------------------------ #
# Configuration tests
# ------------------------------------------------------------------ #


class TestMedTsLLMConfigs(unittest.TestCase):
    """Tests for different model configurations."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_dataset(seq_len=64, n_classes=2)

    def test_different_seq_len(self):
        """Model works with non-default sequence length."""
        model = _make_model(
            self.dataset,
            seq_len=64,
            n_classes=2,
            word_embeddings=torch.randn(50, 32),
        )
        signal = torch.randn(2, 64)
        label = torch.randint(0, 2, (2, 64))
        out = model(signal=signal, label=label)
        self.assertEqual(out["logit"].shape, (2, 64, 2))
        self.assertTrue(out["loss"].isfinite())

    def test_multivariate_concat(self):
        """Model works with concat covariate mode."""
        dataset = _make_dataset(seq_len=64, n_classes=2)
        model = _make_model(
            dataset,
            seq_len=64,
            n_features=3,
            n_classes=2,
            covariate_mode="concat",
            word_embeddings=torch.randn(50, 32),
        )
        signal = torch.randn(2, 64, 3)
        label = torch.randint(0, 2, (2, 64))
        out = model(signal=signal, label=label)
        self.assertEqual(out["logit"].shape, (2, 64, 2))
        self.assertTrue(out["loss"].isfinite())
        out["loss"].backward()

    def test_univariate_mode(self):
        """Model works with univariate covariate mode (default)."""
        dataset = _make_dataset(seq_len=64, n_classes=3)
        model = _make_model(
            dataset,
            seq_len=64,
            n_features=1,
            n_classes=3,
            covariate_mode="univariate",
            word_embeddings=torch.randn(50, 32),
        )
        signal = torch.randn(2, 64)
        label = torch.randint(0, 3, (2, 64))
        out = model(signal=signal, label=label)
        self.assertEqual(out["logit"].shape, (2, 64, 3))

    def test_small_d_model(self):
        """Model works with minimal dimensions."""
        dataset = _make_dataset(seq_len=32, n_classes=2)
        model = _make_model(
            dataset,
            seq_len=32,
            n_classes=2,
            d_model=8,
            d_ff=16,
            n_heads=2,
            num_tokens=10,
            word_embeddings=torch.randn(20, 16),
        )
        signal = torch.randn(1, 32)
        label = torch.randint(0, 2, (1, 32))
        out = model(signal=signal, label=label)
        self.assertTrue(out["loss"].isfinite())


# ------------------------------------------------------------------ #
# LinearProjection layer tests
# ------------------------------------------------------------------ #


class TestLinearProjection(unittest.TestCase):
    """Tests for LinearProjection ablation layer."""

    def test_output_shape(self):
        """Output shape matches (batch, n_patches, d_llm)."""
        from pyhealth.models._medtsllm.layers import LinearProjection

        layer = LinearProjection(d_model=32, d_llm=64)
        target = torch.randn(2, 10, 32)
        source = torch.randn(50, 64)  # ignored
        out = layer(target, source, source)
        self.assertEqual(out.shape, (2, 10, 64))

    def test_ignores_source(self):
        """Output is the same regardless of source embeddings."""
        from pyhealth.models._medtsllm.layers import LinearProjection

        layer = LinearProjection(d_model=16, d_llm=32)
        target = torch.randn(2, 5, 16)
        source_a = torch.randn(20, 32)
        source_b = torch.randn(100, 32)
        out_a = layer(target, source_a, source_a)
        out_b = layer(target, source_b, source_b)
        self.assertTrue(torch.equal(out_a, out_b))

    def test_gradient_flow(self):
        """Gradients flow through linear projection."""
        from pyhealth.models._medtsllm.layers import LinearProjection

        layer = LinearProjection(d_model=16, d_llm=32)
        target = torch.randn(2, 5, 16, requires_grad=True)
        source = torch.randn(20, 32)
        out = layer(target, source, source)
        out.sum().backward()
        self.assertIsNotNone(target.grad)


# ------------------------------------------------------------------ #
# Prompt builder tests
# ------------------------------------------------------------------ #


class TestPromptBuilder(unittest.TestCase):
    """Tests for build_prompt function."""

    def test_dataset_task_prompt(self):
        """Builds prompt with dataset and task descriptions."""
        from pyhealth.models._medtsllm.prompt import build_prompt

        inputs = {"x_enc": torch.randn(2, 128, 1)}
        prompts = build_prompt(
            inputs,
            dataset_description="Test dataset",
            task_description="Test task",
            include_dataset=True,
            include_task=True,
        )
        self.assertEqual(len(prompts), 2)
        flat = " ".join(prompts[0])
        self.assertIn("Test dataset", flat)
        self.assertIn("Test task", flat)

    def test_no_prompt(self):
        """Empty prompt when all flags are False."""
        from pyhealth.models._medtsllm.prompt import build_prompt

        inputs = {"x_enc": torch.randn(1, 64, 1)}
        prompts = build_prompt(
            inputs,
            include_dataset=False,
            include_task=False,
            include_clip=False,
            include_stats=False,
        )
        # Should still have "Time series:" at minimum
        flat = " ".join(prompts[0])
        self.assertIn("Time series:", flat)

    def test_clip_prompt(self):
        """Per-sample descriptions included when clip=True."""
        from pyhealth.models._medtsllm.prompt import build_prompt

        inputs = {
            "x_enc": torch.randn(2, 64, 1),
            "descriptions": ["age: 51, sex: F", "age: 64, sex: M"],
        }
        prompts = build_prompt(
            inputs,
            include_dataset=False,
            include_task=False,
            include_clip=True,
        )
        flat_0 = " ".join(prompts[0])
        flat_1 = " ".join(prompts[1])
        self.assertIn("age: 51", flat_0)
        self.assertIn("age: 64", flat_1)

    def test_batch_size_matches(self):
        """Number of prompts matches batch size."""
        from pyhealth.models._medtsllm.prompt import build_prompt

        inputs = {"x_enc": torch.randn(5, 32, 1)}
        prompts = build_prompt(inputs)
        self.assertEqual(len(prompts), 5)

    def test_stats_prompt_included(self):
        """Stats prompt appears when include_stats=True."""
        from pyhealth.models._medtsllm.prompt import build_prompt

        inputs = {"x_enc": torch.randn(2, 64, 1)}
        prompts = build_prompt(
            inputs,
            include_dataset=False,
            include_task=False,
            include_stats=True,
        )
        flat = " ".join(prompts[0])
        self.assertIn("Input statistics", flat)
        self.assertIn("min value", flat)
        self.assertIn("max value", flat)
        self.assertIn("median value", flat)
        self.assertIn("trend", flat)
        self.assertIn("lags", flat)

    def test_stats_prompt_omitted_by_default(self):
        """Stats prompt absent when include_stats=False."""
        from pyhealth.models._medtsllm.prompt import build_prompt

        inputs = {"x_enc": torch.randn(2, 64, 1)}
        prompts = build_prompt(
            inputs,
            include_dataset=False,
            include_task=False,
            include_stats=False,
        )
        flat = " ".join(prompts[0])
        self.assertNotIn("Input statistics", flat)


# ------------------------------------------------------------------ #
# compute_lags tests
# ------------------------------------------------------------------ #


class TestComputeLags(unittest.TestCase):
    """Tests for the FFT-based autocorrelation lag helper."""

    def test_shape_univariate(self):
        """Shape is (batch, n_lags) for univariate input."""
        from pyhealth.models._medtsllm.prompt import compute_lags

        x = torch.randn(4, 128)
        lags = compute_lags(x, n_lags=5)
        self.assertEqual(lags.shape, (4, 5))

    def test_shape_multivariate(self):
        """Shape is (batch, n_lags) for 3D input."""
        from pyhealth.models._medtsllm.prompt import compute_lags

        x = torch.randn(4, 128, 3)
        lags = compute_lags(x, n_lags=3)
        self.assertEqual(lags.shape, (4, 3))

    def test_lags_are_long(self):
        """Lag indices are integer-valued."""
        from pyhealth.models._medtsllm.prompt import compute_lags

        x = torch.randn(2, 64, 1)
        lags = compute_lags(x, n_lags=5)
        self.assertEqual(lags.dtype, torch.int64)

    def test_lags_in_range(self):
        """Lag indices fall within sequence length."""
        from pyhealth.models._medtsllm.prompt import compute_lags

        seq_len = 128
        x = torch.randn(2, seq_len)
        lags = compute_lags(x, n_lags=5)
        self.assertTrue((lags >= 0).all())
        self.assertTrue((lags < seq_len).all())


# ------------------------------------------------------------------ #
# Task parameter + default task_description
# ------------------------------------------------------------------ #


class TestTaskParam(unittest.TestCase):
    """Tests for the ``task`` constructor parameter."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_dataset()

    def test_default_task_is_semantic_segmentation(self):
        """Default task is semantic_segmentation."""
        model = _make_model(self.dataset)
        self.assertEqual(model.task, "semantic_segmentation")

    def test_invalid_task_raises(self):
        """Unknown task string raises ValueError."""
        with self.assertRaises(ValueError):
            _make_model(self.dataset, task="not_a_real_task")

    def test_semseg_description(self):
        """semantic_segmentation task_description says 'Classify'."""
        model = _make_model(self.dataset, task="semantic_segmentation")
        self.assertIn("Classify", model.task_description)

    def test_segmentation_description(self):
        """segmentation task_description says 'change points'."""
        model = _make_model(self.dataset, task="segmentation")
        self.assertIn("change points", model.task_description)

    def test_anomaly_description(self):
        """anomaly_detection task_description says 'Reconstruct'."""
        model = _make_model(self.dataset, task="anomaly_detection")
        self.assertIn("Reconstruct", model.task_description)

    def test_reconstruction_description(self):
        """reconstruction task_description says 'Reconstruct'."""
        model = _make_model(self.dataset, task="reconstruction")
        self.assertIn("Reconstruct", model.task_description)

    def test_forecasting_description(self):
        """forecasting task_description says 'Forecast'."""
        model = _make_model(self.dataset, task="forecasting")
        self.assertIn("Forecast", model.task_description)

    def test_explicit_task_description_wins(self):
        """Explicit task_description is not overwritten by default."""
        model = _make_model(
            self.dataset,
            task="semantic_segmentation",
            task_description="custom description here",
        )
        self.assertEqual(
            model.task_description, "custom description here"
        )


# ------------------------------------------------------------------ #
# Prompt config knobs
# ------------------------------------------------------------------ #


class TestPromptConfigKnobs(unittest.TestCase):
    """All four prompt knobs are exposed and default True."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_dataset()

    def test_four_knobs_present(self):
        """prompt_config has dataset/task/patient/stats keys."""
        model = _make_model(self.dataset)
        for key in ("dataset", "task", "patient", "stats"):
            self.assertIn(key, model.prompt_config)

    def test_prompt_defaults(self):
        """dataset/task/patient default True; stats defaults False to
        match the cs598-pyhealth dtp reference config."""
        model = _make_model(self.dataset)
        self.assertTrue(model.prompt_config["dataset"])
        self.assertTrue(model.prompt_config["task"])
        self.assertTrue(model.prompt_config["patient"])
        self.assertFalse(model.prompt_config["stats"])

    def test_individual_toggles(self):
        """Each knob can be toggled independently."""
        model = _make_model(
            self.dataset,
            prompt_dataset=False,
            prompt_task=True,
            prompt_patient=False,
            prompt_stats=False,
        )
        self.assertFalse(model.prompt_config["dataset"])
        self.assertTrue(model.prompt_config["task"])
        self.assertFalse(model.prompt_config["patient"])
        self.assertFalse(model.prompt_config["stats"])


# ------------------------------------------------------------------ #
# reprogramming_layer override (LinearProjection ablation)
# ------------------------------------------------------------------ #


class TestReprogrammingOverride(unittest.TestCase):
    """Override ReprogrammingLayer with LinearProjection."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_dataset()

    def test_linear_projection_override(self):
        """LinearProjection slots into the model."""
        from pyhealth.models._medtsllm.layers import LinearProjection

        # d_model=16, d_llm=64 in _make_model defaults
        linear = LinearProjection(d_model=16, d_llm=64)
        model = _make_model(self.dataset, reprogramming_layer=linear)
        self.assertIs(model.reprogramming_layer, linear)

    def test_forward_with_linear_projection(self):
        """Forward pass works with LinearProjection ablation."""
        from pyhealth.models._medtsllm.layers import LinearProjection

        linear = LinearProjection(d_model=16, d_llm=64)
        model = _make_model(self.dataset, reprogramming_layer=linear)
        batch = _make_batch(self.dataset)
        out = model(**batch)
        self.assertIn("logit", out)
        self.assertTrue(out["loss"].isfinite())


# ------------------------------------------------------------------ #
# Description coercion
# ------------------------------------------------------------------ #


class TestDescriptionCoercion(unittest.TestCase):
    """_coerce_descriptions handles list, tuple, scalar-string."""

    def test_list_passthrough(self):
        from pyhealth.models.medtsllm import _coerce_descriptions

        desc = ["a", "b"]
        self.assertEqual(_coerce_descriptions(desc, bs=2), ["a", "b"])

    def test_tuple_coerced(self):
        from pyhealth.models.medtsllm import _coerce_descriptions

        desc = ("a", "b", "c")
        self.assertEqual(
            _coerce_descriptions(desc, bs=3), ["a", "b", "c"]
        )

    def test_scalar_string_broadcast(self):
        """Single string broadcasts to batch size, not list-of-chars."""
        from pyhealth.models.medtsllm import _coerce_descriptions

        desc = "hello"
        out = _coerce_descriptions(desc, bs=3)
        self.assertEqual(out, ["hello", "hello", "hello"])


# ------------------------------------------------------------------ #
# Patient prompt integration test
# ------------------------------------------------------------------ #


class TestMedTsLLMPatientPrompt(unittest.TestCase):
    """Tests for patient prompt path in model forward."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_dataset()
        # Model with prompt_patient=True but no LLM
        # (synthetic replacement skips prompting, so we test the config)
        cls.model = _make_model(cls.dataset, prompt_patient=True)

    def test_prompt_config_has_patient(self):
        """prompt_config includes patient key."""
        self.assertIn("patient", self.model.prompt_config)
        self.assertTrue(self.model.prompt_config["patient"])

    def test_forward_with_description_kwarg(self):
        """Forward pass works when description is in kwargs."""
        signal = torch.randn(2, 128)
        label = torch.randint(0, 4, (2, 128))
        out = self.model(
            signal=signal,
            label=label,
            description=["age: 51, sex: F", "age: 64, sex: M"],
        )
        self.assertIn("logit", out)
        self.assertTrue(out["loss"].isfinite())

    def test_forward_without_description(self):
        """Forward pass works even without description kwarg."""
        signal = torch.randn(2, 128)
        label = torch.randint(0, 4, (2, 128))
        out = self.model(signal=signal, label=label)
        self.assertIn("logit", out)
        self.assertTrue(out["loss"].isfinite())


# ------------------------------------------------------------------ #
# Task branching in forward (Phase 3)
# ------------------------------------------------------------------ #


def _make_binary_dataset(
    n_samples: int = 4,
    seq_len: int = 128,
    n_features: int = 1,
):
    """Dataset with binary float labels (boundary detection style)."""
    from pyhealth.datasets import create_sample_dataset

    samples = []
    for i in range(n_samples):
        if n_features == 1:
            signal = np.random.randn(seq_len).astype(np.float32)
        else:
            signal = np.random.randn(seq_len, n_features).astype(
                np.float32
            )
        label = np.random.randint(0, 2, size=seq_len).astype(np.float32)
        samples.append({
            "patient_id": f"p{i}",
            "visit_id": "v0",
            "signal": signal,
            "label": label,
        })

    return create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema={"label": "tensor"},
        dataset_name="test_medtsllm_binary",
    )


def _make_unlabeled_dataset(
    n_samples: int = 4,
    seq_len: int = 128,
    n_features: int = 2,
):
    """Multivariate dataset without labels (reconstruction style)."""
    from pyhealth.datasets import create_sample_dataset

    samples = []
    for i in range(n_samples):
        signal = np.random.randn(seq_len, n_features).astype(np.float32)
        samples.append({
            "patient_id": f"p{i}",
            "visit_id": "v0",
            "signal": signal,
        })

    return create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema={},
        dataset_name="test_medtsllm_unlabeled",
    )


class TestTaskOutputShapes(unittest.TestCase):
    """Output head shape and n_outputs_per_step adapt to task."""

    def test_semseg_n_outputs_per_step(self):
        """semantic_segmentation => n_outputs_per_step == n_classes."""
        dataset = _make_dataset(n_classes=4)
        model = _make_model(
            dataset, task="semantic_segmentation", n_classes=4
        )
        self.assertEqual(model.n_outputs_per_step, 4)

    def test_segmentation_n_outputs_per_step(self):
        """segmentation => n_outputs_per_step == 1 (binary logit)."""
        dataset = _make_binary_dataset()
        model = _make_model(dataset, task="segmentation")
        self.assertEqual(model.n_outputs_per_step, 1)

    def test_anomaly_n_outputs_per_step(self):
        """anomaly_detection => n_outputs_per_step == n_features."""
        dataset = _make_unlabeled_dataset(n_features=2)
        model = _make_model(
            dataset,
            task="anomaly_detection",
            n_features=2,
            covariate_mode="concat",
        )
        self.assertEqual(model.n_outputs_per_step, 2)

    def test_reconstruction_n_outputs_per_step(self):
        """reconstruction => n_outputs_per_step == n_features."""
        dataset = _make_unlabeled_dataset(n_features=3)
        model = _make_model(
            dataset,
            task="reconstruction",
            n_features=3,
            covariate_mode="concat",
        )
        self.assertEqual(model.n_outputs_per_step, 3)

    def test_semseg_logit_shape(self):
        """semantic_segmentation logit is (bs, pred_len, n_classes)."""
        dataset = _make_dataset(n_classes=4)
        model = _make_model(
            dataset, task="semantic_segmentation", n_classes=4
        )
        batch = _make_batch(dataset)
        out = model(**batch)
        bs = batch["label"].shape[0]
        self.assertEqual(out["logit"].shape, (bs, 128, 4))

    def test_segmentation_logit_shape(self):
        """segmentation logit is (bs, pred_len) after squeeze."""
        dataset = _make_binary_dataset()
        model = _make_model(dataset, task="segmentation")
        batch = _make_batch(dataset)
        out = model(**batch)
        bs = batch["label"].shape[0]
        self.assertEqual(out["logit"].shape, (bs, 128))

    def test_anomaly_reconstruction_shape(self):
        """anomaly_detection prediction is (bs, pred_len, n_features)."""
        dataset = _make_unlabeled_dataset(n_features=2)
        model = _make_model(
            dataset,
            task="anomaly_detection",
            n_features=2,
            covariate_mode="concat",
        )
        batch = _make_batch(dataset)
        out = model(**batch)
        bs = batch["signal"].shape[0]
        self.assertEqual(out["logit"].shape, (bs, 128, 2))


class TestTaskLosses(unittest.TestCase):
    """Task-specific loss computation."""

    def test_semseg_uses_cross_entropy(self):
        """semseg loss is finite and bounded for random inputs."""
        dataset = _make_dataset(n_classes=4)
        model = _make_model(
            dataset, task="semantic_segmentation", n_classes=4
        )
        batch = _make_batch(dataset)
        out = model(**batch)
        self.assertIn("loss", out)
        self.assertTrue(out["loss"].isfinite())

    def test_segmentation_uses_bce(self):
        """segmentation loss is finite under BCE-with-logits."""
        dataset = _make_binary_dataset()
        model = _make_model(dataset, task="segmentation")
        batch = _make_batch(dataset)
        out = model(**batch)
        self.assertIn("loss", out)
        self.assertTrue(out["loss"].isfinite())

    def test_anomaly_uses_mse_no_label(self):
        """anomaly_detection computes MSE against signal, no label needed."""
        dataset = _make_unlabeled_dataset(n_features=2)
        model = _make_model(
            dataset,
            task="anomaly_detection",
            n_features=2,
            covariate_mode="concat",
        )
        batch = _make_batch(dataset)
        out = model(**batch)
        self.assertIn("loss", out)
        self.assertTrue(out["loss"].isfinite())

    def test_reconstruction_uses_mse(self):
        """reconstruction computes MSE against signal."""
        dataset = _make_unlabeled_dataset(n_features=2)
        model = _make_model(
            dataset,
            task="reconstruction",
            n_features=2,
            covariate_mode="concat",
        )
        batch = _make_batch(dataset)
        out = model(**batch)
        self.assertIn("loss", out)
        self.assertTrue(out["loss"].isfinite())


class TestTaskGradients(unittest.TestCase):
    """Backward pass works across all task types."""

    def _backward_flows(self, model, batch):
        out = model(**batch)
        loss = out["loss"]
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        return has_grad

    def test_semseg_backward(self):
        dataset = _make_dataset(n_classes=4)
        model = _make_model(
            dataset, task="semantic_segmentation", n_classes=4
        )
        self.assertTrue(
            self._backward_flows(model, _make_batch(dataset))
        )

    def test_segmentation_backward(self):
        dataset = _make_binary_dataset()
        model = _make_model(dataset, task="segmentation")
        self.assertTrue(
            self._backward_flows(model, _make_batch(dataset))
        )

    def test_anomaly_backward(self):
        dataset = _make_unlabeled_dataset(n_features=2)
        model = _make_model(
            dataset,
            task="anomaly_detection",
            n_features=2,
            covariate_mode="concat",
        )
        self.assertTrue(
            self._backward_flows(model, _make_batch(dataset))
        )


# ------------------------------------------------------------------ #
# Synthetic example end-to-end test
# ------------------------------------------------------------------ #


class TestSyntheticEndToEnd(unittest.TestCase):
    """End-to-end test with synthetic data through PyHealth pipeline."""

    def test_full_pipeline(self):
        """Create dataset -> split -> model -> train step -> eval."""
        from pyhealth.datasets import (
            create_sample_dataset,
            get_dataloader,
            split_by_patient,
        )
        from pyhealth.models.medtsllm import MedTsLLM

        # Create synthetic samples
        samples = []
        for i in range(6):
            samples.append({
                "patient_id": f"p{i}",
                "visit_id": "v0",
                "signal": np.random.randn(128).astype(np.float32),
                "label": np.random.randint(0, 4, 128).astype(np.int64),
            })

        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"signal": "tensor"},
            output_schema={"label": "tensor"},
            dataset_name="e2e_test",
        )

        # Split
        train_ds, _, test_ds = split_by_patient(
            dataset, ratios=[0.7, 0.0, 0.3]
        )

        # Model
        model = MedTsLLM(
            dataset=dataset,
            seq_len=128,
            n_classes=4,
            backbone=None,
            word_embeddings=torch.randn(50, 32),
            d_model=8,
            d_ff=16,
            n_heads=2,
            num_tokens=10,
        )

        # Train one step
        loader = get_dataloader(train_ds, batch_size=2, shuffle=True)
        batch = next(iter(loader))
        out = model(**batch)
        self.assertIn("loss", out)
        self.assertTrue(out["loss"].isfinite())
        out["loss"].backward()

        # Eval
        model.eval()
        test_loader = get_dataloader(test_ds, batch_size=2, shuffle=False)
        with torch.no_grad():
            for batch in test_loader:
                out = model(**batch)
                self.assertIn("y_prob", out)
                break


# ------------------------------------------------------------------ #
# Error handling tests
# ------------------------------------------------------------------ #


class TestMedTsLLMInvalidInputs(unittest.TestCase):
    """Tests for error handling."""

    def test_no_backbone_or_embeddings(self):
        """Raises ValueError when neither backbone nor embeddings given."""
        from pyhealth.models.medtsllm import MedTsLLM

        dataset = _make_dataset()
        with self.assertRaises(ValueError):
            MedTsLLM(
                dataset=dataset,
                backbone=None,
                word_embeddings=None,
            )


# ------------------------------------------------------------------ #
# Preprocessing cache (_medtsllm_cache) — shared by LUDB / MIT-BIH /
# BIDMC loaders. Lives here rather than a separate file because it
# only exists to serve the MedTsLLM port.
# ------------------------------------------------------------------ #


class TestComputeFingerprint(unittest.TestCase):
    """Fingerprint is a stable hash of (raw file stats, params)."""

    def test_same_inputs_same_fingerprint(self):
        from pyhealth.datasets._medtsllm_cache import compute_fingerprint

        with tempfile.TemporaryDirectory() as tmp:
            raw = os.path.join(tmp, "a.dat")
            open(raw, "w").write("hi")
            fp1 = compute_fingerprint([raw], {"trim": True})
            fp2 = compute_fingerprint([raw], {"trim": True})
            self.assertEqual(fp1, fp2)

    def test_different_params_different_fingerprint(self):
        from pyhealth.datasets._medtsllm_cache import compute_fingerprint

        with tempfile.TemporaryDirectory() as tmp:
            raw = os.path.join(tmp, "a.dat")
            open(raw, "w").write("hi")
            fp1 = compute_fingerprint([raw], {"trim": True})
            fp2 = compute_fingerprint([raw], {"trim": False})
            self.assertNotEqual(fp1, fp2)

    def test_changed_file_changes_fingerprint(self):
        from pyhealth.datasets._medtsllm_cache import compute_fingerprint

        with tempfile.TemporaryDirectory() as tmp:
            raw = os.path.join(tmp, "a.dat")
            open(raw, "w").write("hi")
            fp1 = compute_fingerprint([raw], {})
            open(raw, "w").write("hello world now longer")
            fp2 = compute_fingerprint([raw], {})
            self.assertNotEqual(fp1, fp2)

    def test_returns_string(self):
        from pyhealth.datasets._medtsllm_cache import compute_fingerprint

        with tempfile.TemporaryDirectory() as tmp:
            raw = os.path.join(tmp, "a.dat")
            open(raw, "w").write("hi")
            fp = compute_fingerprint([raw], {})
            self.assertIsInstance(fp, str)
            self.assertGreater(len(fp), 16)


class TestLoadOrBuild(unittest.TestCase):
    """load_or_build skips the builder when the cache is warm."""

    def test_first_call_invokes_builder(self):
        from pyhealth.datasets._medtsllm_cache import load_or_build

        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "c.npz")
            calls = {"n": 0}

            def builder():
                calls["n"] += 1
                return {"x": np.arange(5, dtype=np.int64)}

            result = load_or_build(cache, "fp1", builder)
            self.assertEqual(calls["n"], 1)
            np.testing.assert_array_equal(result["x"], np.arange(5))
            self.assertTrue(os.path.exists(cache))

    def test_second_call_skips_builder(self):
        from pyhealth.datasets._medtsllm_cache import load_or_build

        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "c.npz")
            calls = {"n": 0}

            def builder():
                calls["n"] += 1
                return {"x": np.arange(5, dtype=np.int64)}

            load_or_build(cache, "fp1", builder)
            load_or_build(cache, "fp1", builder)
            self.assertEqual(calls["n"], 1)

    def test_fingerprint_mismatch_rebuilds(self):
        from pyhealth.datasets._medtsllm_cache import load_or_build

        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "c.npz")
            calls = {"n": 0}

            def builder():
                calls["n"] += 1
                return {"x": np.full(3, calls["n"], dtype=np.int64)}

            load_or_build(cache, "fp-old", builder)
            second = load_or_build(cache, "fp-new", builder)
            self.assertEqual(calls["n"], 2)
            np.testing.assert_array_equal(second["x"], np.full(3, 2))

    def test_creates_parent_dirs(self):
        from pyhealth.datasets._medtsllm_cache import load_or_build

        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "nested", "dir", "c.npz")

            def builder():
                return {"x": np.zeros(1, dtype=np.int64)}

            load_or_build(cache, "fp", builder)
            self.assertTrue(os.path.exists(cache))

    def test_preserves_string_arrays(self):
        """Cache round-trips unicode arrays (wfdb annotation symbols)."""
        from pyhealth.datasets._medtsllm_cache import load_or_build

        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "c.npz")

            def builder():
                return {
                    "signal": np.zeros((4, 2), dtype=np.float32),
                    "symbols": np.array(["N", "V", "L", "A"]),
                }

            load_or_build(cache, "fp", builder)

            # Second call hits cache
            result = load_or_build(cache, "fp", lambda: None)  # type: ignore[arg-type]
            self.assertEqual(list(result["symbols"]), ["N", "V", "L", "A"])

    def test_corrupt_cache_rebuilds(self):
        from pyhealth.datasets._medtsllm_cache import load_or_build

        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "c.npz")
            with open(cache, "w") as f:
                f.write("not a real npz")

            calls = {"n": 0}

            def builder():
                calls["n"] += 1
                return {"x": np.arange(2, dtype=np.int64)}

            result = load_or_build(cache, "fp", builder)
            self.assertEqual(calls["n"], 1)
            np.testing.assert_array_equal(result["x"], np.arange(2))


if __name__ == "__main__":
    unittest.main()

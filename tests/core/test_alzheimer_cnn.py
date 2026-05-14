"""Unit tests for AlzheimerCNN, AlzheimerCNNViT, and AlzheimerCNNNormVariant.


Coverage per model:
    - Instantiation with default and custom hyperparameters
    - Forward pass produces the expected output dictionary
    - Output shapes match (batch_size, num_classes)
    - Loss is a scalar tensor with valid gradient
    - Backward pass populates gradients on all trainable parameters
    - Probability outputs sum to 1.0 across the class dimension (softmax)
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset

from pyhealth.models import (
    AlzheimerCNN,
    AlzheimerCNNNormVariant,
    AlzheimerCNNViT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dataset stand-in
# ─────────────────────────────────────────────────────────────────────────────

class _SyntheticOutputProcessor:
    """Minimal stand-in for PyHealth's output processor."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def size(self) -> int:
        return self.num_classes


class _SyntheticAlzheimerDataset(Dataset):
    """Synthetic dataset with the schema attributes BaseModel requires.

    Generates random (1, H, W) image tensors and integer labels in [0, num_classes).
    Used only for unit testing — no real medical data involved.
    """

    input_schema = {"image": "image"}
    output_schema = {"label": "multiclass"}

    def __init__(
        self,
        num_samples: int = 4,
        image_size: int = 128,
        num_classes: int = 4,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.output_processors = {
            "label": _SyntheticOutputProcessor(num_classes)
        }
        # Pre-generate to ensure reproducibility within a test
        torch.manual_seed(0)
        self._images = torch.randn(num_samples, 1, image_size, image_size)
        self._labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "image": self._images[idx],
            "label": int(self._labels[idx].item()),
        }


def _make_batch(
    batch_size: int = 2,
    image_size: int = 128,
    num_classes: int = 4,
) -> dict:
    """Create a synthetic batch dict of the form expected by the models' forward pass."""
    torch.manual_seed(42)
    return {
        "image": torch.randn(batch_size, 1, image_size, image_size),
        "label": torch.randint(0, num_classes, (batch_size,)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────

class TestAlzheimerCNN(unittest.TestCase):
    """Tests for the base AlzheimerCNN model."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_dir = Path(tempfile.mkdtemp(prefix="alzheimer_cnn_test_"))
        cls.dataset = _SyntheticAlzheimerDataset(num_samples=4, num_classes=4)
        cls.batch_size = 2
        cls.num_classes = 4

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.tmp_dir.exists():
            shutil.rmtree(cls.tmp_dir)

    def test_instantiation_with_defaults(self) -> None:
        """Model instantiates with default hyperparameters."""
        model = AlzheimerCNN(dataset=self.dataset)
        self.assertIsInstance(model, AlzheimerCNN)
        self.assertEqual(model.init_channels, 32)

    def test_instantiation_with_custom_hyperparameters(self) -> None:
        """Model accepts custom init_channels, classifier_hidden_dim, and dropout."""
        model = AlzheimerCNN(
            dataset=self.dataset,
            init_channels=16,
            classifier_hidden_dim=64,
            dropout=0.3,
        )
        self.assertEqual(model.init_channels, 16)
        self.assertEqual(model.classifier_hidden_dim, 64)

    def test_forward_returns_expected_keys(self) -> None:
        """Forward pass returns dict with loss, y_prob, y_true, and logit."""
        model = AlzheimerCNN(dataset=self.dataset)
        batch = _make_batch(self.batch_size, num_classes=self.num_classes)
        output = model(**batch)
        self.assertIn("loss", output)
        self.assertIn("y_prob", output)
        self.assertIn("y_true", output)
        self.assertIn("logit", output)

    def test_forward_output_shapes(self) -> None:
        """Logits and probabilities have shape (batch_size, num_classes)."""
        model = AlzheimerCNN(dataset=self.dataset)
        batch = _make_batch(self.batch_size, num_classes=self.num_classes)
        output = model(**batch)
        self.assertEqual(output["logit"].shape, (self.batch_size, self.num_classes))
        self.assertEqual(output["y_prob"].shape, (self.batch_size, self.num_classes))

    def test_softmax_probabilities_sum_to_one(self) -> None:
        """Predicted probabilities sum to 1.0 across the class dimension."""
        model = AlzheimerCNN(dataset=self.dataset)
        batch = _make_batch(self.batch_size, num_classes=self.num_classes)
        output = model(**batch)
        prob_sums = output["y_prob"].sum(dim=1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones(self.batch_size), atol=1e-5))

    def test_loss_is_scalar(self) -> None:
        """Loss is a scalar tensor."""
        model = AlzheimerCNN(dataset=self.dataset)
        batch = _make_batch(self.batch_size, num_classes=self.num_classes)
        output = model(**batch)
        self.assertEqual(output["loss"].ndim, 0)

    def test_gradient_flow(self) -> None:
        """Backward pass populates gradients on all trainable parameters."""
        model = AlzheimerCNN(dataset=self.dataset)
        batch = _make_batch(self.batch_size, num_classes=self.num_classes)
        output = model(**batch)
        output["loss"].backward()
        for name, param in model.named_parameters():
            if name == "_dummy_param":
                continue
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient on {name}")
                self.assertFalse(
                    torch.isnan(param.grad).any(),
                    f"NaN in gradient on {name}",
                )


class TestAlzheimerCNNViT(unittest.TestCase):
    """Tests for the CNN + Vision Transformer hybrid model."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_dir = Path(tempfile.mkdtemp(prefix="alzheimer_cnn_vit_test_"))
        cls.dataset = _SyntheticAlzheimerDataset(num_samples=4, num_classes=4)
        cls.batch_size = 2
        cls.num_classes = 4

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.tmp_dir.exists():
            shutil.rmtree(cls.tmp_dir)

    def test_instantiation_with_defaults(self) -> None:
        """Model instantiates with default hyperparameters."""
        model = AlzheimerCNNViT(dataset=self.dataset)
        self.assertIsInstance(model, AlzheimerCNNViT)
        self.assertEqual(model.embed_dim, 128)

    def test_instantiation_with_custom_hyperparameters(self) -> None:
        """Model accepts custom embed_dim, num_heads, and num_transformer_layers."""
        model = AlzheimerCNNViT(
            dataset=self.dataset,
            embed_dim=64,
            num_heads=2,
            num_transformer_layers=2,
        )
        self.assertEqual(model.embed_dim, 64)

    def test_forward_output_shapes(self) -> None:
        """Logits and probabilities have shape (batch_size, num_classes)."""
        model = AlzheimerCNNViT(
            dataset=self.dataset,
            embed_dim=64,
            num_heads=2,
            num_transformer_layers=2,
        )
        batch = _make_batch(self.batch_size, num_classes=self.num_classes)
        output = model(**batch)
        self.assertEqual(output["logit"].shape, (self.batch_size, self.num_classes))
        self.assertEqual(output["y_prob"].shape, (self.batch_size, self.num_classes))

    def test_softmax_probabilities_sum_to_one(self) -> None:
        """Predicted probabilities sum to 1.0 across the class dimension."""
        model = AlzheimerCNNViT(
            dataset=self.dataset,
            embed_dim=64,
            num_heads=2,
            num_transformer_layers=2,
        )
        batch = _make_batch(self.batch_size, num_classes=self.num_classes)
        output = model(**batch)
        prob_sums = output["y_prob"].sum(dim=1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones(self.batch_size), atol=1e-5))

    def test_gradient_flow(self) -> None:
        """Backward pass populates gradients on all trainable parameters."""
        model = AlzheimerCNNViT(
            dataset=self.dataset,
            embed_dim=64,
            num_heads=2,
            num_transformer_layers=2,
        )
        batch = _make_batch(self.batch_size, num_classes=self.num_classes)
        output = model(**batch)
        output["loss"].backward()
        for name, param in model.named_parameters():
            if name == "_dummy_param":
                continue
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient on {name}")


class TestAlzheimerCNNNormVariant(unittest.TestCase):
    """Tests for the swappable-normalization variant."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_dir = Path(tempfile.mkdtemp(prefix="alzheimer_cnn_norm_test_"))
        cls.dataset = _SyntheticAlzheimerDataset(num_samples=4, num_classes=4)
        cls.batch_size = 2
        cls.num_classes = 4

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.tmp_dir.exists():
            shutil.rmtree(cls.tmp_dir)

    def test_instance_norm_variant(self) -> None:
        """norm_type='instance' instantiates and runs forward pass."""
        model = AlzheimerCNNNormVariant(dataset=self.dataset, norm_type="instance")
        self.assertEqual(model.norm_type, "instance")
        batch = _make_batch(self.batch_size, num_classes=self.num_classes)
        output = model(**batch)
        self.assertEqual(output["logit"].shape, (self.batch_size, self.num_classes))

    def test_group_norm_variant(self) -> None:
        """norm_type='group' instantiates and runs forward pass."""
        model = AlzheimerCNNNormVariant(
            dataset=self.dataset, norm_type="group", num_groups=8
        )
        self.assertEqual(model.norm_type, "group")
        batch = _make_batch(self.batch_size, num_classes=self.num_classes)
        output = model(**batch)
        self.assertEqual(output["logit"].shape, (self.batch_size, self.num_classes))

    def test_layer_norm_variant(self) -> None:
        """norm_type='layer' instantiates and runs forward pass."""
        model = AlzheimerCNNNormVariant(dataset=self.dataset, norm_type="layer")
        self.assertEqual(model.norm_type, "layer")
        batch = _make_batch(self.batch_size, num_classes=self.num_classes)
        output = model(**batch)
        self.assertEqual(output["logit"].shape, (self.batch_size, self.num_classes))

    def test_invalid_norm_type_raises(self) -> None:
        """Invalid norm_type raises ValueError."""
        with self.assertRaises(ValueError):
            AlzheimerCNNNormVariant(dataset=self.dataset, norm_type="batch")

    def test_gradient_flow_all_norm_types(self) -> None:
        """Gradient flows correctly for all valid norm types."""
        for norm_type in ("instance", "group", "layer"):
            with self.subTest(norm_type=norm_type):
                model = AlzheimerCNNNormVariant(
                    dataset=self.dataset, norm_type=norm_type
                )
                batch = _make_batch(self.batch_size, num_classes=self.num_classes)
                output = model(**batch)
                output["loss"].backward()
                for name, param in model.named_parameters():
                    if name == "_dummy_param":
                        continue
                    if param.requires_grad:
                        self.assertIsNotNone(
                            param.grad, f"No gradient on {name} ({norm_type})"
                        )


if __name__ == "__main__":
    unittest.main()
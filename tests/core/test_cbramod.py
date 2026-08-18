import unittest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import CBraMod_Wrapper


class TestCBraMod(unittest.TestCase):
    """Test cases for the CBraMod_Wrapper model."""

    def setUp(self):
        """Set up test data and model."""
        n_channels = 16
        patch_size = 200
        n_patches = 10
        n_samples = patch_size * n_patches  # 2000

        self.samples = [
            {
                "patient_id": f"patient-{i}",
                "visit_id": "visit-0",
                "signal": torch.randn(n_channels, n_samples).numpy().tolist(),
                "label": i % 6,
            }
            for i in range(4)
        ]

        self.input_schema = {"signal": "tensor"}
        self.output_schema = {"label": "multiclass"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test_cbramod",
        )

        # Use small model for fast testing
        self.model = CBraMod_Wrapper(
            dataset=self.dataset,
            in_dim=200,
            emb_size=200,
            dim_feedforward=800,
            seq_len=n_patches,
            n_layer=2,
            nhead=8,
            classifier_head=True,
            n_classes=6,
        )

    def test_model_initialization(self):
        """Test that the CBraMod_Wrapper model initializes correctly."""
        self.assertIsInstance(self.model, CBraMod_Wrapper)
        self.assertTrue(self.model.classifier_head)
        self.assertEqual(self.model.emb_size, 200)
        self.assertEqual(self.model.n_classes, 6)
        self.assertEqual(len(self.model.feature_keys), 1)
        self.assertIn("signal", self.model.feature_keys)

    def test_model_forward(self):
        """Test that the CBraMod_Wrapper forward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)
        self.assertIn("embeddings", ret)

        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["y_true"].shape[0], 2)
        self.assertEqual(ret["logit"].shape[0], 2)
        self.assertEqual(ret["logit"].shape[1], 6)  # n_classes
        self.assertEqual(ret["embeddings"].shape[0], 2)
        self.assertEqual(ret["embeddings"].shape[1], 200)  # emb_size
        self.assertEqual(ret["loss"].dim(), 0)

    def test_model_backward(self):
        """Test that the CBraMod_Wrapper backward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters have gradients after backward pass")

    def test_model_without_classifier(self):
        """Test CBraMod_Wrapper without classifier head (encoder only)."""
        model_no_cls = CBraMod_Wrapper(
            dataset=self.dataset,
            in_dim=200,
            emb_size=200,
            dim_feedforward=800,
            seq_len=10,
            n_layer=2,
            nhead=8,
            classifier_head=False,
            n_classes=6,
        )

        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model_no_cls(**data_batch)

        self.assertIn("logit", ret)
        self.assertIn("embeddings", ret)
        self.assertNotIn("loss", ret)
        self.assertNotIn("y_prob", ret)
        self.assertNotIn("y_true", ret)

    def test_model_different_batch_sizes(self):
        """Test CBraMod_Wrapper with different batch sizes."""
        for batch_size in [1, 2, 4]:
            train_loader = get_dataloader(self.dataset, batch_size=batch_size, shuffle=False)
            data_batch = next(iter(train_loader))

            with torch.no_grad():
                ret = self.model(**data_batch)

            actual_batch = min(batch_size, len(self.samples))
            self.assertEqual(ret["y_prob"].shape[0], actual_batch)
            self.assertEqual(ret["y_true"].shape[0], actual_batch)
            self.assertEqual(ret["logit"].shape[0], actual_batch)
            self.assertEqual(ret["embeddings"].shape[0], actual_batch)

    def test_model_output_probabilities(self):
        """Test that output probabilities are valid."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        y_prob = ret["y_prob"]
        self.assertTrue(torch.all(y_prob >= 0), "Probabilities contain negative values")
        self.assertTrue(torch.all(y_prob <= 1), "Probabilities exceed 1")

    def test_missing_signal_raises_error(self):
        """Test that missing 'signal' input raises ValueError."""
        with self.assertRaises((ValueError, KeyError)):
            self.model(label=torch.tensor([0, 1]))

    def test_model_different_n_classes(self):
        """Test CBraMod_Wrapper with different number of classes."""
        n_channels = 16
        patch_size = 200
        n_patches = 10
        n_samples = patch_size * n_patches

        binary_samples = [
            {
                "patient_id": f"patient-{i}",
                "visit_id": "visit-0",
                "signal": torch.randn(n_channels, n_samples).numpy().tolist(),
                "label": i % 2,
            }
            for i in range(4)
        ]

        binary_dataset = create_sample_dataset(
            samples=binary_samples,
            input_schema={"signal": "tensor"},
            output_schema={"label": "binary"},
            dataset_name="test_cbramod_binary",
        )

        model_binary = CBraMod_Wrapper(
            dataset=binary_dataset,
            in_dim=200,
            emb_size=200,
            dim_feedforward=800,
            seq_len=n_patches,
            n_layer=2,
            nhead=8,
            classifier_head=True,
            n_classes=1,
        )

        train_loader = get_dataloader(binary_dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model_binary(**data_batch)

        self.assertEqual(ret["logit"].shape[1], 1)

    def test_model_smaller_config(self):
        """Test CBraMod_Wrapper with a smaller configuration."""
        model_small = CBraMod_Wrapper(
            dataset=self.dataset,
            in_dim=200,
            emb_size=200,
            dim_feedforward=400,
            seq_len=10,
            n_layer=1,
            nhead=4,
            classifier_head=True,
            n_classes=6,
        )

        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model_small(**data_batch)

        self.assertIn("loss", ret)
        self.assertEqual(ret["logit"].shape[1], 6)
        self.assertEqual(ret["embeddings"].shape[1], 200)  # smaller emb_size

    def test_embedding_shape(self):
        """Test that embeddings have the correct shape."""
        train_loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertEqual(ret["embeddings"].shape, (4, 200))  # (batch, emb_size)


if __name__ == "__main__":
    unittest.main()
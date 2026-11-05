import unittest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import CNN, CNNLayer
from pyhealth.models.cnn import CNNBlock


class TestCNN(unittest.TestCase):
    """Test cases for the CNN model."""

    def setUp(self):
        """Set up test data and model."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86", "cond-80", "cond-12"],
                "labs": [[1.0, 2.5], [3.0, 4.0]],
                "label": 1,
            },
            {
                "patient_id": "patient-0",
                "visit_id": "visit-1",
                "conditions": ["cond-33", "cond-86", "cond-80"],
                "labs": [[0.5, 1.0], [1.2, 2.3]],
                "label": 0,
            },
        ]

        self.input_schema = {
            "conditions": "sequence",
            "labs": "tensor",
        }
        self.output_schema = {"label": "binary"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        self.model = CNN(dataset=self.dataset)

    def test_model_initialization(self):
        """Test that the CNN model initializes correctly."""
        self.assertIsInstance(self.model, CNN)
        self.assertEqual(self.model.embedding_dim, 128)
        self.assertEqual(self.model.hidden_dim, 128)
        self.assertEqual(self.model.num_layers, 1)
        self.assertEqual(len(self.model.feature_keys), 2)
        self.assertIn("conditions", self.model.feature_keys)
        self.assertIn("labs", self.model.feature_keys)
        self.assertEqual(self.model.label_key, "label")

    def test_model_forward(self):
        """Test that the CNN model forward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["y_true"].shape[0], 2)
        self.assertEqual(ret["logit"].shape[0], 2)
        self.assertEqual(ret["loss"].dim(), 0)

    def test_model_backward(self):
        """Test that the CNN model backward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters have gradients after backward pass")

    def test_model_with_embedding(self):
        """Test that the CNN model returns embeddings when requested."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        data_batch["embed"] = True

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape[0], 2)
        expected_embed_dim = len(self.model.feature_keys) * self.model.hidden_dim
        self.assertEqual(ret["embed"].shape[1], expected_embed_dim)

    def test_custom_hyperparameters(self):
        """Test CNN model with custom hyperparameters."""
        model = CNN(
            dataset=self.dataset,
            embedding_dim=64,
            hidden_dim=32,
            num_layers=2,
        )

        self.assertEqual(model.embedding_dim, 64)
        self.assertEqual(model.hidden_dim, 32)
        self.assertEqual(model.num_layers, 2)

        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

    def test_model_with_image_input(self):
        """Test CNN model with image input."""
        import os

        image_path = os.path.join(
            os.path.dirname(__file__), "../../test-resources/core/cameraman.tif"
        )

        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "image": image_path,
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "image": image_path,
                "label": 0,
            },
        ]

        input_schema = {"image": "image"}
        output_schema = {"label": "binary"}

        dataset = create_sample_dataset(
            samples=samples,
            input_schema=input_schema,
            output_schema=output_schema,
            dataset_name="test_image",
        )

        model = CNN(dataset=dataset)

        # Check that image feature has spatial_dim=2
        self.assertEqual(model.feature_conv_dims["image"], 2)

        train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["y_true"].shape[0], 2)
        self.assertEqual(ret["logit"].shape[0], 2)
        self.assertEqual(ret["loss"].dim(), 0)

    def test_model_with_mixed_inputs(self):
        """Test CNN model with mixed input types including image."""
        import os

        image_path = os.path.join(
            os.path.dirname(__file__), "../../test-resources/core/cameraman.tif"
        )

        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86"],
                "labs": [[1.0, 2.5]],
                "image": image_path,
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ["cond-33"],
                "labs": [[0.5, 1.0]],
                "image": image_path,
                "label": 0,
            },
        ]

        input_schema = {
            "conditions": "sequence",
            "labs": "tensor",
            "image": "image"
        }
        output_schema = {"label": "binary"}

        dataset = create_sample_dataset(
            samples=samples,
            input_schema=input_schema,
            output_schema=output_schema,
            dataset_name="test_mixed",
        )

        model = CNN(dataset=dataset)

        # Check spatial dimensions for each feature
        self.assertEqual(model.feature_conv_dims["conditions"], 1)  # sequence
        self.assertEqual(model.feature_conv_dims["labs"], 1)       # tensor
        self.assertEqual(model.feature_conv_dims["image"], 2)      # image

        train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["y_true"].shape[0], 2)
        self.assertEqual(ret["logit"].shape[0], 2)
        self.assertEqual(ret["loss"].dim(), 0)

    def test_model_with_multihot_and_1d_tensor_inputs(self):
        """Test CNN model with non-sequence spatial_dim=1 inputs.

        MultiHotProcessor and a TensorProcessor whose per-sample value is a 1D
        vector both embed to [batch, embedding_dim] with no sequence axis. These
        are documented as supported input types and must not crash.
        """
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "demographics": ["asian", "non_hispanic"],
                "vitals": [1.0, 2.5, 3.0],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "demographics": ["white"],
                "vitals": [0.5, 1.0, 2.0],
                "label": 0,
            },
        ]

        input_schema = {"demographics": "multi_hot", "vitals": "tensor"}
        output_schema = {"label": "binary"}

        dataset = create_sample_dataset(
            samples=samples,
            input_schema=input_schema,
            output_schema=output_schema,
            dataset_name="test_multihot",
        )

        model = CNN(dataset=dataset)
        self.assertEqual(model.feature_conv_dims["demographics"], 1)
        self.assertEqual(model.feature_conv_dims["vitals"], 1)

        train_loader = get_dataloader(dataset, batch_size=1, shuffle=False)
        data_batch = next(iter(train_loader))

        ret = model(**data_batch)
        ret["loss"].backward()

        self.assertEqual(ret["y_prob"].shape[0], 1)
        self.assertEqual(ret["logit"].shape[0], 1)
        self.assertEqual(ret["loss"].dim(), 0)
        self.assertTrue(torch.isfinite(ret["loss"]))
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                self.assertTrue(module.training)
                self.assertEqual(module.num_batches_tracked.item(), 0)
                self.assertIsNotNone(module.weight.grad)
                self.assertTrue(torch.isfinite(module.weight.grad).all())

    def test_size_arguments_are_rejected(self):
        """input_size and hidden_size come from embedding_dim and hidden_dim."""
        for name in ("input_size", "hidden_size"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                CNN(dataset=self.dataset, **{name: 8})


class TestCNNBlock(unittest.TestCase):
    """Batch normalization in CNNBlock with a single value per channel."""

    def test_singleton_batch_leaves_running_statistics_unchanged(self):
        block = CNNBlock(4, 8, spatial_dim=1)
        block.train()
        norms = [m for m in block.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        before = [(m.running_mean.clone(), m.running_var.clone()) for m in norms]

        x = torch.randn(1, 4, 1, requires_grad=True)
        block(x).sum().backward()

        self.assertIsNotNone(x.grad)
        for norm, (mean, var) in zip(norms, before):
            self.assertTrue(norm.training)
            self.assertTrue(torch.equal(norm.running_mean, mean))
            self.assertTrue(torch.equal(norm.running_var, var))
            self.assertEqual(norm.num_batches_tracked.item(), 0)

    def test_larger_batch_updates_running_statistics(self):
        block = CNNBlock(4, 8, spatial_dim=1)
        block.train()
        block(torch.randn(2, 4, 1))
        for norm in block.modules():
            if isinstance(norm, torch.nn.BatchNorm1d):
                self.assertEqual(norm.num_batches_tracked.item(), 1)

    def test_unsupported_spatial_dim_raises(self):
        with self.assertRaises(ValueError):
            CNNBlock(4, 8, spatial_dim=4)
        with self.assertRaises(ValueError):
            CNNLayer(4, 8, spatial_dim=4)


if __name__ == "__main__":
    unittest.main()

import unittest
import torch

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.models import CNN


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

        self.dataset = SampleDataset(
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

        dataset = SampleDataset(
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

        dataset = SampleDataset(
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


if __name__ == "__main__":
    unittest.main()

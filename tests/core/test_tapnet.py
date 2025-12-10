import unittest
import torch

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.models import TapNet


class TestTapNet(unittest.TestCase):
    """Test cases for the TapNet model."""

    def setUp(self):
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-1", "cond-2", "cond-3"],
                "labs": [[1.0, 2.0], [3.0, 4.0]],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "conditions": ["cond-2", "cond-4"],
                "labs": [[0.5, 1.5]],
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

        self.model = TapNet(dataset=self.dataset)

    def test_model_initialization(self):
        self.assertIsInstance(self.model, TapNet)
        self.assertEqual(self.model.embedding_dim, 128)
        self.assertEqual(self.model.hidden_dim, 128)
        self.assertEqual(len(self.model.feature_keys), 2)
        self.assertIn("conditions", self.model.feature_keys)
        self.assertIn("labs", self.model.feature_keys)
        self.assertEqual(self.model.label_key, "label")

    def test_model_forward(self):
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
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters have gradients after backward pass")

    def test_model_with_embedding_and_attention(self):
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        data_batch["embed"] = True
        data_batch["return_attn"] = True

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("embed", ret)
        expected_embed_dim = len(self.model.feature_keys) * self.model.hidden_dim * 2
        self.assertEqual(ret["embed"].shape, (2, expected_embed_dim))

        self.assertIn("prototype_attention", ret)
        for key in self.model.feature_keys:
            self.assertIn(key, ret["prototype_attention"])
            self.assertEqual(
                ret["prototype_attention"][key].shape[1], 8
            )  # default num_prototypes

    def test_custom_hyperparameters(self):
        model = TapNet(
            dataset=self.dataset,
            embedding_dim=64,
            hidden_dim=32,
            num_prototypes=4,
            kernel_size=5,
            dropout=0.2,
        )

        self.assertEqual(model.embedding_dim, 64)
        self.assertEqual(model.hidden_dim, 32)

        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertEqual(ret["logit"].shape, (2, 1))


if __name__ == "__main__":
    unittest.main()

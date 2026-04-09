import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import EBCL


class TestEBCL(unittest.TestCase):
    """Test cases for the EBCL model."""

    def setUp(self):
        self.paired_samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["A", "B", "C"],
                "labs": [1.0, 2.0, 3.0],
                "post_conditions": ["B", "C"],
                "post_labs": [1.5, 2.5, 3.5],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ["D", "E"],
                "labs": [0.5, 1.5, 2.5],
                "post_conditions": ["E", "F"],
                "post_labs": [1.0, 1.5, 2.0],
                "label": 0,
            },
        ]
        self.paired_dataset = create_sample_dataset(
            samples=self.paired_samples,
            input_schema={
                "conditions": "sequence",
                "labs": "tensor",
                "post_conditions": "sequence",
                "post_labs": "tensor",
            },
            output_schema={"label": "binary"},
            dataset_name="ebcl_paired",
        )

        self.supervised_samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["A", "B", "C"],
                "labs": [1.0, 2.0, 3.0],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ["D", "E"],
                "labs": [0.5, 1.5, 2.5],
                "label": 0,
            },
        ]
        self.supervised_dataset = create_sample_dataset(
            samples=self.supervised_samples,
            input_schema={
                "conditions": "sequence",
                "labs": "tensor",
            },
            output_schema={"label": "binary"},
            dataset_name="ebcl_supervised",
        )

    def test_model_initialization(self):
        model = EBCL(dataset=self.paired_dataset)
        self.assertIsInstance(model, EBCL)
        self.assertEqual(model.embedding_dim, 128)
        self.assertEqual(model.hidden_dim, 128)
        self.assertEqual(model.projection_dim, 128)
        self.assertEqual(model.label_key, "label")
        self.assertEqual(model.feature_keys, ["conditions", "labs"])
        self.assertEqual(model.post_feature_keys["conditions"], "post_conditions")
        self.assertEqual(model.post_feature_keys["labs"], "post_labs")

    def test_supervised_forward(self):
        model = EBCL(dataset=self.supervised_dataset)
        batch = next(iter(get_dataloader(self.supervised_dataset, batch_size=2, shuffle=False)))

        with torch.no_grad():
            ret = model(**batch)

        self.assertIn("loss", ret)
        self.assertIn("supervised_loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)
        self.assertNotIn("contrastive_loss", ret)
        self.assertEqual(ret["y_prob"].shape, (2, 1))
        self.assertEqual(ret["logit"].shape, (2, 1))
        self.assertEqual(ret["loss"].dim(), 0)

    def test_contrastive_forward(self):
        model = EBCL(dataset=self.paired_dataset)
        batch = next(iter(get_dataloader(self.paired_dataset, batch_size=2, shuffle=False)))

        with torch.no_grad():
            ret = model(**batch)

        self.assertIn("loss", ret)
        self.assertIn("supervised_loss", ret)
        self.assertIn("contrastive_loss", ret)
        self.assertIn("contrastive_logits", ret)
        self.assertIn("post_embed", ret)
        self.assertEqual(ret["contrastive_logits"].shape, (2, 2))
        self.assertEqual(ret["post_embed"].shape, (2, model.hidden_dim))

    def test_model_backward(self):
        model = EBCL(dataset=self.paired_dataset)
        batch = next(iter(get_dataloader(self.paired_dataset, batch_size=2, shuffle=False)))
        ret = model(**batch)
        ret["loss"].backward()

        has_gradient = any(
            parameter.requires_grad and parameter.grad is not None
            for parameter in model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters have gradients after backward pass")

    def test_model_with_embedding(self):
        model = EBCL(dataset=self.paired_dataset)
        batch = next(iter(get_dataloader(self.paired_dataset, batch_size=2, shuffle=False)))
        batch["embed"] = True

        with torch.no_grad():
            ret = model(**batch)

        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape, (2, model.hidden_dim))


if __name__ == "__main__":
    unittest.main()

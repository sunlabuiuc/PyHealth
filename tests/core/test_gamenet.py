import unittest
import torch

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.models import GAMENet


class TestGAMENet(unittest.TestCase):

    def setUp(self):
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": [["cond-33", "cond-86"], ["cond-80", "cond-12"]],
                "procedures": [["proc-45", "proc-23"], ["proc-67"]],
                "drugs": ["drug-1", "drug-2", "drug-3"],
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": [["cond-33"], ["cond-80"]],
                "procedures": [["proc-45"], ["proc-23", "proc-67"]],
                "drugs": ["drug-2", "drug-4"],
            },
            {
                "patient_id": "patient-2",
                "visit_id": "visit-2",
                "conditions": [["cond-86", "cond-80"], ["cond-12"]],
                "procedures": [["proc-45", "proc-67"], ["proc-23"]],
                "drugs": ["drug-1", "drug-4", "drug-5"],
            },
        ]

        self.input_schema = {
            "conditions": "nested_sequence",
            "procedures": "nested_sequence",
        }
        self.output_schema = {"drugs": "multilabel"}

        self.dataset = SampleDataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        self.model = GAMENet(dataset=self.dataset, embedding_dim=64, hidden_dim=64)

    def test_model_initialization(self):
        self.assertIsInstance(self.model, GAMENet)
        self.assertEqual(self.model.embedding_dim, 64)
        self.assertEqual(self.model.hidden_dim, 64)
        self.assertEqual(self.model.num_layers, 1)
        self.assertEqual(len(self.model.feature_keys), 2)
        self.assertIn("conditions", self.model.feature_keys)
        self.assertIn("procedures", self.model.feature_keys)
        self.assertEqual(self.model.label_key, "drugs")

    def test_forward_input_format(self):
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        self.assertIn("conditions", data_batch)
        self.assertIn("procedures", data_batch)
        self.assertIn("drugs", data_batch)

        self.assertEqual(len(data_batch["conditions"].shape), 3)
        self.assertEqual(len(data_batch["procedures"].shape), 3)
        self.assertEqual(len(data_batch["drugs"].shape), 2)

    def test_model_forward(self):
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)

        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["y_true"].shape[0], 2)

        self.assertEqual(ret["loss"].dim(), 0)

    def test_model_backward(self):
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)

        ret["loss"].backward()

        has_gradient = False
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                has_gradient = True
                break
        self.assertTrue(
            has_gradient, "No parameters have gradients after backward pass"
        )

    def test_loss_is_finite(self):
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)

        self.assertTrue(torch.isfinite(ret["loss"]).all())
        self.assertFalse(torch.isnan(ret["loss"]).any())
        self.assertFalse(torch.isinf(ret["loss"]).any())

    def test_output_shapes(self):
        train_loader = get_dataloader(self.dataset, batch_size=3, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        batch_size = data_batch["drugs"].shape[0]
        num_drugs = data_batch["drugs"].shape[1]

        self.assertEqual(ret["y_prob"].shape, (batch_size, num_drugs))
        self.assertEqual(ret["y_true"].shape, (batch_size, num_drugs))
        self.assertEqual(ret["loss"].shape, ())


if __name__ == "__main__":
    unittest.main()


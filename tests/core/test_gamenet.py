import unittest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import GAMENet


class TestGAMENet(unittest.TestCase):

    def setUp(self):
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": [["cond-33", "cond-86"], ["cond-80", "cond-12"]],
                "procedures": [["proc-45", "proc-23"], ["proc-67"]],
                # drugs_hist: per-visit drugs actually administered so far,
                # with the current (target) visit already zeroed out, as
                # produced by pyhealth.tasks.drug_recommendation.
                "drugs_hist": [["drug-2"], []],
                "drugs": ["drug-1", "drug-2", "drug-3"],
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": [["cond-33"], ["cond-80"]],
                "procedures": [["proc-45"], ["proc-23", "proc-67"]],
                "drugs_hist": [["drug-4"], []],
                "drugs": ["drug-2", "drug-4"],
            },
            {
                "patient_id": "patient-2",
                "visit_id": "visit-2",
                "conditions": [["cond-86", "cond-80"], ["cond-12"]],
                "procedures": [["proc-45", "proc-67"], ["proc-23"]],
                "drugs_hist": [["drug-5", "drug-1"], []],
                "drugs": ["drug-1", "drug-4", "drug-5"],
            },
        ]

        self.input_schema = {
            "conditions": "nested_sequence",
            "procedures": "nested_sequence",
            "drugs_hist": "nested_sequence",
        }
        self.output_schema = {"drugs": "multilabel"}

        self.dataset = create_sample_dataset(
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
        self.assertIn("drugs_hist", data_batch)
        self.assertIn("drugs", data_batch)

        self.assertEqual(len(data_batch["conditions"].shape), 3)
        self.assertEqual(len(data_batch["procedures"].shape), 3)
        self.assertEqual(len(data_batch["drugs_hist"].shape), 3)
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
        """Test that the loss is finite."""
        torch.manual_seed(42)  # reproducibility: shuffle + dropout can rarely yield non-finite loss
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertTrue(torch.isfinite(ret["loss"]).all())

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


    def test_missing_drugs_hist_raises(self):
        """Regression test: constructing GAMENet without drugs_hist in the
        input_schema must fail loudly, not silently fall back to zeroed
        history (the original bug)."""
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": [["cond-33"], ["cond-80"]],
                "procedures": [["proc-45"], ["proc-67"]],
                "drugs": ["drug-1", "drug-2"],
            },
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"conditions": "nested_sequence", "procedures": "nested_sequence"},
            output_schema={"drugs": "multilabel"},
            dataset_name="test_missing_hist",
        )
        with self.assertRaises(AssertionError):
            GAMENet(dataset=dataset)

    def test_dynamic_memory_uses_real_drug_history(self):
        """Regression test for the critical bug: the Dynamic Memory's
        values (prev_drugs, Eq. 6 of the GAMENet paper) must be populated
        from each patient's actual drugs_hist, not hardcoded zeros."""
        train_loader = get_dataloader(self.dataset, batch_size=3, shuffle=False)
        data_batch = next(iter(train_loader))

        drugs_hist = data_batch["drugs_hist"]
        prev_drugs = self.model._build_prev_drugs(drugs_hist.to(self.model.device))

        # Every sample in this test set has non-empty history at visit 0
        # (patient-0: drug-2, patient-1: drug-4, patient-2: drug-5/drug-1),
        # so the resulting multi-hot tensor must NOT be all zeros.
        self.assertGreater(prev_drugs.sum().item(), 0)

        # The visit-0 row for patient-0 should have exactly one drug
        # ("drug-2") marked, at the index the label_vocab assigns it.
        label_vocab = self.model.dataset.output_processors["drugs"].label_vocab
        self.assertEqual(prev_drugs[0, 0].sum().item(), 1.0)
        self.assertEqual(prev_drugs[0, 0, label_vocab["drug-2"]].item(), 1.0)

        # The current (target) visit's history was zeroed out by the task
        # convention, so its row must be all zeros.
        self.assertEqual(prev_drugs[0, 1].sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()


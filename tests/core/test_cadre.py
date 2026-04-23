import unittest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import CADRE


class TestCADRE(unittest.TestCase):
    """Test cases for the CADRE model."""

    def setUp(self):
        self.num_drugs = 2
        self.num_genes = 20

        # gene_idx values must be integers; 0 reserved for padding
        self.samples = [
            {
                "patient_id": "patient-0",
                "gene_idx": [1, 2, 3, 4, 0, 0],
                "label": [0,1],
            },
            {
                "patient_id": "patient-1",
                "gene_idx": [5, 6, 7, 8, 9, 0],
                "label": [0,1],
            },
        ]

        self.input_schema = {
            "gene_idx": "sequence",
        }

        self.output_schema = {
            "label": "multilabel",
        }

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="cadre_test",
        )

        self.model = CADRE(
            dataset=self.dataset,
            feature_key="gene_idx",
            label_key="label",
            num_genes=self.num_genes,
            num_drugs=self.num_drugs,
            embedding_dim=16,
            hidden_dim=16,
            attention_size=8,
            attention_head=2,
            dropout=0.1,
        )

    def test_model_initialization(self):
        self.assertIsInstance(self.model, CADRE)
        self.assertEqual(self.model.num_genes, self.num_genes)
        self.assertEqual(self.model.num_drugs, self.num_drugs)

    def test_forward_input_format(self):
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        self.assertIsInstance(batch["gene_idx"], torch.Tensor)
        self.assertIsInstance(batch["label"], torch.Tensor)

    def test_model_forward(self):
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = self.model(**batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

        self.assertEqual(ret["logit"].shape, (2, self.num_drugs))
        self.assertEqual(ret["y_prob"].shape, (2, self.num_drugs))
        self.assertEqual(ret["y_true"].shape, (2, self.num_drugs))
        self.assertEqual(ret["loss"].dim(), 0)

    def test_model_backward(self):
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        ret = self.model(**batch)
        ret["loss"].backward()

        has_grad = any(
            p.requires_grad and p.grad is not None
            for p in self.model.parameters()
        )
        self.assertTrue(has_grad)

    def test_loss_is_finite(self):
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = self.model(**batch)

        self.assertTrue(torch.isfinite(ret["loss"]).all())


if __name__ == "__main__":
    unittest.main()

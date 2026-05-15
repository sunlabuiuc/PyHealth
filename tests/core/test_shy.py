import unittest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import SHy


class TestSHyModel(unittest.TestCase):
    """Tests for the SHy model."""

    def setUp(self):
        self.samples = [
            {
                "patient_id": "p0",
                "diagnoses_hist": [["d1", "d2", "d3"], ["d1", "d4"]],
                "diagnoses": ["d1", "d2"],
            },
            {
                "patient_id": "p1",
                "diagnoses_hist": [["d2", "d3"], ["d4", "d5"], ["d1", "d6"]],
                "diagnoses": ["d3", "d4", "d5"],
            },
            {
                "patient_id": "p2",
                "diagnoses_hist": [["d1", "d6"]],
                "diagnoses": ["d2", "d6"],
            },
            {
                "patient_id": "p3",
                "diagnoses_hist": [["d3", "d4"], ["d5", "d6"]],
                "diagnoses": ["d1"],
            },
        ]

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={"diagnoses_hist": "nested_sequence"},
            output_schema={"diagnoses": "multilabel"},
            dataset_name="test_shy",
        )

        self.model = SHy(
            dataset=self.dataset,
            embedding_dim=16,
            hgnn_dim=16,
            hgnn_layers=1,
            num_tp=2,
            hidden_dim=16,
            num_heads=2,
            dropout=0.0,
        )

    def test_initialization(self):
        """Check model sets up the right keys and params."""
        self.assertIsInstance(self.model, SHy)
        self.assertEqual(self.model.feature_key, "diagnoses_hist")
        self.assertEqual(self.model.label_key, "diagnoses")
        self.assertEqual(self.model.num_tp, 2)

    def test_forward_output_keys(self):
        """Forward pass should return loss, y_prob, y_true, logit."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            out = self.model(**batch)

        self.assertIn("loss", out)
        self.assertIn("y_prob", out)
        self.assertIn("y_true", out)
        self.assertIn("logit", out)
        # loss should be a scalar
        self.assertEqual(out["loss"].dim(), 0)

    def test_output_shapes(self):
        """y_prob and y_true should match (batch, num_labels)."""
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            out = self.model(**batch)

        self.assertEqual(out["y_prob"].shape[1], self.model.output_size)
        self.assertEqual(out["y_true"].shape[1], self.model.output_size)

    def test_backward(self):
        """Make sure gradients flow through the model."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        out = self.model(**batch)
        out["loss"].backward()

        # at least one param should have a gradient
        got_grad = False
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                got_grad = True
                break
        self.assertTrue(got_grad, "backward didn't produce any gradients")

    def test_probabilities_in_range(self):
        """Predicted probs should all be between 0 and 1."""
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            out = self.model(**batch)

        self.assertTrue(torch.all(out["y_prob"] >= 0))
        self.assertTrue(torch.all(out["y_prob"] <= 1))

    def test_loss_not_nan(self):
        """Loss shouldn't be NaN on normal input."""
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            out = self.model(**batch)

        self.assertFalse(torch.isnan(out["loss"]))

    def test_single_phenotype(self):
        """num_tp=1 should still work (no distinctness loss)."""
        model = SHy(
            dataset=self.dataset,
            embedding_dim=16,
            hgnn_dim=16,
            hgnn_layers=1,
            num_tp=1,
            hidden_dim=16,
            num_heads=2,
        )
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            out = model(**batch)

        self.assertFalse(torch.isnan(out["loss"]))
        self.assertEqual(out["y_prob"].shape[0], 2)

    def test_no_hgnn(self):
        """hgnn_layers=0 should fall back to linear projection."""
        model = SHy(
            dataset=self.dataset,
            embedding_dim=16,
            hgnn_dim=16,
            hgnn_layers=0,
            num_tp=2,
            hidden_dim=16,
            num_heads=2,
        )
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            out = model(**batch)

        self.assertIn("loss", out)
        self.assertFalse(torch.isnan(out["loss"]))

    def test_custom_hyperparams(self):
        """Different embedding/hidden sizes should still run."""
        model = SHy(
            dataset=self.dataset,
            embedding_dim=8,
            hgnn_dim=32,
            hgnn_layers=2,
            num_tp=3,
            hidden_dim=32,
            num_heads=4,
            dropout=0.2,
        )
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            out = model(**batch)

        self.assertIn("loss", out)
        self.assertIn("y_prob", out)

    def test_incidence_matrix(self):
        """_build_incidence_matrix should give a valid binary matrix."""
        # fake padded codes: 2 visits, codes padded to length 3
        codes = torch.tensor([[1, 2, 0], [3, 0, 0]])
        H = self.model._build_incidence_matrix(codes)

        # rows = vocab size, cols = num visits
        self.assertEqual(H.shape[0], self.model.vocab_size)
        self.assertEqual(H.shape[1], 2)

        # should only contain 0s and 1s
        self.assertTrue(torch.all((H == 0) | (H == 1)))

        # code 1 should be in visit 0, code 3 should be in visit 1
        self.assertEqual(H[1, 0].item(), 1.0)
        self.assertEqual(H[3, 1].item(), 1.0)
        # padding index 0 should not appear
        self.assertEqual(H[0, 0].item(), 0.0)

    def test_phenotype_extractor_produces_k_subhypergraphs(self):
        """For num_tp=K, the model should produce K phenotype matrices."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        feature_data = batch[self.model.feature_key]
        codes_batch = (
            feature_data[0] if isinstance(feature_data, tuple) else feature_data
        )
        X = self.model.code_embedding(torch.arange(self.model.vocab_size))
        H = self.model._build_incidence_matrix(codes_batch[0])
        tp_mats, tp_embs = self.model._encode_patient(X, H)
        self.assertEqual(len(tp_mats), self.model.num_tp)
        for mat in tp_mats:
            self.assertEqual(mat.shape, H.shape)

    def test_different_num_tp_gives_different_outputs(self):
        """num_tp=1 and num_tp=3 should produce different losses."""
        torch.manual_seed(123)
        model1 = SHy(
            dataset=self.dataset,
            embedding_dim=16,
            hgnn_dim=16,
            hgnn_layers=1,
            num_tp=1,
            hidden_dim=16,
            num_heads=2,
            dropout=0.0,
        )
        torch.manual_seed(123)
        model3 = SHy(
            dataset=self.dataset,
            embedding_dim=16,
            hgnn_dim=16,
            hgnn_layers=1,
            num_tp=3,
            hidden_dim=16,
            num_heads=2,
            dropout=0.0,
        )
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        with torch.no_grad():
            out1 = model1(**batch)
            out3 = model3(**batch)
        self.assertNotEqual(out1["loss"].item(), out3["loss"].item())

    def test_add_false_negatives_changes_incidence(self):
        """add_ratio > 0 should add entries to the incidence matrix."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        feature_data = batch[self.model.feature_key]
        codes_batch = (
            feature_data[0] if isinstance(feature_data, tuple) else feature_data
        )
        X = self.model.code_embedding(torch.arange(self.model.vocab_size))
        H = self.model._build_incidence_matrix(codes_batch[0])
        nz = torch.nonzero(H)
        V, E = nz[:, 0], nz[:, 1]
        X_personal = self.model._run_hgnn(X, V, E)
        ext = self.model.extractors[0]
        enriched = ext._add_false_negatives(X_personal, H, V, E)
        # Enriched should have at least as many nonzeros as original
        self.assertGreaterEqual(enriched.sum().item(), H.sum().item())


class TestDiagnosisPredictionTask(unittest.TestCase):
    """Tests for the diagnosis prediction task classes."""

    def test_mimic3_task_schema(self):
        from pyhealth.tasks import DiagnosisPredictionMIMIC3

        task = DiagnosisPredictionMIMIC3()
        self.assertEqual(task.task_name, "DiagnosisPredictionMIMIC3")
        self.assertIn("diagnoses_hist", task.input_schema)
        self.assertEqual(task.input_schema["diagnoses_hist"], "nested_sequence")
        self.assertIn("diagnoses", task.output_schema)
        self.assertEqual(task.output_schema["diagnoses"], "multilabel")

    def test_mimic4_task_schema(self):
        from pyhealth.tasks import DiagnosisPredictionMIMIC4

        task = DiagnosisPredictionMIMIC4()
        self.assertEqual(task.task_name, "DiagnosisPredictionMIMIC4")
        self.assertIn("diagnoses_hist", task.input_schema)
        self.assertEqual(task.input_schema["diagnoses_hist"], "nested_sequence")
        self.assertIn("diagnoses", task.output_schema)
        self.assertEqual(task.output_schema["diagnoses"], "multilabel")


if __name__ == "__main__":
    unittest.main()

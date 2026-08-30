import unittest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import GAMENet, GAMENetLayer


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


class TestGAMENetLayerDynamicMemoryMasking(unittest.TestCase):
    """Regression tests for GAMENetLayer.forward()'s dynamic-memory
    attention over variable-length (padded) batches.

    Naively slicing DM_keys/DM_values as queries[:, :-1, :] and
    prev_drugs[:, :-1, :] only drops the batch's last column. For a
    patient shorter than the batch's longest sequence, that leaves the
    patient's own current-visit position -- and any padding beyond it --
    inside the attention pool, stealing softmax weight from that
    patient's genuine previous visits. Fixed by masking those positions
    to -inf before the softmax.
    """

    def _make_layer(self, hidden_size, num_drugs, seed=0):
        torch.manual_seed(seed)
        ehr_adj = torch.randint(0, 2, (num_drugs, num_drugs)).float()
        ddi_adj = torch.randint(0, 2, (num_drugs, num_drugs)).float()
        layer = GAMENetLayer(hidden_size, ehr_adj, ddi_adj)
        layer.eval()
        return layer

    def test_dynamic_memory_ignores_own_current_visit_and_padding(self):
        """Perturbing prev_drugs at a shorter patient's own current-visit
        slot and at padding positions -- both of which fall inside the
        naively-sliced DM_values range -- must not change that patient's
        output at all: those positions must receive exactly zero dynamic-
        memory attention weight. This would fail under the pre-fix
        behavior, where nothing masks those positions out of the softmax.
        """
        hidden_size, num_drugs, num_visits = 8, 5, 4
        layer = self._make_layer(hidden_size, num_drugs)

        torch.manual_seed(42)
        queries = torch.randn(2, num_visits, hidden_size)
        prev_drugs = torch.zeros(2, num_visits, num_drugs)
        # Patient 0: 4 valid visits (mask all-ones), real history at 0,1,2.
        prev_drugs[0, 0] = torch.tensor([1.0, 0, 0, 0, 0])
        prev_drugs[0, 1] = torch.tensor([0.0, 1, 0, 0, 0])
        prev_drugs[0, 2] = torch.tensor([0.0, 0, 1, 0, 0])
        # Patient 1: only 2 valid visits (current visit = index 1); one
        # genuine prior visit at index 0.
        prev_drugs[1, 0] = torch.tensor([0.0, 0, 0, 1, 0])

        curr_drugs = torch.randint(0, 2, (2, num_drugs)).float()
        mask = torch.tensor([[1.0, 1, 1, 1], [1.0, 1, 0, 0]])

        with torch.no_grad():
            loss_base, y_prob_base = layer(queries, prev_drugs, curr_drugs, mask)

        prev_drugs_perturbed = prev_drugs.clone()
        # Patient 1's own current-visit slot (index 1) and both padding
        # slots (2, 3) -- all of which should be masked out.
        prev_drugs_perturbed[1, 1] = torch.tensor([1.0, 1, 1, 1, 1])
        prev_drugs_perturbed[1, 2] = torch.tensor([1.0, 1, 1, 1, 1])
        prev_drugs_perturbed[1, 3] = torch.tensor([1.0, 1, 1, 1, 1])

        with torch.no_grad():
            loss_pert, y_prob_pert = layer(queries, prev_drugs_perturbed, curr_drugs, mask)

        torch.testing.assert_close(y_prob_base[1], y_prob_pert[1])
        # Patient 0 (unaffected batch row, full-length sequence) must also
        # be exactly unchanged.
        torch.testing.assert_close(y_prob_base[0], y_prob_pert[0])
        torch.testing.assert_close(loss_base, loss_pert)

    def test_dynamic_memory_zero_for_first_visit_patient(self):
        """A patient whose current visit is their very first visit has
        zero valid previous visits -- the attention row is all -inf
        pre-softmax. This must resolve to a zero dynamic-memory
        contribution, not NaN propagating into the output."""
        hidden_size, num_drugs, num_visits = 8, 5, 3
        layer = self._make_layer(hidden_size, num_drugs, seed=1)

        torch.manual_seed(2)
        queries = torch.randn(1, num_visits, hidden_size)
        prev_drugs = torch.zeros(1, num_visits, num_drugs)
        curr_drugs = torch.randint(0, 2, (1, num_drugs)).float()
        # Single valid visit (the patient's first-ever visit); the rest is
        # padding.
        mask = torch.tensor([[1.0, 0, 0]])

        with torch.no_grad():
            loss, y_prob = layer(queries, prev_drugs, curr_drugs, mask)

        self.assertFalse(torch.isnan(y_prob).any())
        self.assertFalse(torch.isnan(loss).any())

    def test_full_length_patient_unaffected_by_masking_fix(self):
        """Sanity check: for a patient whose sequence spans the batch's
        full padded length (no padding, current visit is the batch's last
        column), the mask covers exactly the same previous-visit positions
        as the original unmasked slice -- output must be identical to a
        run with mask=None (the pre-fix default)."""
        hidden_size, num_drugs, num_visits = 8, 5, 4
        layer = self._make_layer(hidden_size, num_drugs, seed=3)

        torch.manual_seed(4)
        queries = torch.randn(2, num_visits, hidden_size)
        prev_drugs = torch.randint(0, 2, (2, num_visits, num_drugs)).float()
        curr_drugs = torch.randint(0, 2, (2, num_drugs)).float()
        mask_all_ones = torch.ones(2, num_visits)

        with torch.no_grad():
            loss_default, y_prob_default = layer(queries, prev_drugs, curr_drugs, mask=None)
            loss_explicit, y_prob_explicit = layer(
                queries, prev_drugs, curr_drugs, mask=mask_all_ones
            )

        torch.testing.assert_close(y_prob_default, y_prob_explicit)
        torch.testing.assert_close(loss_default, loss_explicit)


if __name__ == "__main__":
    unittest.main()


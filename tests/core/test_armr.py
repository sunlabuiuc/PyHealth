import math
import unittest
import warnings

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import ARMR
from pyhealth.models.armr import (
    PiecewiseTemporalLearning,
    _reverse_padded,
    _split_reused_new,
)
from pyhealth.models.utils import (
    ddi_adjacency,
    ehr_cooccurrence_adjacency,
    gcn_normalize_adjacency,
    history_to_label_index,
)
from pyhealth.trainer import Trainer

SCHEMA = {
    "conditions": "nested_sequence",
    "procedures": "nested_sequence",
    "drugs_hist": "nested_sequence",
}


def _samples():
    return [
        {"patient_id": "p0", "visit_id": "v0",
         "conditions": [["d1", "d2"], ["d3"]],
         "procedures": [["p1"], ["p2"]],
         "drugs_hist": [["A01A"], []],
         "drugs": ["A01A", "B01A"]},
        # a long history: more visits than n_recent, so the Mamba branch runs
        {"patient_id": "p1", "visit_id": "v1",
         "conditions": [["d2"], ["d1"], ["d3"], ["d1"], ["d2"]],
         "procedures": [["p2"], ["p1"], ["p1"], ["p2"], ["p1"]],
         "drugs_hist": [["B01A"], ["C03C", "A01A"], ["B01A"], ["A01A"], []],
         "drugs": ["C03C", "N02B"]},
        # a single visit: no drug history at all
        {"patient_id": "p2", "visit_id": "v2",
         "conditions": [["d3"]],
         "procedures": [["p1"]],
         "drugs_hist": [[]],
         "drugs": ["N02B"]},
    ]


class TestARMR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = create_sample_dataset(
            samples=_samples(), input_schema=SCHEMA,
            output_schema={"drugs": "multilabel"}, dataset_name="armr",
        )
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=3)))

    def test_forward_shapes_with_and_without_graphs(self):
        for use_graph in (False, True):
            torch.manual_seed(0)
            model = ARMR(
                self.dataset, embedding_dim=16, use_graph=use_graph,
                graph_dataset=self.dataset,
            )
            out = model(**self.batch)
            self.assertEqual(out["y_prob"].shape, (3, 4))
            self.assertEqual(out["logit"].shape, (3, 4))
            self.assertEqual(out["y_true"].shape, (3, 4))
            self.assertEqual(out["loss"].dim(), 0)
            self.assertTrue(((out["y_prob"] >= 0) & (out["y_prob"] <= 1)).all())
            out["loss"].backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertFalse(any(torch.isnan(g).any() for g in grads))

    def test_training_reduces_the_loss(self):
        torch.manual_seed(0)
        model = ARMR(self.dataset, embedding_dim=16, use_graph=False, dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        first = None
        for _ in range(20):
            optimizer.zero_grad()
            loss = model(**self.batch)["loss"]
            loss.backward()
            optimizer.step()
            first = first if first is not None else loss.item()
        self.assertLess(loss.item(), first * 0.5)

    def test_trainer_trains_and_evaluates(self):
        torch.manual_seed(0)
        model = ARMR(self.dataset, embedding_dim=16, graph_dataset=self.dataset)
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        trainer = Trainer(
            model=model, metrics=["jaccard_samples", "f1_samples"],
            enable_logging=False,
        )
        trainer.train(train_dataloader=loader, epochs=1)
        scores = trainer.evaluate(get_dataloader(self.dataset, batch_size=2))
        for key in ("loss", "jaccard_samples", "f1_samples"):
            self.assertIn(key, scores)
            self.assertTrue(math.isfinite(scores[key]))

    def test_inference_without_labels(self):
        model = ARMR(self.dataset, embedding_dim=16, use_graph=False)
        model.eval()
        batch = {k: v for k, v in self.batch.items() if k != "drugs"}
        with torch.no_grad():
            out = model(**batch)
        self.assertEqual(sorted(out), ["logit", "y_prob"])

    def test_loss_is_the_weighted_bce_and_margin(self):
        model = ARMR(self.dataset, embedding_dim=16, use_graph=False, alpha=1.0)
        logit = torch.tensor([[2.0, -1.0, 0.5, -3.0]])
        y = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
        self.assertAlmostEqual(
            model._loss(torch.sigmoid(logit), logit, y).item(), bce.item(), places=6
        )
        # the margin term alone: pairs (positive i, negative j), divided by |M|
        model.alpha = 0.0
        prob = torch.tensor([[0.9, 0.1, 0.6, 0.2]])
        expected = sum(
            max(0.0, 1 - (prob[0, i] - prob[0, j]).item())
            for i in (0, 2) for j in (1, 3)
        ) / 4
        self.assertAlmostEqual(
            model._loss(prob, logit, y).item(), expected, places=6
        )

    def test_drug_history_splits_reused_and_new_drugs(self):
        model = ARMR(self.dataset, embedding_dim=16, use_graph=False)
        hist = self.batch["drugs_hist"]
        multihot = model._drug_history(hist, hist.size(1))
        vocab = self.dataset.output_processors["drugs"].label_vocab
        # patient p1: B01A, then C03C and A01A (both new), then B01A and
        # A01A again (both reused)
        reused, new = _split_reused_new(multihot)
        b, c, a = vocab["B01A"], vocab["C03C"], vocab["A01A"]
        self.assertEqual(new[1, 0, b].item(), 1.0)
        self.assertEqual(new[1, 1, c].item(), 1.0)
        self.assertEqual(new[1, 1, a].item(), 1.0)
        self.assertEqual(reused[1, :2].sum().item(), 0.0)
        self.assertEqual(reused[1, 2, b].item(), 1.0)
        self.assertEqual(reused[1, 3, a].item(), 1.0)
        self.assertEqual(new[1, 2:].sum().item(), 0.0)
        self.assertTrue(torch.equal(reused + new, multihot))

    def test_responsive_drugs_decay_reaches_the_score(self):
        torch.manual_seed(0)
        model = ARMR(self.dataset, embedding_dim=16, use_graph=False)
        m = model.label_size
        prev_len = torch.tensor([3])
        # drug 0 given one visit back, or three visits back
        recent = torch.zeros(1, 4, m)
        recent[0, 2, 0] = 1.0
        old = torch.zeros(1, 4, m)
        old[0, 0, 0] = 1.0
        _, rec_recent = model._responsive_drugs(recent, prev_len)
        _, rec_old = model._responsive_drugs(old, prev_len)
        # zero for drugs never taken, larger for more recent use
        self.assertEqual(rec_recent[0, 1:].abs().sum().item(), 0.0)
        self.assertGreater(rec_recent[0, 0].item(), rec_old[0, 0].item())
        self.assertGreater(rec_old[0, 0].item(), 0.0)

    def test_visit_lengths_count_an_empty_middle_visit(self):
        samples = [
            {"patient_id": "p0", "visit_id": "v0",
             "conditions": [["d1"], [], ["d2"]],
             "procedures": [["p1"], [], ["p2"]],
             "drugs_hist": [["A01A"], ["B01A"], []],
             "drugs": ["A01A"]},
            {"patient_id": "p1", "visit_id": "v1",
             "conditions": [["d1"]], "procedures": [["p1"]],
             "drugs_hist": [[]], "drugs": ["B01A"]},
        ]
        dataset = create_sample_dataset(
            samples=samples, input_schema=SCHEMA,
            output_schema={"drugs": "multilabel"}, dataset_name="armr_gap",
        )
        batch = next(iter(get_dataloader(dataset, batch_size=2)))
        model = ARMR(dataset, embedding_dim=16, use_graph=False)
        self.assertEqual(model._visit_lengths(batch["conditions"]).tolist(), [3, 1])
        out = model(**batch)
        self.assertEqual(out["logit"].shape, (2, 2))
        self.assertFalse(torch.isnan(out["loss"]))

    def test_graph_without_graph_dataset_raises(self):
        with self.assertRaisesRegex(ValueError, "graph_dataset"):
            ARMR(self.dataset, embedding_dim=16)
        with self.assertRaisesRegex(ValueError, "use_graph=False"):
            ARMR(self.dataset, embedding_dim=16, use_graph=True)
        # no error, and no warning, with graph_dataset or without the graphs
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            ARMR(self.dataset, embedding_dim=16, graph_dataset=self.dataset)
            ARMR(self.dataset, embedding_dim=16, use_graph=False)

    def test_graph_comes_only_from_graph_dataset(self):
        vocab = self.dataset.output_processors["drugs"].label_vocab
        a, b = vocab["A01A"], vocab["B01A"]
        c, n = vocab["C03C"], vocab["N02B"]
        # p0 (A01A with B01A) only; p1's C03C with N02B stays out
        model = ARMR(
            self.dataset, embedding_dim=16, graph_dataset=self.dataset.subset([0])
        )
        self.assertGreater(model.ehr_adj[a, b].item(), 0.0)
        self.assertEqual(model.ehr_adj[c, n].item(), 0.0)
        full = ARMR(self.dataset, embedding_dim=16, graph_dataset=self.dataset)
        self.assertGreater(full.ehr_adj[c, n].item(), 0.0)

    def test_set_attention_is_normalized_within_each_set(self):
        torch.manual_seed(0)
        model = ARMR(self.dataset, embedding_dim=16, use_graph=False)
        emb = model._drug_embeddings()
        h = torch.randn(3, 2 * model.n_recent, 16)
        # patient 0: drugs 0 and 2 taken; patient 1: all; patient 2: none
        taken = torch.tensor(
            [[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
        )
        for member in (taken, 1.0 - taken):
            q = model._set_attention(emb, h, model.w_reused, member)
            self.assertEqual(q.shape, (3, 4, 16))
            sums = q.sum(dim=1)  # per patient and feature
            for b in range(3):
                expected = 1.0 if member[b].sum() > 0 else 0.0
                self.assertTrue(torch.allclose(sums[b], torch.full((16,), expected)))
            # zero outside the set
            self.assertEqual((q * (1 - member).unsqueeze(-1)).abs().sum().item(), 0.0)
        # an empty set gives zeros with finite gradients
        h.requires_grad_(True)
        model._set_attention(emb, h, model.w_new, taken).sum().backward()
        self.assertTrue(torch.isfinite(h.grad).all())

    def test_decay_parameters_get_a_gradient(self):
        torch.manual_seed(0)
        model = ARMR(self.dataset, embedding_dim=16, use_graph=False)
        model(**self.batch)["loss"].backward()
        for param in (model.decay_scale, model.decay_rate):
            self.assertIsNotNone(param.grad)
            self.assertNotEqual(param.grad.item(), 0.0)

class TestPiecewiseTemporalLearning(unittest.TestCase):
    def test_reverse_padded_keeps_padding_last(self):
        x = torch.arange(8.0).view(2, 4, 1)
        out = _reverse_padded(x, torch.tensor([3, 1]))
        self.assertEqual(out[0, :, 0].tolist(), [2.0, 1.0, 0.0, 3.0])
        self.assertEqual(out[1, :, 0].tolist(), [4.0, 5.0, 6.0, 7.0])

    def test_output_shape_and_padding_independence(self):
        torch.manual_seed(0)
        ptl = PiecewiseTemporalLearning(dim=8, n_recent=2).eval()
        seq = torch.randn(1, 5, 8)
        out = ptl(seq, torch.tensor([5]))
        self.assertEqual(out.shape, (1, 4, 8))
        # extra padded visits do not change the result
        padded = torch.cat([seq, torch.randn(1, 3, 8)], dim=1)
        self.assertTrue(torch.allclose(out, ptl(padded, torch.tensor([5])), atol=1e-6))


class TestDrugHelpers(unittest.TestCase):
    def test_ehr_cooccurrence_adjacency_from_codes_and_multihot(self):
        vocab = {"A01A": 0, "B01A": 1, "C03C": 2, "N02B": 3}
        adj = ehr_cooccurrence_adjacency(
            [["A01A", "B01A", "A01A", "X99X"], torch.tensor([0.0, 0.0, 1.0, 1.0]),
             ["N02B"]],
            vocab,
        )
        expected = torch.zeros(4, 4)
        expected[0, 1] = expected[1, 0] = expected[2, 3] = expected[3, 2] = 1.0
        self.assertTrue(torch.equal(adj, expected))

    def test_gcn_normalize_adjacency_rows_sum_to_one(self):
        adj = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        norm = gcn_normalize_adjacency(adj)
        expected = torch.tensor(
            [[1 / 3, 1 / 3, 1 / 3], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5]]
        )
        self.assertTrue(torch.allclose(norm, expected))
        self.assertTrue(torch.allclose(norm.sum(dim=1), torch.ones(3)))
        # without self-loops, an empty row stays zero (no NaN or inf)
        bare = gcn_normalize_adjacency(torch.tensor([[0.0, 2.0], [0.0, 0.0]]), False)
        self.assertTrue(torch.equal(bare, torch.tensor([[0.0, 1.0], [0.0, 0.0]])))

    def test_history_to_label_index_sends_unknown_codes_to_trash(self):
        mapping = history_to_label_index(
            {"<pad>": 0, "<unk>": 1, "B01A": 2, "X99X": 3}, {"N02B": 0, "B01A": 1}
        )
        self.assertEqual(mapping.tolist(), [2, 2, 1, 2])

    def test_ddi_adjacency_is_a_symmetric_0_1_matrix(self):
        vocab = {"N02B": 0, "B01A": 1, "C03C": 2, "N02A": 3, "M01A": 4}
        adj = ddi_adjacency(vocab)
        self.assertEqual(adj.shape, (5, 5))
        self.assertTrue(torch.equal(adj, adj.T))
        self.assertTrue(((adj == 0) | (adj == 1)).all())
        self.assertGreater(adj.sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()

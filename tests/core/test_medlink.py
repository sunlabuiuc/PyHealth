import unittest
import torch

from pyhealth.datasets import create_sample_dataset
from pyhealth.models import MedLink


class TestMedLink(unittest.TestCase):
    """Basic tests for the MedLink model on pseudo data."""

    def setUp(self):
        # Each "sample" here is a simple patient-record placeholder.
        # The dataset is used to fit SequenceProcessors (vocabularies), which
        # MedLink reuses for processor-native indexing.
        self.samples = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                # query-side codes
                "conditions": ["A", "B", "C"],
                # corpus-side codes ("d_" + feature_key)
                "d_conditions": ["A", "D"],
            },
            {
                "patient_id": "p1",
                "visit_id": "v1",
                "conditions": ["B", "E"],
                "d_conditions": ["C", "E", "F"],
            },
        ]

        # Two sequence-type inputs: conditions and d_conditions
        self.input_schema = {
            "conditions": "sequence",
            "d_conditions": "sequence",
        }
        # No labels are needed; MedLink is self-supervised
        self.output_schema = {}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="medlink_test",
            in_memory=True,
        )

        self.model = MedLink(
            dataset=self.dataset,
            feature_keys=["conditions"],
            embedding_dim=32,
            alpha=0.5,
            beta=0.5,
            gamma=1.0,
        )

    def _make_batch(self):
        # Construct a tiny batch in the format expected by MedLink.forward
        # s_q: list of query sequences
        s_q = [
            ["A", "B", "C"],
            ["B", "E"],
        ]
        # s_p: list of positive corpus sequences
        s_p = [
            ["A", "D"],
            ["C", "E", "F"],
        ]
        # Optionally you could also define negatives s_n = [...]
        batch = {
            "query_id": ["q0", "q1"],
            "id_p": ["p0", "p1"],
            "s_q": s_q,
            "s_p": s_p,
            # no s_n -> defaults to None
        }
        return batch

    def test_model_initialization(self):
        """Model constructs with correct vocabulary size and encoders."""
        self.assertIsInstance(self.model, MedLink)
        self.assertEqual(self.model.feature_key, "conditions")
        self.assertGreater(self.model.vocab_size, 0)
        self.assertIsNotNone(self.model.forward_encoder)
        self.assertIsNotNone(self.model.backward_encoder)

    def test_forward_and_backward(self):
        """Forward pass returns a scalar loss and backward computes gradients."""
        batch = self._make_batch()

        # Forward
        ret = self.model(**batch)
        self.assertIn("loss", ret)
        loss = ret["loss"]
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)  # scalar

        # Backward
        loss.backward()
        has_grad = any(
            p.requires_grad and p.grad is not None
            for p in self.model.parameters()
        )
        self.assertTrue(has_grad, "No gradients after backward pass")

    def test_encoding_helpers(self):
        """encode_queries / encode_corpus produce consistent shapes."""
        queries = [["A", "B"], ["C"]]
        corpus = [["A"], ["B", "C"]]

        q_emb = self.model.encode_queries(queries)
        c_emb = self.model.encode_corpus(corpus)

        self.assertEqual(q_emb.shape[1], self.model.vocab_size)
        self.assertEqual(c_emb.shape[1], self.model.vocab_size)
        self.assertEqual(q_emb.shape[0], len(queries))
        self.assertEqual(c_emb.shape[0], len(corpus))

        scores = self.model.compute_scores(q_emb, c_emb)
        self.assertEqual(scores.shape, (len(queries), len(corpus)))

    def test_feature_key_inference(self):
        """Model should infer feature_keys if not provided."""
        model = MedLink(
            dataset=self.dataset,
            # feature_keys omitted
            embedding_dim=32,
        )
        self.assertEqual(model.feature_key, "conditions")

    def test_hard_negatives_score_query_not_positive_doc(self):
        """Regression: hard negatives must be mined by scoring the QUERY, not
        the positive document. The old code called get_scores(d) on the
        positive doc, so "hard negatives" were docs similar to the answer.
        """
        from pyhealth.models.medlink.utils import get_bm25_hard_negatives

        class FakeBM25:
            def get_scores(self, text):
                if text == "QUERY":  # query ranks pos top, then neg_q
                    return {"pos": 10.0, "neg_q": 9.0, "neg_d": 1.0}
                return {"pos": 10.0, "neg_d": 9.0, "neg_q": 1.0}  # doc-scoring picks neg_d

        corpus = {"pos": "POS", "neg_q": "NQ", "neg_d": "ND"}
        queries = {"q1": "QUERY"}
        qrels = {"q1": {"pos": 1}}
        out = get_bm25_hard_negatives(FakeBM25(), corpus, queries, qrels)
        # scoring the query picks neg_q; scoring the positive doc would pick neg_d
        self.assertEqual(out["q1"], {"pos": 1, "neg_q": -1})

    def test_hard_negatives_exclude_all_positives(self):
        """Regression: with multiple positives, no positive may be chosen as a
        negative. Scoring the query ranks the positives on top, so excluding
        only the current positive would pick another positive as a false negative.
        """
        from pyhealth.models.medlink.utils import get_bm25_hard_negatives

        class FakeBM25:
            def get_scores(self, text):
                return {"pos1": 10.0, "pos2": 9.0, "neg": 8.0}

        corpus = {"pos1": "P1", "pos2": "P2", "neg": "N"}
        queries = {"q1": "QUERY"}
        qrels = {"q1": {"pos1": 1, "pos2": 1}}
        out = get_bm25_hard_negatives(FakeBM25(), corpus, queries, qrels)
        neg_ids = [d for d, lbl in out["q1"].items() if lbl == -1]
        self.assertNotIn("pos1", neg_ids)
        self.assertNotIn("pos2", neg_ids)
        self.assertEqual(neg_ids, ["neg"])

    def test_hard_negatives_preserves_all_positives(self):
        """Regression: with multiple positives, all positives must remain in the
        output when a hard negative is added.
        """
        from pyhealth.models.medlink.utils import (
            get_bm25_hard_negatives,
            get_train_dataloader,
        )

        class FakeBM25:
            def get_scores(self, text):
                return {"pos1": 10.0, "pos2": 9.0, "neg": 8.0}

        corpus = {"pos1": "P1", "pos2": "P2", "neg": "N"}
        queries = {"q1": "QUERY"}
        qrels = {"q1": {"pos1": 1, "pos2": 1}}

        out = get_bm25_hard_negatives(FakeBM25(), corpus, queries, qrels)

        self.assertEqual(
            out["q1"],
            {"pos1": 1, "pos2": 1, "neg": -1},
        )

        dataloader = get_train_dataloader(
            corpus, queries, out, batch_size=2, shuffle=False
        )
        batch = next(iter(dataloader))
        self.assertEqual(batch["id_p"], ["pos1", "pos2"])
        self.assertEqual(batch["s_n"], ["N", "N"])


class TestMedLinkCollateFn(unittest.TestCase):
    """Regression tests for collate_fn's handling of samples with
    heterogeneous keys -- specifically get_train_dataloader's "s_n" key,
    present only for samples where a hard negative was mined.

    The original implementation initialized output keys from only
    samples[0], so depending on sample order it either raised a KeyError
    (samples[0] lacked a key a later sample had) or silently produced a
    shorter, misaligned list for that key (samples[0] had it, a later
    sample didn't) -- both reproduced below.
    """

    def _common_samples(self, first_has_s_n, second_has_s_n):
        s1 = {"query_id": "q1", "id_p": "p1", "s_q": "Q1", "s_p": "P1"}
        s2 = {"query_id": "q2", "id_p": "p2", "s_q": "Q2", "s_p": "P2"}
        if first_has_s_n:
            s1["s_n"] = "N1"
        if second_has_s_n:
            s2["s_n"] = "N2"
        return [s1, s2]

    def test_s_n_present_first_absent_second_does_not_crash(self):
        from pyhealth.models.medlink.utils import collate_fn

        samples = self._common_samples(first_has_s_n=True, second_has_s_n=False)
        out = collate_fn(samples)  # would previously misalign s_n to length 1
        self.assertNotIn("s_n", out)
        self.assertEqual(out["id_p"], ["p1", "p2"])
        self.assertEqual(out["s_q"], ["Q1", "Q2"])

    def test_s_n_absent_first_present_second_does_not_crash(self):
        from pyhealth.models.medlink.utils import collate_fn

        samples = self._common_samples(first_has_s_n=False, second_has_s_n=True)
        out = collate_fn(samples)  # would previously raise KeyError('s_n')
        self.assertNotIn("s_n", out)
        self.assertEqual(out["id_p"], ["p1", "p2"])

    def test_s_n_present_in_all_samples_preserved_and_aligned(self):
        from pyhealth.models.medlink.utils import collate_fn

        samples = self._common_samples(first_has_s_n=True, second_has_s_n=True)
        out = collate_fn(samples)
        self.assertEqual(out["s_n"], ["N1", "N2"])
        self.assertEqual(len(out["s_n"]), len(out["id_p"]))

    def test_s_n_absent_in_all_samples_unaffected(self):
        from pyhealth.models.medlink.utils import collate_fn

        samples = self._common_samples(first_has_s_n=False, second_has_s_n=False)
        out = collate_fn(samples)
        self.assertNotIn("s_n", out)
        self.assertEqual(out["id_p"], ["p1", "p2"])

    def test_all_output_lists_same_length_as_batch(self):
        """General invariant: every key in a collated batch must have
        exactly one entry per input sample, regardless of which samples
        contributed which keys."""
        from pyhealth.models.medlink.utils import collate_fn

        for first, second in [(True, False), (False, True), (True, True), (False, False)]:
            samples = self._common_samples(first, second)
            out = collate_fn(samples)
            for key, values in out.items():
                self.assertEqual(
                    len(values), len(samples),
                    f"key {key!r} has {len(values)} entries, expected {len(samples)}",
                )


if __name__ == "__main__":
    unittest.main()

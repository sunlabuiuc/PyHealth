import unittest
import torch

from pyhealth.datasets import SampleDataset
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

        self.dataset = SampleDataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="medlink_test",
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


if __name__ == "__main__":
    unittest.main()

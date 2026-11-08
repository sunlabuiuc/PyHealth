import importlib.util
import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import KeyClass
from pyhealth.weak_supervision import KeywordLabeler, aggregate_votes, vote_diagnostics


class MockEncoder:
    def encode(self, texts):
        return np.array(
            [[1 + t.count("heart"), 1 + t.count("lung")] for t in texts], dtype=float
        )


class TestKeyClass(unittest.TestCase):
    def test_mining_and_boundaries(self):
        labeler = KeywordLabeler(MockEncoder(), top_k=4, min_df=1).fit(
            ["heart failure", "lung disease", "heart lung"],
            {"cardiac": "heart", "resp": "lung"},
        )
        self.assertEqual(labeler.category_ids_, ("cardiac", "resp"))
        self.assertTrue(any(lf.phrase == "heart failure" for lf in labeler.lfs_))
        votes = labeler.transform(["HEART failure", "heartland", "", "heart lung"])
        self.assertTrue((votes[0] != -1).any())
        self.assertTrue((votes[1:3] == -1).all())
        self.assertEqual(labeler.transform([]).shape, (0, len(labeler.lfs_)))
        self.assertGreater(vote_diagnostics(votes, 2)["cross_category_fraction"], 0)
        # Transform must not learn new phrases from held-out notes.
        self.assertNotIn("heartland", labeler.vectorizer_.vocabulary_)

    def test_majority_abstention_and_validation(self):
        probabilities, covered = aggregate_votes(
            np.array([[0, 1, -1], [-1, -1, -1], [1, 1, 0]]), 2
        )
        np.testing.assert_allclose(
            probabilities, [[0.5, 0.5], [0.5, 0.5], [1 / 3, 2 / 3]]
        )
        self.assertEqual(covered.tolist(), [True, False, True])
        with self.assertRaises(ValueError):
            aggregate_votes(np.array([[2]]), 2)
        p, mask = aggregate_votes(np.empty((2, 0), dtype=int), 2)
        self.assertFalse(mask.any())
        np.testing.assert_allclose(p, 0.5)

    def test_pyhealth_batch_and_gradients(self):
        samples = [
            {
                "patient_id": str(i),
                "embedding": [1.0, float(i)],
                "weak_labels": [0.2, 0.8],
            }
            for i in range(2)
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"embedding": "tensor"},
            output_schema={"weak_labels": "tensor"},
        )
        model = KeyClass(dataset, 2, ["cardiac", "resp"], hidden_dims=(8, 8, 4))
        batch = next(iter(get_dataloader(dataset, batch_size=2)))
        result = model(**batch)
        self.assertEqual(model.mode, "multilabel")
        self.assertEqual(model.label_processor.output_size(), 2)
        self.assertEqual(model.label_processor.label_vocab, {"cardiac": 0, "resp": 1})
        self.assertEqual(list(model.label_processors), ["weak_labels"])
        self.assertEqual(result["y_prob"].shape, (2, 2))
        result["loss"].backward()
        self.assertTrue(torch.isfinite(model.network[0].weight.grad).all())
        model.eval()
        self.assertNotIn("loss", model(embedding=batch["embedding"]))
        with self.assertRaises(ValueError):
            model(embedding=batch["embedding"], weak_labels=[[2.0, 0.0], [0.0, 1.0]])

    @unittest.skipUnless(
        importlib.util.find_spec("snorkel"), "optional Snorkel not installed"
    )
    def test_real_snorkel(self):
        votes = np.tile([[0, 0, -1], [1, -1, 1], [0, 1, 0], [1, 1, 1]], (10, 1))
        votes = np.vstack([votes, [-1, -1, -1]])
        p, covered = aggregate_votes(
            votes, 2, method="snorkel", n_epochs=20, progress_bar=False
        )
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)
        self.assertFalse(covered[-1])
        np.testing.assert_allclose(p[-1], [0.5, 0.5])

"""Covers every metric branch of pyhealth.metrics.multilabel_metrics_fn."""

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pyhealth.metrics import multilabel_metrics_fn

ALL_METRICS = [
    "ddi",  # first on purpose: must not change y_pred for the metrics below
    "roc_auc_micro",
    "roc_auc_macro",
    "roc_auc_weighted",
    "roc_auc_samples",
    "pr_auc_micro",
    "pr_auc_macro",
    "pr_auc_weighted",
    "pr_auc_samples",
    "accuracy",
    "f1_micro",
    "f1_macro",
    "f1_weighted",
    "f1_samples",
    "precision_micro",
    "precision_macro",
    "precision_weighted",
    "precision_samples",
    "recall_micro",
    "recall_macro",
    "recall_weighted",
    "recall_samples",
    "jaccard_micro",
    "jaccard_macro",
    "jaccard_weighted",
    "jaccard_samples",
    "hamming_loss",
    "cwECE",
    "cwECE_adapt",
    "tp",
    "fp",
]


class TestMultilabelMetricBranches(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
        self.y_prob = np.array(
            [[0.9, 0.4, 0.8], [0.2, 0.7, 0.6], [0.8, 0.9, 0.1], [0.1, 0.2, 0.7]]
        )
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        np.save(
            os.path.join(tmpdir.name, "ddi_adj.npy"),
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float),
        )
        patcher = patch("pyhealth.metrics.multilabel.CACHE_PATH", tmpdir.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_every_metric_with_ddi_first(self):
        y_predset = (self.y_prob >= 0.5).astype(int)
        out = multilabel_metrics_fn(
            self.y_true, self.y_prob, metrics=ALL_METRICS, y_predset=y_predset
        )
        expected_keys = {m for m in ALL_METRICS if m != "ddi"} | {"ddi_score"}
        self.assertEqual(set(out), expected_keys)
        self.assertAlmostEqual(out["accuracy"], 11 / 12)
        self.assertAlmostEqual(out["hamming_loss"], 1 / 12)

    def test_prediction_set_metrics_skipped_without_predset(self):
        out = multilabel_metrics_fn(self.y_true, self.y_prob, metrics=["tp", "fp"])
        self.assertEqual(out, {})

    def test_default_metric(self):
        out = multilabel_metrics_fn(self.y_true, self.y_prob)
        self.assertEqual(set(out), {"pr_auc_samples"})

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            multilabel_metrics_fn(self.y_true, self.y_prob, metrics=["nope"])


if __name__ == "__main__":
    unittest.main()

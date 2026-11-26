"""Tests for pyhealth.metrics.multilabel: the "ddi" branch must not clobber
the shared y_pred array used by the other metrics.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pyhealth.metrics import multilabel_metrics_fn


class TestMultilabelDDIMetricOrder(unittest.TestCase):
    """Regression tests: the output of multilabel_metrics_fn must not depend
    on where "ddi" sits in the metrics list.

    The "ddi" branch turns the thresholded (n_samples, n_labels) y_pred into a
    list of label indices per sample. It used to assign that list back to
    y_pred, so every threshold based metric requested after "ddi" received a
    ragged list and sklearn raised "Classification metrics can't handle a mix
    of multilabel-indicator and unknown targets".
    """

    def setUp(self):
        self.y_true = np.array(
            [[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]]
        )
        self.y_prob = np.array(
            [
                [0.9, 0.4, 0.8],
                [0.2, 0.7, 0.6],
                [0.8, 0.9, 0.1],
                [0.1, 0.2, 0.7],
            ]
        )
        # Labels 0 and 1 interact, label 2 interacts with nothing.
        ddi_adj = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        np.save(os.path.join(tmpdir.name, "ddi_adj.npy"), ddi_adj)
        patcher = patch("pyhealth.metrics.multilabel.CACHE_PATH", tmpdir.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_ddi_last_keeps_its_values(self):
        """The ordering the other shipped examples use must not change."""
        out = multilabel_metrics_fn(
            self.y_true,
            self.y_prob,
            metrics=["jaccard_samples", "f1_samples", "ddi"],
        )
        self.assertAlmostEqual(out["jaccard_samples"], 0.9166666666666666)
        self.assertAlmostEqual(out["f1_samples"], 0.95)
        self.assertAlmostEqual(out["ddi_score"], 0.4)

    def test_ddi_first_matches_ddi_last(self):
        """Putting the headline safety metric first used to raise ValueError."""
        ddi_first = multilabel_metrics_fn(
            self.y_true,
            self.y_prob,
            metrics=["ddi", "jaccard_samples", "f1_samples"],
        )
        ddi_last = multilabel_metrics_fn(
            self.y_true,
            self.y_prob,
            metrics=["jaccard_samples", "f1_samples", "ddi"],
        )
        self.assertEqual(ddi_first, ddi_last)

    def test_ddi_in_the_middle_matches_ddi_last(self):
        """The order the docstring itself lists, ddi before hamming_loss."""
        ddi_middle = multilabel_metrics_fn(
            self.y_true,
            self.y_prob,
            metrics=["jaccard_samples", "ddi", "hamming_loss"],
        )
        ddi_last = multilabel_metrics_fn(
            self.y_true,
            self.y_prob,
            metrics=["jaccard_samples", "hamming_loss", "ddi"],
        )
        self.assertEqual(ddi_middle, ddi_last)
        self.assertAlmostEqual(ddi_middle["hamming_loss"], 0.08333333333333333)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from pyhealth.metrics.multiclass import multiclass_metrics_fn


class TestMulticlassRankingMetrics(unittest.TestCase):
    def setUp(self):
        # True-class ranks are 1, 2, 3 and 4 respectively (no ties).
        self.y_true = np.array([0, 1, 2, 3])
        self.y_prob = np.array(
            [
                [0.70, 0.10, 0.15, 0.05],
                [0.60, 0.25, 0.10, 0.05],
                [0.50, 0.30, 0.15, 0.05],
                [0.40, 0.30, 0.20, 0.10],
            ]
        )

    def test_hits_at_custom_k(self):
        result = multiclass_metrics_fn(
            self.y_true, self.y_prob, metrics=["hits@2", "hits@3"]
        )
        self.assertEqual(set(result), {"HITS@2", "HITS@3"})
        self.assertAlmostEqual(result["HITS@2"], 0.5)
        self.assertAlmostEqual(result["HITS@3"], 0.75)

    def test_hits_at_k_larger_than_num_classes(self):
        result = multiclass_metrics_fn(self.y_true, self.y_prob, metrics=["hits@20"])
        self.assertAlmostEqual(result["HITS@20"], 1.0)

    def test_hits_at_n_backward_compatible(self):
        result = multiclass_metrics_fn(self.y_true, self.y_prob, metrics=["hits@n"])
        self.assertEqual(set(result), {"HITS@1", "HITS@5", "HITS@10"})
        self.assertAlmostEqual(result["HITS@1"], 0.25)
        self.assertAlmostEqual(result["HITS@5"], 1.0)
        self.assertAlmostEqual(result["HITS@10"], 1.0)

    def test_invalid_hits_cutoff_raises(self):
        for bad in ["hits@0", "hits@-1", "hits@abc", "hits@", "hits@2.5"]:
            with self.subTest(metric=bad), self.assertRaises(ValueError):
                multiclass_metrics_fn(self.y_true, self.y_prob, metrics=[bad])

    def test_mean_rank(self):
        result = multiclass_metrics_fn(self.y_true, self.y_prob, metrics=["mean_rank"])
        self.assertAlmostEqual(result["mean_rank"], 2.5)
        self.assertAlmostEqual(
            result["mean_reciprocal_rank"], np.mean([1, 1 / 2, 1 / 3, 1 / 4])
        )

    def test_out_of_range_label_raises(self):
        y_true = np.array([2])
        y_prob = np.array([[0.6, 0.4]])
        with self.assertRaises(ValueError):
            multiclass_metrics_fn(y_true, y_prob, metrics=["hits@1"])


if __name__ == "__main__":
    unittest.main()

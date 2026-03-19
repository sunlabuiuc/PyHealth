import builtins
import unittest
from unittest.mock import patch

from pyhealth.metrics.generative import (
    bertscore_f1_score,
    exact_match_score,
    generative_metrics_fn,
    normalize_text_for_exact_match,
)


class TestGenerativeMetrics(unittest.TestCase):
    def test_exact_match_normalization(self):
        y_true = ["The left lung.", "An edema"]
        y_pred = ["left lung", "edema"]

        self.assertEqual(normalize_text_for_exact_match("The left lung."), "left lung")
        self.assertAlmostEqual(exact_match_score(y_true, y_pred), 1.0)

    def test_metric_dispatch(self):
        y_true = ["yes", "no"]
        y_pred = ["yes", "yes"]

        metrics = generative_metrics_fn(y_true=y_true, y_pred=y_pred, metrics=["exact_match"])
        self.assertIn("exact_match", metrics)
        self.assertAlmostEqual(metrics["exact_match"], 0.5)

        with self.assertRaises(ValueError):
            generative_metrics_fn(y_true=y_true, y_pred=y_pred, metrics=["unknown_metric"])

    def test_optional_bertscore_dependency_missing(self):
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "bert_score":
                raise ImportError("mock missing bert_score")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(ImportError):
                bertscore_f1_score(["yes"], ["yes"])


if __name__ == "__main__":
    unittest.main()

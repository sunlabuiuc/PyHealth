import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(
    module_name: str, relative_path: str, stubs: dict[str, object] | None = None
):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs or {}, clear=False):
        spec.loader.exec_module(module)
    return module


def _load_multilabel_module(ddi_rate_score_fn):
    fake_pyhealth = types.ModuleType("pyhealth")
    fake_pyhealth.BASE_CACHE_PATH = "/tmp"

    fake_metrics = types.ModuleType("pyhealth.metrics")
    fake_metrics.__path__ = []
    fake_metrics.ddi_rate_score = ddi_rate_score_fn

    fake_calibration = types.ModuleType("pyhealth.metrics.calibration")
    fake_calibration.ece_classwise = lambda *args, **kwargs: 0.0
    fake_metrics.calibration = fake_calibration

    stubs = {
        "pyhealth": fake_pyhealth,
        "pyhealth.metrics": fake_metrics,
        "pyhealth.metrics.calibration": fake_calibration,
    }
    return _load_module(
        "multilabel_metrics_under_test", "pyhealth/metrics/multilabel.py", stubs
    )


class TestRegressionMetricsQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.regression = _load_module(
            "regression_metrics_under_test", "pyhealth/metrics/regression.py"
        )

    def test_kl_divergence_does_not_change_mse(self):
        x = np.array([0.2, 0.3, 0.5], dtype=float)
        x_rec = np.array([0.1, 0.7, 0.2], dtype=float)

        mse_only = self.regression.regression_metrics_fn(x, x_rec, metrics=["mse"])[
            "mse"
        ]
        mse_with_kl = self.regression.regression_metrics_fn(
            x, x_rec, metrics=["kl_divergence", "mse"]
        )["mse"]

        self.assertAlmostEqual(mse_with_kl, mse_only, places=12)


class TestMultilabelMetricsQuality(unittest.TestCase):
    def test_ddi_metric_does_not_break_followup_metrics(self):
        def fake_ddi_rate_score(pred_labels, ddi_adj):
            return 0.125

        multilabel = _load_multilabel_module(fake_ddi_rate_score)
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_prob = np.array([[0.8, 0.4, 0.9], [0.3, 0.7, 0.2]])

        with patch.object(multilabel.np, "load", return_value=np.zeros((3, 3))):
            scores = multilabel.multilabel_metrics_fn(
                y_true,
                y_prob,
                metrics=["ddi", "f1_micro"],
                threshold=0.5,
            )

        self.assertIn("ddi", scores)
        self.assertIn("ddi_score", scores)
        self.assertIn("f1_micro", scores)
        self.assertEqual(scores["ddi"], scores["ddi_score"])


class TestRankingMetricsQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ranking = _load_module(
            "ranking_metrics_under_test", "pyhealth/metrics/ranking.py"
        )

    def test_invalid_k_values_raise_value_error(self):
        with self.assertRaisesRegex(ValueError, "k_values"):
            self.ranking.ranking_metrics_fn(
                {"q1": {"d1": 1}},
                {"q1": {"d1": 1.0}},
                [0],
            )

    def test_empty_scores_raise_clear_error(self):
        fake_pytrec_eval = types.ModuleType("pytrec_eval")

        class FakeEvaluator:
            def __init__(self, qrels, metrics):
                self.qrels = qrels
                self.metrics = metrics

            def evaluate(self, results):
                return {}

        fake_pytrec_eval.RelevanceEvaluator = FakeEvaluator

        with patch.dict(sys.modules, {"pytrec_eval": fake_pytrec_eval}, clear=False):
            with self.assertRaisesRegex(ValueError, "No ranking scores were produced"):
                self.ranking.ranking_metrics_fn(
                    {"q1": {"d1": 1}},
                    {"q1": {"d1": 1.0}},
                    [1],
                )


if __name__ == "__main__":
    unittest.main()

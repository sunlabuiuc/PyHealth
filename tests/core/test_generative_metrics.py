"""Unit tests for pyhealth.metrics.generative (synthetic-EHR metrics).

Run with::

    python -m unittest tests.core.test_generative_metrics -v
"""

import unittest

import numpy as np
import pandas as pd

from pyhealth.metrics.generative import (
    calc_membership_inference,
    calc_nnaar,
    compute_discriminator_privacy,
    compute_mle,
    compute_prevalence_metrics,
    evaluate_synthetic_ehr,
)
from pyhealth.metrics.generative.utils import (
    convert_cols_to_multihot,
    train_lstm_model,
    train_sklearn_model,
)

SUBJECT_COL, VISIT_COL, CODE_COL, LABEL_COL = "id", "time", "visit_codes", "labels"


def _make_dataframes():
    """Builds small synthetic train/test/synthetic EHR dataframes."""
    train_ehr = pd.DataFrame(
        {
            "visit_codes": [0, 1, 3, 4, 1, 2, 0, 3, 2, 4, 1, 0, 2, 3, 4,
                            1, 0, 2, 3, 4, 1, 0, 2, 3, 4],
            "labels": [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
                       1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
            "time": [0, 0, 1, 1, 0, 1, 2, 2, 3, 3, 1, 2, 3, 4, 4,
                     0, 1, 2, 3, 4, 1, 2, 3, 4, 5],
            "id": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                   3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
        }
    ).astype({"visit_codes": str, "labels": int, "time": int, "id": str})

    test_ehr = pd.DataFrame(
        {
            "visit_codes": [1, 2, 0, 3, 4, 2, 1, 0, 3, 4, 1, 2, 3, 0, 4,
                            2, 1, 3, 0, 4],
            "labels": [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
                       0, 0, 1, 0, 1],
            "time": [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 0, 1, 1, 2, 2,
                     3, 3, 3, 4, 4],
            "id": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                   2, 2, 2, 2, 2],
        }
    ).astype({"visit_codes": str, "labels": int, "time": int, "id": str})

    syn_ehr = pd.DataFrame(
        {
            "visit_codes": [2, 3, 1, 4, 0, 2, 3, 1, 0, 4, 1, 2, 3, 4, 0,
                            2, 1, 3, 4, 0, 2, 1, 3, 4, 0, 1, 2, 3, 4, 0],
            "labels": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1,
                       0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
            "time": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                     3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5],
            "id": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        }
    ).astype({"visit_codes": str, "labels": int, "time": int, "id": str})

    return train_ehr, test_ehr, syn_ehr


class GenerativeMetricsTestCase(unittest.TestCase):
    """Shared fixtures and assertion helpers for the generative metrics."""

    def setUp(self):
        np.random.seed(0)
        self.train_ehr, self.test_ehr, self.syn_ehr = _make_dataframes()
        self.cols = dict(
            subject_col=SUBJECT_COL,
            visit_col=VISIT_COL,
            code_col=CODE_COL,
            label_col=LABEL_COL,
        )

    def assertSummary(self, summary, expected_keys):
        """Asserts a metrics summary has the expected (mean, std) structure."""
        self.assertIsInstance(summary, dict)
        for key in expected_keys:
            self.assertIn(key, summary)
            value = summary[key]
            self.assertIsInstance(value, tuple)
            self.assertEqual(len(value), 2)
            mean, std = value
            self.assertTrue(np.isfinite(mean), f"{key} mean not finite")
            self.assertTrue(np.isfinite(std), f"{key} std not finite")
            self.assertGreaterEqual(std, 0.0)


class TestNNAAR(GenerativeMetricsTestCase):
    def test_calc_nnaar(self):
        summary = calc_nnaar(
            self.train_ehr, self.test_ehr, self.syn_ehr,
            **self.cols, sample_size=10, n_runs=3,
        )
        self.assertSummary(summary, ["nnaar", "aa_es", "aa_ts"])
        for key in ("aa_es", "aa_ts"):
            self.assertGreaterEqual(summary[key][0], 0.0)
            self.assertLessEqual(summary[key][0], 1.0)
        self.assertGreaterEqual(summary["nnaar"][0], -1.0)
        self.assertLessEqual(summary["nnaar"][0], 1.0)


class TestMembershipInference(GenerativeMetricsTestCase):
    def test_calc_membership_inference(self):
        summary = calc_membership_inference(
            self.train_ehr, self.test_ehr, self.syn_ehr,
            **self.cols, num_attack_samples=10, n_runs=3,
        )
        keys = ["MIA_F1", "MIA_Precision", "MIA_Recall", "MIA_Accuracy"]
        self.assertSummary(summary, keys)
        for key in keys:
            self.assertGreaterEqual(summary[key][0], 0.0)
            self.assertLessEqual(summary[key][0], 1.0)


class TestDiscriminatorPrivacy(GenerativeMetricsTestCase):
    def test_discriminator_privacy_lstm(self):
        summary = compute_discriminator_privacy(
            train_fn=train_lstm_model,
            train_ehr=self.train_ehr, test_ehr=self.test_ehr,
            syn_ehr=self.syn_ehr, **self.cols, n_bootstraps=3,
            embed_dim=8, hidden_dim=8, batch_size=8, epochs=2, verbose=False,
        )
        keys = ["Privacy_Discriminator_Accuracy", "Privacy_Score"]
        self.assertSummary(summary, keys)
        self.assertGreaterEqual(summary["Privacy_Score"][0], 0.0)
        self.assertLessEqual(summary["Privacy_Score"][0], 1.0)

    def test_discriminator_privacy_rf(self):
        summary = compute_discriminator_privacy(
            train_fn=train_sklearn_model,
            train_ehr=self.train_ehr, test_ehr=self.test_ehr,
            syn_ehr=self.syn_ehr, **self.cols, n_bootstraps=3, model="rf",
        )
        self.assertSummary(
            summary, ["Privacy_Discriminator_Accuracy", "Privacy_Score"]
        )


class TestMLE(GenerativeMetricsTestCase):
    def test_compute_mle_lstm(self):
        summary = compute_mle(
            train_fn=train_lstm_model,
            train_ehr=self.train_ehr, test_ehr=self.test_ehr,
            syn_ehr=self.syn_ehr, **self.cols, n_bootstraps=3,
            embed_dim=8, hidden_dim=8, batch_size=8, epochs=2, verbose=False,
        )
        keys = [
            "MLE_Real_Accuracy", "MLE_Synth_Accuracy", "MLE_Difference",
            "MLE_Ratio", "MLE_Real_F1", "MLE_Synth_F1",
        ]
        self.assertSummary(summary, keys)
        for key in ("MLE_Real_Accuracy", "MLE_Synth_Accuracy"):
            self.assertGreaterEqual(summary[key][0], 0.0)
            self.assertLessEqual(summary[key][0], 1.0)

    def test_compute_mle_rf(self):
        summary = compute_mle(
            train_fn=train_sklearn_model,
            train_ehr=self.train_ehr, test_ehr=self.test_ehr,
            syn_ehr=self.syn_ehr, **self.cols, n_bootstraps=3, model="rf",
        )
        self.assertSummary(summary, ["MLE_Real_Accuracy", "MLE_Synth_Accuracy"])


class TestPrevalenceMetrics(GenerativeMetricsTestCase):
    def test_compute_prevalence_metrics(self):
        summary = compute_prevalence_metrics(
            self.train_ehr, self.syn_ehr,
            subject_col=SUBJECT_COL, code_col=CODE_COL, n_bootstraps=3,
        )
        keys = ["Prevalence_R2", "Prevalence_Pearson", "Prevalence_RMSE"]
        self.assertSummary(summary, keys)
        self.assertGreaterEqual(summary["Prevalence_Pearson"][0], -1.0)
        self.assertLessEqual(summary["Prevalence_Pearson"][0], 1.0)
        self.assertGreaterEqual(summary["Prevalence_RMSE"][0], 0.0)


class TestConvertColsToMultihot(GenerativeMetricsTestCase):
    def test_convert_cols_to_multihot(self):
        df = self.train_ehr.copy()
        df["gender"] = ["M", "F"] * 12 + ["M"]
        df["age"] = np.arange(len(df), dtype=float)
        out = convert_cols_to_multihot(
            df, code_col=CODE_COL, visit_col=VISIT_COL,
            cat_cols=["gender"], num_cols=["age"], bins_per_num=2,
        )
        self.assertIn("combined_codes", out.columns)
        self.assertEqual(len(out), len(df))
        # Each combined code should fold in the code, the category and the bin.
        first = out["combined_codes"].iloc[0]
        self.assertIn("gender_", first)
        self.assertIn("age_", first)
        # The original dataframe must not be mutated.
        self.assertNotIn("combined_codes", df.columns)


class TestEvaluateSyntheticEHR(GenerativeMetricsTestCase):
    def test_evaluate_all_lstm(self):
        out = evaluate_synthetic_ehr(
            self.train_ehr, self.test_ehr, self.syn_ehr, **self.cols,
            sample_size=10, mode="lstm", metrics="all",
            lstm_params={"embed_dim": 8, "hidden_dim": 8,
                         "batch_size": 8, "epochs": 2},
            n_bootstraps=3, n_runs=3,
        )
        for key in ("nnaar", "MIA_F1", "MLE_Real_Accuracy",
                    "Privacy_Score", "Prevalence_RMSE"):
            self.assertIn(key, out)

    def test_evaluate_privacy_only_rf(self):
        out = evaluate_synthetic_ehr(
            self.train_ehr, self.test_ehr, self.syn_ehr, **self.cols,
            sample_size=10, mode="rf", metrics="privacy",
            n_bootstraps=3, n_runs=3,
        )
        self.assertIn("nnaar", out)
        self.assertNotIn("MLE_Real_Accuracy", out)

    def test_evaluate_utility_only_rf(self):
        out = evaluate_synthetic_ehr(
            self.train_ehr, self.test_ehr, self.syn_ehr, **self.cols,
            mode="rf", metrics="utility", n_bootstraps=3,
        )
        self.assertIn("MLE_Real_Accuracy", out)
        self.assertNotIn("nnaar", out)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            evaluate_synthetic_ehr(
                self.train_ehr, self.test_ehr, self.syn_ehr, **self.cols,
                mode="bad",
            )

    def test_invalid_metrics_raises(self):
        with self.assertRaises(ValueError):
            evaluate_synthetic_ehr(
                self.train_ehr, self.test_ehr, self.syn_ehr, **self.cols,
                metrics="bad",
            )


if __name__ == "__main__":
    unittest.main()

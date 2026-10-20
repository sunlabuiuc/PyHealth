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


def _generate_ehr(
    n_patients, vocab, seed, id_offset=0,
    n_visits_range=(2, 7), n_codes_range=(2, 6),
):
    """Generates a random EHR dataframe with patients drawn from ``vocab``."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_patients):
        pid = str(id_offset + i)
        n_visits = int(rng.integers(*n_visits_range))
        label = int(rng.integers(0, 2))
        for t in range(n_visits):
            n_codes = int(rng.integers(*n_codes_range))
            codes = rng.choice(
                vocab, size=min(n_codes, len(vocab)), replace=False
            )
            for code in codes:
                rows.append(
                    {"id": pid, "time": t,
                     "visit_codes": str(code), "labels": label}
                )
    return pd.DataFrame(rows).astype(
        {"visit_codes": str, "labels": int, "time": int, "id": str}
    )


def _perturb_ehr(df, vocab, frac, seed):
    """Returns a copy of ``df`` with a fraction of codes randomly replaced."""
    rng = np.random.default_rng(seed)
    df = df.copy().reset_index(drop=True)
    mask = rng.random(len(df)) < frac
    new_codes = rng.choice(vocab, size=int(mask.sum()))
    df.loc[mask, "visit_codes"] = [str(c) for c in new_codes]
    return df


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


class TestMetricsBehavior(unittest.TestCase):
    """Sanity checks: metrics should respond to how close synthetic data is.

    Three synthetic datasets are compared against the same real data:

        - ``exact``: an exact copy of the real training data,
        - ``similar``: the training data with ~15% of codes randomly changed,
        - ``different``: independent data over a disjoint code vocabulary.

    A well-behaved metric should rank these consistently (e.g. an exact copy
    is the worst case for privacy and the best case for fidelity).
    """

    VOCAB_REAL = list(range(50))
    VOCAB_DIFF = list(range(100, 150))

    @classmethod
    def setUpClass(cls):
        cls.train_ehr = _generate_ehr(60, cls.VOCAB_REAL, seed=1, id_offset=0)
        cls.test_ehr = _generate_ehr(
            60, cls.VOCAB_REAL, seed=2, id_offset=10000
        )
        cls.syn_exact = cls.train_ehr.copy()
        cls.syn_similar = _perturb_ehr(
            cls.train_ehr, cls.VOCAB_REAL, frac=0.15, seed=3
        )
        cls.syn_different = _generate_ehr(
            60, cls.VOCAB_DIFF, seed=4, id_offset=20000
        )
        cls.cols = dict(
            subject_col=SUBJECT_COL,
            visit_col=VISIT_COL,
            code_col=CODE_COL,
            label_col=LABEL_COL,
        )

    def test_prevalence_orders_by_similarity(self):
        # Prevalence similarity should degrade monotonically: exact > similar
        # > different.
        results = {}
        for name, syn in [
            ("exact", self.syn_exact),
            ("similar", self.syn_similar),
            ("different", self.syn_different),
        ]:
            np.random.seed(0)
            results[name] = compute_prevalence_metrics(
                self.train_ehr, syn,
                subject_col=SUBJECT_COL, code_col=CODE_COL, n_bootstraps=10,
            )

        rmse = {k: v["Prevalence_RMSE"][0] for k, v in results.items()}
        r2 = {k: v["Prevalence_R2"][0] for k, v in results.items()}
        pearson = {k: v["Prevalence_Pearson"][0] for k, v in results.items()}

        # An exact copy has identical code prevalence.
        self.assertAlmostEqual(rmse["exact"], 0.0, places=9)
        self.assertAlmostEqual(r2["exact"], 1.0, places=6)
        self.assertAlmostEqual(pearson["exact"], 1.0, places=6)

        # Error grows / agreement shrinks as synthetic data drifts away.
        self.assertLess(rmse["exact"], rmse["similar"])
        self.assertLess(rmse["similar"], rmse["different"])
        self.assertGreater(r2["exact"], r2["similar"])
        self.assertGreater(r2["similar"], r2["different"])
        self.assertGreaterEqual(pearson["exact"], pearson["similar"])
        self.assertGreater(pearson["similar"], pearson["different"])

    def test_nnaar_flags_exact_copies(self):
        # NNAAR should be high when synthetic data memorizes the training set
        # and near zero otherwise. With proper self-exclusion in the
        # within-set nearest-neighbor search, both exact copies and near-copies
        # (15% perturbed) leak training membership -> high NNAAR, while
        # independent synthetic data (disjoint vocabulary) does not -> ~0.
        nnaar = {}
        for name, syn in [
            ("exact", self.syn_exact),
            ("similar", self.syn_similar),
            ("different", self.syn_different),
        ]:
            np.random.seed(0)
            nnaar[name] = calc_nnaar(
                self.train_ehr, self.test_ehr, syn,
                **self.cols, sample_size=1000, n_runs=3,
            )["nnaar"][0]

        # Memorized / near-memorized synthetic data leaks -> high NNAAR.
        self.assertGreater(nnaar["exact"], 0.3)
        self.assertGreater(nnaar["similar"], 0.3)
        # An exact copy is at least as leaky as a perturbed near-copy.
        self.assertGreaterEqual(nnaar["exact"], nnaar["similar"])
        # Independent synthetic data does not leak -> NNAAR ~ 0.
        self.assertLess(abs(nnaar["different"]), 0.15)
        self.assertGreater(nnaar["exact"], nnaar["different"])
        self.assertGreater(nnaar["similar"], nnaar["different"])

    def test_membership_inference_detects_training_data(self):
        # The attack should succeed when synthetic data is derived from the
        # training set and be near chance when it is unrelated.
        acc = {}
        for name, syn in [
            ("exact", self.syn_exact),
            ("similar", self.syn_similar),
            ("different", self.syn_different),
        ]:
            np.random.seed(0)
            acc[name] = calc_membership_inference(
                self.train_ehr, self.test_ehr, syn,
                **self.cols, num_attack_samples=1000, n_runs=5,
            )["MIA_Accuracy"][0]

        self.assertGreater(acc["exact"], 0.8)
        self.assertGreater(acc["exact"], acc["different"])
        self.assertGreater(acc["similar"], acc["different"])
        self.assertLess(acc["different"], 0.7)

    def test_discriminator_privacy_orders_by_similarity(self):
        # A discriminator easily separates a disjoint-vocabulary synthetic set
        # (accuracy ~1, privacy score ~0) but not data derived from the real
        # data (lower accuracy, higher privacy score).
        score, acc = {}, {}
        for name, syn in [
            ("exact", self.syn_exact),
            ("similar", self.syn_similar),
            ("different", self.syn_different),
        ]:
            np.random.seed(0)
            result = compute_discriminator_privacy(
                train_fn=train_sklearn_model,
                train_ehr=self.train_ehr, test_ehr=self.test_ehr,
                syn_ehr=syn, **self.cols, n_bootstraps=10, model="rf",
            )
            score[name] = result["Privacy_Score"][0]
            acc[name] = result["Privacy_Discriminator_Accuracy"][0]

        # The disjoint-vocabulary set is trivially detected.
        self.assertGreater(acc["different"], 0.8)
        self.assertLess(score["different"], 0.1)
        # Data derived from the real data is harder to flag.
        self.assertGreater(acc["different"], acc["exact"])
        self.assertGreater(acc["different"], acc["similar"])
        self.assertGreater(score["exact"], score["different"])
        self.assertGreater(score["similar"], score["different"])

    def test_mle_orders_by_similarity(self):
        # Utility should be highest for an exact copy and degrade as the
        # synthetic data drifts away from the real data.
        mle = {}
        for name, syn in [
            ("exact", self.syn_exact),
            ("similar", self.syn_similar),
            ("different", self.syn_different),
        ]:
            np.random.seed(0)
            mle[name] = compute_mle(
                train_fn=train_sklearn_model,
                train_ehr=self.train_ehr, test_ehr=self.test_ehr,
                syn_ehr=syn, **self.cols, n_bootstraps=10, model="rf",
            )

        # An exact copy reproduces real utility exactly.
        exact = mle["exact"]
        self.assertAlmostEqual(exact["MLE_Difference"][0], 0.0, places=9)
        self.assertAlmostEqual(exact["MLE_Difference"][1], 0.0, places=9)
        self.assertAlmostEqual(exact["MLE_Ratio"][0], 1.0, places=9)
        self.assertAlmostEqual(
            exact["MLE_Synth_Accuracy"][0], exact["MLE_Real_Accuracy"][0],
            places=9,
        )

        # Synthetic-trained accuracy degrades monotonically.
        diff = {k: abs(v["MLE_Difference"][0]) for k, v in mle.items()}
        ratio = {k: v["MLE_Ratio"][0] for k, v in mle.items()}
        self.assertLessEqual(diff["exact"], diff["similar"])
        self.assertLess(diff["similar"], diff["different"])
        self.assertGreaterEqual(ratio["exact"], ratio["similar"])
        self.assertGreater(ratio["similar"], ratio["different"])
        self.assertLess(ratio["different"], 1.0)


if __name__ == "__main__":
    unittest.main()

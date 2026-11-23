"""Unit tests for the MotionAge participant-level evaluation helpers.

Most tests use small hand-built arrays and tables. One end-to-end test runs
the real pipeline (synthetic NHANES -> dataset -> task -> MotionAge ->
Trainer.inference) on 5 synthetic participants in a temporary directory.
No real NHANES data is used.
"""

import math
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pyhealth.metrics.motionage import (
    age_acceleration,
    aggregate_window_predictions,
    evaluate_feature_sets,
    label_concordance,
    participant_fields,
)
from pyhealth.models.motionage import MotionAgeMapping


class TestAggregateWindowPredictions(unittest.TestCase):
    def test_averages_probabilities_not_logits(self):
        """Mean of 0.5 and 0.99 is 0.745; averaging logits would give ~0.91."""
        table = aggregate_window_predictions(["a", "a"], [0.5, 0.99], [1, 1])
        self.assertAlmostEqual(table.loc[0, "risk"], 0.745)
        mean_logit = (0.0 + math.log(0.99 / 0.01)) / 2
        self.assertNotAlmostEqual(table.loc[0, "risk"], 1 / (1 + math.exp(-mean_logit)))

    def test_counts_windows_and_ignores_order(self):
        ids, prob, label = (
            ["b", "a", "b", "a", "a"],
            [0.2, 0.1, 0.4, 0.3, 0.5],
            [0, 1, 0, 1, 1],
        )
        table = aggregate_window_predictions(ids, prob, label)
        order = [4, 2, 0, 3, 1]
        shuffled = aggregate_window_predictions(
            [ids[i] for i in order], [prob[i] for i in order], [label[i] for i in order]
        )
        pd.testing.assert_frame_equal(table, shuffled)
        self.assertEqual(table["patient_id"].tolist(), ["a", "b"])
        self.assertEqual(table["n_windows"].tolist(), [3, 2])
        self.assertEqual(table["label"].tolist(), [1, 0])
        np.testing.assert_allclose(table["risk"], [0.3, 0.3])

    def test_accepts_column_vectors(self):
        """Trainer.inference returns [n, 1] arrays."""
        table = aggregate_window_predictions(
            ["a", "b"], np.array([[0.2], [0.6]]), np.array([[0.0], [1.0]])
        )
        self.assertEqual(table["label"].tolist(), [0, 1])

    def test_errors(self):
        with self.assertRaises(ValueError):
            aggregate_window_predictions(["a", "a"], [0.1, 0.2], [0, 1])  # mixed label
        with self.assertRaises(ValueError):
            aggregate_window_predictions(["a"], [0.1, 0.2], [0, 0])  # lengths
        with self.assertRaises(ValueError):
            aggregate_window_predictions([], [], [])
        with self.assertRaises(ValueError):
            aggregate_window_predictions(["a"], [1.5], [0])
        with self.assertRaises(ValueError):
            aggregate_window_predictions(["a"], [float("nan")], [0])


class TestParticipantFields(unittest.TestCase):
    def test_one_row_per_participant(self):
        samples = [
            {"patient_id": "b", "age": 45.0, "sex": 1, "income": float("nan")},
            {"patient_id": "a", "age": 61.0, "sex": 2, "income": float("nan")},
            {"patient_id": "b", "age": 45.0, "sex": 1, "income": float("nan")},
        ]
        table = participant_fields(samples, fields=("age", "sex", "income"))
        self.assertEqual(table["patient_id"].tolist(), ["a", "b"])
        self.assertEqual(table["age"].tolist(), [61.0, 45.0])
        self.assertTrue(table["income"].isna().all())

    def test_errors(self):
        varying = [
            {"patient_id": "a", "age": 61.0, "sex": 2},
            {"patient_id": "a", "age": 62.0, "sex": 2},
        ]
        with self.assertRaises(ValueError):
            participant_fields(varying)
        with self.assertRaises(KeyError):
            participant_fields([{"patient_id": "a", "age": 61.0}])


class TestAgeAcceleration(unittest.TestCase):
    def test_values_and_nan(self):
        np.testing.assert_array_equal(
            age_acceleration([52.0, 40.0, np.nan], [50, 45, 60]), [2.0, -5.0, np.nan]
        )

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            age_acceleration([1.0, 2.0], [1.0])

    def test_with_fitted_mapping(self):
        """On the fitted line, MotionAge equals age and acceleration is 0;
        a riskier participant of the same age is older."""
        age = np.repeat(np.arange(40.0, 70.0), 2)
        sex = np.tile([1, 2], age.size // 2)
        risk = 1 / (1 + np.exp(-(-8 + 0.1 * age)))
        mapping = MotionAgeMapping(min_participants=2).fit(risk, age, sex)
        accel = age_acceleration(mapping.transform(risk, sex), age)
        np.testing.assert_allclose(accel, 0.0, atol=1e-6)
        riskier = age_acceleration(mapping.transform([0.5], [1]), [50.0])
        self.assertGreater(riskier[0], 0)


class TestLabelConcordance(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(label_concordance([0, 0, 1, 1], [1, 2, 3, 4]), 1.0)
        self.assertEqual(label_concordance([0, 0, 1, 1], [4, 3, 2, 1]), 0.0)
        self.assertEqual(label_concordance([0, 1, 0, 1], [5, 5, 5, 5]), 0.5)

    def test_equals_auroc(self):
        rng = np.random.default_rng(0)
        label = rng.integers(0, 2, 200)
        score = rng.normal(size=200) + label
        self.assertAlmostEqual(
            label_concordance(label, score), roc_auc_score(label, score)
        )

    def test_missing_scores_and_one_class(self):
        self.assertEqual(label_concordance([0, 1, 1], [1.0, 2.0, np.nan]), 1.0)
        self.assertTrue(math.isnan(label_concordance([1, 1], [1.0, 2.0])))
        self.assertTrue(math.isnan(label_concordance([0, 1], [1.0, np.nan])))

    def test_errors(self):
        with self.assertRaises(ValueError):
            label_concordance([0, 1], [1.0])
        with self.assertRaises(ValueError):
            label_concordance([0, 2], [1.0, 2.0])


class TestEvaluateFeatureSets(unittest.TestCase):
    @staticmethod
    def frame(n: int, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        signal = rng.normal(size=n)
        label = (rng.random(n) < 1 / (1 + np.exp(-3 * signal))).astype(int)
        return pd.DataFrame(
            {"signal": signal, "noise": rng.normal(size=n), "label": label}
        )

    def test_signal_beats_noise_and_delta(self):
        train, test = self.frame(300, 0), self.frame(200, 1)
        table = evaluate_feature_sets(
            train,
            test,
            {"noise": ["noise"], "signal": ["signal", "noise"]},
            baseline="noise",
        ).set_index("feature_set")
        self.assertGreater(table.loc["signal", "test_auroc"], 0.8)
        self.assertLess(abs(table.loc["noise", "test_auroc"] - 0.5), 0.15)
        self.assertEqual(table.loc["noise", "delta_auroc"], 0.0)
        self.assertAlmostEqual(
            table.loc["signal", "delta_auroc"],
            table.loc["signal", "test_auroc"] - table.loc["noise", "test_auroc"],
        )
        self.assertEqual(table.loc["signal", "features"], "signal,noise")
        self.assertEqual(table.loc["signal", "test_events"], test["label"].sum())

    def test_drops_rows_with_missing_features(self):
        train, test = self.frame(100, 0), self.frame(50, 1)
        train.loc[:9, "signal"] = np.nan
        test.loc[:4, "signal"] = np.nan
        row = evaluate_feature_sets(train, test, {"s": ["signal"]}).iloc[0]
        self.assertEqual((row["n_train"], row["n_test"]), (90, 45))

    def test_one_class_test_set_warns(self):
        train, test = self.frame(100, 0), self.frame(20, 1)
        test["label"] = 0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            row = evaluate_feature_sets(train, test, {"s": ["signal"]}).iloc[0]
        self.assertTrue(math.isnan(row["test_auroc"]))
        self.assertTrue(any("one class" in str(w.message) for w in caught))

    def test_errors(self):
        train, test = self.frame(50, 0), self.frame(20, 1)
        with self.assertRaises(KeyError):
            evaluate_feature_sets(train, test, {"x": ["missing"]})
        with self.assertRaises(KeyError):
            evaluate_feature_sets(train, test, {"s": ["signal"]}, baseline="other")
        with self.assertRaises(ValueError):
            evaluate_feature_sets(train, test, {})
        with self.assertRaises(ValueError):
            evaluate_feature_sets(train.assign(label=1), test, {"s": ["signal"]})


class TestEndToEnd(unittest.TestCase):
    """Synthetic NHANES -> dataset -> task -> MotionAge -> participant table."""

    def test_participant_table_from_pipeline(self):
        from pyhealth.datasets import get_dataloader
        from pyhealth.datasets.nhanes import NHANESDataset
        from pyhealth.datasets.nhanes_synthetic import generate_synthetic_nhanes
        from pyhealth.models.motionage import MotionAge
        from pyhealth.tasks.nhanes_mortality_prediction import (
            NHANESMortalityPrediction,
        )
        from pyhealth.trainer import Trainer

        with tempfile.TemporaryDirectory() as tmp:
            # 5 participants: only the first two edge cases (died within 5
            # years, aged 72; died after 5 years, aged 64) are labelled adults
            # with accelerometer data.
            truth = generate_synthetic_nhanes(
                f"{tmp}/raw", n_participants=5, n_days=2
            ).participants
            dataset = NHANESDataset(root=f"{tmp}/raw", cache_dir=f"{tmp}/cache")
            task = NHANESMortalityPrediction(seq_len=288, covariate_set=None)
            samples = dataset.set_task(task)
            model = MotionAge(samples, hidden_dim=8, num_layers=1)
            trainer = Trainer(model=model, enable_logging=False)
            loader = get_dataloader(samples, batch_size=4, shuffle=False)
            trainer.train(loader, epochs=1)
            y_true, y_prob, _, ids = trainer.inference(loader, return_patient_ids=True)
            table = aggregate_window_predictions(ids, y_prob, y_true).merge(
                participant_fields(samples), on="patient_id", validate="1:1"
            )

        expected = truth.iloc[:2].assign(patient_id=lambda f: f["SEQN"].astype(str))
        self.assertEqual(table["patient_id"].tolist(), expected["patient_id"].tolist())
        self.assertEqual(table["label"].tolist(), [1, 0])
        self.assertEqual(table["age"].tolist(), expected["age"].astype(float).tolist())
        self.assertEqual(table["sex"].tolist(), expected["sex"].tolist())
        self.assertEqual(int(table["n_windows"].sum()), len(samples))
        self.assertTrue(table["risk"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()

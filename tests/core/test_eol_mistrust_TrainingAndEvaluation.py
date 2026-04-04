import importlib.util
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.errors import MergeError

try:
    from sklearn.exceptions import ConvergenceWarning  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    ConvergenceWarning = None


def _load_dataset_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "pyhealth" / "datasets" / "eol_mistrust.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pyhealth.datasets.eol_mistrust_training_eval_tests",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_model_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "pyhealth" / "models" / "eol_mistrust.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pyhealth.models.eol_mistrust_training_eval_tests",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _RecordingProbEstimator:
    def __init__(self, probabilities, coef_values=None):
        self.probabilities = list(probabilities)
        self.coef_values = coef_values
        self.fit_X = None
        self.fit_y = None
        self.predicted_shape = None
        self.coef_ = None

    def fit(self, X, y):
        self.fit_X = X.copy()
        self.fit_y = pd.Series(y).copy()
        if self.coef_values is None:
            self.coef_ = [[0.1] * X.shape[1]]
        else:
            self.coef_ = [list(self.coef_values)]
        return self

    def predict_proba(self, X):
        probs = self.probabilities[: len(X)]
        matrix = [[1.0 - prob, prob] for prob in probs]
        self.predicted_shape = (len(matrix), len(matrix[0]) if matrix else 0)
        return matrix


class _EstimatorFactorySequence:
    def __init__(self, estimator_builders):
        self.estimator_builders = list(estimator_builders)
        self.created = []

    def __call__(self):
        index = min(len(self.created), len(self.estimator_builders) - 1)
        estimator = self.estimator_builders[index]()
        self.created.append(estimator)
        return estimator


class _SplitRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, X, y, test_size, random_state):
        features = X.reset_index(drop=True)
        labels = pd.Series(y).reset_index(drop=True)
        n_rows = len(features)
        n_test = max(1, int(round(n_rows * test_size)))
        n_train = max(1, n_rows - n_test)
        if n_train + n_test > n_rows:
            n_test = n_rows - n_train
        if n_test == 0:
            n_test = 1
            n_train = max(0, n_rows - 1)

        self.calls.append(
            {
                "random_state": random_state,
                "test_size": test_size,
                "n_rows": n_rows,
                "n_train": n_train,
                "n_test": n_test,
                "train_indices": list(range(n_train)),
                "test_indices": list(range(n_train, n_train + n_test)),
                "train_hadm_ids": list(features.iloc[:n_train]["hadm_id"]) if "hadm_id" in features.columns else [],
                "test_hadm_ids": list(features.iloc[n_train : n_train + n_test]["hadm_id"]) if "hadm_id" in features.columns else [],
            }
        )

        return (
            features.iloc[:n_train].copy(),
            features.iloc[n_train : n_train + n_test].copy(),
            labels.iloc[:n_train].copy(),
            labels.iloc[n_train : n_train + n_test].copy(),
        )


class _AUCRecorder:
    def __init__(self, value=0.75):
        self.value = float(value)
        self.calls = []

    def __call__(self, y_true, y_prob):
        self.calls.append(
            {
                "y_true": list(pd.Series(y_true)),
                "y_prob": list(pd.Series(y_prob)),
                "value": self.value,
            }
        )
        return self.value


class _AUCSequenceRecorder:
    def __init__(self, values):
        self.values = [float(value) for value in values]
        self.calls = []

    def __call__(self, y_true, y_prob):
        index = len(self.calls)
        value = self.values[index]
        self.calls.append(
            {
                "y_true": list(pd.Series(y_true)),
                "y_prob": list(pd.Series(y_prob)),
                "value": value,
            }
        )
        return value


class _DeterministicSplitRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, X, y, test_size, random_state):
        features = X.reset_index(drop=True)
        labels = pd.Series(y).reset_index(drop=True)
        rng = np.random.RandomState(random_state)
        indices = np.arange(len(features))
        if len(indices) > 0:
            rng.shuffle(indices)
        n_test = max(1, int(round(len(indices) * test_size))) if len(indices) else 0
        n_train = max(0, len(indices) - n_test)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        train_features = features.iloc[train_idx].reset_index(drop=True)
        test_features = features.iloc[test_idx].reset_index(drop=True)
        train_labels = labels.iloc[train_idx].reset_index(drop=True)
        test_labels = labels.iloc[test_idx].reset_index(drop=True)
        self.calls.append(
            {
                "random_state": random_state,
                "train_indices": list(map(int, train_idx)),
                "test_indices": list(map(int, test_idx)),
                "train_hadm_ids": list(train_features["hadm_id"]) if "hadm_id" in train_features.columns else [],
                "test_hadm_ids": list(test_features["hadm_id"]) if "hadm_id" in test_features.columns else [],
                "n_rows": len(features),
                "n_train": len(train_features),
                "n_test": len(test_features),
            }
        )
        return train_features, test_features, train_labels, test_labels


class _FailingEstimator:
    def fit(self, X, y):
        del X, y
        raise RuntimeError("estimator fit failed")


class TestEOLMistrustTrainingAndEvaluation(unittest.TestCase):
    """Synthetic training/evaluation contract tests for the EOL mistrust workflow."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = _load_dataset_module()
        cls.model = _load_model_module()

    def setUp(self):
        hadm_ids = list(range(1001, 1013))
        self.feature_matrix = pd.DataFrame(
            [
                {
                    "hadm_id": hadm_id,
                    "Riker-SAS Scale Score: Agitated": int(index % 2 == 0),
                    "Education Readiness: No": int(index % 3 == 0),
                    "Pain Level: 7-Mod to Severe": int(index % 4 in {0, 1}),
                    "Richmond-RAS Scale: 0 Alert and Calm": int(index % 2 == 1),
                    "Restraint Device: Soft Limb": int(index % 3 == 1),
                    "Pain Present: No": int(index % 4 in {2, 3}),
                }
                for index, hadm_id in enumerate(hadm_ids)
            ]
        )
        self.note_labels = pd.DataFrame(
            [
                {
                    "hadm_id": hadm_id,
                    "noncompliance_label": int(index % 2 == 0),
                    "autopsy_label": int(index % 3 == 0),
                }
                for index, hadm_id in enumerate(hadm_ids)
            ]
        )
        self.note_corpus = pd.DataFrame(
            [
                {
                    "hadm_id": hadm_id,
                    "note_text": (
                        "Patient is non-complian and refused medication."
                        if index % 2 == 0
                        else "Patient remained calm. Date:[**5-1-18**] discussed."
                    ),
                }
                for index, hadm_id in enumerate(hadm_ids)
            ]
        )
        self.demographics = pd.DataFrame(
            [
                {"hadm_id": hadm_id, "race": "WHITE" if index < 6 else "BLACK"}
                for index, hadm_id in enumerate(hadm_ids)
            ]
        )
        self.eol_cohort = pd.DataFrame(
            [
                {"hadm_id": hadm_id, "race": "WHITE" if index < 6 else "BLACK"}
                for index, hadm_id in enumerate(hadm_ids[:10])
            ]
        )
        self.treatment_totals = pd.DataFrame(
            [
                {
                    "hadm_id": hadm_id,
                    "total_vent_min": float(200 + 50 * index),
                    "total_vaso_min": float(20 + 10 * (index % 5)),
                }
                for index, hadm_id in enumerate(hadm_ids[:10])
            ]
        )
        self.acuity_scores = pd.DataFrame(
            [
                {
                    "hadm_id": hadm_id,
                    "oasis": float(15 + index),
                    "sapsii": float(25 + index),
                }
                for index, hadm_id in enumerate(hadm_ids)
            ]
        )
        self.final_model_table = pd.DataFrame(
            [
                {
                    "hadm_id": hadm_id,
                    "age": float(40 + index),
                    "los_days": float(1.5 + 0.25 * index),
                    "gender_f": int(index % 2 == 1),
                    "gender_m": int(index % 2 == 0),
                    "insurance_private": int(index % 3 == 0),
                    "insurance_public": int(index % 3 == 1),
                    "insurance_self_pay": int(index % 3 == 2),
                    "race_white": int(index < 6),
                    "race_black": int(index >= 6),
                    "race_asian": 0,
                    "race_hispanic": 0,
                    "race_native_american": 0,
                    "race_other": 0,
                    "noncompliance_score_z": float(-1.5 + 0.3 * index),
                    "autopsy_score_z": float(1.2 - 0.2 * index),
                    "negative_sentiment_score_z": float(-0.9 + 0.18 * index),
                    "left_ama": int(index % 2 == 0),
                    "code_status_dnr_dni_cmo": int(index % 3 == 0),
                    "in_hospital_mortality": int(index % 4 == 0),
                }
                for index, hadm_id in enumerate(hadm_ids)
            ]
        )

    def _pending_real_data(self, requirement: str) -> None:
        self.skipTest(requirement)

    def test_proxy_metric_training_inputs_align_rows_and_binary_labels(self):
        factory_non = _EstimatorFactorySequence(
            [lambda: _RecordingProbEstimator([0.8] * len(self.feature_matrix))]
        )
        non_model = self.model.fit_proxy_mistrust_model(
            self.feature_matrix,
            self.note_labels,
            "noncompliance_label",
            estimator_factory=factory_non,
        )

        self.assertEqual(non_model.fit_X.shape[0], len(self.feature_matrix))
        self.assertEqual(len(non_model.fit_y), len(self.feature_matrix))
        self.assertTrue(set(non_model.fit_y.unique()).issubset({0, 1}))

        factory_auto = _EstimatorFactorySequence(
            [lambda: _RecordingProbEstimator([0.3] * len(self.feature_matrix))]
        )
        auto_model = self.model.fit_proxy_mistrust_model(
            self.feature_matrix,
            self.note_labels,
            "autopsy_label",
            estimator_factory=factory_auto,
        )

        self.assertEqual(auto_model.fit_X.shape[0], len(self.feature_matrix))
        self.assertEqual(len(auto_model.fit_y), len(self.feature_matrix))
        self.assertTrue(set(auto_model.fit_y.unique()).issubset({0, 1}))

    def test_proxy_metric_predict_proba_outputs_have_two_columns_and_unit_interval(self):
        non_estimator = _RecordingProbEstimator([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.55, 0.45, 0.65, 0.35])
        auto_estimator = _RecordingProbEstimator([0.2, 0.7, 0.3, 0.8, 0.4, 0.6, 0.5, 0.55, 0.45, 0.65, 0.35, 0.75])

        non_scores = self.model.build_proxy_probability_scores(
            self.feature_matrix,
            self.note_labels,
            "noncompliance_label",
            estimator_factory=lambda: non_estimator,
        )
        auto_scores = self.model.build_proxy_probability_scores(
            self.feature_matrix,
            self.note_labels,
            "autopsy_label",
            estimator_factory=lambda: auto_estimator,
        )

        self.assertEqual(non_estimator.predicted_shape, (len(self.feature_matrix), 2))
        self.assertEqual(auto_estimator.predicted_shape, (len(self.feature_matrix), 2))
        self.assertTrue(non_scores["noncompliance_score"].between(0.0, 1.0).all())
        self.assertTrue(auto_scores["autopsy_score"].between(0.0, 1.0).all())

    def test_synthetic_proxy_models_converge_without_warning_with_default_max_iter(self):
        if ConvergenceWarning is None:
            self.skipTest("scikit-learn is unavailable in the current environment.")

        feature_matrix = pd.DataFrame(
            [
                {
                    "hadm_id": 2000 + index,
                    "Riker-SAS Scale Score: Agitated": int(index % 2 == 0),
                    "Education Readiness: No": int(index % 2 == 0),
                    "Pain Level: 7-Mod to Severe": int(index % 3 == 0),
                    "Richmond-RAS Scale: 0 Alert and Calm": int(index % 2 == 1),
                    "Restraint Device: Soft Limb": int(index % 4 in {0, 1}),
                    "Pain Present: No": int(index % 4 in {2, 3}),
                }
                for index in range(40)
            ]
        )
        note_labels = pd.DataFrame(
            [
                {
                    "hadm_id": 2000 + index,
                    "noncompliance_label": int(index % 2 == 0),
                    "autopsy_label": int(index % 4 in {0, 1}),
                }
                for index in range(40)
            ]
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.model.fit_proxy_mistrust_model(feature_matrix, note_labels, "noncompliance_label")
            self.model.fit_proxy_mistrust_model(feature_matrix, note_labels, "autopsy_label")

        convergence_warnings = [
            warning for warning in caught if isinstance(warning.message, ConvergenceWarning)
        ]
        self.assertEqual(convergence_warnings, [])

    def test_mistrust_score_arrays_are_finite_and_z_normalized(self):
        factory = _EstimatorFactorySequence(
            [
                lambda: _RecordingProbEstimator([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.55, 0.45, 0.65, 0.35]),
                lambda: _RecordingProbEstimator([0.2, 0.7, 0.3, 0.8, 0.4, 0.6, 0.5, 0.55, 0.45, 0.65, 0.35, 0.75]),
            ]
        )
        scores = self.model.build_mistrust_score_table(
            self.feature_matrix,
            self.note_labels,
            self.note_corpus,
            estimator_factory=factory,
            sentiment_fn=lambda text: (-0.6 if "non" in text else 0.2, 0.0),
        )

        for column in self.model.MISTRUST_SCORE_COLUMNS:
            values = pd.to_numeric(scores[column], errors="coerce")
            self.assertFalse(values.isna().any())
            self.assertTrue(np.isfinite(values).all())
            self.assertLess(abs(float(values.mean())), 0.01)
            self.assertGreaterEqual(float(values.std(ddof=0)), 0.99)
            self.assertLessEqual(float(values.std(ddof=0)), 1.01)

    def test_synthetic_race_gap_analysis_matches_expected_directional_pattern(self):
        mistrust_scores = pd.DataFrame(
            [
                {"hadm_id": 1, "noncompliance_score_z": -1.4, "autopsy_score_z": -0.1, "negative_sentiment_score_z": -1.1},
                {"hadm_id": 2, "noncompliance_score_z": -1.0, "autopsy_score_z": 0.0, "negative_sentiment_score_z": -0.8},
                {"hadm_id": 3, "noncompliance_score_z": -0.8, "autopsy_score_z": 0.2, "negative_sentiment_score_z": -0.6},
                {"hadm_id": 4, "noncompliance_score_z": -0.6, "autopsy_score_z": 0.3, "negative_sentiment_score_z": -0.5},
                {"hadm_id": 5, "noncompliance_score_z": 0.8, "autopsy_score_z": 0.1, "negative_sentiment_score_z": 0.7},
                {"hadm_id": 6, "noncompliance_score_z": 1.0, "autopsy_score_z": 0.0, "negative_sentiment_score_z": 0.9},
                {"hadm_id": 7, "noncompliance_score_z": 1.2, "autopsy_score_z": 0.2, "negative_sentiment_score_z": 1.1},
                {"hadm_id": 8, "noncompliance_score_z": 1.4, "autopsy_score_z": 0.3, "negative_sentiment_score_z": 1.3},
            ]
        )
        demographics = pd.DataFrame(
            [
                {"hadm_id": 1, "race": "WHITE"},
                {"hadm_id": 2, "race": "WHITE"},
                {"hadm_id": 3, "race": "WHITE"},
                {"hadm_id": 4, "race": "WHITE"},
                {"hadm_id": 5, "race": "BLACK"},
                {"hadm_id": 6, "race": "BLACK"},
                {"hadm_id": 7, "race": "BLACK"},
                {"hadm_id": 8, "race": "BLACK"},
            ]
        )

        results = self.model.run_race_gap_analysis(mistrust_scores, demographics).set_index("metric")
        self.assertTrue(results.loc["noncompliance_score_z", "black_median_higher"])
        self.assertTrue(results.loc["negative_sentiment_score_z", "black_median_higher"])
        self.assertLess(float(results.loc["noncompliance_score_z", "pvalue"]), 0.05)
        self.assertLess(float(results.loc["negative_sentiment_score_z", "pvalue"]), 0.05)
        self.assertGreater(float(results.loc["autopsy_score_z", "pvalue"]), 0.05)

    def test_downstream_feature_configurations_match_required_widths_and_are_finite(self):
        configs = self.model.get_downstream_feature_configurations()
        self.assertEqual(len(configs["Baseline"]), 7)
        self.assertEqual(len(configs["Baseline + ALL"]), 16)

        for columns in configs.values():
            values = self.final_model_table[columns].apply(pd.to_numeric, errors="coerce")
            self.assertFalse(values.isna().any().any())
            self.assertTrue(np.isfinite(values.to_numpy(dtype=float)).all())

    def test_downstream_evaluation_uses_100_random_states_and_approximate_60_40_splits(self):
        split_recorder = _SplitRecorder()
        auc_recorder = _AUCRecorder(0.76)

        results = self.model.evaluate_downstream_predictions(
            self.final_model_table,
            feature_configurations={"Baseline": self.model.BASELINE_FEATURE_COLUMNS},
            task_map={"Left AMA": "left_ama"},
            estimator_factory=lambda: _RecordingProbEstimator([0.2, 0.8, 0.3, 0.7, 0.4]),
            split_fn=split_recorder,
            auc_fn=auc_recorder,
            repetitions=100,
        )

        self.assertEqual(len(split_recorder.calls), 100)
        self.assertEqual({call["random_state"] for call in split_recorder.calls}, set(range(100)))
        self.assertTrue(all(call["test_size"] == 0.4 for call in split_recorder.calls))
        self.assertTrue(all(call["n_train"] == 7 for call in split_recorder.calls))
        self.assertTrue(all(call["n_test"] == 5 for call in split_recorder.calls))
        self.assertEqual(int(results.loc[0, "n_repeats"]), 100)

    def test_downstream_models_run_without_warning_and_auc_values_are_non_degenerate(self):
        split_recorder = _SplitRecorder()
        auc_recorder = _AUCRecorder(0.74)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = self.model.evaluate_downstream_predictions(
                self.final_model_table,
                feature_configurations={"Baseline + ALL": self.model.get_downstream_feature_configurations()["Baseline + ALL"]},
                task_map={"Code Status": "code_status_dnr_dni_cmo"},
                estimator_factory=lambda: _RecordingProbEstimator([0.2, 0.8, 0.3, 0.7, 0.4]),
                split_fn=split_recorder,
                auc_fn=auc_recorder,
                repetitions=100,
            )

        if ConvergenceWarning is not None:
            convergence_warnings = [
                warning for warning in caught if isinstance(warning.message, ConvergenceWarning)
            ]
            self.assertEqual(convergence_warnings, [])
        self.assertTrue(all(0.5 <= call["value"] <= 1.0 for call in auc_recorder.calls))
        self.assertTrue(0.5 <= float(results.loc[0, "auc_mean"]) <= 1.0)

    def test_training_and_evaluation_pipeline_returns_expected_sections(self):
        factory = _EstimatorFactorySequence(
            [
                lambda: _RecordingProbEstimator([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.55, 0.45, 0.65, 0.35]),
                lambda: _RecordingProbEstimator([0.2, 0.7, 0.3, 0.8, 0.4, 0.6, 0.5, 0.55, 0.45, 0.65, 0.35, 0.75]),
            ]
        )
        outputs = self.model.run_full_eol_mistrust_modeling(
            feature_matrix=self.feature_matrix,
            note_labels=self.note_labels,
            note_corpus=self.note_corpus,
            demographics=self.demographics,
            eol_cohort=self.eol_cohort,
            treatment_totals=self.treatment_totals,
            acuity_scores=self.acuity_scores,
            final_model_table=self.final_model_table,
            estimator_factory=factory,
            sentiment_fn=lambda text: (-0.6 if "non" in text else 0.2, 0.0),
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.77),
            repetitions=5,
        )

        self.assertEqual(
            set(outputs.keys()),
            {
                "mistrust_scores",
                "feature_weight_summaries",
                "race_gap_results",
                "race_treatment_results",
                "race_treatment_by_acuity_results",
                "trust_treatment_results",
                "trust_treatment_by_acuity_results",
                "acuity_correlations",
                "downstream_auc_results",
            },
        )

    def test_proxy_metric_models_use_full_all_cohort_and_never_call_split_function(self):
        factory = _EstimatorFactorySequence(
            [
                lambda: _RecordingProbEstimator([0.9] * len(self.feature_matrix)),
                lambda: _RecordingProbEstimator([0.1] * len(self.feature_matrix)),
            ]
        )

        with patch.object(self.model, "train_test_split", side_effect=AssertionError("split should not be used")):
            scores = self.model.build_mistrust_score_table(
                self.feature_matrix,
                self.note_labels,
                self.note_corpus,
                estimator_factory=factory,
                sentiment_fn=lambda text: (-0.6 if "non" in text else 0.2, 0.0),
            )

        self.assertEqual(scores["hadm_id"].tolist(), self.feature_matrix["hadm_id"].tolist())
        self.assertEqual(factory.created[0].fit_X.shape[0], len(self.feature_matrix))
        self.assertEqual(factory.created[1].fit_X.shape[0], len(self.feature_matrix))

    def test_sentiment_metric_does_not_instantiate_or_fit_any_estimator(self):
        with patch.object(self.model, "LogisticRegression", side_effect=AssertionError("estimator should not be used")):
            scores = self.model.build_negative_sentiment_mistrust_scores(
                self.note_corpus,
                sentiment_fn=lambda text: (-0.6 if "non" in text else 0.2, 0.0),
            )
        self.assertEqual(scores.columns.tolist(), ["hadm_id", "negative_sentiment_score"])

    def test_downstream_train_and_test_partitions_are_disjoint_for_every_run(self):
        split_recorder = _DeterministicSplitRecorder()
        self.model.evaluate_downstream_predictions(
            self.final_model_table,
            feature_configurations={"Baseline": self.model.BASELINE_FEATURE_COLUMNS},
            task_map={"Left AMA": "left_ama"},
            estimator_factory=lambda: _RecordingProbEstimator([0.2, 0.8, 0.3, 0.7, 0.4]),
            split_fn=split_recorder,
            auc_fn=_AUCRecorder(0.76),
            repetitions=10,
        )

        for call in split_recorder.calls:
            self.assertTrue(set(call["train_indices"]).isdisjoint(set(call["test_indices"])))

    def test_downstream_splits_are_reproducible_for_same_random_state(self):
        recorder_one = _DeterministicSplitRecorder()
        recorder_two = _DeterministicSplitRecorder()
        kwargs = {
            "final_model_table": self.final_model_table,
            "feature_configurations": {"Baseline": self.model.BASELINE_FEATURE_COLUMNS},
            "task_map": {"Left AMA": "left_ama"},
            "estimator_factory": lambda: _RecordingProbEstimator([0.2, 0.8, 0.3, 0.7, 0.4]),
            "auc_fn": _AUCRecorder(0.74),
            "repetitions": 5,
        }
        self.model.evaluate_downstream_predictions(split_fn=recorder_one, **kwargs)
        self.model.evaluate_downstream_predictions(split_fn=recorder_two, **kwargs)

        self.assertEqual(recorder_one.calls, recorder_two.calls)

    def test_downstream_drops_missing_rows_before_splitting_and_keeps_n_stable(self):
        table = self.final_model_table.copy()
        table.loc[0, "left_ama"] = np.nan
        table.loc[1, "age"] = np.nan
        split_recorder = _SplitRecorder()

        results = self.model.evaluate_downstream_predictions(
            table,
            feature_configurations={"Baseline": self.model.BASELINE_FEATURE_COLUMNS},
            task_map={"Left AMA": "left_ama"},
            estimator_factory=lambda: _RecordingProbEstimator([0.2, 0.8, 0.3, 0.7]),
            split_fn=split_recorder,
            auc_fn=_AUCRecorder(0.71),
            repetitions=10,
        )

        self.assertTrue(all(call["n_rows"] == 10 for call in split_recorder.calls))
        self.assertEqual(int(results.loc[0, "n_rows"]), 10)

    def test_feature_configurations_do_not_include_any_target_column(self):
        target_columns = set(self.model.get_downstream_task_map().values())
        for columns in self.model.get_downstream_feature_configurations().values():
            self.assertTrue(target_columns.isdisjoint(set(columns)))

    def test_proxy_feature_weight_summary_preserves_feature_to_coefficient_alignment(self):
        coef_values = [0.4, 0.1, -0.3, -0.5, 0.2, 0.0]
        summary = self.model.build_noncompliance_feature_weight_summary(
            self.feature_matrix,
            self.note_labels,
            estimator_factory=lambda: _RecordingProbEstimator([0.8] * len(self.feature_matrix), coef_values=coef_values),
            top_n=6,
        )

        by_feature = summary["all"].set_index("feature")["weight"].to_dict()
        feature_columns = [column for column in self.feature_matrix.columns if column != "hadm_id"]
        expected = dict(zip(feature_columns, coef_values))
        self.assertEqual(by_feature, expected)

    def test_weight_aggregation_output_schema_placeholder(self):
        self.skipTest(
            "Average coefficient aggregation across 100 downstream runs is not yet exposed by a dedicated public API."
        )

    def test_mistrust_score_arrays_preserve_hadm_alignment_after_training_and_normalization(self):
        factory = _EstimatorFactorySequence(
            [
                lambda: _RecordingProbEstimator([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.55, 0.45, 0.65, 0.35]),
                lambda: _RecordingProbEstimator([0.2, 0.7, 0.3, 0.8, 0.4, 0.6, 0.5, 0.55, 0.45, 0.65, 0.35, 0.75]),
            ]
        )
        scores = self.model.build_mistrust_score_table(
            self.feature_matrix,
            self.note_labels,
            self.note_corpus,
            estimator_factory=factory,
            sentiment_fn=lambda text: (-0.6 if "non" in text else 0.2, 0.0),
        )
        self.assertEqual(scores["hadm_id"].tolist(), sorted(self.feature_matrix["hadm_id"].tolist()))

    def test_mistrust_score_normalization_happens_after_raw_score_generation(self):
        non = self.model.build_proxy_probability_scores(
            self.feature_matrix,
            self.note_labels,
            "noncompliance_label",
            estimator_factory=lambda: _RecordingProbEstimator([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.55, 0.45, 0.65, 0.35]),
        )
        auto = self.model.build_proxy_probability_scores(
            self.feature_matrix,
            self.note_labels,
            "autopsy_label",
            estimator_factory=lambda: _RecordingProbEstimator([0.2, 0.7, 0.3, 0.8, 0.4, 0.6, 0.5, 0.55, 0.45, 0.65, 0.35, 0.75]),
        )
        sentiment = self.model.build_negative_sentiment_mistrust_scores(
            self.note_corpus,
            sentiment_fn=lambda text: (-0.6 if "non" in text else 0.2, 0.0),
        )
        raw = non.merge(auto, on="hadm_id").merge(sentiment, on="hadm_id")
        manual = self.model.z_normalize_scores(
            raw,
            columns=["noncompliance_score", "autopsy_score", "negative_sentiment_score"],
        ).rename(
            columns={
                "noncompliance_score": "noncompliance_score_z",
                "autopsy_score": "autopsy_score_z",
                "negative_sentiment_score": "negative_sentiment_score_z",
            }
        )

        combined = self.model.build_mistrust_score_table(
            self.feature_matrix,
            self.note_labels,
            self.note_corpus,
            estimator_factory=_EstimatorFactorySequence(
                [
                    lambda: _RecordingProbEstimator([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.55, 0.45, 0.65, 0.35]),
                    lambda: _RecordingProbEstimator([0.2, 0.7, 0.3, 0.8, 0.4, 0.6, 0.5, 0.55, 0.45, 0.65, 0.35, 0.75]),
                ]
            ),
            sentiment_fn=lambda text: (-0.6 if "non" in text else 0.2, 0.0),
        )
        pd.testing.assert_frame_equal(manual.reset_index(drop=True), combined.reset_index(drop=True))

    def test_mann_whitney_is_called_two_sided(self):
        calls = []

        def _fake_mannwhitneyu(left, right, alternative):
            calls.append({"left": list(left), "right": list(right), "alternative": alternative})

            class _Result:
                statistic = 1.0
                pvalue = 0.04

            return _Result()

        with patch.object(self.model, "mannwhitneyu", side_effect=_fake_mannwhitneyu):
            self.model.run_race_gap_analysis(
                pd.DataFrame(
                    [
                        {"hadm_id": 1, "noncompliance_score_z": 0.1, "autopsy_score_z": 0.2, "negative_sentiment_score_z": 0.3},
                        {"hadm_id": 2, "noncompliance_score_z": 0.4, "autopsy_score_z": 0.5, "negative_sentiment_score_z": 0.6},
                    ]
                ),
                pd.DataFrame(
                    [
                        {"hadm_id": 1, "race": "WHITE"},
                        {"hadm_id": 2, "race": "BLACK"},
                    ]
                ),
                score_columns=["noncompliance_score_z"],
            )

        self.assertEqual([call["alternative"] for call in calls], ["two-sided"])

    def test_pearson_is_called_after_inner_join_and_pairwise_dropna(self):
        calls = []

        def _fake_pearsonr(left, right):
            calls.append((list(left), list(right)))
            return 0.2, 0.5

        mistrust = pd.DataFrame(
            [
                {"hadm_id": 1, "noncompliance_score_z": 0.1},
                {"hadm_id": 2, "noncompliance_score_z": np.nan},
                {"hadm_id": 3, "noncompliance_score_z": 0.3},
                {"hadm_id": 4, "noncompliance_score_z": 0.4},
            ]
        )
        acuity = pd.DataFrame(
            [
                {"hadm_id": 2, "oasis": 20.0},
                {"hadm_id": 3, "oasis": 30.0},
                {"hadm_id": 4, "oasis": 40.0},
                {"hadm_id": 5, "oasis": 50.0},
            ]
        )

        with patch.object(self.model, "pearsonr", side_effect=_fake_pearsonr):
            self.model.run_acuity_control_analysis(
                mistrust,
                acuity,
                score_columns=["noncompliance_score_z"],
                acuity_columns=("oasis",),
            )

        self.assertEqual(calls, [([0.3, 0.4], [30.0, 40.0])])

    def test_cdf_plot_helper_placeholder(self):
        self.skipTest(
            "CDF visualization helpers are not yet exposed by a public plotting API in the EOL mistrust model module."
        )

    def test_trust_stratification_direction_is_locked_to_top_n_as_high_mistrust(self):
        mistrust = pd.DataFrame(
            [
                {"hadm_id": 1, "noncompliance_score_z": 0.9},
                {"hadm_id": 2, "noncompliance_score_z": 0.8},
                {"hadm_id": 3, "noncompliance_score_z": 0.2},
                {"hadm_id": 4, "noncompliance_score_z": 0.1},
            ]
        )
        eol = pd.DataFrame(
            [
                {"hadm_id": 1, "race": "WHITE"},
                {"hadm_id": 2, "race": "BLACK"},
                {"hadm_id": 3, "race": "WHITE"},
                {"hadm_id": 4, "race": "BLACK"},
            ]
        )
        treatments = pd.DataFrame(
            [
                {"hadm_id": 1, "total_vent_min": 100.0, "total_vaso_min": 10.0},
                {"hadm_id": 2, "total_vent_min": 90.0, "total_vaso_min": 9.0},
                {"hadm_id": 3, "total_vent_min": 30.0, "total_vaso_min": 3.0},
                {"hadm_id": 4, "total_vent_min": 20.0, "total_vaso_min": 2.0},
            ]
        )

        result = self.model.run_trust_based_treatment_analysis(
            eol,
            mistrust,
            treatments,
            score_columns=["noncompliance_score_z"],
            treatment_columns=("total_vent_min",),
            group_sizes={"total_vent_min": 2},
        )
        row = result.iloc[0]
        self.assertEqual(int(row["n_high"]), 2)
        self.assertEqual(int(row["n_low"]), 2)
        self.assertEqual(float(row["median_high"]), 95.0)
        self.assertEqual(float(row["median_low"]), 25.0)
        self.assertEqual(float(row["median_gap"]), 70.0)

    def test_all_six_configs_are_evaluated_for_all_three_tasks_even_with_fewer_usable_rows(self):
        table = self.final_model_table.copy()
        table.loc[0, "code_status_dnr_dni_cmo"] = np.nan
        table.loc[1, "autopsy_score_z"] = np.nan

        results = self.model.evaluate_downstream_predictions(
            table,
            estimator_factory=lambda: _RecordingProbEstimator([0.2, 0.8, 0.3, 0.7, 0.4]),
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.72),
            repetitions=1,
        )

        self.assertEqual(len(results), 18)
        self.assertEqual(set(results["configuration"]), set(self.model.get_downstream_feature_configurations()))
        self.assertEqual(set(results["task"]), set(self.model.get_downstream_task_map()))
        by_task = results.groupby("task")["n_rows"].max().to_dict()
        self.assertGreater(by_task["Left AMA"], results.loc[results["task"] == "Code Status", "n_rows"].min())

    def test_downstream_auc_output_schema_is_fixed_and_complete(self):
        results = self.model.evaluate_downstream_predictions(
            self.final_model_table,
            feature_configurations={"Baseline": self.model.BASELINE_FEATURE_COLUMNS},
            task_map={"Left AMA": "left_ama"},
            estimator_factory=lambda: _RecordingProbEstimator([0.2, 0.8, 0.3, 0.7, 0.4]),
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.71),
            repetitions=2,
        )
        self.assertEqual(
            results.columns.tolist(),
            [
                "task",
                "configuration",
                "target_column",
                "n_rows",
                "n_features",
                "n_repeats",
                "n_valid_auc",
                "auc_mean",
                "auc_std",
            ],
        )

    def test_average_coefficient_output_schema_placeholder(self):
        self.skipTest(
            "Average coefficient summaries across downstream runs are not yet emitted by a dedicated public API."
        )

    def test_downstream_auc_uses_test_set_probabilities_not_labels_or_train_outputs(self):
        split_recorder = _SplitRecorder()
        auc_recorder = _AUCRecorder(0.79)

        self.model.evaluate_downstream_predictions(
            self.final_model_table,
            feature_configurations={"Baseline": self.model.BASELINE_FEATURE_COLUMNS},
            task_map={"Left AMA": "left_ama"},
            estimator_factory=lambda: _RecordingProbEstimator([0.21, 0.81, 0.31, 0.71, 0.41]),
            split_fn=split_recorder,
            auc_fn=auc_recorder,
            repetitions=1,
        )

        self.assertEqual(auc_recorder.calls[0]["y_prob"], [0.21, 0.81, 0.31, 0.71, 0.41][: split_recorder.calls[0]["n_test"]])
        self.assertNotEqual(auc_recorder.calls[0]["y_prob"], auc_recorder.calls[0]["y_true"])

    def test_single_class_splits_are_counted_via_n_valid_auc(self):
        table = self.final_model_table.copy()
        table["left_ama"] = 0
        results = self.model.evaluate_downstream_predictions(
            table,
            feature_configurations={"Baseline": self.model.BASELINE_FEATURE_COLUMNS},
            task_map={"Left AMA": "left_ama"},
            estimator_factory=lambda: _RecordingProbEstimator([0.2, 0.8, 0.3, 0.7, 0.4]),
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.7),
            repetitions=5,
        )
        row = results.iloc[0]
        self.assertEqual(int(row["n_valid_auc"]), 0)
        self.assertTrue(pd.isna(row["auc_mean"]))
        self.assertTrue(pd.isna(row["auc_std"]))

    def test_proxy_training_rejects_duplicate_hadm_ids(self):
        duplicated = pd.concat([self.feature_matrix, self.feature_matrix.iloc[[0]]], ignore_index=True)
        with self.assertRaises(MergeError):
            self.model.build_proxy_probability_scores(
                duplicated,
                self.note_labels,
                "noncompliance_label",
                estimator_factory=lambda: _RecordingProbEstimator([0.8] * len(duplicated)),
            )

    def test_empty_usable_cohort_returns_nan_auc_row(self):
        table = self.final_model_table.copy()
        table["left_ama"] = np.nan
        results = self.model.evaluate_downstream_predictions(
            table,
            feature_configurations={"Baseline": self.model.BASELINE_FEATURE_COLUMNS},
            task_map={"Left AMA": "left_ama"},
            estimator_factory=lambda: _RecordingProbEstimator([0.2, 0.8]),
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.7),
            repetitions=3,
        )
        row = results.iloc[0]
        self.assertEqual(int(row["n_rows"]), 0)
        self.assertEqual(int(row["n_valid_auc"]), 0)
        self.assertTrue(pd.isna(row["auc_mean"]))
        self.assertTrue(pd.isna(row["auc_std"]))

    def test_estimator_fit_failures_propagate(self):
        with self.assertRaisesRegex(RuntimeError, "estimator fit failed"):
            self.model.build_proxy_probability_scores(
                self.feature_matrix,
                self.note_labels,
                "noncompliance_label",
                estimator_factory=lambda: _FailingEstimator(),
            )

        with self.assertRaisesRegex(RuntimeError, "estimator fit failed"):
            self.model.evaluate_downstream_predictions(
                self.final_model_table,
                feature_configurations={"Baseline": self.model.BASELINE_FEATURE_COLUMNS},
                task_map={"Left AMA": "left_ama"},
                estimator_factory=lambda: _FailingEstimator(),
                split_fn=_SplitRecorder(),
                auc_fn=_AUCRecorder(0.7),
                repetitions=1,
            )

    def test_statistical_backend_failures_propagate(self):
        with patch.object(self.model, "mannwhitneyu", side_effect=RuntimeError("mw failed")):
            with self.assertRaisesRegex(RuntimeError, "mw failed"):
                self.model.run_race_gap_analysis(
                    pd.DataFrame(
                        [
                            {"hadm_id": 1, "noncompliance_score_z": 0.1, "autopsy_score_z": 0.2, "negative_sentiment_score_z": 0.3},
                            {"hadm_id": 2, "noncompliance_score_z": 0.4, "autopsy_score_z": 0.5, "negative_sentiment_score_z": 0.6},
                        ]
                    ),
                    pd.DataFrame(
                        [
                            {"hadm_id": 1, "race": "WHITE"},
                            {"hadm_id": 2, "race": "BLACK"},
                        ]
                    ),
                    score_columns=["noncompliance_score_z"],
                )

        with patch.object(self.model, "pearsonr", side_effect=RuntimeError("pearson failed")):
            with self.assertRaisesRegex(RuntimeError, "pearson failed"):
                self.model.run_acuity_control_analysis(
                    pd.DataFrame(
                        [
                            {"hadm_id": 1, "noncompliance_score_z": 0.1},
                            {"hadm_id": 2, "noncompliance_score_z": 0.2},
                        ]
                    ),
                    pd.DataFrame(
                        [
                            {"hadm_id": 1, "oasis": 1.0},
                            {"hadm_id": 2, "oasis": 2.0},
                        ]
                    ),
                    score_columns=["noncompliance_score_z"],
                    acuity_columns=("oasis",),
                )

    def test_repeated_runs_do_not_mutate_input_frames(self):
        feature_matrix = self.feature_matrix.copy(deep=True)
        note_labels = self.note_labels.copy(deep=True)
        note_corpus = self.note_corpus.copy(deep=True)
        final_model_table = self.final_model_table.copy(deep=True)

        self.model.build_mistrust_score_table(
            feature_matrix,
            note_labels,
            note_corpus,
            estimator_factory=_EstimatorFactorySequence(
                [
                    lambda: _RecordingProbEstimator([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.55, 0.45, 0.65, 0.35]),
                    lambda: _RecordingProbEstimator([0.2, 0.7, 0.3, 0.8, 0.4, 0.6, 0.5, 0.55, 0.45, 0.65, 0.35, 0.75]),
                ]
            ),
            sentiment_fn=lambda text: (-0.6 if "non" in text else 0.2, 0.0),
        )
        self.model.evaluate_downstream_predictions(
            final_model_table,
            feature_configurations={"Baseline": self.model.BASELINE_FEATURE_COLUMNS},
            task_map={"Left AMA": "left_ama"},
            estimator_factory=lambda: _RecordingProbEstimator([0.2, 0.8, 0.3, 0.7, 0.4]),
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.72),
            repetitions=2,
        )

        pd.testing.assert_frame_equal(feature_matrix, self.feature_matrix)
        pd.testing.assert_frame_equal(note_labels, self.note_labels)
        pd.testing.assert_frame_equal(note_corpus, self.note_corpus)
        pd.testing.assert_frame_equal(final_model_table, self.final_model_table)

    def test_proxy_and_downstream_paths_prevent_label_and_target_leakage(self):
        proxy_estimator = _RecordingProbEstimator([0.8] * len(self.feature_matrix))
        self.model.build_proxy_probability_scores(
            self.feature_matrix,
            self.note_labels.assign(left_ama=1),
            "noncompliance_label",
            estimator_factory=lambda: proxy_estimator,
        )
        self.assertEqual(
            proxy_estimator.fit_X.columns.tolist(),
            [column for column in self.feature_matrix.columns if column != "hadm_id"],
        )

        downstream_factory = _EstimatorFactorySequence(
            [lambda: _RecordingProbEstimator([0.2, 0.8, 0.3, 0.7, 0.4])]
        )
        self.model.evaluate_downstream_predictions(
            self.final_model_table.assign(noncompliance_label=1, autopsy_label=0),
            feature_configurations={"Baseline": self.model.BASELINE_FEATURE_COLUMNS},
            task_map={"Left AMA": "left_ama"},
            estimator_factory=downstream_factory,
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.7),
            repetitions=1,
        )
        downstream_columns = downstream_factory.created[0].fit_X.columns.tolist()
        self.assertEqual(downstream_columns, self.model.BASELINE_FEATURE_COLUMNS)
        self.assertTrue(
            {"left_ama", "code_status_dnr_dni_cmo", "in_hospital_mortality", "noncompliance_label", "autopsy_label"}.isdisjoint(
                set(downstream_columns)
            )
        )

    def test_downstream_result_n_features_exactly_match_configuration_widths(self):
        results = self.model.evaluate_downstream_predictions(
            self.final_model_table,
            estimator_factory=lambda: _RecordingProbEstimator([0.2, 0.8, 0.3, 0.7, 0.4]),
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.72),
            repetitions=1,
        )

        expected_widths = {
            name: len(columns)
            for name, columns in self.model.get_downstream_feature_configurations().items()
        }
        for row in results.itertuples(index=False):
            self.assertEqual(int(row.n_features), expected_widths[row.configuration])

    def test_downstream_result_target_columns_exactly_match_task_map(self):
        results = self.model.evaluate_downstream_predictions(
            self.final_model_table,
            estimator_factory=lambda: _RecordingProbEstimator([0.2, 0.8, 0.3, 0.7, 0.4]),
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.72),
            repetitions=1,
        )

        expected_targets = self.model.get_downstream_task_map()
        for row in results.itertuples(index=False):
            self.assertEqual(row.target_column, expected_targets[row.task])

    def test_downstream_result_task_and_configuration_row_order_is_stable(self):
        results = self.model.evaluate_downstream_predictions(
            self.final_model_table,
            estimator_factory=lambda: _RecordingProbEstimator([0.2, 0.8, 0.3, 0.7, 0.4]),
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.72),
            repetitions=1,
        )

        expected_order = [
            (task_name, config_name)
            for task_name in self.model.get_downstream_task_map().keys()
            for config_name in self.model.get_downstream_feature_configurations().keys()
        ]
        actual_order = list(zip(results["task"], results["configuration"]))
        self.assertEqual(actual_order, expected_order)

    def test_downstream_auc_std_is_zero_when_auc_backend_returns_constant_value(self):
        results = self.model.evaluate_downstream_predictions(
            self.final_model_table,
            feature_configurations={"Baseline": self.model.BASELINE_FEATURE_COLUMNS},
            task_map={"Left AMA": "left_ama"},
            estimator_factory=lambda: _RecordingProbEstimator([0.2, 0.8, 0.3, 0.7, 0.4]),
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.73),
            repetitions=5,
        )
        self.assertEqual(float(results.loc[0, "auc_std"]), 0.0)

    def test_downstream_auc_mean_and_std_match_known_per_run_values(self):
        auc_values = [0.61, 0.63, 0.67, 0.69, 0.70]
        auc_recorder = _AUCSequenceRecorder(auc_values)
        results = self.model.evaluate_downstream_predictions(
            self.final_model_table,
            feature_configurations={"Baseline": self.model.BASELINE_FEATURE_COLUMNS},
            task_map={"Left AMA": "left_ama"},
            estimator_factory=lambda: _RecordingProbEstimator([0.2, 0.8, 0.3, 0.7, 0.4]),
            split_fn=_SplitRecorder(),
            auc_fn=auc_recorder,
            repetitions=len(auc_values),
        )

        self.assertAlmostEqual(float(results.loc[0, "auc_mean"]), float(np.mean(auc_values)))
        self.assertAlmostEqual(float(results.loc[0, "auc_std"]), float(np.std(auc_values, ddof=0)))

    def test_mistrust_score_table_is_deterministic_under_shuffled_input_rows(self):
        factory_one = _EstimatorFactorySequence(
            [
                lambda: _RecordingProbEstimator([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.55, 0.45, 0.65, 0.35]),
                lambda: _RecordingProbEstimator([0.2, 0.7, 0.3, 0.8, 0.4, 0.6, 0.5, 0.55, 0.45, 0.65, 0.35, 0.75]),
            ]
        )
        factory_two = _EstimatorFactorySequence(
            [
                lambda: _RecordingProbEstimator([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.55, 0.45, 0.65, 0.35]),
                lambda: _RecordingProbEstimator([0.2, 0.7, 0.3, 0.8, 0.4, 0.6, 0.5, 0.55, 0.45, 0.65, 0.35, 0.75]),
            ]
        )

        shuffled_features = self.feature_matrix.sample(frac=1.0, random_state=7).reset_index(drop=True)
        shuffled_labels = self.note_labels.sample(frac=1.0, random_state=11).reset_index(drop=True)
        shuffled_notes = self.note_corpus.sample(frac=1.0, random_state=13).reset_index(drop=True)

        scores_one = self.model.build_mistrust_score_table(
            self.feature_matrix,
            self.note_labels,
            self.note_corpus,
            estimator_factory=factory_one,
            sentiment_fn=lambda text: (-0.6 if "non" in text else 0.2, 0.0),
        )
        scores_two = self.model.build_mistrust_score_table(
            shuffled_features,
            shuffled_labels,
            shuffled_notes,
            estimator_factory=factory_two,
            sentiment_fn=lambda text: (-0.6 if "non" in text else 0.2, 0.0),
        )
        pd.testing.assert_frame_equal(scores_one.reset_index(drop=True), scores_two.reset_index(drop=True))

    def test_race_gap_analysis_ignores_other_races_even_with_extreme_values(self):
        mistrust = pd.DataFrame(
            [
                {"hadm_id": 1, "noncompliance_score_z": 0.1},
                {"hadm_id": 2, "noncompliance_score_z": 0.2},
                {"hadm_id": 3, "noncompliance_score_z": 0.3},
                {"hadm_id": 4, "noncompliance_score_z": 999.0},
            ]
        )
        demographics = pd.DataFrame(
            [
                {"hadm_id": 1, "race": "WHITE"},
                {"hadm_id": 2, "race": "BLACK"},
                {"hadm_id": 3, "race": "BLACK"},
                {"hadm_id": 4, "race": "ASIAN"},
            ]
        )
        result = self.model.run_race_gap_analysis(
            mistrust,
            demographics,
            score_columns=["noncompliance_score_z"],
        ).iloc[0]
        self.assertEqual(int(result["n_black"]), 2)
        self.assertEqual(int(result["n_white"]), 1)
        self.assertEqual(float(result["median_black"]), 0.25)
        self.assertEqual(float(result["median_white"]), 0.1)

    def test_trust_based_group_size_override_takes_precedence_over_race_based_counts(self):
        mistrust = pd.DataFrame(
            [
                {"hadm_id": 1, "noncompliance_score_z": 0.9},
                {"hadm_id": 2, "noncompliance_score_z": 0.8},
                {"hadm_id": 3, "noncompliance_score_z": 0.7},
                {"hadm_id": 4, "noncompliance_score_z": 0.1},
            ]
        )
        eol = pd.DataFrame(
            [
                {"hadm_id": 1, "race": "WHITE"},
                {"hadm_id": 2, "race": "WHITE"},
                {"hadm_id": 3, "race": "BLACK"},
                {"hadm_id": 4, "race": "BLACK"},
            ]
        )
        treatments = pd.DataFrame(
            [
                {"hadm_id": 1, "total_vent_min": 50.0, "total_vaso_min": 5.0},
                {"hadm_id": 2, "total_vent_min": 40.0, "total_vaso_min": 4.0},
                {"hadm_id": 3, "total_vent_min": 30.0, "total_vaso_min": 3.0},
                {"hadm_id": 4, "total_vent_min": 20.0, "total_vaso_min": 2.0},
            ]
        )

        result = self.model.run_trust_based_treatment_analysis(
            eol,
            mistrust,
            treatments,
            score_columns=["noncompliance_score_z"],
            treatment_columns=("total_vent_min",),
            group_sizes={"total_vent_min": 1},
        ).iloc[0]
        self.assertEqual(int(result["stratification_n"]), 1)
        self.assertEqual(int(result["n_high"]), 1)
        self.assertEqual(int(result["n_low"]), 3)

    def test_acuity_correlation_output_contains_each_pair_exactly_once(self):
        results = self.model.run_acuity_control_analysis(
            self.final_model_table[["hadm_id", *self.model.MISTRUST_SCORE_COLUMNS]],
            self.acuity_scores,
        )
        pairs = [tuple(sorted((row.feature_a, row.feature_b))) for row in results.itertuples(index=False)]
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_run_full_modeling_is_deterministic_with_same_mocks(self):
        kwargs = {
            "feature_matrix": self.feature_matrix,
            "note_labels": self.note_labels,
            "note_corpus": self.note_corpus,
            "demographics": self.demographics,
            "eol_cohort": self.eol_cohort,
            "treatment_totals": self.treatment_totals,
            "acuity_scores": self.acuity_scores,
            "final_model_table": self.final_model_table,
            "sentiment_fn": lambda text: (-0.6 if "non" in text else 0.2, 0.0),
            "split_fn": _DeterministicSplitRecorder(),
            "auc_fn": _AUCRecorder(0.77),
            "repetitions": 3,
        }
        outputs_one = self.model.run_full_eol_mistrust_modeling(
            estimator_factory=_EstimatorFactorySequence(
                [
                    lambda: _RecordingProbEstimator([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.55, 0.45, 0.65, 0.35]),
                    lambda: _RecordingProbEstimator([0.2, 0.7, 0.3, 0.8, 0.4, 0.6, 0.5, 0.55, 0.45, 0.65, 0.35, 0.75]),
                ]
            ),
            **kwargs,
        )
        outputs_two = self.model.run_full_eol_mistrust_modeling(
            estimator_factory=_EstimatorFactorySequence(
                [
                    lambda: _RecordingProbEstimator([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.55, 0.45, 0.65, 0.35]),
                    lambda: _RecordingProbEstimator([0.2, 0.7, 0.3, 0.8, 0.4, 0.6, 0.5, 0.55, 0.45, 0.65, 0.35, 0.75]),
                ]
            ),
            **kwargs,
        )

        def _assert_nested_equal(left, right):
            if isinstance(left, pd.DataFrame):
                pd.testing.assert_frame_equal(left, right)
                return
            if isinstance(left, dict):
                self.assertEqual(set(left.keys()), set(right.keys()))
                for nested_key in left:
                    _assert_nested_equal(left[nested_key], right[nested_key])
                return
            self.assertEqual(left, right)

        for key in outputs_one:
            _assert_nested_equal(outputs_one[key], outputs_two[key])

    def test_real_data_noncompliance_and_autopsy_training_matrix_matches_expected_scale(self):
        self._pending_real_data(
            "Noncompliance/autopsy proxy training on the ALL cohort should use about 48,273 rows, about 620 binary features, and binary labels aligned one-to-one with the feature matrix."
        )

    def test_real_data_proxy_models_converge_and_retain_nonzero_weights(self):
        self._pending_real_data(
            "Noncompliance and autopsy proxy logistic models should converge with max_iter=1000 and retain at least 5 nonzero coefficients each on real MIMIC-III data."
        )

    def test_real_data_proxy_probability_outputs_and_score_arrays_are_finite(self):
        self._pending_real_data(
            "Real-data proxy predict_proba outputs should have shape (n_patients, 2), values in [0, 1], and all score arrays should be finite with no nulls or infinities."
        )

    def test_real_data_proxy_weight_sanity_matches_expected_noncompliance_signals(self):
        self._pending_real_data(
            "The strongest noncompliance coefficients on real data should point to agitation/Riker positively and alert/calm negatively."
        )

    def test_real_data_proxy_weight_sanity_matches_expected_autopsy_signals(self):
        self._pending_real_data(
            "The strongest autopsy coefficients on real data should point to restraint-related features positively and pain/proxy-related features negatively."
        )

    def test_real_data_mistrust_scores_show_expected_black_white_gap_pattern(self):
        self._pending_real_data(
            "On real data, Black admissions should have significantly higher noncompliance and sentiment mistrust scores, while autopsy mistrust should remain non-significant."
        )

    def test_real_data_downstream_labels_match_expected_counts_and_positive_rates(self):
        self._pending_real_data(
            "Left AMA, Code Status, and in-hospital mortality labels should match the expected real-data cohort sizes and positive-rate bands."
        )

    def test_real_data_downstream_feature_tables_are_complete_and_finite(self):
        self._pending_real_data(
            "Baseline and Baseline+ALL downstream feature sets should have exactly 7 and 16 columns respectively, with no nulls or infinities in any used feature column."
        )

    def test_real_data_downstream_split_loop_uses_expected_100_seeded_60_40_splits(self):
        self._pending_real_data(
            "Each real-data downstream experiment should use 100 distinct random seeds (0..99) and approximately 60/40 train/test splits."
        )

    def test_real_data_downstream_models_converge_and_auc_runs_are_non_degenerate(self):
        self._pending_real_data(
            "Every real-data downstream model fit should converge without warning, and every single-run AUC should stay between 0.5 and 1.0."
        )

    def test_real_data_race_based_treatment_disparity_matches_expected_counts_and_direction(self):
        self._pending_real_data(
            "Race-based treatment disparity on real EOL admissions should match the expected White/Black sample sizes, p-values, and median-gap directions for ventilation and vasopressors."
        )

    def test_real_data_race_based_treatment_cdf_plots_have_expected_visual_elements(self):
        self._pending_real_data(
            "Real-data ventilation and vasopressor CDF plots should each contain exactly two curves and two dotted median lines."
        )

    def test_real_data_trust_based_group_sizes_match_black_reference_counts(self):
        self._pending_real_data(
            "For each metric-by-treatment pair, the real-data high-mistrust group size should equal the corresponding Black group size from the race-based treatment analysis."
        )

    def test_real_data_trust_based_disparity_matches_expected_pvalues_and_median_gaps(self):
        self._pending_real_data(
            "Real-data trust-based disparity results should match the expected p-value and median-gap bands across noncompliance, autopsy, and sentiment metrics for ventilation and vasopressors."
        )

    def test_real_data_trust_based_ventilation_gap_exceeds_race_based_gap(self):
        self._pending_real_data(
            "On real data, noncompliance and autopsy mistrust ventilation gaps should each exceed 1.5x the race-based ventilation gap."
        )

    def test_real_data_acuity_control_correlations_match_expected_ranges(self):
        self._pending_real_data(
            "Real-data acuity-control correlations should keep mistrust-vs-acuity weak while matching the expected OASIS-SAPSII and noncompliance-autopsy reference bands."
        )

    def test_real_data_downstream_auc_means_match_expected_reference_bands(self):
        self._pending_real_data(
            "Real-data downstream mean AUCs for Baseline and Baseline+ALL should fall within the expected paper-aligned ranges for Left AMA, Code Status, and Mortality."
        )

    def test_real_data_downstream_relative_ranking_and_improvement_match_reference_pattern(self):
        self._pending_real_data(
            "On real data, Baseline+ALL should be the best (or within 0.005 of best) for all tasks, mortality should improve by 0.02-0.06, and the single-mistrust configs should improve the expected target tasks."
        )

    def test_real_data_downstream_auc_variability_and_average_weights_match_reference_pattern(self):
        self._pending_real_data(
            "Real-data downstream AUC standard deviations and average Baseline+ALL mistrust-feature weights should match the expected reference ranges and directions."
        )


if __name__ == "__main__":
    unittest.main()

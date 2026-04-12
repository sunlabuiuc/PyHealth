import importlib.util
import importlib
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from pyhealth.datasets.sample_dataset import create_sample_dataset
from pyhealth.datasets.utils import get_dataloader
from pyhealth.models.base_model import BaseModel
from pyhealth.trainer import Trainer


def _load_model_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "pyhealth" / "models" / "eol_mistrust.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pyhealth.models.eol_mistrust_model_tests",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_classifier_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "pyhealth"
        / "models"
        / "eol_mistrust_classifier.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pyhealth.models.eol_mistrust_classifier_tests",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_dataset_class_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "pyhealth"
        / "datasets"
        / "eol_mistrust_dataset.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pyhealth.datasets.eol_mistrust_classifier_integration_tests",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_task_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "pyhealth"
        / "tasks"
        / "eol_mistrust.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pyhealth.tasks.eol_mistrust_classifier_integration_tests",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_dataset_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "pyhealth" / "datasets" / "eol_mistrust.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pyhealth.datasets.eol_mistrust_model_integration_tests",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeProbEstimator:
    def __init__(self, probabilities):
        self.probabilities = list(probabilities)
        self.was_fit = False
        self.fit_X = None
        self.fit_y = None
        self.coef_ = None

    def fit(self, X, y):
        self.was_fit = True
        self.fit_X = X.copy() if hasattr(X, "copy") else X
        self.fit_y = y.copy() if hasattr(y, "copy") else y
        self.coef_ = [[0.1] * X.shape[1]]
        return self

    def predict_proba(self, X):
        probs = self.probabilities[: len(X)]
        return [[1.0 - prob, prob] for prob in probs]


class _MalformedProbEstimator:
    def fit(self, X, y):
        del X, y
        self.coef_ = [[0.1]]
        return self

    def predict_proba(self, X):
        return [[1.0] for _ in range(len(X))]


class _NoCoefEstimator:
    def fit(self, X, y):
        del X, y
        return self


class _SplitRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, X, y, test_size, random_state):
        frame = X.reset_index(drop=True)
        labels = pd.Series(y).reset_index(drop=True)
        self.calls.append(
            {
                "random_state": random_state,
                "test_size": test_size,
                "n_rows": len(frame),
            }
        )
        train_idx = [0, 1, 2, 3]
        test_idx = [4, 5]
        return (
            frame.iloc[train_idx].copy(),
            frame.iloc[test_idx].copy(),
            labels.iloc[train_idx].copy(),
            labels.iloc[test_idx].copy(),
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
            }
        )
        return self.value


class _GroupSplitRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, n_splits, test_size, random_state):
        self.calls.append(
            {
                "n_splits": n_splits,
                "test_size": test_size,
                "random_state": random_state,
            }
        )
        outer = self

        class _Splitter:
            def split(self, X, y, groups):
                del y
                outer.calls[-1]["n_rows"] = len(X)
                outer.calls[-1]["groups"] = list(pd.Series(groups).reset_index(drop=True))
                yield [0, 1, 2, 3], [4, 5]

        return _Splitter()


class TestEOLMistrustModel(unittest.TestCase):
    """Model-level unit tests for the EOL mistrust workflow."""

    @classmethod
    def setUpClass(cls):
        cls.module = _load_model_module()
        cls.dataset_module = _load_dataset_module()

    def setUp(self):
        self.feature_matrix = pd.DataFrame(
            [
                {"hadm_id": 101, "Education Readiness: No": 1, "Pain Level: 7-Mod to Severe": 0},
                {"hadm_id": 102, "Education Readiness: No": 0, "Pain Level: 7-Mod to Severe": 1},
                {"hadm_id": 103, "Education Readiness: No": 1, "Pain Level: 7-Mod to Severe": 1},
                {"hadm_id": 104, "Education Readiness: No": 0, "Pain Level: 7-Mod to Severe": 0},
                {"hadm_id": 105, "Education Readiness: No": 1, "Pain Level: 7-Mod to Severe": 0},
                {"hadm_id": 106, "Education Readiness: No": 0, "Pain Level: 7-Mod to Severe": 1},
            ]
        )
        self.note_labels = pd.DataFrame(
            [
                {"hadm_id": 101, "noncompliance_label": 1, "autopsy_label": 0},
                {"hadm_id": 102, "noncompliance_label": 0, "autopsy_label": 1},
                {"hadm_id": 103, "noncompliance_label": 1, "autopsy_label": 0},
                {"hadm_id": 104, "noncompliance_label": 0, "autopsy_label": 1},
                {"hadm_id": 105, "noncompliance_label": 1, "autopsy_label": 0},
                {"hadm_id": 106, "noncompliance_label": 0, "autopsy_label": 1},
            ]
        )
        self.note_corpus = pd.DataFrame(
            [
                {"hadm_id": 101, "note_text": "Patient is noncompliant and angry."},
                {"hadm_id": 102, "note_text": "Patient is calm and cooperative."},
                {"hadm_id": 103, "note_text": "Autopsy discussed with family."},
                {"hadm_id": 104, "note_text": "Patient refused medication repeatedly."},
                {"hadm_id": 105, "note_text": "Date:[**5-1-18**] good rapport."},
                {"hadm_id": 106, "note_text": "non-adher to follow up plan."},
            ]
        )
        self.demographics = pd.DataFrame(
            [
                {"hadm_id": 101, "race": "WHITE"},
                {"hadm_id": 102, "race": "BLACK"},
                {"hadm_id": 103, "race": "BLACK"},
                {"hadm_id": 104, "race": "WHITE"},
                {"hadm_id": 105, "race": "ASIAN"},
                {"hadm_id": 106, "race": "OTHER"},
            ]
        )
        self.eol_cohort = pd.DataFrame(
            [
                {"hadm_id": 101, "race": "WHITE"},
                {"hadm_id": 102, "race": "BLACK"},
                {"hadm_id": 103, "race": "BLACK"},
                {"hadm_id": 104, "race": "WHITE"},
            ]
        )
        self.treatment_totals = pd.DataFrame(
            [
                {"hadm_id": 101, "total_vent_min": 10.0, "total_vaso_min": 5.0},
                {"hadm_id": 102, "total_vent_min": 40.0, "total_vaso_min": 20.0},
                {"hadm_id": 103, "total_vent_min": 80.0, "total_vaso_min": None},
                {"hadm_id": 104, "total_vent_min": 5.0, "total_vaso_min": 10.0},
            ]
        )
        self.acuity_scores = pd.DataFrame(
            [
                {"hadm_id": 101, "oasis": 10.0, "sapsii": 20.0},
                {"hadm_id": 102, "oasis": 15.0, "sapsii": 25.0},
                {"hadm_id": 103, "oasis": 20.0, "sapsii": 30.0},
                {"hadm_id": 104, "oasis": 25.0, "sapsii": 35.0},
                {"hadm_id": 105, "oasis": 30.0, "sapsii": 40.0},
                {"hadm_id": 106, "oasis": 35.0, "sapsii": 45.0},
            ]
        )
        self.final_model_table = pd.DataFrame(
            [
                {
                    "hadm_id": hadm_id,
                    "subject_id": [201, 202, 201, 203, 202, 204][index],
                    "age": float(50 + index),
                    "los_days": float(1 + index),
                    "gender_f": int(index % 2 == 1),
                    "gender_m": int(index % 2 == 0),
                    "insurance_private": int(index % 2 == 0),
                    "insurance_public": int(index % 2 == 1),
                    "insurance_self_pay": 0,
                    "race_white": int(hadm_id in {101, 104}),
                    "race_black": int(hadm_id in {102, 103}),
                    "race_asian": int(hadm_id == 105),
                    "race_hispanic": 0,
                    "race_native_american": 0,
                    "race_other": int(hadm_id == 106),
                    "noncompliance_score_z": [-1.2, -0.8, -0.4, 0.2, 0.8, 1.4][index],
                    "autopsy_score_z": [1.2, 0.8, 0.4, -0.2, -0.8, -1.4][index],
                    "negative_sentiment_score_z": [-0.5, -0.2, 0.0, 0.3, 0.7, 1.1][index],
                    "left_ama": int(index % 2 == 1),
                    "code_status_dnr_dni_cmo": int(index % 2 == 0),
                    "in_hospital_mortality": int(index % 2 == 1),
                }
                for index, hadm_id in enumerate([101, 102, 103, 104, 105, 106])
            ]
        )

    def _get_callable(self, name):
        self.assertTrue(
            hasattr(self.module, name),
            msg=f"Model module is missing expected callable: {name}",
        )
        attr = getattr(self.module, name)
        self.assertTrue(callable(attr), msg=f"Expected model attribute {name} to be callable")
        return attr

    def _sentiment_fn(self, text):
        return (-0.6 if ("non" in text or "refused" in text) else 0.2, 0.0)

    def test_module_exports_expected_core_api(self):
        expected = {
            "EOLMistrustModel",
            "build_mistrust_score_table",
            "evaluate_downstream_predictions",
            "run_full_eol_mistrust_modeling",
            "run_race_gap_analysis",
            "run_trust_based_treatment_analysis",
        }
        self.assertTrue(expected.issubset(set(self.module.__all__)))

    def test_package_import_path_exposes_model_module_api(self):
        try:
            imported = importlib.import_module("pyhealth.models.eol_mistrust")
        except ModuleNotFoundError as exc:
            if exc.name == "dask":
                self.skipTest("pyhealth.models package import currently requires optional dask dependency")
            raise
        self.assertTrue(hasattr(imported, "EOLMistrustModel"))
        self.assertTrue(callable(getattr(imported, "build_mistrust_score_table")))

    def test_get_downstream_task_map_returns_required_three_tasks(self):
        task_map = self._get_callable("get_downstream_task_map")()
        self.assertEqual(
            list(task_map.keys()),
            ["Left AMA", "Code Status", "In-hospital mortality"],
        )
        self.assertEqual(len(task_map), 3)

    def test_get_downstream_feature_configurations_returns_required_widths_and_copy(self):
        get_configs = self._get_callable("get_downstream_feature_configurations")
        configs = get_configs()
        self.assertEqual(
            {name: len(columns) for name, columns in configs.items()},
            {
                "Baseline": 7,
                "Baseline + Race": 13,
                "Baseline + Noncompliant": 8,
                "Baseline + Autopsy": 8,
                "Baseline + Neg-Sentiment": 8,
                "Baseline + ALL": 16,
            },
        )
        configs["Baseline"].append("should_not_leak")
        fresh = get_configs()
        self.assertNotIn("should_not_leak", fresh["Baseline"])

    def test_downstream_configuration_names_membership_and_constant_lists_match_requirements(self):
        configs = self._get_callable("get_downstream_feature_configurations")()
        self.assertEqual(
            list(configs.keys()),
            [
                "Baseline",
                "Baseline + Race",
                "Baseline + Noncompliant",
                "Baseline + Autopsy",
                "Baseline + Neg-Sentiment",
                "Baseline + ALL",
            ],
        )
        self.assertEqual(
            self.module.MISTRUST_SCORE_COLUMNS,
            ["noncompliance_score_z", "autopsy_score_z", "negative_sentiment_score_z"],
        )
        self.assertEqual(
            self.module.BASELINE_FEATURE_COLUMNS,
            [
                "age",
                "los_days",
                "gender_f",
                "gender_m",
                "insurance_private",
                "insurance_public",
                "insurance_self_pay",
            ],
        )
        self.assertEqual(
            configs["Baseline + ALL"],
            self.module.BASELINE_FEATURE_COLUMNS
            + self.module.RACE_FEATURE_COLUMNS
            + self.module.MISTRUST_SCORE_COLUMNS,
        )
        self.assertEqual(len(self.module.RACE_FEATURE_COLUMNS), 6)

    def test_fit_proxy_mistrust_model_uses_full_cohort_and_default_estimator_params(self):
        fit_proxy_mistrust_model = self._get_callable("fit_proxy_mistrust_model")
        created = []

        class _RecordingLogisticRegression:
            def __init__(self, *args, **kwargs):
                del args
                self.kwargs = kwargs
                self.fit_X = None
                self.fit_y = None
                created.append(self)

            def fit(self, X, y):
                self.fit_X = X.copy()
                self.fit_y = y.copy()
                self.coef_ = [[0.1] * X.shape[1]]
                return self

            def predict_proba(self, X):
                return [[0.4, 0.6] for _ in range(len(X))]

        with patch.object(self.module, "LogisticRegression", _RecordingLogisticRegression):
            estimator = fit_proxy_mistrust_model(
                self.feature_matrix,
                self.note_labels,
                "noncompliance_label",
            )

        self.assertEqual(len(created), 1)
        self.assertIs(estimator, created[0])
        self.assertEqual(created[0].kwargs.get("penalty"), "l1")
        self.assertEqual(created[0].kwargs.get("C"), 0.1)
        self.assertEqual(created[0].kwargs.get("solver"), "liblinear")
        self.assertEqual(created[0].kwargs.get("max_iter"), 100)
        self.assertEqual(created[0].kwargs.get("tol"), 0.01)
        self.assertEqual(len(created[0].fit_X), len(self.feature_matrix))
        self.assertEqual(len(created[0].fit_y), len(self.note_labels))

    def test_fit_proxy_mistrust_model_returns_constant_estimator_for_single_class_labels(self):
        fit_proxy_mistrust_model = self._get_callable("fit_proxy_mistrust_model")
        note_labels = self.note_labels.assign(noncompliance_label=0)

        estimator = fit_proxy_mistrust_model(
            self.feature_matrix,
            note_labels,
            "noncompliance_label",
            estimator_factory=lambda: (_ for _ in ()).throw(AssertionError("factory should not be called")),
        )

        probabilities = estimator.predict_proba(self.feature_matrix.drop(columns=["hadm_id"]))
        self.assertTrue(all(row[1] == 0.0 for row in probabilities))
        self.assertEqual(estimator.coef_.shape, (1, self.feature_matrix.shape[1] - 1))

    def test_build_proxy_probability_scores_uses_predict_proba_not_decision_function(self):
        """Proxy scores must use predict_proba (positive-class probability)
        matching the paper methodology, not decision_function (raw log-odds)."""
        import numpy as np

        build_proxy_probability_scores = self._get_callable("build_proxy_probability_scores")
        feature_matrix = self.feature_matrix.iloc[[2, 0, 1]].copy()
        note_labels = self.note_labels.iloc[[1, 2, 0]].copy()

        # Estimator where decision_function and predict_proba return DIFFERENT values
        class _SplitEstimator:
            def __init__(self):
                self.was_fit = False
            def fit(self, X, y):
                self.was_fit = True
                self.coef_ = [[0.1] * X.shape[1]]
                return self
            def decision_function(self, X):
                return np.array([-1.5, 2.3, 0.0])  # raw log-odds (unbounded)
            def predict_proba(self, X):
                return [[0.9, 0.1], [0.3, 0.7], [0.5, 0.5]]  # probabilities [0,1]

        estimator = _SplitEstimator()
        scores = build_proxy_probability_scores(
            feature_matrix=feature_matrix,
            note_labels=note_labels,
            label_column="noncompliance_label",
            estimator_factory=lambda: estimator,
        )

        self.assertEqual(scores["hadm_id"].tolist(), [101, 102, 103])
        # Must match predict_proba[:,1] output, NOT decision_function
        self.assertAlmostEqual(scores.iloc[0]["noncompliance_score"], 0.1)
        self.assertAlmostEqual(scores.iloc[1]["noncompliance_score"], 0.7)
        self.assertAlmostEqual(scores.iloc[2]["noncompliance_score"], 0.5)
        self.assertTrue(estimator.was_fit)

    def test_build_proxy_probability_scores_names_autopsy_output_column(self):
        build_proxy_probability_scores = self._get_callable("build_proxy_probability_scores")
        scores = build_proxy_probability_scores(
            feature_matrix=self.feature_matrix.iloc[:2],
            note_labels=self.note_labels.iloc[:2],
            label_column="autopsy_label",
            estimator_factory=lambda: _FakeProbEstimator([0.3, 0.6]),
        )
        self.assertEqual(scores.columns.tolist(), ["hadm_id", "autopsy_score"])

    def test_build_proxy_probability_scores_returns_constant_scores_for_single_class_labels(self):
        build_proxy_probability_scores = self._get_callable("build_proxy_probability_scores")
        note_labels = self.note_labels.iloc[:3].assign(noncompliance_label=0)

        scores = build_proxy_probability_scores(
            feature_matrix=self.feature_matrix.iloc[:3],
            note_labels=note_labels,
            label_column="noncompliance_label",
            estimator_factory=lambda: (_ for _ in ()).throw(AssertionError("factory should not be called")),
        )

        self.assertEqual(scores["hadm_id"].tolist(), [101, 102, 103])
        self.assertEqual(scores["noncompliance_score"].tolist(), [0.0, 0.0, 0.0])

    def test_build_proxy_probability_scores_missing_required_columns_raise_clear_errors(self):
        build_proxy_probability_scores = self._get_callable("build_proxy_probability_scores")
        with self.assertRaisesRegex(ValueError, "noncompliance_label"):
            build_proxy_probability_scores(
                feature_matrix=self.feature_matrix,
                note_labels=self.note_labels.drop(columns=["noncompliance_label"]),
                label_column="noncompliance_label",
                estimator_factory=lambda: _FakeProbEstimator([0.1] * len(self.feature_matrix)),
            )
        with self.assertRaisesRegex(ValueError, "hadm_id"):
            build_proxy_probability_scores(
                feature_matrix=self.feature_matrix.drop(columns=["hadm_id"]),
                note_labels=self.note_labels,
                label_column="noncompliance_label",
                estimator_factory=lambda: _FakeProbEstimator([0.1] * len(self.note_labels)),
            )

    def test_build_proxy_probability_scores_trains_on_labeled_rows_only_scores_all(self):
        build_proxy_probability_scores = self._get_callable("build_proxy_probability_scores")

        note_labels_with_nan = self.note_labels.copy()
        note_labels_with_nan["autopsy_label"] = [
            1.0, 0.0, float("nan"), float("nan"), float("nan"), float("nan"),
        ]

        fit_sizes = []

        class _TrackingEstimator:
            def __init__(self):
                self.coef_ = None

            def fit(self, X, y):
                fit_sizes.append(len(X))
                self.coef_ = [[0.1] * X.shape[1]]
                return self

            def predict_proba(self, X):
                return [[0.5, 0.5]] * len(X)

        scores = build_proxy_probability_scores(
            feature_matrix=self.feature_matrix,
            note_labels=note_labels_with_nan,
            label_column="autopsy_label",
            estimator_factory=_TrackingEstimator,
        )

        self.assertEqual(fit_sizes, [2], msg="Should train on 2 labeled rows only")
        self.assertEqual(len(scores), 6, msg="Should score all 6 rows")
        self.assertEqual(scores.columns.tolist(), ["hadm_id", "autopsy_score"])

    def test_build_proxy_probability_scores_preserves_feature_column_order_for_estimator_fit(self):
        build_proxy_probability_scores = self._get_callable("build_proxy_probability_scores")

        class _RecordingEstimator:
            def __init__(self):
                self.fit_columns = None
                self.coef_ = None

            def fit(self, X, y):
                del y
                self.fit_columns = list(X.columns)
                self.coef_ = [[0.1] * X.shape[1]]
                return self

            def predict_proba(self, X):
                return [[0.5, 0.5]] * len(X)

        estimator = _RecordingEstimator()
        build_proxy_probability_scores(
            feature_matrix=self.feature_matrix,
            note_labels=self.note_labels,
            label_column="noncompliance_label",
            estimator_factory=lambda: estimator,
        )
        self.assertEqual(
            estimator.fit_columns,
            ["Education Readiness: No", "Pain Level: 7-Mod to Severe"],
        )

    def test_build_proxy_probability_scores_keeps_only_inner_join_hadm_ids(self):
        build_proxy_probability_scores = self._get_callable("build_proxy_probability_scores")
        feature_matrix = self.feature_matrix.iloc[:4].copy()
        note_labels = self.note_labels.iloc[2:].copy()

        scores = build_proxy_probability_scores(
            feature_matrix=feature_matrix,
            note_labels=note_labels,
            label_column="noncompliance_label",
            estimator_factory=lambda: _FakeProbEstimator([0.2, 0.8]),
        )

        self.assertEqual(scores["hadm_id"].tolist(), [103, 104])

    def test_build_proxy_probability_scores_predict_proba_returns_correct_scores(self):
        """predict_proba output must yield correct positive-class scores for each input row."""
        build_proxy_probability_scores = self._get_callable("build_proxy_probability_scores")
        scores = build_proxy_probability_scores(
            feature_matrix=self.feature_matrix.iloc[:2],
            note_labels=self.note_labels.iloc[:2],
            label_column="noncompliance_label",
            estimator_factory=lambda: _FakeProbEstimator([-0.5, 1.2]),
        )
        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(scores.iloc[0]["noncompliance_score"], -0.5)
        self.assertAlmostEqual(scores.iloc[1]["noncompliance_score"], 1.2)

    def test_build_negative_sentiment_mistrust_scores_uses_whitespace_cleanup_and_negates_polarity(self):
        build_negative_sentiment_mistrust_scores = self._get_callable(
            "build_negative_sentiment_mistrust_scores"
        )
        note_corpus = pd.DataFrame(
            [
                {"hadm_id": 202, "note_text": "Date:[**5-1-18**]   calm   rapport"},
                {"hadm_id": 201, "note_text": " patient   refused   medication "},
                {"hadm_id": 203, "note_text": ""},
            ]
        )
        seen = []

        def _sentiment_fn(text):
            seen.append(text)
            if "refused medication" in text:
                return (-0.5, 0.0)
            return (0.25, 0.0)

        scores = build_negative_sentiment_mistrust_scores(note_corpus, sentiment_fn=_sentiment_fn)

        self.assertEqual(
            seen,
            ["Date:[**5-1-18**] calm rapport", "patient refused medication"],
        )
        self.assertEqual(scores["hadm_id"].tolist(), [201, 202, 203])
        by_hadm = scores.set_index("hadm_id")
        self.assertEqual(by_hadm.loc[201, "negative_sentiment_score"], 0.5)
        self.assertEqual(by_hadm.loc[202, "negative_sentiment_score"], -0.25)
        self.assertEqual(by_hadm.loc[203, "negative_sentiment_score"], 0.0)

    def test_build_negative_sentiment_mistrust_scores_missing_note_text_raises_and_empty_schema_is_stable(self):
        build_negative_sentiment_mistrust_scores = self._get_callable(
            "build_negative_sentiment_mistrust_scores"
        )
        with self.assertRaisesRegex(ValueError, "note_text"):
            build_negative_sentiment_mistrust_scores(
                pd.DataFrame([{"hadm_id": 1, "text": "oops"}]),
                sentiment_fn=self._sentiment_fn,
            )

        empty = build_negative_sentiment_mistrust_scores(
            pd.DataFrame(columns=["hadm_id", "note_text"]),
            sentiment_fn=self._sentiment_fn,
        )
        self.assertEqual(empty.columns.tolist(), ["hadm_id", "negative_sentiment_score"])
        self.assertTrue(empty.empty)

    def test_build_negative_sentiment_mistrust_scores_batches_default_backend(self):
        build_negative_sentiment_mistrust_scores = self._get_callable(
            "build_negative_sentiment_mistrust_scores"
        )
        note_corpus = pd.DataFrame(
            [
                {"hadm_id": 202, "note_text": "Date:[**5-1-18**]   calm   rapport"},
                {"hadm_id": 201, "note_text": " patient   refused   medication "},
                {"hadm_id": 203, "note_text": ""},
            ]
        )
        seen_batches = []

        def _batch_backend(texts):
            seen_batches.append(list(texts))
            outputs = []
            for text in texts:
                if "refused medication" in text:
                    outputs.append((-0.5, 0.0))
                else:
                    outputs.append((0.25, 0.0))
            return outputs

        with patch.object(
            self.module,
            "_default_sentiment_batch_backend",
            side_effect=_batch_backend,
        ):
            scores = build_negative_sentiment_mistrust_scores(note_corpus)

        self.assertEqual(
            seen_batches,
            [["Date:[**5-1-18**] calm rapport", "patient refused medication", ""]],
        )
        self.assertEqual(scores["hadm_id"].tolist(), [201, 202, 203])
        by_hadm = scores.set_index("hadm_id")
        self.assertEqual(by_hadm.loc[201, "negative_sentiment_score"], 0.5)
        self.assertEqual(by_hadm.loc[202, "negative_sentiment_score"], -0.25)
        self.assertEqual(by_hadm.loc[203, "negative_sentiment_score"], -0.25)

    def test_z_normalize_scores_normalizes_independently_and_handles_constant_column(self):
        z_normalize_scores = self._get_callable("z_normalize_scores")
        score_table = pd.DataFrame(
            [
                {"hadm_id": 1, "a": 1.0, "b": 5.0, "keep": 10.0},
                {"hadm_id": 2, "a": 2.0, "b": 5.0, "keep": 20.0},
                {"hadm_id": 3, "a": 3.0, "b": 5.0, "keep": 30.0},
            ]
        )

        normalized = z_normalize_scores(score_table, columns=["a", "b"])

        self.assertAlmostEqual(float(normalized["a"].mean()), 0.0, places=7)
        self.assertAlmostEqual(float(normalized["a"].std(ddof=0)), 1.0, places=7)
        self.assertTrue((normalized["b"] == 0.0).all())
        self.assertEqual(normalized["keep"].tolist(), [10.0, 20.0, 30.0])

    def test_z_normalize_scores_leaves_hadm_id_untouched_and_raises_for_missing_column(self):
        z_normalize_scores = self._get_callable("z_normalize_scores")
        score_table = pd.DataFrame(
            [
                {"hadm_id": 10, "a": 1.0},
                {"hadm_id": 20, "a": 2.0},
            ]
        )
        normalized = z_normalize_scores(score_table, columns=["a"])
        self.assertEqual(normalized["hadm_id"].tolist(), [10, 20])

        with self.assertRaisesRegex(ValueError, "missing_col"):
            z_normalize_scores(score_table, columns=["missing_col"])

    def test_build_mistrust_score_table_outputs_required_columns_and_shared_hadm_ids(self):
        build_mistrust_score_table = self._get_callable("build_mistrust_score_table")
        feature_matrix = self.feature_matrix.iloc[:4].copy()
        note_labels = self.note_labels.iloc[1:5].copy()
        note_corpus = self.note_corpus.iloc[2:].copy()

        scores = build_mistrust_score_table(
            feature_matrix=feature_matrix,
            note_labels=note_labels,
            note_corpus=note_corpus,
            estimator_factory=lambda: _FakeProbEstimator([0.2, 0.8, 0.4]),
            sentiment_fn=self._sentiment_fn,
        )

        self.assertEqual(
            scores.columns.tolist(),
            [
                "hadm_id",
                "noncompliance_score_z",
                "autopsy_score_z",
                "negative_sentiment_score_z",
            ],
        )
        self.assertEqual(scores["hadm_id"].tolist(), [103, 104])
        self.assertTrue(pd.api.types.is_float_dtype(scores["noncompliance_score_z"]))
        self.assertTrue(pd.api.types.is_float_dtype(scores["autopsy_score_z"]))
        self.assertTrue(pd.api.types.is_float_dtype(scores["negative_sentiment_score_z"]))

    def test_build_mistrust_score_table_missing_required_columns_raise_and_dependencies_are_called(self):
        build_mistrust_score_table = self._get_callable("build_mistrust_score_table")
        with self.assertRaisesRegex(ValueError, "noncompliance_label"):
            build_mistrust_score_table(
                self.feature_matrix,
                self.note_labels.drop(columns=["noncompliance_label"]),
                self.note_corpus,
                estimator_factory=lambda: _FakeProbEstimator([0.1] * 6),
                sentiment_fn=self._sentiment_fn,
            )
        with self.assertRaisesRegex(ValueError, "note_text"):
            build_mistrust_score_table(
                self.feature_matrix,
                self.note_labels,
                self.note_corpus.drop(columns=["note_text"]),
                estimator_factory=lambda: _FakeProbEstimator([0.1] * 6),
                sentiment_fn=self._sentiment_fn,
            )

        estimator_calls = []
        sentiment_calls = []

        def _factory():
            estimator_calls.append("estimator")
            return _FakeProbEstimator([0.1] * 6)

        def _sentiment(text):
            sentiment_calls.append(text)
            return self._sentiment_fn(text)

        build_mistrust_score_table(
            self.feature_matrix,
            self.note_labels,
            self.note_corpus,
            estimator_factory=_factory,
            sentiment_fn=_sentiment,
        )
        self.assertEqual(len(estimator_calls), 2)
        self.assertEqual(len(sentiment_calls), len(self.note_corpus))

    def test_summarize_feature_weights_returns_positive_and_negative_rankings(self):
        summarize_feature_weights = self._get_callable("summarize_feature_weights")

        class _Estimator:
            coef_ = [[0.7, -0.1, -2.0, 1.2]]

        summary = summarize_feature_weights(
            _Estimator(),
            ["Riker-SAS Scale: Agitated", "Pain Present: No", "State: Alert", "Orientation: Oriented 3x"],
            top_n=2,
        )

        self.assertEqual(set(summary.keys()), {"all", "positive", "negative"})
        self.assertEqual(summary["positive"]["feature"].tolist(), ["Orientation: Oriented 3x", "Riker-SAS Scale: Agitated"])
        self.assertEqual(summary["negative"]["feature"].tolist(), ["State: Alert", "Pain Present: No"])

    def test_summarize_feature_weights_raises_for_missing_coef_or_misaligned_length(self):
        summarize_feature_weights = self._get_callable("summarize_feature_weights")
        with self.assertRaisesRegex(ValueError, "coef_"):
            summarize_feature_weights(_NoCoefEstimator(), ["a", "b"])

        class _WrongShapeEstimator:
            coef_ = [0.1, 0.2]

        with self.assertRaisesRegex(ValueError, "shape"):
            summarize_feature_weights(_WrongShapeEstimator(), ["a", "b"])

        class _BadEstimator:
            coef_ = [[0.1]]

        with self.assertRaisesRegex(ValueError, "align"):
            summarize_feature_weights(_BadEstimator(), ["a", "b"])

    def test_feature_weight_summary_wrappers_use_correct_labels(self):
        with patch.object(self.module, "build_proxy_feature_weight_summary", return_value={"all": pd.DataFrame()}) as patched:
            self.module.build_noncompliance_feature_weight_summary(self.feature_matrix, self.note_labels)
            self.assertEqual(patched.call_args.kwargs["label_column"], "noncompliance_label")

        with patch.object(self.module, "build_proxy_feature_weight_summary", return_value={"all": pd.DataFrame()}) as patched:
            self.module.build_autopsy_feature_weight_summary(self.feature_matrix, self.note_labels)
            self.assertEqual(patched.call_args.kwargs["label_column"], "autopsy_label")

    def test_run_race_gap_analysis_filters_to_white_and_black_and_computes_direction(self):
        run_race_gap_analysis = self._get_callable("run_race_gap_analysis")
        mistrust_scores = pd.DataFrame(
            [
                {"hadm_id": 101, "noncompliance_score_z": 0.0, "autopsy_score_z": 0.1, "negative_sentiment_score_z": 0.2},
                {"hadm_id": 102, "noncompliance_score_z": 2.0, "autopsy_score_z": 2.1, "negative_sentiment_score_z": 2.2},
                {"hadm_id": 103, "noncompliance_score_z": 3.0, "autopsy_score_z": 3.1, "negative_sentiment_score_z": 3.2},
                {"hadm_id": 104, "noncompliance_score_z": 1.0, "autopsy_score_z": 1.1, "negative_sentiment_score_z": 1.2},
                {"hadm_id": 105, "noncompliance_score_z": 99.0, "autopsy_score_z": 99.1, "negative_sentiment_score_z": 99.2},
            ]
        )

        results = run_race_gap_analysis(mistrust_scores, self.demographics, score_columns=["noncompliance_score_z"])

        self.assertEqual(results.shape[0], 1)
        row = results.iloc[0]
        self.assertEqual(row["n_black"], 2)
        self.assertEqual(row["n_white"], 2)
        self.assertAlmostEqual(float(row["median_black"]), 2.5)
        self.assertAlmostEqual(float(row["median_white"]), 0.5)
        self.assertTrue(bool(row["black_median_higher"]))

    def test_run_race_gap_analysis_returns_nan_when_one_group_is_missing(self):
        run_race_gap_analysis = self._get_callable("run_race_gap_analysis")
        demographics = pd.DataFrame([{"hadm_id": 1, "race": "WHITE"}, {"hadm_id": 2, "race": "WHITE"}])
        scores = pd.DataFrame(
            [
                {"hadm_id": 1, "noncompliance_score_z": 0.1},
                {"hadm_id": 2, "noncompliance_score_z": 0.2},
            ]
        )

        results = run_race_gap_analysis(scores, demographics, score_columns=["noncompliance_score_z"])
        row = results.iloc[0]
        self.assertEqual(row["n_black"], 0)
        self.assertTrue(pd.isna(row["pvalue"]))
        self.assertTrue(pd.isna(row["median_black"]))

    def test_run_race_gap_analysis_output_contract_and_missing_columns_raise(self):
        run_race_gap_analysis = self._get_callable("run_race_gap_analysis")
        scores = pd.DataFrame(
            [
                {"hadm_id": 101, "noncompliance_score_z": 0.1},
                {"hadm_id": 102, "noncompliance_score_z": 0.2},
            ]
        )
        demographics = pd.DataFrame(
            [
                {"hadm_id": 101, "race": "WHITE"},
                {"hadm_id": 102, "race": "BLACK"},
            ]
        )
        results = run_race_gap_analysis(scores, demographics, score_columns=["noncompliance_score_z"])
        self.assertTrue(
            {
                "metric",
                "n_black",
                "n_white",
                "median_black",
                "median_white",
                "median_gap_black_minus_white",
                "statistic",
                "pvalue",
                "black_median_higher",
            }.issubset(results.columns)
        )

        with self.assertRaisesRegex(ValueError, "race"):
            run_race_gap_analysis(scores, demographics.drop(columns=["race"]), score_columns=["noncompliance_score_z"])
        with self.assertRaisesRegex(ValueError, "hadm_id"):
            run_race_gap_analysis(scores.drop(columns=["hadm_id"]), demographics, score_columns=["noncompliance_score_z"])

    def test_run_race_based_treatment_analysis_uses_non_null_rows_and_black_minus_white_gap(self):
        run_race_based_treatment_analysis = self._get_callable("run_race_based_treatment_analysis")
        results = run_race_based_treatment_analysis(self.eol_cohort, self.treatment_totals).set_index("treatment")

        vent = results.loc["total_vent_min"]
        vaso = results.loc["total_vaso_min"]
        self.assertEqual(vent["n_black"], 2)
        self.assertEqual(vent["n_white"], 2)
        self.assertAlmostEqual(float(vent["median_gap_black_minus_white"]), 52.5)
        self.assertEqual(vaso["n_black"], 1)
        self.assertEqual(vaso["n_white"], 2)

    def test_run_race_based_treatment_analysis_missing_columns_raise(self):
        run_race_based_treatment_analysis = self._get_callable("run_race_based_treatment_analysis")
        with self.assertRaisesRegex(ValueError, "total_vaso_min"):
            run_race_based_treatment_analysis(
                self.eol_cohort,
                self.treatment_totals.drop(columns=["total_vaso_min"]),
            )
        with self.assertRaisesRegex(ValueError, "race"):
            run_race_based_treatment_analysis(
                self.eol_cohort.drop(columns=["race"]),
                self.treatment_totals,
            )

    def test_run_race_based_treatment_analysis_by_acuity_partitions_each_treatment_into_three_bins(self):
        run_race_based_treatment_analysis_by_acuity = self._get_callable(
            "run_race_based_treatment_analysis_by_acuity"
        )
        results = run_race_based_treatment_analysis_by_acuity(
            self.eol_cohort,
            self.treatment_totals,
            self.acuity_scores,
        )
        self.assertEqual(results.shape[0], 6)
        self.assertEqual(set(results["treatment"]), {"total_vent_min", "total_vaso_min"})
        self.assertEqual(set(results["severity_bin"]), {"low", "medium", "high"})
        counts = results.groupby("treatment")["severity_bin"].nunique().to_dict()
        self.assertEqual(counts, {"total_vent_min": 3, "total_vaso_min": 3})

    def test_run_trust_based_treatment_analysis_uses_explicit_group_size_and_tie_breaks_by_hadm_id(self):
        run_trust_based_treatment_analysis = self._get_callable("run_trust_based_treatment_analysis")
        eol = pd.DataFrame(
            [
                {"hadm_id": 1, "race": "WHITE"},
                {"hadm_id": 2, "race": "BLACK"},
                {"hadm_id": 3, "race": "WHITE"},
            ]
        )
        scores = pd.DataFrame(
            [
                {"hadm_id": 1, "noncompliance_score_z": 0.9},
                {"hadm_id": 2, "noncompliance_score_z": 0.9},
                {"hadm_id": 3, "noncompliance_score_z": 0.1},
            ]
        )
        treatments = pd.DataFrame(
            [
                {"hadm_id": 1, "total_vent_min": 10.0, "total_vaso_min": 1.0},
                {"hadm_id": 2, "total_vent_min": 100.0, "total_vaso_min": 2.0},
                {"hadm_id": 3, "total_vent_min": 1.0, "total_vaso_min": 3.0},
            ]
        )

        results = run_trust_based_treatment_analysis(
            eol,
            scores,
            treatments,
            score_columns=["noncompliance_score_z"],
            treatment_columns=["total_vent_min"],
            group_sizes={"total_vent_min": 1},
        )
        row = results.iloc[0]
        self.assertEqual(row["stratification_n"], 1)
        self.assertEqual(row["n_high"], 1)
        self.assertEqual(row["n_low"], 2)
        self.assertAlmostEqual(float(row["median_high"]), 10.0)

    def test_run_trust_based_treatment_analysis_derives_group_size_from_race_based_counts(self):
        run_trust_based_treatment_analysis = self._get_callable("run_trust_based_treatment_analysis")
        scores = pd.DataFrame(
            [
                {"hadm_id": 101, "noncompliance_score_z": 0.9},
                {"hadm_id": 102, "noncompliance_score_z": 0.8},
                {"hadm_id": 103, "noncompliance_score_z": 0.7},
                {"hadm_id": 104, "noncompliance_score_z": 0.1},
            ]
        )

        results = run_trust_based_treatment_analysis(
            self.eol_cohort,
            scores,
            self.treatment_totals,
            score_columns=["noncompliance_score_z"],
            treatment_columns=["total_vent_min"],
        )

        self.assertEqual(int(results.iloc[0]["stratification_n"]), 2)

    def test_run_trust_based_treatment_analysis_handles_invalid_group_sizes_and_full_cartesian_output(self):
        run_trust_based_treatment_analysis = self._get_callable("run_trust_based_treatment_analysis")
        scores = pd.DataFrame(
            [
                {"hadm_id": 101, "noncompliance_score_z": 0.9, "autopsy_score_z": 0.1},
                {"hadm_id": 102, "noncompliance_score_z": 0.8, "autopsy_score_z": 0.2},
                {"hadm_id": 103, "noncompliance_score_z": 0.7, "autopsy_score_z": 0.3},
                {"hadm_id": 104, "noncompliance_score_z": 0.6, "autopsy_score_z": 0.4},
            ]
        )

        results = run_trust_based_treatment_analysis(
            self.eol_cohort,
            scores,
            self.treatment_totals,
            score_columns=["noncompliance_score_z", "autopsy_score_z"],
            treatment_columns=["total_vent_min", "total_vaso_min"],
            group_sizes={"total_vent_min": 0, "total_vaso_min": 10},
        )
        self.assertEqual(results.shape[0], 4)
        self.assertTrue(results["median_gap"].isna().all())

        valid = run_trust_based_treatment_analysis(
            self.eol_cohort,
            scores,
            self.treatment_totals,
            score_columns=["noncompliance_score_z"],
            treatment_columns=["total_vent_min"],
            group_sizes={"total_vent_min": 2},
        )
        row = valid.iloc[0]
        self.assertAlmostEqual(float(row["median_gap"]), float(row["median_high"]) - float(row["median_low"]))

    def test_run_trust_based_treatment_analysis_by_acuity_returns_metric_treatment_bin_rows(self):
        run_trust_based_treatment_analysis_by_acuity = self._get_callable(
            "run_trust_based_treatment_analysis_by_acuity"
        )
        results = run_trust_based_treatment_analysis_by_acuity(
            self.eol_cohort,
            self.final_model_table[["hadm_id", "noncompliance_score_z"]],
            self.treatment_totals,
            self.acuity_scores,
            score_columns=["noncompliance_score_z"],
        )
        self.assertEqual(results.shape[0], 6)
        self.assertEqual(set(results["metric"]), {"noncompliance_score_z"})
        self.assertEqual(set(results["treatment"]), {"total_vent_min", "total_vaso_min"})
        self.assertEqual(set(results["severity_bin"]), {"low", "medium", "high"})

    def test_run_acuity_control_analysis_returns_pairwise_correlations(self):
        run_acuity_control_analysis = self._get_callable("run_acuity_control_analysis")
        mistrust_scores = pd.DataFrame(
            [
                {"hadm_id": 101, "noncompliance_score_z": 0.1, "autopsy_score_z": 0.2, "negative_sentiment_score_z": 0.3},
                {"hadm_id": 102, "noncompliance_score_z": 0.2, "autopsy_score_z": 0.3, "negative_sentiment_score_z": 0.4},
                {"hadm_id": 103, "noncompliance_score_z": 0.3, "autopsy_score_z": 0.4, "negative_sentiment_score_z": 0.5},
                {"hadm_id": 104, "noncompliance_score_z": 0.4, "autopsy_score_z": 0.5, "negative_sentiment_score_z": 0.6},
            ]
        )
        acuity = self.acuity_scores.iloc[:4].copy()

        results = run_acuity_control_analysis(mistrust_scores, acuity)

        self.assertEqual(results.shape[0], 10)
        pairs = set(zip(results["feature_a"], results["feature_b"]))
        self.assertIn(("noncompliance_score_z", "autopsy_score_z"), pairs)
        self.assertIn(("oasis", "sapsii"), pairs)

    def test_run_acuity_control_analysis_output_contract_low_sample_and_missing_columns(self):
        run_acuity_control_analysis = self._get_callable("run_acuity_control_analysis")
        low_sample_scores = pd.DataFrame(
            [{"hadm_id": 1, "noncompliance_score_z": 0.1, "autopsy_score_z": 0.2, "negative_sentiment_score_z": 0.3}]
        )
        low_sample_acuity = pd.DataFrame([{"hadm_id": 1, "oasis": 10.0, "sapsii": 20.0}])
        results = run_acuity_control_analysis(low_sample_scores, low_sample_acuity)
        self.assertTrue({"feature_a", "feature_b", "correlation", "pvalue", "n"}.issubset(results.columns))
        self.assertTrue(results["correlation"].isna().all())

        with self.assertRaisesRegex(ValueError, "oasis"):
            run_acuity_control_analysis(
                self.final_model_table[["hadm_id", "noncompliance_score_z", "autopsy_score_z", "negative_sentiment_score_z"]],
                self.acuity_scores.drop(columns=["oasis"]),
            )

    def test_evaluate_downstream_predictions_returns_all_task_configuration_rows(self):
        evaluate_downstream_predictions = self._get_callable("evaluate_downstream_predictions")
        split_recorder = _SplitRecorder()
        auc_recorder = _AUCRecorder(0.8)

        results = evaluate_downstream_predictions(
            self.final_model_table,
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9]),
            split_fn=split_recorder,
            auc_fn=auc_recorder,
            repetitions=1,
        )

        self.assertEqual(results.shape[0], 18)
        self.assertEqual(set(results["task"]), {"Left AMA", "Code Status", "In-hospital mortality"})
        self.assertEqual(set(results["configuration"]), set(self.module.DOWNSTREAM_FEATURE_CONFIGS.keys()))

    def test_evaluate_downstream_predictions_uses_random_states_zero_through_ninety_nine_and_test_size_point_four(self):
        evaluate_downstream_predictions = self._get_callable("evaluate_downstream_predictions")
        split_recorder = _SplitRecorder()

        results = evaluate_downstream_predictions(
            self.final_model_table,
            feature_configurations={"Baseline": self.module.BASELINE_FEATURE_COLUMNS},
            task_map={"Left AMA": "left_ama"},
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9]),
            split_fn=split_recorder,
            auc_fn=_AUCRecorder(0.6),
            repetitions=100,
        )

        self.assertEqual(results.shape[0], 1)
        self.assertEqual([call["random_state"] for call in split_recorder.calls], list(range(100)))
        self.assertTrue(all(call["test_size"] == 0.4 for call in split_recorder.calls))

    def test_evaluate_downstream_predictions_uses_default_estimator_metric_and_dropna(self):
        evaluate_downstream_predictions = self._get_callable("evaluate_downstream_predictions")
        table = self.final_model_table.copy()
        table = table.drop(columns=["subject_id"])
        table.loc[0, "age"] = None
        table.loc[1, "left_ama"] = None

        created = []
        split_calls = []
        auc_calls = []

        class _RecordingLogisticRegression:
            def __init__(self, *args, **kwargs):
                del args
                self.kwargs = kwargs
                created.append(self)

            def fit(self, X, y):
                self.fit_X = X.copy()
                self.fit_y = y.copy()
                self.coef_ = [[0.1] * X.shape[1]]
                return self

            def predict_proba(self, X):
                return [[0.9, 0.1], [0.1, 0.9]]

        def _split_fn(X, y, test_size, random_state):
            split_calls.append({"n_rows": len(X), "test_size": test_size, "random_state": random_state})
            frame = X.reset_index(drop=True)
            labels = pd.Series(y).reset_index(drop=True)
            return frame.iloc[:2].copy(), frame.iloc[2:4].copy(), labels.iloc[:2].copy(), labels.iloc[2:4].copy()

        def _auc_fn(y_true, y_prob):
            auc_calls.append({"y_true": list(pd.Series(y_true)), "y_prob": list(pd.Series(y_prob))})
            return 0.77

        with patch.object(self.module, "LogisticRegression", _RecordingLogisticRegression), \
             patch.object(self.module, "train_test_split", _split_fn), \
             patch.object(self.module, "roc_auc_score", _auc_fn):
            results = evaluate_downstream_predictions(
                table,
                feature_configurations={"Baseline": self.module.BASELINE_FEATURE_COLUMNS},
                task_map={"Left AMA": "left_ama"},
                repetitions=1,
            )

        self.assertEqual(split_calls[0]["n_rows"], 4)
        self.assertEqual(created[0].kwargs.get("penalty"), "l1")
        self.assertEqual(created[0].kwargs.get("C"), 0.1)
        self.assertEqual(created[0].kwargs.get("solver"), "liblinear")
        self.assertEqual(created[0].kwargs.get("max_iter"), 100)
        self.assertEqual(created[0].kwargs.get("tol"), 0.01)
        self.assertEqual(auc_calls[0]["y_prob"], [0.1, 0.9])
        self.assertEqual(int(results.iloc[0]["n_valid_auc"]), 1)

    def test_evaluate_downstream_predictions_uses_group_shuffle_split_by_subject_id_by_default(self):
        evaluate_downstream_predictions = self._get_callable("evaluate_downstream_predictions")
        group_split_recorder = _GroupSplitRecorder()

        with patch.object(self.module, "GroupShuffleSplit", side_effect=group_split_recorder), \
             patch.object(
                 self.module,
                 "train_test_split",
                 side_effect=AssertionError("group-aware default split should not call train_test_split"),
             ):
            results = evaluate_downstream_predictions(
                self.final_model_table,
                feature_configurations={"Baseline": self.module.BASELINE_FEATURE_COLUMNS},
                task_map={"Left AMA": "left_ama"},
                estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9]),
                auc_fn=_AUCRecorder(0.6),
                repetitions=1,
            )

        self.assertEqual(results.shape[0], 1)
        self.assertEqual(group_split_recorder.calls[0]["n_splits"], 1)
        self.assertEqual(group_split_recorder.calls[0]["test_size"], 0.4)
        self.assertEqual(group_split_recorder.calls[0]["random_state"], 0)
        self.assertEqual(
            group_split_recorder.calls[0]["groups"],
            [201, 202, 201, 203, 202, 204],
        )

    def test_evaluate_downstream_predictions_returns_nan_for_single_class_target(self):
        evaluate_downstream_predictions = self._get_callable("evaluate_downstream_predictions")
        table = self.final_model_table.copy()
        table["left_ama"] = 0

        results = evaluate_downstream_predictions(
            table,
            feature_configurations={"Baseline": self.module.BASELINE_FEATURE_COLUMNS},
            task_map={"Left AMA": "left_ama"},
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9]),
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.5),
            repetitions=3,
        )

        row = results.iloc[0]
        self.assertEqual(int(row["n_valid_auc"]), 0)
        self.assertTrue(pd.isna(row["auc_mean"]))
        self.assertTrue(pd.isna(row["auc_std"]))

    def test_evaluate_downstream_predictions_uses_exact_required_feature_sets(self):
        evaluate_downstream_predictions = self._get_callable("evaluate_downstream_predictions")
        seen_columns = []

        class _RecordingEstimator:
            def __init__(self):
                self.coef_ = None

            def fit(self, X, y):
                del y
                seen_columns.append(list(X.columns))
                self.coef_ = [[0.1] * X.shape[1]]
                return self

            def predict_proba(self, X):
                return [[0.9, 0.1], [0.1, 0.9]]

        evaluate_downstream_predictions(
            self.final_model_table,
            estimator_factory=lambda: _RecordingEstimator(),
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.8),
            repetitions=1,
        )
        expected = []
        for _task in self.module.DOWNSTREAM_TASK_MAP:
            for _config, columns in self.module.DOWNSTREAM_FEATURE_CONFIGS.items():
                expected.append(list(columns))
        self.assertEqual(seen_columns, expected)

    def test_evaluate_downstream_predictions_computes_auc_mean_and_std_correctly(self):
        evaluate_downstream_predictions = self._get_callable("evaluate_downstream_predictions")
        values = [0.2, 0.6]

        def _auc_fn(y_true, y_prob):
            del y_true, y_prob
            return values.pop(0)

        results = evaluate_downstream_predictions(
            self.final_model_table,
            feature_configurations={"Baseline": self.module.BASELINE_FEATURE_COLUMNS},
            task_map={"Left AMA": "left_ama"},
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9]),
            split_fn=_SplitRecorder(),
            auc_fn=_auc_fn,
            repetitions=2,
        )
        row = results.iloc[0]
        self.assertAlmostEqual(float(row["auc_mean"]), 0.4, places=7)
        self.assertAlmostEqual(float(row["auc_std"]), 0.2, places=7)

    def test_evaluate_downstream_predictions_can_use_task_specific_estimator_factories(self):
        evaluate_downstream_predictions = self._get_callable("evaluate_downstream_predictions")
        created = []

        class _RecordingEstimator:
            def __init__(self, task_name):
                self.task_name = task_name
                self.coef_ = None

            def fit(self, X, y):
                del y
                created.append({"task": self.task_name, "n_features": X.shape[1]})
                self.coef_ = [[0.1] * X.shape[1]]
                return self

            def predict_proba(self, X):
                return [[0.9, 0.1], [0.1, 0.9]]

        def _resolver(task_name, _config_name):
            return lambda: _RecordingEstimator(task_name)

        evaluate_downstream_predictions(
            self.final_model_table,
            feature_configurations={"Baseline": self.module.BASELINE_FEATURE_COLUMNS},
            task_map={
                "Left AMA": "left_ama",
                "Code Status": "code_status_dnr_dni_cmo",
            },
            downstream_estimator_factory_resolver=_resolver,
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.8),
            repetitions=1,
        )

        self.assertEqual(
            [entry["task"] for entry in created],
            ["Left AMA", "Code Status"],
        )

    def test_build_logistic_cv_estimator_factory_uses_logistic_regression_cv_with_adaptive_folds(self):
        build_logistic_cv_estimator_factory = self._get_callable("build_logistic_cv_estimator_factory")
        captured = []

        class _RecordingLogisticRegressionCV:
            def __init__(self, *args, **kwargs):
                del args
                self.kwargs = kwargs
                captured.append(self)

            def fit(self, X, y):
                del X, y
                self.coef_ = [[0.1, 0.2]]
                self.C_ = [self.kwargs["Cs"][0]]
                return self

            def predict_proba(self, X):
                return [[0.8, 0.2] for _ in range(len(X))]

        factory = build_logistic_cv_estimator_factory(
            Cs=[0.01, 0.1, 1.0],
            class_weight="balanced",
            scoring="roc_auc",
        )
        estimator = factory()

        with patch.object(self.module, "LogisticRegressionCV", _RecordingLogisticRegressionCV):
            estimator.fit(
                pd.DataFrame({"x1": [0, 1, 0, 1], "x2": [1, 0, 1, 0]}),
                pd.Series([0, 1, 0, 1]),
            )

        self.assertEqual(captured[0].kwargs["Cs"], [0.01, 0.1, 1.0])
        self.assertEqual(captured[0].kwargs["class_weight"], "balanced")
        self.assertEqual(captured[0].kwargs["scoring"], "roc_auc")
        self.assertEqual(captured[0].kwargs["cv"], 2)

    def test_evaluate_downstream_average_weights_uses_raw_training_features_without_second_scaling(self):
        evaluate_downstream_average_weights = self._get_callable("evaluate_downstream_average_weights")

        created = []

        class _RecordingEstimator:
            def __init__(self):
                self.coef_ = None
                created.append(self)

            def fit(self, X, y):
                self.fit_X = X.copy() if hasattr(X, "copy") else X
                self.fit_y = y.copy() if hasattr(y, "copy") else y
                self.coef_ = [[0.1] * X.shape[1]]
                return self

        split_recorder = _SplitRecorder()
        results = evaluate_downstream_average_weights(
            self.final_model_table,
            feature_configurations={"Baseline": self.module.BASELINE_FEATURE_COLUMNS},
            task_map={"Code Status": "code_status_dnr_dni_cmo"},
            estimator_factory=lambda: _RecordingEstimator(),
            split_fn=split_recorder,
            repetitions=1,
        )

        self.assertEqual(int(results.iloc[0]["n_valid_weights"]), 1)
        self.assertIsInstance(created[0].fit_X, pd.DataFrame)
        expected_train = (
            self.final_model_table[self.module.BASELINE_FEATURE_COLUMNS]
            .reset_index(drop=True)
            .iloc[[0, 1, 2, 3]]
            .copy()
        )
        pd.testing.assert_frame_equal(created[0].fit_X.reset_index(drop=True), expected_train)

    def test_evaluate_downstream_average_weights_uses_group_shuffle_split_by_subject_id_by_default(self):
        evaluate_downstream_average_weights = self._get_callable("evaluate_downstream_average_weights")
        group_split_recorder = _GroupSplitRecorder()
        feature_count = len(self.module.BASELINE_FEATURE_COLUMNS)

        class _RecordingEstimator:
            def __init__(self):
                self.coef_ = None

            def fit(self, X, y):
                del X, y
                self.coef_ = [[0.1] * feature_count]
                return self

        with patch.object(self.module, "GroupShuffleSplit", side_effect=group_split_recorder), \
             patch.object(
                 self.module,
                 "train_test_split",
                 side_effect=AssertionError("group-aware default split should not call train_test_split"),
             ):
            results = evaluate_downstream_average_weights(
                self.final_model_table,
                feature_configurations={"Baseline": self.module.BASELINE_FEATURE_COLUMNS},
                task_map={"Code Status": "code_status_dnr_dni_cmo"},
                estimator_factory=lambda: _RecordingEstimator(),
                repetitions=1,
            )

        self.assertEqual(int(results.iloc[0]["n_valid_weights"]), 1)
        self.assertEqual(
            group_split_recorder.calls[0]["groups"],
            [201, 202, 201, 203, 202, 204],
        )

    def test_evaluate_downstream_average_weights_returns_nan_for_single_class_target(self):
        evaluate_downstream_average_weights = self._get_callable("evaluate_downstream_average_weights")
        table = self.final_model_table.copy()
        table["code_status_dnr_dni_cmo"] = 0

        results = evaluate_downstream_average_weights(
            table,
            feature_configurations={"Baseline": self.module.BASELINE_FEATURE_COLUMNS},
            task_map={"Code Status": "code_status_dnr_dni_cmo"},
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9]),
            split_fn=_SplitRecorder(),
            repetitions=3,
        )

        self.assertEqual(int(results.iloc[0]["n_valid_weights"]), 0)
        self.assertTrue(pd.isna(results.iloc[0]["weight_mean"]))
        self.assertTrue(pd.isna(results.iloc[0]["weight_std"]))

    def test_evaluate_downstream_average_weights_can_use_task_specific_estimator_factories(self):
        evaluate_downstream_average_weights = self._get_callable("evaluate_downstream_average_weights")
        created = []

        class _RecordingEstimator:
            def __init__(self, task_name):
                self.task_name = task_name
                self.coef_ = None

            def fit(self, X, y):
                del y
                created.append({"task": self.task_name, "columns": list(X.columns)})
                self.coef_ = [[0.1] * X.shape[1]]
                return self

        def _resolver(task_name, _config_name):
            return lambda: _RecordingEstimator(task_name)

        evaluate_downstream_average_weights(
            self.final_model_table,
            feature_configurations={"Baseline": self.module.BASELINE_FEATURE_COLUMNS},
            task_map={
                "Left AMA": "left_ama",
                "Code Status": "code_status_dnr_dni_cmo",
            },
            downstream_estimator_factory_resolver=_resolver,
            split_fn=_SplitRecorder(),
            repetitions=1,
        )

        self.assertEqual(
            [entry["task"] for entry in created],
            ["Left AMA", "Code Status"],
        )

    def test_duplicate_hadm_ids_raise_in_proxy_and_race_gap_merges(self):
        build_proxy_probability_scores = self._get_callable("build_proxy_probability_scores")
        run_race_gap_analysis = self._get_callable("run_race_gap_analysis")

        duplicate_features = pd.concat([self.feature_matrix, self.feature_matrix.iloc[[0]]], ignore_index=True)
        with self.assertRaises(Exception):
            build_proxy_probability_scores(
                duplicate_features,
                self.note_labels,
                "noncompliance_label",
                estimator_factory=lambda: _FakeProbEstimator([0.1] * 7),
            )

        duplicate_scores = pd.concat(
            [
                self.final_model_table[["hadm_id", "noncompliance_score_z"]],
                self.final_model_table[["hadm_id", "noncompliance_score_z"]].iloc[[0]],
            ],
            ignore_index=True,
        )
        with self.assertRaises(Exception):
            run_race_gap_analysis(
                duplicate_scores,
                self.demographics,
                score_columns=["noncompliance_score_z"],
            )

    def test_run_full_eol_mistrust_modeling_returns_expected_sections(self):
        run_full_eol_mistrust_modeling = self._get_callable("run_full_eol_mistrust_modeling")

        outputs = run_full_eol_mistrust_modeling(
            feature_matrix=self.feature_matrix,
            note_labels=self.note_labels,
            note_corpus=self.note_corpus,
            demographics=self.demographics,
            eol_cohort=self.eol_cohort,
            treatment_totals=self.treatment_totals,
            acuity_scores=self.acuity_scores,
            final_model_table=self.final_model_table,
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
            sentiment_fn=self._sentiment_fn,
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.8),
            repetitions=1,
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
        self.assertIn("noncompliance", outputs["feature_weight_summaries"])
        self.assertIn("autopsy", outputs["feature_weight_summaries"])

    def test_run_full_eol_mistrust_modeling_preserves_proxy_summary_order(self):
        run_full_eol_mistrust_modeling = self._get_callable("run_full_eol_mistrust_modeling")

        outputs = run_full_eol_mistrust_modeling(
            feature_matrix=self.feature_matrix,
            note_labels=self.note_labels,
            note_corpus=self.note_corpus,
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
            sentiment_fn=self._sentiment_fn,
            repetitions=1,
        )

        self.assertEqual(
            list(outputs["feature_weight_summaries"].keys()),
            ["noncompliance", "autopsy"],
        )

    def test_run_full_eol_mistrust_modeling_merges_missing_mistrust_columns_into_final_table(self):
        run_full_eol_mistrust_modeling = self._get_callable("run_full_eol_mistrust_modeling")
        final_without_scores = self.final_model_table.drop(columns=self.module.MISTRUST_SCORE_COLUMNS)

        captured = {}

        def _fake_downstream(final_model_table, **kwargs):
            del kwargs
            captured["columns"] = final_model_table.columns.tolist()
            return pd.DataFrame([{"task": "Left AMA"}])

        with patch.object(self.module, "evaluate_downstream_predictions", side_effect=_fake_downstream):
            outputs = run_full_eol_mistrust_modeling(
                feature_matrix=self.feature_matrix,
                note_labels=self.note_labels,
                note_corpus=self.note_corpus,
                final_model_table=final_without_scores,
                estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
                sentiment_fn=self._sentiment_fn,
                repetitions=1,
            )

        self.assertTrue(set(self.module.MISTRUST_SCORE_COLUMNS).issubset(set(captured["columns"])))
        self.assertIn("downstream_auc_results", outputs)

    def test_run_full_eol_mistrust_modeling_does_not_overwrite_existing_mistrust_columns(self):
        run_full_eol_mistrust_modeling = self._get_callable("run_full_eol_mistrust_modeling")
        final_with_scores = self.final_model_table.copy()
        final_with_scores["noncompliance_score_z"] = 999.0

        captured = {}

        def _fake_downstream(final_model_table, **kwargs):
            del kwargs
            captured["scores"] = final_model_table["noncompliance_score_z"].tolist()
            return pd.DataFrame([{"task": "Left AMA"}])

        with patch.object(self.module, "evaluate_downstream_predictions", side_effect=_fake_downstream):
            run_full_eol_mistrust_modeling(
                feature_matrix=self.feature_matrix,
                note_labels=self.note_labels,
                note_corpus=self.note_corpus,
                final_model_table=final_with_scores,
                estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
                sentiment_fn=self._sentiment_fn,
                repetitions=1,
            )

        self.assertEqual(captured["scores"], [999.0] * len(final_with_scores))

    def test_baseline_feature_columns_align_with_real_dataset_baseline_only_output(self):
        admissions = pd.DataFrame(
            [
                {"hadm_id": 11, "subject_id": 21, "admittime": "2100-01-01 00:00:00", "dischtime": "2100-01-03 00:00:00", "ethnicity": "WHITE", "insurance": "Medicare", "discharge_location": "HOME", "hospital_expire_flag": 0, "has_chartevents_data": 1},
                {"hadm_id": 12, "subject_id": 22, "admittime": "2100-01-02 00:00:00", "dischtime": "2100-01-04 00:00:00", "ethnicity": "BLACK/AFRICAN AMERICAN", "insurance": "Private", "discharge_location": "LEFT AGAINST MEDICAL ADVICE", "hospital_expire_flag": 0, "has_chartevents_data": 1},
            ]
        )
        patients = pd.DataFrame(
            [
                {"subject_id": 21, "gender": "M", "dob": "2070-01-01 00:00:00"},
                {"subject_id": 22, "gender": "F", "dob": "2070-01-01 00:00:00"},
            ]
        )
        icustays = pd.DataFrame(
            [
                {"hadm_id": 11, "icustay_id": 1101, "intime": "2100-01-01 00:00:00", "outtime": "2100-01-01 13:00:00"},
                {"hadm_id": 12, "icustay_id": 1201, "intime": "2100-01-02 00:00:00", "outtime": "2100-01-02 13:00:00"},
            ]
        )
        d_items = pd.DataFrame(
            [{"itemid": 128, "label": "Code Status", "dbsource": "carevue"}]
        )
        chartevents = pd.DataFrame(
            [{"hadm_id": 12, "itemid": 128, "value": "DNR/DNI", "icustay_id": 1201}]
        )
        mistrust_scores = pd.DataFrame(
            [
                {"hadm_id": 11, "noncompliance_score_z": 0.0, "autopsy_score_z": 0.0, "negative_sentiment_score_z": 0.0},
                {"hadm_id": 12, "noncompliance_score_z": 0.0, "autopsy_score_z": 0.0, "negative_sentiment_score_z": 0.0},
            ]
        )

        base = self.dataset_module.build_base_admissions(admissions, patients)
        demographics = self.dataset_module.build_demographics_table(base)
        all_cohort = self.dataset_module.build_all_cohort(base, icustays)
        baseline_only = self.dataset_module.build_final_model_table(
            demographics=demographics,
            all_cohort=all_cohort,
            admissions=base,
            chartevents=chartevents,
            d_items=d_items,
            mistrust_scores=mistrust_scores,
            include_race=False,
            include_mistrust=False,
        )

        self.assertEqual(
            [column for column in baseline_only.columns if column in self.module.BASELINE_FEATURE_COLUMNS],
            self.module.BASELINE_FEATURE_COLUMNS,
        )
        self.assertFalse(any(column in baseline_only.columns for column in self.module.RACE_FEATURE_COLUMNS))
        self.assertFalse(any(column in baseline_only.columns for column in self.module.MISTRUST_SCORE_COLUMNS))
        self.assertEqual(
            set(baseline_only.columns),
            {
                "hadm_id",
                *self.module.BASELINE_FEATURE_COLUMNS,
                "left_ama",
                "code_status_dnr_dni_cmo",
                "in_hospital_mortality",
            },
        )

    def test_dataset_model_integration_smoke_flow_runs_without_column_renaming(self):
        admissions = pd.DataFrame(
            [
                {"hadm_id": 1, "subject_id": 11, "admittime": "2100-01-01 00:00:00", "dischtime": "2100-01-03 00:00:00", "ethnicity": "WHITE", "insurance": "Medicare", "discharge_location": "HOME", "hospital_expire_flag": 0, "has_chartevents_data": 1},
                {"hadm_id": 2, "subject_id": 12, "admittime": "2100-01-02 00:00:00", "dischtime": "2100-01-04 00:00:00", "ethnicity": "BLACK/AFRICAN AMERICAN", "insurance": "Private", "discharge_location": "LEFT AGAINST MEDICAL ADVICE", "hospital_expire_flag": 0, "has_chartevents_data": 1},
                {"hadm_id": 3, "subject_id": 13, "admittime": "2100-01-03 00:00:00", "dischtime": "2100-01-05 00:00:00", "ethnicity": "ASIAN", "insurance": "Medicare", "discharge_location": "SNF", "hospital_expire_flag": 0, "has_chartevents_data": 1},
                {"hadm_id": 4, "subject_id": 14, "admittime": "2100-01-04 00:00:00", "dischtime": "2100-01-06 00:00:00", "ethnicity": "HISPANIC OR LATINO", "insurance": "Private", "discharge_location": "HOME", "hospital_expire_flag": 1, "has_chartevents_data": 1},
                {"hadm_id": 5, "subject_id": 15, "admittime": "2100-01-05 00:00:00", "dischtime": "2100-01-07 00:00:00", "ethnicity": "AMERICAN INDIAN/ALASKA NATIVE", "insurance": "Self Pay", "discharge_location": "HOME", "hospital_expire_flag": 0, "has_chartevents_data": 1},
                {"hadm_id": 6, "subject_id": 16, "admittime": "2100-01-06 00:00:00", "dischtime": "2100-01-08 00:00:00", "ethnicity": "OTHER", "insurance": "Medicare", "discharge_location": "HOME", "hospital_expire_flag": 0, "has_chartevents_data": 1},
            ]
        )
        patients = pd.DataFrame(
            [
                {"subject_id": 11, "gender": "M", "dob": "2070-01-01 00:00:00"},
                {"subject_id": 12, "gender": "F", "dob": "2070-01-01 00:00:00"},
                {"subject_id": 13, "gender": "M", "dob": "2070-01-01 00:00:00"},
                {"subject_id": 14, "gender": "F", "dob": "2070-01-01 00:00:00"},
                {"subject_id": 15, "gender": "M", "dob": "2070-01-01 00:00:00"},
                {"subject_id": 16, "gender": "F", "dob": "2070-01-01 00:00:00"},
            ]
        )
        icustays = pd.DataFrame(
            [
                {"hadm_id": 1, "icustay_id": 101, "intime": "2100-01-01 00:00:00", "outtime": "2100-01-01 13:00:00"},
                {"hadm_id": 2, "icustay_id": 102, "intime": "2100-01-02 00:00:00", "outtime": "2100-01-02 13:00:00"},
                {"hadm_id": 3, "icustay_id": 103, "intime": "2100-01-03 00:00:00", "outtime": "2100-01-03 13:00:00"},
                {"hadm_id": 4, "icustay_id": 104, "intime": "2100-01-04 00:00:00", "outtime": "2100-01-04 13:00:00"},
                {"hadm_id": 5, "icustay_id": 105, "intime": "2100-01-05 00:00:00", "outtime": "2100-01-05 13:00:00"},
                {"hadm_id": 6, "icustay_id": 106, "intime": "2100-01-06 00:00:00", "outtime": "2100-01-06 13:00:00"},
            ]
        )
        d_items = pd.DataFrame(
            [
                {"itemid": 1, "label": "Education Readiness", "dbsource": "carevue"},
                {"itemid": 2, "label": "Pain Level", "dbsource": "carevue"},
                {"itemid": 128, "label": "Code Status", "dbsource": "carevue"},
            ]
        )
        chartevents = pd.DataFrame(
            [
                {"hadm_id": 1, "itemid": 1, "value": "No", "icustay_id": 101},
                {"hadm_id": 1, "itemid": 2, "value": "7-Mod to Severe", "icustay_id": 101},
                {"hadm_id": 1, "itemid": 128, "value": "Full Code", "icustay_id": 101},
                {"hadm_id": 2, "itemid": 1, "value": "Yes", "icustay_id": 102},
                {"hadm_id": 2, "itemid": 128, "value": "DNR/DNI", "icustay_id": 102},
                {"hadm_id": 3, "itemid": 1, "value": "No", "icustay_id": 103},
                {"hadm_id": 3, "itemid": 2, "value": "None", "icustay_id": 103},
                {"hadm_id": 4, "itemid": 2, "value": "7-Mod to Severe", "icustay_id": 104},
                {"hadm_id": 5, "itemid": 1, "value": "Yes", "icustay_id": 105},
                {"hadm_id": 6, "itemid": 2, "value": "None", "icustay_id": 106},
            ]
        )
        noteevents = pd.DataFrame(
            [
                {"hadm_id": 1, "category": "Nursing", "text": "Patient is noncompliant and angry.", "iserror": None},
                {"hadm_id": 2, "category": "Nursing", "text": "Patient is calm and cooperative.", "iserror": None},
                {"hadm_id": 3, "category": "Nursing", "text": "Autopsy discussed with family.", "iserror": None},
                {"hadm_id": 4, "category": "Nursing", "text": "Patient refused medication repeatedly.", "iserror": None},
                {"hadm_id": 5, "category": "Nursing", "text": "Date:[**5-1-18**] good rapport.", "iserror": None},
                {"hadm_id": 6, "category": "Nursing", "text": "non-adher to follow up plan.", "iserror": None},
            ]
        )
        ventdurations = pd.DataFrame(
            [
                {"icustay_id": 103, "ventnum": 1, "starttime": "2100-01-03 00:00:00", "endtime": "2100-01-03 01:00:00", "duration_hours": 1.0},
                {"icustay_id": 104, "ventnum": 1, "starttime": "2100-01-04 00:00:00", "endtime": "2100-01-04 02:00:00", "duration_hours": 2.0},
            ]
        )
        vasopressordurations = pd.DataFrame(
            [
                {"icustay_id": 103, "vasonum": 1, "starttime": "2100-01-03 03:00:00", "endtime": "2100-01-03 04:00:00", "duration_hours": 1.0},
                {"icustay_id": 104, "vasonum": 1, "starttime": "2100-01-04 05:00:00", "endtime": "2100-01-04 07:00:00", "duration_hours": 2.0},
            ]
        )
        oasis = pd.DataFrame(
            [
                {"hadm_id": 1, "icustay_id": 101, "oasis": 10},
                {"hadm_id": 2, "icustay_id": 102, "oasis": 12},
                {"hadm_id": 3, "icustay_id": 103, "oasis": 20},
                {"hadm_id": 4, "icustay_id": 104, "oasis": 25},
                {"hadm_id": 5, "icustay_id": 105, "oasis": 8},
                {"hadm_id": 6, "icustay_id": 106, "oasis": 9},
            ]
        )
        sapsii = pd.DataFrame(
            [
                {"hadm_id": 1, "icustay_id": 101, "sapsii": 30},
                {"hadm_id": 2, "icustay_id": 102, "sapsii": 35},
                {"hadm_id": 3, "icustay_id": 103, "sapsii": 50},
                {"hadm_id": 4, "icustay_id": 104, "sapsii": 55},
                {"hadm_id": 5, "icustay_id": 105, "sapsii": 20},
                {"hadm_id": 6, "icustay_id": 106, "sapsii": 22},
            ]
        )

        base = self.dataset_module.build_base_admissions(admissions, patients)
        demographics = self.dataset_module.build_demographics_table(base)
        all_cohort = self.dataset_module.build_all_cohort(base, icustays)
        eol_cohort = self.dataset_module.build_eol_cohort(base, demographics)
        feature_matrix = self.dataset_module.build_chartevent_feature_matrix(
            chartevents,
            d_items,
            allowed_labels={"Education Readiness", "Pain Level"},
            all_hadm_ids=all_cohort["hadm_id"].tolist(),
        )
        note_labels = self.dataset_module.build_note_labels(
            noteevents,
            all_hadm_ids=all_cohort["hadm_id"].tolist(),
        )
        note_corpus = self.dataset_module.build_note_corpus(
            noteevents,
            all_hadm_ids=all_cohort["hadm_id"].tolist(),
        )
        treatment_totals = self.dataset_module.build_treatment_totals(
            icustays,
            ventdurations,
            vasopressordurations,
        )
        acuity_scores = self.dataset_module.build_acuity_scores(oasis, sapsii)

        scores = self.module.build_mistrust_score_table(
            feature_matrix,
            note_labels,
            note_corpus,
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
            sentiment_fn=self._sentiment_fn,
        )
        final_model_table = self.dataset_module.build_final_model_table(
            demographics,
            all_cohort,
            base,
            chartevents,
            d_items,
            scores,
            include_race=True,
            include_mistrust=True,
        )
        outputs = self.module.run_full_eol_mistrust_modeling(
            feature_matrix=feature_matrix,
            note_labels=note_labels,
            note_corpus=note_corpus,
            demographics=demographics,
            eol_cohort=eol_cohort,
            treatment_totals=treatment_totals,
            acuity_scores=acuity_scores,
            final_model_table=final_model_table,
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
            sentiment_fn=self._sentiment_fn,
            split_fn=lambda X, y, test_size, random_state: (
                X.reset_index(drop=True).iloc[: max(1, len(X) - 1)].copy(),
                X.reset_index(drop=True).iloc[max(1, len(X) - 1) :].copy(),
                pd.Series(y).reset_index(drop=True).iloc[: max(1, len(X) - 1)].copy(),
                pd.Series(y).reset_index(drop=True).iloc[max(1, len(X) - 1) :].copy(),
            ),
            auc_fn=_AUCRecorder(0.7),
            repetitions=1,
        )

        self.assertEqual(scores.shape[1], 4)
        self.assertEqual(final_model_table.shape[1], 20)
        self.assertNotIn("subject_id", final_model_table.columns)
        self.assertEqual(outputs["downstream_auc_results"].shape[0], 18)
        self.assertEqual(scores["hadm_id"].tolist(), final_model_table["hadm_id"].tolist())

    def test_dataset_and_model_proxy_probability_scores_match_exactly(self):
        model_scores = self.module.build_proxy_probability_scores(
            feature_matrix=self.feature_matrix,
            note_labels=self.note_labels,
            label_column="noncompliance_label",
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
        )
        dataset_scores = self.dataset_module.build_proxy_probability_scores(
            feature_matrix=self.feature_matrix,
            note_labels=self.note_labels,
            label_column="noncompliance_label",
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
        )
        pd.testing.assert_frame_equal(model_scores, dataset_scores)

    def test_dataset_and_model_negative_sentiment_scores_match_exactly_for_non_empty_notes(self):
        note_corpus = self.note_corpus.iloc[:5].copy()
        model_scores = self.module.build_negative_sentiment_mistrust_scores(
            note_corpus=note_corpus,
            sentiment_fn=self._sentiment_fn,
        )
        dataset_scores = self.dataset_module.build_negative_sentiment_scores(
            note_corpus=note_corpus,
            sentiment_fn=self._sentiment_fn,
        )
        pd.testing.assert_frame_equal(model_scores, dataset_scores)

    def test_dataset_and_model_z_normalize_scores_match_exactly(self):
        score_table = pd.DataFrame(
            [
                {"hadm_id": 1, "score_a": 1.0, "score_b": 5.0},
                {"hadm_id": 2, "score_a": 2.0, "score_b": 5.0},
                {"hadm_id": 3, "score_a": 3.0, "score_b": 5.0},
            ]
        )
        model_scores = self.module.z_normalize_scores(score_table, columns=["score_a", "score_b"])
        dataset_scores = self.dataset_module.z_normalize_scores(
            score_table,
            columns=["score_a", "score_b"],
        )
        pd.testing.assert_frame_equal(model_scores, dataset_scores)

    def test_dataset_and_model_mistrust_score_tables_match_on_shared_inputs(self):
        model_scores = self.module.build_mistrust_score_table(
            feature_matrix=self.feature_matrix,
            note_labels=self.note_labels,
            note_corpus=self.note_corpus,
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
            sentiment_fn=self._sentiment_fn,
        )
        dataset_scores = self.dataset_module.build_mistrust_score_table(
            feature_matrix=self.feature_matrix,
            note_labels=self.note_labels,
            note_corpus=self.note_corpus,
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
            sentiment_fn=self._sentiment_fn,
        )
        pd.testing.assert_frame_equal(model_scores, dataset_scores)

    def test_run_race_gap_analysis_calls_mannwhitneyu_with_filtered_vectors_per_metric(self):
        run_race_gap_analysis = self._get_callable("run_race_gap_analysis")
        mistrust_scores = pd.DataFrame(
            [
                {"hadm_id": 101, "noncompliance_score_z": 0.1, "autopsy_score_z": 0.5},
                {"hadm_id": 102, "noncompliance_score_z": 0.2, "autopsy_score_z": 0.6},
                {"hadm_id": 103, "noncompliance_score_z": 0.3, "autopsy_score_z": 0.7},
                {"hadm_id": 104, "noncompliance_score_z": 0.4, "autopsy_score_z": 0.8},
            ]
        )
        demographics = pd.DataFrame(
            [
                {"hadm_id": 101, "race": "WHITE"},
                {"hadm_id": 102, "race": "BLACK"},
                {"hadm_id": 103, "race": "OTHER"},
                {"hadm_id": 104, "race": "BLACK"},
            ]
        )
        calls = []

        class _Result:
            statistic = 7.0
            pvalue = 0.04

        def _fake_mwu(left, right, alternative):
            calls.append(
                {
                    "left": list(pd.Series(left, dtype=float)),
                    "right": list(pd.Series(right, dtype=float)),
                    "alternative": alternative,
                }
            )
            return _Result()

        with patch.object(self.module, "mannwhitneyu", side_effect=_fake_mwu):
            results = run_race_gap_analysis(
                mistrust_scores=mistrust_scores,
                demographics=demographics,
                score_columns=["noncompliance_score_z", "autopsy_score_z"],
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(
            calls,
            [
                {
                    "left": [0.2, 0.4],
                    "right": [0.1],
                    "alternative": "two-sided",
                },
                {
                    "left": [0.6, 0.8],
                    "right": [0.5],
                    "alternative": "two-sided",
                },
            ],
        )
        self.assertEqual(results["statistic"].tolist(), [7.0, 7.0])
        self.assertEqual(results["pvalue"].tolist(), [0.04, 0.04])

    def test_run_acuity_control_analysis_calls_pearsonr_for_each_pair_with_pairwise_filtered_vectors(self):
        run_acuity_control_analysis = self._get_callable("run_acuity_control_analysis")
        mistrust_scores = pd.DataFrame(
            [
                {
                    "hadm_id": 101,
                    "noncompliance_score_z": 0.1,
                    "autopsy_score_z": 0.4,
                    "negative_sentiment_score_z": 0.7,
                },
                {
                    "hadm_id": 102,
                    "noncompliance_score_z": 0.2,
                    "autopsy_score_z": float("nan"),
                    "negative_sentiment_score_z": 0.8,
                },
                {
                    "hadm_id": 103,
                    "noncompliance_score_z": 0.3,
                    "autopsy_score_z": 0.6,
                    "negative_sentiment_score_z": 0.9,
                },
            ]
        )
        acuity_scores = pd.DataFrame(
            [
                {"hadm_id": 101, "oasis": 10.0, "sapsii": 20.0},
                {"hadm_id": 102, "oasis": 11.0, "sapsii": 21.0},
                {"hadm_id": 103, "oasis": 12.0, "sapsii": 22.0},
            ]
        )
        calls = []

        def _fake_pearson(left, right):
            calls.append(
                {
                    "left": list(pd.Series(left, dtype=float)),
                    "right": list(pd.Series(right, dtype=float)),
                }
            )
            return (0.25, 0.5)

        with patch.object(self.module, "pearsonr", side_effect=_fake_pearson):
            results = run_acuity_control_analysis(mistrust_scores, acuity_scores)

        self.assertEqual(len(calls), 10)
        self.assertEqual(
            calls[0],
            {
                "left": [0.1, 0.3],
                "right": [0.4, 0.6],
            },
        )
        self.assertEqual(
            calls[-1],
            {
                "left": [10.0, 11.0, 12.0],
                "right": [20.0, 21.0, 22.0],
            },
        )
        self.assertEqual(len(results), 10)
        self.assertTrue((results["correlation"] == 0.25).all())
        self.assertTrue((results["pvalue"] == 0.5).all())

    def test_analysis_outputs_use_stable_column_order_contracts(self):
        race_gap = self.module.run_race_gap_analysis(
            self.final_model_table[["hadm_id", *self.module.MISTRUST_SCORE_COLUMNS]],
            self.demographics,
        )
        self.assertEqual(
            race_gap.columns.tolist(),
            [
                "metric",
                "n_black",
                "n_white",
                "median_black",
                "median_white",
                "median_gap_black_minus_white",
                "statistic",
                "pvalue",
                "black_median_higher",
            ],
        )

        race_treatment = self.module.run_race_based_treatment_analysis(
            self.eol_cohort,
            self.treatment_totals,
        )
        self.assertEqual(
            race_treatment.columns.tolist(),
            [
                "treatment",
                "n_black",
                "n_white",
                "median_black",
                "median_white",
                "median_gap_black_minus_white",
                "statistic",
                "pvalue",
            ],
        )

        trust_treatment = self.module.run_trust_based_treatment_analysis(
            self.eol_cohort,
            self.final_model_table[["hadm_id", *self.module.MISTRUST_SCORE_COLUMNS]],
            self.treatment_totals,
            group_sizes={"total_vent_min": 1, "total_vaso_min": 1},
        )
        self.assertEqual(
            trust_treatment.columns.tolist(),
            [
                "metric",
                "treatment",
                "stratification_n",
                "n_high",
                "n_low",
                "median_high",
                "median_low",
                "median_gap",
                "statistic",
                "pvalue",
            ],
        )

        acuity = self.module.run_acuity_control_analysis(
            self.final_model_table[["hadm_id", *self.module.MISTRUST_SCORE_COLUMNS]],
            self.acuity_scores,
        )
        self.assertEqual(
            acuity.columns.tolist(),
            ["feature_a", "feature_b", "correlation", "pvalue", "n"],
        )

        downstream = self.module.evaluate_downstream_predictions(
            self.final_model_table,
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9]),
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.8),
            repetitions=1,
        )
        self.assertEqual(
            downstream.columns.tolist(),
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

    def test_empty_valid_inputs_return_stable_analysis_and_downstream_schemas(self):
        empty_scores = pd.DataFrame(columns=["hadm_id", *self.module.MISTRUST_SCORE_COLUMNS])
        empty_demographics = pd.DataFrame(columns=["hadm_id", "race"])
        empty_eol = pd.DataFrame(columns=["hadm_id", "race"])
        empty_treatments = pd.DataFrame(columns=["hadm_id", "total_vent_min", "total_vaso_min"])
        empty_acuity = pd.DataFrame(columns=["hadm_id", "oasis", "sapsii"])
        empty_final = self.final_model_table.head(0).copy()

        race_gap = self.module.run_race_gap_analysis(empty_scores, empty_demographics)
        self.assertEqual(race_gap["metric"].tolist(), self.module.MISTRUST_SCORE_COLUMNS)
        self.assertTrue((race_gap["n_black"] == 0).all())
        self.assertTrue((race_gap["n_white"] == 0).all())
        self.assertTrue(race_gap["pvalue"].isna().all())

        race_treatment = self.module.run_race_based_treatment_analysis(empty_eol, empty_treatments)
        self.assertEqual(race_treatment["treatment"].tolist(), ["total_vent_min", "total_vaso_min"])
        self.assertTrue((race_treatment["n_black"] == 0).all())
        self.assertTrue(race_treatment["pvalue"].isna().all())

        trust_treatment = self.module.run_trust_based_treatment_analysis(
            empty_eol,
            empty_scores,
            empty_treatments,
        )
        self.assertEqual(len(trust_treatment), 6)
        self.assertTrue((trust_treatment["stratification_n"] == 0).all())
        self.assertTrue(trust_treatment["median_gap"].isna().all())

        acuity = self.module.run_acuity_control_analysis(empty_scores, empty_acuity)
        self.assertEqual(len(acuity), 10)
        self.assertTrue(acuity["correlation"].isna().all())
        self.assertTrue((acuity["n"] == 0).all())

        downstream = self.module.evaluate_downstream_predictions(
            empty_final,
            repetitions=2,
        )
        self.assertEqual(len(downstream), 18)
        self.assertTrue((downstream["n_rows"] == 0).all())
        self.assertTrue((downstream["n_valid_auc"] == 0).all())
        self.assertTrue(downstream["auc_mean"].isna().all())

    def test_evaluate_downstream_predictions_is_seed_stable_for_repeated_identical_runs(self):
        kwargs = {
            "final_model_table": self.final_model_table,
            "estimator_factory": lambda: _FakeProbEstimator([0.1, 0.9]),
            "split_fn": _SplitRecorder(),
            "auc_fn": _AUCRecorder(0.77),
            "repetitions": 4,
        }
        first = self.module.evaluate_downstream_predictions(**kwargs)
        second = self.module.evaluate_downstream_predictions(
            final_model_table=self.final_model_table,
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9]),
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.77),
            repetitions=4,
        )
        pd.testing.assert_frame_equal(first, second)


class TestEOLMistrustClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_classifier_module()
        cls.dataset_class_module = _load_dataset_class_module()
        cls.task_module = _load_task_module()
        cls._tmp_dirs: list[Path] = []
        cls._default_full = cls._build_route(
            dataset_prepare_mode="default",
            cache_subdir="cache_default_shared",
        )
        cls._paperlike_full = cls._build_route(
            dataset_prepare_mode="paper_like",
            cache_subdir="cache_paperlike_shared",
        )

    @classmethod
    def tearDownClass(cls):
        for path in getattr(cls, "_tmp_dirs", []):
            shutil.rmtree(path, ignore_errors=True)
        cls._tmp_dirs = []

    @classmethod
    def _build_route(
        cls,
        *,
        dataset_prepare_mode: str,
        cache_subdir: str,
    ) -> dict[str, object]:
        temp_dir = Path(
            tempfile.mkdtemp(dir=Path(__file__).resolve().parents[2])
        )
        cls._tmp_dirs.append(temp_dir)

        cls._write_minimal_root(temp_dir)
        cls._write_full_feature_tables(temp_dir)

        dataset_cls = cls.dataset_class_module.EOLMistrustDataset
        dataset = dataset_cls(
            root=str(temp_dir),
            tables=None,
            dataset_prepare_mode=dataset_prepare_mode,
            cache_dir=temp_dir / cache_subdir,
            num_workers=1,
        )
        task = cls.task_module.EOLMistrustMortalityPredictionMIMIC3(
            include_notes=True,
            dataset_prepare_mode=dataset_prepare_mode,
        )
        sample_dataset = dataset.set_task(task, num_workers=1)
        model = cls.module.EOLMistrustClassifier(
            dataset=sample_dataset,
            embedding_dim=8,
            hidden_dim=16,
            text_hash_buckets=64,
        )
        batch = next(
            iter(
                get_dataloader(
                    sample_dataset,
                    batch_size=2,
                    shuffle=False,
                )
            )
        )
        outputs = model(**batch)
        sample_by_visit = {
            int(sample_dataset[index]["visit_id"]): sample_dataset[index]
            for index in range(len(sample_dataset))
        }
        return {
            "dataset": dataset,
            "task": task,
            "sample_dataset": sample_dataset,
            "model": model,
            "outputs": outputs,
            "sample_by_visit": sample_by_visit,
        }

    @staticmethod
    def _write_minimal_root(root: Path) -> None:
        (root / "mimiciii_clinical").mkdir(parents=True, exist_ok=True)
        (root / "mimiciii_notes").mkdir(parents=True, exist_ok=True)
        (root / "mimiciii_derived").mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {
                    "subject_id": 1,
                    "gender": "F",
                    "dob": "2070-01-01 00:00:00",
                    "dod": "",
                    "dod_hosp": "",
                    "dod_ssn": "",
                    "expire_flag": 0,
                },
                {
                    "subject_id": 2,
                    "gender": "M",
                    "dob": "2065-01-01 00:00:00",
                    "dod": "",
                    "dod_hosp": "",
                    "dod_ssn": "",
                    "expire_flag": 0,
                },
            ]
        ).to_csv(root / "mimiciii_clinical" / "patients.csv", index=False)

        pd.DataFrame(
            [
                {
                    "row_id": 1,
                    "subject_id": 1,
                    "hadm_id": 101,
                    "admittime": "2100-01-01 00:00:00",
                    "dischtime": "2100-01-03 00:00:00",
                    "deathtime": "",
                    "admission_type": "EMERGENCY",
                    "admission_location": "EMERGENCY ROOM",
                    "discharge_location": "HOME",
                    "insurance": "Private",
                    "language": "ENGLISH",
                    "religion": "CATHOLIC",
                    "marital_status": "MARRIED",
                    "ethnicity": "WHITE",
                    "edregtime": "",
                    "edouttime": "",
                    "diagnosis": "SEPSIS",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "row_id": 2,
                    "subject_id": 2,
                    "hadm_id": 102,
                    "admittime": "2100-02-01 00:00:00",
                    "dischtime": "2100-02-04 12:00:00",
                    "deathtime": "",
                    "admission_type": "EMERGENCY",
                    "admission_location": "EMERGENCY ROOM",
                    "discharge_location": "HOME",
                    "insurance": "Medicare",
                    "language": "ENGLISH",
                    "religion": "CATHOLIC",
                    "marital_status": "WIDOWED",
                    "ethnicity": "BLACK/AFRICAN AMERICAN",
                    "edregtime": "",
                    "edouttime": "",
                    "diagnosis": "PNEUMONIA",
                    "hospital_expire_flag": 1,
                    "has_chartevents_data": 1,
                },
            ]
        ).to_csv(root / "mimiciii_clinical" / "admissions.csv", index=False)

        pd.DataFrame(
            [
                {
                    "row_id": 1,
                    "subject_id": 1,
                    "hadm_id": 101,
                    "icustay_id": 1001,
                    "dbsource": "metavision",
                    "first_careunit": "MICU",
                    "last_careunit": "MICU",
                    "first_wardid": 1,
                    "last_wardid": 1,
                    "intime": "2100-01-01 00:00:00",
                    "outtime": "2100-01-02 00:00:00",
                    "los": 1.0,
                },
                {
                    "row_id": 2,
                    "subject_id": 2,
                    "hadm_id": 102,
                    "icustay_id": 1002,
                    "dbsource": "metavision",
                    "first_careunit": "MICU",
                    "last_careunit": "MICU",
                    "first_wardid": 1,
                    "last_wardid": 1,
                    "intime": "2100-02-01 00:00:00",
                    "outtime": "2100-02-02 00:00:00",
                    "los": 1.0,
                },
            ]
        ).to_csv(root / "mimiciii_clinical" / "icustays.csv", index=False)

        pd.DataFrame(
            [
                {
                    "row_id": 1,
                    "subject_id": 1,
                    "hadm_id": 101,
                    "chartdate": "2100-01-01",
                    "charttime": "2100-01-01 12:00:00",
                    "storetime": "2100-01-01 13:00:00",
                    "category": "Nursing",
                    "description": "Report",
                    "cgid": 1,
                    "iserror": 0,
                    "text": "Family meeting note and goals of care discussion.",
                },
                {
                    "row_id": 2,
                    "subject_id": 2,
                    "hadm_id": 102,
                    "chartdate": "2100-02-01",
                    "charttime": "2100-02-01 12:00:00",
                    "storetime": "2100-02-01 13:00:00",
                    "category": "Nursing",
                    "description": "Report",
                    "cgid": 2,
                    "iserror": 0,
                    "text": "Patient declining treatment and family distressed.",
                },
            ]
        ).to_csv(root / "mimiciii_notes" / "noteevents.csv", index=False)

    @staticmethod
    def _write_full_feature_tables(root: Path) -> None:
        pd.DataFrame(
            [
                {
                    "itemid": 128,
                    "label": "Code Status",
                    "abbreviation": "",
                    "dbsource": "carevue",
                    "linksto": "chartevents",
                    "category": "",
                    "unitname": "",
                    "param_type": "",
                    "conceptid": "",
                }
            ]
        ).to_csv(root / "mimiciii_clinical" / "d_items.csv", index=False)

        pd.DataFrame(
            [
                {
                    "subject_id": 1,
                    "hadm_id": 101,
                    "icustay_id": 1001,
                    "itemid": 128,
                    "charttime": "2100-01-01 08:00:00",
                    "storetime": "2100-01-01 08:30:00",
                    "cgid": 1,
                    "value": "DNR/DNI",
                    "valuenum": "",
                    "valueuom": "",
                    "warning": "",
                    "error": "",
                    "resultstatus": "",
                    "stopped": "",
                },
                {
                    "subject_id": 2,
                    "hadm_id": 102,
                    "icustay_id": 1002,
                    "itemid": 128,
                    "charttime": "2100-02-04 11:00:00",
                    "storetime": "2100-02-04 11:30:00",
                    "cgid": 2,
                    "value": "Full Code",
                    "valuenum": "",
                    "valueuom": "",
                    "warning": "",
                    "error": "",
                    "resultstatus": "",
                    "stopped": "",
                },
                {
                    "subject_id": 2,
                    "hadm_id": 102,
                    "icustay_id": 1002,
                    "itemid": 128,
                    "charttime": "2100-02-01 08:00:00",
                    "storetime": "2100-02-01 08:30:00",
                    "cgid": 2,
                    "value": "DNR/DNI",
                    "valuenum": "",
                    "valueuom": "",
                    "warning": "",
                    "error": "",
                    "resultstatus": "",
                    "stopped": "",
                },
            ]
        ).to_csv(root / "mimiciii_clinical" / "chartevents.csv", index=False)

        pd.DataFrame(
            [
                {"subject_id": 1, "hadm_id": 101, "seq_num": 1, "icd9_code": "0389"},
                {"subject_id": 2, "hadm_id": 102, "seq_num": 1, "icd9_code": "486"},
            ]
        ).to_csv(root / "mimiciii_clinical" / "diagnoses_icd.csv", index=False)

        pd.DataFrame(
            [
                {"subject_id": 1, "hadm_id": 101, "seq_num": 1, "icd9_code": "3893"},
                {"subject_id": 2, "hadm_id": 102, "seq_num": 1, "icd9_code": "9671"},
            ]
        ).to_csv(root / "mimiciii_clinical" / "procedures_icd.csv", index=False)

        pd.DataFrame(
            [
                {
                    "subject_id": 1,
                    "hadm_id": 101,
                    "startdate": "2100-01-01 00:00:00",
                    "enddate": "2100-01-02 00:00:00",
                    "drug": "Aspirin",
                    "drug_type": "MAIN",
                    "drug_name_poe": "Aspirin",
                    "drug_name_generic": "Aspirin",
                    "formulary_drug_cd": "ASP",
                    "gsn": "",
                    "ndc": "",
                    "prod_strength": "81 mg",
                    "dose_val_rx": "81",
                    "dose_unit_rx": "mg",
                    "form_val_disp": "1",
                    "form_unit_disp": "tab",
                    "route": "PO",
                },
                {
                    "subject_id": 2,
                    "hadm_id": 102,
                    "startdate": "2100-02-01 00:00:00",
                    "enddate": "2100-02-02 00:00:00",
                    "drug": "Heparin",
                    "drug_type": "MAIN",
                    "drug_name_poe": "Heparin",
                    "drug_name_generic": "Heparin",
                    "formulary_drug_cd": "HEP",
                    "gsn": "",
                    "ndc": "",
                    "prod_strength": "5000 unit",
                    "dose_val_rx": "5000",
                    "dose_unit_rx": "unit",
                    "form_val_disp": "1",
                    "form_unit_disp": "dose",
                    "route": "IV",
                },
            ]
        ).to_csv(root / "mimiciii_clinical" / "prescriptions.csv", index=False)

    def test_classifier_inherits_base_model_and_supports_task_like_inputs(self):
        samples = [
            {
                "patient_id": "p1",
                "visit_id": "v1",
                "conditions": ["4019", "25000"],
                "procedures": ["3893"],
                "drugs": ["Aspirin"],
                "age": 70.0,
                "los_days": 5.0,
                "gender": "F",
                "insurance": "Private",
                "race": "WHITE",
                "clinical_notes": "family meeting note",
                "label": 1,
            },
            {
                "patient_id": "p2",
                "visit_id": "v2",
                "conditions": ["486"],
                "procedures": ["9671"],
                "drugs": ["Heparin"],
                "age": 80.0,
                "los_days": 8.0,
                "gender": "M",
                "insurance": "Public",
                "race": "BLACK",
                "clinical_notes": "patient declining treatment",
                "label": 0,
            },
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={
                "conditions": "sequence",
                "procedures": "sequence",
                "drugs": "sequence",
                "age": "tensor",
                "los_days": "tensor",
                "gender": "text",
                "insurance": "text",
                "race": "text",
                "clinical_notes": "text",
            },
            output_schema={"label": "binary"},
            dataset_name="eol_mistrust_classifier_test",
        )

        model = self.module.EOLMistrustClassifier(
            dataset=dataset,
            embedding_dim=8,
            hidden_dim=16,
            text_hash_buckets=64,
        )
        self.assertTrue(issubclass(self.module.EOLMistrustClassifier, BaseModel))

        batch = next(iter(get_dataloader(dataset, batch_size=2, shuffle=False)))
        outputs = model(**batch)

        self.assertEqual(set(outputs.keys()), {"loss", "y_prob", "y_true", "logit"})
        self.assertEqual(tuple(outputs["logit"].shape), (2, 1))
        self.assertEqual(tuple(outputs["y_prob"].shape), (2, 1))
        self.assertEqual(tuple(outputs["y_true"].shape), (2, 1))

        outputs["loss"].backward()
        self.assertIsNotNone(model.output_layer.weight.grad)
        self.assertTrue(torch.isfinite(outputs["loss"]).item())

    def test_classifier_runs_on_samples_from_eol_mistrust_dataset_task_pipeline(self):
        dataset_cls = self.dataset_class_module.EOLMistrustDataset
        task = self.task_module.EOLMistrustMortalityPredictionMIMIC3(
            include_notes=True
        )

        temp_dir = tempfile.mkdtemp(dir=Path(__file__).resolve().parents[2])
        try:
            root = Path(temp_dir)
            self._write_minimal_root(root)
            dataset = dataset_cls(
                root=str(root),
                tables=["noteevents"],
                cache_dir=root / "cache",
                num_workers=1,
            )
            sample_dataset = dataset.set_task(task, num_workers=1)
            model = self.module.EOLMistrustClassifier(
                dataset=sample_dataset,
                embedding_dim=8,
                hidden_dim=16,
                text_hash_buckets=64,
            )

            batch = next(iter(get_dataloader(sample_dataset, batch_size=2, shuffle=False)))
            outputs = model(**batch)
            del batch
            del model
            del sample_dataset
            del dataset
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(tuple(outputs["logit"].shape), (2, 1))
        self.assertEqual(tuple(outputs["y_prob"].shape), (2, 1))
        self.assertEqual(tuple(outputs["y_true"].shape), (2, 1))
        self.assertTrue(torch.isfinite(outputs["loss"]).item())

    def test_classifier_runs_end_to_end_for_normal_full_feature_path(self):
        results = self._default_full

        dataset = results["dataset"]
        task = results["task"]
        sample_by_visit = results["sample_by_visit"]
        outputs = results["outputs"]

        self.assertEqual(dataset.dataset_prepare_mode, "default")
        self.assertFalse(dataset.paper_like_dataset_prepare)
        self.assertEqual(task.dataset_prepare_mode, "default")
        self.assertIn("diagnoses_icd", dataset.tables)
        self.assertIn("procedures_icd", dataset.tables)
        self.assertIn("prescriptions", dataset.tables)
        self.assertIn("chartevents", dataset.tables)
        self.assertIn("noteevents", dataset.tables)
        self.assertGreater(
            int(torch.count_nonzero(sample_by_visit[101]["conditions"]).item()),
            0,
        )
        self.assertGreater(
            int(torch.count_nonzero(sample_by_visit[101]["procedures"]).item()),
            0,
        )
        self.assertGreater(
            int(torch.count_nonzero(sample_by_visit[101]["drugs"]).item()),
            0,
        )
        self.assertAlmostEqual(
            float(sample_by_visit[102]["los_days"].view(-1)[0].item()),
            3.5,
        )
        self.assertEqual(tuple(outputs["logit"].shape), (2, 1))
        self.assertTrue(torch.isfinite(outputs["loss"]).item())

    def test_classifier_runs_end_to_end_for_paper_like_full_feature_path(self):
        results = self._paperlike_full

        dataset = results["dataset"]
        task = results["task"]
        sample_by_visit = results["sample_by_visit"]
        outputs = results["outputs"]

        self.assertEqual(dataset.dataset_prepare_mode, "paper_like")
        self.assertTrue(dataset.paper_like_dataset_prepare)
        self.assertEqual(task.dataset_prepare_mode, "paper_like")
        self.assertEqual(task.code_status_mode, "paper_like")
        self.assertGreater(
            int(torch.count_nonzero(sample_by_visit[101]["conditions"]).item()),
            0,
        )
        self.assertGreater(
            int(torch.count_nonzero(sample_by_visit[101]["procedures"]).item()),
            0,
        )
        self.assertGreater(
            int(torch.count_nonzero(sample_by_visit[101]["drugs"]).item()),
            0,
        )
        self.assertAlmostEqual(
            float(sample_by_visit[102]["los_days"].view(-1)[0].item()),
            12.0,
        )
        self.assertEqual(tuple(outputs["logit"].shape), (2, 1))
        self.assertTrue(torch.isfinite(outputs["loss"]).item())

    def test_classifier_can_train_and_evaluate_on_normal_full_feature_path(self):
        sample_dataset = self._default_full["sample_dataset"]
        model = self.module.EOLMistrustClassifier(
            dataset=sample_dataset,
            embedding_dim=8,
            hidden_dim=16,
            text_hash_buckets=64,
        )

        train_loader = get_dataloader(sample_dataset, batch_size=2, shuffle=True)
        eval_loader = get_dataloader(sample_dataset, batch_size=2, shuffle=False)
        trainer = Trainer(
            model=model,
            metrics=["accuracy"],
            device="cpu",
            enable_logging=False,
        )
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=eval_loader,
            test_dataloader=eval_loader,
            epochs=1,
            monitor="accuracy",
            load_best_model_at_last=False,
        )
        scores = trainer.evaluate(eval_loader)

        self.assertIn("accuracy", scores)
        self.assertIn("loss", scores)
        self.assertGreaterEqual(float(scores["accuracy"]), 0.0)
        self.assertLessEqual(float(scores["accuracy"]), 1.0)
        self.assertTrue(torch.isfinite(torch.tensor(float(scores["loss"]))).item())


if __name__ == "__main__":
    unittest.main()

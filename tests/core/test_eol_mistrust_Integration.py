import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def _load_dataset_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "pyhealth" / "datasets" / "eol_mistrust.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pyhealth.datasets.eol_mistrust_integration_tests",
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
        "pyhealth.models.eol_mistrust_integration_tests",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_example_module():
    module_path = Path(__file__).resolve().parents[2] / "examples" / "eol_mistrust.py"
    spec = importlib.util.spec_from_file_location(
        "examples.eol_mistrust_integration_tests",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeProbEstimator:
    def __init__(self, probabilities):
        self.probabilities = list(probabilities)
        self.coef_ = None
        self.fit_X = None
        self.fit_y = None

    def fit(self, X, y):
        self.fit_X = X.copy()
        self.fit_y = y.copy()
        self.coef_ = [[0.1] * X.shape[1]]
        return self

    def predict_proba(self, X):
        probs = self.probabilities[: len(X)]
        return [[1.0 - prob, prob] for prob in probs]


class _SplitRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, X, y, test_size, random_state):
        features = X.reset_index(drop=True)
        labels = pd.Series(y).reset_index(drop=True)
        self.calls.append(
            {
                "test_size": test_size,
                "random_state": random_state,
                "n_rows": len(features),
            }
        )
        train_idx = [0, 1, 2, 3]
        test_idx = [4, 5]
        return (
            features.iloc[train_idx].copy(),
            features.iloc[test_idx].copy(),
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


class TestEOLMistrustIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = _load_dataset_module()
        cls.model = _load_model_module()

    def setUp(self):
        self.admissions = pd.DataFrame(
            [
                {
                    "hadm_id": 101,
                    "subject_id": 201,
                    "admittime": "2100-01-01 00:00:00",
                    "dischtime": "2100-01-03 00:00:00",
                    "ethnicity": "WHITE",
                    "insurance": "Medicare",
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 102,
                    "subject_id": 202,
                    "admittime": "2100-01-02 00:00:00",
                    "dischtime": "2100-01-04 00:00:00",
                    "ethnicity": "BLACK/AFRICAN AMERICAN",
                    "insurance": "Private",
                    "discharge_location": "LEFT AGAINST MEDICAL ADVICE",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 103,
                    "subject_id": 203,
                    "admittime": "2100-01-03 00:00:00",
                    "dischtime": "2100-01-05 00:00:00",
                    "ethnicity": "ASIAN",
                    "insurance": "Medicaid",
                    "discharge_location": "SNF",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 104,
                    "subject_id": 204,
                    "admittime": "2100-01-04 00:00:00",
                    "dischtime": "2100-01-06 00:00:00",
                    "ethnicity": "HISPANIC OR LATINO",
                    "insurance": "Private",
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 1,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 105,
                    "subject_id": 205,
                    "admittime": "2100-01-05 00:00:00",
                    "dischtime": "2100-01-07 00:00:00",
                    "ethnicity": "AMERICAN INDIAN/ALASKA NATIVE",
                    "insurance": "Self Pay",
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 106,
                    "subject_id": 206,
                    "admittime": "2100-01-06 00:00:00",
                    "dischtime": "2100-01-08 00:00:00",
                    "ethnicity": "UNKNOWN/NOT SPECIFIED",
                    "insurance": "Medicare",
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
            ]
        )
        self.patients = pd.DataFrame(
            [
                {"subject_id": 201, "gender": "M", "dob": "1800-01-01 00:00:00"},
                {"subject_id": 202, "gender": "F", "dob": "2070-01-01 00:00:00"},
                {"subject_id": 203, "gender": "M", "dob": "2070-01-01 00:00:00"},
                {"subject_id": 204, "gender": "F", "dob": "2070-01-01 00:00:00"},
                {"subject_id": 205, "gender": "M", "dob": "2070-01-01 00:00:00"},
                {"subject_id": 206, "gender": "F", "dob": "2070-01-01 00:00:00"},
            ]
        )
        self.icustays = pd.DataFrame(
            [
                {"hadm_id": 101, "icustay_id": 1001, "intime": "2100-01-01 00:00:00", "outtime": "2100-01-01 13:00:00"},
                {"hadm_id": 101, "icustay_id": 1002, "intime": "2100-01-01 14:00:00", "outtime": "2100-01-01 18:00:00"},
                {"hadm_id": 102, "icustay_id": 1003, "intime": "2100-01-02 00:00:00", "outtime": "2100-01-02 13:00:00"},
                {"hadm_id": 103, "icustay_id": 1004, "intime": "2100-01-03 00:00:00", "outtime": "2100-01-03 13:00:00"},
                {"hadm_id": 104, "icustay_id": 1005, "intime": "2100-01-04 00:00:00", "outtime": "2100-01-04 13:00:00"},
                {"hadm_id": 105, "icustay_id": 1006, "intime": "2100-01-05 00:00:00", "outtime": "2100-01-05 13:00:00"},
                {"hadm_id": 106, "icustay_id": 1007, "intime": "2100-01-06 00:00:00", "outtime": "2100-01-06 13:00:00"},
            ]
        )
        self.d_items = pd.DataFrame(
            [
                {"itemid": 1, "label": "Education Readiness", "dbsource": "carevue"},
                {"itemid": 2, "label": "Pain Level", "dbsource": "carevue"},
                {"itemid": 3, "label": "Richmond-RAS Scale Assessment", "dbsource": "metavision"},
                {"itemid": 128, "label": "Code Status", "dbsource": "carevue"},
            ]
        )
        self.chartevents = pd.DataFrame(
            [
                {"hadm_id": 101, "itemid": 1, "value": "No", "icustay_id": 1001},
                {"hadm_id": 101, "itemid": 1, "value": "No", "icustay_id": 1001},
                {"hadm_id": 101, "itemid": 2, "value": "7-Mod to Severe", "icustay_id": 1001},
                {"hadm_id": 101, "itemid": 128, "value": "Full Code", "icustay_id": 1001},
                {"hadm_id": 102, "itemid": 3, "value": "0 Alert and Calm", "icustay_id": 1003},
                {"hadm_id": 102, "itemid": 128, "value": "DNR/DNI", "icustay_id": 1003},
                {"hadm_id": 103, "itemid": 2, "value": "None", "icustay_id": 1004},
                {"hadm_id": 104, "itemid": 128, "value": "Comfort Measures", "icustay_id": 1005},
                {"hadm_id": 105, "itemid": 1, "value": "Yes", "icustay_id": 1006},
                {"hadm_id": 106, "itemid": 2, "value": "None", "icustay_id": 1007},
            ]
        )
        self.noteevents = pd.DataFrame(
            [
                {"hadm_id": 101, "category": "Nursing", "text": "Patient is noncompliant and refused medication.", "iserror": None},
                {"hadm_id": 102, "category": "Nursing", "text": "Family provided autopsy consent and autopsy was performed.", "iserror": None},
                {"hadm_id": 103, "category": "Nursing", "text": "Patient is non-adher to the follow up plan.", "iserror": None},
                {"hadm_id": 104, "category": "Nursing", "text": "Date:[**5-1-18**]  patient   has   good rapport.", "iserror": None},
                {"hadm_id": 105, "category": "Nursing", "text": "this note should be dropped", "iserror": 1},
                {"hadm_id": 106, "category": "Nursing", "text": "", "iserror": None},
            ]
        )
        self.ventdurations = pd.DataFrame(
            [
                {"icustay_id": 1004, "ventnum": 1, "starttime": "2100-01-03 00:00:00", "endtime": "2100-01-03 01:00:00", "duration_hours": 1.0},
                {"icustay_id": 1004, "ventnum": 2, "starttime": "2100-01-03 11:00:00", "endtime": "2100-01-03 13:00:00", "duration_hours": 2.0},
                {"icustay_id": 1005, "ventnum": 1, "starttime": "2100-01-04 00:00:00", "endtime": "2100-01-04 02:00:00", "duration_hours": 2.0},
            ]
        )
        self.vasopressordurations = pd.DataFrame(
            [
                {"icustay_id": 1004, "vasonum": 1, "starttime": "2100-01-03 03:00:00", "endtime": "2100-01-03 04:00:00", "duration_hours": 1.0},
                {"icustay_id": 1005, "vasonum": 1, "starttime": "2100-01-04 05:00:00", "endtime": "2100-01-04 07:00:00", "duration_hours": 2.0},
            ]
        )
        self.oasis = pd.DataFrame(
            [
                {"hadm_id": 101, "icustay_id": 1001, "oasis": 10},
                {"hadm_id": 102, "icustay_id": 1003, "oasis": 12},
                {"hadm_id": 103, "icustay_id": 1004, "oasis": 20},
                {"hadm_id": 104, "icustay_id": 1005, "oasis": 25},
                {"hadm_id": 105, "icustay_id": 1006, "oasis": 8},
                {"hadm_id": 106, "icustay_id": 1007, "oasis": 9},
            ]
        )
        self.sapsii = pd.DataFrame(
            [
                {"hadm_id": 101, "icustay_id": 1001, "sapsii": 30},
                {"hadm_id": 102, "icustay_id": 1003, "sapsii": 35},
                {"hadm_id": 103, "icustay_id": 1004, "sapsii": 50},
                {"hadm_id": 104, "icustay_id": 1005, "sapsii": 55},
                {"hadm_id": 105, "icustay_id": 1006, "sapsii": 20},
                {"hadm_id": 106, "icustay_id": 1007, "sapsii": 22},
            ]
        )

    def _sentiment_fn(self, text):
        if "non" in text or "refused" in text:
            return (-0.6, 0.0)
        return (0.2, 0.0)

    def _build_core_artifacts(self):
        base = self.dataset.build_base_admissions(self.admissions, self.patients)
        demographics = self.dataset.build_demographics_table(base)
        all_cohort = self.dataset.build_all_cohort(base, self.icustays)
        eol_cohort = self.dataset.build_eol_cohort(base, demographics)
        feature_matrix = self.dataset.build_chartevent_feature_matrix(
            self.chartevents,
            self.d_items,
            all_hadm_ids=all_cohort["hadm_id"].tolist(),
        )
        note_labels = self.dataset.build_note_labels(self.noteevents, all_hadm_ids=all_cohort["hadm_id"].tolist())
        note_corpus = self.dataset.build_note_corpus(self.noteevents, all_hadm_ids=all_cohort["hadm_id"].tolist())
        treatment_totals = self.dataset.build_treatment_totals(self.icustays, self.ventdurations, self.vasopressordurations)
        acuity_scores = self.dataset.build_acuity_scores(self.oasis, self.sapsii)
        mistrust_scores = self.model.build_mistrust_score_table(
            feature_matrix=feature_matrix,
            note_labels=note_labels,
            note_corpus=note_corpus,
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
            sentiment_fn=self._sentiment_fn,
        )
        final_model_table = self.dataset.build_final_model_table(
            demographics=demographics,
            all_cohort=all_cohort,
            admissions=base,
            chartevents=self.chartevents,
            d_items=self.d_items,
            mistrust_scores=mistrust_scores,
            include_race=True,
            include_mistrust=True,
        )
        return {
            "base": base,
            "demographics": demographics,
            "all_cohort": all_cohort,
            "eol_cohort": eol_cohort,
            "feature_matrix": feature_matrix,
            "note_labels": note_labels,
            "note_corpus": note_corpus,
            "treatment_totals": treatment_totals,
            "acuity_scores": acuity_scores,
            "mistrust_scores": mistrust_scores,
            "final_model_table": final_model_table,
        }

    def _build_valid_environment(self):
        hadm_ids = list(range(1, 50003))
        subject_ids = list(range(100001, 150003))
        admissions = pd.DataFrame(
            {
                "hadm_id": hadm_ids,
                "subject_id": subject_ids,
                "admittime": ["2100-01-01 00:00:00"] * len(hadm_ids),
                "dischtime": ["2100-01-02 00:00:00"] * len(hadm_ids),
                "ethnicity": ["WHITE"] * len(hadm_ids),
                "insurance": ["Medicare"] * len(hadm_ids),
                "discharge_location": ["HOME"] * len(hadm_ids),
                "hospital_expire_flag": [0] * len(hadm_ids),
                "has_chartevents_data": [1] * len(hadm_ids),
            }
        )
        patients = pd.DataFrame(
            {
                "subject_id": subject_ids,
                "gender": ["M"] * len(subject_ids),
                "dob": ["2070-01-01 00:00:00"] * len(subject_ids),
            }
        )
        icustays = pd.DataFrame(
            {
                "hadm_id": hadm_ids,
                "icustay_id": list(range(700001, 750003)),
                "intime": ["2100-01-01 00:00:00"] * len(hadm_ids),
                "outtime": ["2100-01-01 13:00:00"] * len(hadm_ids),
            }
        )
        noteevents = pd.DataFrame([{"hadm_id": 1, "category": "Nursing", "text": "ok", "iserror": None}])
        chartevents = pd.DataFrame([{"hadm_id": 1, "itemid": 1, "value": "No", "icustay_id": 700001}])
        d_items = pd.DataFrame([{"itemid": 1, "label": "Education Readiness", "dbsource": "carevue"}])
        ventdurations = pd.DataFrame([{"icustay_id": 700001, "ventnum": 1, "starttime": "2100-01-01 00:00:00", "endtime": "2100-01-01 01:00:00", "duration_hours": 1.0}])
        vasopressordurations = pd.DataFrame([{"icustay_id": 700001, "vasonum": 1, "starttime": "2100-01-01 02:00:00", "endtime": "2100-01-01 03:00:00", "duration_hours": 1.0}])
        oasis = pd.DataFrame([{"hadm_id": 1, "icustay_id": 700001, "oasis": 10}])
        sapsii = pd.DataFrame([{"hadm_id": 1, "icustay_id": 700001, "sapsii": 30}])
        return (
            {
                "admissions": admissions,
                "patients": patients,
                "icustays": icustays,
                "noteevents": noteevents,
                "chartevents": chartevents,
                "d_items": d_items,
            },
            {
                "ventdurations": ventdurations,
                "vasopressordurations": vasopressordurations,
                "oasis": oasis,
                "sapsii": sapsii,
            },
        )

    def _build_deliverable_artifacts(self):
        artifacts = self._build_core_artifacts()
        return {
            "base_admissions": artifacts["base"],
            "eol_cohort": artifacts["eol_cohort"],
            "all_cohort": artifacts["all_cohort"],
            "treatment_totals": artifacts["treatment_totals"],
            "chartevent_feature_matrix": artifacts["feature_matrix"],
            "note_labels": artifacts["note_labels"],
            "mistrust_scores": artifacts["mistrust_scores"],
            "acuity_scores": artifacts["acuity_scores"],
            "final_model_table": artifacts["final_model_table"],
        }

    def test_dataset_public_api_exposes_expected_function_contracts(self):
        expected = {
            "build_base_admissions",
            "build_demographics_table",
            "build_all_cohort",
            "build_eol_cohort",
            "build_chartevent_feature_matrix",
            "build_note_corpus",
            "build_note_labels",
            "build_treatment_totals",
            "build_acuity_scores",
            "build_final_model_table",
            "validate_database_environment",
        }
        for name in expected:
            self.assertTrue(hasattr(self.dataset, name), msg=name)
            self.assertTrue(callable(getattr(self.dataset, name)), msg=name)

    def test_model_public_api_exposes_expected_function_contracts(self):
        expected = {
            "fit_proxy_mistrust_model",
            "build_proxy_probability_scores",
            "build_negative_sentiment_mistrust_scores",
            "z_normalize_scores",
            "build_mistrust_score_table",
            "summarize_feature_weights",
            "run_race_gap_analysis",
            "run_race_based_treatment_analysis",
            "run_trust_based_treatment_analysis",
            "run_acuity_control_analysis",
            "evaluate_downstream_predictions",
            "run_full_eol_mistrust_modeling",
        }
        for name in expected:
            self.assertTrue(hasattr(self.model, name), msg=name)
            self.assertTrue(callable(getattr(self.model, name)), msg=name)

    def test_dataset_helper_unit_rules_cover_mapping_and_whitespace_cleanup(self):
        self.assertEqual(self.dataset.map_ethnicity("BLACK/AFRICAN AMERICAN"), "BLACK")
        self.assertEqual(self.dataset.map_ethnicity("HISPANIC OR LATINO"), "HISPANIC")
        self.assertEqual(
            self.dataset.map_ethnicity("AMERICAN INDIAN/ALASKA NATIVE"),
            "NATIVE AMERICAN",
        )
        self.assertEqual(self.dataset.map_insurance("Medicare"), "Public")
        self.assertEqual(self.dataset.map_insurance("Private"), "Private")
        self.assertEqual(self.dataset.map_insurance("Self Pay"), "Self-Pay")
        self.assertEqual(self.dataset.map_insurance("Other Plan"), "Self-Pay")
        self.assertEqual(
            self.dataset.prepare_note_text_for_sentiment(" Date:[**5-1-18**]   calm   rapport "),
            "Date:[**5-1-18**] calm rapport",
        )

    def test_dataset_build_demographics_table_caps_age_and_computes_expected_columns(self):
        base = self.dataset.build_base_admissions(self.admissions, self.patients)
        demographics = self.dataset.build_demographics_table(base)
        self.assertEqual(
            demographics.columns.tolist(),
            [
                "hadm_id",
                "subject_id",
                "gender",
                "admittime",
                "dischtime",
                "ethnicity",
                "insurance_raw",
                "race",
                "age",
                "los_hours",
                "los_days",
                "insurance",
                "insurance_group",
            ],
        )
        by_hadm = demographics.set_index("hadm_id")
        self.assertEqual(float(by_hadm.loc[101, "age"]), 90.0)
        self.assertAlmostEqual(float(by_hadm.loc[101, "los_hours"]), 48.0, places=7)
        self.assertEqual(by_hadm.loc[105, "race"], "NATIVE AMERICAN")

    def test_dataset_build_base_admissions_filters_has_chartevents_and_enforces_unique_hadm(self):
        admissions = pd.concat(
            [
                self.admissions,
                pd.DataFrame(
                    [
                        {
                            "hadm_id": 107,
                            "subject_id": 207,
                            "admittime": "2100-01-07 00:00:00",
                            "dischtime": "2100-01-08 00:00:00",
                            "ethnicity": "WHITE",
                            "insurance": "Medicare",
                            "discharge_location": "HOME",
                            "hospital_expire_flag": 0,
                            "has_chartevents_data": 0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        patients = pd.concat(
            [
                self.patients,
                pd.DataFrame([{"subject_id": 207, "gender": "M", "dob": "2070-01-01 00:00:00"}]),
            ],
            ignore_index=True,
        )
        base = self.dataset.build_base_admissions(admissions, patients)
        self.assertNotIn(107, base["hadm_id"].tolist())
        self.assertEqual(len(base), len(set(base["hadm_id"])))

    def test_dataset_build_all_and_eol_cohorts_respect_duration_boundaries(self):
        base = self.dataset.build_base_admissions(self.admissions, self.patients)
        demographics = self.dataset.build_demographics_table(base)

        boundary_demo = pd.DataFrame([{"hadm_id": 1, "los_hours": 24.0}, {"hadm_id": 2, "los_hours": 24.01}])
        boundary_base = pd.DataFrame(
            [
                {"hadm_id": 1, "discharge_location": "SNF", "hospital_expire_flag": 0},
                {"hadm_id": 2, "discharge_location": "SNF", "hospital_expire_flag": 0},
            ]
        )
        eol = self.dataset.build_eol_cohort(boundary_base, boundary_demo)
        self.assertEqual(eol["hadm_id"].tolist(), [2])

        all_cohort = self.dataset.build_all_cohort(base, self.icustays)
        self.assertEqual(all_cohort["hadm_id"].tolist(), [101, 102, 103, 104, 105, 106])
        self.assertEqual(len(all_cohort), len(set(all_cohort["hadm_id"])))
        full_eol = self.dataset.build_eol_cohort(base, demographics)
        self.assertEqual(full_eol["hadm_id"].tolist(), [103, 104])

    def test_dataset_build_all_cohort_includes_any_icu_stay(self):
        base = pd.DataFrame([{"hadm_id": 1}, {"hadm_id": 2}])
        icustays = pd.DataFrame(
            [
                {
                    "hadm_id": 1,
                    "icustay_id": 1,
                    "intime": "2100-01-01 00:00:00",
                    "outtime": "2100-01-01 12:00:00",
                },
                {
                    "hadm_id": 2,
                    "icustay_id": 2,
                    "intime": "2100-01-01 00:00:00",
                    "outtime": "2100-01-01 11:59:00",
                },
            ]
        )
        cohort = self.dataset.build_all_cohort(base, icustays)
        self.assertEqual(cohort["hadm_id"].tolist(), [1, 2])

    def test_dataset_note_corpus_and_labels_filter_errors_and_capture_required_phrases(self):
        all_hadm_ids = [101, 102, 103, 104, 105, 106]
        note_corpus = self.dataset.build_note_corpus(self.noteevents, all_hadm_ids=all_hadm_ids)
        note_labels = self.dataset.build_note_labels(self.noteevents, all_hadm_ids=all_hadm_ids)

        self.assertEqual(note_corpus.columns.tolist(), ["hadm_id", "note_text"])
        self.assertEqual(note_corpus["hadm_id"].tolist(), all_hadm_ids)
        self.assertEqual(note_corpus.set_index("hadm_id").loc[105, "note_text"], "")
        by_hadm = note_labels.set_index("hadm_id")
        self.assertEqual(int(by_hadm.loc[101, "noncompliance_label"]), 1)
        self.assertEqual(int(by_hadm.loc[102, "autopsy_label"]), 1)
        self.assertEqual(int(by_hadm.loc[103, "noncompliance_label"]), 0)

    def test_dataset_build_note_corpus_concatenates_with_single_spaces_and_drops_only_iserror_one(self):
        notes = pd.DataFrame(
            [
                {"hadm_id": 1, "category": "Nursing", "text": "first", "iserror": None},
                {"hadm_id": 1, "category": "Nursing", "text": "second", "iserror": 0},
                {"hadm_id": 1, "category": "Nursing", "text": "third", "iserror": 1},
                {"hadm_id": 2, "category": "Nursing", "text": " lone ", "iserror": None},
            ]
        )
        corpus = self.dataset.build_note_corpus(notes, all_hadm_ids=[1, 2, 3])
        by_hadm = corpus.set_index("hadm_id")
        self.assertEqual(by_hadm.loc[1, "note_text"], "first second")
        self.assertEqual(by_hadm.loc[2, "note_text"], "lone")
        self.assertEqual(by_hadm.loc[3, "note_text"], "")

    def test_dataset_identify_table2_itemids_and_feature_matrix_support_partial_matching_binary_rows_and_zero_rows(self):
        itemids = self.dataset.identify_table2_itemids(self.d_items)
        self.assertEqual(itemids, {1, 2, 3})

        feature_matrix = self.dataset.build_chartevent_feature_matrix(
            self.chartevents,
            self.d_items,
            all_hadm_ids=[101, 102, 103, 104],
        )
        self.assertEqual(feature_matrix["hadm_id"].tolist(), [101, 102, 103, 104])
        self.assertEqual(
            sorted([column for column in feature_matrix.columns if column != "hadm_id"]),
            [
                "Education Readiness: No",
                "Education Readiness: Yes",
                "Pain Level: 7-Mod to Severe",
                "Pain Level: None",
                "Richmond-RAS Scale Assessment: 0 Alert and Calm",
            ],
        )
        row_101 = feature_matrix.set_index("hadm_id").loc[101]
        self.assertEqual(int(row_101["Education Readiness: No"]), 1)
        self.assertEqual(int(row_101["Pain Level: 7-Mod to Severe"]), 1)
        row_104 = feature_matrix.set_index("hadm_id").loc[104]
        self.assertTrue((row_104.fillna(0).astype(int) == 0).all())

    def test_dataset_identify_table2_itemids_does_not_overmatch_unrelated_labels(self):
        d_items = pd.DataFrame(
            [
                {"itemid": 1, "label": "Education Readiness", "dbsource": "carevue"},
                {"itemid": 2, "label": "Readiness Scoreboard", "dbsource": "carevue"},
                {"itemid": 3, "label": "Orientation", "dbsource": "carevue"},
                {"itemid": 4, "label": "Random Unrelated Label", "dbsource": "carevue"},
            ]
        )
        itemids = self.dataset.identify_table2_itemids(d_items)
        self.assertEqual(itemids, {1, 3})

    def test_dataset_feature_matrix_is_binary_integer_typed_and_sorted_by_hadm(self):
        feature_matrix = self.dataset.build_chartevent_feature_matrix(
            self.chartevents.sample(frac=1.0, random_state=0),
            self.d_items,
            all_hadm_ids=[106, 103, 102, 101],
        )
        self.assertEqual(feature_matrix["hadm_id"].tolist(), [101, 102, 103, 106])
        for column in feature_matrix.columns:
            if column == "hadm_id":
                continue
            self.assertTrue(pd.api.types.is_integer_dtype(feature_matrix[column]))
            self.assertTrue(set(feature_matrix[column].dropna().unique()).issubset({0, 1}))

    def test_dataset_build_treatment_totals_merges_gap_boundary_and_outputs_sorted_schema(self):
        boundary_icu = pd.DataFrame(
            [{"hadm_id": 1, "icustay_id": 99, "intime": "2100-01-01", "outtime": "2100-01-02"}]
        )
        boundary_vent = pd.DataFrame(
            [
                {"icustay_id": 99, "ventnum": 1, "starttime": "2100-01-01 00:00:00", "endtime": "2100-01-01 01:00:00", "duration_hours": 1.0},
                {"icustay_id": 99, "ventnum": 2, "starttime": "2100-01-01 11:00:00", "endtime": "2100-01-01 12:00:00", "duration_hours": 1.0},
                {"icustay_id": 99, "ventnum": 3, "starttime": "2100-01-01 22:01:00", "endtime": "2100-01-01 23:01:00", "duration_hours": 1.0},
            ]
        )
        empty_vaso = pd.DataFrame(columns=["icustay_id", "vasonum", "starttime", "endtime", "duration_hours"])
        totals = self.dataset.build_treatment_totals(boundary_icu, boundary_vent, empty_vaso)
        self.assertEqual(totals.columns.tolist(), ["hadm_id", "total_vent_min", "total_vaso_min"])
        row = totals.fillna(0).set_index("hadm_id").loc[1]
        self.assertEqual(float(row["total_vent_min"]), 780.0)

    def test_dataset_build_final_model_table_returns_exact_full_schema_order(self):
        artifacts = self._build_core_artifacts()
        final_model_table = artifacts["final_model_table"]
        self.assertEqual(
            final_model_table.columns.tolist(),
            [
                "hadm_id",
                "age",
                "los_days",
                "gender_f",
                "gender_m",
                "insurance_private",
                "insurance_public",
                "insurance_self_pay",
                "race_white",
                "race_black",
                "race_asian",
                "race_hispanic",
                "race_native_american",
                "race_other",
                "noncompliance_score_z",
                "autopsy_score_z",
                "negative_sentiment_score_z",
                "left_ama",
                "in_hospital_mortality",
                "code_status_dnr_dni_cmo",
            ],
        )
        self.assertEqual(final_model_table["hadm_id"].tolist(), [101, 102, 103, 104, 105, 106])
        self.assertEqual(int(final_model_table.set_index("hadm_id").loc[102, "left_ama"]), 1)
        self.assertEqual(
            int(final_model_table.set_index("hadm_id").loc[102, "code_status_dnr_dni_cmo"]),
            1,
        )

    def test_dataset_validate_database_environment_accepts_valid_minimal_environment(self):
        raw_tables, materialized_views = self._build_valid_environment()
        summary = self.dataset.validate_database_environment(raw_tables, materialized_views)
        self.assertEqual(summary["database_flavor"], "postgresql")
        self.assertEqual(summary["schema_name"], "mimiciii")
        self.assertGreater(summary["base_admissions_rows"], 50000)
        self.assertIn("admissions", summary["raw_tables"])
        self.assertIn("oasis", summary["materialized_views"])

    def test_dataset_z_normalize_scores_turns_constant_columns_to_zero(self):
        score_table = pd.DataFrame(
            [
                {"hadm_id": 1, "score_a": 1.0, "score_b": 5.0},
                {"hadm_id": 2, "score_a": 2.0, "score_b": 5.0},
                {"hadm_id": 3, "score_a": 3.0, "score_b": 5.0},
            ]
        )
        normalized = self.dataset.z_normalize_scores(score_table, columns=["score_a", "score_b"])
        self.assertEqual(normalized["hadm_id"].tolist(), [1, 2, 3])
        self.assertAlmostEqual(float(normalized["score_a"].mean()), 0.0, places=7)
        self.assertTrue((normalized["score_b"] == 0.0).all())

    def test_model_helper_units_cover_score_column_and_note_cleanup_rules(self):
        self.assertEqual(self.model._score_column_name("noncompliance_label"), "noncompliance_score")
        self.assertEqual(self.model._score_column_name("custom_target"), "custom_target_score")
        self.assertEqual(
            self.model._prepare_note_text_for_sentiment(" Date:[**5-1-18**]   calm   rapport "),
            "Date:[**5-1-18**] calm rapport",
        )
        self.assertEqual(self.model._prepare_note_text_for_sentiment(None), "")

    def test_model_fit_and_proxy_probability_functions_use_full_input_and_sorted_positive_scores(self):
        artifacts = self._build_core_artifacts()
        created = []

        class _RecordingLogisticRegression:
            def __init__(self, *args, **kwargs):
                del args
                self.kwargs = kwargs
                self.coef_ = None
                self.fit_X = None
                self.fit_y = None
                created.append(self)

            def fit(self, X, y):
                self.fit_X = X.copy()
                self.fit_y = y.copy()
                self.coef_ = [[0.1] * X.shape[1]]
                return self

            def predict_proba(self, X):
                return [[0.2, 0.8] for _ in range(len(X))]

        with patch.object(self.model, "LogisticRegression", _RecordingLogisticRegression):
            self.model.fit_proxy_mistrust_model(
                artifacts["feature_matrix"],
                artifacts["note_labels"],
                "noncompliance_label",
            )

        self.assertEqual(created[0].kwargs["penalty"], "l1")
        self.assertEqual(created[0].kwargs["C"], 0.1)
        self.assertEqual(created[0].kwargs["solver"], "liblinear")
        self.assertEqual(created[0].kwargs["max_iter"], 1000)
        self.assertEqual(len(created[0].fit_X), len(artifacts["feature_matrix"]))

        scores = self.model.build_proxy_probability_scores(
            artifacts["feature_matrix"],
            artifacts["note_labels"],
            "autopsy_label",
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
        )
        self.assertEqual(scores.columns.tolist(), ["hadm_id", "autopsy_score"])
        self.assertEqual(scores["hadm_id"].tolist(), sorted(scores["hadm_id"].tolist()))

    def test_model_build_proxy_probability_scores_supports_nonstandard_label_names_and_bad_probability_shapes(self):
        feature_matrix = pd.DataFrame(
            [
                {"hadm_id": 1, "feature_a": 1},
                {"hadm_id": 2, "feature_a": 0},
            ]
        )
        note_labels = pd.DataFrame(
            [
                {"hadm_id": 1, "custom_target": 1},
                {"hadm_id": 2, "custom_target": 0},
            ]
        )
        scores = self.model.build_proxy_probability_scores(
            feature_matrix,
            note_labels,
            "custom_target",
            estimator_factory=lambda: _FakeProbEstimator([0.2, 0.8]),
        )
        self.assertEqual(scores.columns.tolist(), ["hadm_id", "custom_target_score"])

        class _MalformedProbEstimator:
            def fit(self, X, y):
                del X, y
                self.coef_ = [[0.1]]
                return self

            def predict_proba(self, X):
                return [[1.0] for _ in range(len(X))]

        with self.assertRaises(IndexError):
            self.model.build_proxy_probability_scores(
                feature_matrix,
                note_labels,
                "custom_target",
                estimator_factory=lambda: _MalformedProbEstimator(),
            )

    def test_model_negative_sentiment_and_normalization_functions_return_stable_schemas(self):
        artifacts = self._build_core_artifacts()
        sentiment_scores = self.model.build_negative_sentiment_mistrust_scores(
            artifacts["note_corpus"],
            sentiment_fn=self._sentiment_fn,
        )
        self.assertEqual(sentiment_scores.columns.tolist(), ["hadm_id", "negative_sentiment_score"])
        self.assertEqual(sentiment_scores["hadm_id"].tolist(), sorted(sentiment_scores["hadm_id"].tolist()))

        normalized = self.model.z_normalize_scores(
            pd.DataFrame(
                [
                    {"hadm_id": 1, "score_a": 1.0, "score_b": 10.0},
                    {"hadm_id": 2, "score_a": 2.0, "score_b": 10.0},
                    {"hadm_id": 3, "score_a": 3.0, "score_b": 10.0},
                ]
            ),
            columns=["score_a", "score_b"],
        )
        self.assertEqual(normalized["hadm_id"].tolist(), [1, 2, 3])
        self.assertTrue((normalized["score_b"] == 0.0).all())

    def test_model_negative_sentiment_handles_none_and_whitespace_only_notes(self):
        note_corpus = pd.DataFrame(
            [
                {"hadm_id": 2, "note_text": "   "},
                {"hadm_id": 1, "note_text": None},
            ]
        )
        seen = []

        def _sentiment(text):
            seen.append(text)
            return (0.25, 0.0)

        scores = self.model.build_negative_sentiment_mistrust_scores(note_corpus, sentiment_fn=_sentiment)
        self.assertEqual(seen, ["", ""])
        self.assertEqual(scores["hadm_id"].tolist(), [1, 2])
        self.assertEqual(scores["negative_sentiment_score"].tolist(), [-0.25, -0.25])

    def test_model_z_normalize_scores_handles_all_nan_columns(self):
        score_table = pd.DataFrame(
            [
                {"hadm_id": 1, "score_a": float("nan")},
                {"hadm_id": 2, "score_a": float("nan")},
            ]
        )
        normalized = self.model.z_normalize_scores(score_table, columns=["score_a"])
        self.assertEqual(normalized["hadm_id"].tolist(), [1, 2])
        self.assertTrue((normalized["score_a"] == 0.0).all())

    def test_model_build_mistrust_score_table_returns_required_schema_sorted_unique_and_float_scores(self):
        artifacts = self._build_core_artifacts()
        mistrust_scores = artifacts["mistrust_scores"]
        self.assertEqual(
            mistrust_scores.columns.tolist(),
            [
                "hadm_id",
                "noncompliance_score_z",
                "autopsy_score_z",
                "negative_sentiment_score_z",
            ],
        )
        self.assertEqual(mistrust_scores["hadm_id"].tolist(), sorted(mistrust_scores["hadm_id"].tolist()))
        self.assertEqual(len(mistrust_scores), len(set(mistrust_scores["hadm_id"])))
        for column in mistrust_scores.columns[1:]:
            self.assertTrue(pd.api.types.is_float_dtype(mistrust_scores[column]))

    def test_model_summarize_feature_weights_returns_sorted_positive_and_negative_rankings(self):
        estimator = _FakeProbEstimator([0.1, 0.9])
        estimator.coef_ = [[0.7, -0.2, 0.1]]
        summary = self.model.summarize_feature_weights(
            estimator,
            ["Education Readiness: No", "Pain Level: None", "State: Alert"],
            top_n=2,
        )
        self.assertEqual(set(summary.keys()), {"all", "positive", "negative"})
        self.assertEqual(summary["positive"]["feature"].tolist(), ["Education Readiness: No", "State: Alert"])
        self.assertEqual(summary["negative"]["feature"].tolist(), ["Pain Level: None", "State: Alert"])

    def test_model_race_gap_treatment_and_acuity_functions_return_required_schemas(self):
        artifacts = self._build_core_artifacts()
        race_gap = self.model.run_race_gap_analysis(
            artifacts["mistrust_scores"],
            artifacts["demographics"],
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

        race_treatment = self.model.run_race_based_treatment_analysis(
            artifacts["eol_cohort"],
            artifacts["treatment_totals"],
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

        trust_treatment = self.model.run_trust_based_treatment_analysis(
            artifacts["eol_cohort"],
            artifacts["mistrust_scores"],
            artifacts["treatment_totals"],
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

        acuity = self.model.run_acuity_control_analysis(
            artifacts["mistrust_scores"],
            artifacts["acuity_scores"],
        )
        self.assertEqual(
            acuity.columns.tolist(),
            ["feature_a", "feature_b", "correlation", "pvalue", "n"],
        )
        self.assertTrue(
            ((acuity["feature_a"] == "oasis") & (acuity["feature_b"] == "sapsii")).any()
            or ((acuity["feature_a"] == "sapsii") & (acuity["feature_b"] == "oasis")).any()
        )

    def test_model_run_trust_based_treatment_analysis_handles_invalid_n_and_exact_median_gap(self):
        eol = pd.DataFrame(
            [
                {"hadm_id": 1, "race": "WHITE"},
                {"hadm_id": 2, "race": "BLACK"},
                {"hadm_id": 3, "race": "BLACK"},
            ]
        )
        scores = pd.DataFrame(
            [
                {"hadm_id": 1, "noncompliance_score_z": 0.1},
                {"hadm_id": 2, "noncompliance_score_z": 0.9},
                {"hadm_id": 3, "noncompliance_score_z": 0.5},
            ]
        )
        treatments = pd.DataFrame(
            [
                {"hadm_id": 1, "total_vent_min": 10.0},
                {"hadm_id": 2, "total_vent_min": 40.0},
                {"hadm_id": 3, "total_vent_min": 20.0},
            ]
        )
        invalid = self.model.run_trust_based_treatment_analysis(
            eol,
            scores,
            treatments,
            score_columns=["noncompliance_score_z"],
            treatment_columns=["total_vent_min"],
            group_sizes={"total_vent_min": 0},
        )
        self.assertTrue(pd.isna(invalid.loc[0, "median_gap"]))

        valid = self.model.run_trust_based_treatment_analysis(
            eol,
            scores,
            treatments,
            score_columns=["noncompliance_score_z"],
            treatment_columns=["total_vent_min"],
            group_sizes={"total_vent_min": 1},
        )
        self.assertEqual(float(valid.loc[0, "median_high"]), 40.0)
        self.assertEqual(float(valid.loc[0, "median_low"]), 15.0)
        self.assertEqual(float(valid.loc[0, "median_gap"]), 25.0)

    def test_model_evaluate_downstream_predictions_returns_18_rows_and_respects_split_contract(self):
        artifacts = self._build_core_artifacts()
        final_model_table = artifacts["final_model_table"].copy()
        final_model_table["left_ama"] = [0, 1, 0, 1, 0, 1]
        final_model_table["code_status_dnr_dni_cmo"] = [1, 0, 1, 0, 1, 0]
        final_model_table["in_hospital_mortality"] = [0, 1, 0, 1, 0, 1]
        splitter = _SplitRecorder()
        auc_recorder = _AUCRecorder(0.8)
        results = self.model.evaluate_downstream_predictions(
            final_model_table,
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9]),
            split_fn=splitter,
            auc_fn=auc_recorder,
            repetitions=3,
        )
        self.assertEqual(len(results), 18)
        self.assertEqual(set(results["task"]), set(self.model.DOWNSTREAM_TASK_MAP.keys()))
        self.assertEqual(set(results["configuration"]), set(self.model.DOWNSTREAM_FEATURE_CONFIGS.keys()))
        self.assertTrue((results["n_repeats"] == 3).all())
        self.assertEqual(splitter.calls[0]["test_size"], 0.4)
        self.assertEqual(splitter.calls[0]["random_state"], 0)
        self.assertTrue(all(abs(float(value) - 0.8) < 1e-9 for value in results["auc_mean"]))

    def test_model_evaluate_downstream_predictions_drops_null_rows_before_splitting(self):
        artifacts = self._build_core_artifacts()
        final_model_table = artifacts["final_model_table"].copy()
        final_model_table["left_ama"] = [0, 1, 0, 1, 0, 1]
        final_model_table.loc[0, "age"] = float("nan")
        calls = []

        def _splitter(X, y, test_size, random_state):
            features = X.reset_index(drop=True)
            labels = pd.Series(y).reset_index(drop=True)
            calls.append(
                {
                    "n_rows": len(features),
                    "test_size": test_size,
                    "random_state": random_state,
                }
            )
            return (
                features.iloc[:3].copy(),
                features.iloc[3:].copy(),
                labels.iloc[:3].copy(),
                labels.iloc[3:].copy(),
            )

        self.model.evaluate_downstream_predictions(
            final_model_table,
            task_map={"Left AMA": "left_ama"},
            feature_configurations={"Baseline": self.model.BASELINE_FEATURE_COLUMNS},
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9]),
            split_fn=_splitter,
            auc_fn=_AUCRecorder(0.7),
            repetitions=1,
        )
        self.assertEqual(calls[0]["n_rows"], 5)

    def test_model_run_full_eol_mistrust_modeling_returns_expected_sections_and_aligned_outputs(self):
        artifacts = self._build_core_artifacts()
        outputs = self.model.run_full_eol_mistrust_modeling(
            feature_matrix=artifacts["feature_matrix"],
            note_labels=artifacts["note_labels"],
            note_corpus=artifacts["note_corpus"],
            demographics=artifacts["demographics"],
            eol_cohort=artifacts["eol_cohort"],
            treatment_totals=artifacts["treatment_totals"],
            acuity_scores=artifacts["acuity_scores"],
            final_model_table=artifacts["final_model_table"],
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
            sentiment_fn=self._sentiment_fn,
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.7),
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
        self.assertEqual(
            outputs["mistrust_scores"]["hadm_id"].tolist(),
            artifacts["final_model_table"]["hadm_id"].tolist(),
        )
        self.assertEqual(len(outputs["downstream_auc_results"]), 18)

    def test_model_run_full_eol_mistrust_modeling_returns_only_base_outputs_when_optional_inputs_absent(self):
        artifacts = self._build_core_artifacts()
        outputs = self.model.run_full_eol_mistrust_modeling(
            feature_matrix=artifacts["feature_matrix"],
            note_labels=artifacts["note_labels"],
            note_corpus=artifacts["note_corpus"],
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
            sentiment_fn=self._sentiment_fn,
            repetitions=1,
        )
        self.assertEqual(set(outputs.keys()), {"mistrust_scores", "feature_weight_summaries"})

    def test_dataset_public_functions_raise_clear_errors_for_missing_required_columns(self):
        with self.assertRaisesRegex(ValueError, "subject_id"):
            self.dataset.build_base_admissions(
                self.admissions,
                self.patients.drop(columns=["subject_id"]),
            )

        with self.assertRaisesRegex(ValueError, "value"):
            self.dataset.build_chartevent_feature_matrix(
                self.chartevents.drop(columns=["value"]),
                self.d_items,
            )

        artifacts = self._build_core_artifacts()
        with self.assertRaisesRegex(ValueError, "noncompliance_score_z"):
            self.dataset.build_final_model_table(
                demographics=artifacts["demographics"],
                all_cohort=artifacts["all_cohort"],
                admissions=artifacts["base"],
                chartevents=self.chartevents,
                d_items=self.d_items,
                mistrust_scores=artifacts["mistrust_scores"].drop(columns=["noncompliance_score_z"]),
            )

        raw_tables, materialized_views = self._build_valid_environment()
        del raw_tables["admissions"]
        with self.assertRaisesRegex(ValueError, "Missing required raw tables"):
            self.dataset.validate_database_environment(raw_tables, materialized_views)

    def test_dataset_empty_input_contracts_return_stable_schemas(self):
        empty_notes = pd.DataFrame(columns=["hadm_id", "category", "text", "iserror"])
        note_corpus = self.dataset.build_note_corpus(empty_notes, all_hadm_ids=[1, 2])
        self.assertEqual(note_corpus.columns.tolist(), ["hadm_id", "note_text"])
        self.assertEqual(note_corpus["hadm_id"].tolist(), [1, 2])
        self.assertEqual(note_corpus["note_text"].tolist(), ["", ""])

        note_labels = self.dataset.build_note_labels(empty_notes, all_hadm_ids=[1, 2])
        self.assertEqual(
            note_labels.columns.tolist(),
            ["hadm_id", "noncompliance_label", "autopsy_label"],
        )
        self.assertTrue((note_labels["noncompliance_label"] == 0).all())
        self.assertTrue((note_labels["autopsy_label"] == 0).all())

        empty_treatments = self.dataset.build_treatment_totals(
            self.icustays,
            pd.DataFrame(columns=["icustay_id", "ventnum", "starttime", "endtime", "duration_hours"]),
            pd.DataFrame(columns=["icustay_id", "vasonum", "starttime", "endtime", "duration_hours"]),
        )
        self.assertEqual(
            empty_treatments.columns.tolist(),
            ["hadm_id", "total_vent_min", "total_vaso_min"],
        )
        self.assertTrue(empty_treatments.empty)

        empty_acuity = self.dataset.build_acuity_scores(
            pd.DataFrame(columns=["hadm_id", "icustay_id", "oasis"]),
            pd.DataFrame(columns=["hadm_id", "icustay_id", "sapsii"]),
        )
        self.assertEqual(empty_acuity.columns.tolist(), ["hadm_id", "oasis", "sapsii"])
        self.assertTrue(empty_acuity.empty)

    def test_dataset_private_span_merge_helper_covers_empty_single_overlap_and_gap_boundaries(self):
        empty = pd.DataFrame(columns=["starttime", "endtime"])
        self.assertEqual(self.dataset._merge_spans_for_hadm(empty), 0.0)

        single = pd.DataFrame(
            [{"starttime": pd.Timestamp("2100-01-01 00:00:00"), "endtime": pd.Timestamp("2100-01-01 01:00:00")}]
        )
        self.assertEqual(self.dataset._merge_spans_for_hadm(single), 60.0)

        overlap = pd.DataFrame(
            [
                {"starttime": pd.Timestamp("2100-01-01 00:00:00"), "endtime": pd.Timestamp("2100-01-01 01:00:00")},
                {"starttime": pd.Timestamp("2100-01-01 00:30:00"), "endtime": pd.Timestamp("2100-01-01 01:30:00")},
            ]
        )
        self.assertEqual(self.dataset._merge_spans_for_hadm(overlap), 90.0)

        boundary = pd.DataFrame(
            [
                {"starttime": pd.Timestamp("2100-01-01 00:00:00"), "endtime": pd.Timestamp("2100-01-01 01:00:00")},
                {"starttime": pd.Timestamp("2100-01-01 11:00:00"), "endtime": pd.Timestamp("2100-01-01 12:00:00")},
                {"starttime": pd.Timestamp("2100-01-01 22:01:00"), "endtime": pd.Timestamp("2100-01-01 23:01:00")},
            ]
        )
        self.assertEqual(self.dataset._merge_spans_for_hadm(boundary), 780.0)

    def test_dataset_build_final_model_table_schema_toggles_include_race_and_mistrust(self):
        artifacts = self._build_core_artifacts()
        baseline_only = self.dataset.build_final_model_table(
            demographics=artifacts["demographics"],
            all_cohort=artifacts["all_cohort"],
            admissions=artifacts["base"],
            chartevents=self.chartevents,
            d_items=self.d_items,
            mistrust_scores=artifacts["mistrust_scores"],
            include_race=False,
            include_mistrust=False,
        )
        self.assertEqual(
            baseline_only.columns.tolist(),
            [
                "hadm_id",
                "age",
                "los_days",
                "gender_f",
                "gender_m",
                "insurance_private",
                "insurance_public",
                "insurance_self_pay",
                "left_ama",
                "in_hospital_mortality",
                "code_status_dnr_dni_cmo",
            ],
        )

        race_only = self.dataset.build_final_model_table(
            demographics=artifacts["demographics"],
            all_cohort=artifacts["all_cohort"],
            admissions=artifacts["base"],
            chartevents=self.chartevents,
            d_items=self.d_items,
            mistrust_scores=artifacts["mistrust_scores"],
            include_race=True,
            include_mistrust=False,
        )
        self.assertTrue(all(column in race_only.columns for column in self.model.RACE_FEATURE_COLUMNS))
        self.assertFalse(any(column in race_only.columns for column in self.model.MISTRUST_SCORE_COLUMNS))

    def test_dataset_function_outputs_are_deterministic_across_repeated_runs(self):
        base_one = self.dataset.build_base_admissions(self.admissions, self.patients)
        base_two = self.dataset.build_base_admissions(self.admissions, self.patients)
        pd.testing.assert_frame_equal(base_one, base_two)

        labels_one = self.dataset.build_note_labels(self.noteevents, all_hadm_ids=[101, 102, 103, 104, 105, 106])
        labels_two = self.dataset.build_note_labels(self.noteevents, all_hadm_ids=[101, 102, 103, 104, 105, 106])
        pd.testing.assert_frame_equal(labels_one, labels_two)

        matrix_one = self.dataset.build_chartevent_feature_matrix(
            self.chartevents,
            self.d_items,
            all_hadm_ids=[101, 102, 103, 104, 105, 106],
        )
        matrix_two = self.dataset.build_chartevent_feature_matrix(
            self.chartevents,
            self.d_items,
            all_hadm_ids=[101, 102, 103, 104, 105, 106],
        )
        pd.testing.assert_frame_equal(matrix_one, matrix_two)

    def test_model_private_metric_helpers_cover_empty_nan_and_small_sample_edges(self):
        statistic, pvalue, med_left, med_right, n_left, n_right = self.model._make_metric_result(
            pd.Series([], dtype=float),
            pd.Series([1.0, 2.0], dtype=float),
        )
        self.assertTrue(pd.isna(statistic))
        self.assertTrue(pd.isna(pvalue))
        self.assertTrue(pd.isna(med_left))
        self.assertTrue(pd.isna(med_right))
        self.assertEqual((n_left, n_right), (0, 2))

        corr, corr_pvalue, n = self.model._pearson_with_pvalue(
            pd.Series([1.0]),
            pd.Series([2.0]),
        )
        self.assertTrue(pd.isna(corr))
        self.assertTrue(pd.isna(corr_pvalue))
        self.assertEqual(n, 1)

        corr_nan, corr_nan_pvalue, n_nan = self.model._pearson_with_pvalue(
            pd.Series([float("nan"), float("nan")]),
            pd.Series([1.0, 2.0]),
        )
        self.assertTrue(pd.isna(corr_nan))
        self.assertTrue(pd.isna(corr_nan_pvalue))
        self.assertEqual(n_nan, 0)

    def test_model_public_functions_raise_clear_errors_for_missing_required_columns(self):
        with self.assertRaisesRegex(ValueError, "note_text"):
            self.model.build_negative_sentiment_mistrust_scores(
                pd.DataFrame([{"hadm_id": 1, "text": "oops"}]),
                sentiment_fn=self._sentiment_fn,
            )

        with self.assertRaisesRegex(ValueError, "race"):
            self.model.run_race_gap_analysis(
                pd.DataFrame([{"hadm_id": 1, "noncompliance_score_z": 0.1}]),
                pd.DataFrame([{"hadm_id": 1}]),
                score_columns=["noncompliance_score_z"],
            )

        with self.assertRaisesRegex(ValueError, "left_ama"):
            self.model.evaluate_downstream_predictions(
                self._build_core_artifacts()["final_model_table"].drop(columns=["left_ama"]),
                task_map={"Left AMA": "left_ama"},
                feature_configurations={"Baseline": self.model.BASELINE_FEATURE_COLUMNS},
                repetitions=1,
            )

    def test_model_public_functions_propagate_estimator_errors(self):
        artifacts = self._build_core_artifacts()

        class _FitFailureEstimator:
            def fit(self, X, y):
                del X, y
                raise RuntimeError("fit failed")

        with self.assertRaisesRegex(RuntimeError, "fit failed"):
            self.model.build_proxy_probability_scores(
                artifacts["feature_matrix"],
                artifacts["note_labels"],
                "noncompliance_label",
                estimator_factory=lambda: _FitFailureEstimator(),
            )

        n_features = len(self.model.BASELINE_FEATURE_COLUMNS)

        class _PredictFailureEstimator:
            def fit(self, X, y):
                del X, y
                self.coef_ = [[0.1] * n_features]
                return self

            def predict_proba(self, X):
                del X
                raise RuntimeError("predict failed")

        final_model_table = artifacts["final_model_table"].copy()
        final_model_table["left_ama"] = [0, 1, 0, 1, 0, 1]
        with self.assertRaisesRegex(RuntimeError, "predict failed"):
            self.model.evaluate_downstream_predictions(
                final_model_table,
                task_map={"Left AMA": "left_ama"},
                feature_configurations={"Baseline": self.model.BASELINE_FEATURE_COLUMNS},
                estimator_factory=lambda: _PredictFailureEstimator(),
                split_fn=_SplitRecorder(),
                repetitions=1,
            )

    def test_duplicate_key_contracts_are_enforced_or_deduplicated_as_expected(self):
        duplicate_patients = pd.concat([self.patients, self.patients.iloc[[0]]], ignore_index=True)
        with self.assertRaises(Exception):
            self.dataset.build_base_admissions(self.admissions, duplicate_patients)

        artifacts = self._build_core_artifacts()
        duplicate_features = pd.concat(
            [artifacts["feature_matrix"], artifacts["feature_matrix"].iloc[[0]]],
            ignore_index=True,
        )
        with self.assertRaises(Exception):
            self.model.build_proxy_probability_scores(
                duplicate_features,
                artifacts["note_labels"],
                "noncompliance_label",
                estimator_factory=lambda: _FakeProbEstimator([0.1] * (len(duplicate_features))),
            )

    def test_model_function_outputs_are_deterministic_across_repeated_runs(self):
        artifacts = self._build_core_artifacts()
        scores_one = self.model.build_mistrust_score_table(
            artifacts["feature_matrix"],
            artifacts["note_labels"],
            artifacts["note_corpus"],
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
            sentiment_fn=self._sentiment_fn,
        )
        scores_two = self.model.build_mistrust_score_table(
            artifacts["feature_matrix"],
            artifacts["note_labels"],
            artifacts["note_corpus"],
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
            sentiment_fn=self._sentiment_fn,
        )
        pd.testing.assert_frame_equal(scores_one, scores_two)

        final_model_table = artifacts["final_model_table"].copy()
        final_model_table["left_ama"] = [0, 1, 0, 1, 0, 1]
        final_model_table["code_status_dnr_dni_cmo"] = [1, 0, 1, 0, 1, 0]
        final_model_table["in_hospital_mortality"] = [0, 1, 0, 1, 0, 1]
        downstream_one = self.model.evaluate_downstream_predictions(
            final_model_table,
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9]),
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.77),
            repetitions=2,
        )
        downstream_two = self.model.evaluate_downstream_predictions(
            final_model_table,
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9]),
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.77),
            repetitions=2,
        )
        pd.testing.assert_frame_equal(downstream_one, downstream_two)

    def test_integration_end_to_end_pipeline_runs_from_dataset_sources_to_model_outputs(self):
        artifacts = self._build_core_artifacts()
        final_model_table = artifacts["final_model_table"].copy()
        final_model_table["left_ama"] = [0, 1, 0, 1, 0, 1]
        final_model_table["code_status_dnr_dni_cmo"] = [1, 0, 1, 0, 1, 0]
        final_model_table["in_hospital_mortality"] = [0, 1, 0, 1, 0, 1]
        outputs = self.model.run_full_eol_mistrust_modeling(
            feature_matrix=artifacts["feature_matrix"],
            note_labels=artifacts["note_labels"],
            note_corpus=artifacts["note_corpus"],
            demographics=artifacts["demographics"],
            eol_cohort=artifacts["eol_cohort"],
            treatment_totals=artifacts["treatment_totals"],
            acuity_scores=artifacts["acuity_scores"],
            final_model_table=final_model_table,
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
            sentiment_fn=self._sentiment_fn,
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.7),
            repetitions=1,
        )
        self.assertEqual(outputs["mistrust_scores"]["hadm_id"].tolist(), [101, 102, 103, 104, 105, 106])
        self.assertEqual(outputs["downstream_auc_results"].shape[0], 18)

    def test_integration_stage_to_stage_contracts_accept_upstream_outputs_without_translation(self):
        base = self.dataset.build_base_admissions(self.admissions, self.patients)
        demographics = self.dataset.build_demographics_table(base)
        all_cohort = self.dataset.build_all_cohort(base, self.icustays)
        feature_matrix = self.dataset.build_chartevent_feature_matrix(
            self.chartevents,
            self.d_items,
            all_hadm_ids=all_cohort["hadm_id"].tolist(),
        )
        note_labels = self.dataset.build_note_labels(self.noteevents, all_hadm_ids=all_cohort["hadm_id"].tolist())
        note_corpus = self.dataset.build_note_corpus(self.noteevents, all_hadm_ids=all_cohort["hadm_id"].tolist())
        mistrust_scores = self.model.build_mistrust_score_table(
            feature_matrix,
            note_labels,
            note_corpus,
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
            sentiment_fn=self._sentiment_fn,
        )
        final_model_table = self.dataset.build_final_model_table(
            demographics=demographics,
            all_cohort=all_cohort,
            admissions=base,
            chartevents=self.chartevents,
            d_items=self.d_items,
            mistrust_scores=mistrust_scores,
            include_race=True,
            include_mistrust=True,
        )
        results = self.model.evaluate_downstream_predictions(
            final_model_table.assign(
                left_ama=[0, 1, 0, 1, 0, 1],
                code_status_dnr_dni_cmo=[1, 0, 1, 0, 1, 0],
                in_hospital_mortality=[0, 1, 0, 1, 0, 1],
            ),
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9]),
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.6),
            repetitions=1,
        )
        self.assertEqual(results.shape[0], 18)

    def test_integration_optional_input_permutations_return_expected_output_sections(self):
        artifacts = self._build_core_artifacts()
        base_kwargs = {
            "feature_matrix": artifacts["feature_matrix"],
            "note_labels": artifacts["note_labels"],
            "note_corpus": artifacts["note_corpus"],
            "estimator_factory": lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
            "sentiment_fn": self._sentiment_fn,
            "repetitions": 1,
        }
        only_required = self.model.run_full_eol_mistrust_modeling(**base_kwargs)
        self.assertEqual(set(only_required.keys()), {"mistrust_scores", "feature_weight_summaries"})

        with_demo = self.model.run_full_eol_mistrust_modeling(
            **base_kwargs,
            demographics=artifacts["demographics"],
        )
        self.assertIn("race_gap_results", with_demo)

        with_treatment = self.model.run_full_eol_mistrust_modeling(
            **base_kwargs,
            eol_cohort=artifacts["eol_cohort"],
            treatment_totals=artifacts["treatment_totals"],
        )
        self.assertTrue({"race_treatment_results", "trust_treatment_results"}.issubset(with_treatment))

        with_acuity = self.model.run_full_eol_mistrust_modeling(
            **base_kwargs,
            acuity_scores=artifacts["acuity_scores"],
        )
        self.assertIn("acuity_correlations", with_acuity)

    def test_integration_data_alignment_preserves_shared_hadm_ids_across_artifacts(self):
        artifacts = self._build_core_artifacts()
        all_ids = set(artifacts["all_cohort"]["hadm_id"])
        self.assertEqual(set(artifacts["feature_matrix"]["hadm_id"]), all_ids)
        self.assertEqual(set(artifacts["note_labels"]["hadm_id"]), all_ids)
        self.assertEqual(set(artifacts["note_corpus"]["hadm_id"]), all_ids)
        self.assertEqual(set(artifacts["mistrust_scores"]["hadm_id"]), all_ids)
        self.assertEqual(set(artifacts["final_model_table"]["hadm_id"]), all_ids)
        self.assertTrue(set(artifacts["eol_cohort"]["hadm_id"]).issubset(all_ids))

    def test_integration_ordering_shuffle_invariance_produces_identical_outputs(self):
        original = self._build_core_artifacts()
        self.admissions = self.admissions.sample(frac=1.0, random_state=1).reset_index(drop=True)
        self.patients = self.patients.sample(frac=1.0, random_state=2).reset_index(drop=True)
        self.icustays = self.icustays.sample(frac=1.0, random_state=3).reset_index(drop=True)
        self.chartevents = self.chartevents.sample(frac=1.0, random_state=4).reset_index(drop=True)
        self.noteevents = self.noteevents.sample(frac=1.0, random_state=5).reset_index(drop=True)
        shuffled = self._build_core_artifacts()
        for key in ("base", "demographics", "all_cohort", "eol_cohort", "feature_matrix", "note_labels", "note_corpus", "treatment_totals", "acuity_scores", "mistrust_scores", "final_model_table"):
            pd.testing.assert_frame_equal(original[key], shuffled[key])

    def test_integration_controlled_nulls_only_reduce_affected_stage_rows(self):
        artifacts = self._build_core_artifacts()
        note_corpus = artifacts["note_corpus"].copy()
        note_corpus.loc[note_corpus["hadm_id"] == 101, "note_text"] = None
        scores = self.model.build_mistrust_score_table(
            artifacts["feature_matrix"],
            artifacts["note_labels"],
            note_corpus,
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
            sentiment_fn=self._sentiment_fn,
        )
        self.assertEqual(len(scores), 6)

        final_model_table = artifacts["final_model_table"].copy()
        final_model_table["left_ama"] = [0, 1, 0, 1, 0, 1]
        final_model_table.loc[0, "age"] = float("nan")
        results = self.model.evaluate_downstream_predictions(
            final_model_table,
            task_map={"Left AMA": "left_ama"},
            feature_configurations={"Baseline": self.model.BASELINE_FEATURE_COLUMNS},
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9]),
            split_fn=(lambda X, y, test_size, random_state: (X.iloc[:3], X.iloc[3:], pd.Series(y).iloc[:3], pd.Series(y).iloc[3:])),
            auc_fn=_AUCRecorder(0.5),
            repetitions=1,
        )
        self.assertEqual(int(results.loc[0, "n_rows"]), 5)

    def test_integration_duplicate_cardinality_violation_fails_at_join_boundary(self):
        artifacts = self._build_core_artifacts()
        duplicate_labels = pd.concat(
            [artifacts["note_labels"], artifacts["note_labels"].iloc[[0]]],
            ignore_index=True,
        )
        with self.assertRaises(Exception):
            self.model.build_mistrust_score_table(
                artifacts["feature_matrix"],
                duplicate_labels,
                artifacts["note_corpus"],
                estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
                sentiment_fn=self._sentiment_fn,
            )

    def test_integration_write_read_round_trip_artifacts_remain_consumable(self):
        deliverables = self._build_deliverable_artifacts()
        with tempfile.TemporaryDirectory() as tmpdir:
            self.dataset.write_minimal_deliverables(deliverables, tmpdir)
            final_model_table = pd.read_csv(Path(tmpdir) / "final_model_table.csv")
            mistrust_scores = pd.read_csv(Path(tmpdir) / "mistrust_scores.csv")
            acuity_scores = pd.read_csv(Path(tmpdir) / "acuity_scores.csv")

            final_model_table["left_ama"] = [0, 1, 0, 1, 0, 1]
            final_model_table["code_status_dnr_dni_cmo"] = [1, 0, 1, 0, 1, 0]
            final_model_table["in_hospital_mortality"] = [0, 1, 0, 1, 0, 1]

            downstream = self.model.evaluate_downstream_predictions(
                final_model_table,
                estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9]),
                split_fn=_SplitRecorder(),
                auc_fn=_AUCRecorder(0.55),
                repetitions=1,
            )
            acuity = self.model.run_acuity_control_analysis(mistrust_scores, acuity_scores)
            self.assertEqual(downstream.shape[0], 18)
            self.assertEqual(acuity.shape[1], 5)

    def test_integration_cross_component_dataset_outputs_feed_model_without_translation(self):
        artifacts = self._build_core_artifacts()
        race_gap = self.model.run_race_gap_analysis(artifacts["mistrust_scores"], artifacts["demographics"])
        acuity = self.model.run_acuity_control_analysis(artifacts["mistrust_scores"], artifacts["acuity_scores"])
        self.assertEqual(set(race_gap["metric"]), set(self.model.MISTRUST_SCORE_COLUMNS))
        self.assertTrue((acuity["n"] >= 2).all())

    def test_integration_configuration_variants_are_consumable_with_matching_feature_sets(self):
        artifacts = self._build_core_artifacts()
        base = artifacts["base"]
        demographics = artifacts["demographics"]
        all_cohort = artifacts["all_cohort"]
        mistrust_scores = artifacts["mistrust_scores"]

        tables = {
            "baseline": (
                self.dataset.build_final_model_table(
                    demographics, all_cohort, base, self.chartevents, self.d_items, mistrust_scores, include_race=False, include_mistrust=False
                ),
                {"Baseline": self.model.BASELINE_FEATURE_COLUMNS},
            ),
            "race": (
                self.dataset.build_final_model_table(
                    demographics, all_cohort, base, self.chartevents, self.d_items, mistrust_scores, include_race=True, include_mistrust=False
                ),
                {"Baseline + Race": self.model.BASELINE_FEATURE_COLUMNS + self.model.RACE_FEATURE_COLUMNS},
            ),
            "mistrust": (
                self.dataset.build_final_model_table(
                    demographics, all_cohort, base, self.chartevents, self.d_items, mistrust_scores, include_race=False, include_mistrust=True
                ),
                {"Baseline + ALL": self.model.BASELINE_FEATURE_COLUMNS + self.model.MISTRUST_SCORE_COLUMNS},
            ),
        }
        for table, configs in tables.values():
            table = table.copy()
            table["left_ama"] = [0, 1, 0, 1, 0, 1]
            results = self.model.evaluate_downstream_predictions(
                table,
                feature_configurations=configs,
                task_map={"Left AMA": "left_ama"},
                estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9]),
                split_fn=_SplitRecorder(),
                auc_fn=_AUCRecorder(0.66),
                repetitions=1,
            )
            self.assertEqual(results.shape[0], 1)

    def test_integration_broken_intermediate_artifact_propagates_clear_error(self):
        artifacts = self._build_core_artifacts()
        broken_feature_matrix = artifacts["feature_matrix"].drop(columns=["hadm_id"])
        with self.assertRaisesRegex(ValueError, "hadm_id"):
            self.model.run_full_eol_mistrust_modeling(
                feature_matrix=broken_feature_matrix,
                note_labels=artifacts["note_labels"],
                note_corpus=artifacts["note_corpus"],
                estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
                sentiment_fn=self._sentiment_fn,
                repetitions=1,
            )

    def test_integration_full_pipeline_is_reproducible_across_repeated_runs(self):
        artifacts = self._build_core_artifacts()
        final_model_table = artifacts["final_model_table"].copy()
        final_model_table["left_ama"] = [0, 1, 0, 1, 0, 1]
        final_model_table["code_status_dnr_dni_cmo"] = [1, 0, 1, 0, 1, 0]
        final_model_table["in_hospital_mortality"] = [0, 1, 0, 1, 0, 1]
        kwargs = {
            "feature_matrix": artifacts["feature_matrix"],
            "note_labels": artifacts["note_labels"],
            "note_corpus": artifacts["note_corpus"],
            "demographics": artifacts["demographics"],
            "eol_cohort": artifacts["eol_cohort"],
            "treatment_totals": artifacts["treatment_totals"],
            "acuity_scores": artifacts["acuity_scores"],
            "final_model_table": final_model_table,
            "estimator_factory": lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
            "sentiment_fn": self._sentiment_fn,
            "split_fn": _SplitRecorder(),
            "auc_fn": _AUCRecorder(0.61),
            "repetitions": 1,
        }
        first = self.model.run_full_eol_mistrust_modeling(**kwargs)
        second = self.model.run_full_eol_mistrust_modeling(**kwargs)
        for key in first:
            if isinstance(first[key], dict):
                for inner_key in first[key]:
                    if isinstance(first[key][inner_key], dict):
                        for leaf_key in first[key][inner_key]:
                            pd.testing.assert_frame_equal(
                                first[key][inner_key][leaf_key],
                                second[key][inner_key][leaf_key],
                            )
                    else:
                        pd.testing.assert_frame_equal(first[key][inner_key], second[key][inner_key])
            else:
                pd.testing.assert_frame_equal(first[key], second[key])

    def test_integration_extra_nonbreaking_columns_do_not_change_results(self):
        original = self._build_core_artifacts()
        self.admissions["unused_admissions_col"] = "x"
        self.patients["unused_patients_col"] = "y"
        self.icustays["unused_icu_col"] = "z"
        self.chartevents["unused_event_col"] = "q"
        self.noteevents["unused_note_col"] = "r"
        with_extra = self._build_core_artifacts()
        pd.testing.assert_frame_equal(
            original["demographics"],
            with_extra["demographics"],
        )
        self.assertTrue(set(original["all_cohort"].columns).issubset(with_extra["all_cohort"].columns))
        pd.testing.assert_frame_equal(
            original["all_cohort"][original["all_cohort"].columns],
            with_extra["all_cohort"][original["all_cohort"].columns],
        )
        pd.testing.assert_frame_equal(original["feature_matrix"], with_extra["feature_matrix"])
        pd.testing.assert_frame_equal(original["note_labels"], with_extra["note_labels"])
        pd.testing.assert_frame_equal(original["note_corpus"], with_extra["note_corpus"])
        pd.testing.assert_frame_equal(original["mistrust_scores"], with_extra["mistrust_scores"])
        self.assertTrue(set(original["base"].columns).issubset(with_extra["base"].columns))
        pd.testing.assert_frame_equal(
            original["base"][original["base"].columns],
            with_extra["base"][original["base"].columns],
        )
        pd.testing.assert_frame_equal(
            original["final_model_table"],
            with_extra["final_model_table"],
        )

    def test_integration_package_import_and_direct_load_modules_are_compatible(self):
        dataset_pkg = importlib.import_module("pyhealth.datasets.eol_mistrust")
        model_pkg = importlib.import_module("pyhealth.models.eol_mistrust")
        self.assertTrue(callable(dataset_pkg.build_final_model_table))
        self.assertTrue(callable(model_pkg.run_full_eol_mistrust_modeling))
        self.assertEqual(model_pkg.MISTRUST_SCORE_COLUMNS, self.model.MISTRUST_SCORE_COLUMNS)

    def test_example_run_task_demo_uses_stable_mkdtemp_cache_dir(self):
        example_module = _load_example_module()
        captured = {}

        class _FakeDataset:
            def __init__(self, *args, **kwargs):
                del args
                captured.update(kwargs)

            def stats(self):
                return None

            def set_task(self, task, num_workers=0):
                del task, num_workers
                return self

        with patch.object(example_module.tempfile, "mkdtemp", return_value="stable-cache-dir"), patch.object(
            example_module, "MIMIC3Dataset", _FakeDataset
        ):
            example_module.run_task_demo(Path("root"), Path("config"))

        self.assertEqual(captured["cache_dir"], "stable-cache-dir")

    def test_integration_minimal_boundary_scale_pipeline_runs_with_two_admissions(self):
        admissions = pd.DataFrame(
            [
                {"hadm_id": 1, "subject_id": 10, "admittime": "2100-01-01 00:00:00", "dischtime": "2100-01-02 00:00:00", "ethnicity": "WHITE", "insurance": "Medicare", "discharge_location": "HOME", "hospital_expire_flag": 0, "has_chartevents_data": 1},
                {"hadm_id": 2, "subject_id": 11, "admittime": "2100-01-02 00:00:00", "dischtime": "2100-01-03 00:00:00", "ethnicity": "BLACK/AFRICAN AMERICAN", "insurance": "Private", "discharge_location": "SNF", "hospital_expire_flag": 0, "has_chartevents_data": 1},
            ]
        )
        patients = pd.DataFrame(
            [
                {"subject_id": 10, "gender": "M", "dob": "2070-01-01 00:00:00"},
                {"subject_id": 11, "gender": "F", "dob": "2070-01-01 00:00:00"},
            ]
        )
        icustays = pd.DataFrame(
            [
                {"hadm_id": 1, "icustay_id": 1, "intime": "2100-01-01 00:00:00", "outtime": "2100-01-01 12:00:00"},
                {"hadm_id": 2, "icustay_id": 2, "intime": "2100-01-02 00:00:00", "outtime": "2100-01-02 12:00:00"},
            ]
        )
        d_items = pd.DataFrame([{"itemid": 1, "label": "Education Readiness", "dbsource": "carevue"}])
        chartevents = pd.DataFrame(
            [
                {"hadm_id": 1, "itemid": 1, "value": "No", "icustay_id": 1},
                {"hadm_id": 2, "itemid": 1, "value": "Yes", "icustay_id": 2},
            ]
        )
        noteevents = pd.DataFrame(
            [
                {"hadm_id": 1, "category": "Nursing", "text": "noncompliant", "iserror": None},
                {"hadm_id": 2, "category": "Nursing", "text": "autopsy", "iserror": None},
            ]
        )
        base = self.dataset.build_base_admissions(admissions, patients)
        all_cohort = self.dataset.build_all_cohort(base, icustays)
        feature_matrix = self.dataset.build_chartevent_feature_matrix(chartevents, d_items, all_hadm_ids=all_cohort["hadm_id"].tolist())
        note_labels = self.dataset.build_note_labels(noteevents, all_hadm_ids=all_cohort["hadm_id"].tolist())
        note_corpus = self.dataset.build_note_corpus(noteevents, all_hadm_ids=all_cohort["hadm_id"].tolist())
        scores = self.model.build_mistrust_score_table(
            feature_matrix,
            note_labels,
            note_corpus,
            estimator_factory=lambda: _FakeProbEstimator([0.2, 0.8]),
            sentiment_fn=self._sentiment_fn,
        )
        self.assertEqual(scores["hadm_id"].tolist(), [1, 2])

    def test_integration_outputs_are_consumable_by_simple_consumer_operations(self):
        artifacts = self._build_core_artifacts()
        final_model_table = artifacts["final_model_table"].copy()
        final_model_table["left_ama"] = [0, 1, 0, 1, 0, 1]
        final_model_table["code_status_dnr_dni_cmo"] = [1, 0, 1, 0, 1, 0]
        final_model_table["in_hospital_mortality"] = [0, 1, 0, 1, 0, 1]
        outputs = self.model.run_full_eol_mistrust_modeling(
            feature_matrix=artifacts["feature_matrix"],
            note_labels=artifacts["note_labels"],
            note_corpus=artifacts["note_corpus"],
            demographics=artifacts["demographics"],
            eol_cohort=artifacts["eol_cohort"],
            treatment_totals=artifacts["treatment_totals"],
            acuity_scores=artifacts["acuity_scores"],
            final_model_table=final_model_table,
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
            sentiment_fn=self._sentiment_fn,
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.7),
            repetitions=1,
        )
        grouped = outputs["downstream_auc_results"].groupby("task").size().to_dict()
        indexed = outputs["mistrust_scores"].set_index("hadm_id")
        trust_counts = outputs["trust_treatment_results"].groupby("treatment").size().to_dict()
        self.assertEqual(grouped, {task: 6 for task in self.model.DOWNSTREAM_TASK_MAP.keys()})
        self.assertEqual(indexed.index.tolist(), [101, 102, 103, 104, 105, 106])
        self.assertEqual(
            trust_counts,
            {"total_vent_min": 3, "total_vaso_min": 3},
        )

    def test_integration_resume_from_existing_artifact_directory_is_idempotent(self):
        deliverables = self._build_deliverable_artifacts()
        with tempfile.TemporaryDirectory() as tmpdir:
            self.dataset.write_minimal_deliverables(deliverables, tmpdir)
            first_contents = {
                path.name: path.read_text()
                for path in Path(tmpdir).glob("*.csv")
            }
            self.dataset.write_minimal_deliverables(deliverables, tmpdir)
            second_contents = {
                path.name: path.read_text()
                for path in Path(tmpdir).glob("*.csv")
            }
        self.assertEqual(first_contents, second_contents)

    def test_integration_write_side_effects_do_not_mutate_in_memory_artifacts(self):
        deliverables = self._build_deliverable_artifacts()
        before = {key: value.copy(deep=True) for key, value in deliverables.items()}
        with tempfile.TemporaryDirectory() as tmpdir:
            self.dataset.write_minimal_deliverables(deliverables, tmpdir)
        for key in deliverables:
            pd.testing.assert_frame_equal(deliverables[key], before[key])

    def test_integration_multi_output_results_are_internally_consistent(self):
        artifacts = self._build_core_artifacts()
        final_model_table = artifacts["final_model_table"].copy()
        final_model_table["left_ama"] = [0, 1, 0, 1, 0, 1]
        final_model_table["code_status_dnr_dni_cmo"] = [1, 0, 1, 0, 1, 0]
        final_model_table["in_hospital_mortality"] = [0, 1, 0, 1, 0, 1]
        outputs = self.model.run_full_eol_mistrust_modeling(
            feature_matrix=artifacts["feature_matrix"],
            note_labels=artifacts["note_labels"],
            note_corpus=artifacts["note_corpus"],
            demographics=artifacts["demographics"],
            eol_cohort=artifacts["eol_cohort"],
            treatment_totals=artifacts["treatment_totals"],
            acuity_scores=artifacts["acuity_scores"],
            final_model_table=final_model_table,
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9, 0.3, 0.7, 0.4, 0.6]),
            sentiment_fn=self._sentiment_fn,
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.7),
            repetitions=1,
        )
        self.assertEqual(set(outputs["race_gap_results"]["metric"]), set(self.model.MISTRUST_SCORE_COLUMNS))
        self.assertEqual(set(outputs["trust_treatment_results"]["metric"]), set(self.model.MISTRUST_SCORE_COLUMNS))
        self.assertEqual(set(outputs["downstream_auc_results"]["configuration"]), set(self.model.DOWNSTREAM_FEATURE_CONFIGS.keys()))
        self.assertEqual(set(outputs["feature_weight_summaries"].keys()), {"noncompliance", "autopsy"})

    def test_integration_fixed_golden_workflow_matches_expected_snapshot(self):
        artifacts = self._build_core_artifacts()
        final_model_table = artifacts["final_model_table"].copy()
        final_model_table["left_ama"] = [0, 1, 0, 1, 0, 1]
        final_model_table["code_status_dnr_dni_cmo"] = [1, 0, 1, 0, 1, 0]
        final_model_table["in_hospital_mortality"] = [0, 1, 0, 1, 0, 1]
        downstream = self.model.evaluate_downstream_predictions(
            final_model_table,
            estimator_factory=lambda: _FakeProbEstimator([0.1, 0.9]),
            split_fn=_SplitRecorder(),
            auc_fn=_AUCRecorder(0.7),
            repetitions=1,
        )
        snapshot = {
            "all_hadm_ids": artifacts["all_cohort"]["hadm_id"].tolist(),
            "eol_hadm_ids": artifacts["eol_cohort"]["hadm_id"].tolist(),
            "mistrust_first_row": {
                "hadm_id": int(artifacts["mistrust_scores"].iloc[0]["hadm_id"]),
                "columns": artifacts["mistrust_scores"].columns.tolist(),
            },
            "downstream_rows": int(len(downstream)),
            "downstream_first": {
                "task": downstream.iloc[0]["task"],
                "configuration": downstream.iloc[0]["configuration"],
                "auc_mean": round(float(downstream.iloc[0]["auc_mean"]), 6),
            },
        }
        self.assertEqual(
            snapshot,
            {
                "all_hadm_ids": [101, 102, 103, 104, 105, 106],
                "eol_hadm_ids": [103, 104],
                "mistrust_first_row": {
                    "hadm_id": 101,
                    "columns": [
                        "hadm_id",
                        "noncompliance_score_z",
                        "autopsy_score_z",
                        "negative_sentiment_score_z",
                    ],
                },
                "downstream_rows": 18,
                "downstream_first": {
                    "task": "Left AMA",
                    "configuration": "Baseline",
                    "auc_mean": 0.7,
                },
            },
        )


if __name__ == "__main__":
    unittest.main()

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def _load_eol_mistrust_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "pyhealth" / "datasets" / "eol_mistrust.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pyhealth.datasets.eol_mistrust_module_tests",
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

    def fit(self, X, y):
        self.was_fit = True
        self.fit_X = X.copy() if hasattr(X, "copy") else X
        self.fit_y = y.copy() if hasattr(y, "copy") else y
        return self

    def predict_proba(self, X):
        probs = self.probabilities[: len(X)]
        return [[1.0 - prob, prob] for prob in probs]


class TestEOLMistrustModuleImplementation(unittest.TestCase):
    """Module-facing tests for the EOL mistrust implementation."""

    @classmethod
    def setUpClass(cls):
        cls.module = _load_eol_mistrust_module()

    def setUp(self):
        self.all_hadm_ids = [301, 302, 303, 304, 305, 306]
        self.admissions = pd.DataFrame(
            [
                {
                    "hadm_id": 301,
                    "subject_id": 1,
                    "admittime": "2100-01-01 00:00:00",
                    "dischtime": "2100-01-02 00:00:00",
                    "ethnicity": "WHITE",
                    "insurance": "Medicare",
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 302,
                    "subject_id": 2,
                    "admittime": "2100-02-01 00:00:00",
                    "dischtime": "2100-02-02 12:00:00",
                    "ethnicity": "BLACK/AFRICAN AMERICAN",
                    "insurance": "Private",
                    "discharge_location": "HOME HOSPICE",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 303,
                    "subject_id": 3,
                    "admittime": "2100-03-01 00:00:00",
                    "dischtime": "2100-03-01 20:00:00",
                    "ethnicity": "ASIAN - CHINESE",
                    "insurance": "Medicaid",
                    "discharge_location": "SKILLED NURSING FACILITY",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 304,
                    "subject_id": 4,
                    "admittime": "2100-04-01 00:00:00",
                    "dischtime": "2100-04-01 10:00:00",
                    "ethnicity": "WHITE - RUSSIAN",
                    "insurance": "Government",
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 1,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 305,
                    "subject_id": 5,
                    "admittime": "2100-05-01 00:00:00",
                    "dischtime": "2100-05-02 06:00:00",
                    "ethnicity": "HISPANIC OR LATINO",
                    "insurance": "Self Pay",
                    "discharge_location": "LEFT AGAINST MEDICAL ADVICE",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 306,
                    "subject_id": 6,
                    "admittime": "2100-06-01 00:00:00",
                    "dischtime": "2100-06-02 00:00:00",
                    "ethnicity": "WHITE",
                    "insurance": "Private",
                    "discharge_location": "TRANSFER AGAINST MEDICAL ADVICE REVIEW",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 307,
                    "subject_id": 7,
                    "admittime": "2100-07-01 00:00:00",
                    "dischtime": "2100-07-02 00:00:00",
                    "ethnicity": "BLACK/CAPE VERDEAN",
                    "insurance": "Medicare",
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 0,
                },
            ]
        )
        self.patients = pd.DataFrame(
            [
                {"subject_id": 1, "gender": "M", "dob": "2070-01-01 00:00:00"},
                {"subject_id": 2, "gender": "F", "dob": "2068-02-01 00:00:00"},
                {"subject_id": 3, "gender": "M", "dob": "2072-03-01 00:00:00"},
                {"subject_id": 4, "gender": "F", "dob": "2050-04-01 00:00:00"},
                {"subject_id": 5, "gender": "M", "dob": "2076-05-01 00:00:00"},
                {"subject_id": 6, "gender": "F", "dob": "2065-06-01 00:00:00"},
                {"subject_id": 7, "gender": "M", "dob": "2060-07-01 00:00:00"},
            ]
        )
        self.icustays = pd.DataFrame(
            [
                {
                    "hadm_id": 301,
                    "icustay_id": 3011,
                    "intime": "2100-01-01 00:00:00",
                    "outtime": "2100-01-01 11:00:00",
                },
                {
                    "hadm_id": 301,
                    "icustay_id": 3012,
                    "intime": "2100-01-01 12:00:00",
                    "outtime": "2100-01-01 23:00:00",
                },
                {
                    "hadm_id": 302,
                    "icustay_id": 3021,
                    "intime": "2100-02-01 00:00:00",
                    "outtime": "2100-02-01 13:00:00",
                },
                {
                    "hadm_id": 302,
                    "icustay_id": 3022,
                    "intime": "2100-02-02 00:00:00",
                    "outtime": "2100-02-02 08:00:00",
                },
                {
                    "hadm_id": 303,
                    "icustay_id": 3031,
                    "intime": "2100-03-01 00:00:00",
                    "outtime": "2100-03-01 12:00:00",
                },
                {
                    "hadm_id": 304,
                    "icustay_id": 3041,
                    "intime": "2100-04-01 00:00:00",
                    "outtime": "2100-04-01 14:00:00",
                },
                {
                    "hadm_id": 305,
                    "icustay_id": 3051,
                    "intime": "2100-05-01 00:00:00",
                    "outtime": "2100-05-01 15:00:00",
                },
                {
                    "hadm_id": 306,
                    "icustay_id": 3061,
                    "intime": "2100-06-01 00:00:00",
                    "outtime": "2100-06-01 16:00:00",
                },
                {
                    "hadm_id": 307,
                    "icustay_id": 3071,
                    "intime": "2100-07-01 00:00:00",
                    "outtime": "2100-07-01 18:00:00",
                },
            ]
        )
        self.noteevents = pd.DataFrame(
            [
                {
                    "hadm_id": 302,
                    "category": "Nursing",
                    "text": "Patient was NONCOMPLIANT with care plan. Family provided AUTOPSY consent.",
                    "iserror": 0,
                },
                {
                    "hadm_id": 303,
                    "category": "Physician",
                    "text": "Patient remained non-adher with follow up after counseling.",
                    "iserror": None,
                },
                {
                    "hadm_id": 304,
                    "category": "Nursing",
                    "text": "Patient refuses medication.",
                    "iserror": 0,
                },
                {
                    "hadm_id": 304,
                    "category": "Discharge",
                    "text": "Autopsy requested.",
                    "iserror": 1,
                },
                {
                    "hadm_id": 305,
                    "category": "Nursing",
                    "text": "Patient refused treatment. Date:[**5-1-18**]",
                    "iserror": 0,
                },
            ]
        )
        self.d_items = pd.DataFrame(
            [
                {"itemid": 10, "label": "Riker-SAS Scale Score", "dbsource": "carevue"},
                {"itemid": 11, "label": "Richmond-RAS Scale", "dbsource": "metavision"},
                {"itemid": 12, "label": "Pain Level", "dbsource": "metavision"},
                {"itemid": 13, "label": "Family Meeting Note", "dbsource": "carevue"},
                {"itemid": 14, "label": "Education Readiness Status", "dbsource": "carevue"},
                {"itemid": 128, "label": "Code Status", "dbsource": "carevue"},
                {"itemid": 223758, "label": "Code Status", "dbsource": "metavision"},
                {"itemid": 999, "label": "Code Status", "dbsource": "carevue"},
                {"itemid": 777, "label": "Unrelated Measure", "dbsource": "carevue"},
            ]
        )
        self.chartevents = pd.DataFrame(
            [
                {"hadm_id": 302, "itemid": 10, "value": "Agitated", "icustay_id": 3021},
                {"hadm_id": 302, "itemid": 14, "value": "No", "icustay_id": 3021},
                {"hadm_id": 302, "itemid": 128, "value": "DNR/DNI", "icustay_id": 3021},
                {"hadm_id": 303, "itemid": 11, "value": "0 Alert and Calm", "icustay_id": 3031},
                {"hadm_id": 303, "itemid": 13, "value": "Family Requested", "icustay_id": 3031},
                {
                    "hadm_id": 303,
                    "itemid": 223758,
                    "value": "Comfort Measures Only",
                    "icustay_id": 3031,
                },
                {"hadm_id": 304, "itemid": 12, "value": "7-Mod to Severe", "icustay_id": 3041},
                {"hadm_id": 304, "itemid": 999, "value": "DNR", "icustay_id": 3041},
                {"hadm_id": 305, "itemid": 128, "value": "Full Code", "icustay_id": 3051},
                {"hadm_id": 306, "itemid": 777, "value": "Noise", "icustay_id": 3061},
            ]
        )
        self.ventdurations = pd.DataFrame(
            [
                {
                    "icustay_id": 3021,
                    "ventnum": 1,
                    "starttime": "2100-02-01 00:00:00",
                    "endtime": "2100-02-01 02:00:00",
                    "duration_hours": 2.0,
                },
                {
                    "icustay_id": 3021,
                    "ventnum": 2,
                    "starttime": "2100-02-01 11:30:00",
                    "endtime": "2100-02-01 12:30:00",
                    "duration_hours": 1.0,
                },
                {
                    "icustay_id": 3021,
                    "ventnum": 3,
                    "starttime": "2100-02-01 23:31:00",
                    "endtime": "2100-02-02 00:31:00",
                    "duration_hours": 1.0,
                },
            ]
        )
        self.vasopressordurations = pd.DataFrame(
            [
                {
                    "icustay_id": 3031,
                    "vasonum": 1,
                    "starttime": "2100-03-01 01:00:00",
                    "endtime": "2100-03-01 03:00:00",
                    "duration_hours": 2.0,
                },
                {
                    "icustay_id": 3031,
                    "vasonum": 2,
                    "starttime": "2100-03-01 02:30:00",
                    "endtime": "2100-03-01 05:00:00",
                    "duration_hours": 2.5,
                },
                {
                    "icustay_id": 3031,
                    "vasonum": 3,
                    "starttime": "2100-03-01 14:00:00",
                    "endtime": "2100-03-01 15:00:00",
                    "duration_hours": 1.0,
                },
            ]
        )
        self.oasis = pd.DataFrame(
            [
                {"hadm_id": 302, "icustay_id": 3021, "oasis": 12},
                {"hadm_id": 302, "icustay_id": 3022, "oasis": 25},
                {"hadm_id": 303, "icustay_id": 3031, "oasis": 18},
                {"hadm_id": 304, "icustay_id": 3041, "oasis": 30},
                {"hadm_id": 305, "icustay_id": 3051, "oasis": 9},
                {"hadm_id": 306, "icustay_id": 3061, "oasis": 7},
            ]
        )
        self.sapsii = pd.DataFrame(
            [
                {"hadm_id": 302, "icustay_id": 3021, "sapsii": 40},
                {"hadm_id": 302, "icustay_id": 3022, "sapsii": 60},
                {"hadm_id": 303, "icustay_id": 3031, "sapsii": 35},
                {"hadm_id": 304, "icustay_id": 3041, "sapsii": 70},
                {"hadm_id": 305, "icustay_id": 3051, "sapsii": 15},
                {"hadm_id": 306, "icustay_id": 3061, "sapsii": 12},
            ]
        )

    def _pending_real_data(self, requirement: str) -> None:
        self.skipTest(requirement)

    def _get_callable(self, name):
        self.assertTrue(
            hasattr(self.module, name),
            msg=f"Implement `{name}` in pyhealth.datasets.eol_mistrust",
        )
        attr = getattr(self.module, name)
        self.assertTrue(callable(attr), msg=f"`{name}` must be callable")
        return attr

    def _build_base(self):
        return self._get_callable("build_base_admissions")(self.admissions, self.patients)

    def _build_demographics(self):
        return self._get_callable("build_demographics_table")(self._build_base())

    def _build_all(self):
        return self._get_callable("build_all_cohort")(self._build_base(), self.icustays)

    def _build_eol(self):
        return self._get_callable("build_eol_cohort")(self._build_base(), self._build_demographics())

    def _build_feature_matrix(self):
        build_chartevent_feature_matrix = self._get_callable("build_chartevent_feature_matrix")
        return build_chartevent_feature_matrix(
            self.chartevents,
            self.d_items,
            allowed_labels={
                "Riker-SAS Scale Score",
                "Richmond-RAS Scale",
                "Pain Level",
                "Family Meeting Note",
                "Education Readiness Status",
            },
            all_hadm_ids=self.all_hadm_ids,
        )

    def _build_note_labels(self):
        return self._get_callable("build_note_labels")(
            self.noteevents,
            all_hadm_ids=self.all_hadm_ids,
        )

    def _build_note_corpus(self):
        return self._get_callable("build_note_corpus")(
            self.noteevents,
            all_hadm_ids=self.all_hadm_ids,
        )

    def _zero_mistrust_scores(self, hadm_ids=None):
        if hadm_ids is None:
            hadm_ids = self.all_hadm_ids
        return pd.DataFrame(
            [
                {
                    "hadm_id": hadm_id,
                    "noncompliance_score_z": 0.0,
                    "autopsy_score_z": 0.0,
                    "negative_sentiment_score_z": 0.0,
                }
                for hadm_id in hadm_ids
            ]
        )

    def _required_downstream_feature_configs(self):
        baseline_features = [
            "age",
            "los_days",
            "gender_f",
            "gender_m",
            "insurance_private",
            "insurance_public",
            "insurance_self_pay",
        ]
        race_features = [
            "race_white",
            "race_black",
            "race_asian",
            "race_hispanic",
            "race_native_american",
            "race_other",
        ]
        mistrust_features = {
            "Baseline + Noncompliant": ["noncompliance_score_z"],
            "Baseline + Autopsy": ["autopsy_score_z"],
            "Baseline + Neg-Sentiment": ["negative_sentiment_score_z"],
        }
        return {
            "Baseline": baseline_features,
            "Baseline + Race": baseline_features + race_features,
            "Baseline + Noncompliant": baseline_features + mistrust_features["Baseline + Noncompliant"],
            "Baseline + Autopsy": baseline_features + mistrust_features["Baseline + Autopsy"],
            "Baseline + Neg-Sentiment": baseline_features + mistrust_features["Baseline + Neg-Sentiment"],
            "Baseline + ALL": baseline_features
            + race_features
            + [
                "noncompliance_score_z",
                "autopsy_score_z",
                "negative_sentiment_score_z",
            ],
        }

    def _build_mistrust_scores(self):
        build_mistrust_score_table = self._get_callable("build_mistrust_score_table")
        probability_sequences = [
            [0.05, 0.90, 0.10, 0.80, 0.20, 0.40],
            [0.15, 0.70, 0.20, 0.30, 0.60, 0.50],
        ]
        created = []

        def estimator_factory():
            estimator = _FakeProbEstimator(probability_sequences[len(created)])
            created.append(estimator)
            return estimator

        sentiment_map = {
            "Patient was NONCOMPLIANT with care plan. Family provided AUTOPSY consent.": -0.6,
            "Patient remained non-adher with follow up after counseling.": -0.2,
            "Patient refuses medication.": 0.1,
            "Patient refused treatment. Date:[**5-1-18**]": -0.4,
            "": 0.0,
        }

        scores = build_mistrust_score_table(
            feature_matrix=self._build_feature_matrix(),
            note_labels=self._build_note_labels(),
            note_corpus=self._build_note_corpus(),
            estimator_factory=estimator_factory,
            sentiment_fn=lambda text: (sentiment_map[text], 0.0),
        )
        return scores, created

    def _assert_hadm_unique(self, df, message):
        self.assertIn("hadm_id", df.columns, msg=f"{message} must include hadm_id")
        self.assertTrue(df["hadm_id"].is_unique, msg=f"{message} must be unique on hadm_id")

    def test_all_cohort_contains_distinct_hadm_ids_with_any_icu_stay(self):
        all_cohort = self._build_all()
        self.assertEqual(set(all_cohort["hadm_id"]), set(self.all_hadm_ids))
        self.assertNotIn(307, set(all_cohort["hadm_id"]))
        self._assert_hadm_unique(all_cohort, "ALL cohort")

    def test_all_cohort_size_is_within_expected_mimic_range(self):
        self._pending_real_data(
            "ALL cohort size on real MIMIC-III data should be within 46,000-50,000 admissions."
        )

    def test_eol_cohort_applies_los_and_discharge_criteria(self):
        eol = self._build_eol()
        self.assertEqual(set(eol["hadm_id"]), {302})
        by_hadm = eol.set_index("hadm_id")
        self.assertEqual(by_hadm.loc[302, "discharge_category"], "Hospice")
        self._assert_hadm_unique(eol, "EOL cohort")

    def test_eol_cohort_requires_stay_longer_than_twenty_four_hours(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        build_demographics_table = self._get_callable("build_demographics_table")
        build_eol_cohort = self._get_callable("build_eol_cohort")
        admissions = pd.DataFrame(
            [
                {
                    "hadm_id": 920,
                    "subject_id": 920,
                    "admittime": "2100-09-01 00:00:00",
                    "dischtime": "2100-09-02 00:00:00",
                    "ethnicity": "WHITE",
                    "insurance": "Medicare",
                    "discharge_location": "HOME HOSPICE",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 921,
                    "subject_id": 921,
                    "admittime": "2100-09-01 00:00:00",
                    "dischtime": "2100-09-02 00:01:00",
                    "ethnicity": "BLACK/AFRICAN AMERICAN",
                    "insurance": "Private",
                    "discharge_location": "HOME HOSPICE",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
            ]
        )
        patients = pd.DataFrame(
            [
                {"subject_id": 920, "gender": "M", "dob": "2070-09-01 00:00:00"},
                {"subject_id": 921, "gender": "F", "dob": "2070-09-01 00:00:00"},
            ]
        )

        base = build_base_admissions(admissions, patients)
        demographics = build_demographics_table(base)
        eol = build_eol_cohort(base, demographics)
        self.assertNotIn(920, set(eol["hadm_id"]))
        self.assertIn(921, set(eol["hadm_id"]))

    def test_eol_cohort_size_is_within_expected_mimic_range(self):
        self._pending_real_data(
            "EOL cohort size on real MIMIC-III data should remain near the expected reference scale of roughly 11,000 admissions."
        )

    def test_demographics_age_caps_shifted_mimiciii_ages_at_ninety(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        build_demographics_table = self._get_callable("build_demographics_table")
        admissions = pd.DataFrame(
            [
                {
                    "hadm_id": 901,
                    "subject_id": 91,
                    "admittime": "2100-01-01 00:00:00",
                    "dischtime": "2100-01-02 00:00:00",
                    "ethnicity": "WHITE",
                    "insurance": "Medicare",
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                }
            ]
        )
        patients = pd.DataFrame(
            [
                {"subject_id": 91, "gender": "M", "dob": "1800-01-01 00:00:00"},
            ]
        )

        base = build_base_admissions(admissions, patients)
        demographics = build_demographics_table(base).set_index("hadm_id")
        self.assertEqual(demographics.loc[901, "age"], 90.0)

    def test_mistrust_scores_trained_on_all_can_merge_into_eol_by_hadm_id(self):
        eol = self._build_eol()
        scores, created = self._build_mistrust_scores()
        merged = eol[["hadm_id"]].merge(scores, on="hadm_id", how="left")
        self.assertEqual(len(created), 2)
        self.assertEqual(set(merged["hadm_id"]), set(eol["hadm_id"]))
        self.assertTrue(
            merged[
                [
                    "noncompliance_score_z",
                    "autopsy_score_z",
                    "negative_sentiment_score_z",
                ]
            ]
            .notna()
            .all()
            .all()
        )

    def test_mistrust_score_merge_leaves_null_for_eol_admissions_absent_from_all(self):
        scores, _ = self._build_mistrust_scores()
        eol_like = pd.DataFrame({"hadm_id": [302, 303, 999]})
        merged = eol_like.merge(scores, on="hadm_id", how="left").set_index("hadm_id")
        self.assertTrue(
            merged.loc[
                999,
                [
                    "noncompliance_score_z",
                    "autopsy_score_z",
                    "negative_sentiment_score_z",
                ],
            ]
            .isna()
            .all()
        )

    def test_chartevent_feature_matrix_rows_match_all_cohort(self):
        feature_matrix = self._build_feature_matrix()
        self.assertEqual(set(feature_matrix["hadm_id"]), set(self.all_hadm_ids))
        self._assert_hadm_unique(feature_matrix, "Chartevent feature matrix")

    def test_chartevent_feature_matrix_keeps_zero_rows_for_no_matching_events(self):
        feature_matrix = self._build_feature_matrix().fillna(0).set_index("hadm_id")
        zero_row = feature_matrix.loc[306]
        self.assertTrue((zero_row == 0).all())

    def test_chartevent_feature_matrix_feature_count_is_within_expected_range(self):
        self._pending_real_data(
            "Real-data chartevent feature dimensionality should be within 550-700 columns."
        )

    def test_real_data_chartevent_feature_matrix_cells_are_binary(self):
        self._pending_real_data(
            "On real data, every non-hadm_id cell in the chartevent feature matrix should be binary 0/1."
        )

    def test_chartevent_feature_matrix_is_binary_and_keeps_rare_features(self):
        feature_matrix = self._build_feature_matrix().fillna(0)
        feature_columns = [column for column in feature_matrix.columns if column != "hadm_id"]
        self.assertTrue(feature_columns)
        self.assertTrue(
            feature_matrix[feature_columns].isin([0, 1]).all().all(),
            msg="All chartevent features must be binary.",
        )
        self.assertIn("Family Meeting Note: Family Requested", feature_matrix.columns)
        self.assertEqual(
            int(feature_matrix["Family Meeting Note: Family Requested"].sum()),
            1,
            msg="Rare one-off chart features must be preserved.",
        )

    def test_chartevent_feature_columns_use_label_colon_value_names(self):
        feature_matrix = self._build_feature_matrix()
        expected_columns = {
            "Riker-SAS Scale Score: Agitated",
            "Education Readiness Status: No",
            "Richmond-RAS Scale: 0 Alert and Calm",
            "Pain Level: 7-Mod to Severe",
        }
        self.assertTrue(
            expected_columns.issubset(set(feature_matrix.columns)),
            msg='Feature columns must be named in the form "label: value".',
        )

    def test_chartevent_feature_matrix_counts_repeated_label_value_once_per_admission(self):
        build_chartevent_feature_matrix = self._get_callable("build_chartevent_feature_matrix")
        chartevents = pd.concat(
            [
                self.chartevents,
                pd.DataFrame(
                    [
                        {
                            "hadm_id": 302,
                            "itemid": 10,
                            "value": "Agitated",
                            "icustay_id": 3021,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        feature_matrix = build_chartevent_feature_matrix(
            chartevents,
            self.d_items,
            allowed_labels={
                "Riker-SAS Scale Score",
                "Richmond-RAS Scale",
                "Pain Level",
                "Family Meeting Note",
                "Education Readiness Status",
            },
            all_hadm_ids=self.all_hadm_ids,
        ).set_index("hadm_id")
        self.assertIn("Riker-SAS Scale Score: Agitated", feature_matrix.columns)
        self.assertEqual(feature_matrix.loc[302, "Riker-SAS Scale Score: Agitated"], 1)

    def test_chartevent_feature_matrix_excludes_unmatched_itemids(self):
        feature_matrix = self._build_feature_matrix()
        joined_columns = " | ".join(feature_matrix.columns)
        self.assertNotIn("Unrelated Measure", joined_columns)

    def test_table2_item_matching_is_case_insensitive_partial_and_cross_dbsource(self):
        identify_table2_itemids = self._get_callable("identify_table2_itemids")
        matched = identify_table2_itemids(self.d_items)
        self.assertTrue({10, 11, 12, 13, 14}.issubset(matched))

    def test_note_aggregation_keeps_non_error_notes_and_concatenates_per_admission(self):
        note_corpus = self._build_note_corpus()
        self._assert_hadm_unique(note_corpus, "Note corpus")
        by_hadm = note_corpus.set_index("hadm_id")
        self.assertIn("AUTOPSY consent.", by_hadm.loc[302, "note_text"])
        self.assertIn("non-adher", by_hadm.loc[303, "note_text"])
        self.assertNotIn("Autopsy requested.", by_hadm.loc[304, "note_text"])
        self.assertEqual(by_hadm.loc[306, "note_text"], "")

    def test_note_aggregation_restricts_to_all_cohort_hadm_ids(self):
        build_note_corpus = self._get_callable("build_note_corpus")
        notes = pd.concat(
            [
                self.noteevents,
                pd.DataFrame(
                    [
                        {
                            "hadm_id": 999,
                            "category": "Nursing",
                            "text": "Out-of-cohort note should not survive the ALL cohort filter.",
                            "iserror": 0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        note_corpus = build_note_corpus(notes, all_hadm_ids=self.all_hadm_ids)
        self.assertEqual(set(note_corpus["hadm_id"]), set(self.all_hadm_ids))
        self.assertNotIn(999, set(note_corpus["hadm_id"]))

    def test_note_aggregation_joins_multiple_notes_with_single_space_separator(self):
        build_note_corpus = self._get_callable("build_note_corpus")
        notes = pd.DataFrame(
            [
                {"hadm_id": 1, "category": "Nursing", "text": "First\t note", "iserror": 0},
                {"hadm_id": 1, "category": "Physician", "text": "Second\nnote", "iserror": 0},
            ]
        )
        corpus = build_note_corpus(notes).set_index("hadm_id")
        self.assertEqual(corpus.loc[1, "note_text"], "First note Second note")

    def test_note_coverage_exceeds_40000_admissions(self):
        self._pending_real_data(
            "Real-data note coverage should exceed 40,000 admissions in the ALL cohort."
        )

    def test_total_note_count_exceeds_expected_reference_scale(self):
        self._pending_real_data(
            "The raw clinical note corpus used for note aggregation should contain at least about 800,000 notes on real MIMIC-III data."
        )

    def test_noncompliance_label_matches_only_noncompliant_case_insensitively(self):
        build_note_labels = self._get_callable("build_note_labels")
        phrases = [
            "noncompliant",
        ]
        notes = pd.DataFrame(
            [
                {
                    "hadm_id": index + 1,
                    "category": "Nursing",
                    "text": f"Patient documented as {phrase.upper()} during stay.",
                    "iserror": 0,
                }
                for index, phrase in enumerate(phrases)
            ]
        )
        labels = build_note_labels(notes).set_index("hadm_id")
        for hadm_id in range(1, len(phrases) + 1):
            self.assertEqual(labels.loc[hadm_id, "noncompliance_label"], 1)

    def test_noncompliance_label_does_not_fire_on_hyphenated_refusal_or_noncompliance_variants(self):
        build_note_labels = self._get_callable("build_note_labels")
        phrases = [
            "non-complian",
            "non-adher",
            "refuses medication",
            "refused treatment",
            "noncompliance",
        ]
        notes = pd.DataFrame(
            [
                {
                    "hadm_id": index + 1,
                    "category": "Nursing",
                    "text": f"Patient documented as {phrase.upper()} during stay.",
                    "iserror": 0,
                }
                for index, phrase in enumerate(phrases)
            ]
        )
        labels = build_note_labels(notes).set_index("hadm_id")
        for hadm_id in range(1, len(phrases) + 1):
            self.assertEqual(labels.loc[hadm_id, "noncompliance_label"], 0)

    def test_noncompliance_positive_rate_is_within_expected_range(self):
        self._pending_real_data(
            "Noncompliance label prevalence on real data should be between 1% and 30%."
        )

    def test_autopsy_label_distinguishes_consent_decline_and_ambiguous_mentions(self):
        build_note_labels = self._get_callable("build_note_labels")
        notes = pd.DataFrame(
            [
                {
                    "hadm_id": 1,
                    "category": "Nursing",
                    "text": "AUTOPSY consent obtained and autopsy was performed.",
                    "iserror": 0,
                },
                {
                    "hadm_id": 2,
                    "category": "Nursing",
                    "text": "Autopsy declined by family. No autopsy will be performed.",
                    "iserror": 0,
                },
                {
                    "hadm_id": 3,
                    "category": "Nursing",
                    "text": "Autopsy was discussed with the family.",
                    "iserror": 0,
                },
            ]
        )
        labels = build_note_labels(notes).set_index("hadm_id")
        self.assertEqual(labels.loc[1, "autopsy_label"], 1)
        self.assertEqual(labels.loc[2, "autopsy_label"], 0)
        self.assertEqual(labels.loc[3, "autopsy_label"], 0)

    def test_autopsy_positive_rate_is_within_expected_range(self):
        self._pending_real_data(
            "Autopsy label prevalence on real data should be between 10% and 50%."
        )

    def test_black_autopsy_rate_exceeds_white_autopsy_rate(self):
        self._pending_real_data(
            "Black admission autopsy rate should exceed White admission autopsy rate."
        )

    def test_sentiment_preprocessing_uses_whitespace_tokenize_then_rejoin(self):
        prepare_note_text_for_sentiment = self._get_callable("prepare_note_text_for_sentiment")
        cleaned = prepare_note_text_for_sentiment(
            "Patient\trefused\n\n treatment   Date:[**5-1-18**]"
        )
        self.assertEqual(cleaned, "Patient refused treatment Date:[**5-1-18**]")

    def test_noncompliance_proxy_model_uses_l1_liblinear_logistic_regression(self):
        build_proxy_probability_scores = self._get_callable("build_proxy_probability_scores")
        created = []

        class _RecordingLogisticRegression:
            def __init__(self, *args, **kwargs):
                created.append(kwargs)

            def fit(self, X, y):
                del X, y
                return self

            def predict_proba(self, X):
                return [[0.25, 0.75] for _ in range(len(X))]

        feature_matrix = pd.DataFrame(
            [{"hadm_id": 1, "feature_a": 1}, {"hadm_id": 2, "feature_a": 0}]
        )
        labels = pd.DataFrame(
            [{"hadm_id": 1, "noncompliance_label": 1}, {"hadm_id": 2, "noncompliance_label": 0}]
        )

        with patch.object(self.module, "LogisticRegression", _RecordingLogisticRegression):
            build_proxy_probability_scores(feature_matrix, labels, "noncompliance_label")

        self.assertEqual(created[0].get("penalty"), "l1")
        self.assertEqual(created[0].get("C"), 0.1)
        self.assertEqual(created[0].get("solver"), "liblinear")
        self.assertEqual(created[0].get("max_iter"), 1000)

    def test_autopsy_proxy_model_uses_l1_liblinear_logistic_regression(self):
        build_proxy_probability_scores = self._get_callable("build_proxy_probability_scores")
        created = []

        class _RecordingLogisticRegression:
            def __init__(self, *args, **kwargs):
                created.append(kwargs)

            def fit(self, X, y):
                del X, y
                return self

            def predict_proba(self, X):
                return [[0.40, 0.60] for _ in range(len(X))]

        feature_matrix = pd.DataFrame(
            [{"hadm_id": 1, "feature_a": 1}, {"hadm_id": 2, "feature_a": 0}]
        )
        labels = pd.DataFrame(
            [{"hadm_id": 1, "autopsy_label": 1}, {"hadm_id": 2, "autopsy_label": 0}]
        )

        with patch.object(self.module, "LogisticRegression", _RecordingLogisticRegression):
            build_proxy_probability_scores(feature_matrix, labels, "autopsy_label")

        self.assertEqual(created[0].get("penalty"), "l1")
        self.assertEqual(created[0].get("C"), 0.1)
        self.assertEqual(created[0].get("solver"), "liblinear")
        self.assertEqual(created[0].get("max_iter"), 1000)

    def test_proxy_models_fit_on_full_all_cohort_without_train_test_split(self):
        build_proxy_probability_scores = self._get_callable("build_proxy_probability_scores")
        estimator = _FakeProbEstimator([0.9, 0.2, 0.8, 0.1, 0.4, 0.3])
        feature_matrix = self._build_feature_matrix()
        labels = self._build_note_labels()
        scores = build_proxy_probability_scores(
            feature_matrix,
            labels,
            "noncompliance_label",
            estimator_factory=lambda: estimator,
        )
        self.assertTrue(estimator.was_fit)
        self.assertEqual(len(estimator.fit_X), len(feature_matrix))
        self.assertEqual(set(scores["hadm_id"]), set(self.all_hadm_ids))

    def test_proxy_model_scores_use_predict_proba_positive_class(self):
        build_proxy_probability_scores = self._get_callable("build_proxy_probability_scores")
        estimator = _FakeProbEstimator([0.3, 0.8])
        feature_matrix = pd.DataFrame(
            [{"hadm_id": 1, "feature_a": 1}, {"hadm_id": 2, "feature_a": 0}]
        )
        labels = pd.DataFrame(
            [{"hadm_id": 1, "noncompliance_label": 1}, {"hadm_id": 2, "noncompliance_label": 0}]
        )
        scores = build_proxy_probability_scores(
            feature_matrix,
            labels,
            "noncompliance_label",
            estimator_factory=lambda: estimator,
        )
        self.assertEqual(list(scores["noncompliance_score"]), [0.3, 0.8])

    def test_sentiment_mistrust_score_is_negative_polarity(self):
        build_negative_sentiment_scores = self._get_callable("build_negative_sentiment_scores")
        note_corpus = pd.DataFrame(
            [
                {"hadm_id": 1, "note_text": "very negative"},
                {"hadm_id": 2, "note_text": "neutral"},
                {"hadm_id": 3, "note_text": "positive"},
            ]
        )
        polarity_map = {"very negative": -0.5, "neutral": 0.0, "positive": 0.2}
        scores = build_negative_sentiment_scores(
            note_corpus,
            sentiment_fn=lambda text: (polarity_map[text], 0.0),
        ).set_index("hadm_id")
        self.assertAlmostEqual(scores.loc[1, "negative_sentiment_score"], 0.5)
        self.assertAlmostEqual(scores.loc[2, "negative_sentiment_score"], 0.0)
        self.assertAlmostEqual(scores.loc[3, "negative_sentiment_score"], -0.2)

    def test_negative_sentiment_scores_send_cleaned_text_to_sentiment_function(self):
        build_negative_sentiment_scores = self._get_callable("build_negative_sentiment_scores")
        seen = []

        def sentiment_fn(text):
            seen.append(text)
            return (0.0, 0.0)

        note_corpus = pd.DataFrame(
            [{"hadm_id": 1, "note_text": "Patient\trefused\n treatment   Date:[**5-1-18**]"}]
        )
        build_negative_sentiment_scores(note_corpus, sentiment_fn=sentiment_fn)
        self.assertEqual(seen, ["Patient refused treatment Date:[**5-1-18**]"])

    def test_each_mistrust_score_is_normalized_independently(self):
        scores, _ = self._build_mistrust_scores()
        for column in [
            "noncompliance_score_z",
            "autopsy_score_z",
            "negative_sentiment_score_z",
        ]:
            self.assertAlmostEqual(float(scores[column].mean()), 0.0, places=7)
            self.assertAlmostEqual(float(scores[column].std(ddof=0)), 1.0, places=7)

    def test_normalized_scores_have_mean_near_zero_and_unit_variance(self):
        scores, _ = self._build_mistrust_scores()
        for column in [
            "noncompliance_score_z",
            "autopsy_score_z",
            "negative_sentiment_score_z",
        ]:
            self.assertLess(abs(float(scores[column].mean())), 0.01)
            self.assertGreaterEqual(float(scores[column].std(ddof=0)), 0.99)
            self.assertLessEqual(float(scores[column].std(ddof=0)), 1.01)

    def test_noncompliance_feature_weights_have_expected_signals(self):
        self._pending_real_data(
            "Largest positive noncompliance coefficient should contain agitat or riker, and largest negative should contain alert."
        )

    def test_noncompliance_feature_weight_validation_includes_pain_and_calm_checks(self):
        self._pending_real_data(
            "Feature-weight validation should also confirm pain-related features rank positively and alert/calm or no-pain features rank negatively."
        )

    def test_autopsy_feature_weights_have_expected_signals(self):
        self._pending_real_data(
            "Autopsy feature-weight validation should confirm restraint and orientation signals rank positively while no-pain, proxy, or family-communication signals rank negatively."
        )

    def test_race_gap_validation_for_mistrust_scores_matches_expected_directionality(self):
        self._pending_real_data(
            "Noncompliance and sentiment scores should be higher for Black admissions with p < 0.05; autopsy score p should remain > 0.05."
        )

    def test_noncompliance_race_gap_is_significant_with_black_median_higher(self):
        self._pending_real_data(
            "Noncompliance mistrust should show a significant White-vs-Black gap with Black median higher than White median."
        )

    def test_sentiment_race_gap_is_significant_with_black_median_higher(self):
        self._pending_real_data(
            "Negative-sentiment mistrust should show a significant White-vs-Black gap with Black median higher than White median."
        )

    def test_autopsy_race_gap_is_non_significant(self):
        self._pending_real_data(
            "Autopsy-derived mistrust should remain non-significant between White and Black admissions."
        )

    def test_race_gap_validation_merges_scores_with_race_and_restricts_to_white_and_black(self):
        self._pending_real_data(
            "Race-gap validation must merge mistrust scores with race and restrict analysis to White and Black admissions only."
        )

    def test_race_gap_validation_uses_two_sided_mann_whitney_for_each_metric(self):
        self._pending_real_data(
            "Race-gap validation must use two-sided Mann-Whitney tests separately for noncompliance, autopsy, and sentiment metrics."
        )

    def test_treatment_disparity_uses_admission_level_vent_and_vaso_totals(self):
        build_treatment_totals = self._get_callable("build_treatment_totals")
        totals = build_treatment_totals(
            self.icustays,
            self.ventdurations,
            self.vasopressordurations,
        ).fillna(0).set_index("hadm_id")
        self.assertEqual(totals.loc[302, "total_vent_min"], 810.0)
        self.assertEqual(totals.loc[303, "total_vaso_min"], 840.0)

    def test_treatment_totals_respect_exact_six_hundred_minute_merge_boundary(self):
        build_treatment_totals = self._get_callable("build_treatment_totals")
        icustays = pd.DataFrame(
            [
                {
                    "hadm_id": 950,
                    "icustay_id": 9501,
                    "intime": "2100-09-01 00:00:00",
                    "outtime": "2100-09-01 12:00:00",
                },
                {
                    "hadm_id": 950,
                    "icustay_id": 9502,
                    "intime": "2100-09-01 20:00:00",
                    "outtime": "2100-09-02 04:00:00",
                },
            ]
        )
        ventdurations = pd.DataFrame(
            [
                {
                    "icustay_id": 9501,
                    "ventnum": 1,
                    "starttime": "2100-09-01 00:00:00",
                    "endtime": "2100-09-01 01:00:00",
                    "duration_hours": 1.0,
                },
                {
                    "icustay_id": 9501,
                    "ventnum": 2,
                    "starttime": "2100-09-01 11:00:00",
                    "endtime": "2100-09-01 12:00:00",
                    "duration_hours": 1.0,
                },
                {
                    "icustay_id": 9502,
                    "ventnum": 3,
                    "starttime": "2100-09-01 22:01:00",
                    "endtime": "2100-09-01 23:01:00",
                    "duration_hours": 1.0,
                },
            ]
        )
        empty_vaso = pd.DataFrame(
            columns=["icustay_id", "vasonum", "starttime", "endtime", "duration_hours"]
        )

        totals = build_treatment_totals(icustays, ventdurations, empty_vaso).fillna(0).set_index(
            "hadm_id"
        )
        self.assertEqual(totals.loc[950, "total_vent_min"], 780.0)

    def test_race_based_treatment_disparity_restricts_eol_to_white_and_black(self):
        self._pending_real_data(
            "Race-based treatment disparity analysis must restrict the EOL cohort to race in {WHITE, BLACK}."
        )

    def test_race_based_treatment_disparity_drops_null_treatment_durations_per_test(self):
        self._pending_real_data(
            "Race-based treatment disparity analysis must drop null treatment durations separately for ventilation and vasopressor tests."
        )

    def test_race_based_treatment_disparity_uses_two_sided_mann_whitney(self):
        self._pending_real_data(
            "Race-based treatment disparity analysis must compare Black vs White with two-sided Mann-Whitney tests."
        )

    def test_race_based_treatment_disparity_records_black_sample_sizes_for_later_use(self):
        self._pending_real_data(
            "Race-based treatment disparity analysis must record Black sample sizes for ventilation and vasopressors for later mistrust-group stratification."
        )

    def test_race_based_treatment_disparity_expected_black_sample_sizes_match_reference(self):
        self._pending_real_data(
            "Race-based treatment disparity analysis should recover Black sample sizes approximately equal to n_black_vent ~= 510 and n_black_vaso ~= 453."
        )

    def test_treatment_disparity_keeps_only_non_null_scores_and_treatments(self):
        self._pending_real_data(
            "Treatment-disparity analysis must keep only rows with non-null treatment duration and non-null mistrust score."
        )

    def test_treatment_disparity_stratification_uses_top_n_high_mistrust(self):
        self._pending_real_data(
            "For each metric-treatment pair, admissions must be sorted descending by score and high mistrust must be defined as the top N rows."
        )

    def test_treatment_disparity_uses_black_sample_size_as_group_size(self):
        self._pending_real_data(
            "High-mistrust group size N must equal the corresponding Black sample size from the race-based treatment analysis."
        )

    def test_treatment_disparity_uses_treatment_specific_black_group_sizes(self):
        self._pending_real_data(
            "Trust-based treatment disparity analysis must use N = n_black_vent for ventilation and N = n_black_vaso for vasopressors."
        )

    def test_treatment_disparity_computes_mann_whitney_and_median_gap(self):
        self._pending_real_data(
            "Treatment-disparity analysis must compute two-sided Mann-Whitney U and median(high) - median(low)."
        )

    def test_noncompliance_ventilation_treatment_gap_matches_reference_direction(self):
        self._pending_real_data(
            "Noncompliance-based ventilation disparity should be strongly significant with a large positive median gap near the paper reference."
        )

    def test_autopsy_ventilation_treatment_gap_matches_reference_direction(self):
        self._pending_real_data(
            "Autopsy-based ventilation disparity should be strongly significant with a large positive median gap near the paper reference."
        )

    def test_sentiment_ventilation_treatment_gap_matches_reference_direction(self):
        self._pending_real_data(
            "Sentiment-based ventilation disparity should remain significant with a smaller positive median gap near the paper reference."
        )

    def test_ventilation_trust_based_gaps_exceed_race_gap_for_noncompliance_and_autopsy(self):
        self._pending_real_data(
            "For ventilation, noncompliance and autopsy mistrust gaps must each exceed 1.5x the race-based gap."
        )

    def test_sentiment_vasopressor_result_remains_non_significant(self):
        self._pending_real_data(
            "Sentiment-based vasopressor disparity should remain non-significant with p > 0.10."
        )

    def test_acuity_scores_merge_to_mistrust_scores_by_hadm_id(self):
        build_acuity_scores = self._get_callable("build_acuity_scores")
        acuity = build_acuity_scores(self.oasis, self.sapsii)
        scores, _ = self._build_mistrust_scores()
        merged = scores.merge(acuity, on="hadm_id", how="inner")
        self.assertEqual(set(merged["hadm_id"]), set(acuity["hadm_id"]))
        self.assertTrue({"oasis", "sapsii"}.issubset(merged.columns))

    def test_acuity_aggregation_rule_is_deterministic_for_multiple_icu_stays(self):
        build_acuity_scores = self._get_callable("build_acuity_scores")
        first = build_acuity_scores(self.oasis, self.sapsii)
        second = build_acuity_scores(self.oasis, self.sapsii)
        self.assertTrue(first.equals(second))
        self.assertEqual(len(first.loc[first["hadm_id"] == 302]), 1)

    def test_acuity_correlations_match_expected_ranges(self):
        self._pending_real_data(
            "OASIS-SAPSII correlation should be in 0.60-0.75, each mistrust-acuity correlation should have |r| < 0.15, and noncompliance-autopsy correlation should be in 0.15-0.35."
        )

    def test_acuity_control_uses_pairwise_pearson_correlations_across_all_five_metrics(self):
        self._pending_real_data(
            "Acuity-control analysis must compute pairwise Pearson correlations across noncompliance, autopsy, sentiment, OASIS, and SAPS II."
        )

    def test_oasis_sapsii_correlation_matches_expected_reference_range(self):
        self._pending_real_data(
            "OASIS-SAPSII correlation should remain within the expected reference range around 0.679."
        )

    def test_mistrust_acuity_correlations_remain_weak(self):
        self._pending_real_data(
            "Each mistrust-to-acuity Pearson correlation should have absolute value below 0.15."
        )

    def test_noncompliance_autopsy_correlation_matches_expected_reference_range(self):
        self._pending_real_data(
            "Noncompliance-to-autopsy mistrust correlation should remain within the expected reference band around 0.262."
        )

    def test_left_ama_target_definition_is_exact(self):
        build_final_model_table = self._get_callable("build_final_model_table")
        final = build_final_model_table(
            demographics=self._build_demographics(),
            all_cohort=self._build_all(),
            admissions=self._build_base(),
            chartevents=self.chartevents,
            d_items=self.d_items,
            mistrust_scores=pd.DataFrame(
                [
                    {
                        "hadm_id": hadm_id,
                        "noncompliance_score_z": 0.0,
                        "autopsy_score_z": 0.0,
                        "negative_sentiment_score_z": 0.0,
                    }
                    for hadm_id in self.all_hadm_ids
                ]
            ),
            include_race=False,
            include_mistrust=False,
        ).set_index("hadm_id")
        self.assertEqual(final.loc[305, "left_ama"], 1)
        self.assertEqual(final.loc[306, "left_ama"], 0)

    def test_code_status_target_uses_required_itemids_and_values(self):
        build_code_status_target = getattr(self.module, "_build_code_status_target")
        target = build_code_status_target(self.chartevents, self.d_items).set_index("hadm_id")
        self.assertEqual(target.loc[302, "code_status_dnr_dni_cmo"], 1)
        self.assertEqual(target.loc[303, "code_status_dnr_dni_cmo"], 1)
        self.assertEqual(target.loc[305, "code_status_dnr_dni_cmo"], 0)
        self.assertNotIn(304, set(target.index))

    def test_code_status_task_excludes_admissions_without_charted_code_status(self):
        build_code_status_target = getattr(self.module, "_build_code_status_target")
        target = build_code_status_target(self.chartevents, self.d_items)
        self.assertNotIn(306, set(target["hadm_id"]))

    def test_in_hospital_mortality_target_comes_from_hospital_expire_flag(self):
        build_final_model_table = self._get_callable("build_final_model_table")
        final = build_final_model_table(
            demographics=self._build_demographics(),
            all_cohort=self._build_all(),
            admissions=self._build_base(),
            chartevents=self.chartevents,
            d_items=self.d_items,
            mistrust_scores=pd.DataFrame(
                [
                    {
                        "hadm_id": hadm_id,
                        "noncompliance_score_z": 0.0,
                        "autopsy_score_z": 0.0,
                        "negative_sentiment_score_z": 0.0,
                    }
                    for hadm_id in self.all_hadm_ids
                ]
            ),
            include_race=False,
            include_mistrust=False,
        ).set_index("hadm_id")
        self.assertEqual(final.loc[304, "in_hospital_mortality"], 1)
        self.assertEqual(final.loc[302, "in_hospital_mortality"], 0)

    def test_downstream_feature_configurations_have_exact_required_widths(self):
        final = self._get_callable("build_final_model_table")(
            demographics=self._build_demographics(),
            all_cohort=self._build_all(),
            admissions=self._build_base(),
            chartevents=self.chartevents,
            d_items=self.d_items,
            mistrust_scores=self._zero_mistrust_scores(),
            include_race=True,
            include_mistrust=True,
        )
        expected_widths = {
            "Baseline": 7,
            "Baseline + Race": 13,
            "Baseline + Noncompliant": 8,
            "Baseline + Autopsy": 8,
            "Baseline + Neg-Sentiment": 8,
            "Baseline + ALL": 16,
        }
        for name, columns in self._required_downstream_feature_configs().items():
            self.assertTrue(set(columns).issubset(set(final.columns)), msg=name)
            self.assertEqual(len(columns), expected_widths[name], msg=name)

    def test_downstream_configuration_names_match_required_six_configs(self):
        final = self._get_callable("build_final_model_table")(
            demographics=self._build_demographics(),
            all_cohort=self._build_all(),
            admissions=self._build_base(),
            chartevents=self.chartevents,
            d_items=self.d_items,
            mistrust_scores=self._zero_mistrust_scores(),
            include_race=True,
            include_mistrust=True,
        )
        configuration_map = self._required_downstream_feature_configs()
        self.assertEqual(
            set(configuration_map),
            {
                "Baseline",
                "Baseline + Race",
                "Baseline + Noncompliant",
                "Baseline + Autopsy",
                "Baseline + Neg-Sentiment",
                "Baseline + ALL",
            },
        )
        for columns in configuration_map.values():
            self.assertTrue(set(columns).issubset(set(final.columns)))

    def test_downstream_outputs_cover_all_three_tasks_and_six_configurations(self):
        self._pending_real_data(
            "Downstream results must cover all three tasks across all six required configurations."
        )

    def test_downstream_result_table_has_eighteen_task_configuration_entries(self):
        self._pending_real_data(
            "Downstream outputs should expose 18 task-configuration result entries: 3 tasks x 6 configurations."
        )

    def test_final_model_table_contains_required_downstream_feature_columns(self):
        final = self._get_callable("build_final_model_table")(
            demographics=self._build_demographics(),
            all_cohort=self._build_all(),
            admissions=self._build_base(),
            chartevents=self.chartevents,
            d_items=self.d_items,
            mistrust_scores=pd.DataFrame(
                [
                    {
                        "hadm_id": hadm_id,
                        "noncompliance_score_z": 0.0,
                        "autopsy_score_z": 0.0,
                        "negative_sentiment_score_z": 0.0,
                    }
                    for hadm_id in self.all_hadm_ids
                ]
            ),
            include_race=True,
            include_mistrust=True,
        )
        required_columns = {
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
        }
        self.assertTrue(required_columns.issubset(set(final.columns)))

    def test_final_model_table_native_american_admission_sets_race_native_american_to_one(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        build_demographics_table = self._get_callable("build_demographics_table")
        build_all_cohort = self._get_callable("build_all_cohort")
        build_final_model_table = self._get_callable("build_final_model_table")
        admissions = pd.DataFrame(
            [
                {
                    "hadm_id": 990,
                    "subject_id": 90,
                    "admittime": "2100-01-01 00:00:00",
                    "dischtime": "2100-01-02 00:00:00",
                    "ethnicity": "AMERICAN INDIAN/ALASKA NATIVE",
                    "insurance": "Medicare",
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                }
            ]
        )
        patients = pd.DataFrame(
            [{"subject_id": 90, "gender": "F", "dob": "2070-01-01 00:00:00"}]
        )
        icustays = pd.DataFrame(
            [
                {
                    "hadm_id": 990,
                    "icustay_id": 9901,
                    "intime": "2100-01-01 00:00:00",
                    "outtime": "2100-01-01 13:00:00",
                }
            ]
        )
        base = build_base_admissions(admissions, patients)
        demographics = build_demographics_table(base)
        all_cohort = build_all_cohort(base, icustays)
        final = build_final_model_table(
            demographics=demographics,
            all_cohort=all_cohort,
            admissions=base,
            chartevents=pd.DataFrame(columns=["hadm_id", "itemid", "value", "icustay_id"]),
            d_items=pd.DataFrame(columns=["itemid", "label", "dbsource"]),
            mistrust_scores=pd.DataFrame(
                [
                    {
                        "hadm_id": 990,
                        "noncompliance_score_z": 0.0,
                        "autopsy_score_z": 0.0,
                        "negative_sentiment_score_z": 0.0,
                    }
                ]
            ),
            include_race=True,
            include_mistrust=True,
        ).set_index("hadm_id")
        self.assertEqual(final.loc[990, "race_native_american"], 1)

    def test_downstream_evaluation_runs_100_random_60_40_splits(self):
        self._pending_real_data(
            "Each downstream task-configuration pair must run 100 repetitions with random_state 0..99 and a 60/40 train/test split."
        )

    def test_downstream_evaluation_uses_random_states_zero_through_ninety_nine(self):
        self._pending_real_data(
            "Downstream evaluation must use the exact sequence of random_state values 0 through 99."
        )

    def test_downstream_evaluation_uses_sixty_forty_train_test_split(self):
        self._pending_real_data(
            "Downstream evaluation must use a 60/40 train/test split for every task and configuration."
        )

    def test_downstream_evaluation_drops_rows_with_null_target_or_required_features(self):
        self._pending_real_data(
            "Downstream evaluation must drop rows with null targets or null required feature values before fitting each task/configuration pair."
        )

    def test_downstream_estimator_and_metric_match_spec(self):
        self._pending_real_data(
            'Downstream evaluation must use LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000) and roc_auc_score.'
        )

    def test_downstream_auc_uses_predicted_probabilities_on_the_test_split(self):
        self._pending_real_data(
            "Downstream evaluation must compute ROC AUC from predicted probabilities on the held-out test split."
        )

    def test_downstream_results_report_mean_and_std_auc(self):
        self._pending_real_data(
            "Downstream prediction output must report mean AUC and standard deviation across the 100 splits."
        )

    def test_downstream_benchmark_ranges_match_expected_baselines(self):
        self._pending_real_data(
            "Regression checks should compare AMA, Code Status, and Mortality baseline AUCs against the expected benchmark ranges."
        )

    def test_baseline_ama_auc_matches_expected_reference_range(self):
        self._pending_real_data(
            "Baseline AMA AUC should remain near the paper reference mean and standard deviation."
        )

    def test_baseline_code_status_auc_matches_expected_reference_range(self):
        self._pending_real_data(
            "Baseline Code Status AUC should remain near the paper reference mean and standard deviation."
        )

    def test_baseline_mortality_auc_matches_expected_reference_range(self):
        self._pending_real_data(
            "Baseline Mortality AUC should remain near the paper reference mean and standard deviation."
        )

    def test_baseline_plus_all_is_best_or_near_best(self):
        self._pending_real_data(
            "Baseline + ALL should be the best configuration or within 0.005 of the best across downstream tasks."
        )

    def test_mortality_improvement_from_baseline_to_all_is_in_expected_range(self):
        self._pending_real_data(
            "Mortality AUC improvement from Baseline to Baseline + ALL should be between 0.02 and 0.06."
        )

    def test_module_produces_required_model_artifacts(self):
        self._pending_real_data(
            "Module implementation must produce treatment disparity results, acuity correlations, and downstream prediction outputs in addition to the admission-level tables."
        )

    def test_module_outputs_include_binary_chartevent_feature_matrix_artifact(self):
        self._pending_real_data(
            "Final outputs must include the binary chart-event feature matrix artifact keyed by hadm_id."
        )

    def test_module_outputs_include_note_derived_label_artifact(self):
        self._pending_real_data(
            "Final outputs must include the note-derived label table with noncompliance_label and autopsy_label columns."
        )

    def test_module_outputs_include_three_normalized_mistrust_scores(self):
        self._pending_real_data(
            "Final outputs must include the three independently normalized mistrust score columns."
        )

    def test_module_outputs_include_treatment_disparity_results(self):
        self._pending_real_data(
            "Final outputs must include treatment disparity results for all required metric/treatment pairs."
        )

    def test_module_outputs_include_acuity_correlation_results(self):
        self._pending_real_data(
            "Final outputs must include acuity-correlation results across mistrust and acuity measures."
        )

    def test_module_outputs_include_downstream_auc_results_for_all_tasks_and_configs(self):
        self._pending_real_data(
            "Final outputs must include downstream mean-plus-std AUC results for all tasks and all six configurations."
        )

    def test_model_artifacts_are_unique_and_aligned_on_hadm_id(self):
        all_cohort = self._build_all()
        feature_matrix = self._build_feature_matrix()
        note_labels = self._build_note_labels()
        note_corpus = self._build_note_corpus()
        mistrust_scores, _ = self._build_mistrust_scores()
        acuity = self._get_callable("build_acuity_scores")(self.oasis, self.sapsii)
        final = self._get_callable("build_final_model_table")(
            demographics=self._build_demographics(),
            all_cohort=all_cohort,
            admissions=self._build_base(),
            chartevents=self.chartevents,
            d_items=self.d_items,
            mistrust_scores=mistrust_scores,
            include_race=True,
            include_mistrust=True,
        )
        expected_hadm_ids = {
            "feature_matrix": set(all_cohort["hadm_id"]),
            "note_labels": set(all_cohort["hadm_id"]),
            "note_corpus": set(all_cohort["hadm_id"]),
            "mistrust_scores": set(all_cohort["hadm_id"]),
            "acuity": set(acuity["hadm_id"]),
            "final": set(final["hadm_id"]),
        }
        for name, artifact in [
            ("feature_matrix", feature_matrix),
            ("note_labels", note_labels),
            ("note_corpus", note_corpus),
            ("mistrust_scores", mistrust_scores),
            ("acuity", acuity),
            ("final", final),
        ]:
            self._assert_hadm_unique(artifact, "Artifact")
            self.assertEqual(set(artifact["hadm_id"]), expected_hadm_ids[name])


if __name__ == "__main__":
    unittest.main()

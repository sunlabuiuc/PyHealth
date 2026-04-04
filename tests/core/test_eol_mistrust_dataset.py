import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def _load_eol_mistrust_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "pyhealth" / "datasets" / "eol_mistrust.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pyhealth.datasets.eol_mistrust_dataset_tests",
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
        n = len(X)
        probs = self.probabilities[:n]
        return [[1.0 - prob, prob] for prob in probs]


class TestEOLMistrustPreprocessing(unittest.TestCase):
    """TDD spec for the end-of-life mistrust data-preparation pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.module = _load_eol_mistrust_module()

    def setUp(self):
        self.admissions = pd.DataFrame(
            [
                {
                    "hadm_id": 100,
                    "subject_id": 1,
                    "admittime": "2100-01-01 00:00:00",
                    "dischtime": "2100-01-02 00:00:00",
                    "ethnicity": "WHITE - RUSSIAN",
                    "insurance": "Medicare",
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 101,
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
                    "hadm_id": 102,
                    "subject_id": 3,
                    "admittime": "2100-03-01 00:00:00",
                    "dischtime": "2100-03-01 05:00:00",
                    "ethnicity": "HISPANIC OR LATINO",
                    "insurance": "Self Pay",
                    "discharge_location": "SKILLED NURSING FACILITY",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 103,
                    "subject_id": 4,
                    "admittime": "2100-04-01 00:00:00",
                    "dischtime": "2100-04-01 20:00:00",
                    "ethnicity": "ASIAN - CHINESE",
                    "insurance": "Medicaid",
                    "discharge_location": "SKILLED NURSING FACILITY",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 104,
                    "subject_id": 5,
                    "admittime": "2100-05-01 00:00:00",
                    "dischtime": "2100-05-01 10:00:00",
                    "ethnicity": "PATIENT DECLINED TO ANSWER",
                    "insurance": "Government",
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 1,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 105,
                    "subject_id": 6,
                    "admittime": "2100-06-01 00:00:00",
                    "dischtime": "2100-06-02 06:00:00",
                    "ethnicity": "WHITE",
                    "insurance": "Private",
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 0,
                },
                {
                    "hadm_id": 106,
                    "subject_id": 7,
                    "admittime": "2100-07-01 00:00:00",
                    "dischtime": "2100-07-02 06:00:00",
                    "ethnicity": "WHITE",
                    "insurance": "Private",
                    "discharge_location": "LEFT AGAINST MEDICAL ADVICE",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 107,
                    "subject_id": 8,
                    "admittime": "2100-08-01 00:00:00",
                    "dischtime": "2100-08-02 00:00:00",
                    "ethnicity": "BLACK/CAPE VERDEAN",
                    "insurance": "Medicare",
                    "discharge_location": "HOME HOSPICE",
                    "hospital_expire_flag": 1,
                    "has_chartevents_data": 1,
                },
            ]
        )
        self.patients = pd.DataFrame(
            [
                {"subject_id": 1, "gender": "M", "dob": "2070-01-01 00:00:00"},
                {"subject_id": 2, "gender": "F", "dob": "1800-01-01 00:00:00"},
                {"subject_id": 3, "gender": "F", "dob": "2080-03-01 00:00:00"},
                {"subject_id": 4, "gender": "M", "dob": "2060-04-01 00:00:00"},
                {"subject_id": 5, "gender": "F", "dob": "2050-05-01 00:00:00"},
                {"subject_id": 6, "gender": "M", "dob": "2040-06-01 00:00:00"},
                {"subject_id": 7, "gender": "F", "dob": "2075-07-01 00:00:00"},
                {"subject_id": 8, "gender": "M", "dob": "2035-08-01 00:00:00"},
            ]
        )
        self.icustays = pd.DataFrame(
            [
                {"hadm_id": 100, "icustay_id": 1001, "intime": "2100-01-01 00:00:00", "outtime": "2100-01-01 11:00:00"},
                {"hadm_id": 100, "icustay_id": 1002, "intime": "2100-01-01 12:00:00", "outtime": "2100-01-01 23:00:00"},
                {"hadm_id": 101, "icustay_id": 1011, "intime": "2100-02-01 00:00:00", "outtime": "2100-02-01 13:00:00"},
                {"hadm_id": 103, "icustay_id": 1031, "intime": "2100-04-01 00:00:00", "outtime": "2100-04-01 12:00:00"},
                {"hadm_id": 104, "icustay_id": 1041, "intime": "2100-05-01 01:00:00", "outtime": "2100-05-01 10:00:00"},
                {"hadm_id": 105, "icustay_id": 1051, "intime": "2100-06-01 00:00:00", "outtime": "2100-06-01 14:00:00"},
                {"hadm_id": 106, "icustay_id": 1061, "intime": "2100-07-01 00:00:00", "outtime": "2100-07-01 15:00:00"},
                {"hadm_id": 107, "icustay_id": 1071, "intime": "2100-08-01 00:00:00", "outtime": "2100-08-01 13:00:00"},
            ]
        )
        self.noteevents = pd.DataFrame(
            [
                {
                    "hadm_id": 101,
                    "category": "Nursing",
                    "text": "Patient refuses treatment and was noncompliant with medication.  Date:[**5-1-18**]",
                    "iserror": 0,
                },
                {
                    "hadm_id": 101,
                    "category": "Physician",
                    "text": "Autopsy was discussed with the family.",
                    "iserror": 0,
                },
                {
                    "hadm_id": 103,
                    "category": "Nursing",
                    "text": "Cooperative patient. Follows commands.",
                    "iserror": 0,
                },
                {
                    "hadm_id": 104,
                    "category": "Discharge",
                    "text": "AUTOPSY requested.",
                    "iserror": 1,
                },
                {
                    "hadm_id": 104,
                    "category": "Nursing",
                    "text": "No concerns documented.",
                    "iserror": 0,
                },
                {
                    "hadm_id": 106,
                    "category": "Nursing",
                    "text": "Patient remains nonadherent with follow up plan.",
                    "iserror": 0,
                },
            ]
        )
        self.d_items = pd.DataFrame(
            [
                {"itemid": 1, "label": "Education Readiness", "dbsource": "carevue"},
                {"itemid": 2, "label": "Pain Level", "dbsource": "metavision"},
                {"itemid": 3, "label": "Code Status", "dbsource": "carevue"},
            ]
        )
        self.chartevents = pd.DataFrame(
            [
                {"hadm_id": 101, "itemid": 1, "value": "No", "icustay_id": 1011},
                {"hadm_id": 101, "itemid": 1, "value": "No", "icustay_id": 1011},
                {"hadm_id": 101, "itemid": 2, "value": "7-Mod to Severe", "icustay_id": 1011},
                {"hadm_id": 101, "itemid": 3, "value": "Full Code", "icustay_id": 1011},
                {"hadm_id": 103, "itemid": 1, "value": "Yes", "icustay_id": 1031},
                {"hadm_id": 103, "itemid": 3, "value": "DNR/DNI", "icustay_id": 1031},
                {"hadm_id": 104, "itemid": 3, "value": "Full Code", "icustay_id": 1041},
                {"hadm_id": 106, "itemid": 3, "value": "Full Code", "icustay_id": 1061},
                {"hadm_id": 107, "itemid": 3, "value": "Comfort Measures Only", "icustay_id": 1071},
            ]
        )
        self.ventdurations = pd.DataFrame(
            [
                {
                    "icustay_id": 1011,
                    "ventnum": 1,
                    "starttime": "2100-02-01 00:00:00",
                    "endtime": "2100-02-01 02:00:00",
                    "duration_hours": 2.0,
                },
                {
                    "icustay_id": 1011,
                    "ventnum": 2,
                    "starttime": "2100-02-01 11:30:00",
                    "endtime": "2100-02-01 12:30:00",
                    "duration_hours": 1.0,
                },
                {
                    "icustay_id": 1011,
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
                    "icustay_id": 1031,
                    "vasonum": 1,
                    "starttime": "2100-04-01 01:00:00",
                    "endtime": "2100-04-01 03:00:00",
                    "duration_hours": 2.0,
                },
                {
                    "icustay_id": 1031,
                    "vasonum": 2,
                    "starttime": "2100-04-01 02:30:00",
                    "endtime": "2100-04-01 05:00:00",
                    "duration_hours": 2.5,
                },
                {
                    "icustay_id": 1031,
                    "vasonum": 3,
                    "starttime": "2100-04-01 14:00:00",
                    "endtime": "2100-04-01 15:00:00",
                    "duration_hours": 1.0,
                },
            ]
        )
        self.oasis = pd.DataFrame(
            [
                {"hadm_id": 101, "icustay_id": 1011, "oasis": 15},
                {"hadm_id": 103, "icustay_id": 1031, "oasis": 20},
                {"hadm_id": 106, "icustay_id": 1061, "oasis": 8},
                {"hadm_id": 107, "icustay_id": 1071, "oasis": 30},
            ]
        )
        self.sapsii = pd.DataFrame(
            [
                {"hadm_id": 101, "icustay_id": 1011, "sapsii": 42},
                {"hadm_id": 103, "icustay_id": 1031, "sapsii": 55},
                {"hadm_id": 106, "icustay_id": 1061, "sapsii": 12},
                {"hadm_id": 107, "icustay_id": 1071, "sapsii": 70},
            ]
        )
        self.mistrust_scores = pd.DataFrame(
            [
                {
                    "hadm_id": 101,
                    "noncompliance_score_z": 1.2,
                    "autopsy_score_z": 0.7,
                    "negative_sentiment_score_z": 0.9,
                },
                {
                    "hadm_id": 103,
                    "noncompliance_score_z": -0.3,
                    "autopsy_score_z": -0.2,
                    "negative_sentiment_score_z": -0.1,
                },
                {
                    "hadm_id": 106,
                    "noncompliance_score_z": 0.8,
                    "autopsy_score_z": -0.4,
                    "negative_sentiment_score_z": 0.2,
                },
                {
                    "hadm_id": 107,
                    "noncompliance_score_z": -1.0,
                    "autopsy_score_z": 1.1,
                    "negative_sentiment_score_z": -1.0,
                },
            ]
        )

    def _get_callable(self, name):
        self.assertTrue(
            hasattr(self.module, name),
            msg=f"Implement `{name}` in pyhealth.datasets.eol_mistrust",
        )
        attr = getattr(self.module, name)
        self.assertTrue(callable(attr), msg=f"`{name}` must be callable")
        return attr

    def _assert_hadm_unique(self, df, msg_prefix):
        self.assertIn("hadm_id", df.columns, msg=f"{msg_prefix} must include hadm_id")
        self.assertTrue(
            df["hadm_id"].is_unique,
            msg=f"{msg_prefix} must be unique at the admission level",
        )

    def _build_database_environment_inputs(
        self,
        num_admissions=50010,
        include_multiple_icustays=True,
    ):
        hadm_ids = list(range(200000, 200000 + num_admissions))
        subject_ids = list(range(300000, 300000 + num_admissions))
        icustay_ids = list(range(400000, 400000 + num_admissions))

        admissions = pd.DataFrame(
            {
                "hadm_id": hadm_ids,
                "subject_id": subject_ids,
                "admittime": ["2100-01-01 00:00:00"] * num_admissions,
                "dischtime": ["2100-01-02 00:00:00"] * num_admissions,
                "ethnicity": ["WHITE"] * num_admissions,
                "insurance": ["Medicare"] * num_admissions,
                "discharge_location": ["HOME"] * num_admissions,
                "hospital_expire_flag": [0] * num_admissions,
                "has_chartevents_data": [1] * num_admissions,
            }
        )
        patients = pd.DataFrame(
            {
                "subject_id": subject_ids,
                "gender": ["M"] * num_admissions,
                "dob": ["2070-01-01 00:00:00"] * num_admissions,
            }
        )
        icustays = pd.DataFrame(
            {
                "hadm_id": hadm_ids,
                "icustay_id": icustay_ids,
                "intime": ["2100-01-01 00:00:00"] * num_admissions,
                "outtime": ["2100-01-01 13:00:00"] * num_admissions,
            }
        )
        if include_multiple_icustays:
            icustays = pd.concat(
                [
                    icustays,
                    pd.DataFrame(
                        [
                            {
                                "hadm_id": hadm_ids[0],
                                "icustay_id": 999999,
                                "intime": "2100-01-02 00:00:00",
                                "outtime": "2100-01-02 14:00:00",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

        noteevents = pd.DataFrame(
            [
                {
                    "hadm_id": hadm_ids[0],
                    "category": "Nursing",
                    "text": "Patient refuses treatment and autopsy discussed.",
                    "iserror": 0,
                }
            ]
        )
        d_items = pd.DataFrame(
            [
                {"itemid": 1, "label": "Education Readiness", "dbsource": "carevue"},
            ]
        )
        chartevents = pd.DataFrame(
            [
                {
                    "hadm_id": hadm_ids[0],
                    "itemid": 1,
                    "value": "No",
                    "icustay_id": icustay_ids[0],
                }
            ]
        )
        ventdurations = pd.DataFrame(
            [
                {
                    "icustay_id": icustay_ids[0],
                    "ventnum": 1,
                    "starttime": "2100-01-01 00:00:00",
                    "endtime": "2100-01-01 01:00:00",
                    "duration_hours": 1.0,
                }
            ]
        )
        vasopressordurations = pd.DataFrame(
            [
                {
                    "icustay_id": icustay_ids[0],
                    "vasonum": 1,
                    "starttime": "2100-01-01 02:00:00",
                    "endtime": "2100-01-01 03:00:00",
                    "duration_hours": 1.0,
                }
            ]
        )
        oasis = pd.DataFrame(
            [
                {"hadm_id": hadm_ids[0], "icustay_id": icustay_ids[0], "oasis": 15},
            ]
        )
        sapsii = pd.DataFrame(
            [
                {"hadm_id": hadm_ids[0], "icustay_id": icustay_ids[0], "sapsii": 42},
            ]
        )

        raw_tables = {
            "admissions": admissions,
            "patients": patients,
            "icustays": icustays,
            "noteevents": noteevents,
            "chartevents": chartevents,
            "d_items": d_items,
        }
        materialized_views = {
            "ventdurations": ventdurations,
            "vasopressordurations": vasopressordurations,
            "oasis": oasis,
            "sapsii": sapsii,
        }
        return raw_tables, materialized_views

    def test_map_ethnicity_matches_required_categories(self):
        map_ethnicity = self._get_callable("map_ethnicity")
        self.assertEqual(map_ethnicity("WHITE - RUSSIAN"), "WHITE")
        self.assertEqual(map_ethnicity("BLACK/AFRICAN AMERICAN"), "BLACK")
        self.assertEqual(map_ethnicity("ASIAN - CHINESE"), "ASIAN")
        self.assertEqual(map_ethnicity("HISPANIC OR LATINO"), "HISPANIC")
        self.assertEqual(map_ethnicity("AMERICAN INDIAN/ALASKA NATIVE"), "NATIVE AMERICAN")
        self.assertEqual(map_ethnicity("PATIENT DECLINED TO ANSWER"), "OTHER")
        self.assertEqual(map_ethnicity(None), "OTHER")

    def test_validate_database_environment_rejects_non_mimiciii_schema(self):
        validate_database_environment = self._get_callable("validate_database_environment")
        raw_tables, materialized_views = self._build_database_environment_inputs(
            num_admissions=50001
        )

        with self.assertRaisesRegex(ValueError, "Database schema must be mimiciii."):
            validate_database_environment(
                raw_tables,
                materialized_views,
                schema_name="mimiciv",
            )

    def test_validate_database_environment_rejects_non_postgres_flavor(self):
        validate_database_environment = self._get_callable("validate_database_environment")
        raw_tables, materialized_views = self._build_database_environment_inputs(
            num_admissions=50001
        )

        with self.assertRaisesRegex(ValueError, "Database flavor must be PostgreSQL."):
            validate_database_environment(
                raw_tables,
                materialized_views,
                database_flavor="sqlite",
            )

    def test_validate_database_environment_accepts_postgres_alias(self):
        validate_database_environment = self._get_callable("validate_database_environment")
        raw_tables, materialized_views = self._build_database_environment_inputs(
            num_admissions=50001
        )

        summary = validate_database_environment(
            raw_tables,
            materialized_views,
            database_flavor="postgres",
        )

        self.assertEqual(summary["database_flavor"], "postgres")

    def test_validate_database_environment_requires_all_required_raw_tables(self):
        validate_database_environment = self._get_callable("validate_database_environment")
        raw_tables, materialized_views = self._build_database_environment_inputs(
            num_admissions=50001
        )
        raw_tables.pop("noteevents")

        with self.assertRaisesRegex(ValueError, "Missing required raw tables: noteevents"):
            validate_database_environment(raw_tables, materialized_views)

    def test_validate_database_environment_requires_all_required_materialized_views(self):
        validate_database_environment = self._get_callable("validate_database_environment")
        raw_tables, materialized_views = self._build_database_environment_inputs(
            num_admissions=50001
        )
        materialized_views.pop("oasis")

        with self.assertRaisesRegex(ValueError, "Missing required materialized views: oasis"):
            validate_database_environment(raw_tables, materialized_views)

    def test_validate_database_environment_requires_required_raw_table_columns(self):
        validate_database_environment = self._get_callable("validate_database_environment")
        raw_tables, materialized_views = self._build_database_environment_inputs(
            num_admissions=50001
        )
        raw_tables["admissions"] = raw_tables["admissions"].drop(
            columns=["has_chartevents_data"]
        )

        with self.assertRaisesRegex(
            ValueError,
            "admissions is missing required columns: has_chartevents_data",
        ):
            validate_database_environment(raw_tables, materialized_views)

    def test_validate_database_environment_requires_required_materialized_view_columns(self):
        validate_database_environment = self._get_callable("validate_database_environment")
        raw_tables, materialized_views = self._build_database_environment_inputs(
            num_admissions=50001
        )
        materialized_views["ventdurations"] = materialized_views["ventdurations"].drop(
            columns=["endtime"]
        )

        with self.assertRaisesRegex(
            ValueError,
            "ventdurations is missing required columns: endtime",
        ):
            validate_database_environment(raw_tables, materialized_views)

    def test_validate_database_environment_requires_large_base_admissions_backbone(self):
        validate_database_environment = self._get_callable("validate_database_environment")
        raw_tables, materialized_views = self._build_database_environment_inputs(
            num_admissions=10
        )

        with self.assertRaisesRegex(ValueError, "must exceed 50,000 rows"):
            validate_database_environment(raw_tables, materialized_views)

    def test_validate_database_environment_enforces_50000_row_boundary(self):
        validate_database_environment = self._get_callable("validate_database_environment")

        raw_tables_50000, materialized_views_50000 = self._build_database_environment_inputs(
            num_admissions=50000
        )
        with self.assertRaisesRegex(ValueError, "must exceed 50,000 rows"):
            validate_database_environment(raw_tables_50000, materialized_views_50000)

        raw_tables_50001, materialized_views_50001 = self._build_database_environment_inputs(
            num_admissions=50001
        )
        summary = validate_database_environment(raw_tables_50001, materialized_views_50001)
        self.assertEqual(summary["base_admissions_rows"], 50001)

    def test_validate_database_environment_rejects_null_subject_id_in_base_backbone(self):
        validate_database_environment = self._get_callable("validate_database_environment")
        raw_tables, materialized_views = self._build_database_environment_inputs(
            num_admissions=50001
        )
        raw_tables["admissions"].loc[0, "subject_id"] = pd.NA

        with self.assertRaisesRegex(ValueError, "Base admissions contains null subject_id"):
            validate_database_environment(raw_tables, materialized_views)

    def test_validate_database_environment_rejects_null_hadm_id_in_base_backbone(self):
        validate_database_environment = self._get_callable("validate_database_environment")
        raw_tables, materialized_views = self._build_database_environment_inputs(
            num_admissions=50001
        )
        raw_tables["admissions"].loc[0, "hadm_id"] = pd.NA

        with self.assertRaisesRegex(ValueError, "Base admissions contains null hadm_id"):
            validate_database_environment(raw_tables, materialized_views)

    def test_validate_database_environment_requires_non_null_icustay_bridge_keys(self):
        validate_database_environment = self._get_callable("validate_database_environment")

        for column in ["hadm_id", "icustay_id"]:
            with self.subTest(column=column):
                raw_tables, materialized_views = self._build_database_environment_inputs(
                    num_admissions=50001
                )
                raw_tables["icustays"].loc[0, column] = pd.NA

                with self.assertRaisesRegex(
                    ValueError,
                    "icustays must provide non-null hadm_id and icustay_id",
                ):
                    validate_database_environment(raw_tables, materialized_views)

    def test_validate_database_environment_requires_accessible_note_text(self):
        validate_database_environment = self._get_callable("validate_database_environment")
        raw_tables, materialized_views = self._build_database_environment_inputs(
            num_admissions=50001
        )
        raw_tables["noteevents"]["text"] = pd.NA

        with self.assertRaisesRegex(ValueError, "noteevents.text must be accessible"):
            validate_database_environment(raw_tables, materialized_views)

    def test_validate_database_environment_requires_accessible_chartevent_values(self):
        validate_database_environment = self._get_callable("validate_database_environment")
        raw_tables, materialized_views = self._build_database_environment_inputs(
            num_admissions=50001
        )
        raw_tables["chartevents"]["value"] = pd.NA

        with self.assertRaisesRegex(ValueError, "chartevents.value must be accessible"):
            validate_database_environment(raw_tables, materialized_views)

    def test_validate_database_environment_requires_ventdurations_to_join_to_icustays(self):
        validate_database_environment = self._get_callable("validate_database_environment")
        raw_tables, materialized_views = self._build_database_environment_inputs(
            num_admissions=50001
        )
        materialized_views["ventdurations"]["icustay_id"] = 123456789

        with self.assertRaisesRegex(
            ValueError,
            "ventdurations must join to icustays through icustay_id",
        ):
            validate_database_environment(raw_tables, materialized_views)

    def test_validate_database_environment_requires_vasopressors_to_join_to_icustays(self):
        validate_database_environment = self._get_callable("validate_database_environment")
        raw_tables, materialized_views = self._build_database_environment_inputs(
            num_admissions=50001
        )
        materialized_views["vasopressordurations"]["icustay_id"] = 123456789

        with self.assertRaisesRegex(
            ValueError,
            "vasopressordurations must join to icustays through icustay_id",
        ):
            validate_database_environment(raw_tables, materialized_views)

    def test_validate_database_environment_requires_chartevents_to_join_to_d_items(self):
        validate_database_environment = self._get_callable("validate_database_environment")
        raw_tables, materialized_views = self._build_database_environment_inputs(
            num_admissions=50001
        )
        raw_tables["chartevents"]["itemid"] = 999999

        with self.assertRaisesRegex(
            ValueError,
            "chartevents must join to d_items through itemid",
        ):
            validate_database_environment(raw_tables, materialized_views)

    def test_validate_database_environment_requires_nonempty_admission_level_acuity(self):
        validate_database_environment = self._get_callable("validate_database_environment")
        raw_tables, materialized_views = self._build_database_environment_inputs(
            num_admissions=50001
        )
        materialized_views["oasis"] = pd.DataFrame(
            columns=["hadm_id", "icustay_id", "oasis"]
        )
        materialized_views["sapsii"] = pd.DataFrame(
            columns=["hadm_id", "icustay_id", "sapsii"]
        )

        with self.assertRaisesRegex(
            ValueError,
            "oasis and sapsii must join back to admissions on hadm_id",
        ):
            validate_database_environment(raw_tables, materialized_views)

    def test_validate_database_environment_returns_summary_for_valid_inputs(self):
        validate_database_environment = self._get_callable("validate_database_environment")
        raw_tables, materialized_views = self._build_database_environment_inputs(
            num_admissions=50001
        )

        summary = validate_database_environment(raw_tables, materialized_views)

        self.assertEqual(summary["database_flavor"], "postgresql")
        self.assertEqual(summary["schema_name"], "mimiciii")
        self.assertEqual(summary["base_admissions_rows"], 50001)
        self.assertEqual(
            summary["raw_tables"],
            ["admissions", "chartevents", "d_items", "icustays", "noteevents", "patients"],
        )
        self.assertEqual(
            summary["materialized_views"],
            ["oasis", "sapsii", "vasopressordurations", "ventdurations"],
        )
        self.assertTrue(summary["supports_multiple_icustays_per_hadm"])

    def test_validate_database_environment_reports_when_multiple_icustays_are_absent(self):
        validate_database_environment = self._get_callable("validate_database_environment")
        raw_tables, materialized_views = self._build_database_environment_inputs(
            num_admissions=50001,
            include_multiple_icustays=False,
        )

        summary = validate_database_environment(raw_tables, materialized_views)

        self.assertFalse(summary["supports_multiple_icustays_per_hadm"])

    def test_map_insurance_matches_required_categories(self):
        map_insurance = self._get_callable("map_insurance")
        self.assertEqual(map_insurance("Medicare"), "Public")
        self.assertEqual(map_insurance("Medicaid"), "Public")
        self.assertEqual(map_insurance("Government"), "Public")
        self.assertEqual(map_insurance("Private"), "Private")
        self.assertEqual(map_insurance("Self Pay"), "Self-Pay")

    def test_build_base_admissions_raises_clear_error_when_required_columns_are_missing(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        admissions_missing = self.admissions.drop(columns=["has_chartevents_data"])
        with self.assertRaisesRegex(ValueError, "has_chartevents_data"):
            build_base_admissions(admissions_missing, self.patients)

    def test_build_base_admissions_filters_has_chartevents_and_joins_patients(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        base = build_base_admissions(self.admissions, self.patients)
        self.assertIsInstance(base, pd.DataFrame)
        self.assertIn("gender", base.columns)
        self.assertIn("dob", base.columns)
        self.assertNotIn(105, set(base["hadm_id"]))
        self.assertEqual(len(base), 7)
        self._assert_hadm_unique(base, "Base admissions")

    def test_build_base_admissions_rejects_duplicate_patient_rows_for_subject_id(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        duplicated_patients = pd.concat(
            [
                self.patients,
                pd.DataFrame(
                    [
                        {
                            "subject_id": 1,
                            "gender": "F",
                            "dob": "2071-01-01 00:00:00",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        with self.assertRaises(pd.errors.MergeError):
            build_base_admissions(self.admissions, duplicated_patients)

    def test_build_demographics_table_applies_age_los_race_and_insurance_rules(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        build_demographics_table = self._get_callable("build_demographics_table")
        base = build_base_admissions(self.admissions, self.patients)
        demographics = build_demographics_table(base)

        required = {
            "hadm_id",
            "subject_id",
            "race",
            "age",
            "los_hours",
            "los_days",
            "insurance",
            "gender",
        }
        self.assertTrue(required.issubset(set(demographics.columns)))

        by_hadm = demographics.set_index("hadm_id")
        self.assertEqual(by_hadm.loc[101, "race"], "BLACK")
        self.assertEqual(by_hadm.loc[103, "race"], "ASIAN")
        self.assertEqual(by_hadm.loc[104, "race"], "OTHER")
        self.assertEqual(by_hadm.loc[101, "age"], 90.0)
        self.assertAlmostEqual(by_hadm.loc[103, "los_hours"], 20.0)
        self.assertAlmostEqual(by_hadm.loc[106, "los_days"], 30.0 / 24.0)
        self.assertEqual(by_hadm.loc[104, "insurance"], "Public")
        self.assertEqual(by_hadm.loc[106, "insurance"], "Private")
        self._assert_hadm_unique(demographics, "Demographics table")

    def test_build_eol_cohort_enforces_los_filter_and_discharge_priority(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        build_demographics_table = self._get_callable("build_demographics_table")
        build_eol_cohort = self._get_callable("build_eol_cohort")

        base = build_base_admissions(self.admissions, self.patients)
        demographics = build_demographics_table(base)
        eol = build_eol_cohort(base, demographics)

        self.assertIsInstance(eol, pd.DataFrame)
        self.assertEqual(set(eol["hadm_id"]), {101, 103, 104, 107})

        by_hadm = eol.set_index("hadm_id")
        self.assertEqual(by_hadm.loc[101, "discharge_category"], "Hospice")
        self.assertEqual(by_hadm.loc[103, "discharge_category"], "Skilled Nursing Facility")
        self.assertEqual(by_hadm.loc[104, "discharge_category"], "Deceased")
        self.assertEqual(
            by_hadm.loc[107, "discharge_category"],
            "Deceased",
            msg="Death must take priority over hospice when both indicators are present",
        )
        self.assertNotIn(102, set(eol["hadm_id"]))
        self._assert_hadm_unique(eol, "EOL cohort")

    def test_build_eol_cohort_enforces_exact_six_hour_boundary(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        build_demographics_table = self._get_callable("build_demographics_table")
        build_eol_cohort = self._get_callable("build_eol_cohort")

        admissions = pd.DataFrame(
            [
                {
                    "hadm_id": 891,
                    "subject_id": 891,
                    "admittime": "2100-09-01 00:00:00",
                    "dischtime": "2100-09-01 06:00:00",
                    "ethnicity": "WHITE",
                    "insurance": "Medicare",
                    "discharge_location": "HOME HOSPICE",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 892,
                    "subject_id": 892,
                    "admittime": "2100-09-01 00:00:00",
                    "dischtime": "2100-09-01 05:59:00",
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
                {"subject_id": 891, "gender": "M", "dob": "2070-09-01 00:00:00"},
                {"subject_id": 892, "gender": "F", "dob": "2070-09-01 00:00:00"},
            ]
        )

        base = build_base_admissions(admissions, patients)
        demographics = build_demographics_table(base)
        eol = build_eol_cohort(base, demographics)

        self.assertIn(891, set(eol["hadm_id"]))
        self.assertNotIn(892, set(eol["hadm_id"]))

    def test_build_eol_cohort_accepts_snf_discharge_text(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        build_demographics_table = self._get_callable("build_demographics_table")
        build_eol_cohort = self._get_callable("build_eol_cohort")

        admissions = pd.DataFrame(
            [
                {
                    "hadm_id": 901,
                    "subject_id": 91,
                    "admittime": "2100-09-01 00:00:00",
                    "dischtime": "2100-09-01 12:00:00",
                    "ethnicity": "WHITE",
                    "insurance": "Medicare",
                    "discharge_location": "SNF",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                }
            ]
        )
        patients = pd.DataFrame(
            [
                {"subject_id": 91, "gender": "M", "dob": "2070-09-01 00:00:00"},
            ]
        )

        base = build_base_admissions(admissions, patients)
        demographics = build_demographics_table(base)
        eol = build_eol_cohort(base, demographics)

        self.assertEqual(set(eol["hadm_id"]), {901})

    def test_build_all_cohort_requires_a_single_icu_stay_of_at_least_12_hours(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        build_all_cohort = self._get_callable("build_all_cohort")

        base = build_base_admissions(self.admissions, self.patients)
        all_cohort = build_all_cohort(base, self.icustays)

        self.assertIsInstance(all_cohort, pd.DataFrame)
        self.assertEqual(set(all_cohort["hadm_id"]), {101, 103, 106, 107})
        self.assertNotIn(
            100,
            set(all_cohort["hadm_id"]),
            msg="Two 11-hour ICU stays must not qualify; at least one stay must be >= 12 hours",
        )
        self.assertNotIn(
            105,
            set(all_cohort["hadm_id"]),
            msg="Admissions excluded by has_chartevents_data should stay excluded downstream",
        )
        self._assert_hadm_unique(all_cohort, "ALL cohort")

    def test_build_all_cohort_remains_unique_when_multiple_qualifying_icu_stays_exist(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        build_all_cohort = self._get_callable("build_all_cohort")

        admissions = pd.DataFrame(
            [
                {
                    "hadm_id": 801,
                    "subject_id": 81,
                    "admittime": "2100-08-01 00:00:00",
                    "dischtime": "2100-08-02 00:00:00",
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
                {"subject_id": 81, "gender": "M", "dob": "2070-08-01 00:00:00"},
            ]
        )
        icustays = pd.DataFrame(
            [
                {
                    "hadm_id": 801,
                    "icustay_id": 8011,
                    "intime": "2100-08-01 00:00:00",
                    "outtime": "2100-08-01 13:00:00",
                },
                {
                    "hadm_id": 801,
                    "icustay_id": 8012,
                    "intime": "2100-08-01 14:00:00",
                    "outtime": "2100-08-02 04:00:00",
                },
            ]
        )

        base = build_base_admissions(admissions, patients)
        all_cohort = build_all_cohort(base, icustays)

        self.assertEqual(list(all_cohort["hadm_id"]), [801])
        self._assert_hadm_unique(all_cohort, "ALL cohort with multiple qualifying ICU stays")

    def test_build_treatment_totals_raises_clear_error_on_missing_required_columns(self):
        build_treatment_totals = self._get_callable("build_treatment_totals")
        vent_missing = self.ventdurations.drop(columns=["starttime"])
        with self.assertRaisesRegex(ValueError, "starttime"):
            build_treatment_totals(self.icustays, vent_missing, self.vasopressordurations)

    def test_build_treatment_totals_merges_overlapping_and_short_gap_spans(self):
        build_treatment_totals = self._get_callable("build_treatment_totals")
        totals = build_treatment_totals(
            self.icustays,
            self.ventdurations,
            self.vasopressordurations,
        )

        self.assertIsInstance(totals, pd.DataFrame)
        self.assertTrue({"hadm_id", "total_vent_min", "total_vaso_min"}.issubset(totals.columns))

        by_hadm = totals.fillna(0).set_index("hadm_id")
        self.assertEqual(
            by_hadm.loc[101, "total_vent_min"],
            810.0,
            msg="Vent spans with a gap <= 600 minutes must merge before summing by hadm_id",
        )
        self.assertEqual(
            by_hadm.loc[103, "total_vaso_min"],
            840.0,
            msg="Overlapping vasopressor spans and gaps <= 600 minutes must merge into one span",
        )
        self._assert_hadm_unique(totals, "Treatment totals")

    def test_build_treatment_totals_uses_icustay_bridge_and_respects_600_minute_boundary(self):
        build_treatment_totals = self._get_callable("build_treatment_totals")
        icustays = pd.DataFrame(
            [
                {"hadm_id": 200, "icustay_id": 2001, "intime": "2100-09-01 00:00:00", "outtime": "2100-09-01 12:00:00"},
                {"hadm_id": 200, "icustay_id": 2002, "intime": "2100-09-01 20:00:00", "outtime": "2100-09-02 04:00:00"},
            ]
        )
        ventdurations = pd.DataFrame(
            [
                {
                    "hadm_id": 999,
                    "icustay_id": 2001,
                    "ventnum": 1,
                    "starttime": "2100-09-01 00:00:00",
                    "endtime": "2100-09-01 01:00:00",
                    "duration_hours": 1.0,
                },
                {
                    "hadm_id": 999,
                    "icustay_id": 2001,
                    "ventnum": 2,
                    "starttime": "2100-09-01 11:00:00",
                    "endtime": "2100-09-01 12:00:00",
                    "duration_hours": 1.0,
                },
                {
                    "hadm_id": 999,
                    "icustay_id": 2002,
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
        totals = build_treatment_totals(icustays, ventdurations, empty_vaso)
        row = totals.fillna(0).set_index("hadm_id").loc[200]
        self.assertEqual(
            row["total_vent_min"],
            780.0,
            msg="Gap == 600 must merge, gap == 601 must not merge, and hadm_id must be derived from icustays",
        )

    def test_prepare_note_text_for_sentiment_collapses_whitespace_only(self):
        prepare_note_text_for_sentiment = self._get_callable("prepare_note_text_for_sentiment")
        cleaned = prepare_note_text_for_sentiment(
            "Patient\trefuses\n\n treatment   Date:[**5-1-18**]"
        )
        self.assertEqual(
            cleaned,
            "Patient refuses treatment Date:[**5-1-18**]",
            msg="Whitespace tokenization should collapse whitespace without stripping de-identification markers",
        )
        self.assertEqual(prepare_note_text_for_sentiment("   \n\t "), "")
        self.assertEqual(prepare_note_text_for_sentiment(None), "")

    def test_build_note_corpus_concatenates_non_error_notes_and_can_include_missing_admissions(self):
        build_note_corpus = self._get_callable("build_note_corpus")
        corpus = build_note_corpus(
            self.noteevents,
            all_hadm_ids=[101, 103, 104, 106, 107],
        )

        self.assertIsInstance(corpus, pd.DataFrame)
        self.assertTrue({"hadm_id", "note_text"}.issubset(corpus.columns))
        self._assert_hadm_unique(corpus, "Note corpus")

        by_hadm = corpus.set_index("hadm_id")
        self.assertEqual(set(corpus["hadm_id"]), {101, 103, 104, 106, 107})
        self.assertIn("Patient refuses treatment", by_hadm.loc[101, "note_text"])
        self.assertIn("Autopsy was discussed", by_hadm.loc[101, "note_text"])
        self.assertNotIn(
            "AUTOPSY requested",
            by_hadm.loc[104, "note_text"],
            msg="Error notes must be excluded before admission-level concatenation",
        )
        self.assertEqual(
            by_hadm.loc[107, "note_text"],
            "",
            msg="Admissions without notes should still appear when all_hadm_ids is provided",
        )

    def test_build_note_corpus_filters_out_notes_for_hadm_ids_outside_requested_all_cohort(self):
        build_note_corpus = self._get_callable("build_note_corpus")
        notes = pd.concat(
            [
                self.noteevents,
                pd.DataFrame(
                    [
                        {
                            "hadm_id": 999,
                            "category": "Nursing",
                            "text": "Outside cohort note that should be dropped.",
                            "iserror": 0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        corpus = build_note_corpus(notes, all_hadm_ids=[101, 103, 104])

        self.assertEqual(set(corpus["hadm_id"]), {101, 103, 104})
        self.assertNotIn(999, set(corpus["hadm_id"]))

    def test_build_note_corpus_preserves_empty_strings_after_left_join(self):
        build_note_corpus = self._get_callable("build_note_corpus")
        corpus = build_note_corpus(self.noteevents, all_hadm_ids=[101, 103, 999]).set_index(
            "hadm_id"
        )
        self.assertEqual(corpus.loc[999, "note_text"], "")
        self.assertFalse(pd.isna(corpus.loc[999, "note_text"]))

    def test_build_note_corpus_raises_clear_error_when_required_columns_are_missing(self):
        build_note_corpus = self._get_callable("build_note_corpus")
        notes_missing = self.noteevents.drop(columns=["text"])
        with self.assertRaisesRegex(ValueError, "text"):
            build_note_corpus(notes_missing)

    def test_build_note_labels_ignores_error_notes_and_extracts_rule_based_labels(self):
        build_note_labels = self._get_callable("build_note_labels")
        labels = build_note_labels(self.noteevents)

        self.assertIsInstance(labels, pd.DataFrame)
        self.assertTrue(
            {"hadm_id", "noncompliance_label", "autopsy_label"}.issubset(labels.columns)
        )

        by_hadm = labels.set_index("hadm_id")
        self.assertEqual(by_hadm.loc[101, "noncompliance_label"], 1)
        self.assertEqual(by_hadm.loc[101, "autopsy_label"], 1)
        self.assertEqual(by_hadm.loc[103, "noncompliance_label"], 0)
        self.assertEqual(by_hadm.loc[104, "autopsy_label"], 0)
        self.assertEqual(by_hadm.loc[106, "noncompliance_label"], 1)
        self._assert_hadm_unique(labels, "Note labels")

    def test_build_note_labels_can_include_all_hadm_ids_with_zero_defaults(self):
        build_note_labels = self._get_callable("build_note_labels")
        labels = build_note_labels(
            self.noteevents,
            all_hadm_ids=[101, 103, 104, 106, 107],
        )
        self._assert_hadm_unique(labels, "Note labels with all admissions")
        by_hadm = labels.set_index("hadm_id")
        self.assertEqual(set(labels["hadm_id"]), {101, 103, 104, 106, 107})
        self.assertEqual(by_hadm.loc[107, "noncompliance_label"], 0)
        self.assertEqual(by_hadm.loc[107, "autopsy_label"], 0)

    def test_build_note_labels_raises_clear_error_when_required_columns_are_missing(self):
        build_note_labels = self._get_callable("build_note_labels")
        notes_missing = self.noteevents.drop(columns=["iserror"])
        with self.assertRaisesRegex(ValueError, "iserror"):
            build_note_labels(notes_missing)

    def test_build_note_labels_avoids_simple_false_positives(self):
        build_note_labels = self._get_callable("build_note_labels")
        notes = pd.concat(
            [
                self.noteevents,
                pd.DataFrame(
                    [
                        {
                            "hadm_id": 108,
                            "category": "Nursing",
                            "text": "Medication compliance reviewed with patient. No autopsy planned.",
                            "iserror": 0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        labels = build_note_labels(notes).set_index("hadm_id")
        self.assertEqual(
            labels.loc[108, "noncompliance_label"],
            0,
            msg="Substring rules should not fire on generic compliance mentions",
        )
        self.assertEqual(
            labels.loc[108, "autopsy_label"],
            1,
            msg="Autopsy matching should be case-insensitive and based on substring presence",
        )

    def test_build_note_labels_matches_hyphenated_noncompliance_phrases(self):
        build_note_labels = self._get_callable("build_note_labels")
        notes = pd.DataFrame(
            [
                {
                    "hadm_id": 201,
                    "category": "Nursing",
                    "text": "Patient is non-complian with medications.",
                    "iserror": 0,
                },
                {
                    "hadm_id": 202,
                    "category": "Nursing",
                    "text": "Patient remains non-adher to treatment plan.",
                    "iserror": 0,
                },
            ]
        )

        labels = build_note_labels(notes).set_index("hadm_id")
        self.assertEqual(labels.loc[201, "noncompliance_label"], 1)
        self.assertEqual(labels.loc[202, "noncompliance_label"], 1)

    def test_build_note_labels_matches_literal_noncompliance_and_noncompliant_terms(self):
        build_note_labels = self._get_callable("build_note_labels")
        notes = pd.DataFrame(
            [
                {
                    "hadm_id": 211,
                    "category": "Nursing",
                    "text": "Team documented ongoing noncompliance with medications.",
                    "iserror": 0,
                },
                {
                    "hadm_id": 212,
                    "category": "Nursing",
                    "text": "Patient was described as noncompliant during rounds.",
                    "iserror": 0,
                },
            ]
        )

        labels = build_note_labels(notes).set_index("hadm_id")
        self.assertEqual(labels.loc[211, "noncompliance_label"], 1)
        self.assertEqual(labels.loc[212, "noncompliance_label"], 1)

    def test_identify_table2_itemids_discovers_matching_labels_across_dbsources(self):
        identify_table2_itemids = self._get_callable("identify_table2_itemids")
        d_items = pd.DataFrame(
            [
                {"itemid": 10, "label": "Education Readiness", "dbsource": "carevue"},
                {"itemid": 11, "label": "Education Readiness", "dbsource": "metavision"},
                {"itemid": 12, "label": "Pain Level", "dbsource": "carevue"},
                {"itemid": 13, "label": "Follows Commands", "dbsource": "metavision"},
                {"itemid": 14, "label": "Completely Unrelated Label", "dbsource": "carevue"},
            ]
        )
        itemids = identify_table2_itemids(d_items)
        self.assertIsInstance(itemids, (set, list, tuple))
        itemids = set(itemids)
        self.assertTrue({10, 11, 12, 13}.issubset(itemids))
        self.assertNotIn(
            14,
            itemids,
            msg="Only Table 2 concepts should be selected from d_items",
        )

    def test_identify_table2_itemids_supports_case_insensitive_partial_label_matching(self):
        identify_table2_itemids = self._get_callable("identify_table2_itemids")
        d_items = pd.DataFrame(
            [
                {
                    "itemid": 20,
                    "label": "Richmond-RAS Scale Assessment",
                    "dbsource": "carevue",
                },
                {
                    "itemid": 21,
                    "label": "pain level verbal response",
                    "dbsource": "metavision",
                },
                {
                    "itemid": 22,
                    "label": "SOCIAL WORK CONSULT NOTE",
                    "dbsource": "carevue",
                },
                {
                    "itemid": 23,
                    "label": "Completely unrelated field",
                    "dbsource": "metavision",
                },
            ]
        )

        itemids = set(identify_table2_itemids(d_items))
        self.assertIn(20, itemids)
        self.assertIn(21, itemids)
        self.assertIn(22, itemids)
        self.assertNotIn(23, itemids)

    def test_identify_table2_itemids_raises_clear_error_when_required_columns_are_missing(self):
        identify_table2_itemids = self._get_callable("identify_table2_itemids")
        d_items_missing = self.d_items.drop(columns=["label"])
        with self.assertRaisesRegex(ValueError, "label"):
            identify_table2_itemids(d_items_missing)

    def test_build_chartevent_feature_matrix_creates_binary_label_value_features(self):
        build_chartevent_feature_matrix = self._get_callable("build_chartevent_feature_matrix")
        feature_matrix = build_chartevent_feature_matrix(
            self.chartevents,
            self.d_items,
            allowed_labels={"Education Readiness", "Pain Level"},
        )

        self.assertIsInstance(feature_matrix, pd.DataFrame)
        expected_columns = {
            "hadm_id",
            "Education Readiness: No",
            "Education Readiness: Yes",
            "Pain Level: 7-Mod to Severe",
        }
        self.assertTrue(expected_columns.issubset(set(feature_matrix.columns)))

        by_hadm = feature_matrix.fillna(0).set_index("hadm_id")
        self.assertEqual(by_hadm.loc[101, "Education Readiness: No"], 1)
        self.assertEqual(by_hadm.loc[101, "Pain Level: 7-Mod to Severe"], 1)
        self.assertEqual(by_hadm.loc[103, "Education Readiness: Yes"], 1)
        self.assertEqual(
            by_hadm.loc[101, "Education Readiness: No"],
            1,
            msg="Repeated charted values must stay binary at the admission level",
        )
        self._assert_hadm_unique(feature_matrix, "Chartevent feature matrix")

    def test_build_chartevent_feature_matrix_can_preserve_all_hadm_ids_with_zero_rows(self):
        build_chartevent_feature_matrix = self._get_callable("build_chartevent_feature_matrix")
        feature_matrix = build_chartevent_feature_matrix(
            self.chartevents,
            self.d_items,
            allowed_labels={"Education Readiness", "Pain Level"},
            all_hadm_ids=[101, 103, 106, 107],
        ).fillna(0)
        self._assert_hadm_unique(feature_matrix, "Chartevent feature matrix with all admissions")
        self.assertEqual(set(feature_matrix["hadm_id"]), {101, 103, 106, 107})
        zero_row = feature_matrix.set_index("hadm_id").loc[106]
        self.assertTrue(
            (zero_row == 0).all(),
            msg="Admissions without matching chart features should still appear as all-zero rows when all_hadm_ids is provided",
        )

    def test_build_chartevent_feature_matrix_normalizes_values_and_ignores_blank_entries(self):
        build_chartevent_feature_matrix = self._get_callable("build_chartevent_feature_matrix")
        chartevents = pd.concat(
            [
                self.chartevents,
                pd.DataFrame(
                    [
                        {"hadm_id": 103, "itemid": 1, "value": " No ", "icustay_id": 1031},
                        {"hadm_id": 103, "itemid": 2, "value": "", "icustay_id": 1031},
                        {"hadm_id": 103, "itemid": 2, "value": None, "icustay_id": 1031},
                    ]
                ),
            ],
            ignore_index=True,
        )
        feature_matrix = build_chartevent_feature_matrix(
            chartevents,
            self.d_items,
            allowed_labels={"Education Readiness", "Pain Level"},
        ).fillna(0).set_index("hadm_id")
        self.assertIn(
            "Education Readiness: No",
            feature_matrix.columns,
            msg="Feature columns must preserve the required 'label: value' naming scheme",
        )
        self.assertEqual(
            feature_matrix.loc[103, "Education Readiness: No"],
            1,
            msg="Value normalization should trim whitespace and lowercase to stable feature keys",
        )
        self.assertNotIn(
            "Pain Level: ",
            set(feature_matrix.columns),
            msg="Blank values should not produce empty feature columns",
        )

    def test_build_chartevent_feature_matrix_deduplicates_repeated_pairs_to_one_binary_value(self):
        build_chartevent_feature_matrix = self._get_callable("build_chartevent_feature_matrix")
        chartevents = pd.DataFrame(
            [
                {"hadm_id": 301, "itemid": 1, "value": "No", "icustay_id": 3011},
                {"hadm_id": 301, "itemid": 1, "value": "No", "icustay_id": 3011},
                {"hadm_id": 301, "itemid": 1, "value": "No", "icustay_id": 3011},
            ]
        )
        d_items = pd.DataFrame(
            [
                {"itemid": 1, "label": "Education Readiness", "dbsource": "carevue"},
            ]
        )

        feature_matrix = build_chartevent_feature_matrix(
            chartevents,
            d_items,
            allowed_labels={"Education Readiness"},
        ).set_index("hadm_id")

        self.assertIn(
            "Education Readiness: No",
            feature_matrix.columns,
            msg="Repeated label/value pairs must map into the required single binary feature column",
        )
        self.assertEqual(feature_matrix.loc[301, "Education Readiness: No"], 1)

    def test_build_chartevent_feature_matrix_preserves_rare_single_occurrence_features(self):
        build_chartevent_feature_matrix = self._get_callable("build_chartevent_feature_matrix")
        chartevents = pd.DataFrame(
            [
                {"hadm_id": 401, "itemid": 41, "value": "Yes", "icustay_id": 4011},
            ]
        )
        d_items = pd.DataFrame(
            [
                {"itemid": 41, "label": "Family Meeting", "dbsource": "carevue"},
            ]
        )

        feature_matrix = build_chartevent_feature_matrix(
            chartevents,
            d_items,
            allowed_labels={"Family Meeting"},
            all_hadm_ids=[401, 402],
        ).fillna(0).set_index("hadm_id")

        self.assertIn(
            "Family Meeting: Yes",
            feature_matrix.columns,
            msg="Rare one-off chart-event features must not be pruned from the matrix",
        )
        self.assertEqual(feature_matrix.loc[401, "Family Meeting: Yes"], 1)
        self.assertEqual(feature_matrix.loc[402, "Family Meeting: Yes"], 0)

    def test_build_chartevent_feature_matrix_outputs_binary_values_under_duplicates_and_missing_rows(self):
        build_chartevent_feature_matrix = self._get_callable("build_chartevent_feature_matrix")
        chartevents = pd.DataFrame(
            [
                {"hadm_id": 501, "itemid": 1, "value": "No", "icustay_id": 5011},
                {"hadm_id": 501, "itemid": 1, "value": "No", "icustay_id": 5011},
                {"hadm_id": 501, "itemid": 2, "value": "7-Mod to Severe", "icustay_id": 5011},
                {"hadm_id": 502, "itemid": 2, "value": None, "icustay_id": 5021},
            ]
        )
        feature_matrix = build_chartevent_feature_matrix(
            chartevents,
            self.d_items,
            allowed_labels={"Education Readiness", "Pain Level"},
            all_hadm_ids=[501, 502, 503],
        ).fillna(0)

        feature_columns = [column for column in feature_matrix.columns if column != "hadm_id"]
        self.assertTrue(feature_columns)
        self.assertTrue(feature_matrix[feature_columns].isin([0, 1]).all().all())

    def test_build_chartevent_feature_matrix_raises_clear_error_when_required_columns_are_missing(self):
        build_chartevent_feature_matrix = self._get_callable("build_chartevent_feature_matrix")
        chartevents_missing = self.chartevents.drop(columns=["itemid"])
        with self.assertRaisesRegex(ValueError, "itemid"):
            build_chartevent_feature_matrix(chartevents_missing, self.d_items)

    def test_z_normalize_scores_standardizes_each_metric_independently(self):
        z_normalize_scores = self._get_callable("z_normalize_scores")
        raw = pd.DataFrame(
            [
                {"hadm_id": 1, "noncompliance_score": 1.0, "autopsy_score": 10.0, "negative_sentiment_score": -0.2},
                {"hadm_id": 2, "noncompliance_score": 2.0, "autopsy_score": 20.0, "negative_sentiment_score": 0.0},
                {"hadm_id": 3, "noncompliance_score": 3.0, "autopsy_score": 30.0, "negative_sentiment_score": 0.2},
            ]
        )
        normalized = z_normalize_scores(
            raw,
            columns=[
                "noncompliance_score",
                "autopsy_score",
                "negative_sentiment_score",
            ],
        )

        for col in [
            "noncompliance_score",
            "autopsy_score",
            "negative_sentiment_score",
        ]:
            self.assertAlmostEqual(normalized[col].mean(), 0.0, places=7)
            self.assertAlmostEqual(normalized[col].std(ddof=0), 1.0, places=7)

    def test_z_normalize_scores_returns_zero_for_zero_variance_columns(self):
        z_normalize_scores = self._get_callable("z_normalize_scores")
        raw = pd.DataFrame(
            [
                {"hadm_id": 1, "noncompliance_score": 5.0},
                {"hadm_id": 2, "noncompliance_score": 5.0},
                {"hadm_id": 3, "noncompliance_score": 5.0},
            ]
        )
        normalized = z_normalize_scores(raw, columns=["noncompliance_score"])
        self.assertTrue((normalized["noncompliance_score"] == 0.0).all())

    def test_build_acuity_scores_produces_unique_admission_level_table(self):
        build_acuity_scores = self._get_callable("build_acuity_scores")
        acuity = build_acuity_scores(self.oasis, self.sapsii)
        self.assertIsInstance(acuity, pd.DataFrame)
        self.assertTrue({"hadm_id", "oasis", "sapsii"}.issubset(acuity.columns))
        self._assert_hadm_unique(acuity, "Acuity scores")
        self.assertEqual(set(acuity["hadm_id"]), {101, 103, 106, 107})

    def test_build_acuity_scores_raises_clear_error_when_required_columns_are_missing(self):
        build_acuity_scores = self._get_callable("build_acuity_scores")
        oasis_missing = self.oasis.drop(columns=["oasis"])
        with self.assertRaisesRegex(ValueError, "oasis"):
            build_acuity_scores(oasis_missing, self.sapsii)

    def test_build_acuity_scores_uses_max_when_multiple_icu_stays_share_a_hadm_id(self):
        build_acuity_scores = self._get_callable("build_acuity_scores")
        oasis = pd.DataFrame(
            [
                {"hadm_id": 101, "icustay_id": 1011, "oasis": 15},
                {"hadm_id": 101, "icustay_id": 1012, "oasis": 22},
            ]
        )
        sapsii = pd.DataFrame(
            [
                {"hadm_id": 101, "icustay_id": 1011, "sapsii": 42},
                {"hadm_id": 101, "icustay_id": 1012, "sapsii": 50},
            ]
        )

        acuity = build_acuity_scores(oasis, sapsii).set_index("hadm_id")
        self.assertEqual(acuity.loc[101, "oasis"], 22)
        self.assertEqual(acuity.loc[101, "sapsii"], 50)

    def test_build_proxy_probability_scores_fits_estimator_and_uses_predict_proba_output(self):
        build_proxy_probability_scores = self._get_callable("build_proxy_probability_scores")
        feature_matrix = pd.DataFrame(
            [
                {"hadm_id": 101, "feature_a": 1, "feature_b": 1},
                {"hadm_id": 103, "feature_a": 0, "feature_b": 0},
                {"hadm_id": 106, "feature_a": 1, "feature_b": 0},
                {"hadm_id": 107, "feature_a": 0, "feature_b": 0},
            ]
        )
        note_labels = pd.DataFrame(
            [
                {"hadm_id": 101, "noncompliance_label": 1},
                {"hadm_id": 103, "noncompliance_label": 0},
                {"hadm_id": 106, "noncompliance_label": 1},
                {"hadm_id": 107, "noncompliance_label": 0},
            ]
        )
        created = []

        def estimator_factory():
            estimator = _FakeProbEstimator([0.9, 0.2, 0.8, 0.1])
            created.append(estimator)
            return estimator

        scores = build_proxy_probability_scores(
            feature_matrix=feature_matrix,
            note_labels=note_labels,
            label_column="noncompliance_label",
            estimator_factory=estimator_factory,
        )

        self.assertEqual(len(created), 1)
        self.assertTrue(created[0].was_fit)
        self.assertEqual(list(created[0].fit_y), [1, 0, 1, 0])
        self.assertNotIn(
            "hadm_id",
            set(created[0].fit_X.columns),
            msg="hadm_id must not be used as a predictive feature",
        )
        self._assert_hadm_unique(scores, "Proxy probability scores")
        by_hadm = scores.set_index("hadm_id")
        self.assertAlmostEqual(by_hadm.loc[101, "noncompliance_score"], 0.9)
        self.assertAlmostEqual(by_hadm.loc[103, "noncompliance_score"], 0.2)
        self.assertAlmostEqual(by_hadm.loc[106, "noncompliance_score"], 0.8)
        self.assertAlmostEqual(by_hadm.loc[107, "noncompliance_score"], 0.1)

    def test_build_proxy_probability_scores_uses_l1_liblinear_logistic_regression_by_default(self):
        build_proxy_probability_scores = self._get_callable("build_proxy_probability_scores")
        feature_matrix = pd.DataFrame(
            [
                {"hadm_id": 101, "feature_a": 1},
                {"hadm_id": 103, "feature_a": 0},
            ]
        )
        note_labels = pd.DataFrame(
            [
                {"hadm_id": 101, "noncompliance_label": 1},
                {"hadm_id": 103, "noncompliance_label": 0},
            ]
        )

        created = []

        class _RecordingLogisticRegression:
            def __init__(self, *args, **kwargs):
                created.append(kwargs)

            def fit(self, X, y):
                return self

            def predict_proba(self, X):
                return [[0.1, 0.9] for _ in range(len(X))]

        with patch.object(self.module, "LogisticRegression", _RecordingLogisticRegression):
            build_proxy_probability_scores(
                feature_matrix=feature_matrix,
                note_labels=note_labels,
                label_column="noncompliance_label",
            )

        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].get("penalty"), "l1")
        self.assertEqual(created[0].get("solver"), "liblinear")
        self.assertEqual(created[0].get("max_iter"), 1000)

    def test_build_proxy_probability_scores_sorts_by_hadm_and_aligns_features_with_labels(self):
        build_proxy_probability_scores = self._get_callable("build_proxy_probability_scores")
        feature_matrix = pd.DataFrame(
            [
                {"hadm_id": 106, "feature_a": 0, "feature_b": 1},
                {"hadm_id": 101, "feature_a": 1, "feature_b": 0},
                {"hadm_id": 103, "feature_a": 0, "feature_b": 0},
            ]
        )
        note_labels = pd.DataFrame(
            [
                {"hadm_id": 103, "noncompliance_label": 0},
                {"hadm_id": 101, "noncompliance_label": 1},
                {"hadm_id": 106, "noncompliance_label": 1},
            ]
        )
        created = []

        def estimator_factory():
            estimator = _FakeProbEstimator([0.9, 0.2, 0.8])
            created.append(estimator)
            return estimator

        scores = build_proxy_probability_scores(
            feature_matrix=feature_matrix,
            note_labels=note_labels,
            label_column="noncompliance_label",
            estimator_factory=estimator_factory,
        )

        self.assertEqual(list(created[0].fit_X["feature_a"]), [1, 0, 0])
        self.assertEqual(list(created[0].fit_X["feature_b"]), [0, 0, 1])
        self.assertEqual(list(created[0].fit_y), [1, 0, 1])
        self.assertEqual(list(scores["hadm_id"]), [101, 103, 106])

    def test_build_proxy_probability_scores_raises_clear_error_when_required_columns_are_missing(self):
        build_proxy_probability_scores = self._get_callable("build_proxy_probability_scores")
        note_labels_missing = pd.DataFrame([{"hadm_id": 101}])
        feature_matrix = pd.DataFrame([{"hadm_id": 101, "feature_a": 1}])
        with self.assertRaisesRegex(ValueError, "noncompliance_label"):
            build_proxy_probability_scores(
                feature_matrix=feature_matrix,
                note_labels=note_labels_missing,
                label_column="noncompliance_label",
            )

    def test_build_negative_sentiment_scores_negates_sentiment_polarity(self):
        build_negative_sentiment_scores = self._get_callable("build_negative_sentiment_scores")
        note_corpus = pd.DataFrame(
            [
                {"hadm_id": 101, "note_text": "very negative"},
                {"hadm_id": 103, "note_text": "neutral"},
                {"hadm_id": 106, "note_text": "positive"},
            ]
        )
        polarity_map = {
            "very negative": -0.6,
            "neutral": 0.0,
            "positive": 0.25,
        }

        def sentiment_fn(text):
            return (polarity_map[text], 0.0)

        scores = build_negative_sentiment_scores(
            note_corpus,
            sentiment_fn=sentiment_fn,
        )

        self._assert_hadm_unique(scores, "Negative sentiment scores")
        by_hadm = scores.set_index("hadm_id")
        self.assertAlmostEqual(by_hadm.loc[101, "negative_sentiment_score"], 0.6)
        self.assertAlmostEqual(by_hadm.loc[103, "negative_sentiment_score"], 0.0)
        self.assertAlmostEqual(by_hadm.loc[106, "negative_sentiment_score"], -0.25)

    def test_build_negative_sentiment_scores_passes_whitespace_cleaned_text_to_sentiment(self):
        build_negative_sentiment_scores = self._get_callable("build_negative_sentiment_scores")
        note_corpus = pd.DataFrame(
            [
                {
                    "hadm_id": 201,
                    "note_text": "Patient   refuses \n   treatment   Date:[**5-1-18**]",
                }
            ]
        )
        seen = []

        def sentiment_fn(text):
            seen.append(text)
            return (-0.3, 0.0)

        scores = build_negative_sentiment_scores(note_corpus, sentiment_fn=sentiment_fn)

        self.assertEqual(seen, ["Patient refuses treatment Date:[**5-1-18**]"])
        self.assertAlmostEqual(scores.loc[0, "negative_sentiment_score"], 0.3)

    def test_build_negative_sentiment_scores_handles_empty_notes_as_zero(self):
        build_negative_sentiment_scores = self._get_callable("build_negative_sentiment_scores")
        note_corpus = pd.DataFrame(
            [
                {"hadm_id": 101, "note_text": ""},
                {"hadm_id": 103, "note_text": "non-empty"},
            ]
        )

        def sentiment_fn(text):
            if text == "":
                raise AssertionError("sentiment_fn should not be called on empty note text")
            return (0.4, 0.0)

        scores = build_negative_sentiment_scores(note_corpus, sentiment_fn=sentiment_fn)
        by_hadm = scores.set_index("hadm_id")
        self.assertAlmostEqual(by_hadm.loc[101, "negative_sentiment_score"], 0.0)
        self.assertAlmostEqual(by_hadm.loc[103, "negative_sentiment_score"], -0.4)

    def test_build_negative_sentiment_scores_raises_clear_error_when_required_columns_are_missing(self):
        build_negative_sentiment_scores = self._get_callable("build_negative_sentiment_scores")
        note_corpus_missing = pd.DataFrame([{"hadm_id": 101}])
        with self.assertRaisesRegex(ValueError, "note_text"):
            build_negative_sentiment_scores(note_corpus_missing)

    def test_build_mistrust_score_table_constructs_all_three_normalized_scores_from_inputs(self):
        build_mistrust_score_table = self._get_callable("build_mistrust_score_table")
        feature_matrix = pd.DataFrame(
            [
                {"hadm_id": 101, "feature_a": 1, "feature_b": 1},
                {"hadm_id": 103, "feature_a": 0, "feature_b": 0},
                {"hadm_id": 106, "feature_a": 1, "feature_b": 0},
                {"hadm_id": 107, "feature_a": 0, "feature_b": 0},
            ]
        )
        note_labels = pd.DataFrame(
            [
                {"hadm_id": 101, "noncompliance_label": 1, "autopsy_label": 1},
                {"hadm_id": 103, "noncompliance_label": 0, "autopsy_label": 0},
                {"hadm_id": 106, "noncompliance_label": 1, "autopsy_label": 0},
                {"hadm_id": 107, "noncompliance_label": 0, "autopsy_label": 1},
            ]
        )
        note_corpus = pd.DataFrame(
            [
                {"hadm_id": 101, "note_text": "negative note"},
                {"hadm_id": 103, "note_text": "neutral note"},
                {"hadm_id": 106, "note_text": "slightly positive note"},
                {"hadm_id": 107, "note_text": "very positive note"},
            ]
        )
        probability_sequences = [
            [0.9, 0.2, 0.8, 0.1],
            [0.7, 0.1, 0.3, 0.6],
        ]
        created = []

        def estimator_factory():
            estimator = _FakeProbEstimator(probability_sequences[len(created)])
            created.append(estimator)
            return estimator

        polarity_map = {
            "negative note": -0.5,
            "neutral note": 0.0,
            "slightly positive note": 0.2,
            "very positive note": 0.6,
        }

        def sentiment_fn(text):
            return (polarity_map[text], 0.0)

        scores = build_mistrust_score_table(
            feature_matrix=feature_matrix,
            note_labels=note_labels,
            note_corpus=note_corpus,
            estimator_factory=estimator_factory,
            sentiment_fn=sentiment_fn,
        )

        self.assertEqual(len(created), 2, msg="Two proxy models should be fit: noncompliance and autopsy")
        self.assertTrue(all(est.was_fit for est in created))
        self._assert_hadm_unique(scores, "Mistrust score table")
        required_columns = {
            "hadm_id",
            "noncompliance_score_z",
            "autopsy_score_z",
            "negative_sentiment_score_z",
        }
        self.assertTrue(required_columns.issubset(scores.columns))

        for col in [
            "noncompliance_score_z",
            "autopsy_score_z",
            "negative_sentiment_score_z",
        ]:
            self.assertAlmostEqual(scores[col].mean(), 0.0, places=7)
            self.assertAlmostEqual(scores[col].std(ddof=0), 1.0, places=7)

        by_hadm = scores.set_index("hadm_id")
        self.assertGreater(
            by_hadm.loc[101, "noncompliance_score_z"],
            by_hadm.loc[103, "noncompliance_score_z"],
        )
        self.assertGreater(
            by_hadm.loc[101, "autopsy_score_z"],
            by_hadm.loc[103, "autopsy_score_z"],
        )
        self.assertGreater(
            by_hadm.loc[101, "negative_sentiment_score_z"],
            by_hadm.loc[107, "negative_sentiment_score_z"],
            msg="Negative sentiment score must be based on -1 * polarity before normalization",
        )

    def test_build_mistrust_score_table_keeps_only_hadm_ids_present_in_all_score_sources(self):
        build_mistrust_score_table = self._get_callable("build_mistrust_score_table")
        feature_matrix = pd.DataFrame(
            [
                {"hadm_id": 101, "feature_a": 1},
                {"hadm_id": 103, "feature_a": 0},
                {"hadm_id": 106, "feature_a": 1},
            ]
        )
        note_labels = pd.DataFrame(
            [
                {"hadm_id": 101, "noncompliance_label": 1, "autopsy_label": 0},
                {"hadm_id": 103, "noncompliance_label": 0, "autopsy_label": 1},
                {"hadm_id": 106, "noncompliance_label": 1, "autopsy_label": 0},
            ]
        )
        note_corpus = pd.DataFrame(
            [
                {"hadm_id": 101, "note_text": "negative"},
                {"hadm_id": 106, "note_text": "neutral"},
            ]
        )

        mistrust = build_mistrust_score_table(
            feature_matrix=feature_matrix,
            note_labels=note_labels,
            note_corpus=note_corpus,
            estimator_factory=lambda: _FakeProbEstimator([0.9, 0.2, 0.8]),
            sentiment_fn=lambda text: (-0.1 if text == "negative" else 0.0, 0.0),
        )

        self.assertEqual(list(mistrust["hadm_id"]), [101, 106])

    def test_build_mistrust_score_table_raises_clear_error_when_required_columns_are_missing(self):
        build_mistrust_score_table = self._get_callable("build_mistrust_score_table")
        note_labels_missing = pd.DataFrame([{"hadm_id": 101, "noncompliance_label": 1}])
        feature_matrix = pd.DataFrame([{"hadm_id": 101, "feature_a": 1}])
        note_corpus = pd.DataFrame([{"hadm_id": 101, "note_text": "note"}])
        with self.assertRaisesRegex(ValueError, "autopsy_label"):
            build_mistrust_score_table(
                feature_matrix=feature_matrix,
                note_labels=note_labels_missing,
                note_corpus=note_corpus,
            )

    def test_build_final_model_table_contains_baseline_optional_features_and_targets(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        build_demographics_table = self._get_callable("build_demographics_table")
        build_all_cohort = self._get_callable("build_all_cohort")
        build_final_model_table = self._get_callable("build_final_model_table")

        base = build_base_admissions(self.admissions, self.patients)
        demographics = build_demographics_table(base)
        all_cohort = build_all_cohort(base, self.icustays)

        code_status_items = pd.DataFrame(
            [
                {"itemid": 1, "label": "Education Readiness", "dbsource": "carevue"},
                {"itemid": 2, "label": "Pain Level", "dbsource": "metavision"},
                {"itemid": 128, "label": "Code Status", "dbsource": "carevue"},
                {"itemid": 223758, "label": "Code Status", "dbsource": "metavision"},
            ]
        )
        code_status_events = pd.DataFrame(
            [
                {"hadm_id": 101, "itemid": 1, "value": "No", "icustay_id": 1011},
                {"hadm_id": 101, "itemid": 2, "value": "7-Mod to Severe", "icustay_id": 1011},
                {"hadm_id": 101, "itemid": 128, "value": "Full Code", "icustay_id": 1011},
                {"hadm_id": 103, "itemid": 128, "value": "DNR/DNI", "icustay_id": 1031},
                {"hadm_id": 104, "itemid": 128, "value": "Full Code", "icustay_id": 1041},
                {"hadm_id": 106, "itemid": 128, "value": "Full Code", "icustay_id": 1061},
                {
                    "hadm_id": 107,
                    "itemid": 223758,
                    "value": "Comfort Measures Only",
                    "icustay_id": 1071,
                },
            ]
        )

        final_table = build_final_model_table(
            demographics=demographics,
            all_cohort=all_cohort,
            admissions=base,
            chartevents=code_status_events,
            d_items=code_status_items,
            mistrust_scores=self.mistrust_scores,
            include_race=True,
            include_mistrust=True,
        )

        self.assertIsInstance(final_table, pd.DataFrame)
        required_columns = {
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
            "code_status_dnr_dni_cmo",
            "in_hospital_mortality",
        }
        self.assertTrue(required_columns.issubset(set(final_table.columns)))
        self.assertEqual(set(final_table["hadm_id"]), {101, 103, 106, 107})

        by_hadm = final_table.set_index("hadm_id")
        self.assertEqual(by_hadm.loc[106, "left_ama"], 1)
        self.assertEqual(by_hadm.loc[101, "left_ama"], 0)
        self.assertEqual(by_hadm.loc[103, "code_status_dnr_dni_cmo"], 1)
        self.assertEqual(by_hadm.loc[101, "code_status_dnr_dni_cmo"], 0)
        self.assertEqual(by_hadm.loc[107, "code_status_dnr_dni_cmo"], 1)
        self.assertEqual(by_hadm.loc[107, "in_hospital_mortality"], 1)
        self.assertEqual(by_hadm.loc[101, "in_hospital_mortality"], 0)
        self._assert_hadm_unique(final_table, "Final model table")

    def test_build_final_model_table_left_ama_requires_exact_discharge_location_match(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        build_demographics_table = self._get_callable("build_demographics_table")
        build_all_cohort = self._get_callable("build_all_cohort")
        build_final_model_table = self._get_callable("build_final_model_table")

        admissions = pd.DataFrame(
            [
                {
                    "hadm_id": 201,
                    "subject_id": 21,
                    "admittime": "2100-01-01 00:00:00",
                    "dischtime": "2100-01-02 00:00:00",
                    "ethnicity": "WHITE",
                    "insurance": "Medicare",
                    "discharge_location": "LEFT AGAINST MEDICAL ADVICE",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 202,
                    "subject_id": 22,
                    "admittime": "2100-02-01 00:00:00",
                    "dischtime": "2100-02-02 00:00:00",
                    "ethnicity": "BLACK/AFRICAN AMERICAN",
                    "insurance": "Private",
                    "discharge_location": "TRANSFER AGAINST MEDICAL ADVICE REVIEW",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
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
                {
                    "hadm_id": 201,
                    "icustay_id": 2011,
                    "intime": "2100-01-01 00:00:00",
                    "outtime": "2100-01-01 13:00:00",
                },
                {
                    "hadm_id": 202,
                    "icustay_id": 2021,
                    "intime": "2100-02-01 00:00:00",
                    "outtime": "2100-02-01 13:00:00",
                },
            ]
        )

        base = build_base_admissions(admissions, patients)
        demographics = build_demographics_table(base)
        all_cohort = build_all_cohort(base, icustays)
        mistrust_scores = pd.DataFrame(
            [
                {
                    "hadm_id": 201,
                    "noncompliance_score_z": 0.0,
                    "autopsy_score_z": 0.0,
                    "negative_sentiment_score_z": 0.0,
                },
                {
                    "hadm_id": 202,
                    "noncompliance_score_z": 0.0,
                    "autopsy_score_z": 0.0,
                    "negative_sentiment_score_z": 0.0,
                },
            ]
        )
        empty_chartevents = pd.DataFrame(columns=["hadm_id", "itemid", "value", "icustay_id"])
        empty_d_items = pd.DataFrame(columns=["itemid", "label", "dbsource"])

        final_table = build_final_model_table(
            demographics=demographics,
            all_cohort=all_cohort,
            admissions=base,
            chartevents=empty_chartevents,
            d_items=empty_d_items,
            mistrust_scores=mistrust_scores,
            include_race=True,
            include_mistrust=True,
        ).set_index("hadm_id")

        self.assertEqual(final_table.loc[201, "left_ama"], 1)
        self.assertEqual(final_table.loc[202, "left_ama"], 0)

    def test_build_final_model_table_code_status_uses_only_required_itemids(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        build_demographics_table = self._get_callable("build_demographics_table")
        build_all_cohort = self._get_callable("build_all_cohort")
        build_final_model_table = self._get_callable("build_final_model_table")

        admissions = pd.DataFrame(
            [
                {
                    "hadm_id": 301,
                    "subject_id": 31,
                    "admittime": "2100-03-01 00:00:00",
                    "dischtime": "2100-03-02 00:00:00",
                    "ethnicity": "WHITE",
                    "insurance": "Medicare",
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 302,
                    "subject_id": 32,
                    "admittime": "2100-03-01 00:00:00",
                    "dischtime": "2100-03-02 00:00:00",
                    "ethnicity": "BLACK/AFRICAN AMERICAN",
                    "insurance": "Private",
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 303,
                    "subject_id": 33,
                    "admittime": "2100-03-01 00:00:00",
                    "dischtime": "2100-03-02 00:00:00",
                    "ethnicity": "ASIAN",
                    "insurance": "Medicare",
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
            ]
        )
        patients = pd.DataFrame(
            [
                {"subject_id": 31, "gender": "M", "dob": "2070-01-01 00:00:00"},
                {"subject_id": 32, "gender": "F", "dob": "2070-01-01 00:00:00"},
                {"subject_id": 33, "gender": "M", "dob": "2070-01-01 00:00:00"},
            ]
        )
        icustays = pd.DataFrame(
            [
                {
                    "hadm_id": 301,
                    "icustay_id": 3011,
                    "intime": "2100-03-01 00:00:00",
                    "outtime": "2100-03-01 13:00:00",
                },
                {
                    "hadm_id": 302,
                    "icustay_id": 3021,
                    "intime": "2100-03-01 00:00:00",
                    "outtime": "2100-03-01 13:00:00",
                },
                {
                    "hadm_id": 303,
                    "icustay_id": 3031,
                    "intime": "2100-03-01 00:00:00",
                    "outtime": "2100-03-01 13:00:00",
                },
            ]
        )
        d_items = pd.DataFrame(
            [
                {"itemid": 999, "label": "Code Status", "dbsource": "carevue"},
                {"itemid": 128, "label": "Code Status", "dbsource": "carevue"},
                {"itemid": 223758, "label": "Code Status", "dbsource": "metavision"},
            ]
        )
        chartevents = pd.DataFrame(
            [
                {"hadm_id": 301, "itemid": 999, "value": "DNR/DNI", "icustay_id": 3011},
                {"hadm_id": 302, "itemid": 128, "value": "DNR/DNI", "icustay_id": 3021},
                {
                    "hadm_id": 303,
                    "itemid": 223758,
                    "value": "Comfort Measures Only",
                    "icustay_id": 3031,
                },
            ]
        )

        base = build_base_admissions(admissions, patients)
        demographics = build_demographics_table(base)
        all_cohort = build_all_cohort(base, icustays)
        mistrust_scores = pd.DataFrame(
            [
                {
                    "hadm_id": 301,
                    "noncompliance_score_z": 0.0,
                    "autopsy_score_z": 0.0,
                    "negative_sentiment_score_z": 0.0,
                },
                {
                    "hadm_id": 302,
                    "noncompliance_score_z": 0.0,
                    "autopsy_score_z": 0.0,
                    "negative_sentiment_score_z": 0.0,
                },
                {
                    "hadm_id": 303,
                    "noncompliance_score_z": 0.0,
                    "autopsy_score_z": 0.0,
                    "negative_sentiment_score_z": 0.0,
                },
            ]
        )

        final_table = build_final_model_table(
            demographics=demographics,
            all_cohort=all_cohort,
            admissions=base,
            chartevents=chartevents,
            d_items=d_items,
            mistrust_scores=mistrust_scores,
            include_race=True,
            include_mistrust=True,
        ).set_index("hadm_id")

        self.assertEqual(final_table.loc[301, "code_status_dnr_dni_cmo"], 0)
        self.assertEqual(final_table.loc[302, "code_status_dnr_dni_cmo"], 1)
        self.assertEqual(final_table.loc[303, "code_status_dnr_dni_cmo"], 1)

    def test_build_code_status_target_excludes_admissions_without_charted_code_status(self):
        build_code_status_target = getattr(self.module, "_build_code_status_target")
        d_items = pd.DataFrame(
            [
                {"itemid": 128, "label": "Code Status", "dbsource": "carevue"},
                {"itemid": 223758, "label": "Code Status", "dbsource": "metavision"},
                {"itemid": 1, "label": "Education Readiness", "dbsource": "carevue"},
            ]
        )
        chartevents = pd.DataFrame(
            [
                {"hadm_id": 601, "itemid": 128, "value": "DNR/DNI", "icustay_id": 6011},
                {"hadm_id": 602, "itemid": 1, "value": "No", "icustay_id": 6021},
            ]
        )

        target = build_code_status_target(chartevents, d_items)
        self.assertEqual(set(target["hadm_id"]), {601})

    def test_build_final_model_table_supports_baseline_only_configuration(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        build_demographics_table = self._get_callable("build_demographics_table")
        build_all_cohort = self._get_callable("build_all_cohort")
        build_final_model_table = self._get_callable("build_final_model_table")

        base = build_base_admissions(self.admissions, self.patients)
        demographics = build_demographics_table(base)
        all_cohort = build_all_cohort(base, self.icustays)
        final_table = build_final_model_table(
            demographics=demographics,
            all_cohort=all_cohort,
            admissions=base,
            chartevents=self.chartevents,
            d_items=self.d_items,
            mistrust_scores=self.mistrust_scores,
            include_race=False,
            include_mistrust=False,
        )
        self.assertNotIn("race_white", final_table.columns)
        self.assertNotIn("noncompliance_score_z", final_table.columns)
        self.assertIn("age", final_table.columns)
        self.assertIn("left_ama", final_table.columns)

    def test_build_final_model_table_baseline_only_columns_match_required_set(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        build_demographics_table = self._get_callable("build_demographics_table")
        build_all_cohort = self._get_callable("build_all_cohort")
        build_final_model_table = self._get_callable("build_final_model_table")

        base = build_base_admissions(self.admissions, self.patients)
        demographics = build_demographics_table(base)
        all_cohort = build_all_cohort(base, self.icustays)
        final_table = build_final_model_table(
            demographics=demographics,
            all_cohort=all_cohort,
            admissions=base,
            chartevents=self.chartevents,
            d_items=self.d_items,
            mistrust_scores=self.mistrust_scores,
            include_race=False,
            include_mistrust=False,
        )
        expected_columns = {
            "hadm_id",
            "age",
            "los_days",
            "gender_f",
            "gender_m",
            "insurance_private",
            "insurance_public",
            "insurance_self_pay",
            "left_ama",
            "code_status_dnr_dni_cmo",
            "in_hospital_mortality",
        }
        self.assertEqual(set(final_table.columns), expected_columns)
        self.assertEqual(len(final_table.columns), 11)

    def test_build_final_model_table_race_one_hot_covers_all_required_categories(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        build_demographics_table = self._get_callable("build_demographics_table")
        build_all_cohort = self._get_callable("build_all_cohort")
        build_final_model_table = self._get_callable("build_final_model_table")

        admissions = pd.DataFrame(
            [
                {
                    "hadm_id": 701,
                    "subject_id": 71,
                    "admittime": "2100-07-01 00:00:00",
                    "dischtime": "2100-07-02 00:00:00",
                    "ethnicity": "WHITE",
                    "insurance": "Medicare",
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 702,
                    "subject_id": 72,
                    "admittime": "2100-07-01 00:00:00",
                    "dischtime": "2100-07-02 00:00:00",
                    "ethnicity": "BLACK/AFRICAN AMERICAN",
                    "insurance": "Private",
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 703,
                    "subject_id": 73,
                    "admittime": "2100-07-01 00:00:00",
                    "dischtime": "2100-07-02 00:00:00",
                    "ethnicity": "ASIAN - CHINESE",
                    "insurance": "Medicare",
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 704,
                    "subject_id": 74,
                    "admittime": "2100-07-01 00:00:00",
                    "dischtime": "2100-07-02 00:00:00",
                    "ethnicity": "HISPANIC OR LATINO",
                    "insurance": "Private",
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 705,
                    "subject_id": 75,
                    "admittime": "2100-07-01 00:00:00",
                    "dischtime": "2100-07-02 00:00:00",
                    "ethnicity": "AMERICAN INDIAN/ALASKA NATIVE",
                    "insurance": "Medicare",
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
                {
                    "hadm_id": 706,
                    "subject_id": 76,
                    "admittime": "2100-07-01 00:00:00",
                    "dischtime": "2100-07-02 00:00:00",
                    "ethnicity": "PATIENT DECLINED TO ANSWER",
                    "insurance": "Private",
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 0,
                    "has_chartevents_data": 1,
                },
            ]
        )
        patients = pd.DataFrame(
            [
                {"subject_id": 71, "gender": "M", "dob": "2070-07-01 00:00:00"},
                {"subject_id": 72, "gender": "F", "dob": "2070-07-01 00:00:00"},
                {"subject_id": 73, "gender": "M", "dob": "2070-07-01 00:00:00"},
                {"subject_id": 74, "gender": "F", "dob": "2070-07-01 00:00:00"},
                {"subject_id": 75, "gender": "M", "dob": "2070-07-01 00:00:00"},
                {"subject_id": 76, "gender": "F", "dob": "2070-07-01 00:00:00"},
            ]
        )
        icustays = pd.DataFrame(
            [
                {"hadm_id": 701, "icustay_id": 7011, "intime": "2100-07-01 00:00:00", "outtime": "2100-07-01 13:00:00"},
                {"hadm_id": 702, "icustay_id": 7021, "intime": "2100-07-01 00:00:00", "outtime": "2100-07-01 13:00:00"},
                {"hadm_id": 703, "icustay_id": 7031, "intime": "2100-07-01 00:00:00", "outtime": "2100-07-01 13:00:00"},
                {"hadm_id": 704, "icustay_id": 7041, "intime": "2100-07-01 00:00:00", "outtime": "2100-07-01 13:00:00"},
                {"hadm_id": 705, "icustay_id": 7051, "intime": "2100-07-01 00:00:00", "outtime": "2100-07-01 13:00:00"},
                {"hadm_id": 706, "icustay_id": 7061, "intime": "2100-07-01 00:00:00", "outtime": "2100-07-01 13:00:00"},
            ]
        )
        mistrust_scores = pd.DataFrame(
            [
                {
                    "hadm_id": hadm_id,
                    "noncompliance_score_z": 0.0,
                    "autopsy_score_z": 0.0,
                    "negative_sentiment_score_z": 0.0,
                }
                for hadm_id in [701, 702, 703, 704, 705, 706]
            ]
        )

        base = build_base_admissions(admissions, patients)
        demographics = build_demographics_table(base)
        all_cohort = build_all_cohort(base, icustays)
        final_table = build_final_model_table(
            demographics=demographics,
            all_cohort=all_cohort,
            admissions=base,
            chartevents=pd.DataFrame(columns=["hadm_id", "itemid", "value", "icustay_id"]),
            d_items=pd.DataFrame(columns=["itemid", "label", "dbsource"]),
            mistrust_scores=mistrust_scores,
            include_race=True,
            include_mistrust=False,
        ).set_index("hadm_id")

        self.assertEqual(final_table.loc[701, "race_white"], 1)
        self.assertEqual(final_table.loc[702, "race_black"], 1)
        self.assertEqual(final_table.loc[703, "race_asian"], 1)
        self.assertEqual(final_table.loc[704, "race_hispanic"], 1)
        self.assertEqual(final_table.loc[705, "race_native_american"], 1)
        self.assertEqual(final_table.loc[706, "race_other"], 1)

    def test_write_minimal_deliverables_creates_required_artifact_files(self):
        write_minimal_deliverables = self._get_callable("write_minimal_deliverables")
        artifacts = {
            "base_admissions": pd.DataFrame([{"hadm_id": 101}]),
            "eol_cohort": pd.DataFrame([{"hadm_id": 101}]),
            "all_cohort": pd.DataFrame([{"hadm_id": 101}]),
            "treatment_totals": pd.DataFrame([{"hadm_id": 101, "total_vent_min": 810.0, "total_vaso_min": 0.0}]),
            "chartevent_feature_matrix": pd.DataFrame([{"hadm_id": 101, "feature_a": 1}]),
            "note_labels": pd.DataFrame([{"hadm_id": 101, "noncompliance_label": 1, "autopsy_label": 1}]),
            "mistrust_scores": pd.DataFrame([{"hadm_id": 101, "noncompliance_score_z": 1.0}]),
            "acuity_scores": pd.DataFrame([{"hadm_id": 101, "oasis": 15, "sapsii": 42}]),
            "final_model_table": pd.DataFrame([{"hadm_id": 101, "left_ama": 0}]),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_minimal_deliverables(artifacts, output_dir)

            expected_files = {
                "base_admissions.csv",
                "eol_cohort.csv",
                "all_cohort.csv",
                "treatment_totals.csv",
                "chartevent_feature_matrix.csv",
                "note_labels.csv",
                "mistrust_scores.csv",
                "acuity_scores.csv",
                "final_model_table.csv",
            }
            written_files = {path.name for path in output_dir.iterdir()}
            self.assertEqual(expected_files, written_files)

    def test_write_minimal_deliverables_sorts_by_hadm_id_and_writes_without_index(self):
        write_minimal_deliverables = self._get_callable("write_minimal_deliverables")
        artifacts = {
            "base_admissions": pd.DataFrame([{"hadm_id": 103}, {"hadm_id": 101}]),
            "eol_cohort": pd.DataFrame([{"hadm_id": 103}, {"hadm_id": 101}]),
            "all_cohort": pd.DataFrame([{"hadm_id": 103}, {"hadm_id": 101}]),
            "treatment_totals": pd.DataFrame(
                [
                    {"hadm_id": 103, "total_vent_min": 0.0, "total_vaso_min": 840.0},
                    {"hadm_id": 101, "total_vent_min": 810.0, "total_vaso_min": 0.0},
                ]
            ),
            "chartevent_feature_matrix": pd.DataFrame(
                [
                    {"hadm_id": 103, "feature_a": 1},
                    {"hadm_id": 101, "feature_a": 0},
                ]
            ),
            "note_labels": pd.DataFrame(
                [
                    {"hadm_id": 103, "noncompliance_label": 0, "autopsy_label": 0},
                    {"hadm_id": 101, "noncompliance_label": 1, "autopsy_label": 1},
                ]
            ),
            "mistrust_scores": pd.DataFrame(
                [
                    {"hadm_id": 103, "noncompliance_score_z": -0.3},
                    {"hadm_id": 101, "noncompliance_score_z": 1.2},
                ]
            ),
            "acuity_scores": pd.DataFrame(
                [
                    {"hadm_id": 103, "oasis": 20, "sapsii": 55},
                    {"hadm_id": 101, "oasis": 15, "sapsii": 42},
                ]
            ),
            "final_model_table": pd.DataFrame(
                [
                    {"hadm_id": 103, "left_ama": 0},
                    {"hadm_id": 101, "left_ama": 0},
                ]
            ),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_minimal_deliverables(artifacts, output_dir)
            base_admissions = pd.read_csv(output_dir / "base_admissions.csv")
            self.assertEqual(list(base_admissions["hadm_id"]), [101, 103])
            self.assertNotIn("Unnamed: 0", base_admissions.columns)

    def test_write_minimal_deliverables_raises_when_required_artifact_is_missing(self):
        write_minimal_deliverables = self._get_callable("write_minimal_deliverables")
        artifacts = {
            "base_admissions": pd.DataFrame([{"hadm_id": 101}]),
            "eol_cohort": pd.DataFrame([{"hadm_id": 101}]),
            "all_cohort": pd.DataFrame([{"hadm_id": 101}]),
            "treatment_totals": pd.DataFrame(
                [{"hadm_id": 101, "total_vent_min": 810.0, "total_vaso_min": 0.0}]
            ),
            "chartevent_feature_matrix": pd.DataFrame([{"hadm_id": 101, "feature_a": 1}]),
            "note_labels": pd.DataFrame(
                [{"hadm_id": 101, "noncompliance_label": 1, "autopsy_label": 0}]
            ),
            "mistrust_scores": pd.DataFrame(
                [{"hadm_id": 101, "noncompliance_score_z": 0.0}]
            ),
            "acuity_scores": pd.DataFrame([{"hadm_id": 101, "oasis": 15, "sapsii": 42}]),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(KeyError):
                write_minimal_deliverables(artifacts, Path(temp_dir))

    def test_write_minimal_deliverables_sorts_nullable_integer_hadm_ids(self):
        write_minimal_deliverables = self._get_callable("write_minimal_deliverables")
        artifacts = {
            "base_admissions": pd.DataFrame({"hadm_id": pd.Series([103, 101], dtype="Int64")}),
            "eol_cohort": pd.DataFrame({"hadm_id": pd.Series([103, 101], dtype="Int64")}),
            "all_cohort": pd.DataFrame({"hadm_id": pd.Series([103, 101], dtype="Int64")}),
            "treatment_totals": pd.DataFrame(
                {
                    "hadm_id": pd.Series([103, 101], dtype="Int64"),
                    "total_vent_min": [0.0, 810.0],
                    "total_vaso_min": [840.0, 0.0],
                }
            ),
            "chartevent_feature_matrix": pd.DataFrame({"hadm_id": pd.Series([103, 101], dtype="Int64")}),
            "note_labels": pd.DataFrame(
                {
                    "hadm_id": pd.Series([103, 101], dtype="Int64"),
                    "noncompliance_label": [0, 1],
                    "autopsy_label": [0, 1],
                }
            ),
            "mistrust_scores": pd.DataFrame(
                {
                    "hadm_id": pd.Series([103, 101], dtype="Int64"),
                    "noncompliance_score_z": [-0.3, 1.2],
                }
            ),
            "acuity_scores": pd.DataFrame(
                {
                    "hadm_id": pd.Series([103, 101], dtype="Int64"),
                    "oasis": [20, 15],
                    "sapsii": [55, 42],
                }
            ),
            "final_model_table": pd.DataFrame(
                {
                    "hadm_id": pd.Series([103, 101], dtype="Int64"),
                    "left_ama": [0, 0],
                }
            ),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_minimal_deliverables(artifacts, output_dir)
            final_table = pd.read_csv(output_dir / "final_model_table.csv")
            self.assertEqual(list(final_table["hadm_id"]), [101, 103])

    def test_data_contract_build_base_admissions_output_schema_dtypes_and_uniqueness_are_stable(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        base = build_base_admissions(self.admissions, self.patients)

        self.assertEqual(
            base.columns.tolist(),
            [
                "hadm_id",
                "subject_id",
                "admittime",
                "dischtime",
                "ethnicity",
                "insurance",
                "discharge_location",
                "hospital_expire_flag",
                "has_chartevents_data",
                "gender",
                "dob",
            ],
        )
        self.assertEqual(base["hadm_id"].tolist(), sorted(base["hadm_id"].tolist()))
        self._assert_hadm_unique(base, "Base admissions contract")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(base["admittime"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(base["dischtime"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(base["dob"]))

    def test_data_contract_build_demographics_table_output_dtypes_and_uniqueness_are_stable(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        build_demographics_table = self._get_callable("build_demographics_table")
        base = build_base_admissions(self.admissions, self.patients)
        demographics = build_demographics_table(base)

        self.assertEqual(demographics["hadm_id"].tolist(), sorted(demographics["hadm_id"].tolist()))
        self._assert_hadm_unique(demographics, "Demographics contract")
        self.assertTrue(pd.api.types.is_float_dtype(demographics["age"]))
        self.assertTrue(pd.api.types.is_float_dtype(demographics["los_hours"]))
        self.assertTrue(pd.api.types.is_float_dtype(demographics["los_days"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(demographics["admittime"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(demographics["dischtime"]))

    def test_data_contract_build_eol_and_all_cohorts_are_sorted_unique_and_key_aligned(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        build_demographics_table = self._get_callable("build_demographics_table")
        build_eol_cohort = self._get_callable("build_eol_cohort")
        build_all_cohort = self._get_callable("build_all_cohort")
        base = build_base_admissions(self.admissions, self.patients)
        demographics = build_demographics_table(base)
        eol = build_eol_cohort(base, demographics)
        all_cohort = build_all_cohort(base, self.icustays)

        self.assertEqual(eol["hadm_id"].tolist(), sorted(eol["hadm_id"].tolist()))
        self.assertEqual(all_cohort["hadm_id"].tolist(), sorted(all_cohort["hadm_id"].tolist()))
        self._assert_hadm_unique(eol, "EOL cohort contract")
        self._assert_hadm_unique(all_cohort, "ALL cohort contract")
        self.assertTrue(set(eol["hadm_id"]).issubset(set(base["hadm_id"])))
        self.assertTrue(set(all_cohort["hadm_id"]).issubset(set(base["hadm_id"])))
        self.assertTrue(eol["discharge_category"].notna().all())

    def test_data_contract_build_treatment_totals_output_schema_dtypes_and_uniqueness_are_stable(self):
        build_treatment_totals = self._get_callable("build_treatment_totals")
        totals = build_treatment_totals(self.icustays, self.ventdurations, self.vasopressordurations)

        self.assertEqual(totals.columns.tolist(), ["hadm_id", "total_vent_min", "total_vaso_min"])
        self.assertEqual(totals["hadm_id"].tolist(), sorted(totals["hadm_id"].tolist()))
        self._assert_hadm_unique(totals, "Treatment totals contract")
        self.assertTrue(pd.api.types.is_float_dtype(pd.to_numeric(totals["total_vent_min"], errors="coerce")))
        self.assertTrue(pd.api.types.is_float_dtype(pd.to_numeric(totals["total_vaso_min"], errors="coerce")))

    def test_data_contract_build_note_corpus_and_labels_outputs_are_sorted_unique_and_typed(self):
        build_note_corpus = self._get_callable("build_note_corpus")
        build_note_labels = self._get_callable("build_note_labels")
        corpus = build_note_corpus(self.noteevents, all_hadm_ids=[101, 103, 104, 106, 107])
        labels = build_note_labels(self.noteevents, all_hadm_ids=[101, 103, 104, 106, 107])

        self.assertEqual(corpus["hadm_id"].tolist(), [101, 103, 104, 106, 107])
        self.assertEqual(labels["hadm_id"].tolist(), [101, 103, 104, 106, 107])
        self._assert_hadm_unique(corpus, "Note corpus contract")
        self._assert_hadm_unique(labels, "Note labels contract")
        self.assertTrue(pd.api.types.is_object_dtype(corpus["note_text"]))
        self.assertTrue(pd.api.types.is_integer_dtype(labels["noncompliance_label"]))
        self.assertTrue(pd.api.types.is_integer_dtype(labels["autopsy_label"]))
        self.assertTrue(set(labels["noncompliance_label"].unique()).issubset({0, 1}))
        self.assertTrue(set(labels["autopsy_label"].unique()).issubset({0, 1}))

    def test_data_contract_build_chartevent_feature_matrix_output_is_binary_integer_sorted_and_unique(self):
        build_chartevent_feature_matrix = self._get_callable("build_chartevent_feature_matrix")
        matrix = build_chartevent_feature_matrix(
            self.chartevents,
            self.d_items,
            all_hadm_ids=[101, 103, 104, 106, 107],
        )

        self.assertEqual(matrix["hadm_id"].tolist(), [101, 103, 104, 106, 107])
        self._assert_hadm_unique(matrix, "Feature matrix contract")
        for column in matrix.columns:
            if column == "hadm_id":
                continue
            self.assertTrue(pd.api.types.is_integer_dtype(matrix[column]), msg=column)
            self.assertTrue(set(matrix[column].dropna().unique()).issubset({0, 1}), msg=column)

    def test_data_contract_build_acuity_scores_output_is_sorted_unique_and_numeric(self):
        build_acuity_scores = self._get_callable("build_acuity_scores")
        acuity = build_acuity_scores(self.oasis, self.sapsii)

        self.assertEqual(acuity["hadm_id"].tolist(), sorted(acuity["hadm_id"].tolist()))
        self._assert_hadm_unique(acuity, "Acuity contract")
        self.assertTrue(pd.api.types.is_numeric_dtype(acuity["oasis"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(acuity["sapsii"]))

    def test_data_contract_build_final_model_table_binary_columns_are_integer_and_zero_one(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        build_demographics_table = self._get_callable("build_demographics_table")
        build_all_cohort = self._get_callable("build_all_cohort")
        build_final_model_table = self._get_callable("build_final_model_table")

        base = build_base_admissions(self.admissions, self.patients)
        demographics = build_demographics_table(base)
        all_cohort = build_all_cohort(base, self.icustays)
        final_table = build_final_model_table(
            demographics=demographics,
            all_cohort=all_cohort,
            admissions=base,
            chartevents=self.chartevents,
            d_items=self.d_items,
            mistrust_scores=self.mistrust_scores,
            include_race=True,
            include_mistrust=True,
        )

        binary_columns = [
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
            "left_ama",
            "code_status_dnr_dni_cmo",
            "in_hospital_mortality",
        ]
        for column in binary_columns:
            self.assertTrue(pd.api.types.is_integer_dtype(final_table[column]), msg=column)
            self.assertTrue(set(final_table[column].unique()).issubset({0, 1}), msg=column)

    def test_data_contract_write_minimal_deliverables_round_trip_preserves_columns_and_row_counts(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        build_demographics_table = self._get_callable("build_demographics_table")
        build_eol_cohort = self._get_callable("build_eol_cohort")
        build_all_cohort = self._get_callable("build_all_cohort")
        build_treatment_totals = self._get_callable("build_treatment_totals")
        build_note_corpus = self._get_callable("build_note_corpus")
        build_note_labels = self._get_callable("build_note_labels")
        build_chartevent_feature_matrix = self._get_callable("build_chartevent_feature_matrix")
        build_acuity_scores = self._get_callable("build_acuity_scores")
        build_mistrust_score_table = self._get_callable("build_mistrust_score_table")
        build_final_model_table = self._get_callable("build_final_model_table")
        write_minimal_deliverables = self._get_callable("write_minimal_deliverables")

        base = build_base_admissions(self.admissions, self.patients)
        demographics = build_demographics_table(base)
        eol = build_eol_cohort(base, demographics)
        all_cohort = build_all_cohort(base, self.icustays)
        treatments = build_treatment_totals(self.icustays, self.ventdurations, self.vasopressordurations)
        note_corpus = build_note_corpus(self.noteevents, all_hadm_ids=list(all_cohort["hadm_id"]))
        note_labels = build_note_labels(self.noteevents, all_hadm_ids=list(all_cohort["hadm_id"]))
        feature_matrix = build_chartevent_feature_matrix(
            self.chartevents,
            self.d_items,
            allowed_labels={"Education Readiness", "Pain Level"},
            all_hadm_ids=list(all_cohort["hadm_id"]),
        )
        acuity = build_acuity_scores(self.oasis, self.sapsii)
        mistrust_scores = build_mistrust_score_table(
            feature_matrix=feature_matrix,
            note_labels=note_labels,
            note_corpus=note_corpus,
            estimator_factory=lambda: _FakeProbEstimator([0.9, 0.2, 0.8, 0.1]),
            sentiment_fn=lambda text: (
                {
                    "Patient refuses treatment and was noncompliant with medication. Date:[**5-1-18**] Autopsy was discussed with the family.": -0.5,
                    "Cooperative patient. Follows commands.": 0.0,
                    "Patient remains nonadherent with follow up plan.": -0.2,
                    "": 0.0,
                }[text],
                0.0,
            ),
        )
        final_table = build_final_model_table(
            demographics=demographics,
            all_cohort=all_cohort,
            admissions=base,
            chartevents=self.chartevents,
            d_items=self.d_items,
            mistrust_scores=mistrust_scores,
            include_race=True,
            include_mistrust=True,
        )
        artifacts = {
            "base_admissions": base,
            "eol_cohort": eol,
            "all_cohort": all_cohort,
            "treatment_totals": treatments,
            "chartevent_feature_matrix": feature_matrix,
            "note_labels": note_labels,
            "mistrust_scores": mistrust_scores,
            "acuity_scores": acuity,
            "final_model_table": final_table,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_minimal_deliverables(artifacts, output_dir)
            round_trip = {
                "base_admissions": pd.read_csv(output_dir / "base_admissions.csv"),
                "eol_cohort": pd.read_csv(output_dir / "eol_cohort.csv"),
                "all_cohort": pd.read_csv(output_dir / "all_cohort.csv"),
                "treatment_totals": pd.read_csv(output_dir / "treatment_totals.csv"),
                "chartevent_feature_matrix": pd.read_csv(output_dir / "chartevent_feature_matrix.csv"),
                "note_labels": pd.read_csv(output_dir / "note_labels.csv"),
                "mistrust_scores": pd.read_csv(output_dir / "mistrust_scores.csv"),
                "acuity_scores": pd.read_csv(output_dir / "acuity_scores.csv"),
                "final_model_table": pd.read_csv(output_dir / "final_model_table.csv"),
            }

        for key, original in artifacts.items():
            self.assertEqual(round_trip[key].shape[0], original.shape[0], msg=key)
            self.assertEqual(round_trip[key].columns.tolist(), original.columns.tolist(), msg=key)

    def test_end_to_end_artifact_assembly_smoke_spec(self):
        build_base_admissions = self._get_callable("build_base_admissions")
        build_demographics_table = self._get_callable("build_demographics_table")
        build_eol_cohort = self._get_callable("build_eol_cohort")
        build_all_cohort = self._get_callable("build_all_cohort")
        build_treatment_totals = self._get_callable("build_treatment_totals")
        build_note_corpus = self._get_callable("build_note_corpus")
        build_chartevent_feature_matrix = self._get_callable("build_chartevent_feature_matrix")
        build_note_labels = self._get_callable("build_note_labels")
        build_acuity_scores = self._get_callable("build_acuity_scores")
        build_mistrust_score_table = self._get_callable("build_mistrust_score_table")
        build_final_model_table = self._get_callable("build_final_model_table")
        write_minimal_deliverables = self._get_callable("write_minimal_deliverables")

        base = build_base_admissions(self.admissions, self.patients)
        demographics = build_demographics_table(base)
        eol = build_eol_cohort(base, demographics)
        all_cohort = build_all_cohort(base, self.icustays)
        treatments = build_treatment_totals(
            self.icustays,
            self.ventdurations,
            self.vasopressordurations,
        )
        note_corpus = build_note_corpus(
            self.noteevents,
            all_hadm_ids=list(all_cohort["hadm_id"]),
        )
        note_labels = build_note_labels(
            self.noteevents,
            all_hadm_ids=list(all_cohort["hadm_id"]),
        )
        feature_matrix = build_chartevent_feature_matrix(
            self.chartevents,
            self.d_items,
            allowed_labels={"Education Readiness", "Pain Level"},
            all_hadm_ids=list(all_cohort["hadm_id"]),
        )
        acuity = build_acuity_scores(self.oasis, self.sapsii)
        mistrust_scores = build_mistrust_score_table(
            feature_matrix=feature_matrix,
            note_labels=note_labels,
            note_corpus=note_corpus,
            estimator_factory=lambda: _FakeProbEstimator([0.9, 0.2, 0.8, 0.1]),
            sentiment_fn=lambda text: (
                {
                    "Patient refuses treatment and was noncompliant with medication. Date:[**5-1-18**] Autopsy was discussed with the family.": -0.5,
                    "Cooperative patient. Follows commands.": 0.0,
                    "Patient remains nonadherent with follow up plan.": -0.2,
                    "": 0.0,
                }[text],
                0.0,
            ),
        )
        final_table = build_final_model_table(
            demographics=demographics,
            all_cohort=all_cohort,
            admissions=base,
            chartevents=self.chartevents,
            d_items=self.d_items,
            mistrust_scores=mistrust_scores,
            include_race=True,
            include_mistrust=True,
        )

        artifacts = {
            "base_admissions": base,
            "eol_cohort": eol,
            "all_cohort": all_cohort,
            "treatment_totals": treatments,
            "chartevent_feature_matrix": feature_matrix,
            "note_labels": note_labels,
            "mistrust_scores": mistrust_scores,
            "acuity_scores": acuity,
            "final_model_table": final_table,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_minimal_deliverables(artifacts, output_dir)
            self.assertEqual(len(list(output_dir.iterdir())), 9)


if __name__ == "__main__":
    unittest.main()

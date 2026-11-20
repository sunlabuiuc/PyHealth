"""Tests for TabularMortalityPredictionEICU on small synthetic patients.

Every patient is built in memory from a handful of made-up events, so the
tests read no eICU files and write nothing to disk.

AI assistance:
    Generated with Claude Fable 5.2 (Anthropic).
"""

import math
import unittest
from pathlib import Path

import polars as pl
import torch
import yaml

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset
from pyhealth.tasks import TabularMortalityPredictionEICU

DEFAULTS = TabularMortalityPredictionEICU.DEFAULT_FEATURES
LACTATE = DEFAULTS.index("lactate_last_early")
AGE = DEFAULTS.index("age")
ALBUMIN = DEFAULTS.index("albumin_first_early")
CREATININE = DEFAULTS.index("creatinine_last_early")

# Every column of the eICU config, as in the dataset's event dataframe.
_CONFIG = yaml.safe_load(
    (
        Path(__file__).parents[2] / "pyhealth" / "datasets" / "configs" / "eicu.yaml"
    ).read_text()
)
COLUMNS = [
    f"{table}/{column}"
    for table, config in _CONFIG["tables"].items()
    for column in config["attributes"]
]


def _event(event_type, **attrs):
    # Attributes are strings in the dataset's event dataframe.
    row = {"event_type": event_type, "timestamp": None}
    for key, value in attrs.items():
        row[f"{event_type}/{key}"] = None if value is None else str(value)
    return row


def _stay(stay_id=10, status="Alive", age="60", gender="Male", ethnicity="Caucasian"):
    return _event(
        "patient",
        patientunitstayid=stay_id,
        unitdischargestatus=status,
        age=age,
        gender=gender,
        ethnicity=ethnicity,
    )


def _lab(minute, name, value, stay_id=10):
    return _event(
        "lab",
        patientunitstayid=stay_id,
        labresultoffset=minute,
        labname=name,
        labresult=value,
    )


def _frame(rows, patient_id="p1"):
    df = pl.from_dicts(rows, infer_schema_length=None)
    return df.with_columns(
        pl.lit(patient_id).alias("patient_id"),
        pl.col("timestamp").cast(pl.Datetime("ms")),
        *(
            (pl.col(c) if c in df.columns else pl.lit(None)).cast(pl.String).alias(c)
            for c in COLUMNS
        ),
    )


def _patient(*rows, patient_id="p1"):
    return Patient(patient_id=patient_id, data_source=_frame(list(rows), patient_id))


class TestTabularMortalityPredictionEICU(unittest.TestCase):
    def setUp(self):
        self.task = TabularMortalityPredictionEICU()

    def test_schema_and_defaults(self):
        self.assertEqual(self.task.task_name, "TabularMortalityPredictionEICU")
        self.assertEqual(self.task.input_schema, {"features": "tensor"})
        self.assertEqual(self.task.output_schema, {"mortality": "binary"})
        self.assertEqual(self.task.features, list(DEFAULTS))
        self.assertEqual(len(self.task.features), 10)
        self.assertEqual(self.task.window_hours, 24.0)
        self.assertTrue(set(DEFAULTS) <= set(self.task.SUPPORTED_FEATURES))

    def test_one_sample_per_stay_with_its_own_labs_and_label(self):
        patient = _patient(
            _stay(10, status="Alive"),
            _stay(11, status="Expired"),
            _lab(60, "creatinine", 1.0, stay_id=10),
            _lab(60, "creatinine", 4.0, stay_id=11),
        )
        samples = self.task(patient)
        self.assertEqual([s["stay_id"] for s in samples], [10, 11])
        self.assertEqual([s["mortality"] for s in samples], [0, 1])
        self.assertEqual([s["features"][CREATININE] for s in samples], [1.0, 4.0])
        self.assertEqual({s["patient_id"] for s in samples}, {"p1"})
        self.assertTrue(all(len(s["features"]) == 10 for s in samples))

    def test_stay_without_discharge_status_is_skipped(self):
        patient = _patient(_stay(10, status=None), _stay(11, status=""), _stay(12))
        self.assertEqual([s["stay_id"] for s in self.task(patient)], [12])

    def test_first_and_last_values(self):
        # Rows are out of time order on purpose.
        patient = _patient(
            _stay(),
            _lab(600, "creatinine", 2.0),
            _lab(1200, "creatinine", 3.0),
            _lab(60, "creatinine", 1.0),
            _lab(300, "albumin", 2.5),
            _lab(30, "albumin", 3.1),
        )
        features = self.task(patient)[0]["features"]
        self.assertEqual(features[CREATININE], 3.0)  # last
        self.assertEqual(features[ALBUMIN], 3.1)  # first

    def test_values_outside_the_window_are_ignored(self):
        patient = _patient(
            _stay(),
            _lab(-30, "creatinine", 9.0),  # before ICU admission
            _lab(24 * 60 + 1, "creatinine", 8.0),  # after the window
            _lab(-1, "albumin", 9.0),
            _lab(0, "albumin", 3.0),  # the window starts at admission
            _lab(24 * 60, "lactate", 2.0),  # and includes its last minute
        )
        features = self.task(patient)[0]["features"]
        self.assertTrue(math.isnan(features[CREATININE]))
        self.assertEqual(features[ALBUMIN], 3.0)
        self.assertEqual(features[LACTATE], 2.0)

    def test_window_hours_sets_the_window(self):
        patient = _patient(
            _stay(), _lab(60, "creatinine", 1.0), _lab(600, "creatinine", 2.0)
        )
        short = TabularMortalityPredictionEICU(window_hours=6)
        self.assertEqual(short(patient)[0]["features"][CREATININE], 1.0)
        self.assertEqual(self.task(patient)[0]["features"][CREATININE], 2.0)

    def test_missing_and_non_numeric_labs_are_nan(self):
        patient = _patient(
            _stay(),
            _lab(60, "creatinine", "see comment"),
            _lab(60, "albumin", None),
            _lab(60, "troponin - I", 0.4),  # not a feature
        )
        features = self.task(patient)[0]["features"]
        for index in (CREATININE, ALBUMIN, LACTATE):
            self.assertTrue(math.isnan(features[index]))
        self.assertEqual(features[AGE], 60.0)

    def test_non_numeric_result_does_not_hide_a_numeric_one(self):
        patient = _patient(
            _stay(), _lab(60, "creatinine", 1.5), _lab(120, "creatinine", "pending")
        )
        self.assertEqual(self.task(patient)[0]["features"][CREATININE], 1.5)

    def test_age_and_sex(self):
        task = TabularMortalityPredictionEICU(features=["age", "is_female"])
        patient = _patient(
            _stay(10, age="> 89", gender="Female"),
            _stay(11, age="45", gender="Male"),
            _stay(12, age=None, gender="Unknown"),
        )
        first, second, third = (s["features"] for s in task(patient))
        self.assertEqual(first, [90.0, 1.0])
        self.assertEqual(second, [45.0, 0.0])
        self.assertTrue(all(math.isnan(value) for value in third))

    def test_group_fields_are_carried(self):
        patient = _patient(
            _stay(10, gender="Female", ethnicity="Hispanic"),
            _stay(11, gender=None, ethnicity=None),
        )
        samples = self.task(patient)
        self.assertEqual(
            [(s["gender"], s["ethnicity"]) for s in samples],
            [("Female", "Hispanic"), ("", "")],
        )

    def test_patient_without_stays_gives_no_samples(self):
        self.assertEqual(self.task(_patient(_lab(60, "creatinine", 1.0))), [])

    def test_custom_features_set_the_columns(self):
        task = TabularMortalityPredictionEICU(
            features=["bun_first_early", "is_female", "bun_last_early"]
        )
        patient = _patient(_stay(), _lab(10, "BUN", 20), _lab(500, "BUN", 35))
        self.assertEqual(task(patient)[0]["features"], [20.0, 0.0, 35.0])

    def test_invalid_arguments_raise(self):
        bad_features = {
            "unknown name": ["age", "heart_rate_max"],
            "empty": [],
            "repeated name": ["age", "age"],
        }
        for case, features in bad_features.items():
            with self.subTest(case=case), self.assertRaises(ValueError):
                TabularMortalityPredictionEICU(features=features)
        with self.assertRaises(TypeError):
            TabularMortalityPredictionEICU(features="age")
        for window_hours in (0, -1, float("nan"), float("inf"), "24", True):
            with self.subTest(window_hours=window_hours), self.assertRaises(ValueError):
                TabularMortalityPredictionEICU(window_hours=window_hours)
        with self.assertRaisesRegex(ValueError, "heart_rate_max"):
            TabularMortalityPredictionEICU(features=["heart_rate_max"])

    def test_arguments_are_task_attributes(self):
        # set_task builds its cache key from the task's attributes.
        default = vars(self.task)
        self.assertEqual(set(default), {"features", "window_hours"})
        self.assertNotEqual(
            vars(TabularMortalityPredictionEICU(window_hours=6)), default
        )
        self.assertNotEqual(
            vars(TabularMortalityPredictionEICU(features=["age"])), default
        )

    def test_pre_filter_requires_the_lab_table(self):
        df = _patient(_stay()).data_source.lazy().select(pl.exclude("^lab/.*$"))
        with self.assertRaisesRegex(ValueError, "requires the eICU lab table"):
            self.task.pre_filter(df)
        # Demographic features alone need no lab table.
        static = TabularMortalityPredictionEICU(features=["age", "is_female"])
        self.assertEqual(static.pre_filter(df).collect().height, 1)

    def test_pre_filter_keeps_only_the_rows_the_task_reads(self):
        patient = _patient(
            _stay(),
            _lab(60, "creatinine", 1.0),
            _lab(60, "troponin - I", 0.4),  # not a feature
            _lab(60, "sodium", 140),  # a feature, but not a default one
            _event("diagnosis", patientunitstayid=10, icd9code="038.9"),
        )
        kept = self.task.pre_filter(patient.data_source.lazy()).collect()
        self.assertEqual(sorted(kept["event_type"].to_list()), ["lab", "patient"])
        self.assertEqual(kept["lab/labname"].drop_nulls().to_list(), ["creatinine"])

    def test_samples_are_processed_by_the_schema(self):
        patient = _patient(
            _stay(10, status="Alive"),
            _stay(11, status="Expired"),
            _lab(60, "creatinine", 1.0, stay_id=10),
            _lab(60, "lactate", 5.0, stay_id=11),
        )
        samples = self.task(patient)
        dataset = create_sample_dataset(
            samples, self.task.input_schema, self.task.output_schema
        )
        self.assertEqual(len(dataset), 2)
        for index, sample in enumerate(samples):
            features = dataset[index]["features"]
            self.assertEqual(tuple(features.shape), (10,))
            self.assertEqual(features.dtype, torch.float32)
            # Missing values stay NaN for the model to handle.
            self.assertEqual(
                torch.isnan(features).tolist(),
                [math.isnan(value) for value in sample["features"]],
            )
            self.assertEqual(float(dataset[index]["mortality"]), sample["mortality"])
        self.assertEqual(float(dataset[0]["features"][CREATININE]), 1.0)
        self.assertEqual(float(dataset[1]["features"][LACTATE]), 5.0)


if __name__ == "__main__":
    unittest.main()

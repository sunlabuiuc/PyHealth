"""Tests for InHospitalMortality48hMIMIC3.

Builds a tiny, fully synthetic MIMIC-III-shaped set of CSVs (2-5 patients,
PATIENTS/ADMISSIONS/ICUSTAYS/CHARTEVENTS/LABEVENTS/D_LABITEMS) in a temp
directory and loads them with the real :class:`~pyhealth.datasets.
MIMIC3Dataset` and its default config -- no real/demo MIMIC-III data is
downloaded or read. The task is applied directly to ``Patient`` objects
(``dataset.get_patient(...)``) for most assertions, since that gives the raw
``List[Dict]`` output (the ``features`` list as plain floats) rather than
the tensor a ``set_task``/``TensorProcessor`` round trip would produce; one
test exercises ``set_task`` end to end to confirm the intended user path.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import polars as pl

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks.in_hospital_mortality_48h_mimic3 import (
    FEATURE_NAMES,
    MISSING,
    VARIABLES,
    InHospitalMortality48hMIMIC3,
    _hadm_key,
)

_IDX = {name: i for i, name in enumerate(VARIABLES)}
_N = len(VARIABLES)


def _mean_of(features, variable):
    return features[_IDX[variable]]


def _std_of(features, variable):
    return features[_N + _IDX[variable]]


def _write_csv(path: Path, rows: list, columns: list) -> None:
    data = {col: [row.get(col) for row in rows] for col in columns}
    pl.DataFrame(data, strict=False).write_csv(path)


class TestInHospitalMortality48hMIMIC3(unittest.TestCase):
    """Builds the synthetic MIMIC3Dataset and applies the task exactly once
    for the whole suite (class-scoped fixture): MIMIC3Dataset construction
    dominates per-test wall time, and every test below reads the same fixed
    synthetic cohort, so there is nothing to gain from rebuilding it per
    test method."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls._write_default_cohort(cls.temp_dir)
        cls.dataset = MIMIC3Dataset(
            root=str(cls.temp_dir), tables=["chartevents", "labevents"]
        )
        cls.by_subject = {}
        task = InHospitalMortality48hMIMIC3()
        for pid in cls.dataset.unique_patient_ids:
            samples = task(cls.dataset.get_patient(pid))
            if samples:
                cls.by_subject[int(pid)] = samples
        cls.sample_dataset = cls.dataset.set_task(InHospitalMortality48hMIMIC3())

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @staticmethod
    def _write_default_cohort(temp_dir):
        patients = [
            # p1: adult, 50h ICU stay, rich chartevents+labevents, survives.
            {"subject_id": 1, "gender": "F", "dob": "1980-01-01 00:00:00"},
            # p2: too young (age < 18 at admission).
            {"subject_id": 2, "gender": "M", "dob": "2140-01-01 00:00:00"},
            # p3: adult but short ICU stay (< 48h).
            {"subject_id": 3, "gender": "M", "dob": "1970-01-01 00:00:00"},
            # p4: adult, exactly 48h ICU stay, no chartevents/labevents at
            # all, dies in hospital.
            {"subject_id": 4, "gender": "F", "dob": "1960-01-01 00:00:00"},
        ]
        patient_cols = [
            "subject_id", "gender", "dob", "dod", "dod_hosp", "dod_ssn",
            "expire_flag",
        ]
        _write_csv(temp_dir / "PATIENTS.csv", patients, patient_cols)

        admissions = [
            {
                "subject_id": 1, "hadm_id": 100,
                "admittime": "2150-01-10 00:00:00",
                "dischtime": "2150-01-20 00:00:00",
                "hospital_expire_flag": 0,
            },
            {
                "subject_id": 2, "hadm_id": 200,
                "admittime": "2150-01-10 00:00:00",
                "dischtime": "2150-01-20 00:00:00",
                "hospital_expire_flag": 0,
            },
            {
                "subject_id": 3, "hadm_id": 300,
                "admittime": "2150-01-10 00:00:00",
                "dischtime": "2150-01-12 00:00:00",
                "hospital_expire_flag": 0,
            },
            {
                "subject_id": 4, "hadm_id": 400,
                "admittime": "2150-02-01 00:00:00",
                "dischtime": "2150-02-10 00:00:00",
                "hospital_expire_flag": 1,
            },
        ]
        admission_cols = [
            "subject_id", "hadm_id", "admission_type", "admission_location",
            "insurance", "language", "religion", "marital_status", "ethnicity",
            "edregtime", "edouttime", "deathtime", "diagnosis",
            "discharge_location", "dischtime", "admittime",
            "hospital_expire_flag",
        ]
        _write_csv(temp_dir / "ADMISSIONS.csv", admissions, admission_cols)

        icustays = [
            {
                "subject_id": 1, "hadm_id": 100, "icustay_id": 1000,
                "intime": "2150-01-10 00:00:00", "outtime": "2150-01-12 02:00:00",
            },
            {
                "subject_id": 2, "hadm_id": 200, "icustay_id": 2000,
                "intime": "2150-01-10 00:00:00", "outtime": "2150-01-13 00:00:00",
            },
            {
                "subject_id": 3, "hadm_id": 300, "icustay_id": 3000,
                "intime": "2150-01-10 00:00:00", "outtime": "2150-01-10 10:00:00",
            },
            {
                "subject_id": 4, "hadm_id": 400, "icustay_id": 4000,
                "intime": "2150-02-01 00:00:00", "outtime": "2150-02-03 00:00:00",
            },
        ]
        icustay_cols = [
            "subject_id", "hadm_id", "icustay_id", "intime", "first_careunit",
            "dbsource", "last_careunit", "outtime",
        ]
        _write_csv(temp_dir / "ICUSTAYS.csv", icustays, icustay_cols)

        chartevents = [
            # heart rate (itemid 211): two valid readings + one outlier.
            {"subject_id": 1, "hadm_id": 100, "icustay_id": 1000, "itemid": 211,
             "charttime": "2150-01-10 01:00:00", "valuenum": 80},
            {"subject_id": 1, "hadm_id": 100, "icustay_id": 1000, "itemid": 211,
             "charttime": "2150-01-10 02:00:00", "valuenum": 90},
            {"subject_id": 1, "hadm_id": 100, "icustay_id": 1000, "itemid": 211,
             "charttime": "2150-01-10 03:00:00", "valuenum": 999},  # outlier, dropped
            # capillary refill rate (itemid 3348): one categorical reading.
            {"subject_id": 1, "hadm_id": 100, "icustay_id": 1000, "itemid": 3348,
             "charttime": "2150-01-10 01:00:00", "value": "Normal <3 Secs"},
            # GCS eye opening (itemid 184): two categorical readings.
            {"subject_id": 1, "hadm_id": 100, "icustay_id": 1000, "itemid": 184,
             "charttime": "2150-01-10 01:00:00", "value": "4 Spontaneously"},
            {"subject_id": 1, "hadm_id": 100, "icustay_id": 1000, "itemid": 184,
             "charttime": "2150-01-10 05:00:00", "value": "To Pain"},
            # weight in lb (itemid 3581): one reading, unit-converted.
            {"subject_id": 1, "hadm_id": 100, "icustay_id": 1000, "itemid": 3581,
             "charttime": "2150-01-10 01:00:00", "valuenum": 154},
            # weight already in kg (itemid 3580): must NOT be lb-converted.
            {"subject_id": 1, "hadm_id": 100, "icustay_id": 1000, "itemid": 3580,
             "charttime": "2150-01-10 02:00:00", "valuenum": 70},
            # temperature in F (itemid 678): one reading, unit-converted.
            {"subject_id": 1, "hadm_id": 100, "icustay_id": 1000, "itemid": 678,
             "charttime": "2150-01-10 01:00:00", "valuenum": 98.6},
            # FiO2 charted as a percentage (itemid 3420): rescaled to a fraction.
            {"subject_id": 1, "hadm_id": 100, "icustay_id": 1000, "itemid": 3420,
             "charttime": "2150-01-10 01:00:00", "valuenum": 40},
            # SpO2 charted as a fraction (itemid 646): rescaled to a percentage.
            {"subject_id": 1, "hadm_id": 100, "icustay_id": 1000, "itemid": 646,
             "charttime": "2150-01-10 01:00:00", "valuenum": 0.95},
            # outside the 48h window: must not be counted.
            {"subject_id": 1, "hadm_id": 100, "icustay_id": 1000, "itemid": 211,
             "charttime": "2150-01-12 01:00:00", "valuenum": 40},
        ]
        chart_cols = [
            "subject_id", "hadm_id", "icustay_id", "itemid", "charttime",
            "value", "valuenum", "valueuom",
        ]
        _write_csv(temp_dir / "CHARTEVENTS.csv", chartevents, chart_cols)

        labevents = [
            # glucose (lab itemid 50931): one reading.
            {"subject_id": 1, "hadm_id": 100, "itemid": 50931,
             "charttime": "2150-01-10 02:00:00", "valuenum": 110},
        ]
        lab_cols = [
            "subject_id", "hadm_id", "itemid", "charttime", "value",
            "valuenum", "valueuom", "flag",
        ]
        _write_csv(temp_dir / "LABEVENTS.csv", labevents, lab_cols)
        d_labitems = [
            {
                "itemid": 50931,
                "label": "GLUCOSE",
                "fluid": "BLOOD",
                "category": "CHEMISTRY",
            }
        ]
        _write_csv(
            temp_dir / "D_LABITEMS.csv",
            d_labitems,
            ["itemid", "label", "fluid", "category"],
        )

    def test_feature_names_length(self):
        self.assertEqual(len(FEATURE_NAMES), 34)
        self.assertEqual(len(VARIABLES), 17)

    def test_age_and_los_exclusions(self):
        by_subject = self.by_subject
        # p2 (age < 18) and p3 (LOS < 48h) produce no samples.
        self.assertNotIn(2, by_subject)
        self.assertNotIn(3, by_subject)
        # p1 and p4 (LOS == 48h exactly) are both included.
        self.assertIn(1, by_subject)
        self.assertIn(4, by_subject)

    def test_patient4_all_missing_and_died(self):
        sample = self.by_subject[4][0]
        self.assertEqual(sample["mortality"], 1)
        self.assertEqual(sample["features"], [MISSING] * 34)

    def test_heart_rate_mean_std_excludes_outlier_and_out_of_window(self):
        sample = self.by_subject[1][0]
        # Only the two in-window, in-range readings (80, 90) count.
        self.assertAlmostEqual(_mean_of(sample["features"], "heart_rate"), 85.0)
        self.assertAlmostEqual(
            _std_of(sample["features"], "heart_rate"), 50.0 ** 0.5, places=6
        )

    def test_single_reading_mean_set_std_missing(self):
        features = self.by_subject[1][0]["features"]
        self.assertAlmostEqual(
            _mean_of(features, "capillary_refill_rate"), 0.0
        )  # "Normal <3 Secs" -> 0.0
        self.assertEqual(_std_of(features, "capillary_refill_rate"), MISSING)

    def test_categorical_gcs_mapping_and_stats(self):
        features = self.by_subject[1][0]["features"]
        # "4 Spontaneously" -> 4.0, "To Pain" -> 2.0
        self.assertAlmostEqual(
            _mean_of(features, "glascow_coma_scale_eye_opening"), 3.0
        )
        self.assertAlmostEqual(
            _std_of(features, "glascow_coma_scale_eye_opening"), 2.0 ** 0.5, places=6
        )

    def test_unit_conversions(self):
        features = self.by_subject[1][0]["features"]
        # Two readings: 154 lb (itemid 3581, converted) and 70 kg (itemid
        # 3580, already kg -- must NOT be converted a second time).
        self.assertAlmostEqual(
            _mean_of(features, "weight"), (154 * 0.45359237 + 70) / 2, places=4
        )
        self.assertAlmostEqual(_mean_of(features, "temperature"), 37.0, places=4)

    def test_weight_already_in_kg_is_not_double_converted(self):
        """Regression test for itemid 3580 ("Present Weight (kg)"), which
        was previously misclassified as pounds and lb-converted."""
        features = self.by_subject[1][0]["features"]
        mean = _mean_of(features, "weight")
        # If 70 kg were wrongly treated as 70 lb, the mean would collapse
        # toward ~45% of its correct value; both raw readings (69.86, 70)
        # are close together, so the mean must land near them, not near
        # (69.86 + 70*0.4536) / 2 =~ 50.8.
        self.assertGreater(mean, 65.0)

    def test_fio2_percentage_reading_is_rescaled_to_fraction(self):
        features = self.by_subject[1][0]["features"]
        self.assertAlmostEqual(
            _mean_of(features, "fraction_inspired_oxygen"), 0.40, places=4
        )

    def test_oxygen_saturation_fraction_reading_is_rescaled_to_percentage(self):
        features = self.by_subject[1][0]["features"]
        self.assertAlmostEqual(
            _mean_of(features, "oxygen_saturation"), 95.0, places=4
        )

    def test_lab_sourced_value(self):
        features = self.by_subject[1][0]["features"]
        self.assertAlmostEqual(_mean_of(features, "glucose"), 110.0)
        self.assertEqual(_std_of(features, "glucose"), MISSING)

    def test_unmeasured_variable_is_missing(self):
        features = self.by_subject[1][0]["features"]
        for variable in ("diastolic_blood_pressure", "systolic_blood_pressure", "ph"):
            self.assertEqual(_mean_of(features, variable), MISSING)
            self.assertEqual(_std_of(features, variable), MISSING)

    def test_set_task_integration_yields_expected_count(self):
        self.assertEqual(len(self.sample_dataset), 2)


class TestHadmKey(unittest.TestCase):
    """Unit tests for the hadm_id join-key coercion helper."""

    def test_int_float_and_numeric_string_all_coerce_equal(self):
        self.assertEqual(_hadm_key(100), _hadm_key(100.0))
        self.assertEqual(_hadm_key("100"), _hadm_key(100))
        self.assertEqual(_hadm_key("100.0"), _hadm_key(100))

    def test_none_coerces_to_none_without_raising(self):
        """Regression test: a naive int(float(value)) raises TypeError on
        None, which would crash __call__ for every patient instead of just
        skipping the one record with a missing hadm_id."""
        self.assertIsNone(_hadm_key(None))


class TestHadmIdCrossTableTypeMismatch(unittest.TestCase):
    """Regression test: ADMISSIONS and ICUSTAYS are scanned as independent
    tables, so their hadm_id columns can be inferred with different dtypes
    on real data (one file writes "500", the other "500.0"). Without the
    _hadm_key coercion, the dict-keyed join would silently miss the match
    and the (otherwise eligible) ICU stay would be dropped with no error.
    """

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        patient_cols = [
            "subject_id", "gender", "dob", "dod", "dod_hosp", "dod_ssn",
            "expire_flag",
        ]
        _write_csv(
            self.temp_dir / "PATIENTS.csv",
            [{"subject_id": 9, "gender": "F", "dob": "1980-01-01 00:00:00"}],
            patient_cols,
        )
        _write_csv(
            self.temp_dir / "ADMISSIONS.csv",
            # hadm_id written without a decimal point: likely inferred Int64.
            [{
                "subject_id": 9, "hadm_id": "500",
                "admittime": "2150-01-10 00:00:00",
                "dischtime": "2150-01-20 00:00:00",
                "hospital_expire_flag": 0,
            }],
            [
                "subject_id", "hadm_id", "admission_type", "admission_location",
                "insurance", "language", "religion", "marital_status",
                "ethnicity", "edregtime", "edouttime", "deathtime", "diagnosis",
                "discharge_location", "dischtime", "admittime",
                "hospital_expire_flag",
            ],
        )
        _write_csv(
            self.temp_dir / "ICUSTAYS.csv",
            # Same hadm_id, written WITH a decimal point: likely Float64.
            [{
                "subject_id": 9, "hadm_id": "500.0", "icustay_id": 9000,
                "intime": "2150-01-10 00:00:00", "outtime": "2150-01-12 02:00:00",
            }],
            [
                "subject_id", "hadm_id", "icustay_id", "intime",
                "first_careunit", "dbsource", "last_careunit", "outtime",
            ],
        )
        _write_csv(
            self.temp_dir / "CHARTEVENTS.csv",
            [],
            [
                "subject_id", "hadm_id", "icustay_id", "itemid", "charttime",
                "value", "valuenum", "valueuom",
            ],
        )

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_mismatched_hadm_id_dtypes_still_join(self):
        dataset = MIMIC3Dataset(root=str(self.temp_dir), tables=["chartevents"])
        task = InHospitalMortality48hMIMIC3()
        samples = []
        for pid in dataset.unique_patient_ids:
            samples.extend(task(dataset.get_patient(pid)))
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["mortality"], 0)


class TestMissingHadmIdDoesNotCrash(unittest.TestCase):
    """Regression test: an icustay row with no hadm_id must be skipped, not
    crash __call__ for the whole patient (and therefore set_task for every
    patient). This guards the None-handling in _hadm_key specifically."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        patient_cols = [
            "subject_id", "gender", "dob", "dod", "dod_hosp", "dod_ssn",
            "expire_flag",
        ]
        _write_csv(
            self.temp_dir / "PATIENTS.csv",
            [{"subject_id": 7, "gender": "F", "dob": "1980-01-01 00:00:00"}],
            patient_cols,
        )
        _write_csv(self.temp_dir / "ADMISSIONS.csv", [], [
            "subject_id", "hadm_id", "admission_type", "admission_location",
            "insurance", "language", "religion", "marital_status",
            "ethnicity", "edregtime", "edouttime", "deathtime", "diagnosis",
            "discharge_location", "dischtime", "admittime",
            "hospital_expire_flag",
        ])
        _write_csv(
            self.temp_dir / "ICUSTAYS.csv",
            # hadm_id intentionally omitted (null cell).
            [{
                "subject_id": 7, "icustay_id": 7000,
                "intime": "2150-01-10 00:00:00", "outtime": "2150-01-12 02:00:00",
            }],
            [
                "subject_id", "hadm_id", "icustay_id", "intime",
                "first_careunit", "dbsource", "last_careunit", "outtime",
            ],
        )
        _write_csv(
            self.temp_dir / "CHARTEVENTS.csv",
            [],
            [
                "subject_id", "hadm_id", "icustay_id", "itemid", "charttime",
                "value", "valuenum", "valueuom",
            ],
        )

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_missing_hadm_id_is_skipped_not_raised(self):
        dataset = MIMIC3Dataset(root=str(self.temp_dir), tables=["chartevents"])
        task = InHospitalMortality48hMIMIC3()
        samples = []
        for pid in dataset.unique_patient_ids:
            samples.extend(task(dataset.get_patient(pid)))  # must not raise
        self.assertEqual(samples, [])


if __name__ == "__main__":
    unittest.main()

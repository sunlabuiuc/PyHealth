"""Tests for ClinicalPresenceMortalityMIMIC3's cohort extraction.

The fixture is a small frame shaped like MIMIC3Dataset's ``global_event_df``
(string attributes, nanosecond timestamps), so no MIMIC data is needed.
"""

import unittest
from datetime import datetime, timedelta

import polars as pl

from pyhealth.tasks.clinical_presence_mortality_mimic3 import (
    GROUP_FLAGS,
    LAB_NAMES,
    _extract_cohort,
)

T0 = datetime(2150, 6, 1, 8, 0)
EARLY = T0 - timedelta(days=30)
FMT = "%Y-%m-%d %H:%M:%S"

SCHEMA = {
    "patient_id": pl.String,
    "event_type": pl.String,
    "timestamp": pl.Datetime("ns"),
    "patients/gender": pl.String,
    "patients/dob": pl.String,
    "admissions/hadm_id": pl.String,
    "admissions/dischtime": pl.String,
    "admissions/hospital_expire_flag": pl.String,
    "admissions/ethnicity": pl.String,
    "admissions/insurance": pl.String,
    "labevents/itemid": pl.String,
    "labevents/valuenum": pl.String,
}

# Itemids: both glucose ones map to "Glucose"; 50800 is outside the 67 labs
GLUCOSE_A, GLUCOSE_B = 50809, 50931
SODIUM, POTASSIUM, CHLORIDE, CREATININE, HEMOGLOBIN = 50983, 50971, 50902, 50912, 51222
UNMAPPED = 50800


def _patient(pid, age=60, gender="M"):
    return {
        "patient_id": pid,
        "event_type": "patients",
        "patients/gender": gender,
        "patients/dob": (T0 - timedelta(days=365 * age)).strftime(FMT),
    }


def _admission(
    pid,
    hadm,
    start=T0,
    stay_hours=72,
    died=False,
    ethnicity="WHITE",
    insurance="Medicare",
):
    return {
        "patient_id": pid,
        "event_type": "admissions",
        "timestamp": start,
        "admissions/hadm_id": hadm,
        "admissions/dischtime": (start + timedelta(hours=stay_hours)).strftime(FMT),
        "admissions/hospital_expire_flag": "1" if died else "0",
        "admissions/ethnicity": ethnicity,
        "admissions/insurance": insurance,
    }


def _lab(pid, hours, itemid, value, start=T0):
    return {
        "patient_id": pid,
        "event_type": "labevents",
        "timestamp": start + timedelta(hours=hours),
        "labevents/itemid": str(itemid),
        "labevents/valuenum": None if value is None else str(value),
    }


def _simple(pid, hadm, age=60, gender="M", **admission):
    """A patient with one admission and one sodium value at +2 h."""
    return [
        _patient(pid, age=age, gender=gender),
        _admission(pid, hadm, **admission),
        _lab(pid, 2, SODIUM, 140),
    ]


def cohort_events() -> pl.DataFrame:
    """One patient per cohort rule; comments give the expected outcome."""
    rows = [
        # 1: survives; orders at +2 h and +6 h, other rows don't count
        _patient("1"),
        _admission("1", "101"),
        _lab("1", 6, SODIUM, 138),  # listed first but later, so kept
        _lab("1", 2, SODIUM, 140),
        _lab("1", 2, POTASSIUM, 4.1),
        _lab("1", 7, UNMAPPED, 5),
        _lab("1", 8, CHLORIDE, None),
        _lab("1", 30, CREATININE, 1.2),  # outside the window
        # 2: dies on day 5 -> 1; 3: dies on day 9 -> 0; 14: day 8 exactly -> 0
        *_simple("2", "201", stay_hours=5 * 24, died=True),
        *_simple("3", "301", stay_hours=9 * 24, died=True),
        *_simple("14", "1401", stay_hours=8 * 24, died=True),
        # 4: aged 10 -> excluded
        *_simple("4", "401", age=10),
        # 5: last admission lasts 12 h -> excluded, though the first qualifies
        _patient("5"),
        _admission("5", "501", start=EARLY, stay_hours=48),
        _lab("5", 2, SODIUM, 139, start=EARLY),
        _admission("5", "502", stay_hours=12),
        _lab("5", 2, SODIUM, 137),
        # 6: two admissions with labs, the later listed first -> 602's labs only
        _patient("6"),
        _admission("6", "602"),
        _lab("6", 2, SODIUM, 145),
        _admission("6", "601", start=EARLY),
        _lab("6", 2, SODIUM, 130, start=EARLY),
        _lab("6", 3, HEMOGLOBIN, 9.0, start=EARLY),
        # 7: labs only at exactly +0 h and +24 h -> excluded
        _patient("7"),
        _admission("7", "701"),
        _lab("7", 0, SODIUM, 140),
        _lab("7", 24, SODIUM, 141),
        # 8: same-time glucose pair and a potassium with a null copy are dropped
        _patient("8"),
        _admission("8", "801"),
        _lab("8", 1, GLUCOSE_A, 90),
        _lab("8", 3, GLUCOSE_A, 100),
        _lab("8", 3, GLUCOSE_B, 200),
        _lab("8", 4, POTASSIUM, 4.0),
        _lab("8", 4, POTASSIUM, None),
        # 9: every lab value null -> excluded
        _patient("9"),
        _admission("9", "901"),
        _lab("9", 2, SODIUM, None),
        _lab("9", 3, POTASSIUM, None),
        # 10: DOB shifted ~300 years, as MIMIC does for patients over 89
        *_simple("10", "1001", age=300),
        # 11-13: group flags
        *_simple("11", "1101", ethnicity="BLACK/HAITIAN"),
        *_simple("12", "1201", insurance="Self Pay"),
        *_simple("13", "1301", gender="F", insurance="Private"),
        # 15: stays exactly 24 h -> included
        *_simple("15", "1501", stay_hours=24),
    ]
    return pl.DataFrame(rows, schema=SCHEMA)


class TestExtractCohort(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = cohort_events()
        cls.cohort = _extract_cohort(cls.events.lazy())

    def row(self, pid):
        rows = self.cohort.filter(pl.col("patient_id") == pid).to_dicts()
        self.assertEqual(len(rows), 1, f"patient {pid} not in the cohort once")
        return rows[0]

    def test_cohort_membership(self):
        ids = self.cohort["patient_id"].to_list()
        self.assertEqual(
            ids, ["1", "2", "3", "6", "8", "10", "11", "12", "13", "14", "15"]
        )
        self.assertEqual(self.row("6")["hadm_id"], "602")
        self.assertEqual(
            self.cohort.columns,
            [
                "patient_id",
                "hadm_id",
                "mortality",
                *GROUP_FLAGS,
                "n_lab_orders",
                *LAB_NAMES,
            ],
        )

    def test_label_horizon(self):
        labels = dict(self.cohort.select("patient_id", "mortality").iter_rows())
        self.assertEqual(labels["1"], 0)
        self.assertEqual(labels["2"], 1)
        self.assertEqual(labels["3"], 0)
        self.assertEqual(labels["14"], 0)
        longer = _extract_cohort(self.events.lazy(), horizon_days=10)
        self.assertEqual(
            longer.filter(pl.col("patient_id") == "3")["mortality"].item(), 1
        )

    def test_window_dedup_latest(self):
        p1, p6, p8 = self.row("1"), self.row("6"), self.row("8")
        self.assertEqual(p1["Sodium"], 138.0)
        self.assertEqual(p1["Potassium"], 4.1)
        self.assertIsNone(p1["Creatinine"])
        self.assertIsNone(p1["Chloride"])
        self.assertEqual(p6["Sodium"], 145.0)
        self.assertIsNone(p6["Hemoglobin"])
        self.assertEqual(p8["Glucose"], 90.0)
        self.assertIsNone(p8["Potassium"])
        observed = {lab for lab in LAB_NAMES if p1[lab] is not None}
        self.assertEqual(observed, {"Sodium", "Potassium"})

    def test_group_flags(self):
        flags = {
            pid: tuple(self.row(pid)[g] for g in GROUP_FLAGS)
            for pid in ("1", "11", "12", "13")
        }
        self.assertEqual(flags["1"], (0, 0, 1))
        self.assertEqual(flags["11"], (1, 0, 1))
        self.assertEqual(flags["12"], (0, 0, 1))
        self.assertEqual(flags["13"], (0, 1, 0))

    def test_lab_order_counts(self):
        self.assertEqual(self.row("1")["n_lab_orders"], 2)
        self.assertEqual(self.row("6")["n_lab_orders"], 1)
        self.assertEqual(self.row("8")["n_lab_orders"], 1)

    def test_missing_labevents_raises(self):
        no_labs = self.events.drop("labevents/itemid", "labevents/valuenum")
        with self.assertRaisesRegex(ValueError, r"tables=\['labevents'\]"):
            _extract_cohort(no_labs.lazy())


if __name__ == "__main__":
    unittest.main()

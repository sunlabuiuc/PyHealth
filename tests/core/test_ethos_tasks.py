"""Tests for EthosTimelineMEDS, EthosOutcomeMEDS and EthosRolloutRule.

Tasks run directly on an in-memory ``Patient``. All data here is synthetic.
"""

import unittest
from datetime import datetime, timedelta

import polars as pl

from pyhealth.data import Patient
from pyhealth.tasks import (
    EthosOutcomeMEDS,
    EthosRolloutRule,
    EthosTimelineMEDS,
    HospitalReadmissionMEDS,
    ICUAdmissionMEDS,
    ProlongedLengthOfStayMEDS,
)

T0 = datetime.fromisoformat("2024-01-01T08:00:00")
ADMIT = "HOSPITAL_ADMISSION//ER"
HOME = "HOSPITAL_DISCHARGE//HOME"
DIED = "HOSPITAL_DISCHARGE//DIED"
ICU = "ICU_ADMISSION//MICU"
DEATH = "MEDS_DEATH"
GOT = "token_of_interest"


def patient(*rows):
    """Rows are ``(hours or None, code, hadm_id, value)``."""
    frame = pl.DataFrame(
        {
            "patient_id": ["1"] * len(rows),
            "event_type": ["meds"] * len(rows),
            "timestamp": pl.Series(
                [None if r[0] is None else T0 + timedelta(hours=r[0]) for r in rows],
                dtype=pl.Datetime("ms"),
            ),
            "meds/code": pl.Series([r[1] for r in rows], dtype=pl.String),
            "meds/hadm_id": pl.Series([r[2] for r in rows], dtype=pl.Int64),
            "meds/numeric_value": pl.Series([r[3] for r in rows], dtype=pl.Float64),
        }
    )
    return Patient(patient_id="1", data_source=frame)


STAYS = patient(
    (None, "GENDER//F", None, None),
    (-1000.0, "MEDS_BIRTH", None, None),
    (-5.0, "LAB//EARLY", None, 3.0),
    (0.0, ADMIT, 1, None),
    (2.0, ICU, 1, None),
    (3.0, "LAB//DURING", 1, 7.0),
    (48.0, HOME, 1, None),
    (48.0, "DX//AT_DISCHARGE", 1, None),
    (96.0, ADMIT, 2, None),
    (400.0, DIED, 2, None),
    (400.0, DEATH, None, None),
)


class TestEthosTimelineMEDS(unittest.TestCase):
    def test_whole_timeline_and_static_split(self):
        samples = EthosTimelineMEDS()(STAYS)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["patient_id"], "1")
        self.assertEqual(len(sample["timeline"]["code"]), 9)
        self.assertIn(DEATH, sample["timeline"]["code"])
        self.assertNotIn("GENDER//F", sample["timeline"]["code"])
        self.assertEqual(sample["timeline"]["numeric_value"][0], 3.0)
        self.assertEqual(
            sorted(sample["context"]["code"]), ["GENDER//F", "MEDS_BIRTH"]
        )
        self.assertEqual(sample["context"]["start"], T0 + timedelta(hours=-5))

    def test_patient_with_only_static_rows_has_no_sample(self):
        only_static = patient((None, "GENDER//F", None, None))
        self.assertEqual(EthosTimelineMEDS()(only_static), [])


class TestEthosOutcomeMEDS(unittest.TestCase):
    def test_keeps_the_cohort_labels_and_codes_of_the_outcome_task(self):
        for task_class in (
            ICUAdmissionMEDS,
            HospitalReadmissionMEDS,
            ProlongedLengthOfStayMEDS,
        ):
            with self.subTest(task=task_class.__name__):
                outcomes = task_class()(STAYS)
                samples = EthosOutcomeMEDS(task_class)(STAYS)
                label = next(iter(task_class.output_schema))
                self.assertEqual(len(samples), len(outcomes))
                for sample, outcome in zip(samples, outcomes):
                    self.assertNotIn("codes", sample)
                    self.assertEqual(sample[label], outcome[label])
                    self.assertEqual(sample["hadm_id"], outcome["hadm_id"])
                    rows = sample["timeline"]["code"] + sample["context"]["code"]
                    self.assertEqual(sorted(rows), sorted(outcome["codes"]))

    def test_admission_cut_excludes_the_stay(self):
        sample = EthosOutcomeMEDS(ICUAdmissionMEDS)(STAYS)[0]
        self.assertEqual(sample["timeline"]["code"], ["LAB//EARLY", ADMIT])
        self.assertEqual(sample["timeline"]["numeric_value"], [3.0, None])

    def test_discharge_cut_excludes_what_follows(self):
        sample = EthosOutcomeMEDS(HospitalReadmissionMEDS)(STAYS)[0]
        codes = sample["timeline"]["code"]
        self.assertEqual(codes[-2:], [HOME, "DX//AT_DISCHARGE"])
        self.assertNotIn(DIED, codes)
        self.assertNotIn(DEATH, codes)

    def test_patient_without_a_stay_has_no_sample(self):
        lab_only = patient((0.0, "LAB//X", None, 1.0))
        self.assertEqual(EthosOutcomeMEDS(ICUAdmissionMEDS)(lab_only), [])

    def test_names_and_schema(self):
        task = EthosOutcomeMEDS(ProlongedLengthOfStayMEDS, threshold_days=7.0)
        self.assertEqual(task.task_name, "EthosProlongedLengthOfStayMEDS")
        self.assertEqual(task.output_schema, {"prolonged_los": "binary"})
        self.assertEqual(task.input_schema, EthosTimelineMEDS.input_schema)

    def test_rejects_other_tasks(self):
        with self.assertRaises(ValueError):
            EthosOutcomeMEDS(EthosTimelineMEDS)


class TestRolloutRules(unittest.TestCase):
    def test_icu_admission(self):
        rule = EthosOutcomeMEDS(ICUAdmissionMEDS).rule
        self.assertIs(rule.score(ICU, GOT, 1.0), True)
        self.assertIs(rule.score(DEATH, GOT, 1.0), True)
        self.assertIs(rule.score(DIED, GOT, 1.0), True)
        self.assertIs(rule.score(HOME, GOT, 1.0), False)
        self.assertIsNone(rule.score("TIMELINE_END", GOT, 1.0))
        self.assertIsNone(rule.score("LAB//X", "time_limit", 800.0))
        self.assertEqual(rule.time_limit_days, 730.5)

    def test_icu_admission_without_deaths(self):
        rule = EthosOutcomeMEDS(ICUAdmissionMEDS, death_as_positive=False).rule
        self.assertIs(rule.score(ICU, GOT, 1.0), True)
        self.assertIs(rule.score(DEATH, GOT, 1.0), False)
        self.assertIs(rule.score(DIED, GOT, 1.0), False)
        self.assertEqual(
            rule.stop_prefixes,
            EthosOutcomeMEDS(ICUAdmissionMEDS).rule.stop_prefixes,
        )

    def test_prolonged_stay_threshold_is_inclusive(self):
        rule = EthosOutcomeMEDS(ProlongedLengthOfStayMEDS).rule
        self.assertIs(rule.score(HOME, GOT, 10.0), True)
        self.assertIs(rule.score(DEATH, GOT, 11.0), True)
        self.assertIs(rule.score(HOME, GOT, 9.9), False)
        self.assertIs(rule.score(DEATH, GOT, 9.9), False)
        self.assertIsNone(rule.score("TIMELINE_END", GOT, 11.0))
        custom = EthosOutcomeMEDS(ProlongedLengthOfStayMEDS, threshold_days=3.0).rule
        self.assertIs(custom.score(HOME, GOT, 3.0), True)

    def test_readmission(self):
        rule = EthosOutcomeMEDS(HospitalReadmissionMEDS).rule
        self.assertEqual(rule.time_limit_days, 30.0)
        self.assertIs(rule.score(ADMIT, GOT, 1.0), True)
        self.assertIs(rule.score(DEATH, GOT, 1.0), True)
        self.assertIs(rule.score("TIMELINE_END", GOT, 1.0), False)
        self.assertIs(rule.score("LAB//X", "time_limit", 31.0), False)
        custom = EthosOutcomeMEDS(HospitalReadmissionMEDS, window_days=7.0).rule
        self.assertEqual(custom.time_limit_days, 7.0)

    def test_full_followup_does_not_change_the_rollout(self):
        self.assertEqual(
            EthosOutcomeMEDS(HospitalReadmissionMEDS).rule,
            EthosOutcomeMEDS(HospitalReadmissionMEDS, require_full_followup=True).rule,
        )

    def test_failed_rollouts_are_always_dropped(self):
        rule = EthosRolloutRule(stop_prefixes=("X",), keep_time_limited=True)
        self.assertIsNone(rule.score(None, "key_error", 0.0))
        self.assertIsNone(rule.score("X", "token_limit", 0.0))


if __name__ == "__main__":
    unittest.main()

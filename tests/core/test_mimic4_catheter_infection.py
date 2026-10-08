import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
import math

import polars as pl

from pyhealth.data import Event, Patient
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks.catheter_infection import (
    CatheterAssociatedInfectionPredictionMIMIC4,
    CatheterAssociatedInfectionPredictionMIMIC4Temporal,
    CatheterAssociatedInfectionPredictionStageNetMIMIC4,
    CatheterAssociatedInfectionPredictionStageNetMIMIC4Temporal,
    _catheter_episodes,
    _collect_catheter_days,
    _episode_index_date,
    _onset_eligible,
    _parse_colony_count,
    _summarize_urine_specimens,
)


class TestMIMIC4CatheterInfectionPrediction(unittest.TestCase):
    """Dataset-backed tests using synthetic rows in mimic4demo CSV files."""

    def setUp(self):
        test_dir = Path(__file__).parent.parent.parent
        self.demo_dataset_path = str(
            test_dir / "test-resources" / "core" / "mimic4demo"
        )
        tables = ["diagnoses_icd", "procedures_icd", "labevents"]
        self.dataset = MIMIC4Dataset(
            ehr_root=self.demo_dataset_path,
            ehr_tables=tables,
        )

    @staticmethod
    def _to_int(value):
        if hasattr(value, "item"):
            return int(value.item())
        return int(value)

    def test_helper_code_matching(self):
        task = CatheterAssociatedInfectionPredictionMIMIC4()

        self.assertTrue(task._is_catheter_code("Y84.6", 10))
        self.assertTrue(task._is_catheter_code("0T9B70Z", "10"))
        self.assertTrue(task._is_catheter_code("996.31", 9))
        self.assertTrue(task._is_catheter_code("37.22", 9))
        self.assertFalse(task._is_catheter_code("Y84.6", 9))

        self.assertTrue(task._is_infection_code("T83.511A", 10))
        self.assertTrue(task._is_infection_code("T83518D", "10"))
        self.assertTrue(task._is_infection_code("996.64", 9))
        self.assertFalse(task._is_infection_code("T83.511A", 9))
        self.assertFalse(task._is_infection_code("N39.0", 10))

    def test_synthetic_patient_outcomes(self):
        task = CatheterAssociatedInfectionPredictionMIMIC4()
        sample_dataset = self.dataset.set_task(task)

        labels_by_patient = {
            self._to_int(sample["patient_id"]): self._to_int(sample["label"])
            for sample in sample_dataset
        }

        # Positive case: catheter first, later infection admission.
        self.assertIn(91001, labels_by_patient)
        self.assertEqual(labels_by_patient[91001], 1)

        # Negative case: catheter first, no later infection.
        self.assertIn(91002, labels_by_patient)
        self.assertEqual(labels_by_patient[91002], 0)

        # Excluded case: infection before catheter evidence.
        self.assertNotIn(91003, labels_by_patient)

    def test_synthetic_patient_outcomes_stagenet_variant(self):
        task = CatheterAssociatedInfectionPredictionStageNetMIMIC4()
        sample_dataset = self.dataset.set_task(task)

        labels_by_patient = {
            self._to_int(sample["patient_id"]): self._to_int(sample["label"])
            for sample in sample_dataset
        }

        self.assertIn(91001, labels_by_patient)
        self.assertEqual(labels_by_patient[91001], 1)

        self.assertIn(91002, labels_by_patient)
        self.assertEqual(labels_by_patient[91002], 0)

        self.assertNotIn(91003, labels_by_patient)

    def test_missing_defaults(self):
        task = CatheterAssociatedInfectionPredictionMIMIC4()

        self.assertEqual(task._ensure_nonempty_sequence([]), ["<missing>"])

        empty_lab_df = pl.DataFrame()
        lab_vector = task._build_lab_vector(empty_lab_df)
        self.assertEqual(len(lab_vector), len(task.LAB_CATEGORY_ORDER))
        self.assertTrue(all(v == 0.0 for v in lab_vector))

    def test_no_nan_labs_in_nested_outputs(self):
        task = CatheterAssociatedInfectionPredictionMIMIC4()
        sample_dataset = self.dataset.set_task(task)

        for sample in sample_dataset:
            for visit_labs in sample["labs"]:
                self.assertFalse(any(math.isnan(v) for v in visit_labs))


def _make_patient(patient_id, events):
    """Build an in-memory Patient from (event_type, timestamp, attrs) tuples."""
    rows = []
    for event_type, timestamp, attrs in events:
        row = {"patient_id": patient_id, "event_type": event_type, "timestamp": timestamp}
        row.update({f"{event_type}/{k}": v for k, v in attrs.items()})
        rows.append(row)
    columns = sorted({k for row in rows for k in row} - {"patient_id", "event_type", "timestamp"})
    schema = {"patient_id": pl.Utf8, "event_type": pl.Utf8, "timestamp": pl.Datetime("ms")}
    schema.update({c: pl.Utf8 for c in columns})
    df = pl.DataFrame(
        [{c: row.get(c) for c in schema} for row in rows], schema=schema
    )
    return Patient(patient_id=patient_id, data_source=df)


class TestCatheterTemporalHelpers(unittest.TestCase):
    """Pure-function tests for the temporal (NHSN-aligned) CAUTI helpers."""

    ADMIT = date(2150, 1, 1)

    def _d(self, offset):
        return self.ADMIT + timedelta(days=offset)

    def test_episode_split_on_gap(self):
        days = [self._d(0), self._d(1), self._d(3), self._d(4)]
        self.assertEqual(
            _catheter_episodes(days),
            [(self._d(0), self._d(1)), (self._d(3), self._d(4))],
        )

    def test_catheter_day_eligibility(self):
        # Placed day 1, removed day 2 → never reaches catheter day 3.
        self.assertIsNone(_episode_index_date((self._d(0), self._d(1)), self.ADMIT))
        # Placed day 1, in place through day 3 → eligible on day 3.
        self.assertEqual(
            _episode_index_date((self._d(0), self._d(2)), self.ADMIT), self._d(2)
        )

    def test_onset_removed_day_before(self):
        episodes = [(self._d(0), self._d(3))]
        self.assertTrue(_onset_eligible(self._d(4), episodes, self.ADMIT))
        self.assertFalse(_onset_eligible(self._d(5), episodes, self.ADMIT))
        # Catheter day 2 is too early.
        self.assertFalse(_onset_eligible(self._d(1), episodes, self.ADMIT))

    def test_untimed_catheter_never_eligible(self):
        # Hard requirement: no timed catheter episode → no eligible onset.
        self.assertFalse(_onset_eligible(self._d(1), [], self.ADMIT))
        self.assertFalse(_onset_eligible(self._d(5), [], self.ADMIT))

    def test_discharge_before_eligible_day(self):
        episode = (self._d(0), self._d(2))
        # Discharged on hospital day 2 → never reaches an eligible inpatient day.
        self.assertIsNone(
            _episode_index_date(episode, self.ADMIT, disch_date=self._d(1))
        )
        self.assertEqual(
            _episode_index_date(episode, self.ADMIT, disch_date=self._d(2)),
            self._d(2),
        )

    def test_cpt_51701_not_a_catheter_day(self):
        hcpcs = [
            Event("hcpcsevents", datetime(2150, 1, 2), hadm_id="1", hcpcs_cd="51701"),
            Event("hcpcsevents", datetime(2150, 1, 3), hadm_id="1", hcpcs_cd="51702"),
        ]
        days, sources = _collect_catheter_days(self.ADMIT, self._d(9), hcpcs)
        self.assertEqual(days, {self._d(2)})
        self.assertEqual(sources, {"cpt", "cpt_nonindwelling"})

    def test_icu_foley_days(self):
        proc = [
            Event(
                "procedureevents",
                datetime(2150, 1, 1, 10),
                hadm_id="1",
                itemid="229351",
                endtime="2150-01-03 08:00:00",
            )
        ]
        output = [
            Event("outputevents", datetime(2150, 1, 6), hadm_id="1", itemid="226567"),
            Event("outputevents", datetime(2150, 1, 5), hadm_id="1", itemid="226559"),
        ]
        days, sources = _collect_catheter_days(
            self.ADMIT, self._d(9), procedure_events=proc, output_events=output
        )
        self.assertEqual(days, {self._d(0), self._d(1), self._d(2), self._d(4)})
        self.assertEqual(sources, {"icu_proc", "icu_output"})

    def _urine_events(
        self, specimen_id, orgs, comments="", quantity=None, test_itemid="90039"
    ):
        return [
            Event(
                "microbiologyevents",
                datetime(2150, 1, 4),
                micro_specimen_id=specimen_id,
                spec_type_desc="URINE",
                test_itemid=test_itemid,
                org_name=org,
                comments=comments,
                quantity=quantity,
                charttime="2150-01-04 09:00:00",
            )
            for org in orgs
        ]

    def test_urine_culture_criteria(self):
        events = (
            self._urine_events("a", ["ESCHERICHIA COLI"])
            + self._urine_events("b", ["YEAST"])
            + self._urine_events("c", ["E COLI", "ENTEROCOCCUS", "PROTEUS"])
            + self._urine_events("d", ["E COLI"], comments="< 10,000 CFU/mL.")
            + self._urine_events("e", [""], comments="NO GROWTH")
            + self._urine_events("f", ["KLEBSIELLA PNEUMONIAE"], test_itemid="90235")
        )
        reasons = {
            s.specimen_id: s.reason for s in _summarize_urine_specimens(events)
        }
        self.assertEqual(reasons["a"], "qualifies")
        self.assertEqual(reasons["b"], "excluded_organism_only")
        self.assertEqual(reasons["c"], "gt_2_organisms")
        self.assertEqual(reasons["d"], "below_cfu_threshold")
        self.assertEqual(reasons["e"], "no_growth")
        self.assertEqual(reasons["f"], "qualifies")  # reflex urine culture

    def test_non_culture_urine_tests_ignored(self):
        # Chlamydia NAAT (90116) and Legionella antigen (90128) on a urine
        # specimen name an organism but are not urine cultures.
        events = self._urine_events(
            "g", ["CHLAMYDIA TRACHOMATIS"], test_itemid="90116"
        ) + self._urine_events(
            "h", ["LEGIONELLA PNEUMOPHILA SEROGROUP 1"], test_itemid="90128"
        )
        self.assertEqual(_summarize_urine_specimens(events), [])
        # A specimen with both a NAAT row and a culture row keeps only the culture.
        mixed = self._urine_events(
            "i", ["CHLAMYDIA TRACHOMATIS"], test_itemid="90116"
        ) + self._urine_events("i", ["ESCHERICHIA COLI"])
        (spec,) = _summarize_urine_specimens(mixed)
        self.assertEqual(spec.organisms, ("ESCHERICHIA COLI",))

    def test_colony_count_parsing(self):
        self.assertEqual(_parse_colony_count(">100,000 CFU/mL"), 100001)
        self.assertEqual(_parse_colony_count("<10,000 organisms/ml."), 9999)
        self.assertEqual(_parse_colony_count("10,000-100,000 CFU/mL"), 99999)
        self.assertIsNone(_parse_colony_count(None))


class TestCatheterTemporalTask(unittest.TestCase):
    """End-to-end temporal task tests on in-memory synthetic patients."""

    def _admission(self, hadm, admit, disch):
        return ("admissions", admit, {"hadm_id": hadm, "dischtime": disch})

    def _diag(self, hadm, code, disch):
        return (
            "diagnoses_icd",
            disch,
            {"hadm_id": hadm, "icd_code": code, "icd_version": "10"},
        )

    def _urine(self, hadm, specimen, ts, org="ESCHERICHIA COLI", test_itemid="90039"):
        return (
            "microbiologyevents",
            datetime(ts.year, ts.month, ts.day),
            {
                "hadm_id": hadm,
                "micro_specimen_id": specimen,
                "charttime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "spec_type_desc": "URINE",
                "test_itemid": test_itemid,
                "org_name": org,
            },
        )

    def _foley(self, hadm, start, end):
        return (
            "procedureevents",
            start,
            {
                "hadm_id": hadm,
                "itemid": "229351",
                "endtime": end.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    def _run(self, patient, **kwargs):
        task = CatheterAssociatedInfectionPredictionMIMIC4Temporal(
            map_ccscm=False, **kwargs
        )
        return {s["record_id"]: s for s in task(patient)}

    def test_icd_only_catheter_excluded(self):
        # Hard requirement: ICD codes alone cannot establish > 2 catheter days.
        admit, disch = datetime(2150, 1, 1, 8), datetime(2150, 1, 8, 12)
        patient = _make_patient(
            "p1",
            [
                self._admission("10", admit, "2150-01-08 12:00:00"),
                self._diag("10", "T83511A", disch),
                self._urine("10", "s1", datetime(2150, 1, 5, 9)),
            ],
        )
        self.assertEqual(self._run(patient), {})

    def test_short_catheter_excluded(self):
        patient = _make_patient(
            "p1b",
            [
                self._admission("11", datetime(2150, 1, 1, 8), "2150-01-08 12:00:00"),
                self._foley("11", datetime(2150, 1, 2, 10), datetime(2150, 1, 3, 9)),
                self._diag("11", "T83511A", datetime(2150, 1, 8, 12)),
            ],
        )
        self.assertEqual(self._run(patient), {})

    def test_icd_positive_via_m1_with_timed_catheter(self):
        patient = _make_patient(
            "p1c",
            [
                self._admission("12", datetime(2150, 1, 1, 8), "2150-01-08 12:00:00"),
                self._foley("12", datetime(2150, 1, 1, 10), datetime(2150, 1, 5, 10)),
                self._diag("12", "T83511A", datetime(2150, 1, 8, 12)),
            ],
        )
        sample = self._run(patient)["p1c_12"]
        self.assertEqual(sample["label"], 1)
        self.assertEqual(sample["positive_markers"], "M1")
        self.assertEqual(sample["nhsn_strict"], 0)

    def test_culture_only_positive_via_m3_timed(self):
        admit = datetime(2150, 1, 1, 8)
        patient = _make_patient(
            "p2",
            [
                self._admission("20", admit, "2150-01-10 12:00:00"),
                self._foley("20", datetime(2150, 1, 1, 10), datetime(2150, 1, 6, 10)),
                self._urine("20", "s1", datetime(2150, 1, 5, 9)),
            ],
        )
        sample = self._run(patient)["p2_20"]
        self.assertEqual(sample["label"], 1)
        self.assertEqual(sample["positive_markers"], "M3")
        self.assertEqual(sample["nhsn_strict"], 1)
        self.assertEqual(sample["index_time"], "2150-01-03T00:00:00")
        self.assertGreaterEqual(sample["onset_time"], sample["index_time"])
        # Current-admission partial visit is appended (no ICD codes).
        self.assertEqual(sample["conditions"][-1], ["<missing>"])

    def test_culture_before_catheter_day3_is_negative(self):
        admit = datetime(2150, 1, 1, 8)
        patient = _make_patient(
            "p3",
            [
                self._admission("30", admit, "2150-01-10 12:00:00"),
                self._foley("30", datetime(2150, 1, 3, 10), datetime(2150, 1, 8, 10)),
                self._urine("30", "s1", datetime(2150, 1, 4, 9)),
            ],
        )
        sample = self._run(patient)["p3_30"]
        self.assertEqual(sample["label"], 0)
        self.assertEqual(sample["positive_markers"], "")

    def test_rit_suppresses_repeat_and_marker_subset(self):
        patient = _make_patient(
            "p4",
            [
                self._admission("40", datetime(2150, 1, 1, 8), "2150-01-20 12:00:00"),
                self._foley("40", datetime(2150, 1, 1, 10), datetime(2150, 1, 19, 10)),
                self._urine("40", "s1", datetime(2150, 1, 4, 9)),
                self._urine("40", "s2", datetime(2150, 1, 10, 9)),
                self._diag("40", "N390", datetime(2150, 1, 20, 12)),
            ],
        )
        sample = self._run(patient)["p4_40"]
        self.assertEqual(sample["positive_markers"], "M2|M3")
        self.assertTrue(sample["onset_time"].startswith("2150-01-04"))
        # Restricting the label to M1 only turns this admission negative.
        sample_m1 = self._run(patient, positive_markers=("M1",))["p4_40"]
        self.assertEqual(sample_m1["label"], 0)

    def test_urine_naat_does_not_fire_m3(self):
        patient = _make_patient(
            "p7",
            [
                self._admission("70", datetime(2150, 1, 1, 8), "2150-01-10 12:00:00"),
                self._foley("70", datetime(2150, 1, 1, 10), datetime(2150, 1, 6, 10)),
                self._urine(
                    "70",
                    "s1",
                    datetime(2150, 1, 5, 9),
                    org="CHLAMYDIA TRACHOMATIS",
                    test_itemid="90116",
                ),
            ],
        )
        sample = self._run(patient)["p7_70"]
        self.assertEqual(sample["label"], 0)
        self.assertEqual(sample["positive_markers"], "")

    def test_no_catheter_evidence_excluded(self):
        patient = _make_patient(
            "p5",
            [
                self._admission("50", datetime(2150, 1, 1, 8), "2150-01-08 12:00:00"),
                self._urine("50", "s1", datetime(2150, 1, 5, 9)),
                self._diag("50", "N390", datetime(2150, 1, 8, 12)),
            ],
        )
        self.assertEqual(self._run(patient), {})

    def test_stagenet_variant_shapes(self):
        patient = _make_patient(
            "p6",
            [
                self._admission("60", datetime(2150, 1, 1, 8), "2150-01-10 12:00:00"),
                self._foley("60", datetime(2150, 1, 1, 10), datetime(2150, 1, 6, 10)),
                self._urine("60", "s1", datetime(2150, 1, 5, 9)),
            ],
        )
        task = CatheterAssociatedInfectionPredictionStageNetMIMIC4Temporal(
            map_ccscm=False
        )
        (sample,) = task(patient)
        times, codes = sample["icd_codes"]
        self.assertEqual(len(times), len(codes))
        self.assertEqual(len(sample["labs"][1]), len(codes))
        self.assertEqual(sample["label"], 1)


if __name__ == "__main__":
    unittest.main()

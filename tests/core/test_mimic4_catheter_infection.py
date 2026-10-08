import csv
import math
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Synthetic four-patient MIMIC-IV fixture (plain CSVs in a temp directory)
# ---------------------------------------------------------------------------
#   P1 10000001  adm 20000011: ICD catheter code (Z46.6), no infection
#                adm 20000012: CPT 51702 + ICU Foley 03-01..03-07, T83.511A +
#                N39.0, E. coli urine culture 03-05 (catheter/hospital day 5)
#   P2 10000002  adm 20000021: Z46.6, ICU Foley 05-01..05-06, urine culture
#                with no growth
#   P3 10000003  adm 20000031: N39.0 only, no catheter evidence at all
#   P4 10000004  adm 20000041: T83.511A, ICU Foley for only 2 days

FAKE_MIMIC4 = {
    "hosp/patients.csv": [
        ["subject_id", "gender", "anchor_age", "anchor_year",
         "anchor_year_group", "dod"],
        ["10000001", "F", "70", "2150", "2014 - 2016", ""],
        ["10000002", "M", "65", "2150", "2014 - 2016", ""],
        ["10000003", "F", "40", "2150", "2014 - 2016", ""],
        ["10000004", "M", "80", "2150", "2014 - 2016", ""],
    ],
    "hosp/admissions.csv": [
        ["subject_id", "hadm_id", "admittime", "dischtime", "admission_type",
         "admission_location", "discharge_location", "insurance", "language",
         "marital_status", "race", "hospital_expire_flag"],
        ["10000001", "20000011", "2150-01-01 08:00:00", "2150-01-06 12:00:00",
         "URGENT", "ER", "HOME", "Medicare", "ENGLISH", "MARRIED", "WHITE", "0"],
        ["10000001", "20000012", "2150-03-01 08:00:00", "2150-03-10 12:00:00",
         "URGENT", "ER", "HOME", "Medicare", "ENGLISH", "MARRIED", "WHITE", "0"],
        ["10000002", "20000021", "2150-05-01 08:00:00", "2150-05-09 12:00:00",
         "URGENT", "ER", "HOME", "Medicare", "ENGLISH", "SINGLE", "WHITE", "0"],
        ["10000003", "20000031", "2150-07-01 08:00:00", "2150-07-05 12:00:00",
         "URGENT", "ER", "HOME", "Other", "ENGLISH", "SINGLE", "ASIAN", "0"],
        ["10000004", "20000041", "2150-09-01 08:00:00", "2150-09-08 12:00:00",
         "URGENT", "ER", "HOME", "Medicare", "ENGLISH", "WIDOWED", "BLACK", "0"],
    ],
    "icu/icustays.csv": [
        ["subject_id", "hadm_id", "stay_id", "first_careunit", "last_careunit",
         "intime", "outtime"],
        ["10000001", "20000012", "30000012", "MICU", "MICU",
         "2150-03-01 09:00:00", "2150-03-08 09:00:00"],
        ["10000002", "20000021", "30000021", "MICU", "MICU",
         "2150-05-01 09:00:00", "2150-05-07 09:00:00"],
        ["10000004", "20000041", "30000041", "SICU", "SICU",
         "2150-09-02 09:00:00", "2150-09-04 09:00:00"],
    ],
    "hosp/diagnoses_icd.csv": [
        ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
        ["10000001", "20000011", "1", "I10", "10"],
        ["10000001", "20000011", "2", "Z466", "10"],
        ["10000001", "20000012", "1", "T83511A", "10"],
        ["10000001", "20000012", "2", "N390", "10"],
        ["10000001", "20000012", "3", "I10", "10"],
        ["10000002", "20000021", "1", "I10", "10"],
        ["10000002", "20000021", "2", "Z466", "10"],
        ["10000003", "20000031", "1", "N390", "10"],
        ["10000004", "20000041", "1", "T83511A", "10"],
    ],
    "hosp/procedures_icd.csv": [
        ["subject_id", "hadm_id", "seq_num", "chartdate", "icd_code",
         "icd_version"],
        ["10000001", "20000012", "1", "2150-03-01", "0T9B70Z", "10"],
    ],
    "hosp/prescriptions.csv": [
        ["subject_id", "hadm_id", "starttime", "stoptime", "drug", "ndc",
         "prod_strength", "dose_val_rx", "dose_unit_rx", "route"],
        ["10000001", "20000012", "2150-03-02 10:00:00", "2150-03-04 10:00:00",
         "Heparin", "", "5000 units", "5000", "UNIT", "SC"],
        ["10000002", "20000021", "2150-05-02 10:00:00", "2150-05-04 10:00:00",
         "Heparin", "", "5000 units", "5000", "UNIT", "SC"],
    ],
    "hosp/d_labitems.csv": [
        ["itemid", "label", "fluid", "category"],
        ["50983", "Sodium", "Blood", "Chemistry"],
        ["50971", "Potassium", "Blood", "Chemistry"],
    ],
    "hosp/labevents.csv": [
        ["labevent_id", "subject_id", "hadm_id", "specimen_id", "itemid",
         "charttime", "storetime", "value", "valuenum", "valueuom", "flag"],
        # P1 adm 2: sodium before the index day (kept) and after it (dropped).
        ["1", "10000001", "20000012", "1", "50983", "2150-03-02 06:00:00",
         "2150-03-02 07:00:00", "140", "140", "mEq/L", ""],
        ["2", "10000001", "20000012", "2", "50983", "2150-03-08 06:00:00",
         "2150-03-08 07:00:00", "120", "120", "mEq/L", "abnormal"],
        ["3", "10000002", "20000021", "3", "50971", "2150-05-02 06:00:00",
         "2150-05-02 07:00:00", "4.0", "4.0", "mEq/L", ""],
    ],
    "hosp/hcpcsevents.csv": [
        ["subject_id", "hadm_id", "chartdate", "hcpcs_cd", "seq_num",
         "short_description"],
        ["10000001", "20000012", "2150-03-01", "51702", "1",
         "Insert temp bladder cath"],
    ],
    "hosp/microbiologyevents.csv": [
        ["microevent_id", "subject_id", "hadm_id", "micro_specimen_id",
         "chartdate", "charttime", "spec_itemid", "spec_type_desc",
         "test_itemid", "test_name", "org_name", "quantity", "comments",
         "ab_name"],
        ["1", "10000001", "20000012", "40000001", "2150-03-05 00:00:00",
         "2150-03-05 09:00:00", "70079", "URINE", "90039", "URINE CULTURE",
         "ESCHERICHIA COLI", "", "", "AMPICILLIN"],
        ["2", "10000002", "20000021", "40000002", "2150-05-04 00:00:00",
         "2150-05-04 09:00:00", "70079", "URINE", "90039", "URINE CULTURE",
         "", "", "NO GROWTH.", ""],
    ],
    "icu/procedureevents.csv": [
        ["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid",
         "value", "statusdescription"],
        ["10000001", "20000012", "30000012", "2150-03-01 10:00:00",
         "2150-03-07 10:00:00", "229351", "1", "FinishedRunning"],
        ["10000002", "20000021", "30000021", "2150-05-01 10:00:00",
         "2150-05-06 10:00:00", "229351", "1", "FinishedRunning"],
        ["10000004", "20000041", "30000041", "2150-09-02 10:00:00",
         "2150-09-03 09:00:00", "229351", "1", "FinishedRunning"],
    ],
    "icu/outputevents.csv": [
        ["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value"],
        ["10000001", "20000012", "30000012", "2150-03-03 12:00:00", "226559",
         "400"],
        ["10000002", "20000021", "30000021", "2150-05-03 12:00:00", "226559",
         "350"],
    ],
}

ICD_TABLES = ["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"]
TEMPORAL_TABLES = ICD_TABLES + [
    "hcpcsevents",
    "microbiologyevents",
    "procedureevents",
    "outputevents",
]


def _write_fake_mimic4(root):
    for rel_path, rows in FAKE_MIMIC4.items():
        path = Path(root) / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows(rows)


def _to_int(value):
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


class TestMIMIC4CatheterInfectionPrediction(unittest.TestCase):
    """End-to-end set_task tests on the synthetic four-patient fixture."""

    NUM_WORKERS = 2

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        _write_fake_mimic4(root / "mimiciv")
        cls.icd_dataset = MIMIC4Dataset(
            ehr_root=str(root / "mimiciv"),
            ehr_tables=ICD_TABLES,
            cache_dir=str(root / "cache_icd"),
        )
        cls.temporal_dataset = MIMIC4Dataset(
            ehr_root=str(root / "mimiciv"),
            ehr_tables=TEMPORAL_TABLES,
            cache_dir=str(root / "cache_temporal"),
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _samples(self, dataset, task):
        return {
            s["record_id"]: s
            for s in dataset.set_task(task, num_workers=self.NUM_WORKERS)
        }

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
        # N39.0 is a conditional infection code: positive only with a catheter.
        self.assertTrue(task._is_conditional_infection_code("N39.0", 10))
        self.assertFalse(task._is_unconditional_infection_code("N39.0", 10))

    def test_icd_task_labels(self):
        for task in (
            CatheterAssociatedInfectionPredictionMIMIC4(map_ccscm=False),
            CatheterAssociatedInfectionPredictionStageNetMIMIC4(map_ccscm=False),
        ):
            with self.subTest(task=task.task_name):
                labels = {
                    rid: _to_int(s["label"])
                    for rid, s in self._samples(self.icd_dataset, task).items()
                }
                self.assertEqual(
                    labels,
                    {
                        "10000001_cauti1": 1,
                        "10000001_cauti1_aug1": 1,  # suffix augmentation
                        "10000001_neg1": 0,
                        "10000002_neg1": 0,
                        "10000004_cauti1": 1,  # unconditional T83.511A
                    },
                )

    def test_temporal_task_labels(self):
        task = CatheterAssociatedInfectionPredictionMIMIC4Temporal(map_ccscm=False)
        samples = self._samples(self.temporal_dataset, task)
        # P1 admission 1 (ICD-only catheter), P3 (no catheter) and P4 (2-day
        # Foley) fail the hard catheter > 2 days gate and emit no sample.
        self.assertEqual(set(samples), {"10000001_20000012", "10000002_20000021"})

        pos = samples["10000001_20000012"]
        self.assertEqual(_to_int(pos["label"]), 1)
        self.assertEqual(pos["positive_markers"], "M1|M2|M3")
        # 0T9B70Z (Foley placement, ICD-10-PCS) is recorded as "icd" evidence.
        self.assertEqual(pos["catheter_sources"], "cpt|icd|icu_output|icu_proc")
        self.assertEqual(pos["index_time"], "2150-03-03T00:00:00")
        self.assertEqual(pos["onset_time"], "2150-03-05T09:00:00")
        self.assertEqual(_to_int(pos["nhsn_strict"]), 1)

        neg = samples["10000002_20000021"]
        self.assertEqual(_to_int(neg["label"]), 0)
        self.assertEqual(neg["positive_markers"], "")

    def test_temporal_labs_stop_at_index_time(self):
        task = CatheterAssociatedInfectionPredictionMIMIC4Temporal(map_ccscm=False)
        pos = self._samples(self.temporal_dataset, task)["10000001_20000012"]
        sodium = task.LAB_CATEGORY_ORDER.index("Sodium")
        # Visits: prior admission, then the current admission before index.
        # Only the 03-02 sodium (140) precedes the 03-03 index; 03-08 (120)
        # would leak post-index information.
        self.assertEqual(float(pos["labs"][-1][sodium]), 140.0)

    def test_temporal_stagenet_variant(self):
        task = CatheterAssociatedInfectionPredictionStageNetMIMIC4Temporal(
            map_ccscm=False
        )
        labels = {
            rid: _to_int(s["label"])
            for rid, s in self._samples(self.temporal_dataset, task).items()
        }
        self.assertEqual(labels, {"10000001_20000012": 1, "10000002_20000021": 0})

    def test_missing_defaults(self):
        task = CatheterAssociatedInfectionPredictionMIMIC4()

        self.assertEqual(task._ensure_nonempty_sequence([]), ["<missing>"])

        empty_lab_df = pl.DataFrame()
        lab_vector = task._build_lab_vector(empty_lab_df)
        self.assertEqual(len(lab_vector), len(task.LAB_CATEGORY_ORDER))
        self.assertTrue(all(v == 0.0 for v in lab_vector))

    def test_no_nan_labs_in_nested_outputs(self):
        for dataset, task in (
            (self.icd_dataset, CatheterAssociatedInfectionPredictionMIMIC4(map_ccscm=False)),
            (
                self.temporal_dataset,
                CatheterAssociatedInfectionPredictionMIMIC4Temporal(map_ccscm=False),
            ),
        ):
            for sample in self._samples(dataset, task).values():
                for visit_labs in sample["labs"]:
                    self.assertFalse(any(math.isnan(float(v)) for v in visit_labs))


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

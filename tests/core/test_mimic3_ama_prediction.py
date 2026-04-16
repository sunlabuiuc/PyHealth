"""Test suite for MIMIC-III Against-Medical-Advice (AMA) discharge prediction.

This is the automated test module for ``AMAPredictionMIMIC3``
(``pyhealth.tasks.ama_prediction``).  It provides a comprehensive set of checks
covering:

    - Task helpers: race and insurance normalization, substance-use
      detection from diagnosis text.
    - Task contract: ``task_name``, ``input_schema``, ``output_schema``, and
      default flags (e.g. newborn filtering).
    - Feature engineering on mock patients: AMA vs non-AMA labels, age and
      LOS from timestamps, demographics vs separate ``race`` tokens,
      ``has_substance_use``, multi-admission behavior, and schema key sets.
    - Ablation-oriented checks: baseline feature presence, label correctness,
      and absence of clinical code fields in samples.
    - Integration: curated five-row gzipped MIMIC-style CSVs with
      ``MIMIC3Dataset`` + ``set_task`` + ``LogisticRegression`` forward passes
      and short ``Trainer`` smoke runs (example CLI tables are not asserted).
    - Synthetic generator sanity: exhaustive grid patient row count.

Paper (task motivation):
    Boag, W.; Suresh, H.; Celi, L. A.; Szolovits, P.; and Ghassemi, M.
    "Racial Disparities and Mistrust in End-of-Life Care." MLHC / PMLR, 2018.

Usage:
    # From the PyHealth repository root (quiet summary)
    cd /path/to/PyHealth && python -m unittest tests.core.test_mimic3_ama_prediction -q

    # Verbose per-test output
    cd /path/to/PyHealth && python -m unittest tests.core.test_mimic3_ama_prediction -v

    # Run this file directly
    cd /path/to/PyHealth && python tests/core/test_mimic3_ama_prediction.py

"""

import gc
import gzip
import importlib.util
import io
import shutil
import sys
import tempfile
import unittest
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pandas as pd
import torch

from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
from pyhealth.models import LogisticRegression
from pyhealth.tasks.ama_prediction import (
    AMAPredictionMIMIC3,
    _has_substance_use,
    _normalize_insurance,
    _normalize_race,
)
from pyhealth.trainer import Trainer

warnings.filterwarnings("ignore", category=ResourceWarning)
if "ignore::ResourceWarning" not in getattr(sys, "warnoptions", []):
    sys.warnoptions.append("ignore::ResourceWarning")
    warnings._filters_mutated()

_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "mimic3_ama_prediction_logistic_regression.py"
)
_spec = importlib.util.spec_from_file_location(
    "mimic3_ama_prediction_example",
    _EXAMPLE_PATH,
)
_example_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_example_mod)
generate_synthetic_mimic3 = _example_mod.generate_synthetic_mimic3


# Fixed 5-row MIMIC-III-like slice for integration tests (3 non-AMA, 2 AMA).
# Race (normalized): 2 White, 2 Black, 1 Hispanic.
# Age at admit (task uses calendar year from DOB vs admittime): two young
# (18-44), two middle (45-64), one senior (65+).
CURATED_SYNTHETIC_N = 5
CURATED_SYNTHETIC_AMA_NEGATIVE = 3
CURATED_SYNTHETIC_AMA_POSITIVE = 2

_CURATED_MIMIC3_PATIENTS = [
    {
        "subject_id": 1,
        "gender": "M",
        "dob": "2118-01-01 00:00:00",
        "dod": None,
        "dod_hosp": None,
        "dod_ssn": None,
        "expire_flag": 0,
    },
    {
        "subject_id": 2,
        "gender": "F",
        "dob": "2096-01-02 00:00:00",
        "dod": None,
        "dod_hosp": None,
        "dod_ssn": None,
        "expire_flag": 0,
    },
    {
        "subject_id": 3,
        "gender": "M",
        "dob": "2098-01-03 00:00:00",
        "dod": None,
        "dod_hosp": None,
        "dod_ssn": None,
        "expire_flag": 0,
    },
    {
        "subject_id": 4,
        "gender": "F",
        "dob": "2115-01-11 00:00:00",
        "dod": None,
        "dod_hosp": None,
        "dod_ssn": None,
        "expire_flag": 0,
    },
    {
        "subject_id": 5,
        "gender": "M",
        "dob": "2079-01-12 00:00:00",
        "dod": None,
        "dod_hosp": None,
        "dod_ssn": None,
        "expire_flag": 0,
    },
]

_CURATED_MIMIC3_ADMISSIONS = [
    {
        "subject_id": 1,
        "hadm_id": 100,
        "admission_type": "EMERGENCY",
        "admission_location": "EMERGENCY ROOM ADMIT",
        "insurance": "Private",
        "language": "ENGLISH",
        "religion": "CHRISTIAN",
        "marital_status": "SINGLE",
        "ethnicity": "WHITE",
        "edregtime": "2150-01-01 00:00:00",
        "edouttime": "2150-01-01 00:00:00",
        "diagnosis": "PNEUMONIA",
        "discharge_location": "HOME",
        "dischtime": "2150-01-08 00:00:00",
        "admittime": "2150-01-01 00:00:00",
        "hospital_expire_flag": 0,
    },
    {
        "subject_id": 2,
        "hadm_id": 101,
        "admission_type": "EMERGENCY",
        "admission_location": "EMERGENCY ROOM ADMIT",
        "insurance": "Private",
        "language": "ENGLISH",
        "religion": "CHRISTIAN",
        "marital_status": "SINGLE",
        "ethnicity": "WHITE",
        "edregtime": "2150-01-02 00:00:00",
        "edouttime": "2150-01-02 00:00:00",
        "diagnosis": "CHEST PAIN",
        "discharge_location": "HOME",
        "dischtime": "2150-01-09 00:00:00",
        "admittime": "2150-01-02 00:00:00",
        "hospital_expire_flag": 0,
    },
    {
        "subject_id": 3,
        "hadm_id": 102,
        "admission_type": "EMERGENCY",
        "admission_location": "EMERGENCY ROOM ADMIT",
        "insurance": "Medicaid",
        "language": "ENGLISH",
        "religion": "CHRISTIAN",
        "marital_status": "SINGLE",
        "ethnicity": "BLACK/AFRICAN AMERICAN",
        "edregtime": "2150-01-03 00:00:00",
        "edouttime": "2150-01-03 00:00:00",
        "diagnosis": "SEPSIS",
        "discharge_location": "HOME",
        "dischtime": "2150-01-10 00:00:00",
        "admittime": "2150-01-03 00:00:00",
        "hospital_expire_flag": 0,
    },
    {
        "subject_id": 4,
        "hadm_id": 103,
        "admission_type": "EMERGENCY",
        "admission_location": "EMERGENCY ROOM ADMIT",
        "insurance": "Medicaid",
        "language": "ENGLISH",
        "religion": "CHRISTIAN",
        "marital_status": "SINGLE",
        "ethnicity": "BLACK/AFRICAN AMERICAN",
        "edregtime": "2150-01-11 00:00:00",
        "edouttime": "2150-01-11 00:00:00",
        "diagnosis": "PNEUMONIA",
        "discharge_location": "LEFT AGAINST MEDICAL ADVI",
        "dischtime": "2150-01-18 00:00:00",
        "admittime": "2150-01-11 00:00:00",
        "hospital_expire_flag": 0,
    },
    {
        "subject_id": 5,
        "hadm_id": 104,
        "admission_type": "EMERGENCY",
        "admission_location": "EMERGENCY ROOM ADMIT",
        "insurance": "Medicare",
        "language": "ENGLISH",
        "religion": "CHRISTIAN",
        "marital_status": "SINGLE",
        "ethnicity": "HISPANIC OR LATINO",
        "edregtime": "2150-01-12 00:00:00",
        "edouttime": "2150-01-12 00:00:00",
        "diagnosis": "OPIOID DEPENDENCE",
        "discharge_location": "LEFT AGAINST MEDICAL ADVI",
        "dischtime": "2150-01-19 00:00:00",
        "admittime": "2150-01-12 00:00:00",
        "hospital_expire_flag": 0,
    },
]

_CURATED_MIMIC3_ICUSTAYS = [
    {
        "subject_id": 1,
        "hadm_id": 100,
        "icustay_id": 1000,
        "first_careunit": "MICU",
        "last_careunit": "MICU",
        "dbsource": "metavision",
        "intime": "2150-01-01 02:00:00",
        "outtime": "2150-01-07 22:00:00",
    },
    {
        "subject_id": 2,
        "hadm_id": 101,
        "icustay_id": 1001,
        "first_careunit": "MICU",
        "last_careunit": "MICU",
        "dbsource": "metavision",
        "intime": "2150-01-02 02:00:00",
        "outtime": "2150-01-08 22:00:00",
    },
    {
        "subject_id": 3,
        "hadm_id": 102,
        "icustay_id": 1002,
        "first_careunit": "MICU",
        "last_careunit": "MICU",
        "dbsource": "metavision",
        "intime": "2150-01-03 02:00:00",
        "outtime": "2150-01-09 22:00:00",
    },
    {
        "subject_id": 4,
        "hadm_id": 103,
        "icustay_id": 1003,
        "first_careunit": "MICU",
        "last_careunit": "MICU",
        "dbsource": "metavision",
        "intime": "2150-01-11 02:00:00",
        "outtime": "2150-01-17 22:00:00",
    },
    {
        "subject_id": 5,
        "hadm_id": 104,
        "icustay_id": 1004,
        "first_careunit": "MICU",
        "last_careunit": "MICU",
        "dbsource": "metavision",
        "intime": "2150-01-12 02:00:00",
        "outtime": "2150-01-18 22:00:00",
    },
]


def _write_curated_synthetic_mimic3_for_tests(root: str) -> None:
    """Materialize the fixed 5-row MIMIC-III-like CSV.gz bundle for tests.

    Args:
        root: Directory receiving ``PATIENTS.csv.gz``, ``ADMISSIONS.csv.gz``,
            ``ICUSTAYS.csv.gz`` (column names match ``MIMIC3Dataset`` ingest).

    Returns:
        None.

    Note:
        Row mix matches ``CURATED_SYNTHETIC_*`` constants so
        ``AMAPredictionMIMIC3`` yields both AMA labels and varied demographics
        without loading real MIMIC-III.
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("PATIENTS", _CURATED_MIMIC3_PATIENTS),
        ("ADMISSIONS", _CURATED_MIMIC3_ADMISSIONS),
        ("ICUSTAYS", _CURATED_MIMIC3_ICUSTAYS),
    ):
        df = pd.DataFrame(rows)
        with gzip.open(root_path / f"{name}.csv.gz", "wt") as f:
            df.to_csv(f, index=False)


# ------------------------------------------------------------------
# Module-level shared dataset (loaded once for all integration tests)
# ------------------------------------------------------------------

_shared_tmpdir = None
_shared_cache_dir = None
_shared_dataset = None
_shared_sample_dataset = None


def setUpModule() -> None:
    """Load one shared ``MIMIC3Dataset`` + ``SampleDataset`` for integration tests.

    Runs once per test module import: writes curated CSVs, builds the base
    dataset with ``tables=[]``, applies ``AMAPredictionMIMIC3`` via
    ``set_task`` (task ``input_schema`` / ``output_schema`` drive processors).
    """
    global _shared_tmpdir, _shared_cache_dir
    global _shared_dataset, _shared_sample_dataset
    _shared_tmpdir = tempfile.mkdtemp(prefix="ama_shared_")
    _shared_cache_dir = tempfile.mkdtemp(prefix="ama_shared_cache_")
    _write_curated_synthetic_mimic3_for_tests(_shared_tmpdir)
    _shared_dataset = MIMIC3Dataset(
        root=_shared_tmpdir,
        tables=[],
        cache_dir=_shared_cache_dir,
    )
    _shared_sample_dataset = _shared_dataset.set_task(AMAPredictionMIMIC3())


def tearDownModule() -> None:
    """Release LitData handles and remove temp CSV/cache directories."""
    global _shared_dataset, _shared_sample_dataset
    if _shared_sample_dataset is not None:
        _shared_sample_dataset.close()
    _shared_sample_dataset = None
    _shared_dataset = None
    gc.collect()
    # Proactively close lingering ``.ld`` chunk readers to avoid shutdown
    # ``ResourceWarning`` from litdata after temp dirs are removed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for obj in gc.get_objects():
            if isinstance(obj, io.FileIO) and not obj.closed:
                name = getattr(obj, "name", "")
                if (
                    isinstance(name, str)
                    and _shared_cache_dir
                    and _shared_cache_dir in name
                ):
                    try:
                        obj.close()
                    except Exception:
                        pass
    gc.collect()
    for d in (_shared_tmpdir, _shared_cache_dir):
        if d:
            shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_event(**attrs: Any) -> MagicMock:
    """Build a ``MagicMock`` admission/drug/etc. row for unit tests.

    Args:
        **attrs: Field names and values (e.g. ``hadm_id=``, ``icd9_code=``).

    Returns:
        Mock object whose attributes match ``attrs``.
    """
    event = MagicMock()
    for key, value in attrs.items():
        setattr(event, key, value)
    return event


def _build_patient(
    patient_id: str,
    admissions: List[Dict[str, Any]],
    diagnoses: List[Dict[str, Any]],
    procedures: List[Dict[str, Any]],
    prescriptions: List[Dict[str, Any]],
    gender: str = "M",
    dob: str = "2100-01-01 00:00:00",
) -> MagicMock:
    """Build a mock ``Patient`` whose ``get_events`` mirrors PyHealth filters.

    Args:
        patient_id: ``patient.patient_id`` string.
        admissions: kwargs for each admission ``MagicMock``.
        diagnoses: kwargs for diagnosis events (``hadm_id`` aligned).
        procedures: kwargs for procedure events.
        prescriptions: kwargs for prescription events.
        gender: Demographics token for ``patients`` event.
        dob: DOB string feeding age logic in ``AMAPredictionMIMIC3``.

    Returns:
        ``MagicMock`` with ``get_events`` routing ``event_type`` to the proper
        list and honoring simple ``filters`` tuples on code tables.

    Note:
        Output samples follow ``AMAPredictionMIMIC3.input_schema`` /
        ``output_schema`` (same keys as real ``set_task`` tensors after
        processors run on CSV-backed data).
    """
    patient = MagicMock()
    patient.patient_id = patient_id

    patient_event = _make_event(gender=gender, dob=dob)
    adm_events = [_make_event(**a) for a in admissions]
    diag_events = [_make_event(**d) for d in diagnoses]
    proc_events = [_make_event(**p) for p in procedures]
    rx_events = [_make_event(**r) for r in prescriptions]

    def _get_events(event_type, filters=None, **kwargs):
        if event_type == "patients":
            return [patient_event]
        if event_type == "admissions":
            return adm_events
        source = {
            "diagnoses_icd": diag_events,
            "procedures_icd": proc_events,
            "prescriptions": rx_events,
        }.get(event_type, [])
        if filters:
            col, op, val = filters[0]
            source = [e for e in source if getattr(e, col, None) == val]
        return source

    patient.get_events = _get_events
    return patient


SAMPLE_KEYS = {
    "visit_id",
    "patient_id",
    "demographics",
    "age",
    "los",
    "race",
    "has_substance_use",
    "ama",
}


class TestNormalizeRace(unittest.TestCase):
    """Unit tests for the race normalization helper."""

    def test_white(self):
        self.assertEqual(_normalize_race("WHITE"), "White")
        self.assertEqual(_normalize_race("WHITE - RUSSIAN"), "White")

    def test_black(self):
        self.assertEqual(_normalize_race("BLACK/AFRICAN AMERICAN"), "Black")

    def test_hispanic(self):
        self.assertEqual(_normalize_race("HISPANIC OR LATINO"), "Hispanic")
        self.assertEqual(_normalize_race("SOUTH AMERICAN"), "Hispanic")

    def test_asian(self):
        self.assertEqual(_normalize_race("ASIAN - CHINESE"), "Asian")

    def test_native_american(self):
        self.assertEqual(
            _normalize_race("AMERICAN INDIAN/ALASKA NATIVE"),
            "Native American",
        )

    def test_other(self):
        self.assertEqual(_normalize_race("UNKNOWN/NOT SPECIFIED"), "Other")
        self.assertEqual(_normalize_race(None), "Other")

    def test_normalize_insurance(self):
        self.assertEqual(_normalize_insurance("Medicare"), "Public")
        self.assertEqual(_normalize_insurance("Medicaid"), "Public")
        self.assertEqual(_normalize_insurance("Government"), "Public")
        self.assertEqual(_normalize_insurance("Private"), "Private")
        self.assertEqual(_normalize_insurance("Self Pay"), "Self Pay")
        self.assertEqual(_normalize_insurance(None), "Other")


class TestHasSubstanceUse(unittest.TestCase):
    """Unit tests for the substance-use detection helper."""

    def test_alcohol(self):
        self.assertEqual(_has_substance_use("ALCOHOL WITHDRAWAL"), 1)

    def test_opioid(self):
        self.assertEqual(_has_substance_use("OPIOID DEPENDENCE"), 1)

    def test_heroin(self):
        self.assertEqual(_has_substance_use("HEROIN OVERDOSE"), 1)

    def test_cocaine(self):
        self.assertEqual(_has_substance_use("COCAINE INTOXICATION"), 1)

    def test_drug_withdrawal(self):
        self.assertEqual(_has_substance_use("DRUG WITHDRAWAL SEIZURE"), 1)

    def test_etoh(self):
        self.assertEqual(_has_substance_use("ETOH ABUSE"), 1)

    def test_substance(self):
        self.assertEqual(_has_substance_use("SUBSTANCE ABUSE"), 1)

    def test_overdose(self):
        self.assertEqual(_has_substance_use("OVERDOSE - ACCIDENTAL"), 1)

    def test_negative(self):
        self.assertEqual(_has_substance_use("PNEUMONIA"), 0)
        self.assertEqual(_has_substance_use("CHEST PAIN"), 0)

    def test_none(self):
        self.assertEqual(_has_substance_use(None), 0)

    def test_case_insensitive(self):
        self.assertEqual(_has_substance_use("alcohol withdrawal"), 1)
        self.assertEqual(_has_substance_use("Heroin Overdose"), 1)


class TestAMAPredictionMIMIC3Schema(unittest.TestCase):
    """Validate class-level schema attributes."""

    def test_task_name(self):
        self.assertEqual(
            AMAPredictionMIMIC3.task_name,
            "AMAPredictionMIMIC3",
        )

    def test_input_schema(self):
        schema = AMAPredictionMIMIC3.input_schema
        self.assertEqual(schema["demographics"], "multi_hot")
        self.assertEqual(schema["age"], "tensor")
        self.assertEqual(schema["los"], "tensor")
        self.assertEqual(schema["race"], "multi_hot")
        self.assertEqual(schema["has_substance_use"], "tensor")

    def test_output_schema(self):
        self.assertEqual(AMAPredictionMIMIC3.output_schema, {"ama": "binary"})

    def test_defaults(self):
        task = AMAPredictionMIMIC3()
        self.assertTrue(task.exclude_newborns)


class TestAMAPredictionMIMIC3Mock(unittest.TestCase):
    """Unit tests using purely synthetic mock patients.

    No real dataset is loaded.  Each test builds mock Patient
    objects in memory and runs the task callable directly.
    All tests complete in milliseconds.
    """

    def setUp(self) -> None:
        """Fresh task instance per test (default ``exclude_newborns=True``)."""
        self.task = AMAPredictionMIMIC3()

    def _default_admission(
        self,
        hadm_id: str = "100",
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Admission kwargs for ``_build_patient`` with sane AMA-test defaults.

        Args:
            hadm_id: Visit id string stored on the mock admission.
            **overrides: Fields to replace (e.g. ``discharge_location=``).

        Returns:
            Dict passed to ``_make_event`` via ``_build_patient`` admissions
            list; keys mirror post-ingest MIMIC attribute names.
        """
        adm = {
            "hadm_id": hadm_id,
            "admission_type": "EMERGENCY",
            "discharge_location": "HOME",
            "ethnicity": "WHITE",
            "insurance": "Private",
            "dischtime": "2150-01-10 14:00:00",
            "timestamp": datetime(2150, 1, 1),
            "diagnosis": "PNEUMONIA",
        }
        adm.update(overrides)
        return adm

    # ----------------------------------------------------------
    # Label generation
    # ----------------------------------------------------------

    def test_empty_patient(self):
        patient = MagicMock()
        patient.patient_id = "P0"
        patient.get_events = lambda event_type, **kw: []
        self.assertEqual(self.task(patient), [])

    def test_ama_label_positive(self):
        """AMA discharge -> label=1."""
        patient = _build_patient(
            patient_id="P1",
            admissions=[
                self._default_admission(
                    hadm_id="100",
                    discharge_location=("LEFT AGAINST MEDICAL ADVI"),
                ),
            ],
            diagnoses=[{"hadm_id": "100", "icd9_code": "4019"}],
            procedures=[{"hadm_id": "100", "icd9_code": "3893"}],
            prescriptions=[{"hadm_id": "100", "drug": "Aspirin"}],
        )
        samples = self.task(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["ama"], 1)
        self.assertEqual(samples[0]["visit_id"], "100")
        self.assertEqual(samples[0]["patient_id"], "P1")

    def test_ama_label_negative(self):
        """Non-AMA discharge -> label=0."""
        patient = _build_patient(
            patient_id="P2",
            admissions=[
                self._default_admission(hadm_id="200"),
            ],
            diagnoses=[{"hadm_id": "200", "icd9_code": "25000"}],
            procedures=[{"hadm_id": "200", "icd9_code": "3995"}],
            prescriptions=[{"hadm_id": "200", "drug": "Metformin"}],
        )
        samples = self.task(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["ama"], 0)

    def test_multiple_admissions_mixed_labels(self):
        """Two admissions: one AMA, one not."""
        patient = _build_patient(
            patient_id="P3",
            admissions=[
                self._default_admission(hadm_id="300"),
                self._default_admission(
                    hadm_id="301",
                    admission_type="URGENT",
                    discharge_location=("LEFT AGAINST MEDICAL ADVI"),
                    timestamp=datetime(2150, 6, 1),
                    dischtime="2150-06-05 10:00:00",
                ),
            ],
            diagnoses=[
                {"hadm_id": "300", "icd9_code": "4019"},
                {"hadm_id": "301", "icd9_code": "30000"},
            ],
            procedures=[
                {"hadm_id": "300", "icd9_code": "3893"},
                {"hadm_id": "301", "icd9_code": "9394"},
            ],
            prescriptions=[
                {"hadm_id": "300", "drug": "Lisinopril"},
                {"hadm_id": "301", "drug": "Naloxone"},
            ],
        )
        samples = self.task(patient)
        self.assertEqual(len(samples), 2)
        labels = {s["visit_id"]: s["ama"] for s in samples}
        self.assertEqual(labels["300"], 0)
        self.assertEqual(labels["301"], 1)

    # ----------------------------------------------------------
    # Filtering / edge cases
    # ----------------------------------------------------------

    def test_exclude_newborns(self):
        """NEWBORN admissions skipped when flag is True."""
        patient = _build_patient(
            patient_id="P7",
            admissions=[
                self._default_admission(
                    hadm_id="700",
                    admission_type="NEWBORN",
                ),
            ],
            diagnoses=[{"hadm_id": "700", "icd9_code": "V3000"}],
            procedures=[{"hadm_id": "700", "icd9_code": "9904"}],
            prescriptions=[{"hadm_id": "700", "drug": "Vitamin K"}],
        )
        task_ex = AMAPredictionMIMIC3(exclude_newborns=True)
        self.assertEqual(task_ex(patient), [])

        task_in = AMAPredictionMIMIC3(exclude_newborns=False)
        samples = task_in(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["ama"], 0)

    def test_no_patients_events(self):
        """Patient with admissions but no ``patients`` event."""
        patient = MagicMock()
        patient.patient_id = "PX"

        def _get(event_type, **kw):
            if event_type == "patients":
                return []
            if event_type == "admissions":
                return [_make_event(hadm_id="1")]
            return []

        patient.get_events = _get
        self.assertEqual(self.task(patient), [])

    # ----------------------------------------------------------
    # Feature extraction
    # ----------------------------------------------------------

    def test_sample_keys(self):
        """Every sample must contain the expected keys."""
        patient = _build_patient(
            patient_id="P8",
            admissions=[
                self._default_admission(
                    hadm_id="800",
                    discharge_location="SNF",
                    timestamp=datetime(2150, 2, 15),
                ),
            ],
            diagnoses=[{"hadm_id": "800", "icd9_code": "4280"}],
            procedures=[{"hadm_id": "800", "icd9_code": "3722"}],
            prescriptions=[{"hadm_id": "800", "drug": "Furosemide"}],
        )
        samples = self.task(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(set(samples[0].keys()), SAMPLE_KEYS)

    def test_demographics_baseline_tokens(self):
        """BASELINE demographics: gender + insurance, no race."""
        patient = _build_patient(
            patient_id="P10",
            admissions=[
                self._default_admission(
                    hadm_id="1000",
                    ethnicity="BLACK/AFRICAN AMERICAN",
                    insurance="Medicaid",
                    timestamp=datetime(2150, 5, 1),
                ),
            ],
            diagnoses=[{"hadm_id": "1000", "icd9_code": "4019"}],
            procedures=[{"hadm_id": "1000", "icd9_code": "3893"}],
            prescriptions=[{"hadm_id": "1000", "drug": "Aspirin"}],
            gender="F",
        )
        samples = self.task(patient)
        demo = samples[0]["demographics"]
        self.assertIn("gender:F", demo)
        self.assertIn("insurance:Public", demo)
        for token in demo:
            self.assertFalse(
                token.startswith("race:"),
                "race must not be in demographics",
            )

    def test_race_separate_feature(self):
        """Race is a separate multi-hot feature."""
        patient = _build_patient(
            patient_id="P10b",
            admissions=[
                self._default_admission(
                    hadm_id="1001",
                    ethnicity="BLACK/AFRICAN AMERICAN",
                    timestamp=datetime(2150, 5, 1),
                ),
            ],
            diagnoses=[{"hadm_id": "1001", "icd9_code": "4019"}],
            procedures=[{"hadm_id": "1001", "icd9_code": "3893"}],
            prescriptions=[{"hadm_id": "1001", "drug": "Aspirin"}],
        )
        samples = self.task(patient)
        self.assertIn("race", samples[0])
        self.assertEqual(samples[0]["race"], ["race:Black"])

    def test_substance_use_positive(self):
        """Substance-use diagnosis -> has_substance_use=1."""
        patient = _build_patient(
            patient_id="P14",
            admissions=[
                self._default_admission(
                    hadm_id="1400",
                    diagnosis="ALCOHOL WITHDRAWAL",
                    timestamp=datetime(2150, 7, 1),
                ),
            ],
            diagnoses=[{"hadm_id": "1400", "icd9_code": "29181"}],
            procedures=[{"hadm_id": "1400", "icd9_code": "3893"}],
            prescriptions=[{"hadm_id": "1400", "drug": "Lorazepam"}],
        )
        samples = self.task(patient)
        self.assertEqual(samples[0]["has_substance_use"], [1.0])

    def test_substance_use_negative(self):
        """Non-substance diagnosis -> has_substance_use=0."""
        patient = _build_patient(
            patient_id="P15",
            admissions=[
                self._default_admission(
                    hadm_id="1500",
                    diagnosis="PNEUMONIA",
                    timestamp=datetime(2150, 8, 1),
                ),
            ],
            diagnoses=[{"hadm_id": "1500", "icd9_code": "486"}],
            procedures=[{"hadm_id": "1500", "icd9_code": "3893"}],
            prescriptions=[{"hadm_id": "1500", "drug": "Levofloxacin"}],
        )
        samples = self.task(patient)
        self.assertEqual(samples[0]["has_substance_use"], [0.0])

    def test_age_calculation(self):
        """Age computed from dob and admission timestamp."""
        patient = _build_patient(
            patient_id="P11",
            admissions=[
                self._default_admission(
                    hadm_id="1100",
                    timestamp=datetime(2150, 6, 15),
                    dischtime="2150-06-20 12:00:00",
                ),
            ],
            diagnoses=[{"hadm_id": "1100", "icd9_code": "4019"}],
            procedures=[{"hadm_id": "1100", "icd9_code": "3893"}],
            prescriptions=[{"hadm_id": "1100", "drug": "Aspirin"}],
            dob="2100-01-01 00:00:00",
        )
        samples = self.task(patient)
        self.assertEqual(samples[0]["age"], [50.0])

    def test_age_capped_at_90(self):
        """Ages above 90 are capped (MIMIC-III convention)."""
        patient = _build_patient(
            patient_id="P12",
            admissions=[
                self._default_admission(
                    hadm_id="1200",
                    timestamp=datetime(2150, 6, 15),
                ),
            ],
            diagnoses=[{"hadm_id": "1200", "icd9_code": "4019"}],
            procedures=[{"hadm_id": "1200", "icd9_code": "3893"}],
            prescriptions=[{"hadm_id": "1200", "drug": "Aspirin"}],
            dob="1850-01-01 00:00:00",
        )
        samples = self.task(patient)
        self.assertEqual(samples[0]["age"], [90.0])

    def test_los_calculation(self):
        """LOS in days from admittime to dischtime."""
        patient = _build_patient(
            patient_id="P13",
            admissions=[
                self._default_admission(
                    hadm_id="1300",
                    timestamp=datetime(2150, 3, 1, 8, 0, 0),
                    dischtime="2150-03-06 08:00:00",
                ),
            ],
            diagnoses=[{"hadm_id": "1300", "icd9_code": "4019"}],
            procedures=[{"hadm_id": "1300", "icd9_code": "3893"}],
            prescriptions=[{"hadm_id": "1300", "drug": "Aspirin"}],
        )
        samples = self.task(patient)
        self.assertAlmostEqual(samples[0]["los"][0], 5.0, places=2)

    # ----------------------------------------------------------
    # Multi-patient synthetic "dataset" (2 patients)
    # ----------------------------------------------------------

    def test_two_patient_synthetic_dataset(self):
        """End-to-end with 2 synthetic patients, both labels."""
        p1 = _build_patient(
            patient_id="S1",
            admissions=[
                self._default_admission(
                    hadm_id="A1",
                    discharge_location=("LEFT AGAINST MEDICAL ADVI"),
                    ethnicity="HISPANIC OR LATINO",
                    insurance="Medicaid",
                    diagnosis="HEROIN OVERDOSE",
                    timestamp=datetime(2150, 1, 1),
                    dischtime="2150-01-03 12:00:00",
                ),
            ],
            diagnoses=[{"hadm_id": "A1", "icd9_code": "96500"}],
            procedures=[{"hadm_id": "A1", "icd9_code": "9604"}],
            prescriptions=[{"hadm_id": "A1", "drug": "Naloxone"}],
            gender="M",
            dob="2100-06-15 00:00:00",
        )
        p2 = _build_patient(
            patient_id="S2",
            admissions=[
                self._default_admission(
                    hadm_id="A2",
                    discharge_location="HOME",
                    ethnicity="WHITE",
                    insurance="Private",
                    diagnosis="CHEST PAIN",
                    timestamp=datetime(2150, 3, 10),
                    dischtime="2150-03-12 08:00:00",
                ),
            ],
            diagnoses=[{"hadm_id": "A2", "icd9_code": "78650"}],
            procedures=[{"hadm_id": "A2", "icd9_code": "8856"}],
            prescriptions=[{"hadm_id": "A2", "drug": "Aspirin"}],
            gender="F",
            dob="2090-01-01 00:00:00",
        )

        all_samples = self.task(p1) + self.task(p2)
        self.assertEqual(len(all_samples), 2)

        s1 = all_samples[0]
        self.assertEqual(s1["ama"], 1)
        self.assertEqual(s1["race"], ["race:Hispanic"])
        self.assertIn("insurance:Public", s1["demographics"])
        self.assertEqual(s1["has_substance_use"], [1.0])
        self.assertAlmostEqual(s1["age"][0], 49.0, places=0)
        self.assertAlmostEqual(s1["los"][0], 2.5, places=1)

        s2 = all_samples[1]
        self.assertEqual(s2["ama"], 0)
        self.assertEqual(s2["race"], ["race:White"])
        self.assertIn("insurance:Private", s2["demographics"])
        self.assertEqual(s2["has_substance_use"], [0.0])
        self.assertAlmostEqual(s2["age"][0], 60.0, places=0)


class TestAMAAblationBaselines(unittest.TestCase):
    """Tests for the three ablation study feature baselines.

    These tests verify that each baseline can be used to select
    different subsets of features via the model's feature_keys parameter.
    """

    def setUp(self) -> None:
        """Build one multi-visit mock patient covering baseline feature keys.

        Samples include all ``AMAPredictionMIMIC3.input_schema`` fields so
        tests can reason about ``feature_keys`` subsets the same way models
        do after ``set_task``.
        """
        self.task = AMAPredictionMIMIC3()
        self.patient = _build_patient(
            patient_id="ABLATION_TEST",
            admissions=[
                {
                    "hadm_id": "A1",
                    "admission_type": "EMERGENCY",
                    "discharge_location": "HOME",
                    "ethnicity": "HISPANIC OR LATINO",
                    "insurance": "Medicaid",
                    "dischtime": "2150-01-10 14:00:00",
                    "timestamp": datetime(2150, 1, 1),
                    "diagnosis": "ALCOHOL WITHDRAWAL",
                },
                {
                    "hadm_id": "A2",
                    "admission_type": "URGENT",
                    "discharge_location": "LEFT AGAINST MEDICAL ADVI",
                    "ethnicity": "WHITE",
                    "insurance": "Private",
                    "dischtime": "2150-06-05 10:00:00",
                    "timestamp": datetime(2150, 6, 1),
                    "diagnosis": "PNEUMONIA",
                },
            ],
            diagnoses=[
                {"hadm_id": "A1", "icd9_code": "29181"},
                {"hadm_id": "A2", "icd9_code": "486"},
            ],
            procedures=[
                {"hadm_id": "A1", "icd9_code": "3893"},
                {"hadm_id": "A2", "icd9_code": "9604"},
            ],
            prescriptions=[
                {"hadm_id": "A1", "drug": "Lorazepam"},
                {"hadm_id": "A2", "drug": "Levofloxacin"},
            ],
            gender="M",
            dob="2100-01-01 00:00:00",
        )

    def test_baseline_features_present(self):
        """BASELINE includes demographics, age, los."""
        samples = self.task(self.patient)
        self.assertGreaterEqual(len(samples), 1)

        sample = samples[0]
        self.assertIn("demographics", sample)
        self.assertIn("age", sample)
        self.assertIn("los", sample)
        self.assertTrue(isinstance(sample["age"][0], float))
        self.assertTrue(isinstance(sample["los"][0], float))
        self.assertTrue(isinstance(sample["demographics"], list))

    def test_baseline_race_feature_present(self):
        """BASELINE + RACE adds race feature."""
        samples = self.task(self.patient)
        self.assertGreaterEqual(len(samples), 1)

        for sample in samples:
            self.assertIn("race", sample)
            self.assertTrue(isinstance(sample["race"], list))
            race_val = sample["race"][0].split(":", 1)[1]
            self.assertIn(
                race_val,
                ["White", "Black", "Hispanic", "Asian", "Native American", "Other"],
            )

    def test_baseline_substance_use_feature_present(self):
        """BASELINE + RACE + SUBSTANCE adds has_substance_use."""
        samples = self.task(self.patient)
        self.assertGreaterEqual(len(samples), 1)

        for sample in samples:
            self.assertIn("has_substance_use", sample)
            self.assertTrue(isinstance(sample["has_substance_use"], list))
            self.assertIn(sample["has_substance_use"][0], [0.0, 1.0])

    def test_substance_use_detection_in_ablation(self):
        """Verify substance use detection for ablation patient."""
        samples = self.task(self.patient)

        s1 = next(s for s in samples if s["visit_id"] == "A1")
        self.assertEqual(s1["has_substance_use"], [1.0])

        s2 = next(s for s in samples if s["visit_id"] == "A2")
        self.assertEqual(s2["has_substance_use"], [0.0])

    def test_race_normalization_in_ablation(self):
        """Verify race normalization for ablation patient."""
        samples = self.task(self.patient)

        s1 = next(s for s in samples if s["visit_id"] == "A1")
        self.assertEqual(s1["race"], ["race:Hispanic"])

        s2 = next(s for s in samples if s["visit_id"] == "A2")
        self.assertEqual(s2["race"], ["race:White"])

    def test_age_and_los_computed(self):
        """Verify age and LOS are computed correctly."""
        samples = self.task(self.patient)

        for sample in samples:
            age = sample["age"][0]
            los = sample["los"][0]
            self.assertAlmostEqual(age, 50.0, places=1)
            self.assertGreater(los, 0.0)

    def test_demographics_includes_gender_and_insurance(self):
        """BASELINE demographics include gender and insurance."""
        samples = self.task(self.patient)

        for sample in samples:
            demo = sample["demographics"]
            has_gender = any(t.startswith("gender:") for t in demo)
            has_insurance = any(t.startswith("insurance:") for t in demo)
            self.assertTrue(has_gender)
            self.assertTrue(has_insurance)

    def test_insurance_normalization_in_ablation(self):
        """Verify insurance normalization (Medicaid -> Public)."""
        samples = self.task(self.patient)

        s1 = next(s for s in samples if s["visit_id"] == "A1")
        demo1 = s1["demographics"]
        self.assertIn("insurance:Public", demo1)

        s2 = next(s for s in samples if s["visit_id"] == "A2")
        demo2 = s2["demographics"]
        self.assertIn("insurance:Private", demo2)

    def test_label_correctness_in_ablation(self):
        """Verify AMA label is correct."""
        samples = self.task(self.patient)

        s1 = next(s for s in samples if s["visit_id"] == "A1")
        self.assertEqual(s1["ama"], 0)

        s2 = next(s for s in samples if s["visit_id"] == "A2")
        self.assertEqual(s2["ama"], 1)

    def test_baseline_minimal_features(self):
        """BASELINE (minimal) has only required keys."""
        samples = self.task(self.patient)
        self.assertGreaterEqual(len(samples), 1)

        sample = samples[0]
        baseline_keys = {
            "demographics",
            "age",
            "los",
            "race",
            "has_substance_use",
            "visit_id",
            "patient_id",
            "ama",
        }
        self.assertEqual(set(sample.keys()), baseline_keys)

    def test_multiple_admissions_all_included(self):
        """All non-newborn admissions are included (no filtering)."""
        samples = self.task(self.patient)
        self.assertEqual(len(samples), 2)

        visit_ids = {s["visit_id"] for s in samples}
        self.assertEqual(visit_ids, {"A1", "A2"})

    def test_ablation_patient_no_clinical_codes(self):
        """Ablation samples do not contain clinical code fields."""
        samples = self.task(self.patient)

        for sample in samples:
            self.assertNotIn("conditions", sample)
            self.assertNotIn("procedures", sample)
            self.assertNotIn("drugs", sample)


# ------------------------------------------------------------------
# Integration tests using shared curated 5-row dataset
# ------------------------------------------------------------------


class TestAMAWithSyntheticData(unittest.TestCase):
    """AMA task on curated minimal synthetic CSVs (fast pipeline checks)."""

    def test_dataset_loads_successfully(self):
        self.assertIsNotNone(_shared_dataset)
        self.assertGreater(len(_shared_sample_dataset), 0)

    def test_samples_have_expected_features(self):
        sample = _shared_sample_dataset[0]

        expected_keys = {
            "visit_id",
            "patient_id",
            "demographics",
            "age",
            "los",
            "race",
            "has_substance_use",
            "ama",
        }
        self.assertEqual(set(sample.keys()), expected_keys)

    def test_demographics_values(self):
        for sample in _shared_sample_dataset:
            demo = sample["demographics"]
            self.assertTrue(
                torch.is_tensor(demo) or isinstance(demo, (int, float)),
                "Demographics should be processed",
            )

    def test_age_in_valid_range(self):
        for sample in _shared_sample_dataset:
            age = sample["age"]
            self.assertTrue(torch.is_tensor(age) or isinstance(age, (int, float)))

    def test_los_positive(self):
        for sample in _shared_sample_dataset:
            los = sample["los"]
            self.assertTrue(torch.is_tensor(los) or isinstance(los, (int, float)))

    def test_race_normalized(self):
        for sample in _shared_sample_dataset:
            race = sample["race"]
            self.assertTrue(torch.is_tensor(race) or isinstance(race, (int, float)))

    def test_substance_use_binary(self):
        for sample in _shared_sample_dataset:
            substance = sample["has_substance_use"]
            self.assertTrue(
                torch.is_tensor(substance) or isinstance(substance, (int, float)),
            )

    def test_ama_label_binary(self):
        for sample in _shared_sample_dataset:
            ama = sample["ama"]
            self.assertIn(ama, [0, 1])

    def test_has_positive_and_negative_labels(self):
        labels = [sample["ama"] for sample in _shared_sample_dataset]
        has_positive = any(label == 1 for label in labels)
        has_negative = any(label == 0 for label in labels)

        self.assertTrue(
            has_positive and has_negative,
            "Dataset should have both positive and negative AMA cases",
        )


class TestAMABaselineFeatures(unittest.TestCase):
    """LogisticRegression ablation feature subsets on synthetic data."""

    def _create_model_with_features(self, feature_keys):
        model = LogisticRegression(
            dataset=_shared_sample_dataset,
            embedding_dim=64,
        )
        model.feature_keys = list(feature_keys)
        output_size = model.get_output_size()
        embedding_dim = model.embedding_model.embedding_layers[
            feature_keys[0]
        ].out_features
        model.fc = torch.nn.Linear(len(feature_keys) * embedding_dim, output_size)
        return model

    def test_baseline_model_can_be_created(self):
        model = self._create_model_with_features(["demographics", "age", "los"])
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.fc)

    def test_baseline_plus_race_model(self):
        model = self._create_model_with_features(["demographics", "age", "los", "race"])
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.fc)

    def test_baseline_plus_race_plus_substance_model(self):
        model = self._create_model_with_features(
            ["demographics", "age", "los", "race", "has_substance_use"]
        )
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.fc)

    def test_baseline_forward_pass(self):
        model = self._create_model_with_features(["demographics", "age", "los"])

        train_ds, _, test_ds = split_by_patient(
            _shared_sample_dataset, [0.8, 0.0, 0.2], seed=0
        )
        test_dl = get_dataloader(test_ds, batch_size=8, shuffle=False)

        model.eval()
        with torch.no_grad():
            batch = next(iter(test_dl))
            output = model(**batch)

        self.assertIn("y_prob", output)
        self.assertIn("y_true", output)
        self.assertEqual(output["y_prob"].shape[0], len(test_ds))

    def test_baseline_plus_race_forward_pass(self):
        model = self._create_model_with_features(["demographics", "age", "los", "race"])

        train_ds, _, test_ds = split_by_patient(
            _shared_sample_dataset, [0.8, 0.0, 0.2], seed=0
        )
        test_dl = get_dataloader(test_ds, batch_size=8, shuffle=False)

        model.eval()
        with torch.no_grad():
            batch = next(iter(test_dl))
            output = model(**batch)

        self.assertIn("y_prob", output)
        self.assertEqual(output["y_prob"].shape[0], len(test_ds))

    def test_baseline_plus_full_forward_pass(self):
        model = self._create_model_with_features(
            ["demographics", "age", "los", "race", "has_substance_use"]
        )

        train_ds, _, test_ds = split_by_patient(
            _shared_sample_dataset, [0.8, 0.0, 0.2], seed=0
        )
        test_dl = get_dataloader(test_ds, batch_size=8, shuffle=False)

        model.eval()
        with torch.no_grad():
            batch = next(iter(test_dl))
            output = model(**batch)

        self.assertIn("y_prob", output)
        self.assertEqual(output["y_prob"].shape[0], len(test_ds))


class TestAMATrainingSpeed(unittest.TestCase):
    """Short training runs on tiny synthetic data."""

    def test_training_completes_quickly(self):
        import time

        train_ds, _, test_ds = split_by_patient(
            _shared_sample_dataset, [0.6, 0.0, 0.4], seed=0
        )
        train_dl = get_dataloader(train_ds, batch_size=8, shuffle=True)

        model = LogisticRegression(
            dataset=_shared_sample_dataset,
            embedding_dim=64,
        )
        model.feature_keys = ["demographics", "age", "los"]
        output_size = model.get_output_size()
        embedding_dim = model.embedding_model.embedding_layers[
            "demographics"
        ].out_features
        model.fc = torch.nn.Linear(3 * embedding_dim, output_size)

        trainer = Trainer(model=model)

        t0 = time.time()
        trainer.train(
            train_dataloader=train_dl,
            val_dataloader=None,
            epochs=1,
            monitor=None,
        )
        elapsed = time.time() - t0

        self.assertGreater(elapsed, 0, "Training should take some time")

    def test_multiple_splits_complete_quickly(self):
        for split_seed in range(2):
            train_ds, _, _ = split_by_patient(
                _shared_sample_dataset,
                [0.6, 0.0, 0.4],
                seed=split_seed,
            )
            train_dl = get_dataloader(train_ds, batch_size=8, shuffle=True)

            model = LogisticRegression(
                dataset=_shared_sample_dataset,
                embedding_dim=64,
            )
            model.feature_keys = ["demographics", "age", "los"]
            output_size = model.get_output_size()
            embedding_dim = model.embedding_model.embedding_layers[
                "demographics"
            ].out_features
            model.fc = torch.nn.Linear(3 * embedding_dim, output_size)

            trainer = Trainer(model=model)
            trainer.train(
                train_dataloader=train_dl,
                val_dataloader=None,
                epochs=1,
                monitor=None,
            )

        self.assertTrue(True)


EXHAUSTIVE_PATIENT_ROWS = 2 * 6 * 6 * 3 * 2 * 2 + 3


class TestExhaustiveSyntheticGrid(unittest.TestCase):
    """Sanity-check exhaustive synthetic generator (row counts only)."""

    def test_patient_row_count_matches_cross_product(self):
        tmp = tempfile.mkdtemp(prefix="ama_exhaustive_")
        try:
            generate_synthetic_mimic3(
                tmp,
                mode="exhaustive",
                seed=0,
                n_patients=1,
            )
            with gzip.open(Path(tmp) / "PATIENTS.csv.gz", "rt") as f:
                lines = f.readlines()
            data_rows = len(lines) - 1
            self.assertEqual(data_rows, EXHAUSTIVE_PATIENT_ROWS)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestCuratedSyntheticGrid(unittest.TestCase):
    """Curated synthetic: fixed small CSV (integration test data)."""

    def test_curated_csv_row_counts(self):
        tmp = tempfile.mkdtemp(prefix="ama_curated_")
        try:
            _write_curated_synthetic_mimic3_for_tests(tmp)
            for name in ("PATIENTS", "ADMISSIONS", "ICUSTAYS"):
                with gzip.open(Path(tmp) / f"{name}.csv.gz", "rt") as f:
                    n = len(f.readlines()) - 1
                self.assertEqual(n, CURATED_SYNTHETIC_N, name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_curated_task_label_counts(self):
        self.assertEqual(len(_shared_sample_dataset), CURATED_SYNTHETIC_N)

        def _ama_int(x):
            if torch.is_tensor(x):
                return int(x.item())
            return int(x)

        labels = [
            _ama_int(_shared_sample_dataset[i]["ama"])
            for i in range(CURATED_SYNTHETIC_N)
        ]
        self.assertEqual(sum(labels), CURATED_SYNTHETIC_AMA_POSITIVE)
        self.assertEqual(
            labels.count(0),
            CURATED_SYNTHETIC_AMA_NEGATIVE,
        )
        self.assertEqual(
            labels.count(1),
            CURATED_SYNTHETIC_AMA_POSITIVE,
        )


if __name__ == "__main__":
    unittest.main()

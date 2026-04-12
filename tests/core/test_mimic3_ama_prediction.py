import unittest
from datetime import datetime
from unittest.mock import MagicMock

from pyhealth.tasks.ama_prediction import (
    AMAPredictionMIMIC3,
    _has_substance_use,
    _normalize_insurance,
    _normalize_race,
)


def _make_event(**attrs):
    """Create a mock event with the given attributes."""
    event = MagicMock()
    for key, value in attrs.items():
        setattr(event, key, value)
    return event


def _build_patient(
    patient_id,
    admissions,
    diagnoses,
    procedures,
    prescriptions,
    gender="M",
    dob="2100-01-01 00:00:00",
):
    """Build a mock Patient with ``get_events`` that respects filters.

    Uses 2-5 synthetic patients max.  No real dataset is loaded.

    Args:
        patient_id: Patient identifier string.
        admissions: List of dicts for admission events.
        diagnoses: List of dicts for diagnosis events.
        procedures: List of dicts for procedure events.
        prescriptions: List of dicts for prescription events.
        gender: Gender string for the demographics event.
        dob: Date-of-birth string for computing age.
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
            source = [
                e
                for e in source
                if getattr(e, col, None) == val
            ]
        return source

    patient.get_events = _get_events
    return patient


SAMPLE_KEYS = {
    "visit_id",
    "patient_id",
    "conditions",
    "procedures",
    "drugs",
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
        self.assertEqual(
            _normalize_race("WHITE - RUSSIAN"), "White"
        )

    def test_black(self):
        self.assertEqual(
            _normalize_race("BLACK/AFRICAN AMERICAN"), "Black"
        )

    def test_hispanic(self):
        self.assertEqual(
            _normalize_race("HISPANIC OR LATINO"), "Hispanic"
        )
        self.assertEqual(
            _normalize_race("SOUTH AMERICAN"), "Hispanic"
        )

    def test_asian(self):
        self.assertEqual(
            _normalize_race("ASIAN - CHINESE"), "Asian"
        )

    def test_native_american(self):
        self.assertEqual(
            _normalize_race("AMERICAN INDIAN/ALASKA NATIVE"),
            "Native American",
        )

    def test_other(self):
        self.assertEqual(
            _normalize_race("UNKNOWN/NOT SPECIFIED"), "Other"
        )
        self.assertEqual(_normalize_race(None), "Other")

    def test_normalize_insurance(self):
        self.assertEqual(
            _normalize_insurance("Medicare"), "Public"
        )
        self.assertEqual(
            _normalize_insurance("Medicaid"), "Public"
        )
        self.assertEqual(
            _normalize_insurance("Government"), "Public"
        )
        self.assertEqual(
            _normalize_insurance("Private"), "Private"
        )
        self.assertEqual(
            _normalize_insurance("Self Pay"), "Self Pay"
        )
        self.assertEqual(_normalize_insurance(None), "Other")


class TestHasSubstanceUse(unittest.TestCase):
    """Unit tests for the substance-use detection helper."""

    def test_alcohol(self):
        self.assertEqual(
            _has_substance_use("ALCOHOL WITHDRAWAL"), 1
        )

    def test_opioid(self):
        self.assertEqual(
            _has_substance_use("OPIOID DEPENDENCE"), 1
        )

    def test_heroin(self):
        self.assertEqual(
            _has_substance_use("HEROIN OVERDOSE"), 1
        )

    def test_cocaine(self):
        self.assertEqual(
            _has_substance_use("COCAINE INTOXICATION"), 1
        )

    def test_drug_withdrawal(self):
        self.assertEqual(
            _has_substance_use("DRUG WITHDRAWAL SEIZURE"), 1
        )

    def test_etoh(self):
        self.assertEqual(_has_substance_use("ETOH ABUSE"), 1)

    def test_substance(self):
        self.assertEqual(
            _has_substance_use("SUBSTANCE ABUSE"), 1
        )

    def test_overdose(self):
        self.assertEqual(
            _has_substance_use("OVERDOSE - ACCIDENTAL"), 1
        )

    def test_negative(self):
        self.assertEqual(_has_substance_use("PNEUMONIA"), 0)
        self.assertEqual(_has_substance_use("CHEST PAIN"), 0)

    def test_none(self):
        self.assertEqual(_has_substance_use(None), 0)

    def test_case_insensitive(self):
        self.assertEqual(
            _has_substance_use("alcohol withdrawal"), 1
        )
        self.assertEqual(
            _has_substance_use("Heroin Overdose"), 1
        )


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
        self.assertEqual(
            schema["has_substance_use"], "tensor"
        )
        self.assertEqual(schema["conditions"], "sequence")
        self.assertEqual(schema["procedures"], "sequence")
        self.assertEqual(schema["drugs"], "sequence")

    def test_output_schema(self):
        self.assertEqual(
            AMAPredictionMIMIC3.output_schema, {"ama": "binary"}
        )

    def test_defaults(self):
        task = AMAPredictionMIMIC3()
        self.assertTrue(task.exclude_newborns)


class TestAMAPredictionMIMIC3Mock(unittest.TestCase):
    """Unit tests using purely synthetic mock patients.

    No real dataset is loaded.  Each test builds mock Patient
    objects in memory and runs the task callable directly.
    All tests complete in milliseconds.
    """

    def setUp(self):
        self.task = AMAPredictionMIMIC3()

    def _default_admission(self, hadm_id="100", **overrides):
        """Return a standard admission dict."""
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
                    discharge_location=(
                        "LEFT AGAINST MEDICAL ADVI"
                    ),
                ),
            ],
            diagnoses=[
                {"hadm_id": "100", "icd9_code": "4019"}
            ],
            procedures=[
                {"hadm_id": "100", "icd9_code": "3893"}
            ],
            prescriptions=[
                {"hadm_id": "100", "drug": "Aspirin"}
            ],
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
            diagnoses=[
                {"hadm_id": "200", "icd9_code": "25000"}
            ],
            procedures=[
                {"hadm_id": "200", "icd9_code": "3995"}
            ],
            prescriptions=[
                {"hadm_id": "200", "drug": "Metformin"}
            ],
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
                    discharge_location=(
                        "LEFT AGAINST MEDICAL ADVI"
                    ),
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

    def test_skip_missing_diagnoses(self):
        """No diagnosis codes -> skip admission."""
        patient = _build_patient(
            patient_id="P4",
            admissions=[
                self._default_admission(hadm_id="400")
            ],
            diagnoses=[],
            procedures=[
                {"hadm_id": "400", "icd9_code": "3893"}
            ],
            prescriptions=[
                {"hadm_id": "400", "drug": "Aspirin"}
            ],
        )
        self.assertEqual(self.task(patient), [])

    def test_skip_missing_procedures(self):
        """No procedure codes -> skip admission."""
        patient = _build_patient(
            patient_id="P5",
            admissions=[
                self._default_admission(hadm_id="500")
            ],
            diagnoses=[
                {"hadm_id": "500", "icd9_code": "4019"}
            ],
            procedures=[],
            prescriptions=[
                {"hadm_id": "500", "drug": "Aspirin"}
            ],
        )
        self.assertEqual(self.task(patient), [])

    def test_skip_missing_prescriptions(self):
        """No prescriptions -> skip admission."""
        patient = _build_patient(
            patient_id="P6",
            admissions=[
                self._default_admission(
                    hadm_id="600",
                    discharge_location=(
                        "LEFT AGAINST MEDICAL ADVI"
                    ),
                ),
            ],
            diagnoses=[
                {"hadm_id": "600", "icd9_code": "4019"}
            ],
            procedures=[
                {"hadm_id": "600", "icd9_code": "3893"}
            ],
            prescriptions=[],
        )
        self.assertEqual(self.task(patient), [])

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
            diagnoses=[
                {"hadm_id": "700", "icd9_code": "V3000"}
            ],
            procedures=[
                {"hadm_id": "700", "icd9_code": "9904"}
            ],
            prescriptions=[
                {"hadm_id": "700", "drug": "Vitamin K"}
            ],
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
            diagnoses=[
                {"hadm_id": "800", "icd9_code": "4280"}
            ],
            procedures=[
                {"hadm_id": "800", "icd9_code": "3722"}
            ],
            prescriptions=[
                {"hadm_id": "800", "drug": "Furosemide"}
            ],
        )
        samples = self.task(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(set(samples[0].keys()), SAMPLE_KEYS)

    def test_clinical_codes_content(self):
        """Extracted codes match the input data."""
        patient = _build_patient(
            patient_id="P9",
            admissions=[
                self._default_admission(
                    hadm_id="900",
                    timestamp=datetime(2150, 4, 1),
                ),
            ],
            diagnoses=[
                {"hadm_id": "900", "icd9_code": "4019"},
                {"hadm_id": "900", "icd9_code": "25000"},
            ],
            procedures=[
                {"hadm_id": "900", "icd9_code": "3893"},
            ],
            prescriptions=[
                {"hadm_id": "900", "drug": "Lisinopril"},
                {"hadm_id": "900", "drug": "Metformin"},
            ],
        )
        samples = self.task(patient)
        self.assertEqual(
            samples[0]["conditions"], ["4019", "25000"]
        )
        self.assertEqual(samples[0]["procedures"], ["3893"])
        self.assertEqual(
            samples[0]["drugs"],
            ["Lisinopril", "Metformin"],
        )

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
            diagnoses=[
                {"hadm_id": "1000", "icd9_code": "4019"}
            ],
            procedures=[
                {"hadm_id": "1000", "icd9_code": "3893"}
            ],
            prescriptions=[
                {"hadm_id": "1000", "drug": "Aspirin"}
            ],
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
            diagnoses=[
                {"hadm_id": "1001", "icd9_code": "4019"}
            ],
            procedures=[
                {"hadm_id": "1001", "icd9_code": "3893"}
            ],
            prescriptions=[
                {"hadm_id": "1001", "drug": "Aspirin"}
            ],
        )
        samples = self.task(patient)
        self.assertIn("race", samples[0])
        self.assertEqual(
            samples[0]["race"], ["race:Black"]
        )

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
            diagnoses=[
                {"hadm_id": "1400", "icd9_code": "29181"}
            ],
            procedures=[
                {"hadm_id": "1400", "icd9_code": "3893"}
            ],
            prescriptions=[
                {"hadm_id": "1400", "drug": "Lorazepam"}
            ],
        )
        samples = self.task(patient)
        self.assertEqual(
            samples[0]["has_substance_use"], [1.0]
        )

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
            diagnoses=[
                {"hadm_id": "1500", "icd9_code": "486"}
            ],
            procedures=[
                {"hadm_id": "1500", "icd9_code": "3893"}
            ],
            prescriptions=[
                {"hadm_id": "1500", "drug": "Levofloxacin"}
            ],
        )
        samples = self.task(patient)
        self.assertEqual(
            samples[0]["has_substance_use"], [0.0]
        )

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
            diagnoses=[
                {"hadm_id": "1100", "icd9_code": "4019"}
            ],
            procedures=[
                {"hadm_id": "1100", "icd9_code": "3893"}
            ],
            prescriptions=[
                {"hadm_id": "1100", "drug": "Aspirin"}
            ],
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
            diagnoses=[
                {"hadm_id": "1200", "icd9_code": "4019"}
            ],
            procedures=[
                {"hadm_id": "1200", "icd9_code": "3893"}
            ],
            prescriptions=[
                {"hadm_id": "1200", "drug": "Aspirin"}
            ],
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
            diagnoses=[
                {"hadm_id": "1300", "icd9_code": "4019"}
            ],
            procedures=[
                {"hadm_id": "1300", "icd9_code": "3893"}
            ],
            prescriptions=[
                {"hadm_id": "1300", "drug": "Aspirin"}
            ],
        )
        samples = self.task(patient)
        self.assertAlmostEqual(
            samples[0]["los"][0], 5.0, places=2
        )

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
                    discharge_location=(
                        "LEFT AGAINST MEDICAL ADVI"
                    ),
                    ethnicity="HISPANIC OR LATINO",
                    insurance="Medicaid",
                    diagnosis="HEROIN OVERDOSE",
                    timestamp=datetime(2150, 1, 1),
                    dischtime="2150-01-03 12:00:00",
                ),
            ],
            diagnoses=[
                {"hadm_id": "A1", "icd9_code": "96500"}
            ],
            procedures=[
                {"hadm_id": "A1", "icd9_code": "9604"}
            ],
            prescriptions=[
                {"hadm_id": "A1", "drug": "Naloxone"}
            ],
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
            diagnoses=[
                {"hadm_id": "A2", "icd9_code": "78650"}
            ],
            procedures=[
                {"hadm_id": "A2", "icd9_code": "8856"}
            ],
            prescriptions=[
                {"hadm_id": "A2", "drug": "Aspirin"}
            ],
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
        self.assertIn(
            "insurance:Private", s2["demographics"]
        )
        self.assertEqual(s2["has_substance_use"], [0.0])
        self.assertAlmostEqual(s2["age"][0], 60.0, places=0)


if __name__ == "__main__":
    unittest.main()

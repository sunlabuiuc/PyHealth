import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from pyhealth.tasks.ama_prediction import (
    AMAPredictionMIMIC3,
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
    """Build a mock Patient whose ``get_events`` respects *filters*.

    Args:
        patient_id: Patient identifier string.
        admissions: List of admission event dicts (should include
            ``ethnicity``, ``insurance``, ``dischtime``).
        diagnoses: List of diagnosis event dicts (must include ``hadm_id``).
        procedures: List of procedure event dicts (must include ``hadm_id``).
        prescriptions: List of prescription event dicts (must include ``hadm_id``).
        gender: Gender string for the patient demographics event.
        dob: Date of birth string for computing age.
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
    "conditions",
    "procedures",
    "drugs",
    "demographics",
    "age",
    "los",
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
        self.assertEqual(
            _normalize_race("SOUTH AMERICAN"), "Hispanic"
        )

    def test_asian(self):
        self.assertEqual(
            _normalize_race("ASIAN - CHINESE"), "Asian"
        )

    def test_native_american(self):
        self.assertEqual(
            _normalize_race("AMERICAN INDIAN/ALASKA NATIVE"), "Native American"
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


class TestAMAPredictionMIMIC3Schema(unittest.TestCase):
    """Validate class-level schema attributes."""

    def test_task_name(self):
        self.assertEqual(AMAPredictionMIMIC3.task_name, "AMAPredictionMIMIC3")

    def test_input_schema(self):
        schema = AMAPredictionMIMIC3.input_schema
        self.assertEqual(schema["conditions"], "sequence")
        self.assertEqual(schema["procedures"], "sequence")
        self.assertEqual(schema["drugs"], "sequence")
        self.assertEqual(schema["demographics"], "multi_hot")
        self.assertEqual(schema["age"], "tensor")
        self.assertEqual(schema["los"], "tensor")

    def test_output_schema(self):
        self.assertEqual(AMAPredictionMIMIC3.output_schema, {"ama": "binary"})

    def test_defaults(self):
        task = AMAPredictionMIMIC3()
        self.assertTrue(task.exclude_newborns)


class TestAMAPredictionMIMIC3Mock(unittest.TestCase):
    """Unit tests using synthetic mock data (no real dataset needed)."""

    def setUp(self):
        self.task = AMAPredictionMIMIC3()

    def _default_admission(self, hadm_id="100", **overrides):
        """Return a standard admission dict with demographic fields."""
        adm = {
            "hadm_id": hadm_id,
            "admission_type": "EMERGENCY",
            "discharge_location": "HOME",
            "ethnicity": "WHITE",
            "insurance": "Private",
            "dischtime": "2150-01-10 14:00:00",
            "timestamp": datetime(2150, 1, 1),
        }
        adm.update(overrides)
        return adm

    def test_empty_patient(self):
        patient = MagicMock()
        patient.patient_id = "P0"
        patient.get_events = lambda event_type, **kw: []
        self.assertEqual(self.task(patient), [])

    def test_ama_label_positive(self):
        """Admission with AMA discharge should produce label=1."""
        patient = _build_patient(
            patient_id="P1",
            admissions=[
                self._default_admission(
                    hadm_id="100",
                    discharge_location="LEFT AGAINST MEDICAL ADVI",
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
        """Non-AMA discharge should produce label=0."""
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
                    discharge_location="LEFT AGAINST MEDICAL ADVI",
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

    def test_skip_admission_missing_diagnoses(self):
        """Admissions with no diagnosis codes should be excluded."""
        patient = _build_patient(
            patient_id="P4",
            admissions=[self._default_admission(hadm_id="400")],
            diagnoses=[],
            procedures=[{"hadm_id": "400", "icd9_code": "3893"}],
            prescriptions=[{"hadm_id": "400", "drug": "Aspirin"}],
        )
        self.assertEqual(self.task(patient), [])

    def test_skip_admission_missing_procedures(self):
        """Admissions with no procedure codes should be excluded."""
        patient = _build_patient(
            patient_id="P5",
            admissions=[self._default_admission(hadm_id="500")],
            diagnoses=[{"hadm_id": "500", "icd9_code": "4019"}],
            procedures=[],
            prescriptions=[{"hadm_id": "500", "drug": "Aspirin"}],
        )
        self.assertEqual(self.task(patient), [])

    def test_skip_admission_missing_prescriptions(self):
        """Admissions with no prescriptions should be excluded."""
        patient = _build_patient(
            patient_id="P6",
            admissions=[
                self._default_admission(
                    hadm_id="600",
                    discharge_location="LEFT AGAINST MEDICAL ADVI",
                ),
            ],
            diagnoses=[{"hadm_id": "600", "icd9_code": "4019"}],
            procedures=[{"hadm_id": "600", "icd9_code": "3893"}],
            prescriptions=[],
        )
        self.assertEqual(self.task(patient), [])

    def test_exclude_newborns(self):
        """NEWBORN admissions should be excluded by default."""
        patient = _build_patient(
            patient_id="P7",
            admissions=[
                self._default_admission(
                    hadm_id="700", admission_type="NEWBORN"
                ),
            ],
            diagnoses=[{"hadm_id": "700", "icd9_code": "V3000"}],
            procedures=[{"hadm_id": "700", "icd9_code": "9904"}],
            prescriptions=[{"hadm_id": "700", "drug": "Vitamin K"}],
        )
        task_exclude = AMAPredictionMIMIC3(exclude_newborns=True)
        self.assertEqual(task_exclude(patient), [])

        task_include = AMAPredictionMIMIC3(exclude_newborns=False)
        samples = task_include(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["ama"], 0)

    def test_sample_keys(self):
        """Every sample must contain the expected keys."""
        patient = _build_patient(
            patient_id="P8",
            admissions=[
                self._default_admission(
                    hadm_id="800", discharge_location="SNF",
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

    def test_clinical_codes_content(self):
        """Verify extracted codes match the input data."""
        patient = _build_patient(
            patient_id="P9",
            admissions=[
                self._default_admission(
                    hadm_id="900", timestamp=datetime(2150, 4, 1),
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
        self.assertEqual(samples[0]["conditions"], ["4019", "25000"])
        self.assertEqual(samples[0]["procedures"], ["3893"])
        self.assertEqual(samples[0]["drugs"], ["Lisinopril", "Metformin"])

    def test_demographics_tokens(self):
        """Demographics should include prefixed gender, race, insurance."""
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
        self.assertIn("race:Black", demo)
        self.assertIn("insurance:Public", demo)

    def test_age_calculation(self):
        """Age should be computed from dob and admission timestamp."""
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
        """Ages above 90 should be capped (MIMIC-III de-identification)."""
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
        """LOS should be computed in days from admittime to dischtime."""
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


class TestAMAPredictionMIMIC3Integration(unittest.TestCase):
    """Integration test using the MIMIC-III demo dataset.

    The demo dataset contains zero AMA discharge events.  Because the
    ``BinaryLabelProcessor`` requires exactly two unique labels to fit,
    ``set_task()`` cannot complete on this dataset.  Instead we verify
    that the task callable itself produces well-formed samples when
    invoked on real ``Patient`` objects from the loaded dataset.
    """

    @classmethod
    def setUpClass(cls):
        from pyhealth.datasets import MIMIC3Dataset

        cls.cache_dir = tempfile.TemporaryDirectory()
        demo_path = str(
            Path(__file__).parent.parent.parent
            / "test-resources"
            / "core"
            / "mimic3demo"
        )
        cls.dataset = MIMIC3Dataset(
            root=demo_path,
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            cache_dir=cls.cache_dir.name,
        )
        cls.task = AMAPredictionMIMIC3()

    def test_task_callable_on_real_patients(self):
        """Run the task on real Patient objects from the demo dataset."""
        total_samples = 0
        for patient in self.dataset.iter_patients():
            samples = self.task(patient)
            for sample in samples:
                self.assertIn("ama", sample)
                self.assertEqual(sample["ama"], 0)
                self.assertIsInstance(sample["conditions"], list)
                self.assertIsInstance(sample["procedures"], list)
                self.assertIsInstance(sample["drugs"], list)
                self.assertGreater(len(sample["conditions"]), 0)
                self.assertGreater(len(sample["procedures"]), 0)
                self.assertGreater(len(sample["drugs"]), 0)
                self.assertIsInstance(sample["demographics"], list)
                self.assertGreater(len(sample["demographics"]), 0)
                self.assertIsInstance(sample["age"], list)
                self.assertEqual(len(sample["age"]), 1)
                self.assertIsInstance(sample["los"], list)
                self.assertEqual(len(sample["los"]), 1)
                total_samples += 1
        self.assertGreater(total_samples, 0, "Should produce at least one sample")


if __name__ == "__main__":
    unittest.main()

"""Unit tests for NotesLabsMIMIC4, admission-section extraction, and ICDLabsMIMIC4 fixes."""

from datetime import datetime
import unittest


class _DummyEvent:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _DummyPatientWithNotes:
    def __init__(self, note_texts=None, icd_codes=None, lab_df=None, vital_df=None) -> None:
        self.patient_id = "p-1"
        self._admissions = [
            _DummyEvent(
                timestamp=datetime(2020, 1, 1, 0, 0, 0),
                dischtime="2020-01-03 12:00:00",
                hadm_id=101,
                hospital_expire_flag=0,
            )
        ]
        self._patients = [_DummyEvent(anchor_age=55)]
        self._note_texts = note_texts or []
        self._icd_codes = icd_codes or []
        self._lab_df = lab_df
        self._vital_df = vital_df

    def get_events(self, event_type, start=None, end=None, filters=None, return_df=False):
        if event_type == "patients":
            return self._patients
        if event_type == "admissions":
            return self._admissions
        if event_type == "discharge":
            out = []
            for text in self._note_texts:
                out.append(
                    _DummyEvent(
                        timestamp=datetime(2020, 1, 3, 12, 0, 0),
                        text=text,
                    )
                )
            return out
        if event_type in {"diagnoses_icd", "procedures_icd"}:
            return self._icd_codes
        if event_type == "labevents" and return_df:
            return self._lab_df
        if event_type == "chartevents" and return_df:
            return self._vital_df if self._vital_df is not None else pl.DataFrame(
                {
                    "timestamp": [],
                    "chartevents/itemid": [],
                    "chartevents/storetime": [],
                    "chartevents/valuenum": [],
                }
            )
        if event_type == "chartevents" and not return_df:
            return []
        return []


class _DummyPatientMalformedDischtime:
    def __init__(self) -> None:
        self.patient_id = "p-2"
        self._admissions = [
            _DummyEvent(
                timestamp=datetime(2020, 1, 1, 0, 0, 0),
                dischtime="malformed-dischtime",
                hadm_id=102,
                hospital_expire_flag=0,
            )
        ]
        self._patients = [_DummyEvent(anchor_age=60)]

    def get_events(self, event_type, start=None, end=None, filters=None, return_df=False):
        if event_type == "patients":
            return self._patients
        if event_type == "admissions":
            return self._admissions
        if event_type in {"diagnoses_icd", "procedures_icd", "discharge", "radiology"}:
            return []
        if event_type == "labevents" and return_df:
            import polars as pl

            return pl.DataFrame(
                {
                    "timestamp": [],
                    "labevents/itemid": [],
                    "labevents/storetime": [],
                    "labevents/valuenum": [],
                }
            )
        if event_type == "chartevents" and return_df:
            import polars as pl

            return pl.DataFrame(
                {
                    "timestamp": [],
                    "chartevents/itemid": [],
                    "chartevents/storetime": [],
                    "chartevents/valuenum": [],
                }
            )
        return []


class TestExtractAdmissionSections(unittest.TestCase):
    def test_extracts_target_sections(self):
        from pyhealth.tasks.multimodal_mimic4 import BaseMultimodalMIMIC4Task

        text = """Chief Complaint:
Shortness of breath

Past Medical History:
1. Hypertension

Medications on Admission:
1. Metoprolol 25 mg PO BID

Discharge Diagnosis:
Acute MI
"""
        result = BaseMultimodalMIMIC4Task._extract_admission_sections(text)
        self.assertIn("Shortness of breath", result)
        self.assertIn("Hypertension", result)
        self.assertIn("Metoprolol", result)
        self.assertNotIn("Acute MI", result)
        self.assertIn("[SEP]", result)

    def test_fallback_to_first_1024_when_no_sections(self):
        from pyhealth.tasks.multimodal_mimic4 import BaseMultimodalMIMIC4Task

        text = "This is a note with no section headers at all. " * 50
        result = BaseMultimodalMIMIC4Task._extract_admission_sections(text)
        self.assertEqual(result, text[:1024])

    def test_case_insensitive_headers(self):
        from pyhealth.tasks.multimodal_mimic4 import BaseMultimodalMIMIC4Task

        text = """CHIEF COMPLAINT:
Chest pain

Past Medical/Surgical History:
Appendectomy
"""
        result = BaseMultimodalMIMIC4Task._extract_admission_sections(text)
        self.assertIn("Chest pain", result)
        self.assertIn("Appendectomy", result)

    def test_empty_string_fallback(self):
        from pyhealth.tasks.multimodal_mimic4 import BaseMultimodalMIMIC4Task

        result = BaseMultimodalMIMIC4Task._extract_admission_sections("")
        self.assertEqual(result, "")


class TestCollectAdmissionNoteSections(unittest.TestCase):
    def test_collects_sections_and_returns_time_zero(self):
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4

        task = NotesLabsMIMIC4()
        patient = _DummyPatientWithNotes(
            note_texts=["Chief Complaint:\nFever\n\nPast Medical History:\nDiabetes"]
        )
        texts, times = task._collect_admission_note_sections(
            patient, 101, datetime(2020, 1, 1, 0, 0, 0)
        )
        self.assertEqual(len(texts), 1)
        self.assertIn("Fever", texts[0])
        self.assertEqual(times, [0.0])

    def test_missing_note_fallback(self):
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4

        task = NotesLabsMIMIC4()
        patient = _DummyPatientWithNotes(note_texts=[])
        texts, times = task._collect_admission_note_sections(
            patient, 101, datetime(2020, 1, 1, 0, 0, 0)
        )
        self.assertEqual(texts, [""])
        self.assertEqual(times, [0.0])

    def test_no_time_filter_applied(self):
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4

        task = NotesLabsMIMIC4()
        # Discharge note timestamp is 2020-01-03, well outside any 24h window
        patient = _DummyPatientWithNotes(
            note_texts=["Chief Complaint:\nFever"]
        )
        texts, times = task._collect_admission_note_sections(
            patient, 101, datetime(2020, 1, 1, 0, 0, 0)
        )
        self.assertEqual(len(texts), 1)
        self.assertIn("Fever", texts[0])


class TestNotesLabsMIMIC4(unittest.TestCase):
    def test_default_schema_excludes_icd(self):
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4

        task = NotesLabsMIMIC4()
        self.assertNotIn("icd_codes", task.input_schema)
        self.assertNotIn("vitals", task.input_schema)
        self.assertNotIn("vitals_mask", task.input_schema)
        self.assertIn("admission_note_times", task.input_schema)
        self.assertIn("labs", task.input_schema)
        self.assertIn("labs_mask", task.input_schema)

    def test_include_icd_adds_icd_schema(self):
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4

        task = NotesLabsMIMIC4(include_icd=True)
        self.assertIn("icd_codes", task.input_schema)

    def test_include_vitals_adds_vitals_schema(self):
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4

        task = NotesLabsMIMIC4(include_vitals=True)
        self.assertIn("vitals", task.input_schema)
        self.assertIn("vitals_mask", task.input_schema)
        self.assertNotIn("icd_codes", task.input_schema)

    def test_include_vitals_and_icd_adds_both(self):
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4

        task = NotesLabsMIMIC4(include_icd=True, include_vitals=True)
        self.assertIn("vitals", task.input_schema)
        self.assertIn("vitals_mask", task.input_schema)
        self.assertIn("icd_codes", task.input_schema)

    def test_output_structure_with_vitals(self):
        import polars as pl
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4

        task = NotesLabsMIMIC4(window_hours=24, include_vitals=True)
        lab_df = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, 2, 0, 0)],
                "labevents/itemid": ["50824"],
                "labevents/storetime": ["2020-01-01 02:00:00"],
                "labevents/valuenum": [138.0],
            }
        )
        vital_df = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, 1, 0, 0)],
                "chartevents/itemid": ["220045"],  # HeartRate
                "chartevents/storetime": ["2020-01-01 01:00:00"],
                "chartevents/valuenum": [80.0],
            }
        )
        patient = _DummyPatientWithNotes(
            note_texts=["Chief Complaint:\nFever"],
            lab_df=lab_df,
            vital_df=vital_df,
        )
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertIn("vitals", sample)
        self.assertIn("vitals_mask", sample)
        self.assertNotIn("icd_codes", sample)
        vital_times, vital_values = sample["vitals"]
        self.assertGreater(len(vital_times), 0)

    def test_vitals_fallback_when_empty(self):
        import polars as pl
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4

        task = NotesLabsMIMIC4(window_hours=24, include_vitals=True)
        lab_df = pl.DataFrame(
            {
                "timestamp": [],
                "labevents/itemid": [],
                "labevents/storetime": [],
                "labevents/valuenum": [],
            }
        )
        vital_df = pl.DataFrame(
            {
                "timestamp": [],
                "chartevents/itemid": [],
                "chartevents/storetime": [],
                "chartevents/valuenum": [],
            }
        )
        patient = _DummyPatientWithNotes(
            note_texts=["Chief Complaint:\nFever"],
            lab_df=lab_df,
            vital_df=vital_df,
        )
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        vital_times, vital_values = samples[0]["vitals"]
        self.assertEqual(len(vital_times), 1)
        self.assertEqual(len(vital_values[0]), len(task.VITAL_CATEGORY_NAMES))

    def test_output_structure_no_icd(self):
        import polars as pl
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4

        task = NotesLabsMIMIC4(window_hours=24)
        lab_df = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1, 2, 0, 0)],
                "labevents/itemid": ["50824"],  # Sodium
                "labevents/storetime": ["2020-01-01 02:00:00"],
                "labevents/valuenum": [138.0],
            }
        )
        patient = _DummyPatientWithNotes(
            note_texts=["Chief Complaint:\nFever"],
            lab_df=lab_df,
        )
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertIn("admission_note_times", sample)
        self.assertIn("labs", sample)
        self.assertIn("labs_mask", sample)
        self.assertNotIn("icd_codes", sample)
        self.assertEqual(sample["mortality"], 0)

    def test_output_structure_with_icd(self):
        import polars as pl
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4

        task = NotesLabsMIMIC4(window_hours=24, include_icd=True)
        lab_df = pl.DataFrame(
            {
                "timestamp": [],
                "labevents/itemid": [],
                "labevents/storetime": [],
                "labevents/valuenum": [],
            }
        )
        patient = _DummyPatientWithNotes(
            note_texts=["Chief Complaint:\nFever"],
            icd_codes=[_DummyEvent(icd_code="I21")],
            lab_df=lab_df,
        )
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertIn("icd_codes", sample)

    def test_mortality_label_positive(self):
        import polars as pl
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4

        task = NotesLabsMIMIC4(window_hours=24)
        lab_df = pl.DataFrame(
            {
                "timestamp": [],
                "labevents/itemid": [],
                "labevents/storetime": [],
                "labevents/valuenum": [],
            }
        )
        patient = _DummyPatientWithNotes(
            note_texts=["Chief Complaint:\nFever"],
            lab_df=lab_df,
        )
        patient._admissions[0].hospital_expire_flag = 1
        samples = task(patient)
        self.assertEqual(samples[0]["mortality"], 1)


class TestICDLabsMIMIC4Fixes(unittest.TestCase):
    def test_malformed_dischtime_does_not_drop_admission(self):
        from pyhealth.tasks.multimodal_mimic4 import ICDLabsMIMIC4

        task = ICDLabsMIMIC4(window_hours=24)
        patient = _DummyPatientMalformedDischtime()
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertGreater(len(sample["icd_codes"][0]), 0)
        self.assertGreater(len(sample["labs"][0]), 0)
        self.assertGreater(len(sample["labs_mask"][0]), 0)

    def test_missing_icd_code_uses_text_token(self):
        import polars as pl
        from pyhealth.tasks.multimodal_mimic4 import ICDLabsMIMIC4

        task = ICDLabsMIMIC4(window_hours=24)
        lab_df = pl.DataFrame(
            {
                "timestamp": [],
                "labevents/itemid": [],
                "labevents/storetime": [],
                "labevents/valuenum": [],
            }
        )
        patient = _DummyPatientWithNotes(
            note_texts=[],
            icd_codes=[],  # no ICD codes
            lab_df=lab_df,
        )
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        _, icd_visits = samples[0]["icd_codes"]
        self.assertEqual(icd_visits, [[""]])


if __name__ == "__main__":
    unittest.main()

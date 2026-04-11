import importlib.util
import unittest
from pathlib import Path

import pandas as pd


def _load_task_module():
    module_path = Path(__file__).resolve().parents[2] / "pyhealth" / "tasks" / "eol_mistrust.py"
    spec = importlib.util.spec_from_file_location(
        "pyhealth.tasks.eol_mistrust_task_tests",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _DummyEvent:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _DummyPatient:
    def __init__(self, patient_id, events_by_type):
        self.patient_id = patient_id
        self._events_by_type = events_by_type

    def get_events(self, event_type, filters=None):
        events = list(self._events_by_type.get(event_type, []))
        for field, operator, expected in filters or []:
            if operator != "==":
                raise AssertionError(f"Unexpected filter operator in test double: {operator}")
            events = [event for event in events if getattr(event, field, None) == expected]
        return events


class TestEOLMistrustTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_task_module()

    def test_build_code_status_target_has_clear_corrected_and_paper_like_modes(self):
        chartevents = pd.DataFrame(
            [
                {
                    "hadm_id": 101,
                    "itemid": 128,
                    "value": "Full Code",
                    "charttime": "2100-01-01 10:00:00",
                },
                {
                    "hadm_id": 101,
                    "itemid": 128,
                    "value": "DNR/DNI",
                    "charttime": "2100-01-01 08:00:00",
                },
                {
                    "hadm_id": 102,
                    "itemid": 128,
                    "value": "DNR",
                },
                {
                    "hadm_id": 102,
                    "itemid": 128,
                    "value": "Other/Remarks",
                },
            ]
        )

        corrected = self.module.build_code_status_target(chartevents).set_index("hadm_id")
        paper_like = self.module.build_code_status_target(
            chartevents,
            code_status_mode="paper_like",
        ).set_index("hadm_id")

        self.assertEqual(int(corrected.loc[101, "code_status_dnr_dni_cmo"]), 0)
        self.assertEqual(int(paper_like.loc[101, "code_status_dnr_dni_cmo"]), 1)
        self.assertEqual(int(corrected.loc[102, "code_status_dnr_dni_cmo"]), 0)
        self.assertEqual(int(paper_like.loc[102, "code_status_dnr_dni_cmo"]), 1)

    def test_build_target_tables_keep_expected_public_behavior(self):
        admissions = pd.DataFrame(
            [
                {
                    "hadm_id": 201,
                    "discharge_location": "LEFT AGAINST MEDICAL ADVICE",
                    "hospital_expire_flag": 0,
                },
                {
                    "hadm_id": 202,
                    "discharge_location": "HOME",
                    "hospital_expire_flag": 1,
                },
            ]
        )

        left_ama = self.module.build_left_ama_target(admissions).set_index("hadm_id")
        mortality = self.module.build_in_hospital_mortality_target(admissions).set_index("hadm_id")

        self.assertEqual(int(left_ama.loc[201, "left_ama"]), 1)
        self.assertEqual(int(left_ama.loc[202, "left_ama"]), 0)
        self.assertEqual(int(mortality.loc[201, "in_hospital_mortality"]), 0)
        self.assertEqual(int(mortality.loc[202, "in_hospital_mortality"]), 1)

    def test_downstream_task_wrapper_builds_single_admission_sample(self):
        task = self.module.EOLMistrustCodeStatusPredictionMIMIC3(include_notes=True)
        patient = _DummyPatient(
            patient_id="subject-1",
            events_by_type={
                "patients": [
                    _DummyEvent(gender="F", dob="2070-01-01 00:00:00"),
                ],
                "admissions": [
                    _DummyEvent(
                        hadm_id=301,
                        admittime="2100-01-01 00:00:00",
                        dischtime="2100-01-03 12:00:00",
                        discharge_location="HOME",
                        hospital_expire_flag=0,
                        insurance="Private",
                        ethnicity="BLACK/AFRICAN AMERICAN",
                    ),
                ],
                "diagnoses_icd": [
                    _DummyEvent(hadm_id=301, icd9_code="4019"),
                ],
                "procedures_icd": [
                    _DummyEvent(hadm_id=301, icd9_code="3893"),
                ],
                "prescriptions": [
                    _DummyEvent(hadm_id=301, drug="Aspirin"),
                ],
                "chartevents": [
                    _DummyEvent(hadm_id=301, itemid=128, value="Full Code"),
                    _DummyEvent(hadm_id=301, itemid=128, value="Comfort Measures"),
                ],
                "noteevents": [
                    _DummyEvent(hadm_id=301, text="  family   meeting   note  "),
                ],
            },
        )

        samples = task(patient)

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["visit_id"], 301)
        self.assertEqual(sample["patient_id"], "subject-1")
        self.assertEqual(sample["conditions"], ["4019"])
        self.assertEqual(sample["procedures"], ["3893"])
        self.assertEqual(sample["drugs"], ["Aspirin"])
        self.assertEqual(sample["insurance"], "Private")
        self.assertEqual(sample["race"], "BLACK")
        self.assertEqual(sample["clinical_notes"], "family meeting note")
        self.assertEqual(sample["code_status_dnr_dni_cmo"], 1)
        self.assertGreater(sample["age"], 0.0)
        self.assertGreater(sample["los_days"], 0.0)

    def test_task_map_and_wrapper_targets_stay_consistent(self):
        task_map = self.module.get_eol_mistrust_task_map()

        self.assertEqual(
            list(task_map.items()),
            [
                ("Left AMA", "left_ama"),
                ("Code Status", "code_status_dnr_dni_cmo"),
                ("In-hospital mortality", "in_hospital_mortality"),
            ],
        )
        with self.assertRaisesRegex(ValueError, "Unsupported EOL mistrust target"):
            self.module.EOLMistrustDownstreamMIMIC3(target="unknown")


if __name__ == "__main__":
    unittest.main()

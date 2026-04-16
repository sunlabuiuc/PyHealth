"""Tests for InHospitalMortalityTemporalMIMIC4 task.

Uses synthetic mock patients (not demo/real data). Each test builds
dummy Patient objects and calls the task directly.
"""

import unittest
from collections import Counter
from datetime import datetime

from pyhealth.tasks import InHospitalMortalityTemporalMIMIC4


# -- mock classes to simulate PyHealth Patient/Event objects --

class DummyDemographics:
    def __init__(self, anchor_age, anchor_year=2150,
                 anchor_year_group="2017 - 2019"):
        self.anchor_age = anchor_age
        self.anchor_year = anchor_year
        self.anchor_year_group = anchor_year_group


class DummyAdmission:
    def __init__(self, hadm_id, timestamp, hospital_expire_flag):
        self.hadm_id = hadm_id
        self.timestamp = timestamp
        self.hospital_expire_flag = hospital_expire_flag


class DummyDiagnosis:
    def __init__(self, hadm_id, icd_code, icd_version):
        self.hadm_id = hadm_id
        self.icd_code = icd_code
        self.icd_version = icd_version


class DummyProcedure:
    def __init__(self, hadm_id, icd_code, icd_version):
        self.hadm_id = hadm_id
        self.icd_code = icd_code
        self.icd_version = icd_version


class DummyPrescription:
    def __init__(self, hadm_id, drug):
        self.hadm_id = hadm_id
        self.drug = drug


class DummyPatient:
    def __init__(self, patient_id, demographics, admissions,
                 diagnoses, procedures, prescriptions):
        self.patient_id = patient_id
        self._demographics = demographics
        self._admissions = admissions
        self._diagnoses = diagnoses
        self._procedures = procedures
        self._prescriptions = prescriptions

    def get_events(self, event_type, filters=None, **kwargs):
        if event_type == "patients":
            return self._demographics
        elif event_type == "admissions":
            return self._admissions
        elif event_type == "diagnoses_icd":
            events = self._diagnoses
        elif event_type == "procedures_icd":
            events = self._procedures
        elif event_type == "prescriptions":
            events = self._prescriptions
        else:
            return []

        # apply hadm_id filter if provided
        if filters:
            for field, op, value in filters:
                if op == "==":
                    events = [
                        e for e in events
                        if getattr(e, field, None) == value
                    ]
        return events


# -- helpers to build patients quickly --

def make_patient(patient_id, age, admissions_data,
                 anchor_year=2150,
                 anchor_year_group="2017 - 2019"):
    """Build a DummyPatient from a compact spec.

    admissions_data is a list of dicts like:
        {"hadm_id": "1", "year": 2150, "died": False,
         "dx": [("E10", "10")], "px": [("5A19", "10")],
         "drugs": ["Insulin"]}
    """
    demographics = [DummyDemographics(
        age, anchor_year, anchor_year_group
    )]
    admissions = []
    diagnoses = []
    procedures = []
    prescriptions = []

    for a in admissions_data:
        ts = datetime(a["year"], 3, 15)
        flag = "1" if a.get("died", False) else "0"
        admissions.append(
            DummyAdmission(a["hadm_id"], ts, flag)
        )
        for code, ver in a.get("dx", []):
            diagnoses.append(
                DummyDiagnosis(a["hadm_id"], code, ver)
            )
        for code, ver in a.get("px", []):
            procedures.append(
                DummyProcedure(a["hadm_id"], code, ver)
            )
        for drug in a.get("drugs", []):
            prescriptions.append(
                DummyPrescription(a["hadm_id"], drug)
            )

    return DummyPatient(
        patient_id, demographics, admissions,
        diagnoses, procedures, prescriptions,
    )


class TestTemporalMortalityMIMIC4(unittest.TestCase):

    def setUp(self):
        self.task = InHospitalMortalityTemporalMIMIC4()

    # -- schema --

    def test_task_name_and_schemas(self):
        t = InHospitalMortalityTemporalMIMIC4
        self.assertEqual(
            t.task_name, "InHospitalMortalityTemporalMIMIC4"
        )
        self.assertEqual(t.input_schema, {
            "conditions": "sequence",
            "procedures": "sequence",
            "drugs": "sequence",
        })
        self.assertEqual(t.output_schema, {"mortality": "binary"})

    # -- sample processing --

    def test_basic_sample_keys(self):
        patient = make_patient("p1", 45, [{
            "hadm_id": "100", "year": 2150, "died": False,
            "dx": [("E10", "10")],
            "px": [("5A19", "10")],
            "drugs": ["Metformin"],
        }])
        samples = self.task(patient)
        self.assertEqual(len(samples), 1)
        s = samples[0]
        for k in ["patient_id", "admission_id", "conditions",
                   "procedures", "drugs", "mortality",
                   "admission_year"]:
            self.assertIn(k, s)

    def test_multiple_admissions(self):
        patient = make_patient("p1", 50, [
            {"hadm_id": "1", "year": 2150, "died": False,
             "dx": [("E10", "10")], "px": [("5A19", "10")],
             "drugs": ["Insulin"]},
            {"hadm_id": "2", "year": 2152, "died": True,
             "dx": [("I50", "10")], "px": [("02HV", "10")],
             "drugs": ["Furosemide"]},
        ])
        samples = self.task(patient)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["admission_id"], "1")
        self.assertEqual(samples[1]["admission_id"], "2")

    # -- label generation --

    def test_mortality_label_died(self):
        patient = make_patient("p1", 60, [{
            "hadm_id": "1", "year": 2150, "died": True,
            "dx": [("E10", "10")],
            "px": [("5A19", "10")],
            "drugs": ["Insulin"],
        }])
        samples = self.task(patient)
        self.assertEqual(samples[0]["mortality"], 1)

    def test_mortality_label_survived(self):
        patient = make_patient("p1", 60, [{
            "hadm_id": "1", "year": 2150, "died": False,
            "dx": [("E10", "10")],
            "px": [("5A19", "10")],
            "drugs": ["Insulin"],
        }])
        samples = self.task(patient)
        self.assertEqual(samples[0]["mortality"], 0)

    # -- feature extraction --

    def test_admission_year_deshifted(self):
        # anchor_year=2150, group="2017 - 2019" -> midpoint 2018
        # shift = 2150 - 2018 = 132
        # admission in shifted year 2151 -> real year 2151 - 132 = 2019
        patient = make_patient("p1", 40, [{
            "hadm_id": "1", "year": 2151, "died": False,
            "dx": [("E10", "10")],
            "px": [("5A19", "10")],
            "drugs": ["Insulin"],
        }], anchor_year=2150, anchor_year_group="2017 - 2019")
        samples = self.task(patient)
        self.assertEqual(samples[0]["admission_year"], 2019)

    def test_icd_codes_have_version_prefix(self):
        patient = make_patient("p1", 40, [{
            "hadm_id": "1", "year": 2150, "died": False,
            "dx": [("E10", "10"), ("4019", "9")],
            "px": [("5A19", "10")],
            "drugs": ["Insulin"],
        }])
        samples = self.task(patient)
        self.assertIn("10_E10", samples[0]["conditions"])
        self.assertIn("9_4019", samples[0]["conditions"])

    def test_drugs_are_drug_names(self):
        patient = make_patient("p1", 40, [{
            "hadm_id": "1", "year": 2150, "died": False,
            "dx": [("E10", "10")],
            "px": [("5A19", "10")],
            "drugs": ["Insulin", "Metformin"],
        }])
        samples = self.task(patient)
        self.assertEqual(
            samples[0]["drugs"], ["Insulin", "Metformin"]
        )

    # -- edge cases --

    def test_minor_excluded(self):
        patient = make_patient("p1", 17, [{
            "hadm_id": "1", "year": 2150, "died": False,
            "dx": [("E10", "10")],
            "px": [("5A19", "10")],
            "drugs": ["Insulin"],
        }])
        samples = self.task(patient)
        self.assertEqual(samples, [])

    def test_age_18_included(self):
        patient = make_patient("p1", 18, [{
            "hadm_id": "1", "year": 2150, "died": False,
            "dx": [("E10", "10")],
            "px": [("5A19", "10")],
            "drugs": ["Insulin"],
        }])
        samples = self.task(patient)
        self.assertEqual(len(samples), 1)

    def test_no_diagnoses_skipped(self):
        patient = make_patient("p1", 40, [{
            "hadm_id": "1", "year": 2150, "died": False,
            "dx": [],
            "px": [("5A19", "10")],
            "drugs": ["Insulin"],
        }])
        samples = self.task(patient)
        self.assertEqual(samples, [])

    def test_no_procedures_skipped(self):
        patient = make_patient("p1", 40, [{
            "hadm_id": "1", "year": 2150, "died": False,
            "dx": [("E10", "10")],
            "px": [],
            "drugs": ["Insulin"],
        }])
        samples = self.task(patient)
        self.assertEqual(samples, [])

    def test_no_drugs_skipped(self):
        patient = make_patient("p1", 40, [{
            "hadm_id": "1", "year": 2150, "died": False,
            "dx": [("E10", "10")],
            "px": [("5A19", "10")],
            "drugs": [],
        }])
        samples = self.task(patient)
        self.assertEqual(samples, [])

    def test_no_admissions(self):
        demographics = [DummyDemographics(40)]
        patient = DummyPatient("p1", demographics, [], [], [], [])
        samples = self.task(patient)
        self.assertEqual(samples, [])

    def test_missing_expire_flag_skipped(self):
        # admission with hospital_expire_flag=None should be skipped
        demographics = [DummyDemographics(40)]
        adm = DummyAdmission("1", datetime(2150, 1, 1), None)
        dx = DummyDiagnosis("1", "E10", "10")
        px = DummyProcedure("1", "5A19", "10")
        rx = DummyPrescription("1", "Insulin")
        patient = DummyPatient(
            "p1", demographics, [adm], [dx], [px], [rx]
        )
        samples = self.task(patient)
        self.assertEqual(samples, [])

    def test_mixed_valid_and_invalid_admissions(self):
        # one admission with all features, one missing procedures
        patient = make_patient("p1", 50, [
            {"hadm_id": "1", "year": 2150, "died": False,
             "dx": [("E10", "10")], "px": [("5A19", "10")],
             "drugs": ["Insulin"]},
            {"hadm_id": "2", "year": 2151, "died": False,
             "dx": [("I10", "10")], "px": [],
             "drugs": ["Lisinopril"]},
        ])
        samples = self.task(patient)
        # only the first admission should produce a sample
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["admission_id"], "1")


if __name__ == "__main__":
    unittest.main()

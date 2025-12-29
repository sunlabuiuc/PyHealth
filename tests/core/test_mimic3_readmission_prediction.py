from datetime import timedelta
import tempfile
import unittest

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import ReadmissionPredictionMIMIC3


class TestReadmissionPredictionMIMIC3(unittest.TestCase):
    def setUp(self):
        """Seed dataset with neg and pos 15 day readmission examples (min required for sample generation)"""
        self.mock = MockMICIC3Dataset()
        patient = self.mock.add_patient()
        self.admission1 = self.mock.add_admission(patient, "2020-01-01 00:00:00", "2020-01-01 12:00:00")
        self.admission2 = self.mock.add_admission(patient, "2020-01-16 12:00:00", "2020-01-16 12:00:01") # Exactly 15 days later
        self.admission3 = self.mock.add_admission(patient, "2020-01-31 12:00:00", "2020-01-31 12:00:01") # 15 days later less 1 second

    def tearDown(self):
        self.mock.destroy()

    def test_patient_with_pos_and_neg_samples(self):
        dataset = self.mock.create()

        task = ReadmissionPredictionMIMIC3()
        samples = dataset.set_task(task)

        self.assertIn("task_name", vars(ReadmissionPredictionMIMIC3))
        self.assertIn("input_schema", vars(ReadmissionPredictionMIMIC3))
        self.assertIn("output_schema", vars(ReadmissionPredictionMIMIC3))

        self.assertEqual(task.window, timedelta(days=15))

        for sample in samples:
            self.assertIn("visit_id", sample)
            self.assertIn("patient_id", sample)
            self.assertIn("conditions", sample)
            self.assertIn("procedures", sample)
            self.assertIn("drugs", sample)
            self.assertIn("readmission", sample)

        self.assertEqual(len(samples), 2)

        neg_samples = [s for s in samples if s["readmission"] == 0]
        pos_samples = [s for s in samples if s["readmission"] == 1]

        self.assertEqual(len(neg_samples), 1)
        self.assertEqual(len(pos_samples), 1)

        self.assertEqual(neg_samples[0]["visit_id"], str(self.admission1))
        self.assertEqual(pos_samples[0]["visit_id"], str(self.admission2))

        self.assertTrue(all(s["visit_id"] != str(self.admission3) for s in samples)) # Patient's last admission not included

    def test_explicit_time_window(self):
        patient = self.mock.add_patient()
        admission1 = self.mock.add_admission(patient, "2020-01-01 00:00:00", "2020-01-01 12:00:00")
        admission2 = self.mock.add_admission(patient, "2020-01-06 12:00:00", "2020-01-06 12:00:01") # Exactly 5 days later
        admission3 = self.mock.add_admission(patient, "2020-01-11 12:00:00", "2020-01-11 12:00:01") # 5 days later less 1 second
        dataset = self.mock.create()

        samples = dataset.set_task(ReadmissionPredictionMIMIC3(timedelta(days=5)))

        visit1 = [s for s in samples if s["visit_id"] == str(admission1)]
        visit2 = [s for s in samples if s["visit_id"] == str(admission2)]

        self.assertEqual(len(visit1), 1)
        self.assertEqual(len(visit2), 1)

        self.assertEqual(visit1[0]["readmission"], 0)
        self.assertEqual(visit2[0]["readmission"], 1)

        self.assertTrue(all(s["visit_id"] != str(admission3) for s in samples)) # Patient's last admission not included

    def test_patient_with_only_one_visit_is_excluded(self):
        patient = self.mock.add_patient()
        admission = self.mock.add_admission(patient, "2020-01-01 00:00:00", "2020-01-01 12:00:00")
        dataset = self.mock.create()

        samples = dataset.set_task(ReadmissionPredictionMIMIC3())

        self.assertTrue(all(s["patient_id"] != str(patient) for s in samples))
        self.assertTrue(all(s["visit_id"] != str(admission) for s in samples))

    def test_admissions_without_diagnoses_are_excluded(self):
        patient1 = self.mock.add_patient()
        patient2 = self.mock.add_patient()
        admission1 = self.mock.add_admission(patient1, "2020-01-01 00:00:00", "2020-01-01 12:00:00")
        admission2 = self.mock.add_admission(patient1, "2020-01-01 00:00:00", "2020-01-01 12:00:00")
        admission3 = self.mock.add_admission(patient2, "2020-01-01 00:00:00", "2020-01-01 12:00:00", add_diagnosis=False)
        admission4 = self.mock.add_admission(patient2, "2020-01-01 00:00:00", "2020-01-01 12:00:00")
        dataset = self.mock.create()

        samples = dataset.set_task(ReadmissionPredictionMIMIC3())

        visit_ids = [int(s["visit_id"]) for s in samples]

        self.assertIn   (admission1, visit_ids)
        self.assertNotIn(admission2, visit_ids) # Patient's last admission should not be included
        self.assertNotIn(admission3, visit_ids)
        self.assertNotIn(admission4, visit_ids) # Patient's last admission should not be included

    def test_admissions_without_prescriptions_are_excluded(self):
        patient1 = self.mock.add_patient()
        patient2 = self.mock.add_patient()
        admission1 = self.mock.add_admission(patient1, "2020-01-01 00:00:00", "2020-01-01 12:00:00")
        admission2 = self.mock.add_admission(patient1, "2020-01-01 00:00:00", "2020-01-01 12:00:00")
        admission3 = self.mock.add_admission(patient2, "2020-01-01 00:00:00", "2020-01-01 12:00:00", add_prescription=False)
        admission4 = self.mock.add_admission(patient2, "2020-01-01 00:00:00", "2020-01-01 12:00:00")
        dataset = self.mock.create()

        samples = dataset.set_task(ReadmissionPredictionMIMIC3())

        visit_ids = [int(s["visit_id"]) for s in samples]

        self.assertIn   (admission1, visit_ids)
        self.assertNotIn(admission2, visit_ids) # Patient's last admission should not be included
        self.assertNotIn(admission3, visit_ids)
        self.assertNotIn(admission4, visit_ids) # Patient's last admission should not be included

    def test_admissions_without_procedures_are_excluded(self):
        patient1 = self.mock.add_patient()
        patient2 = self.mock.add_patient()
        admission1 = self.mock.add_admission(patient1, "2020-01-01 00:00:00", "2020-01-01 12:00:00")
        admission2 = self.mock.add_admission(patient1, "2020-01-01 00:00:00", "2020-01-01 12:00:00")
        admission3 = self.mock.add_admission(patient2, "2020-01-01 00:00:00", "2020-01-01 12:00:00", add_procedure=False)
        admission4 = self.mock.add_admission(patient2, "2020-01-01 00:00:00", "2020-01-01 12:00:00")
        dataset = self.mock.create()

        samples = dataset.set_task(ReadmissionPredictionMIMIC3())

        visit_ids = [int(s["visit_id"]) for s in samples]

        self.assertIn   (admission1, visit_ids)
        self.assertNotIn(admission2, visit_ids) # Patient's last admission should not be included
        self.assertNotIn(admission3, visit_ids)
        self.assertNotIn(admission4, visit_ids) # Patient's last admission should not be included

    def test_admissions_of_minors_are_excluded(self):
        patient = self.mock.add_patient(dob="2000-01-01 00:00:00")
        admission1 = self.mock.add_admission(patient, admittime="2017-12-31 23:59:59", dischtime="2018-01-01 00:00:00") # Admitted 1 second before turning 18
        admission2 = self.mock.add_admission(patient, admittime="2018-01-01 00:00:00", dischtime="2018-01-01 12:00:00") # Admitted at exactly 18
        admission3 = self.mock.add_admission(patient, admittime="2020-01-01 00:00:00", dischtime="2020-01-01 12:00:00")
        dataset = self.mock.create()

        task = ReadmissionPredictionMIMIC3()
        samples = dataset.set_task(task)

        self.assertTrue(task.exclude_minors)

        visit_ids = [int(s["visit_id"]) for s in samples]

        self.assertNotIn(admission1, visit_ids)
        self.assertIn   (admission2, visit_ids)
        self.assertNotIn(admission3, visit_ids) # Patient's last admission should not be included

    def test_exclude_minors_flag(self):
        patient = self.mock.add_patient(dob="2000-01-01 00:00:00")
        admission1 = self.mock.add_admission(patient, admittime="2017-12-31 23:59:59", dischtime="2018-01-01 00:00:00") # Admitted 1 second before turning 18
        admission2 = self.mock.add_admission(patient, admittime="2018-01-01 00:00:00", dischtime="2018-01-01 12:00:00") # Admitted at exactly 18
        admission3 = self.mock.add_admission(patient, admittime="2020-01-01 00:00:00", dischtime="2020-01-01 12:00:00")
        dataset = self.mock.create()

        samples = dataset.set_task(ReadmissionPredictionMIMIC3(exclude_minors=False))

        visit_ids = [int(s["visit_id"]) for s in samples]

        self.assertIn   (admission1, visit_ids)
        self.assertIn   (admission2, visit_ids)
        self.assertNotIn(admission3, visit_ids) # Patient's last admission should not be included


class MockMICIC3Dataset:
    def __init__(self):
        self.patients =      ["row_id,subject_id,gender,dob,dod,dod_hosp,dod_ssn,expire_flag"]
        self.admissions =    ["row_id,subject_id,hadm_id,admittime,dischtime,deathtime,admission_type,admission_location,discharge_location,insurance,language,religion,marital_status,ethnicity,edregtime,edouttime,diagnosis,hospital_expire_flag,has_chartevents_data"]
        self.icu_stays =     ["subject_id,intime,icustay_id,first_careunit,dbsource,last_careunit,outtime"]
        self.diagnoses =     ["row_id,subject_id,hadm_id,seq_num,icd9_code"]
        self.prescriptions = ["row_id,subject_id,hadm_id,icustay_id,startdate,enddate,drug_type,drug,drug_name_poe,drug_name_generic,formulary_drug_cd,gsn,ndc,prod_strength,dose_val_rx,dose_unit_rx,form_val_disp,form_unit_disp,route"]
        self.procedures =    ["row_id,subject_id,hadm_id,seq_num,icd9_code"]

        self.next_subject_id = 1
        self.next_hadm_id = 1
        self.next_diagnosis_id = 1
        self.next_prescription_id = 1
        self.next_procedure_id = 1

    def add_patient(self,dob: str="2000-01-01 00:00:00") -> int:
        subject_id = self.next_subject_id
        self.next_subject_id += 1
        self.patients.append(f"{subject_id},{subject_id},,{dob},,,,")
        return subject_id

    def add_admission(self,
                      subject_id: int,
                      admittime: str,
                      dischtime: str,
                      add_diagnosis: bool=True,
                      add_prescription: bool=True,
                      add_procedure: bool=True
                      ) -> int:
        hadm_id = self.next_hadm_id
        self.next_hadm_id += 1
        self.admissions.append(f"{hadm_id},{subject_id},{hadm_id},{admittime},{dischtime},,,,,,,,,,,,,,")
        if add_diagnosis: self.add_diagnosis(subject_id, hadm_id)
        if add_prescription: self.add_prescription(subject_id, hadm_id)
        if add_procedure: self.add_procedure(subject_id, hadm_id)
        return hadm_id

    def add_diagnosis(self, subject_id: int, hadm_id: int, seq_num: int=1, icd9_code: str="") -> int:
        row_id = self.next_diagnosis_id
        self.next_diagnosis_id += 1
        self.diagnoses.append(f"{row_id},{subject_id},{hadm_id},{seq_num},{icd9_code}")
        return row_id

    def add_prescription(self, subject_id: int, hadm_id: int) -> int:
        row_id = self.next_prescription_id
        self.next_prescription_id += 1
        self.prescriptions.append(f"{row_id},{subject_id},{hadm_id},,,,,,,,,,,,,,,,")
        return row_id

    def add_procedure(self, subject_id: int, hadm_id: int, seq_num: int=1, icd9_code: str="") -> int:
        row_id = self.next_procedure_id
        self.next_procedure_id += 1
        self.procedures.append(f"{row_id},{subject_id},{hadm_id},{seq_num},{icd9_code}")
        return row_id

    def create(self, tables: list=["diagnoses_icd", "prescriptions", "procedures_icd"]):
        self._dir = tempfile.TemporaryDirectory()

        files = {
            "PATIENTS.csv":       "\n".join(self.patients),
            "ADMISSIONS.csv":     "\n".join(self.admissions),
            "ICUSTAYS.csv":       "\n".join(self.icu_stays),
            "DIAGNOSES_ICD.csv":  "\n".join(self.diagnoses),
            "PRESCRIPTIONS.csv":  "\n".join(self.prescriptions),
            "PROCEDURES_ICD.csv": "\n".join(self.procedures),
        }

        for k, v in files.items():
            with open(f"{self._dir.name}/{k}", 'w') as f: f.write(v)

        return MIMIC3Dataset(root=self._dir.name, tables=tables)

    def destroy(self):
        self._dir.cleanup()


if __name__ == "__main__":
    unittest.main()

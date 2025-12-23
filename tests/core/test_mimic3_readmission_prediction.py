from datetime import timedelta
import os
import shutil
import unittest

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import ReadmissionPredictionMIMIC3

class TestReadmissionPredictionMIMIC3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists("test"):
            shutil.rmtree("test")
        os.makedirs("test")

        patients = [
            "row_id,subject_id,gender,dob,dod,dod_hosp,dod_ssn,expire_flag",
            "1,1,,2000-01-01 00:00:00,,,,",
            "2,2,,2000-01-01 00:00:00,,,,",
            "3,3,,2000-01-01 00:00:00,,,,",
            "4,4,,2000-01-01 00:00:00,,,,",
            "5,5,,2000-01-01 00:00:00,,,,",
            "6,6,,2000-01-01 00:00:00,,,,",
        ]
        with open("test/PATIENTS.csv", 'w') as f:
            f.write("\n".join(patients))

        admissions = [
            "row_id,subject_id,hadm_id,admittime,dischtime,deathtime,admission_type,admission_location,discharge_location,insurance,language,religion,marital_status,ethnicity,edregtime,edouttime,diagnosis,hospital_expire_flag,has_chartevents_data",
            "1,1,1,2020-01-01 00:00:00,2020-01-01 12:00:00,,,,,,,,,,,,,,",
            "2,2,2,2020-01-01 00:00:00,2020-01-01 12:00:00,,,,,,,,,,,,,,",
            "3,2,3,2020-01-31 11:00:00,2020-01-31 12:00:00,,,,,,,,,,,,,,",
            "4,3,4,2020-01-01 00:00:00,2020-01-01 12:00:00,,,,,,,,,,,,,,",
            "5,3,5,2020-01-31 12:00:00,2020-01-31 13:00:00,,,,,,,,,,,,,,",
            "6,4,6,2020-01-01 00:00:00,2020-01-01 12:00:00,,,,,,,,,,,,,,",
            "7,4,7,2020-02-01 00:00:00,2020-02-01 12:00:00,,,,,,,,,,,,,,",
            "8,4,8,2020-02-02 00:00:00,2020-02-02 12:00:00,,,,,,,,,,,,,,",
            "9,5,9,2020-01-01 00:00:00,2020-01-01 12:00:00,,,,,,,,,,,,,,",
            "10,5,10,2020-01-02 00:00:00,2020-01-02 12:00:00,,,,,,,,,,,,,,",
            "11,5,11,2020-01-03 00:00:00,2020-01-03 12:00:00,,,,,,,,,,,,,,",
            "12,5,12,2020-01-04 00:00:00,2020-01-04 12:00:00,,,,,,,,,,,,,,",
            "13,6,13,2017-12-31 23:59:59,2018-01-01 00:00:00,,,,,,,,,,,,,,",
            "14,6,14,2018-01-01 00:00:00,2018-01-01 12:00:00,,,,,,,,,,,,,,",
            "15,6,15,2020-01-01 00:00:00,2020-01-01 12:00:00,,,,,,,,,,,,,,",
        ]
        with open("test/ADMISSIONS.csv", 'w') as f:
            f.write("\n".join(admissions))

        icu_stays = [
            "subject_id,intime,icustay_id,first_careunit,dbsource,last_careunit,outtime",
        ]
        with open("test/ICUSTAYS.csv", 'w') as f:
            f.write("\n".join(icu_stays))

        diagnoses = [
            "row_id,subject_id,hadm_id,seq_num,icd9_code",
            "1,1,1,1,",
            "2,2,2,1,",
            "3,2,3,1,",
            "4,3,4,1,",
            "5,3,5,1,",
            "6,4,6,1,",
            "7,4,7,1,",
            "8,4,8,1,",
            "9,5,9,1,",
            "10,5,10,1,",
            "11,5,12,1,",
            "12,6,13,1,",
            "13,6,14,1,",
            "14,6,15,1,",
        ]
        with open("test/DIAGNOSES_ICD.csv", 'w') as f:
            f.write("\n".join(diagnoses))

        prescriptions = [
            "row_id,subject_id,hadm_id,icustay_id,startdate,enddate,drug_type,drug,drug_name_poe,drug_name_generic,formulary_drug_cd,gsn,ndc,prod_strength,dose_val_rx,dose_unit_rx,form_val_disp,form_unit_disp,route",
            "1,1,1,,,,,,,,,,,,,,,,",
            "2,2,2,,,,,,,,,,,,,,,,",
            "3,2,3,,,,,,,,,,,,,,,,",
            "4,3,4,,,,,,,,,,,,,,,,",
            "5,3,5,,,,,,,,,,,,,,,,",
            "6,4,6,,,,,,,,,,,,,,,,",
            "7,4,7,,,,,,,,,,,,,,,,",
            "8,4,8,,,,,,,,,,,,,,,,",
            "9,5,9,,,,,,,,,,,,,,,,",
            "10,5,11,,,,,,,,,,,,,,,,",
            "11,5,12,,,,,,,,,,,,,,,,",
            "12,6,13,,,,,,,,,,,,,,,,",
            "13,6,14,,,,,,,,,,,,,,,,",
            "14,6,15,,,,,,,,,,,,,,,,",
        ]
        with open("test/PRESCRIPTIONS.csv", 'w') as f:
            f.write("\n".join(prescriptions))

        procedures = [
            "row_id,subject_id,hadm_id,seq_num,icd9_code",
            "1,1,1,1,",
            "2,2,2,1,",
            "3,2,3,1,",
            "4,3,4,1,",
            "5,3,5,1,",
            "6,4,6,1,",
            "7,4,7,1,",
            "8,4,8,1,",
            "9,5,10,1,",
            "10,5,11,1,",
            "11,5,12,1,",
            "12,6,13,1,",
            "13,6,14,1,",
            "14,6,15,1,",
        ]
        with open("test/PROCEDURES_ICD.csv", 'w') as f:
            f.write("\n".join(procedures))

        dataset = MIMIC3Dataset(root="./test", tables=["diagnoses_icd", "prescriptions", "procedures_icd"])

        cls.samples = dataset.set_task(ReadmissionPredictionMIMIC3(timedelta(days=30)))

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("test"):
            shutil.rmtree("test")

    def test_task_schema(self):
        self.assertIn("task_name", vars(ReadmissionPredictionMIMIC3))
        self.assertIn("input_schema", vars(ReadmissionPredictionMIMIC3))
        self.assertIn("output_schema", vars(ReadmissionPredictionMIMIC3))

    def test_sample_schema(self):
        for sample in self.samples:
            self.assertIn("patient_id", sample)
            self.assertIn("admission_id", sample)
            self.assertIn("diagnoses", sample)
            self.assertIn("prescriptions", sample)
            self.assertIn("procedures", sample)
            self.assertIn("readmission", sample)

    def test_expected_num_samples(self):
        self.assertEqual(len(self.samples), 5)

    def test_patient_with_only_one_visit_is_excluded(self):
        self.assertTrue(all(sample["patient_id"] != '1' for sample in self.samples))
        self.assertTrue(all(sample["admission_id"] != '1' for sample in self.samples))

    def test_positive_sample(self):
        samples = [sample for sample in self.samples if sample["admission_id"] == '2']
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["readmission"], 1)

    def test_negative_sample(self):
        samples = [sample for sample in self.samples if sample["admission_id"] == '4']
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["readmission"], 0)

    def test_patient_with_positive_and_negative_samples(self):
        samples = [sample for sample in self.samples if sample["admission_id"] == '6']
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["readmission"], 0)
        samples = [sample for sample in self.samples if sample["admission_id"] == '7']
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["readmission"], 1)

    def test_last_admission_is_excluded_since_no_readmission_data(self):
        samples = [sample for sample in self.samples if sample["admission_id"] in ('3', '5', '8', '12', '15')]
        self.assertEqual(len(samples), 0)

    def test_admissions_without_diagnoses_are_excluded(self):
        self.assertTrue(all(sample["admission_id"] != '11' for sample in self.samples))

    def test_admissions_without_prescriptions_are_excluded(self):
        self.assertTrue(all(sample["admission_id"] != '10' for sample in self.samples))

    def test_admissions_without_procedures_are_excluded(self):
        self.assertTrue(all(sample["admission_id"] != '9' for sample in self.samples))

    def test_admissions_of_minors_are_excluded(self):
        self.assertTrue(all(sample["admission_id"] != '13' for sample in self.samples))
        samples = [sample for sample in self.samples if sample["admission_id"] == '14']
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["readmission"], 0)

if __name__ == "__main__":
    unittest.main()

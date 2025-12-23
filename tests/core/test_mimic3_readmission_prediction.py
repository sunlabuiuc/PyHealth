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
        self.assertEqual(len(self.samples), 2)

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
        pass

    def test_last_admission_is_excluded_since_no_readmission_data(self):
        samples = [sample for sample in self.samples if sample["admission_id"] == '3']
        self.assertEqual(len(samples), 0)
        samples = [sample for sample in self.samples if sample["admission_id"] == '5']
        self.assertEqual(len(samples), 0)

    def test_admissions_without_diagnoses_are_excluded(self):
        pass

    def test_admissions_without_prescriptions_are_excluded(self):
        pass

    def test_admissions_without_procedures_are_excluded(self):
        pass

    def test_admissions_of_minors_are_excluded(self):
        pass

if __name__ == "__main__":
    unittest.main()

import datetime
import os
import sys
import unittest

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.unittests.test_datasets.utils import EHRDatasetStatAssertion

current = os.path.dirname(os.path.realpath(__file__))
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(current)))
sys.path.append(repo_root)


# this test suite verifies the MIMIC3 dataset is consistently parsing the dataset.
# a dataset is qualified if it produces the correct statistics, and if a sample from the dataset
# matches the expected data.
# Synthetic_MIMIC-III dataset provided in the root is a dependancy to the expected values
# used for testing correctness
# like the MIMIC4 dataset, if this test suite fails, it may be due to a regression in the
# code, or due to the dataset at the root chaning.

def _parse_time(time_string: str) -> datetime.datetime:
    return datetime.datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S")

class TestsMimic3Dataset(unittest.TestCase):
    DATASET_NAME = "mimic3-demo"
    ROOT = "https://physionet.org/files/mimiciii-demo/1.4/"
    TABLES = ["DIAGNOSES_ICD", "PRESCRIPTIONS"]
    CODE_MAPPING = {"NDC": ("ATC", {"target_kwargs": {"level": 3}})}
    REFRESH_CACHE = True

    dataset = MIMIC3Dataset(
        dataset_name=DATASET_NAME,
        root=ROOT,
        tables=TABLES,
    )

    def setUp(self):
        pass

    # tests that a single event is correctly parsed
    def test_patient(self):
        selected_patient_id = "10035"

        expected_gender = "M"
        expected_ethnicity = "WHITE"
        expected_birth_datetime = datetime.datetime(2053, 4, 13, 0, 0)
        expected_death_datetime = datetime.datetime(2133, 3, 30, 0, 0)

        expected_visit_id = "110244"
        visit_filter = [("hadm_id", "==", expected_visit_id)]
        expected_num_visits = 1
        expected_encounter_time = datetime.datetime(2129, 3, 3, 16, 6)
        expected_discharge_time = datetime.datetime(2129, 3, 7, 18, 19)

        expected_num_events_diagnoses_icd = 4
        expected_num_events_prescriptions = 35


        actual_patient = self.dataset.get_patient(patient_id=selected_patient_id)
        patient_event = actual_patient.get_events("patients")[0]
        self.assertEqual(expected_gender, patient_event["gender"])
        self.assertEqual(expected_birth_datetime, _parse_time(patient_event["dob"]))
        self.assertEqual(expected_death_datetime, _parse_time(patient_event["dod"]))

        self.assertEqual(expected_num_visits, len(actual_patient.get_events("admissions")))

        visit =  actual_patient.get_events("admissions", filters=visit_filter)[0]
        self.assertEqual(expected_ethnicity, visit["ethnicity"])
        self.assertEqual(expected_encounter_time, visit["timestamp"])
        self.assertEqual(expected_discharge_time, _parse_time(visit["dischtime"]))

        self.assertEqual(
            expected_num_events_diagnoses_icd,
            len(actual_patient.get_events("diagnoses_icd", filters=visit_filter)),
        )
        self.assertEqual(
            expected_num_events_prescriptions,
            len(actual_patient.get_events("prescriptions", filters=visit_filter)),
        )

    def test_statistics(self):
        expected_tables = ["patients", "admissions", "icustays"] + self.TABLES
        self.assertCountEqual(expected_tables, self.dataset.tables)

        EHRDatasetStatAssertion(self.dataset, 0.01).assertEHRStats(
            expected_num_patients=100,
            expected_num_visits=129,
            expected_num_visits_per_patient=1.2900,
            expected_events_per_visit_per_table={
                "diagnoses_icd": 13.6512, "prescriptions": 85.230},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

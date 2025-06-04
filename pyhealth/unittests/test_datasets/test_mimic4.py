import datetime
import os
import sys
import unittest

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.unittests.test_datasets.utils import EHRDatasetStatAssertion

current = os.path.dirname(os.path.realpath(__file__))
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(current)))
sys.path.append(repo_root)


# this test suite verifies the MIMIC4 demo dataset is parsed correctly and produces
# the correct dataset for demoing. To qualify the units under test we check the dataset statistics,
# and a single sample from the dataset.


class TestMimic4Dataset(unittest.TestCase):

    DATASET_NAME = "mimic4-demo"
    ROOT = "https://physionet.org/files/mimic-iv-demo/2.2/"
    TABLES = ["diagnoses_icd", "procedures_icd", "labevents"]
    CODE_MAPPING = {}
    DEV = True  # not needed when using demo set since its 100 patients large
    REFRESH_CACHE = True

    dataset = MIMIC4Dataset(
        dataset_name=DATASET_NAME,
        ehr_root=ROOT,
        ehr_tables=TABLES,
        dev=DEV,
    )

    def setUp(self):
        pass

    # test the dataset integrity based on a single sample.
    def test_patient(self):
        expected_patient_id = "10000032"

        expected_visit_count = 4
        expected_visit_id = "22595853"

        expected_diagnoses_icd_event_count = 8
        expected_labevent_event_count = 57
        expected_encounter_time = datetime.datetime(2180, 5, 6, 22, 23)
        expected_discharge_time = datetime.datetime(2180, 5, 7, 17, 15)
        visit_filter = [("hadm_id", "==", expected_visit_id)]

        actual_patient = self.dataset.get_patient(expected_patient_id)
        self.assertEqual(
            expected_diagnoses_icd_event_count,
            len(actual_patient.get_events("diagnoses_icd", filters=visit_filter)),
        )
        self.assertEqual(
            expected_labevent_event_count,
            len(actual_patient.get_events("labevents", filters=visit_filter))
        )

        self.assertEqual(expected_visit_count, len(actual_patient.get_events("admissions")))
        visit = actual_patient.get_events("admissions", filters=visit_filter)[0]
        self.assertEqual(expected_visit_id, visit["hadm_id"])
        self.assertEqual(expected_encounter_time, visit["timestamp"])
        self.assertEqual(expected_discharge_time, datetime.datetime.strptime(visit["dischtime"], "%Y-%m-%d %H:%M:%S"))

    # checks data integrity based on statistics.
    def test_statistics(self):
        expected_tables = ["patients", "admissions", "icustays"] + self.TABLES

        self.assertCountEqual(expected_tables, self.dataset.tables)

        EHRDatasetStatAssertion(self.dataset, 0.01).assertEHRStats(
            expected_num_patients=100,
            expected_num_visits=275,
            expected_num_visits_per_patient=2.7500,
            # Values are updated due to new group by logic ignoring null columns
            expected_events_per_visit_per_table={
                "diagnoses_icd": 16.3855,
                "procedures_icd": 3.861,
                "labevents": 314.710317
            }
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

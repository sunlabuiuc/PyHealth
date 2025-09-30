import datetime
import unittest

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.unittests.test_datasets.utils import EHRDatasetStatAssertion

import os, sys

current = os.path.dirname(os.path.realpath(__file__))
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(current)))
sys.path.append(repo_root)


# this test suite verifies the MIMIC4 demo dataset is parsed correctly and produces
# the correct dataset for demoing. To qualify the units under test we check the dataset statistics,
# and a single sample from the dataset.


class TestMimic4Dataset(unittest.TestCase):

    DATASET_NAME = "mimic4-demo"
    ROOT = "https://storage.googleapis.com/pyhealth/mimiciv-demo/hosp/"
    TABLES = ["diagnoses_icd", "procedures_icd", "labevents"]
    CODE_MAPPING = {}
    DEV = True  # not needed when using demo set since its 100 patients large
    REFRESH_CACHE = True

    dataset = MIMIC4Dataset(
        dataset_name=DATASET_NAME,
        root=ROOT,
        tables=TABLES,
        code_mapping=CODE_MAPPING,
        dev=DEV,
        refresh_cache=REFRESH_CACHE,
    )

    def setUp(self):
        pass

    # test the dataset integrity based on a single sample.
    def test_patient(self):
        expected_patient_id = "10000032"

        expected_visit_count = 4
        expected_visit_to_id_keys = 3
        expected_visit_to_id_values = "29079034"

        expected_diagnoses_icd_event_count = 8
        expected_procedures_icd_event_count = 1
        expected_labevent_event_count = 57
        expected_event_length = 66
        expected_encounter_time = datetime.datetime(2180, 5, 6, 22, 23)
        expected_discharge_time = datetime.datetime(2180, 5, 7, 17, 15)

        actual_patient_id = list(self.dataset.patients.keys())[0]
        self.assertEqual(expected_patient_id, actual_patient_id)

        actual_visits = self.dataset.patients[actual_patient_id]
        self.assertEqual(expected_visit_count, len(actual_visits))
        self.assertEqual(
            expected_visit_to_id_keys, list(actual_visits.index_to_visit_id.keys())[-1]
        )
        self.assertEqual(
            expected_visit_to_id_values,
            list(actual_visits.index_to_visit_id.values())[-1],
        )

        visit = actual_visits[0]
        self.assertEqual(
            expected_diagnoses_icd_event_count,
            len(visit.event_list_dict["diagnoses_icd"]),
        )
        self.assertEqual(
            expected_procedures_icd_event_count,
            len(visit.event_list_dict["procedures_icd"]),
        )
        self.assertEqual(
            expected_labevent_event_count, len(visit.event_list_dict["labevents"])
        )
        self.assertEqual(expected_event_length, visit.num_events)
        self.assertEqual(expected_encounter_time, visit.encounter_time)
        self.assertEqual(expected_discharge_time, visit.discharge_time)

    # checks data integrity based on statistics.
    def test_statistics(self):

        # self.dataset.stat()

        self.assertEqual(sorted(self.TABLES), sorted(self.dataset.available_tables))

        EHRDatasetStatAssertion(self.dataset, 0.01).assertEHRStats(
            expected_num_patients=100,
            expected_num_visits=275,
            expected_num_visits_per_patient=2.7500,
            expected_events_per_visit_per_table=[16.3855, 2.6255, 288.3891],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

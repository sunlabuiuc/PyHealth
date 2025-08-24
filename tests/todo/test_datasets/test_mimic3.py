import datetime
import unittest

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.unittests.test_datasets.utils import EHRDatasetStatAssertion
import os, sys

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


class TestsMimic3Dataset(unittest.TestCase):
    DATASET_NAME = "mimic3-demo"
    ROOT = "https://storage.googleapis.com/pyhealth/mimiciii-demo/1.4/"
    TABLES = ["DIAGNOSES_ICD", "PRESCRIPTIONS"]
    CODE_MAPPING = {"NDC": ("ATC", {"target_kwargs": {"level": 3}})}
    REFRESH_CACHE = True

    dataset = MIMIC3Dataset(
        dataset_name=DATASET_NAME,
        root=ROOT,
        tables=TABLES,
        code_mapping=CODE_MAPPING,
        refresh_cache=REFRESH_CACHE,
    )

    def setUp(self):
        pass

    # tests that a single event is correctly parsed
    def test_patient(self):

        selected_patient_id = "10035"
        selected_visit_id = "110244"

        expected_geneder = "M"
        expected_ethnicity = "WHITE"
        expected_birth_datetime = datetime.datetime(2053, 4, 13, 0, 0)
        expected_death_datetime = None

        expected_visit_to_id_keys = [0]
        expected_visit_to_id_values = [selected_visit_id]

        expected_visit_id = "110244"
        expected_num_visits = 1
        expected_encounter_time = datetime.datetime(2129, 3, 3, 16, 6)
        expected_discharge_time = datetime.datetime(2129, 3, 7, 18, 19)

        expected_num_events_in_visit = 17
        expected_num_event_types = 2
        expected_num_events_diagnoses_icd = 4
        expected_num_events_prescriptions = 13

        self.assertTrue(selected_patient_id in self.dataset.patients)

        actual_patient = self.dataset.patients[selected_patient_id]

        self.assertEqual(expected_geneder, actual_patient.gender)
        self.assertEqual(expected_ethnicity, actual_patient.ethnicity)
        self.assertEqual(expected_birth_datetime, actual_patient.birth_datetime)
        self.assertEqual(expected_death_datetime, actual_patient.death_datetime)

        self.assertEqual(expected_num_visits, len(actual_patient.visits))

        actual_visit_id = list(actual_patient.visits.keys())[0]

        self.assertEqual(
            expected_visit_to_id_keys, list(actual_patient.index_to_visit_id.keys())
        )
        self.assertEqual(
            expected_visit_to_id_values, list(actual_patient.index_to_visit_id.values())
        )

        self.assertEqual(expected_visit_id, actual_visit_id)
        actual_visit = actual_patient.visits[actual_visit_id]
        self.assertEqual(expected_encounter_time, actual_visit.encounter_time)
        self.assertEqual(expected_discharge_time, actual_visit.discharge_time)
        self.assertEqual(expected_num_events_in_visit, actual_visit.num_events)
        self.assertEqual(
            expected_num_event_types, len(actual_visit.event_list_dict.keys())
        )
        self.assertEqual(
            expected_num_events_diagnoses_icd,
            len(actual_visit.event_list_dict["DIAGNOSES_ICD"]),
        )
        self.assertEqual(
            expected_num_events_prescriptions,
            len(actual_visit.event_list_dict["PRESCRIPTIONS"]),
        )

    def test_statistics(self):
        # self.dataset.stat()

        self.assertEqual(sorted(self.TABLES), sorted(self.dataset.available_tables))

        EHRDatasetStatAssertion(self.dataset, 0.01).assertEHRStats(
            expected_num_patients=100,
            expected_num_visits=129,
            expected_num_visits_per_patient=1.2900,
            expected_events_per_visit_per_table=[13.6512, 56.7597],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

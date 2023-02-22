import datetime
import unittest
import os, sys

from pyhealth.utils import record_dataset_cache
current = os.path.dirname(os.path.realpath(__file__))
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(current)))
sys.path.append(repo_root)

from pyhealth.datasets import MIMIC4Dataset

# this test suite verifies the MIMIC4 demo dataset is parsed correctly and produces
# the correct dataset for demoing. To qualify the units under test we check the dataset statistics,
# and a single sample from the dataset.


class TestMimic4(unittest.TestCase):

    # to test the file this path needs to be updated
    ROOT = "https://storage.googleapis.com/pyhealth/mimiciv-demo/hosp/"
    TABLES = ["diagnoses_icd", "procedures_icd", "labevents"]
    CODE_MAPPING = {}
    DEV = True  # not needed when using demo set since its 100 patients large

    dataset = MIMIC4Dataset(
        root=ROOT,
        tables=TABLES,
        code_mapping=CODE_MAPPING,
        dev=DEV,
        refresh_cache=False,
    )
    
    record_dataset_cache(repo_root, dataset.filepath)

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

        print(self.dataset.info())
        print(self.dataset.stat())

        expected_dev = self.DEV
        expected_num_patients = 100  # for the reduced dataset at
        expected_num_visits = 275
        expected_num_visits_per_patient = 2.7500
        expected_num_events_per_table = [16.3855, 2.6255, 288.3891]

        self.assertEqual(sorted(self.TABLES), sorted(self.dataset.available_tables))

        self.assertEqual(expected_dev, self.dataset.dev)
        self.assertEqual(expected_num_patients, len(self.dataset.patients))

        actual_visits = [len(patient) for patient in self.dataset.patients.values()]
        self.assertEqual(expected_num_visits, sum(actual_visits))

        actual_visits_per_patient = sum(actual_visits) / len(actual_visits)
        self.assertAlmostEqual(
            expected_num_visits_per_patient, actual_visits_per_patient, places=2
        )

        for expected_value, table in zip(
            expected_num_events_per_table, self.dataset.tables
        ):
            actual_num_events = [
                len(v.get_event_list(table))
                for p in self.dataset.patients.values()
                for v in p
            ]

            actual_value_per_event_type = sum(actual_num_events) / len(
                actual_num_events
            )
            self.assertAlmostEqual(
                expected_value, actual_value_per_event_type, places=2
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

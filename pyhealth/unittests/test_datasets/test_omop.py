import datetime
from sqlite3 import Timestamp
import unittest
from pyhealth.data.data import Event
import collections

from pyhealth.datasets import OMOPDataset
from pyhealth.unittests.test_datasets.utils import EHRDatasetStatAssertion


class TestOMOPDataset(unittest.TestCase):
    DATASET_NAME = "omop-demo"
    ROOT = "https://storage.googleapis.com/pyhealth/synpuf1k_omop_cdm_5.2.2/"
    TABLES = [
        "condition_occurrence",
        "procedure_occurrence",
        "drug_exposure",
        "measurement",
    ]
    CODE_MAPPING = {}
    DEV = True  # not needed when using demo set since its 100 patients large
    REFRESH_CACHE = True

    dataset = OMOPDataset(
        dataset_name=DATASET_NAME,
        root=ROOT,
        tables=TABLES,
        code_mapping=CODE_MAPPING,
        dev=DEV,
        refresh_cache=REFRESH_CACHE,
    )

    def setUp(self):
        pass

    def test_patient(self):

        print(self.dataset)

        expected_len_patients = 1000
        selected_patient_id = "100"

        expected_available_tables = [
            "drug_exposure",
            "procedure_occurrence",
            "measurement",
        ]
        expected_birth_datetime = datetime.datetime(1934, 4, 1, 0, 0)
        expected_death_datetime = None
        expected_ethnicity = 8527
        expected_gender = 8532

        expected_len_visit_ids = 47

        selected_visit_index = 16
        expected_visit_id = "5393"

        expected_available_tables_for_visit = [
            "procedure_occurrence",
            "drug_exposure",
            "measurement",
        ]

        expected_visit_discharge_status = 0
        expected_visit_discharge_time = datetime.datetime(2008, 8, 8, 0, 0)
        expected_visit_encounter_time = datetime.datetime(2008, 8, 8, 0, 0)
        expected_visit_len_event_list_dict = 3

        Expected_data = collections.namedtuple(
            "Expected_data_attributes",
            [
                "code",
                "timestamp",
                "vocabulary",
            ],
        )

        expected_event_list_dict = [
            (
                "procedure_occurrence",
                4,
                0,
                Expected_data(
                    "2005280",
                    datetime.datetime(2008, 8, 8, 0, 0),
                    "PROCEDURE_CONCEPT_ID",
                ),
            ),
            (
                "drug_exposure",
                1,
                0,
                Expected_data(
                    "2213483", datetime.datetime(2008, 8, 8, 0, 0), "DRUG_CONCEPT_ID"
                ),
            ),
            (
                "measurement",
                1,
                0,
                Expected_data("2212095", None, "MEASUREMENT_CONCEPT_ID"),
            ),
        ]

        self.assertEqual(expected_len_patients, len(self.dataset.patients))

        actual_patient = self.dataset.patients[selected_patient_id]

        self.assertEqual(
            sorted(expected_available_tables), sorted(actual_patient.available_tables)
        )
        self.assertEqual(expected_birth_datetime, actual_patient.birth_datetime)
        self.assertEqual(expected_death_datetime, actual_patient.death_datetime)
        self.assertEqual(expected_ethnicity, actual_patient.ethnicity)
        self.assertEqual(expected_gender, actual_patient.gender)

        self.assertEqual(expected_len_visit_ids, len(actual_patient.index_to_visit_id))

        self.assertEqual(
            expected_visit_id, actual_patient.index_to_visit_id[selected_visit_index]
        )

        actual_visit_id = actual_patient.index_to_visit_id[selected_visit_index]

        actual_visit = actual_patient.visits[actual_visit_id]

        self.assertEqual(
            expected_available_tables_for_visit, actual_visit.available_tables
        )
        self.assertEqual(expected_visit_discharge_status, actual_visit.discharge_status)
        self.assertEqual(expected_visit_discharge_time, actual_visit.discharge_time)
        self.assertEqual(expected_visit_encounter_time, actual_visit.encounter_time)
        self.assertEqual(
            expected_visit_len_event_list_dict, len(actual_visit.event_list_dict)
        )

        for (
            expected_event_type,
            expected_event_list_len,
            selected_event_index,
            expected_event_data,
        ) in expected_event_list_dict:

            self.assertTrue(expected_event_type in actual_visit.available_tables)

            actual_visit_table = actual_visit.event_list_dict[expected_event_type]

            self.assertEqual(expected_event_list_len, len(actual_visit_table))

            actual_event = actual_visit_table[selected_event_index]

            self.assertEqual(expected_event_data.code, actual_event.code)
            self.assertEqual(expected_event_data.timestamp, actual_event.timestamp)
            self.assertEqual(expected_event_data.vocabulary, actual_event.vocabulary)

    def test_statistics(self):

        EHRDatasetStatAssertion(self.dataset, 0.01).assertEHRStats(
            expected_num_patients=1000,
            expected_num_visits=55261,
            expected_num_visits_per_patient=55.2610,
            expected_events_per_visit_per_table=[0.0000, 2.4886, 0.1387, 0.6253],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

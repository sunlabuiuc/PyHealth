import datetime
from sqlite3 import Timestamp
import unittest
from pyhealth.data.data import Event
import pandas

from pyhealth.datasets import eICUDataset
from pyhealth.unittests.test_datasets.utils import EHRDatasetStatAssertion


class TesteICUDataset(unittest.TestCase):

    # to test the file this path needs to be updated
    DATASET_NAME = "eICU-demo"
    ROOT = "https://storage.googleapis.com/pyhealth/eicu-demo/"
    TABLES = ["diagnosis", "medication", "lab", "treatment", "physicalExam"]
    CODE_MAPPING = {}
    DEV = True  # not needed when using demo set since its 100 patients large
    REFRESH_CACHE = True

    dataset = eICUDataset(
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
        # given parametes:
        selected_patient_id = "002-10009+193705"
        selected_visit_index = 0
        # selected indeces for events defined in `expected_event_data`

        # expect:
        # patient data
        expected_birth_datetime = pandas.Timestamp("1938-01-01 00:00:00")
        expected_death_datetime = None
        expected_ethnicity = "Caucasian"
        expected_gender = "Female"

        # visit data
        expected_visit_len = 1
        expected_visit_id = "224606"
        expected_visit_discharge_status = "Alive"
        expected_discharge_time = datetime.datetime(2014, 1, 4, 0, 45)
        expected_encounter_time = datetime.datetime(2014, 1, 1, 2, 59)

        # visit attribute dict data
        expected_visit_attr_dict_len = 2
        expected_visit_hopital_id = 71
        expected_visit_region = "Midwest"

        # event level data
        expected_event_count = 319

        # during a specified visit assert the event data is correct. Event data is parametrized by tables
        # schema:
        #   event type (from one of the requested tables)
        #       'length': number of events for that event type
        #       'events':
        #           tuple of index of the event in the event array, pyhealth.Event object with hardcoded relevant fields for event at the index
        expected_event_data = {
            "diagnosis": {
                "length": 8,
                "events": [
                    (
                        0,
                        Event(
                            code="567.9",
                            timestamp=pandas.Timestamp("2014-01-01 03:36:00"),
                            vocabulary="ICD9CM",
                        ),
                    ),
                    (
                        1,
                        Event(
                            code="K65.0",
                            timestamp=pandas.Timestamp("2014-01-01 03:36:00"),
                            vocabulary="ICD10CM",
                        ),
                    ),
                ],
            },
            "medication": {
                "length": 38,
                "events": [
                    (
                        0,
                        Event(
                            code="MORPHINE INJ",
                            timestamp=pandas.Timestamp("2013-12-31 21:09:00"),
                            vocabulary="eICU_DRUGNAME",
                        ),
                    ),
                    (
                        5,
                        Event(
                            code="CIPROFLOXACIN IN D5W 400 MG/200ML IV SOLN",
                            timestamp=pandas.Timestamp("2013-12-31 22:43:00"),
                            vocabulary="eICU_DRUGNAME",
                        ),
                    ),
                ],
            },
            "lab": {
                "length": 251,
                "events": [
                    (
                        0,
                        Event(
                            code="sodium",
                            timestamp=pandas.Timestamp("2013-12-31 21:04:00"),
                            vocabulary="eICU_LABNAME",
                        ),
                    ),
                    (
                        2,
                        Event(
                            code="BUN",
                            timestamp=pandas.Timestamp("2013-12-31 21:04:00"),
                            vocabulary="eICU_LABNAME",
                        ),
                    ),
                ],
            },
            "physicalExam": {
                "length": 22,
                "events": [
                    (
                        0,
                        Event(
                            code="notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/Score/scored",
                            timestamp=pandas.Timestamp("2014-01-01 03:05:00"),
                            vocabulary="eICU_PHYSICALEXAMPATH",
                        ),
                    ),
                    (
                        1,
                        Event(
                            code="notes/Progress Notes/Physical Exam/Physical Exam Obtain Options/Performed - Structured",
                            timestamp=pandas.Timestamp("2014-01-01 03:05:00"),
                            vocabulary="eICU_PHYSICALEXAMPATH",
                        ),
                    ),
                ],
            },
        }

        # patient level information
        actual_patient = self.dataset.patients[selected_patient_id]
        self.assertEqual(expected_visit_len, len(actual_patient.visits))
        self.assertEqual(expected_birth_datetime, actual_patient.birth_datetime)
        self.assertEqual(expected_death_datetime, actual_patient.death_datetime)
        self.assertEqual(expected_ethnicity, actual_patient.ethnicity)
        self.assertEqual(expected_gender, actual_patient.gender)

        # visit level information
        actual_visit_id = actual_patient.index_to_visit_id[selected_visit_index]
        self.assertEqual(expected_visit_id, actual_visit_id)

        actual_visit = actual_patient.visits[actual_visit_id]
        self.assertEqual(expected_event_count, actual_visit.num_events)
        self.assertEqual(expected_visit_discharge_status, actual_visit.discharge_status)
        self.assertEqual(expected_discharge_time, actual_visit.discharge_time)
        self.assertEqual(expected_encounter_time, actual_visit.encounter_time)

        # visit attributes
        actual_visit_attributes = actual_visit.attr_dict
        self.assertEqual(expected_visit_attr_dict_len, len(actual_visit_attributes))
        self.assertEqual(
            expected_visit_hopital_id, actual_visit_attributes["hospital_id"]
        )
        self.assertEqual(expected_visit_region, actual_visit_attributes["region"])

        # event level information
        actual_event_list_dict = actual_visit.event_list_dict
        for event_key in expected_event_data:
            actual_event_array = actual_event_list_dict[event_key]
            expected_event = expected_event_data[event_key]

            self.assertEqual(
                expected_event["length"],
                len(actual_event_array),
                f"incorrect num events for'{event_key}'",
            )
            for selected_index, expected_pyhealth_Event in expected_event["events"]:
                error_message = f"incorrect event code on '{event_key}' event, selected index: {selected_index}"

                actual_event = actual_event_array[selected_index]
                self.assertEqual(
                    expected_pyhealth_Event.code, actual_event.code, error_message
                )
                self.assertEqual(
                    expected_pyhealth_Event.timestamp,
                    actual_event.timestamp,
                    error_message,
                )
                self.assertEqual(
                    expected_pyhealth_Event.vocabulary,
                    actual_event.vocabulary,
                    error_message,
                )

    def test_statistics(self):
        # self.dataset.stat()

        self.assertEqual(sorted(self.TABLES), sorted(self.dataset.available_tables))

        EHRDatasetStatAssertion(self.dataset, 0.01).assertEHRStats(
            expected_num_patients=2174,
            expected_num_visits=2520,
            expected_num_visits_per_patient=1.1592,
            expected_events_per_visit_per_table=[
                16.7202,
                17.8345,
                172.4841,
                15.1944,
                33.3563,
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

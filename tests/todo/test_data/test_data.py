from datetime import datetime
import unittest

from pyhealth.data import Event, Patient


class TestEvent(unittest.TestCase):
    def setUp(self):
        self.event = Event(
            type="DIAGNOSES_ICD",
            timestamp=datetime(2012, 1, 1, 0, 0),
            attr_dict={"code": "428.0", "vocabulary": "ICD9CM"}
        )

    def test_type(self):
        self.assertIsInstance(self.event.type, str)
        self.assertIsInstance(self.event.timestamp, datetime)

    def test_attr(self):
        self.assertEqual(self.event.type, "DIAGNOSES_ICD")
        self.assertEqual(self.event.timestamp, datetime(2012, 1, 1, 0, 0))
        attr_dict = self.event.attr_dict
        self.assertEqual(attr_dict["code"], "428.0")
        self.assertEqual(attr_dict["vocabulary"], "ICD9CM")

    def test_repr_and_str(self):
        print(repr(self.event))
        print(str(self.event))


class TestPatient(unittest.TestCase):
    def setUp(self):
        self.event1 = Event(type="diagnosis", timestamp=datetime(2023, 5, 17))
        self.event2 = Event(type="prescription", timestamp=datetime(2023, 5, 18))
        self.event3 = Event(type="lab_test", timestamp=None)
        self.patient = Patient(patient_id="12345",
                               attr_dict={"name": "John Doe", "age": 45})

    def test_attr(self):
        self.assertEqual(self.patient.patient_id, "12345")
        self.assertEqual(self.patient.attr_dict["name"], "John Doe")
        self.assertEqual(self.patient.attr_dict["age"], 45)

    def test_methods(self):
        self.patient.add_event(self.event1)
        self.patient.add_event(self.event2)
        self.patient.add_event(self.event3)
        self.assertEqual(len(self.patient.events), 3)
        self.assertEqual(self.patient.events[0], self.event1)
        self.assertEqual(self.patient.events[1], self.event2)
        self.assertEqual(self.patient.events[2], self.event3)

    def test_repr_and_str(self):
        self.patient.add_event(self.event1)
        self.patient.add_event(self.event2)
        self.patient.add_event(self.event3)
        print(repr(self.patient))
        print(str(self.patient))


if __name__ == "__main__":
    unittest.main()

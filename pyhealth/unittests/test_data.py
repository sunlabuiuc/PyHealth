import unittest
import datetime
from pyhealth.data import Event, Visit, Patient


class TestEvent(unittest.TestCase):
    def setUp(self):
        self.event = Event(
            code="428.0",
            table="DIAGNOSES_ICD",
            vocabulary="ICD9CM",
            visit_id="v001",
            patient_id="p001",
            timestamp=datetime.datetime(2012, 1, 1, 0, 0),
            add_attr1="add_attr1",
            add_attr2={"key": "add_attr2"},
        )

    def test_type(self):
        self.assertIsInstance(self.event.timestamp, datetime.datetime)

    def test_attr(self):
        self.assertEqual(self.event.code, "428.0")
        self.assertEqual(self.event.table, "DIAGNOSES_ICD")
        self.assertEqual(self.event.vocabulary, "ICD9CM")
        self.assertEqual(self.event.visit_id, "v001")
        self.assertEqual(self.event.patient_id, "p001")
        self.assertEqual(self.event.timestamp, datetime.datetime(2012, 1, 1, 0, 0))

        attr_dict = self.event.attr_dict
        self.assertEqual(attr_dict["add_attr1"], "add_attr1")
        self.assertEqual(attr_dict["add_attr2"], {"key": "add_attr2"})


class TestVisit(unittest.TestCase):
    def setUp(self):
        self.event1 = Event(
            code="00069153041",
            table="PRESCRIPTIONS",
            vocabulary="NDC",
            visit_id="v001",
            patient_id="p001",
            dosage="250mg",
        )

        self.event2 = Event(
            code="00069153042",
            table="PRESCRIPTIONS",
            vocabulary="NDC",
            visit_id="v001",
            patient_id="p001",
            method="tablet",
        )

        self.visit = Visit(
            visit_id="v001",
            patient_id="p001",
            encounter_time=datetime.datetime(2012, 1, 1, 0, 0),
            discharge_time=datetime.datetime(2012, 1, 8, 0, 0),
            discharge_status="expired",
        )

    def test_methods(self):
        # add the first event
        self.visit.add_event(self.event1)
        self.assertTrue("PRESCRIPTIONS" in self.visit.available_tables)
        self.assertEqual(self.visit.num_events, 1)
        self.assertEqual(self.visit.get_event_list("PRESCRIPTIONS"), [self.event1])
        self.assertEqual(self.visit.get_code_list("PRESCRIPTIONS"), [self.event1.code])

        # add the second event
        self.visit.add_event(self.event2)
        self.assertEqual(self.visit.num_events, 2)
        self.assertEqual(
            self.visit.get_event_list("PRESCRIPTIONS"), [self.event1, self.event2]
        )
        self.assertEqual(
            self.visit.get_code_list("PRESCRIPTIONS"),
            [self.event1.code, self.event2.code],
        )

    def test_attr(self):
        self.visit.add_event(self.event1)
        self.visit.add_event(self.event2)
        self.assertEqual(self.visit.visit_id, "v001")
        self.assertEqual(self.visit.patient_id, "p001")
        self.assertEqual(self.visit.num_events, 2)
        self.assertEqual(self.visit.encounter_time, datetime.datetime(2012, 1, 1, 0, 0))
        self.assertEqual(self.visit.discharge_time, datetime.datetime(2012, 1, 8, 0, 0))
        self.assertEqual(self.visit.discharge_status, "expired")


class TestPatient(unittest.TestCase):
    def setUp(self):
        self.event = Event(
            code="00069153041",
            table="PRESCRIPTIONS",
            vocabulary="NDC",
            visit_id="v001",
            patient_id="p001",
            dosage="250mg",
        )

        self.visit = Visit(
            visit_id="v001",
            patient_id="p001",
            encounter_time=datetime.datetime(2012, 1, 1, 0, 0),
            discharge_time=datetime.datetime(2012, 1, 8, 0, 0),
            discharge_status="expired",
        )

        self.patient = Patient(
            patient_id="p001",
        )

    def test_methods(self):
        self.patient.add_visit(self.visit)
        self.patient.add_event(self.event)
        self.assertTrue("PRESCRIPTIONS" in self.patient.available_tables)
        self.assertEqual(self.patient.get_visit_by_id("v001"), self.visit)
        self.assertEqual(self.patient.get_visit_by_index(0), self.visit)
        self.assertEqual(self.patient.visits["v001"], self.visit)

    def test_attr(self):
        self.patient.add_visit(self.visit)
        self.patient.add_event(self.event)
        self.assertEqual(self.patient.patient_id, "p001")
        self.assertEqual(self.patient.get_visit_by_index(0).patient_id, "p001")
        self.assertEqual(self.patient.visits["v001"].patient_id, "p001")


if __name__ == "__main__":
    unittest.main()

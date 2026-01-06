import unittest
from datetime import datetime
from unittest.mock import MagicMock
from pyhealth.tasks.patient_linkage_mimic3 import PatientLinkageMIMIC3Task

class TestPatientLinkageMIMIC3Task(unittest.TestCase):
    def setUp(self):
        self.task = PatientLinkageMIMIC3Task()

    def test_call_multiple_admissions(self):
        # Mock patient
        patient = MagicMock()
        patient.patient_id = "123"

        # Mock admissions
        # Adm 1: 2010-01-01, hadm_id=1, codes=["A", "B"]
        # Adm 2: 2011-01-01, hadm_id=2, codes=["C"]
        # Adm 3: 2012-01-01, hadm_id=3, codes=["D", "E"] (Query)
        
        adm1 = MagicMock()
        adm1.timestamp = datetime(2010, 1, 1)
        adm1.attr_dict = {"hadm_id": "1"}

        adm2 = MagicMock()
        adm2.timestamp = datetime(2011, 1, 1)
        adm2.attr_dict = {"hadm_id": "2"}

        adm3 = MagicMock()
        adm3.timestamp = datetime(2012, 1, 1)
        adm3.attr_dict = {"hadm_id": "3"}

        # Mock demographics
        demo = MagicMock()
        demo.attr_dict = {"dob": datetime(1980, 1, 1)}

        # Mock diagnosis events
        # Adm 1
        d1_1 = MagicMock(); d1_1.attr_dict = {"hadm_id": "1", "icd9_code": "A"}
        d1_2 = MagicMock(); d1_2.attr_dict = {"hadm_id": "1", "icd9_code": "B"}
        # Adm 2
        d2_1 = MagicMock(); d2_1.attr_dict = {"hadm_id": "2", "icd9_code": "C"}
        # Adm 3
        d3_1 = MagicMock(); d3_1.attr_dict = {"hadm_id": "3", "icd9_code": "D"}
        d3_2 = MagicMock(); d3_2.attr_dict = {"hadm_id": "3", "icd9_code": "E"}

        # Configure patient.get_events
        def get_events_side_effect(event_type):
            if event_type == "admissions":
                return [adm1, adm2, adm3]
            elif event_type == "patients":
                return [demo]
            elif event_type == "diagnoses_icd":
                return [d1_1, d1_2, d2_1, d3_1, d3_2]
            return []
        
        patient.get_events.side_effect = get_events_side_effect

        results = self.task(patient)
        self.assertEqual(len(results), 1)
        sample = results[0]

        # Check Schema Keys
        expected_keys = {
            "patient_id", "q_visit_id", "q_conditions", "q_timestamp",
            "d_visit_ids", "d_conditions", "d_timestamp", "time_gap_days"
        }
        self.assertTrue(expected_keys.issubset(sample.keys()))

        # Check Query (Adm 3)
        self.assertEqual(sample["q_visit_id"], "3")
        self.assertEqual(sample["q_conditions"], ["", "D", "E"])
        self.assertEqual(sample["q_timestamp"], datetime(2012, 1, 1))

        # Check Database (Adm 1 + Adm 2)
        # Expected order (sorted by time): Adm 1 then Adm 2
        # Adm 1 codes: A, B
        # Adm 2 codes: C
        # Concatenated: A, B, [SEP], C
        # Plus leading ""
        # Note: dict order might vary if not sorted in task, but task sorts admissions. 
        # Inside admission, codes are list append order.
        
        # d_conditions should be ["", "A", "B", "[SEP]", "C"] 
        # OR ["", "C", "[SEP]", "A", "B"] depending on if it processes most recent first or chronological.
        # Task code: `d_visits = admissions[:-1]` where admissions is sorted chronological.
        # So d_visits = [adm1, adm2]
        # Then `for d_visit in d_visits:` -> processed in chronological order.
        expected_d_conditions = ["", "A", "B", "[SEP]", "C"]
        self.assertEqual(sample["d_conditions"], expected_d_conditions)
        
        self.assertEqual(sample["d_visit_ids"], "1|2")
        self.assertEqual(sample["d_timestamp"], datetime(2011, 1, 1)) # Most recent db visit
        
        # Check Time Gap
        # 2012-01-01 - 2011-01-01 = 365 days
        self.assertEqual(sample["time_gap_days"], 365)


    def test_insufficient_admissions(self):
        patient = MagicMock()
        patient.get_events.return_value = [] # No admissions
        self.assertEqual(self.task(patient), [])

if __name__ == '__main__':
    unittest.main()

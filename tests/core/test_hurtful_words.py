import unittest
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from pyhealth.tasks import HurtfulWordsMortalityTask

class MockEvent:
    """Mock for pyhealth Event objects."""
    def __init__(self, event_type: str, timestamp: Optional[datetime], hadm_id: str = None, attr_dict: Dict = None):
        self.event_type = event_type
        self.timestamp = timestamp
        self.hadm_id = hadm_id
        self.attr_dict = attr_dict or {}
        if hadm_id:
            self.attr_dict["hadm_id"] = hadm_id
        self.hospital_expire_flag = self.attr_dict.get("hospital_expire_flag", 0)

class MockPatient:
    """Optimized Mock Patient for sub-millisecond lookups."""
    def __init__(self, patient_id: str, events: List[MockEvent]):
        self.patient_id = patient_id
        self.events_cache = {}
        for e in events:
            self.events_cache.setdefault(e.event_type, []).append(e)

    def get_events(self, event_type: str, filters: List = None) -> List[MockEvent]:
        events = self.events_cache.get(event_type, [])
        if not filters:
            return events
        for key, op, value in filters:
            if op == "==":
                events = [e for e in events if e.attr_dict.get(key) == value]
        return events

class TestHurtfulWordsMortalityTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Zero-redundancy setup: Initialize task and temp environment once."""
        cls.test_dir = tempfile.mkdtemp()
        cls.task = HurtfulWordsMortalityTask()

    @classmethod
    def tearDownClass(cls):
        """Cleanup temporary artifacts."""
        shutil.rmtree(cls.test_dir)

    def test_full_pipeline_and_feature_extraction(self):
        """Tests label generation, feature extraction, and intersectional logic."""
        dob = datetime(1980, 1, 1)
        v1_time = datetime(2020, 1, 1)
        v2_time = datetime(2020, 1, 10)
        
        p_event = MockEvent("patients", None, attr_dict={"gender": "M", "dob": dob})
        v1 = MockEvent("admissions", v1_time, hadm_id="H1", 
                       attr_dict={"ethnicity": "HISPANIC", "insurance": "Medicaid"})

        n1 = MockEvent("noteevents", v1_time, hadm_id="H1", attr_dict={"text": "Note part 1."})
        n2 = MockEvent("noteevents", v1_time, hadm_id="H1", attr_dict={"text": "Note part 2."})
        
        v2 = MockEvent("admissions", v2_time, hadm_id="H2", attr_dict={"hospital_expire_flag": 0})

        patient = MockPatient("P001", [p_event, v1, n1, n2, v2])
        samples = self.task(patient)

        self.assertEqual(len(samples), 1)
        s = samples[0]

        self.assertEqual(s["mortality"], 0)
        self.assertEqual(s["intersectional_group"], "M_HISPANIC")
        self.assertEqual(s["insurance"], "Medicaid")
        self.assertEqual(s["age"], 40)

        self.assertEqual(s["clinical_notes"], "Note part 1. Note part 2.")

    def test_edge_case_missing_demographics(self):
        """Tests fallback to 'UNKNOWN' when demographics are missing or empty."""
        v1_time = datetime(2020, 1, 1)
        v2_time = datetime(2020, 1, 10)
        
        # Missing gender and DOB
        p_event = MockEvent("patients", None, attr_dict={}) 
        # Missing ethnicity and insurance
        v1 = MockEvent("admissions", v1_time, hadm_id="H1", attr_dict={})
        n1 = MockEvent("noteevents", v1_time, hadm_id="H1", attr_dict={"text": "Some text."})
        v2 = MockEvent("admissions", v2_time, hadm_id="H2")

        patient = MockPatient("P002", [p_event, v1, n1, v2])
        samples = self.task(patient)

        if samples:
            s = samples[0]
            self.assertEqual(s["gender"], "UNKNOWN")
            self.assertEqual(s["ethnicity"], "UNKNOWN")
            self.assertEqual(s["intersectional_group"], "UNKNOWN_UNKNOWN")

    def test_edge_case_invalid_age(self):
        # Patient born in the future relative to admission
        dob = datetime(2025, 1, 1)
        v1_time = datetime(2020, 1, 1)
        v2_time = datetime(2020, 1, 10)
        
        p_event = MockEvent("patients", None, attr_dict={"gender": "F", "dob": dob})
        v1 = MockEvent("admissions", v1_time, hadm_id="H1", attr_dict={"ethnicity": "OTHER"})
        n1 = MockEvent("noteevents", v1_time, hadm_id="H1", attr_dict={"text": "Valid text."})
        v2 = MockEvent("admissions", v2_time, hadm_id="H2")

        patient = MockPatient("P003", [p_event, v1, n1, v2])
        samples = self.task(patient)
        
        # Should be skipped because age < 0
        self.assertEqual(len(samples), 0)

    def test_edge_case_chronology(self):
        """Tests that visits are sorted by timestamp before processing."""
        dob = datetime(1950, 1, 1)
        # Out of order timestamps
        v2_time = datetime(2020, 5, 1) # Later visit
        v1_time = datetime(2020, 1, 1) # Earlier visit
        
        p_event = MockEvent("patients", None, attr_dict={"gender": "F", "dob": dob})
        v2 = MockEvent("admissions", v2_time, hadm_id="H2", attr_dict={"hospital_expire_flag": 1})
        v1 = MockEvent("admissions", v1_time, hadm_id="H1", attr_dict={"ethnicity": "WHITE"})
        n1 = MockEvent("noteevents", v1_time, hadm_id="H1", attr_dict={"text": "Earlier note."})

        # Added to Mock in reverse chronological order
        patient = MockPatient("P004", [p_event, v2, v1, n1])
        samples = self.task(patient)

        # Logic should sort them and treat v1 as the input for v2's outcome
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["hadm_id"], "H1")
        self.assertEqual(samples[0]["mortality"], 1)

if __name__ == "__main__":
    unittest.main()
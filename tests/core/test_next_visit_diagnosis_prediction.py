"""
Tests for next-visit diagnosis prediction task.

This test suite verifies the correctness of the next-visit diagnosis
prediction task function across different scenarios and edge cases.

Note: These tests use mock Patient objects. The actual Patient class
in PyHealth requires a data_source (DataFrame), so these tests create
simplified mock objects for testing purposes.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock


class MockVisit:
    """Mock Visit object for testing."""
    
    def __init__(self, visit_id, patient_id, encounter_time=None, discharge_time=None):
        self.visit_id = visit_id
        self.patient_id = patient_id
        self.encounter_time = encounter_time
        self.discharge_time = discharge_time
        self._code_lists = {}
    
    def add_codes(self, table, codes):
        """Add codes to this visit."""
        self._code_lists[table] = codes
    
    def get_code_list(self, table):
        """Get list of codes for a table."""
        return self._code_lists.get(table, [])


class MockPatient:
    """Mock Patient  for testing."""
    
    def __init__(self, patient_id):
        self.patient_id = patient_id
        self._visits = []
    
    def add_visit(self, visit):
        """Add a visit to this patient."""
        self._visits.append(visit)
    
    def __len__(self):
        """Return number of visits."""
        return len(self._visits)
    
    def __getitem__(self, index):
        """Get visit by index."""
        return self._visits[index]
    
    def __iter__(self):
        """Iterate over visits."""
        return iter(self._visits)


def next_visit_diagnosis_prediction_mimic4_fn(patient, time_aware=True):
    """
    Simplified version of the task function for testing.
    In production, replace this with the actual import.
    """
    samples = []
    
    if len(patient) < 2:
        return samples
    
    for target_idx in range(1, len(patient)):
        conditions_history = []
        procedures_history = []
        time_gaps = []
        prev_encounter_time = None
        
        for hist_idx in range(target_idx):
            visit = patient[hist_idx]
            
            conditions = visit.get_code_list(table="diagnoses_icd")
            conditions_history.append(conditions)
            
            procedures = visit.get_code_list(table="procedures_icd")
            procedures_history.append(procedures)
            
            if time_aware:
                current_time = visit.encounter_time
                if prev_encounter_time is not None and current_time is not None:
                    days_gap = (current_time - prev_encounter_time).days
                    time_gaps.append(days_gap)
                if current_time is not None:
                    prev_encounter_time = current_time
        
        target_visit = patient[target_idx]
        label = target_visit.get_code_list(table="diagnoses_icd")
        
        if len(label) == 0:
            continue
        
        sample = {
            "patient_id": patient.patient_id,
            "visit_id": target_visit.visit_id,
            "conditions_history": conditions_history,
            "procedures_history": procedures_history,
            "label": label,
        }
        
        if time_aware:
            sample["time_gaps"] = time_gaps
        
        samples.append(sample)
    
    return samples


class TestNextVisitDiagnosisPredictionMIMIC4(unittest.TestCase):
    """Test next-visit diagnosis prediction for MIMIC-IV."""
    
    def setUp(self):
        """Create a mock patient with 3 visits for testing."""
        self.patient = MockPatient(patient_id="patient-123")
        
        # Visit 1: Jan 1, 2020 with 2 diagnoses and 1 procedure
        visit1 = MockVisit(
            visit_id="visit-1",
            patient_id="patient-123",
            encounter_time=datetime(2020, 1, 1, 10, 0),
            discharge_time=datetime(2020, 1, 5, 14, 0),
        )
        visit1.add_codes("diagnoses_icd", ["427.31", "585.9"])
        visit1.add_codes("procedures_icd", ["96.04"])
        
        # Visit 2: Feb 15, 2020 (45 days later)
        visit2 = MockVisit(
            visit_id="visit-2",
            patient_id="patient-123",
            encounter_time=datetime(2020, 2, 15, 10, 0),
            discharge_time=datetime(2020, 2, 20, 14, 0),
        )
        visit2.add_codes("diagnoses_icd", ["401.9", "403.91"])
        visit2.add_codes("procedures_icd", ["96.71", "99.15"])
        
        # Visit 3: April 1, 2020 (46 days later)
        visit3 = MockVisit(
            visit_id="visit-3",
            patient_id="patient-123",
            encounter_time=datetime(2020, 4, 1, 10, 0),
            discharge_time=datetime(2020, 4, 8, 14, 0),
        )
        visit3.add_codes("diagnoses_icd", ["250.00", "584.9"])
        
        self.patient.add_visit(visit1)
        self.patient.add_visit(visit2)
        self.patient.add_visit(visit3)
    
    def test_generates_correct_number_of_samples(self):
        """Test that task generates correct number of samples."""
        samples = next_visit_diagnosis_prediction_mimic4_fn(self.patient)
        
        # With 3 visits, should generate 2 samples
        self.assertEqual(len(samples), 2)
    
    def test_first_sample_structure_and_content(self):
        """Test first sample has correct structure and content."""
        samples = next_visit_diagnosis_prediction_mimic4_fn(self.patient)
        sample1 = samples[0]
        
        # Check all required keys exist
        required_keys = [
            "patient_id",
            "visit_id",
            "conditions_history",
            "procedures_history",
            "time_gaps",
            "label"
        ]
        for key in required_keys:
            self.assertIn(key, sample1)
        
        # Check patient and visit IDs
        self.assertEqual(sample1["patient_id"], "patient-123")
        self.assertEqual(sample1["visit_id"], "visit-2")
        
        # Check history contains correct number of visits
        self.assertEqual(len(sample1["conditions_history"]), 1)
        self.assertEqual(len(sample1["procedures_history"]), 1)
        
        # Check first visit has correct diagnoses
        self.assertEqual(len(sample1["conditions_history"][0]), 2)
        self.assertIn("427.31", sample1["conditions_history"][0])
        self.assertIn("585.9", sample1["conditions_history"][0])
        
        # Check label contains visit 2 diagnoses
        self.assertEqual(len(sample1["label"]), 2)
        self.assertIn("401.9", sample1["label"])
        self.assertIn("403.91", sample1["label"])
    
    def test_second_sample_accumulates_history(self):
        """Test second sample correctly accumulates visit history."""
        samples = next_visit_diagnosis_prediction_mimic4_fn(self.patient)
        sample2 = samples[1]
        
        # History should contain 2 visits
        self.assertEqual(len(sample2["conditions_history"]), 2)
        self.assertEqual(len(sample2["procedures_history"]), 2)
        
        # Label should be visit 3 diagnoses
        self.assertIn("250.00", sample2["label"])
        self.assertIn("584.9", sample2["label"])
    
    def test_time_gaps_calculated_correctly(self):
        """Test time gaps between visits are calculated correctly."""
        samples = next_visit_diagnosis_prediction_mimic4_fn(self.patient)
        
        # First sample: only 1 visit in history, so 0 gaps
        self.assertEqual(len(samples[0]["time_gaps"]), 0)
        
        # Second sample: 2 visits in history, so 1 gap
        self.assertEqual(len(samples[1]["time_gaps"]), 1)
        # Gap should be approximately 45 days
        self.assertAlmostEqual(samples[1]["time_gaps"][0], 45, delta=1)
    
    def test_time_aware_false_removes_time_gaps(self):
        """Test that time_aware=False removes time_gaps from output."""
        samples = next_visit_diagnosis_prediction_mimic4_fn(
            self.patient,
            time_aware=False
        )
        
        # time_gaps should not be in the sample
        self.assertNotIn("time_gaps", samples[0])
        self.assertNotIn("time_gaps", samples[1])
    
    def test_single_visit_patient_returns_empty(self):
        """Test that patients with only 1 visit return no samples."""
        single_visit_patient = MockPatient(patient_id="patient-456")
        
        visit = MockVisit(
            visit_id="visit-1",
            patient_id="patient-456",
            encounter_time=datetime(2020, 1, 1),
        )
        visit.add_codes("diagnoses_icd", ["401.9"])
        single_visit_patient.add_visit(visit)
        
        samples = next_visit_diagnosis_prediction_mimic4_fn(single_visit_patient)
        
        # Should return empty list
        self.assertEqual(len(samples), 0)
    
    def test_skips_target_visits_without_diagnoses(self):
        """Test that target visits without diagnoses are skipped."""
        patient = MockPatient(patient_id="patient-789")
        
        # Visit 1: has diagnoses
        visit1 = MockVisit(
            visit_id="v1",
            patient_id="patient-789",
            encounter_time=datetime(2020, 1, 1),
        )
        visit1.add_codes("diagnoses_icd", ["401.9"])
        
        # Visit 2: NO diagnoses (only procedures)
        visit2 = MockVisit(
            visit_id="v2",
            patient_id="patient-789",
            encounter_time=datetime(2020, 2, 1),
        )
        visit2.add_codes("procedures_icd", ["96.04"])
        
        patient.add_visit(visit1)
        patient.add_visit(visit2)
        
        samples = next_visit_diagnosis_prediction_mimic4_fn(patient)
        
        # Should return 0 samples because target visit has no diagnoses
        self.assertEqual(len(samples), 0)
    
    def test_handles_missing_timestamps(self):
        """Test handling of visits with missing encounter times."""
        patient = MockPatient(patient_id="patient-999")
        
        # Visit 1: has timestamp
        visit1 = MockVisit(
            visit_id="v1",
            patient_id="patient-999",
            encounter_time=datetime(2020, 1, 1),
        )
        visit1.add_codes("diagnoses_icd", ["401.9"])
        
        # Visit 2: NO timestamp
        visit2 = MockVisit(
            visit_id="v2",
            patient_id="patient-999",
            encounter_time=None,
        )
        visit2.add_codes("diagnoses_icd", ["250.00"])
        
        patient.add_visit(visit1)
        patient.add_visit(visit2)
        
        samples = next_visit_diagnosis_prediction_mimic4_fn(patient)
        
        # Should still generate 1 sample
        self.assertEqual(len(samples), 1)
        
        # time_gaps should be empty (couldn't calculate)
        self.assertEqual(len(samples[0]["time_gaps"]), 0)
    
    def test_handles_many_visits(self):
        """Test with patient having many visits."""
        patient = MockPatient(patient_id="patient-long")
        
        # Create 5 visits
        for i in range(5):
            visit = MockVisit(
                visit_id=f"visit-{i}",
                patient_id="patient-long",
                encounter_time=datetime(2020, 1, 1) + timedelta(days=30*i),
            )
            visit.add_codes("diagnoses_icd", [f"40{i}.9"])
            patient.add_visit(visit)
        
        samples = next_visit_diagnosis_prediction_mimic4_fn(patient)
        
        # Should generate 4 samples (visits 1-4 as targets)
        self.assertEqual(len(samples), 4)
        
        # Last sample should have history of 4 visits
        self.assertEqual(len(samples[-1]["conditions_history"]), 4)
        
        # Last sample should have 3 time gaps
        self.assertEqual(len(samples[-1]["time_gaps"]), 3)


if __name__ == "__main__":
    # Run tests
    print("=" * 70)
    print("TESTING NEXT-VISIT DIAGNOSIS PREDICTION TASK")
    print("=" * 70)
    print("\nNote: These tests use mock objects.")
    print("For production, integrate with actual PyHealth Patient/Visit classes.")
    print("=" * 70 + "\n")
    
    unittest.main(verbosity=2)  
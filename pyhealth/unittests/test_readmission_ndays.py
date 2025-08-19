import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import sys 
import os
current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current)))

from pyhealth.tasks import ReadmissionNDaysMIMIC4


class TestReadmissionNDaysMIMIC4(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock patient
        self.mock_patient = MagicMock()
        
        # Mock demographics data - 25 year old patient
        demographics = MagicMock()
        demographics.__getitem__.side_effect = {"anchor_age": "25"}.get
        self.mock_patient.get_events.return_value = [demographics]
        
        # Base timestamp for tests
        self.base_timestamp = datetime(2023, 1, 1, 10, 0, 0)

    def create_mock_admission(self, timestamp, dischtime):
        """Helper to create mock admission objects."""
        admission = MagicMock()
        admission.timestamp = timestamp
        admission.dischtime = dischtime.strftime("%Y-%m-%d %H:%M:%S")
        admission.hadm_id = f"adm_{timestamp.strftime('%Y%m%d%H%M')}"
        return admission
    
    def test_task_name_changes_with_n_days(self):
        """Test that task name updates according to n_days parameter."""
        task_7 = ReadmissionNDaysMIMIC4(n_days=7)
        task_30 = ReadmissionNDaysMIMIC4()  # Default is 30
        task_90 = ReadmissionNDaysMIMIC4(n_days=90)
        
        self.assertEqual(task_7.task_name, "Readmission7DaysMIMIC4")
        self.assertEqual(task_30.task_name, "Readmission30DaysMIMIC4")
        self.assertEqual(task_90.task_name, "Readmission90DaysMIMIC4")

    def test_underage_patient_returns_empty(self):
        """Test that patients under 18 result in empty samples."""
        # Update mock to represent a 16-year-old
        demographics = MagicMock()
        demographics.__getitem__.side_effect = {"anchor_age": "16"}.get
        mock_underage_patient = MagicMock()
        mock_underage_patient.get_events.return_value = [demographics]
        
        task = ReadmissionNDaysMIMIC4()
        result = task(mock_underage_patient)
        
        self.assertEqual(result, [])

    @patch('pyhealth.tasks.ReadmissionNDaysMIMIC4.__call__')
    def test_correct_n_days_threshold(self, mock_call):
        """Test different readmission thresholds (7, 30, 90 days)."""
        # Setup
        task_7 = ReadmissionNDaysMIMIC4(n_days=7)
        task_30 = ReadmissionNDaysMIMIC4()  # default 30
        task_90 = ReadmissionNDaysMIMIC4(n_days=90)
        
        # Assert the n_days attribute is correctly set
        self.assertEqual(task_7.n_days, 7)
        self.assertEqual(task_30.n_days, 30)
        self.assertEqual(task_90.n_days, 90)

    def test_readmission_calculation(self):
        """Test readmission is calculated correctly for different windows."""
        # Setup admissions with different readmission patterns
        adm1 = self.create_mock_admission(
            self.base_timestamp, 
            self.base_timestamp + timedelta(days=1)
        )
        
        # Readmission after 5 days (positive for 7+ day windows)
        adm2 = self.create_mock_admission(
            self.base_timestamp + timedelta(days=6), 
            self.base_timestamp + timedelta(days=8)
        )
        
        # Readmission after 20 days (positive for 30+ day windows)
        adm3 = self.create_mock_admission(
            self.base_timestamp + timedelta(days=28), 
            self.base_timestamp + timedelta(days=32)
        )
        
        # Readmission after 60 days (positive for 90-day window only)
        adm4 = self.create_mock_admission(
            self.base_timestamp + timedelta(days=92), 
            self.base_timestamp + timedelta(days=95)
        )
        
        # No further readmission
        
        # Create mock patient with admission history and clinical codes
        def mock_get_events(event_type, start=None, end=None):
            if event_type == "patients":
                demographics = MagicMock()
                demographics.__getitem__.side_effect = {"anchor_age": "25"}.get
                return [demographics]
            elif event_type == "admissions":
                return [adm1, adm2, adm3, adm4]
            elif event_type in ["diagnoses_icd", "procedures_icd", "prescriptions"]:
                # Return some mock clinical codes
                mock_event = MagicMock()
                if event_type == "diagnoses_icd":
                    mock_event.icd_version = "ICD10"
                    mock_event.icd_code = "E11.9"
                elif event_type == "procedures_icd":
                    mock_event.icd_version = "ICD10"
                    mock_event.icd_code = "0SG"
                else:  # prescriptions
                    mock_event.drug = "Aspirin"
                return [mock_event]
        
        mock_patient = MagicMock()
        mock_patient.get_events.side_effect = mock_get_events
        mock_patient.patient_id = "test_patient"
        
        # Test with 7-day window
        task_7 = ReadmissionNDaysMIMIC4(n_days=7)
        results_7 = task_7(mock_patient)
        
        # Test with 30-day window
        task_30 = ReadmissionNDaysMIMIC4()  # default 30
        results_30 = task_30(mock_patient)
        
        # Test with 90-day window
        task_90 = ReadmissionNDaysMIMIC4(n_days=90)
        results_90 = task_90(mock_patient)
        
        # Verify correct readmission labeling based on windows
        # First admission: should be positive for all windows
        self.assertEqual(results_7[0]["readmission"], 1)
        self.assertEqual(results_30[0]["readmission"], 1)
        self.assertEqual(results_90[0]["readmission"], 1)
        
        # Second admission: should be positive for 30 and 90 day windows
        self.assertEqual(results_7[1]["readmission"], 0)
        self.assertEqual(results_30[1]["readmission"], 1)
        self.assertEqual(results_90[1]["readmission"], 1)
        
        # Third admission: should be positive only for 90 day window
        self.assertEqual(results_7[2]["readmission"], 0)
        self.assertEqual(results_30[2]["readmission"], 0)
        self.assertEqual(results_90[2]["readmission"], 1)
        
        # Fourth admission: should be negative for all
        self.assertEqual(results_7[3]["readmission"], 0)
        self.assertEqual(results_30[3]["readmission"], 0)
        self.assertEqual(results_90[3]["readmission"], 0)

    def test_short_stay_filtering(self):
        """Test that stays < 12 hours are filtered out."""
        # Create admissions with very short stay (< 12 hours)
        adm_short = self.create_mock_admission(
            self.base_timestamp, 
            self.base_timestamp + timedelta(hours=6)  # 6 hour stay
        )
        
        def mock_get_events(event_type, start=None, end=None):
            if event_type == "patients":
                demographics = MagicMock()
                demographics.__getitem__.side_effect = {"anchor_age": "25"}.get
                return [demographics]
            elif event_type == "admissions":
                return [adm_short]
            else:
                # Return mocked clinical events
                mock_event = MagicMock()
                if event_type == "diagnoses_icd":
                    mock_event.icd_version = "ICD10"
                    mock_event.icd_code = "E11.9"
                elif event_type == "procedures_icd":
                    mock_event.icd_version = "ICD10"
                    mock_event.icd_code = "0SG"
                else:  # prescriptions
                    mock_event.drug = "Aspirin"
                return [mock_event]
        
        mock_patient = MagicMock()
        mock_patient.get_events.side_effect = mock_get_events
        
        task = ReadmissionNDaysMIMIC4()
        results = task(mock_patient)
        
        # Should be filtered out due to short duration
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
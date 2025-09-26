# pyhealth/tasks/test_mimic3_note_tasks.py

import unittest
import re
from unittest.mock import patch, MagicMock
import pandas as pd
from pyhealth.tasks.mimic3_note_tasks import MIMIC3NoteReplaceDeIdTask
from pyhealth.data import Patient
import datetime

class TestMIMIC3NoteReplaceDeIdTask(unittest.TestCase):
    """Test cases for the MIMIC3NoteReplaceDeIdTask class."""

    def setUp(self):
        """Set up test fixtures."""
        self.task = MIMIC3NoteReplaceDeIdTask()
        
        # Create a mock patient
        self.patient = MagicMock(spec=Patient)
        self.patient.patient_id = "12345"
        
        # Create mock note events
        self.note_event1 = MagicMock()
        self.note_event1.attr_dict = {"text": "Patient [**Name (NI) **] was seen by Dr. [**Last Name (NamePattern1) **] on [**2018-01-15**]."}
        self.note_event1.timestamp = datetime.datetime(2018, 1, 15, 10, 30)
        
        self.note_event2 = MagicMock()
        self.note_event2.attr_dict = {"text": ""}  # Empty note
        self.note_event2.timestamp = datetime.datetime(2018, 1, 16, 9, 15)
        
        self.note_event3 = MagicMock()
        self.note_event3.attr_dict = {"text": "Patient is a 95-year-old [**Age over 90 **] male admitted on [**2018-01-10**] to [**Hospital **]."}
        self.note_event3.timestamp = datetime.datetime(2018, 1, 17, 14, 45)

    def test_task_initialization(self):
        """Test task initialization and schema."""
        self.assertEqual(self.task.task_name, "Deidentifying MIMIC-III Clinical Notes")
        self.assertEqual(self.task.input_schema, {"text": "text"})
        self.assertEqual(self.task.output_schema, {"masked_text": "text"})

    def test_is_date(self):
        """Test the is_date method."""
        self.assertTrue(self.task.is_date("2022-01-15"))
        self.assertTrue(self.task.is_date("2022-1-5"))
        self.assertTrue(self.task.is_date("This is in the month of January"))
        self.assertTrue(self.task.is_date("Next year 2024"))
        self.assertFalse(self.task.is_date("This is not a date"))

    def test_repl_deid(self):
        """Test the repl_deid method with different categories."""
        # Create mock Match objects for different PHI categories
        match_date = MagicMock()
        match_date.group.return_value = "2022-01-15"
        self.assertEqual(self.task.repl_deid(match_date), "PHIDATEPHI")
        
        match_hospital = MagicMock()
        match_hospital.group.return_value = "Hospital Name"
        self.assertEqual(self.task.repl_deid(match_hospital), "PHIHOSPITALPHI")
        
        
        match_name = MagicMock()
        match_name.group.return_value = "First Name"
        self.assertEqual(self.task.repl_deid(match_name), "PHINAMEPHI")
        
        match_telephone = MagicMock()
        match_telephone.group.return_value = "telephone number"
        self.assertEqual(self.task.repl_deid(match_telephone), "PHICONTACTPHI")
        
        match_number = MagicMock()
        match_number.group.return_value = "123456"
        self.assertEqual(self.task.repl_deid(match_number), "PHINUMBERPHI")
        
        match_age = MagicMock()
        match_age.group.return_value = "age over 90"
        self.assertEqual(self.task.repl_deid(match_age), "PHIAGEPHI")
        
        match_other = MagicMock()
        match_other.group.return_value = "unknown category"
        self.assertEqual(self.task.repl_deid(match_other), "PHIOTHERPHI")

    def test_replace_deid(self):
        """Test the replace_deid method for PHI replacement."""
        input_text = "Patient [**Name (NI) **] was seen on [**2018-01-15**]."
        expected_output = "Patient PHINAMEPHI was seen on PHIDATEPHI."
        self.assertEqual(self.task.replace_deid(input_text), expected_output)

    def test_mask_gendered_terms(self):
        """Test masking of gendered terms."""
        input_text = "The patient is a 45-year-old male. He was admitted yesterday."
        expected_output = "The patient is a 45-year-old [GEND]. [GEND] was admitted yesterday."
        self.assertEqual(self.task.mask_gendered_terms(input_text), expected_output)

    def test_clean_note_format(self):
        """Test cleaning and standardization of note format."""
        input_text = "1. Patient history\n2. Current medications\n---\nSeen by Dr. Smith, M.D."
        expected_output = "Patient history Current medications Seen by doctor Smith, md"
        self.assertEqual(self.task.clean_note_format(input_text), expected_output)

    def test_process_note(self):
        """Test the complete note processing pipeline."""
        input_text = "Patient [**Name (NI) **] is a 45-year-old male.\nHe was seen by Dr. [**Last Name (NamePattern1) **] on [**2018-01-15**]."
        processed_text = self.task.process_note(input_text)
        
        # Check that PHI markers are replaced
        self.assertNotIn("[**", processed_text)
        self.assertNotIn("**]", processed_text)
        
        # Check that gendered terms are masked
        self.assertIn("[GEND]", processed_text)
        
        # Check that formatting is cleaned
        self.assertNotIn("\n", processed_text)

    @patch('pyhealth.data.Patient')
    def test_call_with_patient(self, mock_patient_class):
        """Test the __call__ method with a patient containing notes."""
        # Set up the patient to return our mock events
        self.patient.get_events.return_value = [self.note_event1, self.note_event2, self.note_event3]
        
        # Call the task with the patient
        results = self.task(self.patient)
        
        # Verify results
        self.assertEqual(len(results), 2)  # Only 2 notes have content
        
        # Check that the patient entity was updated
        self.assertEqual(self.note_event1.attr_dict["phimasked"], True)
        self.assertEqual(self.note_event3.attr_dict["phimasked"], True)
        
        # Check first result
        self.assertEqual(results[0]["patient_id"], "12345")
        self.assertIn("PHINAMEPHI", results[0]["masked_text"])
        self.assertIn("PHIDATEPHI", results[0]["masked_text"])
        
        # Check that empty notes were skipped
        self.patient.get_events.assert_called_once_with(event_type="noteevents")

    def test_call_with_empty_notes(self):
        """Test the __call__ method with a patient having only empty notes."""
        # Set up the patient to return only the empty note event
        self.patient.get_events.return_value = [self.note_event2]
        
        # Call the task with the patient
        results = self.task(self.patient)
        
        # Verify no results are returned for empty notes
        self.assertEqual(len(results), 0)

    def test_integration_process_notes(self):
        """Integration test for processing multiple notes."""
        # Set up more complex notes
        complex_note = (
            "Patient [**Name (NI) **], a 67-year-old female, was admitted to [**Hospital **] on [**2018-01-15**].\n"
            "She was referred by Dr. [**Last Name (NamePattern1) **] from [**Location (un) **].\n"
            "Medical Record Number: [**Medical Record Number(1) **]\n"
            "Phone: [**Telephone/Fax (1) **]\n"
            "Her address is [**Street Address(1) **], [**State **]."
        )
        
        self.note_event1.attr_dict["text"] = complex_note
        self.patient.get_events.return_value = [self.note_event1]
        
        # Process the notes
        results = self.task(self.patient)
        
        # Check that all types of PHI were correctly masked
        processed_text = results[0]["masked_text"]
        self.assertIn("PHINAMEPHI", processed_text)
        self.assertIn("PHIHOSPITALPHI", processed_text)
        self.assertIn("PHIDATEPHI", processed_text)
        self.assertIn("PHILOCATIONPHI", processed_text)
        self.assertIn("PHINUMBERPHI", processed_text)
        self.assertIn("PHICONTACTPHI", processed_text)
        self.assertIn("[GEND]", processed_text)  # Gendered term replacement


if __name__ == "__main__":
    unittest.main()
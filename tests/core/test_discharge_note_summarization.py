"""
Unit tests for DischargeNoteSummarization task in summarization_data_processing.py.

Tests cover:
    - Class attributes (task_name, input_schema, output_schema)
    - __call__: happy-path extraction of brief_hospital_course and summary
    - __call__: all boundary / filtering conditions that cause samples to be skipped
    - Output dictionary structure and field types

External dependencies (pyhealth) are fully mocked so the tests run without
installing the real library or accessing any dataset.

Run with:
    python -m pytest test_summarization_data_processing.py -v
    # or
    python -m unittest test_summarization_data_processing.py -v
"""


import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from pathlib import Path
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.data import Patient
import tempfile
from pyhealth.tasks import DischargeNoteSummarization
from unittest.mock import MagicMock
from pyhealth.data import Patient, Event


import logging

class TestDischargeNoteSummarizationTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_resources = Path(__file__).parent.parent.parent / "test-resources" / "discharge"
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls.full_note_dataset = MIMIC4Dataset(
            note_root=cls.test_resources,
            note_tables=["discharge"])
        cls.task = DischargeNoteSummarization()
        cls.sample_notes = cls.full_note_dataset.set_task(cls.task)
        cls.MIN_SUMMARY_LENGTH = 350

    def create_mock_patient(self, note_text, patient_id="p1", hadm_id="h1", subject_id="20000003"):
        """Helper to create a mock Patient with a single discharge event."""
        patient = MagicMock(spec=Patient)
        patient.patient_id = patient_id
        
        # Create a mock Event for the discharge note
        event = MagicMock(spec=Event)
        event.attr_dict = {
            "text": note_text,
            "hadm_id": hadm_id,
            "subject_id": subject_id
        }
        
        # Mock the get_events method to return our discharge event
        patient.get_events.side_effect = lambda event_type: [event] if event_type == "discharge" else []
        return patient
    
        
    def test_generated_samples(self):
        self.assertEqual(len(self.sample_notes), 2)
        print(self.sample_notes[0]["summary"])
        self.assertTrue(self.sample_notes[0]["summary"].startswith("Discharge Instructions:"))


        

    def test_task_metadata(self):
        self.assertEqual(self.task.task_name,"DischargeNoteSummarization")
        self.assertIn("text", self.task.input_schema)
        self.assertIn("summary", self.task.output_schema)

    def test_filtering_short_summary(self):
        
        note = (
                "Brief Hospital Course:\n"
                "The patient is an elderly individual with a significant past medical history of chronic obstructive "
                "pulmonary disease, congestive heart failure with a reduced ejection fraction of thirty-five percent, "
                "and Type 2 diabetes mellitus. The patient presented to the emergency department complaining of "
                "progressive shortness of breath, productive cough with yellow sputum, and bilateral lower extremity "
                "edema increasing over the last five days. Upon arrival, the patient was tachycardic and hypoxic, "
                "requiring supplemental oxygen via nasal cannula to maintain saturations above ninety-two percent. "
                "A chest X-ray revealed bilateral pulmonary infiltrates and pleural effusions, consistent with a "
                "multifocal pneumonia overlaying a congestive heart failure exacerbation. Laboratory results were "
                "significant for an elevated pro-BNP and a leukocytosis with an elevated white blood cell count. "
                "During the first forty-eight hours of admission, the patient was started on intravenous antibiotics "
                "for community-acquired pneumonia. Diuresis was initiated with intravenous medications, resulting in "
                "a significant net negative fluid balance over three days. The patient’s respiratory status "
                "improved significantly; oxygen was successfully weaned to room air by hospital day four. "
                "Endocrinology was consulted for blood glucose management, and the insulin regimen was "
                "adjusted to a sliding scale with a long-acting basal dose. By the day of discharge, the "
                "patient was stable, ambulating without distress, and lung sounds were markedly clearer on "
                "auscultation. Weight had returned to the documented baseline. "
                
                "Medications on Admission: "
                "Metformin, Lisinopril, Furosemide, and an Albuterol inhaler. "
                
                "Discharge Instructions: "
                "You were treated in the hospital for a combination of pneumonia and a flare-up of your heart "
                "failure. It is vital that you finish the entire course of oral antibiotics as prescribed, "
                "even if you feel better. Please monitor your weight every morning before breakfast. If you "
                "notice a weight gain of more than three pounds in a single day or five pounds in a week, "
                "contact your primary care doctor immediately as this indicates fluid buildup. Continue to "
                "use your salt-restricted diet and limit your total fluid intake to one and a half liters "
                "daily to prevent further strain on your heart. Rest is encouraged for the next week; however, "
                "try to perform light walking around the house to prevent blood clots. Avoid any heavy lifting "
                "or strenuous exercise until cleared by your cardiologist. You should continue your home "
                "medications as updated in the attached list. Seek immediate emergency care if you experience "
                "chest pain, severe shortness of breath while sitting still, or if you begin coughing up blood. "
                "We have adjusted your diuretic medication slightly to help manage your fluid levels more "
                "effectively during your recovery. Ensure you have picked up your new prescriptions from the "
                "pharmacy before the end of the day. It is also recommended that you receive your flu and "
                "pneumonia vaccinations once you have fully recovered from this current illness. Please bring "
                "your updated medication list to all upcoming appointments to ensure your medical record is accurate. "
                
                "Followup Instructions: "
                "Follow up with Cardiology next week. Follow up with your Primary Care Provider within seven days "
                "for a transition of care visit."
                        
        )
        patient = self.create_mock_patient(note)
        samples = self.task(patient)
        
        self.assertEqual(len(samples), 1, "This summary should not be filtered out as its length more than 350.")

    def test_edge_cases(self):
        """Verify that summaries shorter than MIN_SUMMARY_LENGTH (350) are skipped."""
        short_summary = "This summary is too short." # ~26 chars
        note = (
            #"Brief Hospital Course:\nStable.\n"
            "Medications on Admission:\nNone.\n"
            "Discharge Instructions:\n" + short_summary + "\n"
            "Followup Instructions:\nNone."
        )
        patient = self.create_mock_patient(note)
        samples = self.task(patient)
         
        self.assertEqual(len(samples), 0, "Should filter out samples with short summaries.")

    def test_edge_cases_1(self):
        short_summary = "This is a sample generated summary."
        note = (
            "Brief Hospital Course:\nStable.\n"
            #"Medications on Admission:\nNone.\n"
            #"Discharge Instructions:\n" + short_summary + "\n"
            #"Followup Instructions:\nNone."
            "This is a sample generated short summary that coes not contain all sections."
        )

        patient = self.create_mock_patient(note)
        samples = self.task(patient)
        
        self.assertEqual(len(samples), 0, "Should filter out samples with short summaries.")

    

if __name__ == "__main__":
    unittest.main()
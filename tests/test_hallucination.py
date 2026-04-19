import unittest
import polars as pl
from datetime import datetime
from pyhealth.data import Patient
from pyhealth.tasks.hallucination_detection import HallucinationDetectionTask

class TestHallucinationDetectionTask(unittest.TestCase):
    """Unit tests for the HallucinationDetectionTask class.

    This test suite verifies that the hallucination detection task correctly
    processes Polars-backed patient data, distinguishes between faithful and
    hallucinated summaries, and handles edge cases such as missing clinical notes.
    """

    def setUp(self) -> None:
        """Sets up the test environment with synthetic patient data.
        
        This method initializes a mock Polars DataFrame representing a patient
        with multiple visits, ensuring both positive and negative cases are
        available for the task logic to process.
        """
        # Setup a patient with one good visit and one hallucinated visit
        data = pl.DataFrame({
            "timestamp": [
                datetime(2026, 1, 1), datetime(2026, 1, 2), 
                datetime(2026, 2, 1), datetime(2026, 2, 2)
            ],
            "event_type": ["noteevents", "visits", "noteevents", "visits"],
            "noteevents/text": [
                ["Stable",  "vitals", "mild", "cough"], None,
                ["Laceration", "on", "left", "hand"], None
            ],
            "noteevents/visit_id": ["V1", None, "V2", None],
            "visits/visit_id": [None, "V1", None, "V2"],
            "visits/ai_summary": [
                None, ["Patient", "has", "a", "cough"], 
                None, ["Patient", "has", "a", "broken", "leg"] # Hallucination
            ],
            "visits/hallucination_label": [None, 0, None, 1] 
        })

        self.patient = Patient(patient_id="P1", data_source=data)
        self.task = HallucinationDetectionTask()

    def test_processing_and_labels(self) -> None:
        """Verifies that samples are extracted and labels match the source text.

        Checks for correct sample count, label accuracy for both faithful and
        hallucinated summaries, and proper feature extraction of note text.
        """
        samples = self.task(self.patient)
        self.assertEqual(len(samples), 2)
        
        # Check faithful case
        s1 = next(s for s in samples if s["visit_id"] == "V1")
        self.assertEqual(s1["label"], 0)
        self.assertIn("cough", s1["source_text"])

        # Check hallucination case
        s2 = next(s for s in samples if s["visit_id"] == "V2")
        self.assertEqual(s2["label"], 1)
        self.assertIn("broken", s2["summary_text"])

    def test_edge_case_no_notes(self) -> None:
        """Tests that visits without supporting clinical notes are skipped.

        Ensures that the task logic does not generate a sample if there is no
        source text (noteevents) available to ground the summary, preventing
        invalid training pairs.
        """
        # Create a patient who has a visit but zero clinical notes
        empty_data = pl.DataFrame({
            "timestamp": [datetime(2026, 3, 1)],
            "event_type": ["visits"],
            "visits/visit_id": ["V3"],
            "visits/ai_summary": ["Missing notes test."],
            "visits/hallucination_label": [0]
        })
        empty_patient = Patient(patient_id="P2", data_source=empty_data)
        
        samples = self.task(empty_patient)
        # Should return 0 samples because there's no source text to ground it
        self.assertEqual(len(samples), 0)
    
    def test_feature_types(self) -> None:
        """Verifies that extracted features are of the correct data type.
        
        Ensures that the source_text and summary_text are lists, so that 
        they can be processed as proper sequences.
        """
        samples = self.task(self.patient)
        s = samples[0]
        # Verify the Transformer will get lists, not strings
        self.assertIsInstance(s["source_text"], list)
        self.assertIsInstance(s["summary_text"], list)
        self.assertIsInstance(s["label"], int)
    
    def test_edge_case_missing_summary(self) -> None:
        """Tests that visits missing an AI summary are correctly ignored.
        
        If the AI summary is missing, we should not create a sample. This 
        ensures that the sample is not made if the AI summary is missing.
        """
        bad_data = pl.DataFrame({
            "timestamp": [datetime(2026, 4, 1), datetime(2026, 4, 2)],
            "event_type": ["noteevents", "visits"],
            "noteevents/text": [["test"], None],
            "noteevents/visit_id": ["V4", None],
            "visits/visit_id": [None, "V4"],
            "visits/ai_summary": [None, None], # MISSING SUMMARY
            "visits/hallucination_label": [None, 1]
        })
        p = Patient(patient_id="P3", data_source=bad_data)
        samples = self.task(p)
        self.assertEqual(len(samples), 0)

if __name__ == "__main__":
    unittest.main()
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
                "Stable vitals, mild cough.", None,
                "Laceration on left hand.", None
            ],
            "noteevents/visit_id": ["V1", None, "V2", None],
            "visits/visit_id": [None, "V1", None, "V2"],
            "visits/ai_summary": [
                None, "Patient has a cough.", 
                None, "Patient has a broken leg." # Hallucination
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
        """Verifies samples are extracted and labels match the source."""
        samples = self.task(self.patient)
        self.assertEqual(len(samples), 2)
        
        # Check faithful case
        s1 = next(s for s in samples if s["visit_id"] == "V1")
        self.assertEqual(s1["label"], 0)
        self.assertIn("cough", s1["source_text"])

        # Check hallucination case
        s2 = next(s for s in samples if s["visit_id"] == "V2")
        self.assertEqual(s2["label"], 1)
        self.assertIn("broken leg", s2["summary_text"])

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

if __name__ == "__main__":
    unittest.main()
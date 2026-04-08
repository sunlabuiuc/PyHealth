# tests/tasks/test_medlingo_expansion_task.py
import unittest
from types import SimpleNamespace

from pyhealth.tasks.medlingo_expansion import MedLingoExpansionTask

class FakePatient:
    """
    Minimal fake Patient object for task unit testing.
    """

    def __init__(self, patient_id, events):
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type: str):
        if event_type == "medlingo":
            return self._events
        return []

class TestMedLingoExpansionTask(unittest.TestCase):
    def setUp(self):
        self.task = MedLingoExpansionTask()

    def test_returns_expected_sample(self):
        fake_event = SimpleNamespace(
            context="Pt with elevated creat, monitor renal function.",
            term="creat",
            expansion="creatinine",
        )
        fake_patient = FakePatient(patient_id="medlingo_001", events=[fake_event])

        samples = self.task(fake_patient)

        self.assertIsInstance(samples, list)
        self.assertEqual(len(samples), 1)

        sample = samples[0]
        self.assertEqual(sample["patient_id"], "medlingo_001")
        self.assertEqual(
            sample["text"],
            "Clinical snippet: Pt with elevated creat, monitor renal function.\n"
            "Jargon term: creat",
        )
        self.assertEqual(sample["label"], "creatinine")

    def test_raises_for_missing_event(self):
        fake_patient = FakePatient(patient_id="medlingo_002", events=[])

        with self.assertRaisesRegex(
            ValueError, "Expected exactly one medlingo event per patient."
        ):
            self.task(fake_patient)

    def test_raises_for_multiple_events(self):
        fake_event_1 = SimpleNamespace(
            context="Pt with elevated creat.",
            term="creat",
            expansion="creatinine",
        )
        fake_event_2 = SimpleNamespace(
            context="Pt with fx of left wrist.",
            term="fx",
            expansion="fracture",
        )
        fake_patient = FakePatient(
            patient_id="medlingo_003",
            events=[fake_event_1, fake_event_2],
        )

        with self.assertRaisesRegex(
            ValueError, "Expected exactly one medlingo event per patient."
        ):
            self.task(fake_patient)

if __name__ == "__main__":
    unittest.main()
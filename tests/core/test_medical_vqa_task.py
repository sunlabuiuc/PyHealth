import unittest
from dataclasses import dataclass

from pyhealth.tasks import MedicalVQATask


@dataclass
class _DummyEvent:
    image_path: str
    question: str
    answer: str


class _DummyPatient:
    def __init__(self, patient_id: str, events):
        self.patient_id = patient_id
        self._events = events
        self.last_event_type = None

    def get_events(self, event_type=None):
        self.last_event_type = event_type
        return self._events


class TestMedicalVQATask(unittest.TestCase):
    def test_task_schema_attributes(self):
        task = MedicalVQATask()
        self.assertEqual(task.task_name, "MedicalVQA")
        self.assertEqual(task.input_schema, {"image": "image", "question": "text"})
        self.assertEqual(task.output_schema, {"answer": "multiclass"})

    def test_task_converts_events_to_samples(self):
        task = MedicalVQATask()
        patient = _DummyPatient(
            patient_id="patient-1",
            events=[
                _DummyEvent(
                    image_path="/tmp/study_0.png",
                    question="is there a fracture",
                    answer="yes",
                ),
                _DummyEvent(
                    image_path="/tmp/study_1.png",
                    question="is the study normal",
                    answer="no",
                ),
            ],
        )

        samples = task(patient)

        self.assertEqual(patient.last_event_type, "vqarad")
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["patient_id"], "patient-1")
        self.assertEqual(samples[0]["image"], "/tmp/study_0.png")
        self.assertEqual(samples[0]["question"], "is there a fracture")
        self.assertEqual(samples[0]["answer"], "yes")
        self.assertEqual(samples[1]["image"], "/tmp/study_1.png")
        self.assertEqual(samples[1]["question"], "is the study normal")
        self.assertEqual(samples[1]["answer"], "no")

    def test_task_returns_empty_list_for_patient_without_events(self):
        task = MedicalVQATask()
        patient = _DummyPatient(patient_id="patient-2", events=[])
        self.assertEqual(task(patient), [])

    def test_task_raises_for_missing_required_event_attribute(self):
        task = MedicalVQATask()

        class _IncompleteEvent:
            def __init__(self):
                self.image_path = "/tmp/study_missing.png"
                self.question = "is there edema"

        patient = _DummyPatient(patient_id="patient-3", events=[_IncompleteEvent()])

        with self.assertRaises(AttributeError):
            task(patient)


if __name__ == "__main__":
    unittest.main()

from typing import Dict, List
from pyhealth.data import Event, Patient
from pyhealth.tasks.base_task import BaseTask


class SleepQAExtractiveQA(BaseTask):
    """Extractive Question Answering task for SleepQA.

    This task maps SleepQA events into samples containing a passage, 
    a question, and the answer span (text and start index).

    Input Schema:
        passage: raw text context.
        question: the sleep-related query.
    Output Schema:
        answer_text: the ground truth answer string.
        answer_start: char-level start index of the answer.
    """
    task_name = "SleepQAExtractiveQA"
    input_schema = {"passage": "text", "question": "text"}
    output_schema = {"answer_text": "text", "answer_start": "multiclass"}

    def __call__(self, patient: Patient) -> List[Dict]:
        """Processes a patient object into QA samples.

        Args:
            patient: a Patient object containing SleepQA events.

        Returns:
            A list of sample dictionaries.
        """
        samples = []
        for event in patient.get_events(event_type="sleepqa"):
            samples.append({
                "patient_id": patient.patient_id,
                "visit_id": event.visit_id,
                # FIX: Use bracket notation [] instead of .get()
                "passage": event["passage"],
                "question": event["question"],
                "answer_text": event["answer_text"],
                "answer_start": int(event["answer_start"]),
            })
        return samples

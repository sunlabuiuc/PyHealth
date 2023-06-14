from dataclasses import dataclass, field
from typing import Dict

from pyhealth.tasks import TaskTemplate


@dataclass(frozen=True)
class MedicalTranscriptionsClassification(TaskTemplate):
    task_name: str = "MedicalTranscriptionsClassification"
    input_schema: Dict[str, str] = field(default_factory=lambda: {"transcription": "text"})
    output_schema: Dict[str, str] = field(default_factory=lambda: {"label": "label"})

    def __call__(self, patient):
        if patient["transcription"] is None or patient["medical_specialty"] is None:
            return []
        sample = {
            "transcription": patient["transcription"],
            "label": patient["medical_specialty"],
        }
        return [sample]


if __name__ == "__main__":
    task = MedicalTranscriptionsClassification()
    print(task)
    print(type(task))

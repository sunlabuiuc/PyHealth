from typing import Any, Dict, List

from ..data import Patient
from .base_task import BaseTask


class MedicalTranscriptionsClassification(BaseTask):
    """Task for classifying medical transcriptions into medical specialties.

    This task takes medical transcription text as input and predicts the
    corresponding medical specialty. It processes patient records containing
    mtsamples events and extracts transcription and medical specialty
    information.

    Attributes:
        task_name (str): Name of the task
        input_schema (Dict[str, str]): Schema defining input features
        output_schema (Dict[str, str]): Schema defining output features
    """
    task_name: str = "MedicalTranscriptionsClassification"
    input_schema: Dict[str, str] = {"transcription": "text"}
    output_schema: Dict[str, str] = {"medical_specialty": "multiclass"}

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Process a patient record to extract medical transcription samples.

        Args:
            patient (Patient): Patient record containing medical
                transcription events

        Returns:
            List[Dict[str, Any]]: List of samples containing transcription
                and medical specialty
        """
        event = patient.get_events(event_type="mtsamples")
        # There should be only one event
        assert len(event) == 1
        event = event[0]

        transcription_valid = isinstance(event.transcription, str)
        specialty_valid = isinstance(event.medical_specialty, str)

        if transcription_valid and specialty_valid:
            sample = {
                "id": patient.patient_id,
                "transcription": event.transcription,
                "medical_specialty": event.medical_specialty,
            }
            return [sample]
        else:
            return []

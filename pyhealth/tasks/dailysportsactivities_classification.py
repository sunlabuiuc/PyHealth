from typing import Any, Dict, List
import torch

from .base_task import BaseTask


class DSAClassification(BaseTask):
    """A task for classifying activity type.

    This task takes in various sensor readings from units placed on different
    parts of the body to predict what type of activity the person is doing.

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, person, segment_num,
            and sensor_data as well as the activity type
            {
                "patient_id": xxx,
                "person": particular individual completing the activity (one of 8, 1-indexed),
                "segment_num": 5 second segment of activity (one of 60, 1-indexed),
                "sensor_data": 5625 length vector containing values for 9 sensors among 5
                units for 125 readings at 25 Hz,
                "activity": type of activity being performed (one of 19, 1-indexed)
            }
    """

    task_name: str = "DSAClassification"
    input_schema: Dict[str, str] = {
        "person": "raw",
        "segment_num": "raw",
        "sensor_data": "raw"
    }
    output_schema: Dict[str, str] = {"activity": "multiclass"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a time segment to classify the type of activity.

        Args:
            patient: A patient object containing sensor data.

        Returns:
            List[Dict[str, Any]]: A list of samples with person, segment, and
            sensor data as well as activity type

        Raises:
            AssertionError: If the patient has more than one event.
        """
        event = patient.get_events(event_type="dsa")
        # There should be only one event
        assert len(event) == 1
        event = event[0]

        person = int(event.person)
        segment_num = int(event.segment_num)
        sensor_data = torch.tensor(event.sensor_data.split("|"), dtype=float)
        activity = event.activity
        return [{"person": person, "segment_num": segment_num, "sensor_data": sensor_data, "activity": activity}]

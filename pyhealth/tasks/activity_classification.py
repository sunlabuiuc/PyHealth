"""Activity classification for PyHealth.

This module provides a task for classifying daily and sports activity using
motion sensor data from Daily and Sports Activities (DSA) dataset.
"""

from typing import Any, Dict, List, Optional

from .base_task import BaseTask


class ActivityClassification(BaseTask):
    """Task for classifying activity using motion sensor data.

    This task classifies which activity a patient is performing over a time 
    period from multi-sensor time-series data.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The input schema specifying required inputs.
        output_schema (Dict[str, str]): The output schema specifying outputs.

    Note:
        Each time-series is sampled over 5 seconds with 25 Hz frequency.

    Examples:
        >>> from pyhealth.datasets import DSADataset
        >>> from pyhealth.tasks import ActivityClassification
        >>> dataset = DSADataset(root="/path/to/dsa")
        >>> task = ActivityClassification()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "ActivityClassification"
    input_schema: Dict[str, str] = {
        "features": "sequence",
    }
    output_schema: Dict[str, str] = {
        "label": "text",
    }

    def _extract_features(self, activity: Any) -> List[float]:
        """Extract features from an activity event.

        Args:
            activity: An activity event object.

        Returns:
            List of feature values.
        """
        features: List[float] = []

        columns = [
            f"{x}_{z}{y}" 
            for z in ["x", "y", "z"]
            for y in ["acc", "gyro", "mag"]
            for x in ["T", "RA", "LA", "RL", "LL"]
        ]
        for c in columns:
            x = getattr(activity, c, None)
            features.append(x)
 
        return features

    def _extract_label(self, activity: Any) -> Optional[int]:
        """Extract label for an activity event.

        Args:
            activity: An activity event object.

        Returns:
            1 - 19 to represent activity, None if value is invalid or unknown.
        """
        if activity is None:
            return None

        label = getattr(activity, "activity", None)

        return label

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process daily and sports activities data for activity classification.

        Args:
            patient: A patient object containing activities data.

        Returns:
            List[Dict[str, Any]]: A list containing dictionaries with
            patient features and activity label. 

        Note:
            Returns empty list for patients with:
            - No activities
        """
        activities = patient.get_events(event_type="activities")

        if len(activities) == 0:
            return []

        return [
            {
                "patient_id": patient.patient_id,
                "features": self._extract_features(activity),
                "label": self._extract_label(activity),
            }
            for activity in activities
        ]

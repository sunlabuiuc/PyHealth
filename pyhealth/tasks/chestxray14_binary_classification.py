import logging
from typing import Dict, List

from pyhealth.data.data import Event, Patient
from pyhealth.datasets.chestxray14 import ChestXray14Dataset
from pyhealth.tasks.base_task import BaseTask

logger = logging.getLogger(__name__)

class ChestXray14BinaryClassification(BaseTask):
    """
    A PyHealth task class for binary classification of a specific disease
    in the ChestXray14 dataset.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for the task input.
        output_schema (Dict[str, str]): The schema for the task output.
        disease (str): The disease label to classify.
    """
    task_name: str = "ChestXray14BinaryClassification"
    input_schema: Dict[str, str] = {"image": "image"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(self, disease: str) -> None:
        """
        Initializes the ChestXray14BinaryClassification task with a specified disease.

        Args:
            disease (str): The disease to classify in the binary task. Must be one
                           of the predefined class labels in ChestXray14Dataset.

        Raises:
            ValueError: If the specified disease is not a valid class in the dataset.
        """
        if disease not in ChestXray14Dataset.classes:
            msg = "Invalid disease!"
            logger.error(msg)
            raise ValueError(msg)

        self.disease = disease

    def __call__(self, patient: Patient) -> List[Dict[str, str]]:
        """
        Generates a binary classification data sample for a single patient.

        Args:
            patient (Patient): A patient object containing at least one
                               'chestxray14' event.

        Returns:
            List[Dict[str, str]]: A list containing a single dictionary with:
                - 'image': path to the chest X-ray image.
                - 'label': binary label (as a string) for the specified disease.

        Raises:
            ValueError: If the number of chestxray14 events is not exactly one.
        """
        events: List[Event] = patient.get_events(event_type="chestxray14")
        if len(events) != 1:
            msg = f"Expected just 1 event but got {len(events)}!"
            logger.error(msg)
            raise ValueError(msg)

        return [{"image": events[0]["path"], "label": events[0][self.disease]}]

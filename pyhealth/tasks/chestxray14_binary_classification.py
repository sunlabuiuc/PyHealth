"""
PyHealth task for binary classification using the ChestX-ray14 dataset.

Dataset link:
    https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345

Dataset paper: (please cite if you use this dataset)
    Xiaosong Wang, Yifan Peng, Le Lu, et al. "ChestX-ray8: Hospital-scale Chest
    X-ray Database and Benchmarks on Weakly-Supervised Classification and
    Localization of Common Thorax Diseases." 2017 IEEE Conference on Computer
    Vision and Pattern Recognition (CVPR), pp. 3462-3471.

Dataset paper link:
    https://arxiv.org/abs/1705.02315

Author:
    Eric Schrock (ejs9@illinois.edu)
"""
import logging
from typing import Dict, List

from pyhealth.data import Event, Patient
from pyhealth.tasks import BaseTask

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
        from pyhealth.datasets import ChestXray14Dataset # Avoid circular import
        if disease not in ChestXray14Dataset.classes:
            msg = f"Invalid disease: '{disease}'! Must be one of {ChestXray14Dataset.classes}."
            logger.error(msg)
            raise ValueError(msg)

        self.disease = disease

    def __call__(self, patient: Patient) -> List[Dict]:
        """
        Generates binary classification data samples for a single patient.

        Args:
            patient (Patient): A patient object containing at least one
                               'chestxray14' event.

        Returns:
            List[Dict]: A list containing a dictionary for each patient visit with:
                - 'image': path to the chest X-ray image.
                - 'label': binary label for the specified disease.
        """
        events: List[Event] = patient.get_events(event_type="chestxray14")

        samples = []
        for event in events:
            samples.append({"image": event["path"], "label": int(event[self.disease])})

        return samples

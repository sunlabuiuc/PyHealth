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

from ..data import Event, Patient
from .base_task import BaseTask

logger = logging.getLogger(__name__)

class ChestXray14BinaryClassification(BaseTask):
    """
    A PyHealth task class for binary classification of a specific disease
    in the ChestXray14 dataset.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for the task input.
        output_schema (Dict[str, str]): The schema for the task output.
        classes (List[str]): List of diseases that appear in the dataset.
        disease (str): The disease label to classify.
    """
    task_name: str = "ChestXray14BinaryClassification"
    input_schema: Dict[str, str] = {"image": "image"}
    output_schema: Dict[str, str] = {"label": "binary"}
    classes: List[str] = ["atelectasis", "cardiomegaly", "consolidation",
               "edema", "effusion", "emphysema",
               "fibrosis", "hernia", "infiltration",
               "mass", "nodule", "pleural_thickening",
               "pneumonia", "pneumothorax"]

    def __init__(self, disease: str) -> None:
        """
        Initializes the ChestXray14BinaryClassification task with a specified disease.

        Args:
            disease (str): The disease to classify in the binary task. Must be one
                           of the predefined class labels in ChestXray14Dataset.

        Raises:
            ValueError: If the specified disease is not a valid class in the dataset.
        """
        if disease not in self.classes:
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

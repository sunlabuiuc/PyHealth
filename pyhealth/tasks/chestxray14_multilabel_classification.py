"""
PyHealth task for multilabel classification using the ChestX-ray14 dataset.

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

class ChestXray14MultilabelClassification(BaseTask):
    """
    A PyHealth task class for multilabel classification of all fourteen diseases
    in the ChestXray14 dataset.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for the task input.
        output_schema (Dict[str, str]): The schema for the task output.
    """
    task_name: str = "ChestXray14MultilabelClassification"
    input_schema: Dict[str, str] = {"image": "image"}
    output_schema: Dict[str, str] = {"labels": "multilabel"}

    def __call__(self, patient: Patient) -> List[Dict[str, str]]:
        """
        Generates a multilabel classification data sample for a single patient.

        Args:
            patient (Patient): A patient object containing at least one
                               'chestxray14' event.

        Returns:
            List[Dict[str, str]]: A list containing a single dictionary with:
                - 'image': path to the chest X-ray image.
                - 'labels': a list of binary labels (as strings) for the fourteen classes in ChestXray14Dataset.

        Raises:
            ValueError: If the number of chestxray14 events is not exactly one.
        """
        events: List[Event] = patient.get_events(event_type="chestxray14")
        if len(events) != 1:
            msg = f"Expected just 1 event but got {len(events)}!"
            logger.error(msg)
            raise ValueError(msg)

        from ..datasets.chestxray14 import ChestXray14Dataset # Avoid circular import
        return [{"image": events[0]["path"], "labels": [events[0][disease] for disease in ChestXray14Dataset.classes]}]

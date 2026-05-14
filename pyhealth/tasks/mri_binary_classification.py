"""
PyHealth task for binary classification using the MRI dataset.

Dataset link:
    https://www.kaggle.com/datasets/ninadaithal/oasis-1-shinohara

Dataset paper: (please cite if you use this dataset)
   Open Access Series of Imaging Studies (OASIS): Cross-Sectional MRI Data in Young, 
   Middle Aged, Nondemented, and Demented Older Adults. Marcus, DS, Wang, TH, Parker, 
   J, Csernansky, JG, Morris, JC, Buckner, RL. Journal of Cognitive Neuroscience, 
   19, 1498-1507. doi: 10.1162/jocn.2007.19.9.1498

Dataset paper link:
    https://arxiv.org/abs/1911.03740

Author:
    Soheil Golara and Karan Desai
"""

import logging
from typing import Dict, List

from pyhealth.data import Event, Patient
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)


class MRIBinaryClassification(BaseTask):
    """
    A PyHealth task class for binary classification of alzheimer's
    in the MRI dataset.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for the task input.
        output_schema (Dict[str, str]): The schema for the task output.
        disease (str): The disease label to classify.

    Examples:
        >>> from pyhealth.datasets import MRIDataset
        >>> from pyhealth.tasks import MRIBinaryClassification
        >>> dataset = MRIDataset(root="/path/to/mri")
        >>> task = MRIBinaryClassification(disease="alzheimers")
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "MRIBinaryClassification"
    input_schema: Dict[str, str] = {"image": "image"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(self, disease: str) -> None:
        """
        Initializes the MRIBinaryClassification task with a specified disease.

        Args:
            disease (str): The disease to classify in the binary task. Must be one
                           of the predefined class labels in MRIDataset.

        Raises:
            ValueError: If the specified disease is not a valid class in the dataset.
        """
        from pyhealth.datasets import MRIDataset  # Avoid circular import

        if disease not in MRIDataset.classes:
            msg = f"Invalid disease: '{disease}'! Must be one of {MRIDataset.classes}."
            logger.error(msg)
            raise ValueError(msg)

        self.disease = disease

    def __call__(self, patient: Patient) -> List[Dict]:
        """
        Generates binary classification data samples for a single patient.

        Args:
            patient (Patient): A patient object containing at least one
                               'mri' image.

        Returns:
            List[Dict]: A list containing a dictionary for each patient with:
                - 'image': path to the mri image.
                - 'label': binary label for the specified disease.
        """
        events: List[Event] = patient.get_events(event_type="mri")

        samples = []
        for event in events:
            samples.append({"image": event["path"], "label": int(event[self.disease])})

        return samples

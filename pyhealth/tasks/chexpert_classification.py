from typing import Any, Dict, List

from .base_task import BaseTask


class CheXpertCXRClassification(BaseTask):
    """A task for classifying Pneumonia from chest X-ray images.

    This task classifies chest X-ray images into binary Pneumonia disease category.
    It expects a multiple chest X-ray image per patient and returns the
    corresponding Pnemonia label.

    Attributes:
        task_name (str): The name of the task, set to
            "CheXpertCXRClassification".
        input_schema (Dict[str, str]): The input schema specifying the required
            input format. Contains a single key "image" with value "image".
        output_schema (Dict[str, str]): The output schema specifying the output
            format. Contains a single key "Pneumonia" with value "binary".
    """

    task_name: str = "CheXpertCXRClassification"
    input_schema: Dict[str, str] = {"image": "image"}
    output_schema: Dict[str, str] = {"Pneumonia": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient's chest X-ray data to classify Pneumonia status.

        Args:
            patient: A patient object containing chest X-ray data.

        Returns:
            List[Dict[str, Any]]: A list containing a single dictionary with:
                - "image": Path to the chest X-ray image
                - "Pneumonia": The Pneumonia classification label

        Raises:
            AssertionError: If the patient has more than one chest X-ray event.
        """
        event = patient.get_events(event_type="chexpert_cxr")
        # There should be only one event
        assert len(event) == 1
        event = event[0]
        image = event.Path
        Pneumonia = event.Pneumonia
        return [{"image": image, "Pneumonia": Pneumonia}]

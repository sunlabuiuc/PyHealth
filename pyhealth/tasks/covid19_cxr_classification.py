from typing import Any, Dict, List

from .base_task import BaseTask


class COVID19CXRClassification(BaseTask):
    """A task for classifying chest disease from chest X-ray images.

    This task classifies chest X-ray images into different disease categories.
    It expects a single chest X-ray image per patient and returns the
    corresponding disease label.

    Attributes:
        task_name (str): The name of the task, set to
            "COVID19CXRClassification".
        input_schema (Dict[str, str]): The input schema specifying the required
            input format. Contains a single key "image" with value "image".
        output_schema (Dict[str, str]): The output schema specifying the output
            format. Contains a single key "disease" with value "multiclass".
    """

    task_name: str = "COVID19CXRClassification"
    input_schema: Dict[str, str] = {"image": "image"}
    output_schema: Dict[str, str] = {"disease": "multiclass"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient's chest X-ray data to classify COVID-19 status.

        Args:
            patient: A patient object containing chest X-ray data.

        Returns:
            List[Dict[str, Any]]: A list containing a single dictionary with:
                - "image": Path to the chest X-ray image
                - "disease": The disease classification label

        Raises:
            AssertionError: If the patient has more than one chest X-ray event.
        """
        event = patient.get_events(event_type="covid19_cxr")
        # There should be only one event
        assert len(event) == 1
        event = event[0]
        image = event.path
        disease = event.label
        return [{"image": image, "disease": disease}]

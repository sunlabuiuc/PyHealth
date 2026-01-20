from typing import Any, Dict, List
from pyhealth.tasks import BaseTask

class PH2MelanomaClassification(BaseTask):
    """Task for classifying skin lesions in the PH2 dataset.
    
    Labels:
        - Common Nevus
        - Atypical Nevus
        - Melanoma
    """

    task_name: str = "PH2MelanomaClassification"
    input_schema: Dict[str, str] = {"image": "image"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        events = patient.get_events(event_type="ph2_data")
        samples = []

        for event in events:
            image_path = getattr(event, "path", None)
            if not image_path:
                continue

            # Determine label
            if getattr(event, "melanoma", None) == "X":
                label = "melanoma"
            elif getattr(event, "atypical_nevus", None) == "X":
                label = "atypical_nevus"
            elif getattr(event, "common_nevus", None) == "X":
                label = "common_nevus"
            else:
                continue  # skip images with no label

            samples.append({"image": image_path, "label": label})

        return samples

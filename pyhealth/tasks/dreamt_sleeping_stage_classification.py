from typing import Any, Dict, List

from pyhealth.tasks.base_task import BaseTask
import numpy as np


class DREAMTE4SleepingStageClassification(BaseTask):
    """A task for classifying sleep stages from physiological signals.

    This task processes wearable device (E4) data to classify sleep stages (Sleep or Wake)
    using extracted physiological features. It expects sequential physiological measurements
    and returns corresponding sleep stage labels.

    Attributes:
        task_name (str): The name of the task, set to "DREAMTE4SleepingStageClassification".
        input_schema (Dict[str, str]): The input schema specifying the required
            input format. Contains:
            - "features": "text" (physiological features as numpy array)
        output_schema (Dict[str, str]): The output schema specifying the output
            format. Contains:
            - "label": "binary" (sleep stage classification label)

    Note:
        - Input features include HRV, movement, and other physiological metrics
        - Sleep stages are mapped as: "N1", "N2", "N3", "R": 0, "P", "W":1, "Missing": np.nan
        - Each sample represents a 30-second epoch of sleep data
    """

    task_name: str = "DREAMTE4SleepingStageClassification"
    input_schema: Dict[str, str] = {"features": "text"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient's physiological data to classify sleep stages.

        Args:
            patient: A patient object containing physiological recordings and
                sleep stage annotations. Expected to have events of type
                "dreams_features" containing:
                - sid: Subject ID
                - Sleep_Stage: Annotated sleep stage 
                - Various physiological features (HRV metrics, movement, etc.)
                - Respiratory event markers
                - Demographic information

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing:
                - "patient_id": Unique patient identifier
                - "features": Physiological features as numpy array (shape 1×N)
                - "label": Sleep stage classification 0 or 1 

        """
        samples = []
        cols_to_remove = [
            "sid", "Sleep_Stage", "Central_Apnea", "Obstructive_Apnea", 
            "Multiple_Events", "Hypopnea", 'AHI_Severity', 'Obesity', 
            'BMI', "circadian_decay", "circadian_linear", "circadian_cosine",
            "timestamp_start"
        ]
        
        records = patient.get_events(event_type="dreamt_features")
        for record in records:
            attr = record.attr_dict
            features = [float(v) for k, v in attr.items() if k not in cols_to_remove]
            
            sample = {
                "patient_id": patient.patient_id[0],
                "features": np.array(features).reshape(1, -1),
                "label": int(float(attr["Sleep_Stage"])),
            }
            samples.append(sample)

        return samples


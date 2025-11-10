from typing import Any, Dict, List
from .base_task import BaseTask


class BMDHSDiseaseClassification(BaseTask):
    """Multi-label classification task for heart valve diseases.

    This task classifies heart sound recordings into multiple disease categories:
    - AS (Aortic Stenosis)
    - AR (Aortic Regurgitation)
    - MR (Mitral Regurgitation)
    - MS (Mitral Stenosis)

    Each patient can have multiple diseases simultaneously (multi-label).

    The task also provides access to patient metadata including:
    - Age
    - Gender
    - Smoker status (binary)
    - Living environment (urban/rural, binary)

    And 8 heart sound recording filenames.

    Attributes:
        task_name (str): The name of the task, set to "BMDHSDiseaseClassification".
        input_schema (Dict[str, str]): The input schema specifying the required
            input format. Contains:
            - "recording_1" through "recording_8": "audio"
            - "age": "float"
            - "gender": "categorical"
            - "smoker": "binary"
            - "lives": "binary"
        output_schema (Dict[str, str]): The output schema specifying the output
            format. Contains:
            - "diagnosis": "multilabel"
    """

    task_name: str = "BMDHSDiseaseClassification"

    input_schema: Dict[str, str] = {
        # 8 heart sound recordings
        **{f"recording_{i}": "audio" for i in range(1, 9)},
        # Patient metadata
        "age": "regression",
        "gender": "binary",
        "smoker": "binary",
        "lives": "binary",
    }

    output_schema: Dict[str, str] = {
        # Multi-label disease classification
        "diagnosis": "multilabel",
    }

    def __init__(self) -> None:
        try:
            import soundfile
        except ImportError:
            raise ImportError(
                "SoundFile library is required for BMDHSDiseaseClassification task to read .wav files. "
                "Install it with: pip install soundfile"
            )
        super().__init__()

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient's data to extract features and labels.

        Args:
            patient: A patient object containing BMD-HS data.

        Returns:
            List[Dict[str, Any]]: A list containing a single dictionary with:
                - "recording_1" through "recording_8": Paths to recordings
                - "age": Patient age
                - "gender": Patient gender
                - "smoker": Smoking status (binary)
                - "lives": Living environment (binary)
                - "diagnosis": A dictionary with disease labels (AS, AR, MR, MS)

        Raises:
            AssertionError: If the patient has more than one event.
        """

        pid = patient.patient_id

        # Extract recording filenames
        rec_event = patient.get_events(event_type="recordings")[0]
        recordings = {
            f"recording_{i}": rec_event[f"recording_{i}"] for i in range(1, 9)
        }

        # Extract metadata
        meta_event = patient.get_events(event_type="metadata")[0]
        metadata = {
            "age": float(meta_event["Age"]),
            "gender": meta_event["Gender"],
            "smoker": meta_event["Smoker"],
            "lives": meta_event["Lives"],
        }

        # Extract disease labels (multi-label)
        diag_event = patient.get_events(event_type="diagnoses")[0]
        labels = {
            "diagnosis": [
                dis for dis in ["AS", "AR", "MR", "MS"] if int(diag_event[dis])
            ]
        }
        # Combine all features and labels
        sample = {"patient_id": pid, **recordings, **metadata, **labels}

        return [sample]

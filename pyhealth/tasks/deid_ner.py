"""
PyHealth task for NER-based de-identification of clinical text.

Converts PhysioNet De-Identification dataset records into token-level
BIO-tagged NER samples for PHI detection.

Dataset link:
    https://physionet.org/content/deidentifiedmedicaltext/1.0/

Task paper: (please cite if you use this task)
    Johnson, Alistair E.W., et al. "Deidentification of free-text medical
    records using pre-trained bidirectional transformers." Proceedings of
    the ACM Conference on Health, Inference, and Learning (CHIL), 2020.

Paper link:
    https://doi.org/10.1145/3368555.3384455

Author:
    Matt McKenna (mtm16@illinois.edu)
"""

from typing import Dict, List, Optional, Type, Union

from pyhealth.data import Event, Patient
from pyhealth.processors.text_processor import TextProcessor
from pyhealth.tasks import BaseTask


class DeIDNERTask(BaseTask):
    """Token-level NER task for clinical text de-identification.

    Each sample contains a list of tokens and their BIO labels over
    7 PHI categories: AGE, CONTACT, DATE, ID, LOCATION, NAME,
    PROFESSION.

    Supports optional overlapping windowing (paper Section 3.3) to
    handle notes longer than BERT's 512 token limit.

    Args:
        window_size: If set, split notes into overlapping windows of
            this many tokens. Default None (no windowing).
        window_overlap: Number of tokens shared between consecutive
            windows. Default 0.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, Union[str, Type]]): The schema for the task input.
        output_schema (Dict[str, Union[str, Type]]): The schema for the task output.

    Examples:
        >>> from pyhealth.datasets import PhysioNetDeIDDataset
        >>> from pyhealth.tasks import DeIDNERTask
        >>> dataset = PhysioNetDeIDDataset(root="/path/to/data")
        >>> task = DeIDNERTask()
        >>> samples = dataset.set_task(task)
        >>> task_windowed = DeIDNERTask(window_size=100, window_overlap=60)
        >>> samples = dataset.set_task(task_windowed)
    """

    task_name: str = "DeIDNER"
    input_schema: Dict[str, Union[str, Type]] = {"text": TextProcessor}
    output_schema: Dict[str, Union[str, Type]] = {"labels": TextProcessor}

    def __init__(
        self,
        window_size: Optional[int] = None,
        window_overlap: int = 0,
    ):
        self.window_size = window_size
        self.window_overlap = window_overlap

    def __call__(self, patient: Patient) -> List[Dict]:
        """Generate NER samples from a patient's clinical notes.

        Args:
            patient: A Patient object with physionet_deid events.

        Returns:
            List of dicts, each with 'text' (str) and
            'labels' (str) keys. Both are space-joined strings.
        """
        events: List[Event] = patient.get_events(
            event_type="physionet_deid"
        )

        samples = []
        for event in events:
            note_id = event["note_id"]
            words = event["text"].split(" ")
            labels = event["labels"].split(" ")

            if self.window_size is None:
                # No windowing: one sample per note.
                samples.append({
                    "patient_id": patient.patient_id,
                    "note_id": note_id,
                    "token_start": "0",
                    "text": event["text"],
                    "labels": event["labels"],
                })
            else:
                # Overlapping windows (paper Section 3.3).
                step = self.window_size - self.window_overlap
                idx = 0
                while idx < len(words):
                    end = min(idx + self.window_size, len(words))
                    samples.append({
                        "patient_id": patient.patient_id,
                        "note_id": note_id,
                        "token_start": str(idx),
                        "text": " ".join(words[idx:end]),
                        "labels": " ".join(labels[idx:end]),
                    })
                    idx += step

        return samples

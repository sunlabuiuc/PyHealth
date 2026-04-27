"""Audio Question Answering tasks for PyHealth.

This module provides tasks for processing generative text answers
based on medical audio signals using the CaReSound dataset.
"""

from typing import Any, Dict, List
from .base_task import BaseTask


class CaReSoundAQA(BaseTask):
    """Task for Audio Question Answering on respiratory and cardiac sounds.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): Required inputs (audio_path, question).
        output_schema (Dict[str, str]): Required outputs (answer).

    Example:
        >>> from pyhealth.datasets import CaReSoundDataset
        >>> from pyhealth.tasks import CaReSoundAQA
        >>> example_dataset = CaReSoundDataset(root="/Users/rahuld/Downloads/CaReAQA/datasets")
        >>> sample_dataset = example_dataset.set_task(CaReSoundAQA())
    """

    task_name: str = "CaReSoundAQA"
    input_schema: Dict[str, str] = {
        "question": "text",
    }
    output_schema: Dict[str, str] = {"answer": "text"}

    def _safe_str(self, value: Any, default: str = "") -> str:
        """Safely convert value to string, handling None and NaN."""
        if value is None or str(value).lower() == "nan":
            return default
        return str(value)

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient record to extract audio-QA samples."""
        samples: List[Dict[str, Any]] = []
        events = patient.get_events("metadata")

        for event in events:
            # The new engine puts CSV columns into attr_dict
            attr = getattr(event, "attr_dict", {})

            audio_path = str(attr.get("audio_path", ""))
            question = str(attr.get("question", ""))
            answer = str(attr.get("answer", ""))
            hf_split = str(attr.get("hf_split", "unknown"))

            if not audio_path or not question or not answer:
                continue

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": f"v_{patient.patient_id}",
                    "audio_path": audio_path,
                    "question": question,
                    "answer": answer,
                    "original_hf_split": hf_split,
                }
            )

        return samples

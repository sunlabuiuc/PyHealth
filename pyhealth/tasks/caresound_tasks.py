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
    """

    task_name: str = "CaReSoundAQA"
    input_schema: Dict[str, str] = {
        "audio_path": "path",
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

        for visit_id, visit in patient.visits.items():
            audio_path = visit.attr_dict.get("audio_path") if hasattr(visit, "attr_dict") else visit.get("audio_path")
            
            if not audio_path:
                continue
                
            events = visit.attr_dict.get("events", []) if hasattr(visit, "attr_dict") else visit.get("events", [])
            
            for event in events:
                question = self._safe_str(event.get("question"))
                answer = self._safe_str(event.get("answer"))
                hf_split = self._safe_str(event.get("hf_split"), default="unknown")
                
                if not question or not answer:
                    continue

                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "visit_id": visit_id,
                        "audio_path": audio_path,
                        "question": question,
                        "answer": answer,
                        "original_hf_split": hf_split,
                    }
                )

        return samples
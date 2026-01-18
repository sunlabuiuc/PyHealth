from typing import Dict, List, Optional, Sequence, Set

import torch

from .base_task import BaseTask
from .sdoh_utils import TARGET_CODES, codes_to_multihot


class SDOHICD9AdmissionTask(BaseTask):
    """Builds admission-level samples for SDOH ICD-9 V-code detection."""

    task_name: str = "SDOHICD9Admission"
    input_schema: Dict[str, str] = {
        "notes": "raw",
        "note_categories": "raw",
        "chartdates": "raw",
        "patient_id": "raw",
        "visit_id": "raw",
    }
    output_schema: Dict[str, object] = {
        "label": ("tensor", {"dtype": torch.float32}),
    }

    def __init__(
        self,
        target_codes: Optional[Sequence[str]] = None,
        label_source: str = "manual",
    ) -> None:
        self.target_codes = list(target_codes) if target_codes else list(TARGET_CODES)
        if label_source not in {"manual", "true"}:
            raise ValueError("label_source must be 'manual' or 'true'")
        self.label_source = label_source

    def __call__(self, admission: Dict) -> List[Dict]:
        if self.label_source == "manual":
            label_codes: Set[str] = admission.get("manual_codes", set())
        else:
            label_codes = admission.get("true_codes", set())

        sample = dict(admission)
        sample["label_codes"] = sorted(label_codes)
        sample["label"] = codes_to_multihot(label_codes, self.target_codes)
        return [sample]

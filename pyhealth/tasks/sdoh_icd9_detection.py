from typing import Dict, List, Optional, Sequence, Set

import torch

from .base_task import BaseTask
from .sdoh_utils import TARGET_CODES, codes_to_multihot


class SDOHICD9AdmissionTask(BaseTask):
    """Builds admission-level samples for SDOH ICD-9 V-code detection.

    The task attaches a multi-hot label vector and the corresponding label
    codes based on the provided admission dictionary (and optional label_map).
    """

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
        label_map: Optional[Dict[str, Dict[str, Set[str]]]] = None,
    ) -> None:
        """Initialize the admission-level task.

        Args:
            target_codes: Target ICD-9 codes for label vector construction.
            label_source: Which label set to use ("manual" or "true").
            label_map: Optional mapping of HADM_ID to label codes.
        """
        self.target_codes = list(target_codes) if target_codes else list(TARGET_CODES)
        if label_source not in {"manual", "true"}:
            raise ValueError("label_source must be 'manual' or 'true'")
        self.label_source = label_source
        self.label_map = label_map or {}

    def __call__(self, admission: Dict) -> List[Dict]:
        """Build a single labeled sample from an admission dictionary."""
        admission_id = str(admission.get("visit_id", ""))
        if admission_id and admission_id in self.label_map:
            admission = dict(admission)
            admission.setdefault("manual_codes", self.label_map[admission_id]["manual"])
            admission.setdefault("true_codes", self.label_map[admission_id]["true"])

        if self.label_source == "manual":
            label_codes: Set[str] = admission.get("manual_codes", set())
        else:
            label_codes = admission.get("true_codes", set())

        sample = dict(admission)
        sample["label_codes"] = sorted(label_codes)
        sample["label"] = codes_to_multihot(label_codes, self.target_codes)
        return [sample]

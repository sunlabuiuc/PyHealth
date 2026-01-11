import logging
from typing import Dict, List, Optional, Sequence, Set

import pandas as pd
import torch

from ..data import Event, Patient
from .base_task import BaseTask
from .sdoh_utils import TARGET_CODES, codes_to_multihot, parse_codes

logger = logging.getLogger(__name__)


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

        sample = {
            "visit_id": admission["visit_id"],
            "patient_id": admission["patient_id"],
            "notes": admission["notes"],
            "note_categories": admission["note_categories"],
            "chartdates": admission["chartdates"],
            "num_notes": admission.get("num_notes", len(admission["notes"])),
            "text_length": admission.get("text_length", 0),
            "is_gap_case": admission.get("is_gap_case"),
            "manual_codes": admission.get("manual_codes", set()),
            "true_codes": admission.get("true_codes", set()),
            "label_codes": sorted(label_codes),
            "label": codes_to_multihot(label_codes, self.target_codes),
        }
        return [sample]


def load_sdoh_icd9_labels(
    csv_path: str, target_codes: Sequence[str]
) -> Dict[str, Dict[str, Set[str]]]:
    df = pd.read_csv(csv_path)
    if "HADM_ID" not in df.columns:
        raise ValueError("CSV must include HADM_ID column.")

    labels: Dict[str, Dict[str, Set[str]]] = {}
    for hadm_id, group in df.groupby("HADM_ID"):
        first = group.iloc[0]
        labels[str(hadm_id)] = {
            "manual": parse_codes(
                first.get("ADMISSION_MANUAL_LABELS"), target_codes
            ),
            "true": parse_codes(
                first.get("ADMISSION_TRUE_CODES"), target_codes
            ),
        }
    return labels


class SDOHICD9MIMIC3NoteTask(BaseTask):
    """Builds admission-level samples from MIMIC-III noteevents with CSV labels."""

    task_name: str = "SDOHICD9MIMIC3Notes"
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
        label_csv_path: str,
        target_codes: Optional[Sequence[str]] = None,
        label_source: str = "manual",
    ) -> None:
        self.target_codes = list(target_codes) if target_codes else list(TARGET_CODES)
        if label_source not in {"manual", "true"}:
            raise ValueError("label_source must be 'manual' or 'true'")
        self.label_source = label_source
        self.label_map = load_sdoh_icd9_labels(label_csv_path, self.target_codes)

    def __call__(self, patient: Patient) -> List[Dict]:
        notes: List[Event] = patient.get_events(event_type="noteevents")
        if not notes:
            return []

        by_hadm: Dict[str, List[Event]] = {}
        for event in notes:
            hadm_id = str(event.hadm_id)
            if hadm_id not in self.label_map:
                continue
            by_hadm.setdefault(hadm_id, []).append(event)

        samples: List[Dict] = []
        for hadm_id, events in by_hadm.items():
            events.sort(key=lambda e: e.timestamp or "")
            note_texts = [str(e.text) if e.text is not None else "" for e in events]
            note_categories = [str(e.category) if e.category is not None else "" for e in events]
            chartdates = [
                e.timestamp.strftime("%Y-%m-%d") if e.timestamp is not None else "Unknown"
                for e in events
            ]

            label_codes = self.label_map[hadm_id][self.label_source]
            sample = {
                "visit_id": hadm_id,
                "patient_id": patient.patient_id,
                "notes": note_texts,
                "note_categories": note_categories,
                "chartdates": chartdates,
                "num_notes": len(note_texts),
                "text_length": int(sum(len(note) for note in note_texts)),
                "manual_codes": self.label_map[hadm_id]["manual"],
                "true_codes": self.label_map[hadm_id]["true"],
                "label_codes": sorted(label_codes),
                "label": codes_to_multihot(label_codes, self.target_codes),
            }
            samples.append(sample)

        return samples

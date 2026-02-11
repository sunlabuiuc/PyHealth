import logging
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .sample_dataset import SampleDataset, create_sample_dataset
from ..tasks.sdoh_icd9_detection import SDOHICD9AdmissionTask
from ..tasks.sdoh_utils import TARGET_CODES, parse_codes

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = {
    "HADM_ID",
    "SUBJECT_ID",
    "NOTE_CATEGORY",
    "CHARTDATE",
    "FULL_TEXT",
    "ADMISSION_TRUE_CODES",
    "ADMISSION_MANUAL_LABELS",
}


class SDOHICD9NotesDataset:
    """CSV-backed dataset for SDOH ICD-9 V-code detection from notes."""

    def __init__(
        self,
        csv_path: str,
        dataset_name: Optional[str] = None,
        target_codes: Optional[Sequence[str]] = None,
        include_categories: Optional[Sequence[str]] = None,
    ) -> None:
        self.csv_path = csv_path
        self.dataset_name = dataset_name or "sdoh_icd9_notes"
        self.target_codes = list(target_codes) if target_codes else list(TARGET_CODES)
        self.include_categories = (
            {cat.strip().upper() for cat in include_categories}
            if include_categories
            else None
        )
        self._admissions = self._load_admissions()

    def _load_admissions(self) -> List[Dict]:
        df = pd.read_csv(self.csv_path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        admissions: List[Dict] = []
        df["CHARTDATE"] = pd.to_datetime(df["CHARTDATE"], errors="coerce")
        if self.include_categories is not None:
            df = df[
                df["NOTE_CATEGORY"].astype("string")
                .str.upper()
                .isin(self.include_categories)
            ]
        for hadm_id, group in df.groupby("HADM_ID"):
            group = group.sort_values("CHARTDATE")
            first = group.iloc[0]
            notes = [str(text).strip() for text in group["FULL_TEXT"].fillna("")]
            chartdates = [
                dt.strftime("%Y-%m-%d") if pd.notna(dt) else "Unknown"
                for dt in group["CHARTDATE"]
            ]
            admission = {
                "visit_id": str(hadm_id),
                "patient_id": str(first["SUBJECT_ID"]),
                "is_gap_case": first.get("IS_GAP_CASE"),
                "note_categories": [
                    str(cat).strip() for cat in group["NOTE_CATEGORY"].fillna("")
                ],
                "chartdates": chartdates,
                "notes": notes,
                "num_notes": len(notes),
                "text_length": int(sum(len(note) for note in notes)),
                "manual_codes": parse_codes(
                    first["ADMISSION_MANUAL_LABELS"], self.target_codes
                ),
                "true_codes": parse_codes(
                    first["ADMISSION_TRUE_CODES"], self.target_codes
                ),
            }
            admissions.append(admission)
        logger.info("Loaded %d admissions from %s", len(admissions), self.csv_path)
        return admissions

    def set_task(
        self,
        task: Optional[SDOHICD9AdmissionTask] = None,
        label_source: str = "manual",
        in_memory: bool = True,
    ) -> SampleDataset:
        if task is None:
            task = SDOHICD9AdmissionTask(
                target_codes=self.target_codes,
                label_source=label_source,
            )

        samples: List[Dict] = []
        for admission in self._admissions:
            samples.extend(task(admission))

        return create_sample_dataset(
            samples=samples,
            input_schema=task.input_schema,
            output_schema=task.output_schema,
            dataset_name=self.dataset_name,
            task_name=task.task_name,
            in_memory=in_memory,
        )

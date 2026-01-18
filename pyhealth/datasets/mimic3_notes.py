import logging
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .sample_dataset import SampleDataset, create_sample_dataset
from ..tasks.sdoh_icd9_detection import SDOHICD9AdmissionTask
from ..tasks.sdoh_utils import TARGET_CODES, load_sdoh_icd9_labels

logger = logging.getLogger(__name__)


class MIMIC3NotesDataset:
    """Note-only loader for MIMIC-III NOTEEVENTS with label CSV filtering."""

    def __init__(
        self,
        noteevents_path: str,
        label_csv_path: str,
        target_codes: Optional[Sequence[str]] = None,
        hadm_ids: Optional[Iterable[str]] = None,
        include_categories: Optional[Sequence[str]] = None,
        chunksize: int = 200_000,
        dataset_name: Optional[str] = None,
    ) -> None:
        self.noteevents_path = noteevents_path
        self.target_codes = list(target_codes) if target_codes else list(TARGET_CODES)
        self.label_map = load_sdoh_icd9_labels(label_csv_path, self.target_codes)
        if hadm_ids is None:
            hadm_ids = self.label_map.keys()
        self.hadm_ids = {str(x) for x in hadm_ids}
        self.include_categories = (
            {cat.strip().upper() for cat in include_categories}
            if include_categories
            else None
        )
        self.chunksize = chunksize
        self.dataset_name = dataset_name or "mimic3_notes"

    def _load_notes(self) -> Dict[str, List[Dict]]:
        keep_cols = {
            "SUBJECT_ID",
            "HADM_ID",
            "CHARTDATE",
            "CHARTTIME",
            "CATEGORY",
            "TEXT",
        }

        notes_by_hadm: Dict[str, List[Dict]] = {}
        for chunk in pd.read_csv(
            self.noteevents_path,
            chunksize=self.chunksize,
            usecols=lambda c: c.upper() in keep_cols,
            dtype={"SUBJECT_ID": "string", "HADM_ID": "string"},
            low_memory=False,
        ):
            chunk.columns = [c.upper() for c in chunk.columns]
            filtered = chunk[chunk["HADM_ID"].astype("string").isin(self.hadm_ids)]
            if filtered.empty:
                continue
            if self.include_categories is not None:
                filtered = filtered[
                    filtered["CATEGORY"].astype("string")
                    .str.upper()
                    .isin(self.include_categories)
                ]
                if filtered.empty:
                    continue

            charttime = pd.to_datetime(filtered["CHARTTIME"], errors="coerce")
            chartdate = pd.to_datetime(filtered["CHARTDATE"], errors="coerce")
            timestamp = charttime.fillna(chartdate)

            for row, ts in zip(filtered.itertuples(index=False), timestamp):
                hadm_id = str(row.HADM_ID)
                entry = {
                    "patient_id": str(row.SUBJECT_ID) if pd.notna(row.SUBJECT_ID) else "",
                    "text": row.TEXT if pd.notna(row.TEXT) else "",
                    "category": row.CATEGORY if pd.notna(row.CATEGORY) else "",
                    "timestamp": ts,
                }
                notes_by_hadm.setdefault(hadm_id, []).append(entry)

        return notes_by_hadm

    def _build_admissions(self) -> List[Dict]:
        notes_by_hadm = self._load_notes()
        admissions: List[Dict] = []
        for hadm_id, notes in notes_by_hadm.items():
            if hadm_id not in self.label_map:
                continue
            notes.sort(
                key=lambda x: x["timestamp"]
                if pd.notna(x["timestamp"])
                else pd.Timestamp.min
            )
            note_texts = [str(n["text"]) for n in notes]
            note_categories = [str(n["category"]) for n in notes]
            chartdates = [
                n["timestamp"].strftime("%Y-%m-%d") if pd.notna(n["timestamp"]) else "Unknown"
                for n in notes
            ]

            labels = self.label_map[hadm_id]
            admissions.append(
                {
                    "visit_id": hadm_id,
                    "patient_id": notes[0]["patient_id"],
                    "notes": note_texts,
                    "note_categories": note_categories,
                    "chartdates": chartdates,
                    "num_notes": len(note_texts),
                    "text_length": int(sum(len(note) for note in note_texts)),
                    "manual_codes": labels["manual"],
                    "true_codes": labels["true"],
                }
            )

        logger.info("Loaded %d admissions from NOTEEVENTS", len(admissions))
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
        for admission in self._build_admissions():
            samples.extend(task(admission))

        return create_sample_dataset(
            samples=samples,
            input_schema=task.input_schema,
            output_schema=task.output_schema,
            dataset_name=self.dataset_name,
            task_name=task.task_name,
            in_memory=in_memory,
        )

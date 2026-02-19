import logging
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import narwhals as pl
import pandas as pd

from .base_dataset import BaseDataset
from .sample_dataset import SampleDataset, create_sample_dataset
from ..tasks.sdoh_icd9_detection import SDOHICD9AdmissionTask
from ..tasks.sdoh_utils import TARGET_CODES

logger = logging.getLogger(__name__)


class MIMIC3Dataset(BaseDataset):
    """
    A dataset class for handling MIMIC-III data.

    This class is responsible for loading and managing the MIMIC-III dataset,
    which includes tables such as patients, admissions, and icustays.

    Attributes:
        root (str): The root directory where the dataset is stored.
        tables (List[str]): A list of tables to be included in the dataset.
        dataset_name (Optional[str]): The name of the dataset.
        config_path (Optional[str]): The path to the configuration file.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> # Load MIMIC-III dataset with clinical tables
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["diagnoses_icd", "procedures_icd", "labevents"],
        ... )
        >>> dataset.stats()
    """

    def __init__(
        self,
        root: str,
        tables: List[str],
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initializes the MIMIC4Dataset with the specified parameters.

        Args:
            root (str): The root directory where the dataset is stored.
            tables (List[str]): A list of additional tables to include.
            dataset_name (Optional[str]): The name of the dataset. Defaults to "mimic3".
            config_path (Optional[str]): The path to the configuration file. If not provided, a default config is used.
        """
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "mimic3.yaml"
        default_tables = ["patients", "admissions", "icustays"]
        tables = default_tables + tables
        if "prescriptions" in tables:
            warnings.warn(
                "Events from prescriptions table only have date timestamp (no specific time). "
                "This may affect temporal ordering of events.",
                UserWarning,
            )
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "mimic3",
            config_path=config_path,
            **kwargs,
        )
        return

    def preprocess_noteevents(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Table-specific preprocess function which will be called by BaseDataset.load_table().

        Preprocesses the noteevents table by ensuring that the charttime column
        is populated. If charttime is null, it uses chartdate with a default
        time of 00:00:00.

        See: https://mimic.mit.edu/docs/iii/tables/noteevents/#chartdate-charttime-storetime.

        Args:
            df (pl.LazyFrame): The input dataframe containing noteevents data.

        Returns:
            pl.LazyFrame: The processed dataframe with updated charttime
            values.
        """
        df = df.with_columns(
            pl.when(pl.col("charttime").is_null())
            .then(pl.col("chartdate") + pl.lit(" 00:00:00"))
            .otherwise(pl.col("charttime"))
            .alias("charttime")
        )
        return df


class MIMIC3NoteDataset:
    """Note-only loader for MIMIC-III NOTEEVENTS.

    This loader streams NOTEEVENTS in chunks and optionally filters by HADM_IDs
    and note categories. Use set_task() to convert admissions into samples.
    """

    def __init__(
        self,
        noteevents_path: Optional[str] = None,
        root: Optional[str] = None,
        target_codes: Optional[Sequence[str]] = None,
        hadm_ids: Optional[Iterable[str]] = None,
        include_categories: Optional[Sequence[str]] = None,
        chunksize: int = 200_000,
        dataset_name: Optional[str] = None,
    ) -> None:
        """Initialize the note-only loader.

        Args:
            noteevents_path: Path to NOTEEVENTS CSV/CSV.GZ.
            root: MIMIC-III root directory (used if noteevents_path is None).
            target_codes: Target ICD-9 codes for label vector construction.
            hadm_ids: Optional admission IDs to include.
            include_categories: Optional NOTE_CATEGORY values to include.
            chunksize: Number of rows per chunk read.
            dataset_name: Optional dataset name for the SampleDataset.
        """
        if noteevents_path is None:
            if root is None:
                raise ValueError("root is required when noteevents_path is not set.")
            noteevents_path = str(Path(root) / "NOTEEVENTS.csv.gz")
        self.noteevents_path = noteevents_path
        self.target_codes = list(target_codes) if target_codes else list(TARGET_CODES)
        self.hadm_ids = {str(x) for x in hadm_ids} if hadm_ids is not None else None
        self.include_categories = (
            {cat.strip().upper() for cat in include_categories}
            if include_categories
            else None
        )
        self.chunksize = chunksize
        self.dataset_name = dataset_name or "mimic3_note"

    def _load_notes(self) -> Dict[str, List[Dict]]:
        """Load and group note events by admission."""
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
            if self.hadm_ids is not None:
                chunk = chunk[chunk["HADM_ID"].astype("string").isin(self.hadm_ids)]
                if chunk.empty:
                    continue
            if self.include_categories is not None:
                chunk = chunk[
                    chunk["CATEGORY"].astype("string")
                    .str.upper()
                    .isin(self.include_categories)
                ]
                if chunk.empty:
                    continue

            charttime = pd.to_datetime(chunk["CHARTTIME"], errors="coerce")
            chartdate = pd.to_datetime(chunk["CHARTDATE"], errors="coerce")
            timestamp = charttime.fillna(chartdate)

            for row, ts in zip(chunk.itertuples(index=False), timestamp):
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
        """Build admission-level note bundles with timestamps and categories."""
        notes_by_hadm = self._load_notes()
        admissions: List[Dict] = []
        for hadm_id, notes in notes_by_hadm.items():
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

            admissions.append(
                {
                    "visit_id": hadm_id,
                    "patient_id": notes[0]["patient_id"],
                    "notes": note_texts,
                    "note_categories": note_categories,
                    "chartdates": chartdates,
                    "num_notes": len(note_texts),
                    "text_length": int(sum(len(note) for note in note_texts)),
                }
            )

        logger.info("Loaded %d admissions from NOTEEVENTS", len(admissions))
        return admissions

    def set_task(
        self,
        task: Optional[SDOHICD9AdmissionTask] = None,
        label_source: str = "manual",
        label_map: Optional[Dict[str, Dict[str, Set[str]]]] = None,
        in_memory: bool = True,
    ) -> SampleDataset:
        """Apply a task to admissions and return a SampleDataset."""
        if task is None:
            task = SDOHICD9AdmissionTask(
                target_codes=self.target_codes,
                label_source=label_source,
                label_map=label_map,
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

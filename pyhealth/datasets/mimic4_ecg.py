import os
from typing import List, Optional
import pandas as pd

from pyhealth.datasets import BaseDataset


class MIMIC4ECGDataset(BaseDataset):
    """
    MIMIC-IV ECG dataset.

    This dataset loader manages the MIMIC-IV ECG tables:
    - machine_measurements
    - record_list
    - waveform_note_links

    It extends BaseDataset so that schema, splits, and config-driven loading
    are handled consistently with other PyHealth datasets.
    """

    def __init__(
        self,
        root: str,
        tables: Optional[List[str]] = None,
        dataset_name: str = "mimic4_ecg",
        config_path: Optional[str] = None,
        **kwargs,
    ):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "mimic4_ecg.yaml"
            )

        # ECG-specific tables
        default_tables = [
            "machine_measurements",
            "record_list",
            "waveform_note_links",
        ]
        if tables is None:
            tables = default_tables
        else:
            tables = tables + default_tables

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name,
            config_path=config_path,
            **kwargs,
        )
        
        self.data = {}

        for table in self.tables:
            path = os.path.join(root, "ecg", f"{table}.csv")
            if os.path.exists(path):
                self.data[table] = pd.read_csv(path)
            else:
                raise FileNotFoundError(f"Neither path exists: {path}")
        
    def __len__(self):
        # Use record_list as the canonical length
        return len(self.data["record_list"])

    def __getitem__(self, idx: int):
        record = self.data["record_list"].iloc[idx]
        subject_id, study_id = record["subject_id"], record["study_id"]

        mm = self.data["machine_measurements"].query(
            "subject_id == @subject_id and study_id == @study_id"
        )
        notes = self.data["waveform_note_links"].query(
            "subject_id == @subject_id and study_id == @study_id"
        )

        return {
            "subject_id": subject_id,
            "study_id": study_id,
            "record": record.to_dict(),
            "machine_measurements": mm.to_dict(orient="records"),
            "notes": notes.to_dict(orient="records"),
        }

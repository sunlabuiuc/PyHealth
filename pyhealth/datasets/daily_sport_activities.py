from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any

import io
import zipfile

import dask.dataframe as dd
import numpy as np
import pandas as pd
import requests

from pyhealth.datasets import BaseDataset


class DailyAndSportActivitiesDataset(BaseDataset):
    """PyHealth dataset for Daily and Sports Activities data.

    This dataset parses multi-sensor time-series text files into structured
    samples suitable for downstream task processing.

    Expected folder layout example:
        root/
          activity_01/
            subject_01/
              segment_01.txt
              segment_02.txt
          activity_02/
            subject_01/
              segment_01.txt

    Each .txt file is expected to contain numeric sensor values arranged
    row-wise over time.

    Parsed sample format:
        {
            "record_id": str,
            "patient_id": str,
            "activity": str,
            "activity_id": int,
            "segment_id": str,
            "signal": np.ndarray,   # shape: [time_steps, num_features]
        }
    """

    activities: List[str] = ["sitting", "standing", "lying on back", "lying on right side",
                             "ascending stairs", "descending stairs", "standing still in elevator",
                             "moving around in elevator", "walking in parking lot",
                             "walking on 4km/h treadmill 0 incline", "walking on 4km/h treadmill 15 incline",
                             "running on 8km/h treadmill", "using stair stepper", "using cross trainer",
                             "cycling in horizontal position", "cycling in vertical position",
                             "rowing", "jumping", "playing basketball"]

    def __init__(
        self,
        root: str = ".",
        config_path: Optional[str] = str(Path(__file__).parent / "configs" / "daily_sport_activities.yaml"),
        download: bool = False,
        dev: bool = False,
    ):
        self.root_path = Path(root)
        self.root = root

        if download:
            self._download(self.root)

        if not self.root_path.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root_path}")

        if not self.root_path.is_dir():
            raise NotADirectoryError(f"Dataset root is not a directory: {self.root_path}")

        super().__init__(
            root=self.root,
            tables=["daily_sport_activities"],
            dataset_name="daily_sport_activities",
            config_path=config_path,
            dev=dev,
        )     

    @property
    def default_task(self):
        from pyhealth.tasks.daily_sport_activities import DailyAndSportActivitiesTask
        return DailyAndSportActivitiesTask(signal_loader=self.load_signal)
    
    def _download(self, root: str) -> None:
        """Downloads the Daily and Sports Activities dataset and extracts the compressed data.
        
        Args:
            root (str): Root directory of raw data.
        Raises:
            HTTPError: If the file cannot be downloaded.
        """

        dataset_url = "https://archive.ics.uci.edu/static/public/256/daily+and+sports+activities.zip"

        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)

        response = requests.get(dataset_url, timeout=60)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            zf.extractall(root_path)

    def _discover_files(self) -> List[Path]:
        """Find all text files under the dataset root."""
        txt_files = sorted(self.root_path.rglob("*.txt"))

        if not txt_files:
            raise FileNotFoundError(
                f"No .txt files found under dataset root: {root_path.root}"
            )

        return txt_files

    def _infer_metadata_from_path(self, file_path: Path) -> Dict[str, str]:
        """Infer activity, subject, and segment identifiers from the file path."""
        relative_parts = file_path.relative_to(self.root_path).parts

        if len(relative_parts) < 3:
            raise ValueError(
                f"Unexpected file structure for {file_path}. "
                f"Expected at least activity/subject/file.txt"
            )

        activity_id = relative_parts[-3]
        patient_id = relative_parts[-2]
        segment_id = file_path.stem
        record_id = f"{patient_id}_{activity_id}_{segment_id}"

        if not activity_id.startswith("a"):
            raise ValueError(f"Invalid activity folder name: {activity_id}")
        if not patient_id.startswith("p"):
            raise ValueError(f"Invalid subject folder name: {patient_id}")
        if not segment_id.startswith("s"):
            raise ValueError(f"Invalid segment filename: {segment_id}")

        return {
            "record_id": record_id,
            "activity_id": activity_id,
            "patient_id": patient_id,
            "segment_id": segment_id,
        }

    def load_signal(self, file_path: str | Path) -> np.ndarray:
        """Load and validate a 125 x 45 sensor matrix from a text file."""
        file_path = Path(file_path)

        try:
            signal = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Failed to parse numeric data from {file_path}: {e}") from e

        if signal.size == 0:
            raise ValueError(f"Empty signal file: {file_path}")

        if signal.ndim == 1:
            signal = np.expand_dims(signal, axis=1)

        if signal.ndim != 2:
            raise ValueError(
                f"Signal in {file_path} must be 2D after parsing, got shape {signal.shape}"
            )

        if signal.shape != (125, 45):
            raise ValueError(
                f"Signal in {file_path} must have shape (125, 45), got {signal.shape}"
            )

        if not np.isfinite(signal).all():
            raise ValueError(f"Signal contains NaN or Inf values: {file_path}")

        return signal

    def _get_activity_name(self, activity_id: str) -> str:
        idx = int(activity_id[1:]) - 1
        if idx < 0 or idx >= len(self.activities):
            raise ValueError(f"Invalid activity_id: {activity_id}")
        return self.activities[idx]
    
    def _parse_file_to_event_row(self, file_path: Path) -> Dict[str, Any]:
        metadata = self._infer_metadata_from_path(file_path)
        signal = self.load_signal(file_path)

        activity_name = self._get_activity_name(metadata["activity_id"])

        return {
            "patient_id": metadata["patient_id"],
            "event_type": "daily_sport_activities",
            "timestamp": pd.NaT,
            "daily_sport_activities/record_id": metadata["record_id"],
            "daily_sport_activities/visit_id": metadata["segment_id"],
            "daily_sport_activities/activity_id": metadata["activity_id"],
            "daily_sport_activities/activity": activity_name,
            "daily_sport_activities/segment_id": metadata["segment_id"],
            "daily_sport_activities/file_path": str(file_path),
            "daily_sport_activities/n_rows": int(signal.shape[0]),
            "daily_sport_activities/n_cols": int(signal.shape[1]),
            "daily_sport_activities/sampling_rate_hz": 25,
            "daily_sport_activities/duration_seconds": 5,
        }
    
    def load_data(self) -> dd.DataFrame:
        """Load raw segment files into a PyHealth-compatible event dataframe."""
        rows: List[Dict[str, Any]] = []
        txt_files = self._discover_files()

        for file_path in txt_files:
            rows.append(self._parse_file_to_event_row(file_path))

        if not rows:
            raise ValueError("No samples were parsed from the dataset.")

        pdf = pd.DataFrame(rows)
        pdf["patient_id"] = pdf["patient_id"].astype("string")
        pdf["event_type"] = pdf["event_type"].astype("string")
        pdf["timestamp"] = pd.NaT

        return dd.from_pandas(pdf, npartitions=1)

    def parse_data(self) -> List[Dict]:
        """Debug helper: parse raw files into in-memory samples."""
        samples: List[Dict[str, Any]] = []
        txt_files = self._discover_files()

        for file_path in txt_files:
            metadata = self._infer_metadata_from_path(file_path)
            signal = self.load_signal(file_path)
            activity_name = self._get_activity_name(metadata["activity_id"])

            samples.append(
                {
                    "record_id": metadata["record_id"],
                    "patient_id": metadata["patient_id"],
                    "visit_id": metadata["segment_id"],
                    "activity_id": metadata["activity_id"],
                    "activity": activity_name,
                    "segment_id": metadata["segment_id"],
                    "file_path": str(file_path),
                    "signal": signal,
                }
            )

        if not samples:
            raise ValueError("No samples were parsed from the dataset.")

        return samples

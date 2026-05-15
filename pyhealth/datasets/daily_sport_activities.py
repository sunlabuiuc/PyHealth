"""
PyHealth dataset for the Daily and Sports Activity dataset.

Dataset link:
    https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities

Dataset paper:
    Zhang, H.; Zhan, D.; Lin, Y.; He, J.; Zhu, Q.; Shen, Z.-J.; and
    Zheng, Z. 2024. Daily Physical Activity Monitoring: Adaptive Learning
    from Multi-source Motion Sensor Data. Proceedings of the fifth Conference
    on Health, Inference, and Learning, volume 248 of Proceedings of Machine
    Learning Research, 39–54. PMLR

Dataset paper link:
    https://raw.githubusercontent.com/mlresearch/v248/main/assets/zhang24a/zhang24a.pdf

Authors:
    Niam Pattni (npattni2@illinois.edu)
    Sezim Zamirbekova (szami2@illinois.edu)
"""
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
    """
    PyHealth dataset for Daily and Sports Activities data.

    This dataset parses multi-sensor time-series text files into structured
    samples suitable for downstream task processing.

    Expected folder layout example:
        root/
          a01/
            p1/
              s01.txt
              s02.txt
          a02/
            p1/
              s01.txt

    Each .txt file is expected to contain numeric sensor values arranged
    row-wise over time. aXX represents an activity ID, p* represents a subject ID,
    and sXX represents a specific sensor ID.

    Parsed data format:
        - "record_id": str
        - "patient_id": str
        - "visit_id": str
        - "activity_id": str
        - "activity": int
        - "segment_id": str
        - "file_path": str
        - "signal": np.ndarray   # shape: [time_steps, num_features]

    Attributes:
        root (str): Root directory of the raw data.
        root_path (Path): Root directory of the raw data.
        config_path (str): Path to the configuration file.
        activities (List[str]): Ordered list of activities by ID, activities[0] == a01.
    """

    activities: List[str] = ["sitting", "standing", "lying on back",
                             "lying on right side", "ascending stairs",
                             "descending stairs", "standing still in elevator",
                             "moving around in elevator", "walking in parking lot",
                             "walking on 4km/h treadmill 0 incline",
                             "walking on 4km/h treadmill 15 incline",
                             "running on 8km/h treadmill", "using stair stepper",
                             "using cross trainer", "cycling in horizontal position",
                             "cycling in vertical position", "rowing", "jumping",
                             "playing basketball"]

    def __init__(
        self,
        root: str = ".",
        config_path: Optional[str] = str(
            Path(__file__).parent / "configs" / "daily_sport_activities.yaml"
        ),
        download: bool = False,
        dev: bool = False,
    ):
        """
        Initializes the Daily and Sports Activities dataset.

        Args:
            root (str): Root director of the raw data. Defaults to the current
            working directory.
            config_path (Optional[str]): Path to the configuration file.
            Defaults to "../configs/daily_sport_activities.yaml".
            download (bool): Whether to download the dataset or use an existing copy.
            Defaults to False.
            dev (bool): Configures parent BaseDataset. Defaults to False.

        Raises:
            FileNotFoundError: If the dataset cannot be found in the specified
            directory.
            NotADirectoryError: If the specified root path is not a directory.
        """
        self.root_path = Path(root)
        self.root = root

        if download:
            self._download(self.root)

        if not self.root_path.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root_path}")

        if not self.root_path.is_dir():
            raise NotADirectoryError(
                f"Dataset root is not a directory: {self.root_path}"
            )

        super().__init__(
            root=self.root,
            tables=["daily_sport_activities"],
            dataset_name="daily_sport_activities",
            config_path=config_path,
            dev=dev,
        )     

    @property
    def default_task(self):
        """
        Returns the default task for this dataset.

        Returns:
            DailyAndSportActivitiesClassification: The default classification task.

        Example::
            >>> dataset = DailyAndSportActivitiesDataset()
            >>> task = dataset.default_task
        """
        from pyhealth.tasks import DailyAndSportActivitiesClassification
        return DailyAndSportActivitiesClassification(signal_loader=self.load_signal)
    
    def _download(self, root: str) -> None:
        """Downloads the Daily and Sports Activities dataset and extracts the
        compressed data.
        
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
        """
        Find all text files under the dataset root.
        
        Returns:
            List[Path]: List of all paths to relevant text files

        Raises:
            FileNotFoundError: No text files exist in the specified root path.
        """
        txt_files = sorted(self.root_path.rglob("*.txt"))

        if not txt_files:
            raise FileNotFoundError(
                f"No .txt files found under dataset root: {self.root_path.root}"
            )

        return txt_files

    def _infer_metadata_from_path(self, file_path: Path) -> Dict[str, str]:
        """
        Infer activity, subject, and segment identifiers from the file path.
        
        Args:
            file_path (Path): The path to the given text file.
        
        Returns:
            Dict[str, str]: Map from metadata name to value for record, activity,
            patient, and segment IDs

        Raises:
            ValueError: Folder structure does not follow aXX/p*/sXX.txt.
            ValueError: Activity folder name doesn't start with a.
            ValueError: Patient folder name doesn't start with p.
            ValueError: Segment file name doesn't start with s.
        """
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
        """
        Load and validate a 125 x 45 sensor matrix from a text file.
        
        Args:
            file_path (str | Path): The path to the text file to load.

        Returns:
            np.ndarray: The value loaded from the text file of the sensor.

        Raises:
            ValueError: Couldn't read numeric data from the file.
            ValueError: Empty target file.
            ValueError: Shape of parsed data is not 2D.
            ValueError: Shape of parsed data is not (125, 45).
            ValueError: Parsed data contains NaN or Inf values.
        """
        file_path = Path(file_path)

        try:
            signal = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
        except Exception as e:
            raise ValueError(
                f"Failed to parse numeric data from {file_path}: {e}"
            ) from e

        if signal.size == 0:
            raise ValueError(f"Empty signal file: {file_path}")

        if signal.ndim == 1:
            signal = np.expand_dims(signal, axis=1)

        if signal.ndim != 2:
            raise ValueError(
                f"""Signal in {file_path} must be 2D after parsing, got shape
                {signal.shape}"""
            )

        if signal.shape != (125, 45):
            raise ValueError(
                f"Signal in {file_path} must have shape (125, 45), got {signal.shape}"
            )

        if not np.isfinite(signal).all():
            raise ValueError(f"Signal contains NaN or Inf values: {file_path}")

        return signal

    def _get_activity_name(self, activity_id: str) -> str:
        """
        Get activity name from a given ID.

        Args:
            activity_id (str): The ID number XX from the folder "root/aXX/...".

        Returns:
            str: The corresponding activity name from activities[XX - 1]

        Raises:
            ValueError: Activity ID is not within 1 to 19 inclusive.
        """
        idx = int(activity_id[1:]) - 1
        if idx < 0 or idx >= len(self.activities):
            raise ValueError(f"Invalid activity_id: {activity_id}")
        return self.activities[idx]
    
    def _parse_file_to_event_row(self, file_path: Path) -> Dict[str, Any]:
        """
        Reads a target file and folder structure and parses it into an event row.

        Args:
            file_path (Path): Path to target dataset file.

        Returns:
            Dict[str, Any]: Maps attribute name to value from parsed data
        """
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
        """
        Load raw segment files into a PyHealth-compatible event dataframe.
        
        Returns:
            dd.DataFrame: Dask dataframe of event rows from parsed data.

        Raises:
            ValueError: No valid parsed data exists.
        """
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

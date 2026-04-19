"""Daily and Sports Activities (DSA) dataset loader."""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from pyhealth.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Dataset metadata (SleepEDF-style: domain facts live in code, YAML is tables only)
# -----------------------------------------------------------------------------

DSA_PYHEALTH_MANIFEST = "dsa-pyhealth.csv"

_LABEL_MAPPING: Dict[str, str] = {
    "A01": "sitting",
    "A02": "standing",
    "A03": "lying_on_back",
    "A04": "lying_on_right_side",
    "A05": "ascending_stairs",
    "A06": "descending_stairs",
    "A07": "standing_in_elevator_still",
    "A08": "moving_around_in_elevator",
    "A09": "walking_in_parking_lot",
    "A10": "walking_on_treadmill_flat",
    "A11": "walking_on_treadmill_inclined",
    "A12": "running_on_treadmill",
    "A13": "exercising_on_stepper",
    "A14": "exercising_on_cross_trainer",
    "A15": "cycling_on_exercise_bike_horizontal",
    "A16": "cycling_on_exercise_bike_vertical",
    "A17": "rowing",
    "A18": "jumping",
    "A19": "playing_basketball",
}

_UNITS: List[Dict[str, str]] = [
    {"T": "Torso"},
    {"RA": "Right Arm"},
    {"LA": "Left Arm"},
    {"RL": "Right Leg"},
    {"LL": "Left Leg"},
]

_SENSORS: List[Dict[str, str]] = [
    {"xacc": "X-axis Accelerometer"},
    {"yacc": "Y-axis Accelerometer"},
    {"zacc": "Z-axis Accelerometer"},
    {"xgyro": "X-axis Gyroscope"},
    {"ygyro": "Y-axis Gyroscope"},
    {"zgyro": "Z-axis Gyroscope"},
    {"xmag": "X-axis Magnetometer"},
    {"ymag": "Y-axis Magnetometer"},
    {"zmag": "Z-axis Magnetometer"},
]

_SAMPLING_FREQUENCY = 25
_NUM_COLUMNS = 45
_NUM_ROWS = 125

_LAYOUT = {
    "activity_dir_pattern": r"^a\d{2}$",
    "subject_dir_pattern": r"^p\d+$",
    "segment_file_pattern": r"^s\d+\.txt$",
    "code_regex_pattern": r"^A(\d+)$",
    "file_extension": ".txt",
}

_ACTIVITY_DIR_RE = re.compile(_LAYOUT["activity_dir_pattern"])
_SUBJECT_DIR_RE = re.compile(_LAYOUT["subject_dir_pattern"])
_SEGMENT_FILE_RE = re.compile(_LAYOUT["segment_file_pattern"])
_ACTIVITY_CODE_RE = re.compile(_LAYOUT["code_regex_pattern"])

DSA_TABLE_NAME = "segments"


class DSADataset(BaseDataset):
    """Daily and Sports Activities (DSA) time-series dataset (Barshan & Altun, 2010).

    Recordings use five on-body IMU units (torso, two arms, two legs); each unit
    contributes nine columns per row (3-axis accelerometer, gyroscope, and
    magnetometer), so each segment row has 45 comma-separated values. The public
    release is sampled at 25 Hz; each ``.txt`` segment is typically 125 lines (about
    five seconds of data).

    On disk, activities live in folders ``a01`` through ``a19``, subjects in ``p1``
    through ``p8``, and segment files ``s01.txt``, ``s02.txt``, … under each
    subject.

    Dataset is available at:
    https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities

    Citations:
        If you use this dataset, cite: Barshan, B., & Altun, K. (2010). Daily and
        Sports Activities [Dataset]. UCI Machine Learning Repository.
        https://doi.org/10.24432/C5C59F

    Args:
        root str: Dataset root (activity folders; manifest created if missing).
        dataset_name: Passed to :class:`BaseDataset`. Default ``"dsa"``.
        config_path: Path to ``dsa.yaml`` (default: package ``configs/dsa.yaml``).
        cache_dir: Cache directory for :class:`BaseDataset`.
        num_workers: Parallel workers for base pipelines.
        dev: Passed to :class:`BaseDataset` (limits patients when building events).

    Examples:
        >>> from pyhealth.datasets import DSADataset
        >>> dataset = DSADataset(root="/path/to/dsa")
        >>> dataset.stat()
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "dsa.yaml"
            )

        metadata_path = os.path.join(root, DSA_PYHEALTH_MANIFEST)
        if not os.path.exists(metadata_path):
            self.prepare_metadata(root)

        super().__init__(
            root=root,
            tables=[DSA_TABLE_NAME],
            dataset_name=dataset_name or "dsa",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

        self.label_mapping: Dict[str, str] = dict(_LABEL_MAPPING)
        self.units: List[Dict[str, str]] = list(_UNITS)
        self.sensors: List[Dict[str, str]] = list(_SENSORS)
        self.sampling_frequency: int = _SAMPLING_FREQUENCY
        self._num_columns: int = _NUM_COLUMNS
        self._num_rows: int = _NUM_ROWS

        self._manifest_df: pd.DataFrame = pd.read_csv(os.path.join(self.root, self.config.tables[DSA_TABLE_NAME].file_path))

    def prepare_metadata(self, root: str) -> None:
        """Scan ``root`` and write ``dsa-pyhealth.csv`` (``tables.segments``)."""
        rows = []
        for a_dir in sorted(os.listdir(root)):
            if not _ACTIVITY_DIR_RE.match(a_dir):
                continue
            activity_code = a_dir.upper()
            a_path = os.path.join(root, a_dir)
            if not os.path.isdir(a_path):
                continue

            for p_dir in sorted(os.listdir(a_path)):
                if not _SUBJECT_DIR_RE.match(p_dir):
                    continue
                p_path = os.path.join(a_path, p_dir)
                if not os.path.isdir(p_path):
                    continue

                for s_file in sorted(os.listdir(p_path)):
                    if not _SEGMENT_FILE_RE.match(s_file):
                        continue

                    rows.append(
                        {
                            "subject_id": p_dir,
                            "activity_name": _LABEL_MAPPING[activity_code],
                            "activity_code": activity_code,
                            "segment_path": f"{a_dir}/{p_dir}/{s_file}",
                        }
                    )

        if not rows:
            raise ValueError(
                f"No DSA segments under {root}; expected aXX/pY/sZZ.txt layout."
            )

        metadata_path = os.path.join(root, DSA_PYHEALTH_MANIFEST)
        df = pd.DataFrame(rows)
        df = df[["subject_id", "activity_name", "activity_code", "segment_path"]]
        df.to_csv(metadata_path, index=False)

    def get_subject_ids(self) -> List[str]:
        """Return sorted subject IDs from the manifest."""
        return sorted(self._manifest_df["subject_id"].unique().tolist())

    def get_activity_labels(self) -> Dict[str, int]:
        """Map activity name to class index (ordered by activity code)."""
        codes = sorted(self.label_mapping.keys())
        return {self.label_mapping[c]: i for i, c in enumerate(codes)}

    def get_subject_data(self, subject_id: str) -> Dict[str, Any]:
        """Load all segment arrays for one subject."""
        subject_df = self._manifest_df[self._manifest_df["subject_id"] == subject_id]
        if subject_df.empty:
            raise ValueError(f"Subject {subject_id!r} not found in manifest")

        subject_data: Dict[str, Any] = {"id": subject_id, "activities": {}}

        for (activity_name, activity_code), group in subject_df.groupby(
            ["activity_name", "activity_code"]
        ):
            segments = []
            for _, row in group.iterrows():
                segment_path = os.path.join(self.root, row["segment_path"])
                segment_data = self._load_segment(segment_path, subject_id, activity_name)
                segments.append(segment_data)

            subject_data["activities"][activity_name] = {
                "id": activity_code,
                "segments": segments,
            }

        return subject_data

    def _load_segment(
        self,
        file_path: str,
        subject_id: str,
        activity: str,
    ) -> Dict[str, Any]:
        """Load a single segment file and return as dict."""
        try:
            data = np.loadtxt(file_path, delimiter=",", dtype=np.float64)
        except Exception as e:
            raise ValueError(
                f"Failed to parse DSA segment {file_path}; expected a "
                f"{self._num_rows}x{self._num_columns} comma-separated numeric file."
            ) from e

        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_rows, n_cols = data.shape
        if n_rows != self._num_rows:
            raise ValueError(
                f"{file_path} has {n_rows} rows, expected {self._num_rows}"
            )
        if n_cols != self._num_columns:
            raise ValueError(
                f"{file_path} has {n_cols} columns, expected {self._num_columns}"
            )
        if not np.isfinite(data).all():
            raise ValueError(f"{file_path} contains non-finite values (NaN or Inf).")

        return {
            "file_path": Path(file_path),
            "subject_id": subject_id,
            "activity": activity,
            "data": data,
            "num_samples": n_rows,
            "sampling_rate": self.sampling_frequency,
            "segment_filename": os.path.basename(file_path),
        }

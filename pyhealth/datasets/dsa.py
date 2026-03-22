"""Daily and Sports Activities (DSA) dataset loader."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml

from pyhealth.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "dsa.yaml"


def _load_dsa_yaml(config_path: Path) -> Dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or "dataset" not in raw:
        raise ValueError(f"DSA config {config_path} must contain a top-level 'dataset' key.")
    cfg = raw["dataset"]
    if not isinstance(cfg, dict):
        raise ValueError(f"DSA config 'dataset' section must be a mapping in {config_path}.")
    return cfg


class DSADataset(BaseDataset):
    """Daily and Sports Activities (DSA) time-series dataset.

    Expected layout under ``root``::

        root/
            a01/p1/s01.txt
            a01/p1/s02.txt
            ...
            a19/p8/s60.txt

    Each segment file has 125 lines and 45 comma-separated columns (25 Hz × 5 s).
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        cfg_path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG
        self._dsa_config = _load_dsa_yaml(cfg_path)

        super().__init__(
            root=str(Path(root).expanduser().resolve()),
            tables=[],
            dataset_name=dataset_name or str(self._dsa_config.get("internal_name", "dsa")),
            config_path=None,
        )

        self._root_path = Path(self.root)
        task_cfg = self._dsa_config.get("task_configs", {}).get("activity_recognition", {})
        self.label_mapping: Dict[str, str] = dict(task_cfg.get("label_mapping", {}))
        self.units: List[Any] = list(self._dsa_config.get("units", []))
        self.sensors: List[Any] = list(self._dsa_config.get("sensors", []))
        self.sampling_frequency: int = int(self._dsa_config.get("sampling_frequency", 25))
        layout = self._dsa_config.get("layout", {}) or {}
        self._activity_dir_re = re.compile(layout.get("activity_dir_pattern", r"^a\d{2}$"))
        self._subject_dir_re = re.compile(layout.get("subject_dir_pattern", r"^p\d+$"))
        self._segment_file_re = re.compile(layout.get("segment_file_pattern", r"^s\d+\.txt$"))
        self._num_columns = int(self._dsa_config.get("data_structure", {}).get("num_columns", 45))
        self._metadata: Optional[Dict[str, Any]] = None

    def _activity_code_from_dir(self, activity_dir: str) -> Optional[str]:
        m = re.fullmatch(r"a(\d{2})", activity_dir)
        if not m:
            return None
        return f"A{int(m.group(1))}"

    def _load_metadata(self) -> Dict[str, Any]:
        if not self._root_path.is_dir():
            raise FileNotFoundError(f"DSA root is not a directory: {self._root_path}")

        metadata: Dict[str, Any] = {"subjects": {}, "activities": {}, "file_paths": []}

        for activity_dir in sorted(self._root_path.iterdir()):
            if not activity_dir.is_dir() or not self._activity_dir_re.match(activity_dir.name):
                continue

            code = self._activity_code_from_dir(activity_dir.name)
            if code is None:
                continue
            activity_name = self.label_mapping.get(code)
            if activity_name is None:
                logger.warning("Skipping folder %s: no label_mapping entry for %s", activity_dir, code)
                continue

            activity_path = activity_dir
            if activity_name not in metadata["activities"]:
                metadata["activities"][activity_name] = {
                    "id": code,
                    "path": activity_path,
                    "subjects": {},
                }

            for subject_dir in sorted(activity_path.iterdir()):
                if not subject_dir.is_dir() or not self._subject_dir_re.match(subject_dir.name):
                    continue

                subject_id = subject_dir.name
                segment_files = sorted(
                    f.name
                    for f in subject_dir.iterdir()
                    if f.is_file() and self._segment_file_re.match(f.name)
                )

                if subject_id not in metadata["subjects"]:
                    metadata["subjects"][subject_id] = {
                        "id": subject_id,
                        "activities": {},
                    }

                metadata["subjects"][subject_id]["activities"][activity_name] = {
                    "id": code,
                    "path": subject_dir,
                    "segments": segment_files,
                }
                metadata["activities"][activity_name]["subjects"][subject_id] = segment_files

        return metadata

    def get_subject_data(self, subject_id: str) -> Dict[str, Any]:
        """Load all segment data for one subject."""
        return self._load_subject_data(subject_id)

    def _load_subject_data(self, subject_id: str) -> Dict[str, Any]:
        if self._metadata is None:
            self._metadata = self._load_metadata()

        if subject_id not in self._metadata["subjects"]:
            raise ValueError(f"Subject {subject_id!r} not found under {self._root_path}")

        subject_data: Dict[str, Any] = {"id": subject_id, "activities": {}}
        subject_info = self._metadata["subjects"][subject_id]

        for activity_name, activity_info in subject_info["activities"].items():
            activity_data: Dict[str, Any] = {"id": activity_info["id"], "segments": []}
            for segment_file in activity_info["segments"]:
                file_path = Path(activity_info["path"]) / segment_file
                activity_data["segments"].append(
                    self._process_segment(file_path, subject_id, activity_name)
                )
            subject_data["activities"][activity_name] = activity_data

        return subject_data

    def _process_segment(
        self,
        file_path: Path,
        subject_id: str,
        activity: str,
    ) -> Dict[str, Any]:
        with open(file_path, encoding="utf-8") as f:
            lines = [ln for ln in f.readlines() if ln.strip()]

        data = np.zeros((len(lines), self._num_columns), dtype=np.float64)
        for i, line in enumerate(lines):
            values = line.strip().split(",")
            if len(values) != self._num_columns:
                raise ValueError(
                    f"Line {i + 1} in {file_path} has {len(values)} values, expected {self._num_columns}"
                )
            try:
                data[i] = [float(x) for x in values]
            except ValueError as exc:
                raise ValueError(f"Error parsing values in {file_path}: {exc}") from exc

        return {
            "file_path": file_path,
            "subject_id": subject_id,
            "activity": activity,
            "data": data,
            "num_samples": len(lines),
            "sampling_rate": self.sampling_frequency,
            "units": self.units,
            "sensors": self.sensors,
            "timestamp": file_path.name,
        }

    def get_subject_ids(self) -> List[str]:
        if self._metadata is None:
            self._metadata = self._load_metadata()
        return sorted(self._metadata["subjects"].keys())

    def get_activity_labels(self) -> Dict[str, int]:
        """Map each activity name (from config) to an integer label 0 … 18 (A1 … A19)."""
        codes = sorted(self.label_mapping.keys(), key=lambda c: int(str(c)[1:]))
        return {self.label_mapping[c]: idx for idx, c in enumerate(codes)}

    def get_num_subjects(self) -> int:
        if self._metadata is None:
            self._metadata = self._load_metadata()
        return len(self._metadata["subjects"])

    def get_num_activities(self) -> int:
        if self._metadata is None:
            self._metadata = self._load_metadata()
        return len(self._metadata["activities"])

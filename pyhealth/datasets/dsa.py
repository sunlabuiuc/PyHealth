"""Daily and Sports Activities (DSA) dataset loader."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

from pyhealth.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "dsa.yaml"
DSA_TABLE_KEY = "dsa_segments"


def _load_raw_dsa_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate the full DSA YAML (tables + dataset metadata).

    Args:
        config_path: Path to ``dsa.yaml`` containing ``version``, ``tables``,
            and ``dataset`` sections.

    Returns:
        Parsed YAML as a dictionary.

    Raises:
        ValueError: If required top-level keys are missing or have wrong types.
        FileNotFoundError: If ``config_path`` does not exist.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"DSA config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"DSA config {config_path} must be a mapping at the top level."
        )
    for key in ("version", "tables", "dataset"):
        if key not in raw:
            raise ValueError(
                f"DSA config {config_path} must contain top-level key {key!r}."
            )
    if not isinstance(raw["tables"], dict) or DSA_TABLE_KEY not in raw["tables"]:
        raise ValueError(
            f"DSA config {config_path} must define tables.{DSA_TABLE_KEY} for "
            "BaseDataset."
        )
    if not isinstance(raw["dataset"], dict):
        raise ValueError(
            f"DSA config {config_path}: 'dataset' section must be a mapping."
        )
    return raw


def _dataset_section(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return the ``dataset`` subsection from a full raw config dict."""
    sec = raw["dataset"]
    assert isinstance(sec, dict)
    return sec


class DSADataset(BaseDataset):
    """Daily and Sports Activities (DSA) time-series dataset.

    This class follows the same configuration pattern as :class:`MIMIC4EHRDataset`
    and :class:`COVID19CXRDataset`: a YAML file under ``configs/`` defines tables for
    :class:`BaseDataset`, while the ``dataset`` section in that file holds DSA-specific
    metadata (labels, layout patterns, sensor channel descriptions).

    If ``dsa_manifest.csv`` is not yet present under ``root``, it is created by
    scanning the directory tree (same layout as the public DSA release).

    Expected layout under ``root``::

        root/
            dsa_manifest.csv   # created automatically if missing
            a01/p1/s01.txt
            a01/p1/s02.txt
            ...
            a19/p8/s60.txt

    Each segment file has 125 lines and 45 comma-separated columns (25 Hz × 5 s).

    Examples:
        >>> from pyhealth.datasets import DSADataset
        >>> # dataset = DSADataset(root="/path/to/dsa")
        >>> # dataset.get_subject_ids()
        >>> # subject = dataset.get_subject_data("p1")
        >>> # dataset.stats()  # uses BaseDataset event pipeline
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        cfg_path = (
            Path(config_path).expanduser().resolve()
            if config_path
            else _DEFAULT_CONFIG
        )
        raw_cfg = _load_raw_dsa_config(cfg_path)
        dsa_cfg = _dataset_section(raw_cfg)

        root_path = Path(root).expanduser().resolve()
        manifest_rel = raw_cfg["tables"][DSA_TABLE_KEY]["file_path"]
        manifest_path = root_path / manifest_rel

        self._prepare_manifest_if_missing(root_path, dsa_cfg, manifest_path)

        super().__init__(
            root=str(root_path),
            tables=[DSA_TABLE_KEY],
            dataset_name=dataset_name
            or str(dsa_cfg.get("internal_name", "dsa")),
            config_path=str(cfg_path),
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

        self._root_path = root_path
        self._dsa_config = dsa_cfg
        task_cfg = dsa_cfg.get("task_configs", {}).get("activity_recognition", {})
        self.label_mapping: Dict[str, str] = dict(task_cfg.get("label_mapping", {}))
        self.units: List[Dict[str, str]] = [
            u for u in dsa_cfg.get("units", []) if isinstance(u, dict)
        ]
        self.sensors: List[Dict[str, str]] = [
            s for s in dsa_cfg.get("sensors", []) if isinstance(s, dict)
        ]
        self.sampling_frequency: int = int(dsa_cfg.get("sampling_frequency", 25))
        layout = dsa_cfg.get("layout", {}) or {}
        self._activity_dir_re = re.compile(
            layout.get("activity_dir_pattern", r"^a\d{2}$")
        )
        self._subject_dir_re = re.compile(
            layout.get("subject_dir_pattern", r"^p\d+$")
        )
        self._segment_file_re = re.compile(
            layout.get("segment_file_pattern", r"^s\d+\.txt$")
        )
        ds = dsa_cfg.get("data_structure", {}) or {}
        self._num_columns = int(ds.get("num_columns", 45))
        self._metadata: Optional[Dict[str, Any]] = None

    @staticmethod
    def _prepare_manifest_if_missing(
        root_path: Path,
        dsa_cfg: Dict[str, Any],
        manifest_path: Path,
    ) -> None:
        if manifest_path.is_file():
            return
        if not root_path.is_dir():
            raise FileNotFoundError(f"DSA root is not a directory: {root_path}")
        logger.info("Writing DSA manifest to %s", manifest_path)
        df = DSADataset._scan_tree_to_manifest(root_path, dsa_cfg)
        df.to_csv(manifest_path, index=False)

    def prepare_metadata(self, root: Optional[str] = None) -> None:
        """Rebuild ``dsa_manifest.csv`` by scanning the on-disk DSA tree.

        Args:
            root: Dataset root directory. Defaults to :attr:`root`.

        Overwrites the manifest path defined in the ``dsa_segments`` table config.
        """
        root_path = Path(root).expanduser().resolve() if root else Path(self.root)
        assert self.config is not None
        rel = self.config.tables[DSA_TABLE_KEY].file_path
        manifest_path = root_path / rel
        if not root_path.is_dir():
            raise FileNotFoundError(f"DSA root is not a directory: {root_path}")
        logger.info("Regenerating DSA manifest at %s", manifest_path)
        df = self._scan_tree_to_manifest(root_path, self._dsa_config)
        df.to_csv(manifest_path, index=False)
        self._metadata = None

    @staticmethod
    def _scan_tree_to_manifest(
        root_path: Path,
        dsa_cfg: Dict[str, Any],
    ) -> pd.DataFrame:
        """Walk ``root_path`` and build a manifest dataframe."""
        task_cfg = dsa_cfg.get("task_configs", {}).get("activity_recognition", {})
        label_mapping: Dict[str, str] = dict(task_cfg.get("label_mapping", {}))
        layout = dsa_cfg.get("layout", {}) or {}
        activity_re = re.compile(layout.get("activity_dir_pattern", r"^a\d{2}$"))
        subject_re = re.compile(layout.get("subject_dir_pattern", r"^p\d+$"))
        segment_re = re.compile(layout.get("segment_file_pattern", r"^s\d+\.txt$"))

        def activity_code_from_dir(name: str) -> Optional[str]:
            m = re.fullmatch(r"a(\d{2})", name)
            if not m:
                return None
            return f"A{int(m.group(1))}"

        rows: List[Dict[str, str]] = []

        for activity_dir in sorted(root_path.iterdir()):
            if not activity_dir.is_dir() or not activity_re.match(activity_dir.name):
                continue
            code = activity_code_from_dir(activity_dir.name)
            if code is None:
                continue
            activity_name = label_mapping.get(code)
            if activity_name is None:
                logger.warning(
                    "Skipping folder %s: no label_mapping entry for %s",
                    activity_dir,
                    code,
                )
                continue

            for subject_dir in sorted(activity_dir.iterdir()):
                if not subject_dir.is_dir() or not subject_re.match(
                    subject_dir.name
                ):
                    continue
                subject_id = subject_dir.name
                for seg in sorted(subject_dir.iterdir()):
                    if not seg.is_file() or not segment_re.match(seg.name):
                        continue
                    rel = seg.relative_to(root_path).as_posix()
                    rows.append(
                        {
                            "subject_id": subject_id,
                            "activity_name": activity_name,
                            "activity_code": code,
                            "segment_path": rel,
                        }
                    )

        if not rows:
            raise ValueError(
                f"No DSA segment files found under {root_path}. "
                "Expected layout aXX/pY/sZZ.txt."
            )
        return pd.DataFrame(rows)

    def _load_metadata(self) -> Dict[str, Any]:
        """Build subject/activity indexes from the manifest CSV."""
        if not self._root_path.is_dir():
            raise FileNotFoundError(
                f"DSA root is not a directory: {self._root_path}"
            )
        assert self.config is not None
        rel_manifest = self.config.tables[DSA_TABLE_KEY].file_path
        manifest_path = self._root_path / rel_manifest
        if not manifest_path.is_file():
            raise FileNotFoundError(f"DSA manifest not found: {manifest_path}")

        df = pd.read_csv(manifest_path)
        metadata: Dict[str, Any] = {"subjects": {}, "activities": {}}

        for (sid, aname), group in df.groupby(
            ["subject_id", "activity_name"],
            sort=True,
        ):
            sid = str(sid)
            aname = str(aname)
            code = str(group["activity_code"].iloc[0])
            paths = [Path(str(x)) for x in group["segment_path"].tolist()]
            subject_dir = self._root_path / paths[0].parent
            segment_files = sorted(p.name for p in paths)
            activity_dir = self._root_path / paths[0].parts[0]

            if sid not in metadata["subjects"]:
                metadata["subjects"][sid] = {"id": sid, "activities": {}}
            metadata["subjects"][sid]["activities"][aname] = {
                "id": code,
                "path": subject_dir,
                "segments": segment_files,
            }

            if aname not in metadata["activities"]:
                metadata["activities"][aname] = {
                    "id": code,
                    "path": activity_dir,
                    "subjects": {},
                }
            metadata["activities"][aname]["subjects"][sid] = segment_files

        return metadata

    def get_subject_data(self, subject_id: str) -> Dict[str, Any]:
        """Load every segment time series for one subject.

        Args:
            subject_id: Subject directory name (e.g. ``\"p1\"``).

        Returns:
            Nested dict: subject id, then per-activity segment dicts with NumPy
            ``data`` arrays.

        Raises:
            ValueError: If ``subject_id`` is not present in the manifest.
        """
        return self._load_subject_data(subject_id)

    def _load_subject_data(self, subject_id: str) -> Dict[str, Any]:
        if self._metadata is None:
            self._metadata = self._load_metadata()

        if subject_id not in self._metadata["subjects"]:
            raise ValueError(
                f"Subject {subject_id!r} not found under {self._root_path}"
            )

        subject_data: Dict[str, Any] = {"id": subject_id, "activities": {}}
        subject_info = self._metadata["subjects"][subject_id]

        for activity_name, activity_info in subject_info["activities"].items():
            activity_data: Dict[str, Any] = {
                "id": activity_info["id"],
                "segments": [],
            }
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
                    f"Line {i + 1} in {file_path} has {len(values)} values, "
                    f"expected {self._num_columns}"
                )
            try:
                data[i] = [float(x) for x in values]
            except ValueError as exc:
                raise ValueError(
                    f"Error parsing values in {file_path}: {exc}"
                ) from exc

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
        """Return sorted subject ids from the manifest."""
        if self._metadata is None:
            self._metadata = self._load_metadata()
        return sorted(self._metadata["subjects"].keys())

    def get_activity_labels(self) -> Dict[str, int]:
        """Map activity names (from config) to integer labels 0 … 18.

        Returns:
            Mapping from human-readable activity label to class index ordered by
            activity code A1 … A19.
        """
        codes = sorted(self.label_mapping.keys(), key=lambda c: int(str(c)[1:]))
        return {self.label_mapping[c]: idx for idx, c in enumerate(codes)}

    def get_num_subjects(self) -> int:
        """Number of distinct subjects in the manifest."""
        if self._metadata is None:
            self._metadata = self._load_metadata()
        return len(self._metadata["subjects"])

    def get_num_activities(self) -> int:
        """Number of distinct activities present under ``root``."""
        if self._metadata is None:
            self._metadata = self._load_metadata()
        return len(self._metadata["activities"])

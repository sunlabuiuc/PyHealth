"""Daily and Sports Activities (DSA) dataset loader."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from pyhealth.datasets.base_dataset import BaseDataset
from pyhealth.datasets.configs.config import DatasetConfig

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "dsa.yaml"
DSA_TABLE_KEY = "dsa_segments"
_MANIFEST_COLUMNS = (
    "subject_id",
    "activity_name",
    "activity_code",
    "segment_path",
)

_ACTIVITY_CODE_RE = re.compile(r"^A(\d+)$", re.IGNORECASE)


def _activity_code_sort_key(code: str) -> int:
    """Return the numeric part of activity code (e.g. ``A3`` / ``a3`` -> ``3``)."""
    m = _ACTIVITY_CODE_RE.match(code.strip())
    if not m:
        raise ValueError(
            f"Invalid activity code {code!r} in label_mapping; expected like 'A1', 'A12'."
        )
    return int(m.group(1))


def _load_and_validate_dsa_yaml_full(config_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load DSA YAML; validate ``tables`` with Pydantic and return root + ``dataset``."""
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
    DatasetConfig.model_validate(
        {"version": raw["version"], "tables": raw["tables"]}
    )
    return raw, raw["dataset"]


class DSADataset(BaseDataset):
    """Loader for the Daily and Sports Activities (DSA) time-series benchmark.

    The public DSA release contains multivariate sensor recordings from five body-worn
    inertial measurement units (45 channels) sampled at 25 Hz. Activities are
    organized under ``a01``…``a19`` directories, subjects under ``p1``…``p8``, with
    fixed-length segment files ``s01.txt``, … (typically 125 rows × 45 columns per
    file). PyHealth mirrors activity labels from the YAML ``label_mapping`` and builds
    ``dsa_manifest.csv`` under the dataset root on first use so
    :class:`~pyhealth.datasets.base_dataset.BaseDataset` can expose segment rows via
    :meth:`~pyhealth.datasets.base_dataset.BaseDataset.load_table`.

    Dataset is available at:
    https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities

    Data (summary):
    ----------------
    - 19 daily living and sports activity classes, 8 subjects, fixed-length segments
      with five IMU units (accelerometer, gyroscope, magnetometer per unit).

    Citations:
    ---------
    If you use this dataset, please cite:
    1. Barshan, B., & Altun, K. (2010). Daily and Sports Activities [Dataset]. UCI
       Machine Learning Repository. https://doi.org/10.24432/C5C59F.

    References:
    ----------
    [1] https://doi.org/10.24432/C5C59F
    [2] https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities

    Args:
        root: Directory containing activity folders (e.g. ``a01``) or an existing
            ``dsa_manifest.csv`` alongside that tree.
        dataset_name: Optional logical name; defaults to ``internal_name`` from the
            config or ``\"dsa\"``.
        config_path: Path to ``dsa.yaml``. Defaults to the package config next to this
            module.
        cache_dir: Passed through to :class:`BaseDataset` for task/event caches.
        num_workers: Worker processes for :class:`BaseDataset` pipelines.
        dev: If ``True``, :class:`BaseDataset` limits the global event table to the
            first 1000 patient ids when building cached parquets (see parent
            implementation).

    Examples:
        >>> from pyhealth.datasets import DSADataset
        >>> # ds = DSADataset(root="/path/to/dsa")
        >>> # ds.get_subject_ids()
        >>> # ds.get_subject_data("p1")
        >>> # ds.load_table("dsa_segments")
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
        raw_cfg, dsa_cfg = _load_and_validate_dsa_yaml_full(cfg_path)

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
        ds = dsa_cfg.get("data_structure", {}) or {}
        self._num_columns = int(ds.get("num_columns", 45))
        nr = ds.get("num_rows")
        self._num_rows: Optional[int] = int(nr) if nr is not None else None
        self._metadata: Optional[Dict[str, Any]] = None

    @staticmethod
    def _prepare_manifest_if_missing(
        root_path: Path,
        dsa_cfg: Dict[str, Any],
        manifest_path: Path,
    ) -> None:
        """Create ``dsa_manifest.csv`` by scanning the tree if the file is absent.

        Args:
            root_path: Dataset root (activity directories live directly under this).
            dsa_cfg: The ``dataset:`` mapping from ``dsa.yaml``.
            manifest_path: Absolute path to the CSV to create.

        Raises:
            FileNotFoundError: If ``root_path`` is not a directory when a scan is
                needed.
            ValueError: If the scan finds no segment files.
        """
        if manifest_path.is_file():
            return
        if not root_path.is_dir():
            raise FileNotFoundError(f"DSA root is not a directory: {root_path}")
        logger.info("Writing DSA manifest to %s", manifest_path)
        df = DSADataset._scan_tree_to_manifest(root_path, dsa_cfg)
        df.to_csv(manifest_path, index=False)

    def _config_or_raise(self) -> DatasetConfig:
        if self.config is None:
            raise RuntimeError("Dataset config is not loaded.")
        return self.config

    def prepare_metadata(self, root: Optional[str] = None) -> None:
        """Rebuild ``dsa_manifest.csv`` by scanning the on-disk DSA tree.

        Args:
            root: Dataset root directory. Defaults to :attr:`root`.

        Raises:
            FileNotFoundError: If ``root`` is not a directory.
            RuntimeError: If the in-memory config is missing.
            ValueError: If the scan finds no segment files.

        Overwrites the manifest path defined in the ``dsa_segments`` table config.
        """
        root_path = Path(root).expanduser().resolve() if root else Path(self.root)
        cfg = self._config_or_raise()
        rel = cfg.tables[DSA_TABLE_KEY].file_path
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
        """Walk ``root_path`` and build a manifest dataframe.

        Args:
            root_path: Dataset root containing ``aXX/pY/sZZ.txt`` layout.
            dsa_cfg: Parsed ``dataset:`` section from ``dsa.yaml``.

        Returns:
            DataFrame with columns ``subject_id``, ``activity_name``, ``activity_code``,
            ``segment_path``.

        Raises:
            ValueError: If no segment files are found.
        """
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

    def _validate_manifest_dataframe(self, df: pd.DataFrame, manifest_path: Path) -> None:
        """Ensure the manifest has expected columns and non-degenerate contents."""
        missing = [c for c in _MANIFEST_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"DSA manifest {manifest_path} is missing columns: {missing}. "
                f"Expected at least {_MANIFEST_COLUMNS}."
            )
        if df.empty:
            raise ValueError(f"DSA manifest {manifest_path} is empty.")
        for col in _MANIFEST_COLUMNS:
            if df[col].isna().any():
                raise ValueError(
                    f"DSA manifest {manifest_path} has null values in column {col!r}."
                )

    def _resolved_segment_path(self, rel_raw: str) -> Path:
        """Resolve a manifest path under ``root`` (reject path traversal)."""
        rel = Path(str(rel_raw))
        if rel.is_absolute():
            raise ValueError(
                f"segment_path must be relative to dataset root, got {rel_raw!r}"
            )
        full = (self._root_path / rel).resolve()
        try:
            full.relative_to(self._root_path.resolve())
        except ValueError as exc:
            raise ValueError(
                f"segment_path {rel_raw!r} escapes dataset root {self._root_path}"
            ) from exc
        return full

    def _load_metadata(self) -> Dict[str, Any]:
        """Build subject/activity indexes from the manifest CSV.

        Returns:
            Nested ``subjects`` / ``activities`` index as built for
            :meth:`get_subject_data`.

        Raises:
            FileNotFoundError: If root or manifest is missing.
            RuntimeError: If config is not loaded.
            ValueError: If the manifest is invalid or inconsistent.
        """
        if not self._root_path.is_dir():
            raise FileNotFoundError(
                f"DSA root is not a directory: {self._root_path}"
            )
        cfg = self._config_or_raise()
        rel_manifest = cfg.tables[DSA_TABLE_KEY].file_path
        manifest_path = self._root_path / rel_manifest
        if not manifest_path.is_file():
            raise FileNotFoundError(f"DSA manifest not found: {manifest_path}")

        df = pd.read_csv(manifest_path)
        self._validate_manifest_dataframe(df, manifest_path)
        metadata: Dict[str, Any] = {"subjects": {}, "activities": {}}

        for (sid, aname), group in df.groupby(
            ["subject_id", "activity_name"],
            sort=True,
        ):
            sid = str(sid)
            aname = str(aname)
            codes = group["activity_code"].astype(str).unique().tolist()
            if len(codes) != 1:
                raise ValueError(
                    f"Manifest rows for subject {sid!r}, activity {aname!r} "
                    f"disagree on activity_code: {codes}"
                )
            code = codes[0]
            paths = [Path(str(x)) for x in group["segment_path"].tolist()]
            for p in paths:
                rp = self._resolved_segment_path(p.as_posix())
                if not rp.is_file():
                    raise FileNotFoundError(
                        f"Segment file listed in manifest does not exist: {rp}"
                    )
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
        """Internal: load manifest-backed multivariate segments for one subject."""
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
        """Parse one DSA ``.txt`` segment into a numeric array and list-style metadata.

        Args:
            file_path: Absolute path to the segment file.
            subject_id: Subject id (e.g. ``p1``).
            activity: Human-readable activity label for this segment.

        Returns:
            Dict with ``data`` (``numpy.ndarray``), ``segment_filename``, channel
            metadata copies, and shape fields.

        Raises:
            ValueError: If the file is empty, row/column counts disagree with config,
                or floats fail to parse.
        """
        with open(file_path, encoding="utf-8") as f:
            lines = [ln for ln in f.readlines() if ln.strip()]

        if not lines:
            raise ValueError(f"Segment file is empty or whitespace-only: {file_path}")

        if self._num_rows is not None and len(lines) != self._num_rows:
            raise ValueError(
                f"{file_path} has {len(lines)} non-empty lines, expected "
                f"{self._num_rows} per config data_structure.num_rows."
            )

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
            "segment_filename": file_path.name,
        }

    def get_subject_ids(self) -> List[str]:
        """Return sorted subject ids from the manifest."""
        if self._metadata is None:
            self._metadata = self._load_metadata()
        return sorted(self._metadata["subjects"].keys())

    def get_activity_labels(self) -> Dict[str, int]:
        """Map activity names (from config) to integer class indices.

        Indices follow ascending numeric activity code (``A1``, ``A2``, …).

        Returns:
            Mapping from human-readable activity label to class index.

        Raises:
            ValueError: If any ``label_mapping`` key is not of the form ``An`` with
                integer ``n``.
        """
        codes = sorted(self.label_mapping.keys(), key=_activity_code_sort_key)
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

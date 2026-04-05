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
_ACTIVITY_CODE_RE = re.compile(r"^A(\d+)$", re.IGNORECASE)


def _activity_code_sort_key(code: str) -> int:
    m = _ACTIVITY_CODE_RE.match(code.strip())
    if not m:
        raise ValueError(
            f"Invalid activity code {code!r} in label_mapping; expected like 'A1'."
        )
    return int(m.group(1))


def _load_dsa_yaml(config_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not config_path.is_file():
        raise FileNotFoundError(f"DSA config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    DatasetConfig.model_validate(
        {"version": raw["version"], "tables": raw["tables"]}
    )
    return raw, raw["dataset"]


def _scan_dsa_tree(root_path: Path, dsa_cfg: Dict[str, Any]) -> pd.DataFrame:
    task = dsa_cfg.get("task_configs", {}).get("activity_recognition", {})
    labels: Dict[str, str] = dict(task.get("label_mapping", {}))
    layout = dsa_cfg.get("layout", {}) or {}
    a_re = re.compile(layout.get("activity_dir_pattern", r"^a\d{2}$"))
    p_re = re.compile(layout.get("subject_dir_pattern", r"^p\d+$"))
    s_re = re.compile(layout.get("segment_file_pattern", r"^s\d+\.txt$"))
    rows: List[Dict[str, str]] = []

    for seg in sorted(root_path.glob("*/*/*.txt")):
        if not seg.is_file():
            continue
        rel = seg.relative_to(root_path)
        a, p, sname = rel.parts
        if not (a_re.match(a) and p_re.match(p) and s_re.match(sname)):
            continue
        m_num = re.fullmatch(r"a(\d{2})", a, flags=re.IGNORECASE)
        if not m_num:
            continue
        code = f"A{int(m_num.group(1))}"
        act_name = labels.get(code)
        if act_name is None:
            logger.warning("Skipping %s: no label_mapping for %s", rel, code)
            continue
        rows.append(
            {
                "subject_id": p,
                "activity_name": act_name,
                "activity_code": code,
                "segment_path": rel.as_posix(),
            }
        )

    if not rows:
        raise ValueError(
            f"No DSA segments under {root_path}; expected aXX/pY/sZZ.txt layout."
        )
    return pd.DataFrame(rows)


def _write_manifest_csv(
        root_path: Path, dsa_cfg: Dict[str, Any], manifest_path: Path
) -> None:
    if not root_path.is_dir():
        raise FileNotFoundError(f"DSA root is not a directory: {root_path}")
    logger.info("Writing DSA manifest to %s", manifest_path)
    _scan_dsa_tree(root_path, dsa_cfg).to_csv(manifest_path, index=False)


class DSADataset(BaseDataset):
    """Daily and Sports Activities (DSA) time-series dataset (Barshan & Altun, 2010).

    Recordings use five on-body IMU units (torso, two arms, two legs); each unit
    contributes nine columns per row (3-axis accelerometer, gyroscope, and
    magnetometer), so each segment row has 45 comma-separated values. The public
    release is sampled at 25 Hz; each ``.txt`` segment is typically 125 lines (about
    five seconds of data).

    On disk, activities live in folders ``a01`` through ``a19``, subjects in ``p1``
    through ``p8``, and segment files ``s01.txt``, ``s02.txt``, … under each
    subject. PyHealth maps ``aXX`` folder names to activity labels using the
    ``label_mapping`` in ``configs/dsa.yaml``.

    :class:`BaseDataset` reads a single tabular index of segments. The path to that
    CSV (by default ``dsa_manifest.csv`` next to the activity folders) is set in the
    YAML ``tables.dsa_segments.file_path`` entry. If that file is not present under
    ``root`` when you construct this class, the loader walks the tree, matches the
    same layout patterns as in the YAML, and writes the manifest. You can rebuild it
    later with :meth:`prepare_metadata`.

    Dataset is available at:
    https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities

    Citations:
        If you use this dataset, cite: Barshan, B., & Altun, K. (2010). Daily and
        Sports Activities [Dataset]. UCI Machine Learning Repository.
        https://doi.org/10.24432/C5C59F

    Args:
        root: Dataset root (activity folders; manifest created if missing).
        dataset_name: Defaults to ``internal_name`` from config or ``\"dsa\"``.
        config_path: Path to ``dsa.yaml`` (default: package ``configs/dsa.yaml``).
        cache_dir: Cache directory for :class:`BaseDataset`.
        num_workers: Parallel workers for base pipelines.
        dev: Passed to :class:`BaseDataset` (limits patients when building events).

    Examples:
        >>> from pyhealth.datasets import DSADataset
        >>> # DSADataset(root="/path/to/dsa").get_subject_ids()
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
        raw_cfg, dsa_cfg = _load_dsa_yaml(cfg_path)
        root_path = Path(root).expanduser().resolve()
        manifest_path = root_path / raw_cfg["tables"][DSA_TABLE_KEY]["file_path"]
        if not manifest_path.is_file():
            _write_manifest_csv(root_path, dsa_cfg, manifest_path)

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
        self._num_columns: int = int(ds.get("num_columns", 45))
        self._metadata: Optional[Dict[str, Any]] = None

    def prepare_metadata(self, root: str) -> None:
        """Scan ``root`` and overwrite the manifest CSV (``tables.dsa_segments``)."""
        root_path = Path(root).expanduser().resolve()
        manifest_path = root_path / self.config.tables[DSA_TABLE_KEY].file_path
        _write_manifest_csv(root_path, self._dsa_config, manifest_path)
        self._metadata = None

    def _load_metadata(self) -> Dict[str, Any]:
        manifest_path = (
                self._root_path / self.config.tables[DSA_TABLE_KEY].file_path
        )
        df = pd.read_csv(manifest_path)
        out: Dict[str, Any] = {"subjects": {}, "activities": {}}
        for (sid, aname), group in df.groupby(
                ["subject_id", "activity_name"],
                sort=True,
        ):
            sid, aname = str(sid), str(aname)
            code = str(group["activity_code"].iloc[0])
            paths = [Path(str(x)) for x in group["segment_path"].tolist()]
            subject_dir = self._root_path / paths[0].parent
            seg_names = sorted(p.name for p in paths)
            activity_dir = self._root_path / paths[0].parts[0]

            subj = out["subjects"].setdefault(sid, {"id": sid, "activities": {}})
            subj["activities"][aname] = {
                "id": code,
                "path": subject_dir,
                "segments": seg_names,
            }

            act = out["activities"].setdefault(
                aname,
                {"id": code, "path": activity_dir, "subjects": {}},
            )
            act["subjects"][sid] = seg_names

        return out

    def _ensure_metadata(self) -> Dict[str, Any]:
        if self._metadata is None:
            self._metadata = self._load_metadata()
        return self._metadata

    def get_subject_ids(self) -> List[str]:
        """Sorted subject ids from the manifest."""
        return sorted(self._ensure_metadata()["subjects"].keys())

    def get_activity_labels(self) -> Dict[str, int]:
        """Map activity name → class index (ordered by activity code A1, A2, …)."""
        codes = sorted(self.label_mapping.keys(), key=_activity_code_sort_key)
        return {self.label_mapping[c]: i for i, c in enumerate(codes)}

    def get_num_subjects(self) -> int:
        return len(self._ensure_metadata()["subjects"])

    def get_num_activities(self) -> int:
        return len(self._ensure_metadata()["activities"])

    def get_subject_data(self, subject_id: str) -> Dict[str, Any]:
        """Load all segment arrays for one subject (see config for channel layout)."""
        meta = self._ensure_metadata()
        if subject_id not in meta["subjects"]:
            raise ValueError(
                f"Subject {subject_id!r} not found under {self._root_path}"
            )

        subject_data: Dict[str, Any] = {"id": subject_id, "activities": {}}
        for activity_name, activity_info in meta["subjects"][subject_id][
            "activities"
        ].items():
            segs_out = [
                self._segment_payload(
                    Path(activity_info["path"]) / fn,
                    subject_id,
                    activity_name,
                )
                for fn in activity_info["segments"]
            ]
            subject_data["activities"][activity_name] = {
                "id": activity_info["id"],
                "segments": segs_out,
            }
        return subject_data

    def _segment_payload(
            self,
            file_path: Path,
            subject_id: str,
            activity: str,
    ) -> Dict[str, Any]:
        data = np.loadtxt(file_path, delimiter=",", dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        n_rows, n_cols = data.shape
        if n_cols != self._num_columns:
            raise ValueError(
                f"{file_path} has {n_cols} columns, expected {self._num_columns}"
            )
        return {
            "file_path": file_path,
            "subject_id": subject_id,
            "activity": activity,
            "data": data,
            "num_samples": n_rows,
            "sampling_rate": self.sampling_frequency,
            "units": self.units,
            "sensors": self.sensors,
            "segment_filename": file_path.name,
        }

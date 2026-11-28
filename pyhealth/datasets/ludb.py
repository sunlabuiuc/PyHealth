"""LUDB frozen-window dataset.

Contributors: Thi Van Trinh Tran and Zexi Li.
Task: four-class ECG delineation with explicit annotation ignore masks.
Reference: MedTsLLM, https://arxiv.org/abs/2408.07773.
Data: https://physionet.org/content/ludb/1.0.1/.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .base_dataset import BaseDataset
from .splitter import Split


@dataclass(frozen=True)
class LUDBManifestSplit(Split):
    """Use supplied patient membership, without drawing a new split.

    Parts are training then test, ordered numerically by patient and in source
    window order within each patient. No validation set or CV is introduced.

    Examples:
        >>> split = LUDBManifestSplit((2,), (10,))
        >>> [part.tolist() for part in split.split_indices({"10": [0], "2": [1]})]
        [[1], [0]]
    """

    train_patients: tuple[int, ...]
    test_patients: tuple[int, ...]

    def __post_init__(self) -> None:
        train, test = self.train_patients, self.test_patients
        if not train or not test:
            raise ValueError("Both train and test patient sets must be non-empty")
        if len(set(train + test)) != len(train + test):
            raise ValueError("Duplicate or overlapping patient IDs in frozen split")
        if any(not isinstance(pid, (int, np.integer)) or pid < 1
               for pid in train + test):
            raise ValueError("LUDB patient IDs must be positive integers")

    @classmethod
    def from_csv(cls, path: str | Path) -> "LUDBManifestSplit":
        """Read an explicit patient_id,split CSV; reject ambiguous membership.

        Examples:
            >>> split = LUDBManifestSplit.from_csv(  # doctest: +SKIP
            ...     "patient_split.csv",
            ... )
        """
        frame = pd.read_csv(path)
        if not {"patient_id", "split"} <= set(frame):
            raise ValueError("Patient split requires patient_id and split columns")
        ids = pd.to_numeric(frame["patient_id"], errors="raise")
        if ids.isna().any() or (ids % 1 != 0).any():
            raise ValueError("Patient IDs must be integral and non-missing")
        frame["patient_id"] = ids.astype("int64")
        if frame["patient_id"].duplicated().any():
            raise ValueError("Duplicate patient IDs in frozen split")
        if set(frame["split"]) != {"train", "test"}:
            raise ValueError("Patient split must contain exactly train and test")
        return cls(*(
            tuple(sorted(frame.loc[frame["split"] == part, "patient_id"].tolist()))
            for part in ("train", "test")
        ))

    def split_indices(
        self, patient_to_index: Mapping[str, Sequence[int]]
    ) -> list[np.ndarray]:
        """Return exact manifest membership in historical numeric patient order.

        Examples:
            >>> LUDBManifestSplit((1,), (2,)).split_indices({"1": [0], "2": [1]})
            [array([0]), array([1])]
        """
        lookup = {int(pid): indices for pid, indices in patient_to_index.items()}
        expected = set(self.train_patients + self.test_patients)
        if len(lookup) != len(patient_to_index) or set(lookup) != expected:
            raise ValueError("Task patients differ from the frozen patient split")
        return [np.asarray([
            index for pid in sorted(part) for index in lookup[pid]
        ], dtype=np.int64) for part in (self.train_patients, self.test_patients)]

    def to_dict(self) -> dict:
        """Describe the frozen membership for the native processed-cache key.

        Examples:
            >>> LUDBManifestSplit((1,), (2,)).to_dict()["kind"]
            'ludb_manifest'
        """
        return {
            "kind": "ludb_manifest",
            "train_patients": sorted(self.train_patients),
            "test_patients": sorted(self.test_patients),
        }


class LUDBDataset(BaseDataset):
    """Native dataset consuming externally prepared, frozen LUDB artifacts.

    Requires LUDB 1.0.1 WFDB records, a window manifest, patient split, and
    ``mask_root/masks/<patient_id>.npz`` (labels, ignore_mask, lead_names).
    This does not derive annotations, choose exclusions, resample signals,
    rebuild masks, or generate a different patient split. Missing artifacts
    are errors. Caches contain copies and belong in approved project storage.

    Args:
        manifest_path: Frozen window_manifest.csv with both train/test rows.
        records_root: Directory with numeric LUDB WFDB record stems.
        mask_root: Prepared artifact directory containing masks/.
        split_manifest_path: Frozen patient_split.csv.
        cache_dir: Explicit location for native PyHealth caches.
        metadata_path: Optional metadata_prepared.csv; all fields are retained.
        num_workers: Native task-processing workers.

    Examples:
        >>> dataset = LUDBDataset(  # doctest: +SKIP
        ...     "window_manifest.csv", "data", "prepared", "patient_split.csv",
        ...     cache_dir="project_cache", metadata_path="metadata_prepared.csv",
        ... )
        >>> train, test = dataset.set_task(  # doctest: +SKIP
        ...     split=dataset.manifest_split,
        ... )
    """

    def __init__(
        self,
        manifest_path: str | Path,
        records_root: str | Path,
        mask_root: str | Path,
        split_manifest_path: str | Path,
        cache_dir: str | Path,
        metadata_path: str | Path | None = None,
        num_workers: int = 1,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.records_root = Path(records_root).resolve()
        self.mask_root = Path(mask_root).resolve()
        self.metadata_path = Path(metadata_path).resolve() if metadata_path else None
        self.manifest_split = LUDBManifestSplit.from_csv(split_manifest_path)
        frame = pd.read_csv(self.manifest_path)
        required = {
            "patient_id", "lead", "lead_index", "split", "window_index",
            "start_sample", "end_sample_exclusive", "window_length",
            "has_supervised_points",
        }
        if not required <= set(frame) or frame.empty:
            raise ValueError(
                f"Empty manifest or missing columns: {required - set(frame)}"
            )
        for name in ("patient_id", "lead_index", "window_index", "start_sample",
                     "end_sample_exclusive", "window_length"):
            values = pd.to_numeric(frame[name], errors="raise")
            if values.isna().any() or (values % 1 != 0).any():
                raise ValueError(f"Manifest {name} must contain non-missing integers")
        expected = {
            **dict.fromkeys(self.manifest_split.train_patients, "train"),
            **dict.fromkeys(self.manifest_split.test_patients, "test"),
        }
        if set(frame["patient_id"].astype(int)) != set(expected):
            raise ValueError("Manifest patients differ from patient_split.csv")
        for row in frame.itertuples():
            if row.split != expected[int(row.patient_id)]:
                raise ValueError("Manifest split disagrees with patient_split.csv")
        if frame.duplicated(["patient_id", "lead_index", "window_index"]).any():
            raise ValueError("Duplicate patient/lead/window IDs in manifest")
        if not self.records_root.is_dir() or not (self.mask_root / "masks").is_dir():
            raise FileNotFoundError("Missing WFDB records directory or masks directory")
        super().__init__(
            root=str(self.manifest_path.parent), tables=["windows"],
            config_path=str(Path(__file__).parent / "configs" / "ludb.yaml"),
            cache_dir=cache_dir, num_workers=num_workers,
        )

    def _init_cache_dir(self, cache_dir: str | Path | None) -> Path:
        # Resolve before BaseDataset computes config + source-file cache keys.
        self.config.tables["windows"].file_path = self.manifest_path.name
        return super()._init_cache_dir(cache_dir)

    @property
    def default_task(self):
        """Create the ignore-aware task without altering source artifacts.

        Examples:
            >>> task = dataset.default_task  # doctest: +SKIP
        """
        from pyhealth.tasks.ecg_delineation_ludb import ECGDelineationLUDB

        return ECGDelineationLUDB(
            self.records_root, self.mask_root, self.metadata_path,
            patient_ids=self.manifest_split.train_patients
            + self.manifest_split.test_patients,
        )

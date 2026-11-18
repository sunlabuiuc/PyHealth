"""Frozen-label LUDB ECG delineation task.

Contributors: Thi Van Trinh Tran and Zexi Li.
Reference: MedTsLLM, https://arxiv.org/abs/2408.07773.
Dataset: https://physionet.org/content/ludb/1.0.1/.
"""

import hashlib
import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import torch

from pyhealth.data import Patient

from .base_task import BaseTask


class ECGDelineationLUDB(BaseTask):
    """Materialize 512-point windows, preserving frozen labels and ignores.

    ``label`` contains only valid positions for the native time-series label
    processor. ``dense_label`` and ``ignore_mask`` retain the full original
    window for independent audits. Dense targets are outputs, never features.
    All-ignored windows are retained with an empty sparse label. No filtering,
    resampling, padding, or dataset-level normalization is applied here.

    Metadata is optional. When supplied, every CSV field is serialized exactly
    as in the research runner (sorted JSON keys, missing values become null).
    Selecting clinical-only fields would be a separate experimental change.

    Args:
        records_root: Directory of LUDB 1.0.1 WFDB records (5000 x 12, 500 Hz).
        mask_root: Directory containing masks/<patient_id>.npz.
        metadata_path: Optional metadata CSV containing unique integral ID.
        patient_ids: All expected IDs; their artifacts enter task cache keys.

    Examples:
        >>> task = ECGDelineationLUDB(  # doctest: +SKIP
        ...     "data", "prepared", patient_ids=[1, 2],
        ... )
        >>> ECGDelineationLUDB.output_schema["label"][0]
        'timeseries_multiclass'
    """

    task_name = "ecg_delineation_ludb"
    input_schema: ClassVar[dict] = {
        "signal": ("tensor", {"dtype": torch.float32}),
        "ignore_mask": ("tensor", {"dtype": torch.bool}),
        "patient_context": "raw",
        "lead_index": "raw",
        "start_sample": "raw",
        "end_sample_exclusive": "raw",
        "manifest_order": "raw",
    }
    output_schema: ClassVar[dict] = {
        # The label's times are sample indices of the signal window.
        "label": (
            "timeseries_multiclass",
            {"labels": [0, 1, 2, 3], "timeline": ["signal"]},
        ),
        "dense_label": ("tensor", {"dtype": torch.long}),
    }

    def __init__(
        self,
        records_root: str | Path,
        mask_root: str | Path,
        metadata_path: str | Path | None = None,
        *, patient_ids: tuple[int, ...] | list[int],
    ) -> None:
        self.task_version = "frozen-ludb-windows-v1"
        self.records_root = str(Path(records_root).resolve())
        self.mask_root = str(Path(mask_root).resolve())
        self.metadata_path = (
            str(Path(metadata_path).resolve()) if metadata_path else None
        )
        self.patient_ids = tuple(sorted(patient_ids))
        if not self.patient_ids or len(set(patient_ids)) != len(patient_ids) or any(
            not isinstance(pid, (int, np.integer)) or pid < 1 for pid in patient_ids
        ):
            raise ValueError(
                "Supply all expected patient IDs as unique positive integers"
            )
        self.contexts = self._load_contexts()
        # Contents of masks/metadata and record stats enter vars(task), hence
        # native cache keys. Nothing is written into the raw source folders.
        paths = [Path(self.mask_root) / "masks" / f"{pid}.npz"
                 for pid in self.patient_ids]
        if self.metadata_path:
            paths.append(Path(self.metadata_path))
        self.artifact_hashes = tuple(
            (str(path), hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths
        )
        self.record_fingerprints = tuple(
            (str(path), path.stat().st_size, path.stat().st_mtime_ns)
            for pid in self.patient_ids for ext in ("hea", "dat")
            for path in [Path(self.records_root) / f"{pid}.{ext}"]
        )

    def _load_contexts(self) -> dict[int, str]:
        if self.metadata_path is None:
            return {}
        table = pd.read_csv(self.metadata_path)
        if "ID" not in table or table["ID"].duplicated().any():
            raise ValueError("Metadata requires unique ID values")
        ids = pd.to_numeric(table["ID"], errors="raise")
        if ids.isna().any() or (ids % 1 != 0).any():
            raise ValueError("Metadata ID values must be non-missing integers")
        return {
            int(row["ID"]): json.dumps(
                {str(key): None if pd.isna(value) else value
                 for key, value in row.to_dict().items()},
                sort_keys=True, separators=(",", ":"), default=str,
            )
            for _, row in table.iterrows()
        }

    def __call__(self, patient: Patient) -> list[dict]:
        """Return native samples without turning ignored points into background.

        Examples:
            >>> windows = task(patient)  # doctest: +SKIP
        """
        try:
            import wfdb
        except ImportError as exc:
            raise ImportError(
                "LUDB signal reading requires the optional wfdb package"
            ) from exc
        pid = int(patient.patient_id)
        if pid not in self.patient_ids:
            raise ValueError(f"Patient {pid} is not in the supplied frozen cohort")
        events = list(patient.get_events(event_type="windows"))
        if not events:
            raise ValueError(f"Patient {pid} has no manifest windows")
        record = wfdb.rdrecord(str(Path(self.records_root) / str(pid)))
        signal = np.asarray(record.p_signal)
        if signal.shape != (5000, 12) or not np.isclose(record.fs, 500.0):
            raise ValueError(f"Patient {pid}: expected 5000x12 samples at 500 Hz")
        if not np.isfinite(signal).all():
            raise ValueError(f"Patient {pid}: non-finite signal values")
        with np.load(Path(self.mask_root) / "masks" / f"{pid}.npz",
                     allow_pickle=False) as stored:
            labels = np.asarray(stored["labels"]).copy()
            ignore = np.asarray(stored["ignore_mask"]).copy()
            names = [str(value).lower() for value in stored["lead_names"].tolist()]
        if labels.shape != (12, 5000) or ignore.shape != labels.shape:
            raise ValueError(f"Patient {pid}: masks must be 12x5000")
        if not np.issubdtype(labels.dtype, np.integer) or (
            labels.min() < 0 or labels.max() > 3
        ):
            raise ValueError(f"Patient {pid}: labels must be integer classes 0..3")
        if not np.isin(ignore, [False, True]).all():
            raise ValueError(f"Patient {pid}: ignore_mask must be binary")
        if names != [str(name).lower() for name in record.sig_name]:
            raise ValueError(f"Patient {pid}: WFDB/mask lead orders disagree")
        if self.metadata_path and pid not in self.contexts:
            raise ValueError(f"Patient {pid}: metadata is missing")
        samples = []
        for order, event in enumerate(events):
            lead, start, end = (
                int(event.lead_index), int(event.start_sample),
                int(event.end_sample_exclusive),
            )
            if not 0 <= lead < 12 or str(event.lead).lower() != names[lead]:
                raise ValueError(f"Patient {pid}: manifest lead disagrees with masks")
            if not 0 <= start < end <= 5000 or end - start != 512 or (
                int(event.window_length) != 512
            ):
                raise ValueError(f"Patient {pid}: invalid 512-point window bounds")
            dense = labels[lead, start:end].astype(np.int64, copy=True)
            ignored = ignore[lead, start:end].astype(bool, copy=True)
            valid = ~ignored
            flag = str(event.has_supervised_points).strip().lower()
            if flag not in {"true", "false", "1", "0", "yes", "no"}:
                raise ValueError("Invalid has_supervised_points flag")
            if (flag in {"true", "1", "yes"}) != bool(valid.any()):
                raise ValueError("Manifest supervision flag disagrees with frozen mask")
            samples.append({
                "patient_id": str(pid),
                "record_id": f"{pid}:{event.lead}:{int(event.window_index)}",
                "signal": signal[start:end, lead].astype(np.float32, copy=True),
                "dense_label": dense,
                "ignore_mask": ignored,
                "label": {"time": np.flatnonzero(valid).tolist(),
                          "values": dense[valid].tolist()},
                "patient_context": self.contexts.get(pid, ""),
                "lead_index": lead, "start_sample": start,
                "end_sample_exclusive": end, "manifest_order": order,
            })
        return samples

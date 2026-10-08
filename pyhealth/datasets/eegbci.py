from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import mne
import pandas as pd

from .base_dataset import BaseDataset
from pyhealth.tasks.eegbci import EEGMotorImageryEEGBCI, run_type_for_run

logger = logging.getLogger(__name__)

EEGBCI_METADATA_COLUMNS = {
    "patient_id",
    "record_id",
    "subject_id",
    "run",
    "run_type",
    "signal_file",
    "source",
}


class EEGBCIDataset(BaseDataset):
    """PhysioNet EEG Motor Movement/Imagery metadata dataset.

    The source dataset is PhysioNet's EEG Motor Movement/Imagery Dataset
    (``eegmmidb``), version 1.0.0, licensed under the
    Open Data Commons Attribution License v1.0. Cite Schalk (2009),
    https://doi.org/10.13026/C28G6P.

    Args:
        root: Directory containing or receiving EEGBCI EDF files and metadata.
        dataset_name: Optional dataset name prefix. Defaults to ``"eegbci"``.
        config_path: Optional dataset configuration path.
        subjects: Subject identifiers to include. Defaults to ``[1, 2, 3]``.
        runs: Run identifiers to include. Defaults to runs 3 through 14.
        download: Whether MNE may download missing EDF files.
        **kwargs: Additional arguments forwarded to :class:`BaseDataset`.

    Raises:
        FileNotFoundError: If a requested EDF is unavailable and downloading is
            disabled.

    Examples:
        >>> dataset = EEGBCIDataset(
        ...     root="/path/to/eegbci", subjects=[1], runs=[3], download=True
        ... )
        >>> dataset.stats()
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        subjects: Optional[list[int]] = None,
        runs: Optional[list[int]] = None,
        download: bool = False,
        **kwargs,
    ) -> None:
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "eegbci.yaml"
        self.root = root
        self.subjects = self._normalize_selection(
            list(subjects) if subjects is not None else [1, 2, 3]
        )
        self.runs = self._normalize_selection(
            list(runs) if runs is not None else list(range(3, 15))
        )
        self.download = download
        self.selection_key = self._build_selection_key()
        self.metadata_file_name = self._metadata_file_name()
        self.prepare_metadata()
        metadata_key = self._metadata_cache_key()
        dataset_name = dataset_name or "eegbci"
        super().__init__(
            root=root,
            tables=["records"],
            dataset_name=f"{dataset_name}_{self.selection_key}_{metadata_key}",
            config_path=config_path,
            **kwargs,
        )
        if self.config is not None:
            self.config.tables["records"].file_path = self.metadata_file_name

    @staticmethod
    def _normalize_selection(values: list[int]) -> list[int]:
        """Normalize identifiers to sorted, unique integers.

        Args:
            values: Subject or run identifiers.

        Returns:
            The normalized identifiers.
        """
        return sorted({int(value) for value in values})

    def _build_selection_key(self) -> str:
        """Build a stable cache identity for the subject/run selection.

        Returns:
            The selection key.
        """
        payload = {
            "subjects": [int(subject) for subject in self.subjects],
            "runs": [int(run) for run in self.runs],
        }
        digest = hashlib.sha1(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:10]
        subject_part = "-".join(f"{int(subject):03d}" for subject in self.subjects)
        run_part = "-".join(f"{int(run):02d}" for run in self.runs)
        return f"s{subject_part}_r{run_part}_{digest}"

    def _metadata_file_name(self) -> str:
        """Return the selection-specific metadata filename.

        Returns:
            The CSV filename.
        """
        return f"eegbci-pyhealth-{self.selection_key}.csv"

    def _metadata_cache_key(self) -> str:
        """Return a content fingerprint for derived dataset caches.

        Returns:
            A stable fingerprint of the selection-specific metadata CSV.
        """
        csv_path = Path(self.root) / self.metadata_file_name
        return hashlib.sha1(csv_path.read_bytes()).hexdigest()[:10]

    def _find_local_edf(self, subject: int, run: int) -> Path | None:
        """Find the canonical EDF path before a recursive fallback search.

        Args:
            subject: PhysioNet subject identifier.
            run: EEGBCI run identifier.

        Returns:
            The local EDF path, or ``None`` when no matching file exists.
        """
        root = Path(self.root)
        filename = f"S{subject:03d}R{run:02d}.edf"
        canonical_path = (
            root / "files" / "eegmmidb" / "1.0.0" / f"S{subject:03d}" / filename
        )
        if canonical_path.exists():
            return canonical_path
        matches = sorted(root.rglob(filename))
        return matches[0] if matches else None

    def _requested_pairs(self) -> list[tuple[int, int]]:
        """Return requested subject/run pairs in stable order.

        Returns:
            The sorted subject/run pairs.
        """
        return sorted(
            (int(subject), int(run))
            for subject in self.subjects
            for run in self.runs
        )

    def _metadata_matches_request(self, csv_path: Path) -> bool:
        """Check whether cached metadata can be safely reused.

        Valid metadata has the required columns and requested subject/run pairs,
        and every referenced EDF path still exists.

        Args:
            csv_path: Metadata CSV path.

        Returns:
            Whether the cached metadata is reusable.
        """
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return False
        if not EEGBCI_METADATA_COLUMNS.issubset(df.columns):
            return False
        pairs = sorted((int(row.subject_id), int(row.run)) for row in df.itertuples())
        if pairs != self._requested_pairs():
            return False
        return all(Path(str(row.signal_file)).is_file() for row in df.itertuples())

    def prepare_metadata(self) -> None:
        """Reuse valid metadata or write rows for every requested EDF.

        Raises:
            FileNotFoundError: If a requested EDF is unavailable and downloading
                is disabled.
        """
        root = Path(self.root)
        csv_path = root / self.metadata_file_name
        if csv_path.exists() and self._metadata_matches_request(csv_path):
            return

        rows: list[dict] = []
        for subject in self.subjects:
            paths_by_run: dict[int, Path] = {}
            if self.download:
                downloaded = mne.datasets.eegbci.load_data(
                    subject, self.runs, path=str(root), update_path=False
                )
                for path in downloaded:
                    p = Path(path)
                    for run in self.runs:
                        if p.name == f"S{subject:03d}R{run:02d}.edf":
                            paths_by_run[run] = p
            for run in self.runs:
                signal_file = paths_by_run.get(run) or self._find_local_edf(subject, run)
                if signal_file is None:
                    raise FileNotFoundError(
                        f"Missing EEGBCI EDF for subject {subject}, run {run}. "
                        "Pass download=True to fetch it with MNE."
                    )
                rows.append(
                    {
                        "patient_id": f"S{subject:03d}",
                        "record_id": f"R{run:02d}",
                        "subject_id": int(subject),
                        "run": int(run),
                        "run_type": run_type_for_run(run),
                        "signal_file": str(signal_file),
                        "source": "physionet_eegbci",
                    }
                )

        df = pd.DataFrame(rows)
        df.sort_values(["subject_id", "run"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info("Wrote EEGBCI metadata to %s", csv_path)

    @property
    def default_task(self) -> EEGMotorImageryEEGBCI:
        """Return the canonical supervised EEGBCI task.

        Returns:
            An :class:`EEGMotorImageryEEGBCI` task.
        """
        return EEGMotorImageryEEGBCI()

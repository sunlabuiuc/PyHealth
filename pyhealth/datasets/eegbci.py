from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import mne
import pandas as pd

from .base_dataset import BaseDataset
from pyhealth.tasks.eegbci import EEGBCIPatternDiscovery, run_type_for_run

logger = logging.getLogger(__name__)


class EEGBCIDataset(BaseDataset):
    """PhysioNet EEG Motor Movement/Imagery metadata dataset."""

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
        self.subjects = subjects or [1, 2, 3]
        self.runs = runs or list(range(3, 15))
        self.download = download
        self.prepare_metadata()
        super().__init__(
            root=root,
            tables=["records"],
            dataset_name=dataset_name or "eegbci",
            config_path=config_path,
            **kwargs,
        )

    def _find_local_edf(self, subject: int, run: int) -> Path | None:
        root = Path(self.root)
        pattern = f"S{subject:03d}R{run:02d}.edf"
        matches = sorted(root.rglob(pattern))
        return matches[0] if matches else None

    def prepare_metadata(self) -> None:
        root = Path(self.root)
        csv_path = root / "eegbci-pyhealth.csv"
        if csv_path.exists():
            return

        rows: list[dict] = []
        for subject in self.subjects:
            paths_by_run: dict[int, Path] = {}
            if self.download:
                downloaded = mne.datasets.eegbci.load_data(
                    subject, self.runs, path=str(root)
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
    def default_task(self) -> EEGBCIPatternDiscovery:
        return EEGBCIPatternDiscovery()

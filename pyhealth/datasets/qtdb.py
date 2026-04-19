import os
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base_dataset import BaseDataset

QTDB_PHYSIONET_URL = "https://physionet.org/files/qtdb/1.0.0/"
DEFAULT_DOWNLOAD_ROOT = Path.home() / ".cache" / "pyhealth" / "datasets"
NUM_RECORDS = 105
DEV_NUM_RECORDS = 10


class QTDBDataset(BaseDataset):
    """Base ECG dataset for the PhysioNet QT Database (QTDB).

    QTDB contains 105 two-lead 15-minute ECG recordings with multiple
    annotation files (reference beat annotations and waveform delineation
    outputs from manual/automatic methods).

    Dataset:
        https://physionet.org/content/qtdb/1.0.0/

    Design:
        - One table: ``qtdb``
        - One row/event per record
        - ``patient_id`` is the record ID (e.g., ``sel100``)

    Args:
        root: Local path to QTDB data directory, or any parent directory to probe.
            If ``download=True`` and data is not found locally, QTDB is downloaded
            to ``Path.home() / ".cache" / "pyhealth" / "datasets"``.
        dataset_name: Optional dataset name. Defaults to ``"qtdb"``.
        config_path: Optional YAML config path. Defaults to
            ``pyhealth/datasets/configs/qtdb.yaml``.
        dev: If True, keep only first 10 records.
        download: If True and local data is missing, download QTDB using:
            ``wget -r -N -c -np https://physionet.org/files/qtdb/1.0.0/``.
        refresh_cache: Deprecated legacy argument; accepted for compatibility
            but ignored.
        **kwargs: Forwarded to :class:`BaseDataset` (e.g., ``cache_dir``,
            ``num_workers``).

    Examples:
        >>> from pyhealth.datasets import QTDBDataset
        >>> ds = QTDBDataset(root="~/.cache/pyhealth/datasets", download=True, dev=True)
        >>> ds.stats()
        >>> ds.info()
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        dev: bool = False,
        download: bool = False,
        refresh_cache: Optional[bool] = None,
        **kwargs,
    ) -> None:
        if refresh_cache is not None:
            warnings.warn(
                "`refresh_cache` is deprecated for QTDBDataset with BaseDataset and is ignored.",
                DeprecationWarning,
                stacklevel=2,
            )

        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "qtdb.yaml"
            )

        data_root = self._resolve_data_root(root=root, download=download)
        metadata_root = self._prepare_metadata(root=data_root)

        super().__init__(
            root=metadata_root,
            tables=["qtdb"],
            dataset_name=dataset_name or "qtdb",
            config_path=config_path,
            dev=dev,
            **kwargs,
        )

    # ---------------------------------------------------------------------
    # Backward-compat convenience methods
    # ---------------------------------------------------------------------
    def stat(self) -> None:
        """Backward-compatible alias for :meth:`stats`."""
        warnings.warn(
            "`stat()` is deprecated; use `stats()` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.stats()

    def info(self) -> None:
        """Print QTDB summary for the BaseDataset/event-table layout."""
        print(f"Dataset: {self.dataset_name}")
        print("Backend: BaseDataset")
        print("Event table: qtdb (one event per QTDB record)")
        print(f"Root (metadata source): {self.root}")
        print(f"Tables: {self.tables}")
        print(f"Dev mode: {self.dev}")
        print("Lead columns: lead_0, lead_1")
        print("Use `dataset.stats()` for patient/event counts.")

    def preprocess_qtdb(self, df: Any) -> Any:
        """Apply QTDB-specific preprocessing before event caching."""
        if not self.dev:
            return df
        # Keep first DEV_NUM_RECORDS rows after deterministic lexical sort.
        return df.sort("patient_id").head(DEV_NUM_RECORDS)

    # ---------------------------------------------------------------------
    # Data root + download helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _candidate_data_roots(root: str) -> List[Path]:
        """Return possible QTDB data directories for local/downloaded layouts."""
        root_path = Path(root).expanduser().resolve()
        return [
            root_path,
            root_path / "qtdb" / "1.0.0",
            root_path / "physionet.org" / "files" / "qtdb" / "1.0.0",
            root_path / "files" / "qtdb" / "1.0.0",
        ]

    @staticmethod
    def _looks_like_qtdb_data_dir(path: Path) -> bool:
        """Check whether a directory looks like QTDB."""
        if not path.exists() or not path.is_dir():
            return False

        if (path / "RECORDS").exists():
            return True

        try:
            return any(
                p.is_file() and p.name.endswith(".hea") and not p.name.endswith(".hea-")
                for p in path.iterdir()
            )
        except OSError:
            return False

    @staticmethod
    def _download_qtdb(download_root: Path) -> None:
        """Download QTDB from PhysioNet using wget recursive mirror command."""
        cmd = ["wget", "-r", "-N", "-c", "-np", QTDB_PHYSIONET_URL]
        try:
            subprocess.run(cmd, cwd=str(download_root), check=True)
        except FileNotFoundError as e:
            raise RuntimeError(
                "wget is not installed or not in PATH. Please install wget and retry."
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"QTDB download failed via wget command: {' '.join(cmd)}"
            ) from e

    @classmethod
    def _resolve_data_root(cls, root: str, download: bool) -> str:
        """Resolve QTDB data directory, optionally downloading from PhysioNet."""
        for candidate in cls._candidate_data_roots(root):
            if cls._looks_like_qtdb_data_dir(candidate):
                return str(candidate)

        if not download:
            raise FileNotFoundError(
                "QTDB data files not found. Provide a valid local QTDB path "
                "or set download=True to fetch from PhysioNet."
            )

        download_root = DEFAULT_DOWNLOAD_ROOT.resolve()
        download_root.mkdir(parents=True, exist_ok=True)
        cls._download_qtdb(download_root)

        for candidate in cls._candidate_data_roots(str(download_root)):
            if cls._looks_like_qtdb_data_dir(candidate):
                return str(candidate)

        raise FileNotFoundError(
            "QTDB download completed but data directory could not be resolved."
        )

    # ---------------------------------------------------------------------
    # Metadata helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _read_records_list(root_path: Path) -> List[str]:
        """Read record IDs from RECORDS file, fallback to .hea discovery."""
        records_file = root_path / "RECORDS"
        if records_file.exists():
            records = [
                line.strip()
                for line in records_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if records:
                return records

        records = []
        for f in root_path.glob("*.hea"):
            if f.name.endswith(".hea") and not f.name.endswith(".hea-"):
                records.append(f.stem)
        return sorted(set(records))

    @staticmethod
    def _ann_path_or_none(base: Path, ext: str) -> Optional[str]:
        ann = base.with_suffix(f".{ext}")
        return str(ann) if ann.exists() else None

    def _prepare_metadata(self, root: str) -> str:
        """Build ``qtdb-pyhealth.csv`` if missing and return metadata root path."""
        root_path = Path(root).expanduser().resolve()
        metadata_path = root_path / "qtdb-pyhealth.csv"
        if metadata_path.exists():
            return str(root_path)

        records = self._read_records_list(root_path)
        if not records:
            # conservative fallback if directory probing happens before actual files exist
            records = [f"record_{i:03d}" for i in range(NUM_RECORDS)]

        rows: List[Dict[str, object]] = []
        for rec in records:
            base = root_path / rec
            rows.append(
                {
                    "patient_id": rec,
                    "visit_id": "ecg",
                    "record_id": rec,
                    # absolute WFDB base record path (without extension)
                    "signal_file": str(base),
                    # QTDB has two leads per record
                    "lead_0": "0",
                    "lead_1": "1",
                    # commonly used QTDB annotation files
                    "ann_atr": self._ann_path_or_none(base, "atr"),
                    "ann_man": self._ann_path_or_none(base, "man"),
                    "ann_qt1": self._ann_path_or_none(base, "qt1"),
                    "ann_q1c": self._ann_path_or_none(base, "q1c"),
                    "ann_pu": self._ann_path_or_none(base, "pu"),
                    "ann_pu0": self._ann_path_or_none(base, "pu0"),
                    "ann_pu1": self._ann_path_or_none(base, "pu1"),
                }
            )

        df = pd.DataFrame(rows)
        df.sort_values(["patient_id"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        try:
            df.to_csv(metadata_path, index=False)
            return str(root_path)
        except (PermissionError, OSError):
            cache_root = Path.home() / ".cache" / "pyhealth" / "qtdb"
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_metadata_path = cache_root / "qtdb-pyhealth.csv"
            df.to_csv(cache_metadata_path, index=False)
            return str(cache_root)

    @property
    def default_task(self):
        """No default task is enforced for QTDB."""
        return None


if __name__ == "__main__":
    cache_root = (
        Path.home()
        / ".cache"
        / "pyhealth"
        / "datasets"
        / "physionet.org"
        / "files"
        / "qtdb"
        / "1.0.0"
    )
    data_root = os.environ.get("DATA_ROOT", str(cache_root))

    dataset = QTDBDataset(root=data_root, dev=True, download=True)
    dataset.stats()
    dataset.info()

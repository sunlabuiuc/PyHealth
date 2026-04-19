import os
import random
import subprocess
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .base_dataset import BaseDataset

LEADS = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]
NUM_RECORDS = 200
DEV_NUM_RECORDS = 10
FS = 500
N_SAMPLES = 5000
LUDB_PHYSIONET_URL = "https://physionet.org/files/ludb/1.0.1/"
DEFAULT_DOWNLOAD_ROOT = Path.home() / ".cache" / "pyhealth" / "datasets"


class LUDBDataset(BaseDataset):
    """Base ECG dataset for the Lobachevsky University Database (LUDB).

    Modernized LUDB dataset built on :class:`pyhealth.datasets.BaseDataset`.

    Design:
        - One table: ``ludb``
        - One row/event per LUDB record (record ID 1..200)
        - ``patient_id`` is the record ID
        - 12 lead annotation files are represented as 12 columns:
          ``lead_i``, ``lead_ii``, ..., ``lead_v6``

    Notes:
        - Pulse-level splitting is intentionally handled in task logic
          (e.g., ECG delineation task), not at base dataset level.
        - Metadata CSV (``ludb-pyhealth.csv``) is generated automatically if
          missing.

    Dataset:
        https://physionet.org/content/ludb/1.0.1/

    Args:
        root: Local root path for LUDB. This can be:
            - the LUDB ``data/`` directory containing ``*.hea``, ``*.dat``,
              and ``*.<lead>`` files, or
            - any parent directory to probe for existing LUDB files.
              If ``download=True`` and local files are not found, LUDB is
              downloaded to ``Path.home() / ".cache" / "pyhealth" / "datasets"``.
        dataset_name: Optional dataset name. Defaults to ``"ludb"``.
        config_path: Optional YAML config path. Defaults to
            ``pyhealth/datasets/configs/ludb.yaml``.
        dev: If True, uses only the first 10 records.
        download: If True and LUDB files are not found locally, downloads LUDB
            from PhysioNet using:
            ``wget -r -N -c -np https://physionet.org/files/ludb/1.0.1/``.
        refresh_cache: Deprecated compatibility argument from legacy signal API.
            It is accepted for backward compatibility but ignored under
            BaseDataset.
        **kwargs: Forwarded to :class:`BaseDataset` (e.g., ``cache_dir``,
            ``num_workers``).

    Examples:
        >>> from pyhealth.datasets import LUDBDataset
        >>> dataset = LUDBDataset(root="/path/to/physionet.org/files/ludb/1.0.1/data")
        >>> dataset.stats()  # BaseDataset-native dataset statistics
        >>> dataset.info()   # LUDB-specific summary (table/leads/paths)
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
                "`refresh_cache` is deprecated for LUDBDataset with BaseDataset and is ignored.",
                DeprecationWarning,
                stacklevel=2,
            )

        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "ludb.yaml"
            )

        data_root = self._resolve_data_root(root=root, download=download)
        metadata_root = self._prepare_metadata(root=data_root)

        super().__init__(
            root=metadata_root,
            tables=["ludb"],
            dataset_name=dataset_name or "ludb",
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
        """Print LUDB summary for the modern BaseDataset/event-table layout."""
        print(f"Dataset: {self.dataset_name}")
        print("Backend: BaseDataset")
        print("Event table: ludb (one event per LUDB record)")
        print(f"Root (metadata source): {self.root}")
        print(f"Tables: {self.tables}")
        print(f"Dev mode: {self.dev}")
        print(f"Lead columns: {', '.join(f'lead_{lead}' for lead in LEADS)}")
        print("Use `dataset.stats()` for patient/event counts.")

    def preprocess_ludb(self, df: Any) -> Any:
        """Apply LUDB-specific preprocessing before event caching."""
        if not self.dev:
            return df
        try:
            import importlib

            nw = importlib.import_module("narwhals")
        except Exception:
            return df
        return df.filter(nw.col("patient_id").cast(nw.Int64) <= DEV_NUM_RECORDS)

    # ---------------------------------------------------------------------
    # Metadata generation
    # ---------------------------------------------------------------------
    @staticmethod
    def _candidate_data_roots(root: str) -> List[Path]:
        """Return possible LUDB data directories for local or downloaded layouts."""
        root_path = Path(root).expanduser().resolve()
        return [
            root_path,
            root_path / "data",
            root_path / "physionet.org" / "files" / "ludb" / "1.0.1" / "data",
            root_path / "files" / "ludb" / "1.0.1" / "data",
        ]

    @staticmethod
    def _looks_like_ludb_data_dir(path: Path) -> bool:
        """Check whether a directory looks like LUDB data/ (contains numeric .hea)."""
        if not path.exists() or not path.is_dir():
            return False
        try:
            return any(p.suffix == ".hea" and p.stem.isdigit() for p in path.iterdir())
        except OSError:
            return False

    @staticmethod
    def _download_ludb(download_root: Path) -> None:
        """Download LUDB from PhysioNet using wget recursive mirror command."""
        cmd = ["wget", "-r", "-N", "-c", "-np", LUDB_PHYSIONET_URL]
        try:
            subprocess.run(cmd, cwd=str(download_root), check=True)
        except FileNotFoundError as e:
            raise RuntimeError(
                "wget is not installed or not in PATH. Please install wget and retry."
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"LUDB download failed via wget command: {' '.join(cmd)}"
            ) from e

    @classmethod
    def _resolve_data_root(cls, root: str, download: bool) -> str:
        """Resolve LUDB data directory, optionally downloading from PhysioNet."""
        for candidate in cls._candidate_data_roots(root):
            if cls._looks_like_ludb_data_dir(candidate):
                return str(candidate)

        if not download:
            raise FileNotFoundError(
                "LUDB data files not found. Provide a valid local LUDB data path "
                "or set download=True to fetch from PhysioNet."
            )

        download_root = Path(DEFAULT_DOWNLOAD_ROOT).expanduser().resolve()
        download_root.mkdir(parents=True, exist_ok=True)
        cls._download_ludb(download_root)

        for candidate in cls._candidate_data_roots(str(download_root)):
            if cls._looks_like_ludb_data_dir(candidate):
                return str(candidate)

        raise FileNotFoundError(
            "LUDB download completed but data directory could not be resolved."
        )

    @staticmethod
    def _resolve_ludb_csv(root: str) -> Optional[str]:
        """Resolve path to PhysioNet ``ludb.csv`` if available."""
        root_path = Path(root)
        candidates = [
            root_path / "ludb.csv",
            root_path.parent / "ludb.csv",
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        return None

    @staticmethod
    def _discover_record_ids(root: str) -> List[int]:
        """Discover record IDs from ``*.hea`` files in LUDB data directory."""
        ids: List[int] = []
        for filename in os.listdir(root):
            if not filename.endswith(".hea"):
                continue
            stem = filename[:-4]
            if stem.isdigit():
                ids.append(int(stem))
        return sorted(set(ids))

    def _prepare_metadata(self, root: str) -> str:
        """Build ``ludb-pyhealth.csv`` if it does not exist.

        Returns:
            Directory path where ``ludb-pyhealth.csv`` is located.
        """
        root_path = Path(root)
        metadata_path = root_path / "ludb-pyhealth.csv"
        if metadata_path.exists():
            return str(root_path)

        ludb_csv_path = self._resolve_ludb_csv(root)

        # Optional richer metadata from PhysioNet ludb.csv
        meta_by_id: Dict[int, Dict[str, object]] = {}
        if ludb_csv_path is not None:
            raw_df = pd.read_csv(ludb_csv_path)

            # Derive no_p from Rhythms if available
            if "Rhythms" in raw_df.columns:
                afib_pattern = r"Atrial fibrillation|Atrial flutter"
                raw_df["no_p"] = (
                    raw_df["Rhythms"]
                    .astype(str)
                    .str.contains(afib_pattern, na=False)
                    .astype(int)
                )
                raw_df["rhythm_norm"] = (
                    raw_df["Rhythms"].astype(str).str.split("\n").str[0].str.strip()
                )
            else:
                raw_df["no_p"] = 0
                raw_df["rhythm_norm"] = None

            # Normalize electric axis if available
            axis_col = "Electric axis of the heart"
            if axis_col in raw_df.columns:
                raw_df["axis_norm"] = (
                    raw_df[axis_col]
                    .astype(str)
                    .str.split(":")
                    .str[-1]
                    .str.strip()
                    .replace({"nan": None})
                )
            else:
                raw_df["axis_norm"] = None

            if "ID" in raw_df.columns:
                subset = raw_df[
                    ["ID", "rhythm_norm", "axis_norm", "no_p"]
                ].values.tolist()

                for rid_raw, rhythm_raw, axis_raw, no_p_raw in subset:
                    if pd.isna(rid_raw):
                        continue

                    try:
                        rid = int(rid_raw)
                    except (TypeError, ValueError):
                        continue

                    try:
                        no_p = 0 if pd.isna(no_p_raw) else int(no_p_raw)
                    except (TypeError, ValueError):
                        no_p = 0

                    meta_by_id[rid] = {
                        "rhythm": rhythm_raw,
                        "electric_axis": axis_raw,
                        "no_p": no_p,
                    }

        record_ids = self._discover_record_ids(root)
        if not record_ids:
            # Fallback when files are not visible yet
            record_ids = list(range(1, NUM_RECORDS + 1))

        rows: List[Dict[str, object]] = []
        for rid in record_ids:
            pid = str(rid)

            row: Dict[str, object] = {
                "patient_id": pid,
                "visit_id": "ecg",
                "record_id": pid,
                # Absolute WFDB base record path, without extension
                "signal_file": str(root_path / pid),
                "fs": FS,
                "n_samples": N_SAMPLES,
                "rhythm": None,
                "electric_axis": None,
                "no_p": 0,
            }

            # 12 lead columns
            for lead in LEADS:
                col = f"lead_{lead}"
                # store absolute annotation path
                row[col] = str(root_path / f"{pid}.{lead}")

            # enrich from ludb.csv if present
            if rid in meta_by_id:
                row.update(meta_by_id[rid])

            rows.append(row)

        df = pd.DataFrame(rows)
        df.sort_values(["patient_id"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        try:
            df.to_csv(metadata_path, index=False)
            return str(root_path)
        except (PermissionError, OSError):
            cache_root = Path.home() / ".cache" / "pyhealth" / "ludb"
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_metadata_path = cache_root / "ludb-pyhealth.csv"
            df.to_csv(cache_metadata_path, index=False)
            return str(cache_root)

    @property
    def default_task(self):
        """No default task is enforced for LUDB."""
        return None


def get_stratified_ludb_split(
    ludb_csv: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Return stratified train/val/test LUDB record-ID splits.

    Stratification key:
        ``(no_p, rhythm_norm, axis_norm)``

    Rules:
        - Groups with count >= 10 are split proportionally.
        - Smaller groups are pooled and split proportionally together.

    Args:
        ludb_csv: Path to PhysioNet ``ludb.csv``.
        train_ratio: Train fraction.
        val_ratio: Validation fraction.
        seed: Random seed.

    Returns:
        (train_ids, val_ids, test_ids): sorted record IDs (1-indexed).
    """
    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio > 1:
        raise ValueError(
            "train_ratio and val_ratio must be non-negative and sum to <= 1."
        )

    rng = random.Random(seed)
    df = pd.read_csv(ludb_csv)

    if "ID" not in df.columns:
        raise ValueError("Expected column 'ID' in ludb.csv")
    if "Rhythms" not in df.columns:
        raise ValueError("Expected column 'Rhythms' in ludb.csv")
    if "Electric axis of the heart" not in df.columns:
        raise ValueError("Expected column 'Electric axis of the heart' in ludb.csv")

    # no_p=1 if AFib/flutter appears in rhythms
    afib_pattern = r"Atrial fibrillation|Atrial flutter"
    df["no_p"] = df["Rhythms"].str.contains(afib_pattern, na=False).astype(int)

    # primary rhythm: first line
    df["rhythm_norm"] = df["Rhythms"].str.split("\n").str[0].str.strip()

    # normalize axis text
    def _axis(val: str) -> str:
        if pd.isna(val):
            return "unknown"
        return str(val).split(":")[-1].strip()

    df["axis_norm"] = df["Electric axis of the heart"].apply(_axis)

    df["strat_key"] = list(zip(df["no_p"], df["rhythm_norm"], df["axis_norm"]))
    record_ids = df["ID"].astype(int).tolist()

    groups: Dict[tuple, List[int]] = defaultdict(list)
    for rid, key in zip(record_ids, df["strat_key"]):
        groups[key].append(rid)

    train_ids: List[int] = []
    val_ids: List[int] = []
    test_ids: List[int] = []

    threshold = 10
    small_pool: List[int] = []

    for _, ids in groups.items():
        rng.shuffle(ids)
        n = len(ids)
        if n >= threshold:
            n_train = round(n * train_ratio)
            n_val = round(n * val_ratio)
            train_ids.extend(ids[:n_train])
            val_ids.extend(ids[n_train : n_train + n_val])
            test_ids.extend(ids[n_train + n_val :])
        else:
            small_pool.extend(ids)

    rng.shuffle(small_pool)
    n = len(small_pool)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    train_ids.extend(small_pool[:n_train])
    val_ids.extend(small_pool[n_train : n_train + n_val])
    test_ids.extend(small_pool[n_train + n_val :])

    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


if __name__ == "__main__":
    CACHE_ROOT = (
        Path.home()
        / ".cache"
        / "pyhealth"
        / "datasets"
        / "physionet.org"
        / "files"
        / "ludb"
        / "1.0.1"
    )
    DATA_ROOT = os.environ.get("DATA_ROOT", str(CACHE_ROOT / "data"))
    LUDB_CSV = os.environ.get("LUDB_CSV", str(CACHE_ROOT / "ludb.csv"))

    dataset = LUDBDataset(root=DATA_ROOT, dev=True, download=True)
    dataset.stats()
    dataset.info()

    if os.path.exists(LUDB_CSV):
        train_ids, val_ids, test_ids = get_stratified_ludb_split(LUDB_CSV)
        print(
            f"Stratified split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
        )

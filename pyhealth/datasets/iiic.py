# Author: Jathurshan Pradeepkumar
# NetID: jp65
# Paper Title: Development of Expert-Level Classification of Seizures and Rhythmic and Periodic Patterns During EEG Interpretation; Tokenizing Single-Channel EEG with Time-Frequency Motif Learning
# Paper Link: https://pubmed.ncbi.nlm.nih.gov/36878708/ ; https://arxiv.org/pdf/2502.16060
# Description: IIIC Seizure (SPaRCNet) EEG dataset integration for PyHealth

import json
import logging
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base_dataset import BaseDataset
from pyhealth.tasks.iiic_seizure_classification import EEGSeizureIIIC

logger = logging.getLogger(__name__)


class IIICDataset(BaseDataset):
    """Base EEG dataset for the IIIC Seizure and Rhythmic/Periodic Patterns Corpus (SPaRCNet).

    Dataset is available via the Brain Data Science Platform (BDSP):
      - Website / Paper repo: https://github.com/bdsp-core/IIIC-SPaRCNet
      - AWS S3 Access Point:
        ``aws s3 ls s3://arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-restricted-access-point/sparcnet_data/``

    References:
      - Jing, J., Ge, W., Hong, S., Fernandes, M. B., Lin, Z., Yang, C., An, S.,
        Struck, A. F., Herlopian, A., Karakis, I., Halford, J. J., Ng, M. C.,
        Johnson, E. L., Appavu, B. L., Sarkis, R. A., Osman, G., Kaplan, P. W.,
        Dhakar, M. B., Jayagopal, L. A., Sheikh, Z., Taraschenko, O., Schmitt, S.,
        Haider, H. A., Kim, J. A., Swisher, C. B., Gaspard, N., Cervenka, M. C.,
        Rodriguez Ruiz, A. A., Lee, J. W., Tabaeizadeh, M., Gilmore, E. J.,
        Nordstrom, K., Yoo, J. Y., Holmes, M. G., Herman, S. T., Williams, J. A.,
        Pathmanathan, J., Nascimento, F. A., Fan, Z., Nasiri, S., Shashikumar, S. P.,
        Clifford, G. D., Sun, J., & Westover, M. B. (2023).
        "Development of Expert-Level Classification of Seizures and Rhythmic and
        Periodic Patterns During EEG Interpretation." *Neurology*, 100(17), e1750-e1762.
      - Pradeepkumar, J., Piao, X., Chen, Z., & Sun, J. (2026).
        "Tokenizing Single-Channel EEG with Time-Frequency Motif Learning."
        *ICLR 2026*. https://arxiv.org/abs/2502.16060

    The dataset contains 10-second, 16-channel longitudinal bipolar (TCP) montage
    EEG segments sampled at 200 Hz (shape ``(16, 2000)``), annotated by experts
    into six classes:
      - ``0``: Other
      - ``1``: Seizure (SZ)
      - ``2``: Lateralized Periodic Discharges (LPD)
      - ``3``: Generalized Periodic Discharges (GPD)
      - ``4``: Lateralized Rhythmic Delta Activity (LRDA)
      - ``5``: Generalized Rhythmic Delta Activity (GRDA)

    Supported Directory Layouts in ``root``:
      1. **Raw BDSP S3 / SPaRCNet ``.npy`` Layout**:
         Contains ``all_train_X.npy`` (or ``training_X.npy`` / ``10_train_X.npy``),
         ``all_train_Y.npy`` (or ``training_Y.npy`` / ``10_train_Y.npy``), and
         ``all_train_key.npy`` (or ``training_key.npy`` / ``10_train_key.npy``),
         plus optional ``test_X.npy``, ``test_Y.npy``, and ``test_key.npy``.
         When initialized on raw ``.npy`` files, ``prepare_metadata()`` applies the
         TFM-Tokenizer / SPaRCNet filtering pipeline (retaining segments with
         ``votes.sum(axis=1) > vote_threshold``, sorting and deduplicating by
         ``key``, removing flat zero-variance segments with ``std == 0``),
         performs a deterministic patient-level split (``patient_id = key.split('_')[0]``,
         60% train / 20% val / 20% test with ``np.random.RandomState(seed)``),
         saves per-segment ``{split}/{key}.pkl`` files (if ``save_pkl=True``),
         and writes ``train_df.csv``, ``val_df.csv``, ``test_df.csv``, and
         ``iiic-processed.csv``.
      2. **Pre-extracted ``.pkl`` + ``*_df.csv`` Layout** (e.g., ``/srv/local/data/jp65/IIIC/``):
         Contains ``train_df.csv``, ``val_df.csv``, ``test_df.csv`` and ``train/``,
         ``val/``, ``test/`` directories of ``{key}.pkl`` files. ``prepare_metadata()``
         reads these directly and builds ``iiic-processed.csv`` without re-extracting.

    Args:
        root: Root directory containing the IIIC dataset files.
        tables: List of tables to load. Default is ``["signal"]``.
        dataset_name: Optional dataset name. Defaults to ``"iiic"``.
        config_path: Optional path to YAML config file. Defaults to ``"iiic.yaml"``.
        subset: Which raw data subset to load when processing raw ``.npy`` files:
            ``"all"`` (uses ``all_train_*`` / ``training_*``) or ``"10_train"``
            (uses ``10_train_*`` with >= 10 annotator votes). Default is ``"all"``.
        vote_threshold: Minimum total votes required to keep a segment
            (``votes.sum(axis=1) > vote_threshold``). Default is ``3`` (following
            TFM-Tokenizer).
        split_ratios: Patient-level ``(train, val, test)`` split ratios when
            splitting from raw ``.npy`` files. Default is ``(0.6, 0.2, 0.2)``.
        seed: Random seed for deterministic patient-level splitting. Default is ``5``
            (matching ``Biosignal_Token`` / TFM-Tokenizer).
        save_pkl: When processing raw ``.npy`` files, whether to extract each valid
            segment to ``{split}/{key}.pkl``. If ``False``, ``signal_file`` references
            slices in the ``.npy`` file directly via ``path.npy::idx``. Default is ``True``.
        refresh_metadata: If ``True``, rebuild ``iiic-processed.csv`` even if it
            already exists. Default is ``False``.
        **kwargs: Additional keyword arguments passed to :class:`BaseDataset`
            (such as ``dev``, ``cache_dir``, etc.).

    Examples:
        >>> from pyhealth.datasets import IIICDataset
        >>> from pyhealth.tasks import EEGSeizureIIIC
        >>> dataset = IIICDataset(root="/srv/local/data/jp65/IIIC/")  # doctest: +SKIP
        >>> dataset.stats()  # doctest: +SKIP
        >>> sample_dataset = dataset.set_task(EEGSeizureIIIC())  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str,
        tables: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        subset: str = "all",
        vote_threshold: int = 3,
        split_ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
        seed: int = 5,
        save_pkl: bool = True,
        refresh_metadata: bool = False,
        **kwargs,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "iiic.yaml"

        if subset not in ("all", "10_train"):
            raise ValueError("subset must be one of 'all' or '10_train'")

        if len(split_ratios) != 3 or not np.isclose(sum(split_ratios), 1.0):
            raise ValueError(
                f"split_ratios must be a 3-tuple summing to 1.0, got {split_ratios}"
            )

        self.root = str(root)
        self.subset = subset
        self.vote_threshold = int(vote_threshold)
        self.split_ratios = tuple(float(r) for r in split_ratios)
        self.seed = int(seed)
        self.save_pkl = bool(save_pkl)
        self.refresh_metadata = bool(refresh_metadata)

        if tables is None:
            tables = ["signal"]

        self.prepare_metadata()

        # Check if iiic-processed.csv was written to fallback cache directory
        root_path = Path(self.root)
        cache_dir = Path.home() / ".cache" / "pyhealth" / "iiic"
        shared_csv = root_path / "iiic-processed.csv"
        cache_csv = cache_dir / "iiic-processed.csv"
        effective_root = self.root
        if not shared_csv.exists() and cache_csv.exists():
            logger.info(f"Using cached metadata from {cache_dir}")
            effective_root = str(cache_dir)

        super().__init__(
            root=effective_root,
            tables=tables,
            dataset_name=dataset_name or "iiic",
            config_path=config_path,
            **kwargs,
        )

    def prepare_metadata(self) -> None:
        """Build and save ``iiic-processed.csv`` metadata for IIICDataset.

        Supports both:
          1. Pre-extracted layout with ``train_df.csv``, ``val_df.csv``, ``test_df.csv``
             and ``train/``, ``val/``, ``test/`` ``.pkl`` directories.
          2. Raw ``.npy`` files from BDSP S3 ``sparcnet_data/`` or ``IIIC_data/``.
        """
        root = Path(self.root)
        cache_dir = Path.home() / ".cache" / "pyhealth" / "iiic"
        shared_csv = root / "iiic-processed.csv"
        cache_csv = cache_dir / "iiic-processed.csv"

        if not self.refresh_metadata and (shared_csv.exists() or cache_csv.exists()):
            return

        # Case 1: Pre-extracted train_df.csv / val_df.csv / test_df.csv layout
        split_csvs = {
            "train": root / "train_df.csv",
            "val": root / "val_df.csv",
            "test": root / "test_df.csv",
        }
        if any(p.exists() for p in split_csvs.values()) and not self._has_only_raw_npy(root):
            df = self._build_metadata_from_split_csvs(root, split_csvs)
            if not df.empty:
                self._save_metadata_csv(df, shared_csv, cache_csv)
                return

        # Case 2: Raw .npy files (BDSP S3 sparcnet_data/ or IIIC_data/)
        raw_sources = self._discover_raw_npy_sources(root)
        if raw_sources:
            df = self._build_metadata_from_raw_npy(root, cache_dir, raw_sources)
            if not df.empty:
                self._save_metadata_csv(df, shared_csv, cache_csv)
                return

        # Case 3: Scan existing train/ val/ test/ directories for .pkl files directly
        df = self._build_metadata_from_pkl_dirs(root)
        if not df.empty:
            self._save_metadata_csv(df, shared_csv, cache_csv)

    @staticmethod
    def _has_only_raw_npy(root: Path) -> bool:
        """Return True if split directories do not exist even if CSVs do."""
        return not any((root / sp).is_dir() for sp in ("train", "val", "test"))

    @staticmethod
    def _build_metadata_from_split_csvs(
        root: Path, split_csvs: Dict[str, Path]
    ) -> pd.DataFrame:
        """Construct unified metadata dataframe from ``train_df.csv``, ``val_df.csv``, ``test_df.csv``."""
        rows: List[Dict[str, object]] = []
        vote_cols = ["0", "1", "2", "3", "4", "5"]

        for split, csv_path in split_csvs.items():
            if not csv_path.exists():
                continue
            sdf = pd.read_csv(csv_path)
            if "key" not in sdf.columns:
                continue

            has_vote_cols = all(c in sdf.columns for c in vote_cols)
            for row in sdf.to_dict(orient="records"):
                key = str(row["key"]).strip()
                pid = (
                    str(row["pid"]).strip()
                    if "pid" in row and pd.notna(row["pid"])
                    else key.split("_")[0]
                )
                if "majority" in row and pd.notna(row["majority"]):
                    label = int(row["majority"])
                elif "label" in row and pd.notna(row["label"]):
                    label = int(row["label"])
                elif has_vote_cols:
                    votes_arr = [int(row[c]) for c in vote_cols]
                    label = int(np.argmax(votes_arr))
                else:
                    label = 0

                if has_vote_cols:
                    votes_str = json.dumps([int(row[c]) for c in vote_cols])
                else:
                    votes_str = json.dumps([])

                pkl_path = root / split / f"{key}.pkl"
                rows.append(
                    {
                        "patient_id": pid,
                        "record_id": key,
                        "label": label,
                        "votes": votes_str,
                        "split": split,
                        "signal_file": str(pkl_path),
                    }
                )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df.sort_values(["patient_id", "record_id"], inplace=True, na_position="last")
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def _build_metadata_from_pkl_dirs(root: Path) -> pd.DataFrame:
        """Fallback builder that scans ``train/``, ``val/``, ``test/`` for ``.pkl`` files."""
        rows: List[Dict[str, object]] = []
        for split in ("train", "val", "test"):
            split_dir = root / split
            if not split_dir.is_dir():
                continue
            for pkl_path in sorted(split_dir.glob("*.pkl")):
                key = pkl_path.stem
                pid = key.split("_")[0]
                label = 0
                try:
                    with open(pkl_path, "rb") as f:
                        data = pickle.load(f)
                    if isinstance(data, dict) and "label" in data:
                        label = int(data["label"])
                except Exception:
                    pass
                rows.append(
                    {
                        "patient_id": pid,
                        "record_id": key,
                        "label": label,
                        "votes": "[]",
                        "split": split,
                        "signal_file": str(pkl_path),
                    }
                )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df.sort_values(["patient_id", "record_id"], inplace=True, na_position="last")
        df.reset_index(drop=True, inplace=True)
        return df

    def _discover_raw_npy_sources(
        self, root: Path
    ) -> List[Tuple[Path, Path, Path]]:
        """Locate raw ``(X_path, Y_path, key_path)`` triplets in ``root``.

        Checks for:
          - ``10_train_{X,Y,key}.npy`` when ``self.subset == "10_train"``
          - ``all_train_{X,Y,key}.npy`` (official BDSP S3 naming)
          - ``training_{X,Y,key}.npy`` (local ``IIIC_data`` naming)
          - ``test_{X,Y,key}.npy`` or ``10_test_{X,Y,key}.npy`` if all three exist
        """
        sources: List[Tuple[Path, Path, Path]] = []

        if self.subset == "10_train":
            train_candidates = [("10_train_X.npy", "10_train_Y.npy", "10_train_key.npy")]
        else:
            train_candidates = [
                ("all_train_X.npy", "all_train_Y.npy", "all_train_key.npy"),
                ("training_X.npy", "training_Y.npy", "training_key.npy"),
                ("10_train_X.npy", "10_train_Y.npy", "10_train_key.npy"),
            ]

        for x_name, y_name, k_name in train_candidates:
            x_p = self._find_npy_file(root, x_name)
            y_p = self._find_npy_file(root, y_name)
            k_p = self._find_npy_file(root, k_name)
            if x_p is not None and y_p is not None and k_p is not None:
                sources.append((x_p, y_p, k_p))
                break

        # Also check if labeled test set exists (e.g., test_X.npy, test_Y.npy, test_key.npy)
        test_candidates = [
            ("test_X.npy", "test_Y.npy", "test_key.npy"),
            ("10_test_X.npy", "10_test_Y.npy", "10_test_key.npy"),
            ("10_test_X.npy", "test_Y.npy", "test_key.npy"),
        ]
        for x_name, y_name, k_name in test_candidates:
            x_p = self._find_npy_file(root, x_name)
            y_p = self._find_npy_file(root, y_name)
            k_p = self._find_npy_file(root, k_name)
            if x_p is not None and y_p is not None and k_p is not None:
                sources.append((x_p, y_p, k_p))
                break

        return sources

    @staticmethod
    def _find_npy_file(root: Path, filename: str) -> Optional[Path]:
        """Find ``filename`` directly in ``root`` or one level below (e.g. ``17_data_EEG_1115/`` or ``data_1109/``)."""
        direct = root / filename
        if direct.is_file():
            return direct
        for sub in ("17_data_EEG_1115", "data_1109"):
            candidate = root / sub / filename
            if candidate.is_file():
                return candidate
        return None

    def _build_metadata_from_raw_npy(
        self,
        root: Path,
        cache_dir: Path,
        raw_sources: List[Tuple[Path, Path, Path]],
    ) -> pd.DataFrame:
        """Process raw ``.npy`` files following ``Biosignal_Token`` / SPaRCNet pipeline.

        Steps:
          1. Load ``(X, Y, key)`` from each discovered source.
          2. Filter rows where ``Y.sum(axis=1) > self.vote_threshold``.
          3. Sort by ``key`` ascending and drop duplicate ``key``s (``keep='first'``).
          4. Filter out flat/corrupted signals where ``xi.astype(float).std() == 0``.
          5. Extract ``pid = key.split('_')[0]`` and deterministically split unique
             patients into ``train``, ``val``, ``test`` using ``np.random.RandomState(self.seed)``
             according to ``self.split_ratios`` (60% / 20% / 20% by default).
          6. Save per-segment ``{split}/{key}.pkl`` files (if ``self.save_pkl=True``)
             and ``{split}_df.csv`` files.
        """
        df_parts: List[pd.DataFrame] = []
        x_arrays: List[np.ndarray] = []

        for source_id, (x_path, y_path, k_path) in enumerate(raw_sources):
            X_mmap = np.load(x_path, mmap_mode="r")
            Y = np.load(y_path)
            keys = np.load(k_path)

            if Y.ndim == 2:
                votes = pd.DataFrame(Y, columns=[0, 1, 2, 3, 4, 5])
            else:
                one_hot = np.zeros((len(Y), 6), dtype=np.int64)
                for idx_r, lbl in enumerate(Y):
                    one_hot[idx_r, int(lbl)] = 10
                votes = pd.DataFrame(one_hot, columns=[0, 1, 2, 3, 4, 5])

            # Match _read_X: {k: X[i] for i, k in enumerate(key)} (last occurrence wins)
            last_idx_map = {str(k): i for i, k in enumerate(keys)}

            votes["key"] = keys.astype(str)
            votes["majority"] = np.argmax(votes[[0, 1, 2, 3, 4, 5]].values, axis=1)
            mask = (
                votes[[0, 1, 2, 3, 4, 5]].values.sum(axis=1)
                + votes["majority"].values
            ) > self.vote_threshold

            part_df = votes[mask].sort_values(["key"], ascending=True).copy()
            part_df = part_df.reset_index(drop=True)
            part_df = part_df.drop_duplicates(subset=["key"], keep="first").copy()
            part_df["pid"] = part_df["key"].map(lambda s: str(s).split("_")[0])
            part_df["source_id"] = source_id
            part_df["row_idx"] = part_df["key"].map(last_idx_map).astype(np.int64)

            df_parts.append(part_df)
            x_arrays.append(X_mmap)

        if not df_parts:
            return pd.DataFrame()

        combined = pd.concat(df_parts, axis=0, ignore_index=True)

        # Filter out flat zero-std segments
        valid_mask: List[bool] = []
        for rec in combined[["source_id", "row_idx"]].to_dict(orient="records"):
            xi = x_arrays[int(rec["source_id"])][int(rec["row_idx"])]
            valid_mask.append(bool(np.asarray(xi, dtype=np.float64).std() > 0))

        combined = combined[valid_mask].copy()
        combined.reset_index(drop=True, inplace=True)

        if combined.empty:
            return pd.DataFrame()

        # Patient-level split matching Biosignal_Token (seed=5, 60/20/20)
        pid_list = combined["pid"].unique().copy()
        rng = np.random.RandomState(self.seed)
        rng.shuffle(pid_list)

        n_pids = len(pid_list)
        r_train, r_val, _ = self.split_ratios
        train_end = int(r_train * n_pids)
        val_end = int((r_train + r_val) * n_pids)

        # Ensure non-empty splits when n_pids >= 3
        if n_pids >= 3:
            train_end = max(1, min(train_end, n_pids - 2))
            val_end = max(train_end + 1, min(val_end, n_pids - 1))

        train_pids = set(pid_list[:train_end])
        val_pids = set(pid_list[train_end:val_end])

        def _assign_split(pid: str) -> str:
            if pid in train_pids:
                return "train"
            if pid in val_pids:
                return "val"
            return "test"

        combined["split"] = combined["pid"].map(_assign_split)

        # Determine writable target base directory (root or cache_dir)
        target_base = root
        if self.save_pkl:
            try:
                (root / "train").mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError):
                target_base = cache_dir
                target_base.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, object]] = []
        for split in ("train", "val", "test"):
            split_sub = combined[combined["split"] == split].copy()
            if split_sub.empty:
                continue

            if self.save_pkl:
                split_dir = target_base / split
                split_dir.mkdir(parents=True, exist_ok=True)

            # Save <split>_df.csv for full compatibility with Biosignal_Token
            csv_cols = [0, 1, 2, 3, 4, 5, "key", "majority", "pid"]
            try:
                split_sub[csv_cols].to_csv(
                    target_base / f"{split}_df.csv", index=False
                )
            except (PermissionError, OSError):
                pass

            for rec in split_sub.to_dict(orient="records"):
                src_id = int(rec["source_id"])
                r_idx = int(rec["row_idx"])
                key = str(rec["key"])
                pid = str(rec["pid"])
                label = int(rec["majority"])
                votes = [int(rec[i]) for i in range(6)]
                xi = np.asarray(x_arrays[src_id][r_idx])

                if self.save_pkl:
                    pkl_path = target_base / split / f"{key}.pkl"
                    if not pkl_path.exists() or self.refresh_metadata:
                        with open(pkl_path, "wb") as f:
                            pickle.dump({"signal": xi, "label": label}, f)
                    signal_file_str = str(pkl_path)
                else:
                    x_path = raw_sources[src_id][0]
                    signal_file_str = f"{x_path}::{r_idx}"

                rows.append(
                    {
                        "patient_id": pid,
                        "record_id": key,
                        "label": label,
                        "votes": json.dumps(votes),
                        "split": split,
                        "signal_file": signal_file_str,
                    }
                )

        df = pd.DataFrame(rows)
        if not df.empty:
            df.sort_values(
                ["patient_id", "record_id"], inplace=True, na_position="last"
            )
            df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def _save_metadata_csv(
        df: pd.DataFrame, shared_csv: Path, cache_csv: Path
    ) -> None:
        """Write ``iiic-processed.csv`` to ``shared_csv`` or fallback ``cache_csv``."""
        try:
            shared_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(shared_csv, index=False)
            logger.info(f"Wrote IIIC metadata ({len(df)} records) to {shared_csv}")
        except (PermissionError, OSError):
            cache_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_csv, index=False)
            logger.info(
                f"Wrote IIIC metadata ({len(df)} records) to cache: {cache_csv}"
            )

    @property
    def default_task(self) -> EEGSeizureIIIC:
        """Returns the default task for the IIIC dataset: :class:`EEGSeizureIIIC`.

        Returns:
            EEGSeizureIIIC: The default task instance.
        """
        return EEGSeizureIIIC()

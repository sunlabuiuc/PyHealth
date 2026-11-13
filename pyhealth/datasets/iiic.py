# Author: Jathurshan Pradeepkumar
# NetID: jp65
# Paper Title: Tokenizing Single-Channel EEG with Time-Frequency Motif Learning
# Paper Link: https://arxiv.org/pdf/2502.16060
# Description: IIIC Seizure (SPaRCNet) EEG dataset integration for PyHealth

import contextlib
import hashlib
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from pyhealth.tasks.iiic_seizure_classification import EEGSeizureIIIC

from .base_dataset import BaseDataset, _default_cache_dir

logger = logging.getLogger(__name__)


class IIICDataset(BaseDataset):
    """Base EEG dataset for the IIIC Seizure Corpus (SPaRCNet).

    Dataset is available via the Brain Data Science Platform (BDSP):
      - Website / Paper repo: https://github.com/bdsp-core/IIIC-SPaRCNet
      - AWS S3 Access Point:
        ``s3://arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-restricted-access-point/sparcnet_data/``

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
         TFM-Tokenizer (``Biosignal_Token``) / SPaRCNet filtering pipeline:
         retaining segments where ``votes.sum(axis=1) + majority > vote_threshold``
         (intentionally adding ``majority = np.argmax(votes, axis=1)`` to the vote
         sum to match ``Biosignal_Token``'s ``_read_label_with_thres``, which assigns
         ``votes["majority"] = np.argmax(Y, 1)`` to the ``votes`` DataFrame before
         evaluating ``votes.sum(1) > thres``), enforcing ``votes.sum(axis=1) >= 10``
         when ``subset="10_train"``, sorting and deduplicating by ``key``, removing
         flat zero-variance segments with ``std == 0``, performing a deterministic
         patient-level split (``patient_id = key.split('_')[0]``, 60% train /
         20% val / 20% test with ``np.random.RandomState(seed)``), saving
         per-segment ``{split}/{key}.pkl`` files (if ``save_pkl=True``), and
         writing ``train_df.csv``, ``val_df.csv``, ``test_df.csv``,
         ``iiic-processed.csv``, and ``iiic-processed.meta.json``.
      2. **Pre-extracted ``.pkl`` + ``*_df.csv`` Layout**
         (e.g., ``/srv/local/data/jp65/IIIC/``):
         Contains ``train_df.csv``, ``val_df.csv``, ``test_df.csv`` and ``train/``,
         ``val/``, ``test/`` directories of ``{key}.pkl`` files. ``prepare_metadata()``
         reads these directly and builds ``iiic-processed.csv`` without re-extracting.

    Getting the data:
        - Source: the BDSP project ``bdsp-sparcnet``,
          https://bdsp.io/content/bdsp-sparcnet/
        - Access: credentialed. Create a BDSP account, complete its credentialing,
          and sign the project's data use agreement; that grants read access to
          the S3 access point above.
        - ``root``: the folder holding the downloaded IIIC files.

    Args:
        root: Root directory containing the IIIC dataset files.
        tables: List of tables to load. Default is ``["signal"]``.
        dataset_name: Optional dataset name. Defaults to ``"iiic"``.
        config_path: Optional path to YAML config file. Defaults to ``"iiic.yaml"``.
        subset: Which raw data subset to load when processing raw ``.npy`` files:
            ``"all"`` (uses ``all_train_*`` / ``training_*``) or ``"10_train"``
            (prefers ``10_train_*`` if present, and filters to segments with at
            least 10 annotator votes, ``votes.sum(axis=1) >= 10``).
            Default is ``"all"``.
        vote_threshold: Minimum threshold ``thres`` for the reference
            ``Biosignal_Token`` vote filter
            ``votes.sum(axis=1) + majority > vote_threshold`` (where
            ``majority = np.argmax(votes, axis=1)`` is intentionally included in
            the sum to match the reference ``_read_label_with_thres`` pipeline).
            Default is ``3``.
        split_ratios: Patient-level ``(train, val, test)`` split ratios when
            splitting from raw ``.npy`` files. Default is ``(0.6, 0.2, 0.2)``.
        seed: Random seed for deterministic patient-level splitting. Default is ``5``
            (matching ``Biosignal_Token`` / TFM-Tokenizer).
        save_pkl: When processing raw ``.npy`` files, whether to extract each valid
            segment to ``{split}/{key}.pkl``. If ``False``, ``signal_file`` references
            slices in the ``.npy`` file directly via ``path.npy::idx``.
            Default is ``True``.
        refresh_metadata: If ``True``, rebuild ``iiic-processed.csv`` even if a
            cached file with matching build parameters already exists. Note that
            changing ``subset``, ``vote_threshold``, ``split_ratios``, ``seed``,
            or ``save_pkl`` automatically triggers a rebuild via the sidecar
            ``iiic-processed.meta.json`` file. Default is ``False``.
        **kwargs: Additional keyword arguments passed to :class:`BaseDataset`
            (such as ``dev``, ``cache_dir``, etc.).

    Examples:
        >>> from pyhealth.datasets import IIICDataset
        >>> from pyhealth.tasks import EEGSeizureIIIC
        >>> dataset = IIICDataset(root="/srv/local/data/jp65/IIIC/")  # doctest: +SKIP
        >>> dataset.stats()  # doctest: +SKIP
        >>> sample_dataset = dataset.set_task(EEGSeizureIIIC())  # doctest: +SKIP
    """

    METADATA_CSV_FILENAME: str = "iiic-processed.csv"
    METADATA_META_FILENAME: str = "iiic-processed.meta.json"

    def __init__(
        self,
        root: str,
        tables: list[str] | None = None,
        dataset_name: str | None = None,
        config_path: str | None = None,
        subset: str = "all",
        vote_threshold: int = 3,
        split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
        seed: int = 5,
        save_pkl: bool = True,
        refresh_metadata: bool = False,
        cache_dir: str | Path | None = None,
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

        self.root = str(Path(root).expanduser().resolve())
        self.subset = subset
        self.vote_threshold = int(vote_threshold)
        self.split_ratios = tuple(float(r) for r in split_ratios)
        self.seed = int(seed)
        self.save_pkl = bool(save_pkl)
        self.refresh_metadata = bool(refresh_metadata)
        self._user_cache_dir: str | Path | None = cache_dir

        if tables is None:
            tables = ["signal"]

        self.prepare_metadata()

        # Use root if its metadata CSV is valid for current parameters;
        # otherwise use the parameter-keyed metadata cache directory.
        root_path = Path(self.root)
        meta_cache_dir = self._metadata_cache_dir()
        shared_csv = root_path / self.METADATA_CSV_FILENAME
        cache_csv = meta_cache_dir / self.METADATA_CSV_FILENAME
        effective_root = self.root
        if self._is_metadata_valid(shared_csv):
            effective_root = self.root
        elif self._is_metadata_valid(cache_csv):
            logger.info(f"Using cached metadata from {meta_cache_dir}")
            effective_root = str(meta_cache_dir)

        super().__init__(
            root=effective_root,
            tables=tables,
            dataset_name=dataset_name or "iiic",
            config_path=config_path,
            cache_dir=cache_dir,
            **kwargs,
        )

    def _build_params(self) -> dict[str, object]:
        """Return the canonical dictionary of metadata build parameters."""
        root_path = Path(self.root).expanduser().resolve()
        raw_sources = [
            [str(p.resolve()) for p in triplet]
            for triplet in self._discover_raw_npy_sources(root_path)
        ]
        return {
            "root": str(root_path),
            "subset": self.subset,
            "vote_threshold": self.vote_threshold,
            "split_ratios": list(self.split_ratios),
            "seed": self.seed,
            "save_pkl": self.save_pkl,
            "raw_sources": raw_sources,
        }

    def _build_params_digest(self) -> str:
        """Return a 16-char SHA-256 hex digest of ``_build_params()``."""
        payload = json.dumps(self._build_params(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _metadata_cache_dir(self) -> Path:
        """Return the parameter-keyed cache directory for derived IIIC files.

        Uses the ``cache_dir`` argument passed to :class:`IIICDataset` when
        provided, or :func:`~pyhealth.datasets.base_dataset._default_cache_dir`
        otherwise (which respects ``PYHEALTH_REQUIRE_CACHE_DIR``).

        Note:
            Unlike :func:`pyhealth.datasets.chbmit._metadata_cache_dir` (which
            hashes only ``root`` into a hardcoded ``~/.cache/pyhealth/chbmit``
            directory), IIIC's cache directory must (1) key on the full build
            parameter dictionary (:meth:`_build_params`) in addition to ``root``
            so changing ``subset``, ``vote_threshold``, ``split_ratios``, or
            ``seed`` never reuses stale splits, and (2) honor ``cache_dir`` and
            :func:`_default_cache_dir` because restricted-access BDSP EEG
            ``.pkl`` segments may be extracted there when ``root`` is read-only.
        """
        cache_id = f"iiic_{self._build_params_digest()}"
        if self._user_cache_dir is not None:
            return Path(self._user_cache_dir).expanduser().resolve() / cache_id
        return _default_cache_dir(cache_id)

    def _is_metadata_valid(self, csv_path: Path) -> bool:
        """Return True if ``csv_path`` exists and its sidecar matches build params."""
        if not csv_path.is_file():
            return False
        meta_path = csv_path.with_name(self.METADATA_META_FILENAME)
        if not meta_path.is_file():
            return False
        try:
            saved = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return saved == self._build_params()

    @staticmethod
    def _select_writable_base(root: Path, cache_dir: Path) -> Path:
        """Return ``root`` if writable, otherwise create and return ``cache_dir``."""
        if root.is_dir() and os.access(root, os.W_OK):
            try:
                probe = root / ".iiic_write_probe"
                probe.touch()
                probe.unlink()
                return root
            except (PermissionError, OSError):
                pass
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def prepare_metadata(self) -> None:
        """Build and save ``iiic-processed.csv`` metadata for IIICDataset.

        Supports both:
          1. Raw ``.npy`` files from BDSP S3 ``sparcnet_data/`` or ``IIIC_data/``.
          2. Pre-extracted layout with ``train_df.csv``, ``val_df.csv``, ``test_df.csv``
             and ``train/``, ``val/``, ``test/`` ``.pkl`` directories.
        """
        root = Path(self.root)
        cache_dir = self._metadata_cache_dir()
        shared_csv = root / self.METADATA_CSV_FILENAME
        cache_csv = cache_dir / self.METADATA_CSV_FILENAME

        if not self.refresh_metadata and (
            self._is_metadata_valid(shared_csv)
            or self._is_metadata_valid(cache_csv)
        ):
            return

        # Case 1: Raw .npy files (BDSP S3 sparcnet_data/ or IIIC_data/)
        raw_sources = self._discover_raw_npy_sources(root)
        if raw_sources:
            df = self._build_metadata_from_raw_npy(root, cache_dir, raw_sources)
            if not df.empty:
                self._save_metadata_csv(df, shared_csv, cache_csv)
                return

        # Case 2: Pre-extracted train_df.csv / val_df.csv / test_df.csv layout
        split_csvs = {
            "train": root / "train_df.csv",
            "val": root / "val_df.csv",
            "test": root / "test_df.csv",
        }
        has_csvs = any(p.exists() for p in split_csvs.values())
        if has_csvs and not self._has_only_raw_npy(root):
            df = self._build_metadata_from_split_csvs(root, split_csvs)
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
        root: Path, split_csvs: dict[str, Path]
    ) -> pd.DataFrame:
        """Construct unified metadata dataframe from split ``*_df.csv`` files."""
        rows: list[dict[str, object]] = []
        vote_cols = ["0", "1", "2", "3", "4", "5"]

        for split, csv_path in split_csvs.items():
            if not csv_path.exists():
                continue
            sdf = pd.read_csv(csv_path)
            if "key" not in sdf.columns:
                continue

            has_vote_cols = all(c in sdf.columns for c in vote_cols)
            has_label_col = (
                "majority" in sdf.columns
                or "label" in sdf.columns
                or has_vote_cols
            )
            if not has_label_col:
                logger.warning(
                    "Skipping %s: CSV has no 'majority', 'label', or vote columns",
                    csv_path,
                )
                continue

            for row in sdf.to_dict(orient="records"):
                key = str(row["key"]).strip()
                pid = (
                    str(row["pid"]).strip()
                    if "pid" in row and pd.notna(row["pid"])
                    else key.split("_")[0]
                )
                has_valid_votes = has_vote_cols and all(
                    pd.notna(row[c]) for c in vote_cols
                )
                if "majority" in row and pd.notna(row["majority"]):
                    label = int(row["majority"])
                elif "label" in row and pd.notna(row["label"]):
                    label = int(row["label"])
                elif has_valid_votes:
                    votes_arr = [int(row[c]) for c in vote_cols]
                    if sum(votes_arr) <= 0:
                        logger.warning(
                            "Skipping unlabeled row %s in %s: zero total votes",
                            key,
                            csv_path,
                        )
                        continue
                    label = int(np.argmax(votes_arr))
                else:
                    logger.warning(
                        "Skipping unlabeled row %s in %s: missing label and votes",
                        key,
                        csv_path,
                    )
                    continue

                if has_valid_votes:
                    votes_str = json.dumps([int(row[c]) for c in vote_cols])
                else:
                    votes_str = json.dumps([])

                pkl_path = (root / split / f"{key}.pkl").resolve()
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
        """Fallback builder scanning ``train/``, ``val/``, ``test/`` for ``.pkl``."""
        rows: list[dict[str, object]] = []
        for split in ("train", "val", "test"):
            split_dir = root / split
            if not split_dir.is_dir():
                continue
            for pkl_path in sorted(split_dir.glob("*.pkl")):
                key = pkl_path.stem
                pid = key.split("_")[0]
                try:
                    with open(pkl_path, "rb") as f:
                        data = pickle.load(f)
                except (
                    OSError,
                    pickle.UnpicklingError,
                    ValueError,
                    KeyError,
                    EOFError,
                ) as exc:
                    logger.warning(
                        "Skipping unreadable pickle %s: %s", pkl_path, exc
                    )
                    continue

                if (
                    not isinstance(data, dict)
                    or "label" not in data
                    or data["label"] is None
                    or pd.isna(data["label"])
                ):
                    logger.warning(
                        "Skipping unlabeled pickle %s: missing 'label' key",
                        pkl_path,
                    )
                    continue

                try:
                    label = int(data["label"])
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "Skipping pickle %s with invalid label %r: %s",
                        pkl_path,
                        data["label"],
                        exc,
                    )
                    continue

                rows.append(
                    {
                        "patient_id": pid,
                        "record_id": key,
                        "label": label,
                        "votes": "[]",
                        "split": split,
                        "signal_file": str(pkl_path.resolve()),
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
    ) -> list[tuple[Path, Path, Path]]:
        """Locate raw ``(X_path, Y_path, key_path)`` triplets in ``root``.

        Checks for:
          - ``10_train_{X,Y,key}.npy`` when ``self.subset == "10_train"``
          - ``all_train_{X,Y,key}.npy`` (official BDSP S3 naming)
          - ``training_{X,Y,key}.npy`` (local ``IIIC_data`` naming)
          - ``test_{X,Y,key}.npy`` or ``10_test_{X,Y,key}.npy`` if all three exist
        """
        sources: list[tuple[Path, Path, Path]] = []

        if self.subset == "10_train":
            train_candidates = [
                ("10_train_X.npy", "10_train_Y.npy", "10_train_key.npy"),
                ("all_train_X.npy", "all_train_Y.npy", "all_train_key.npy"),
                ("training_X.npy", "training_Y.npy", "training_key.npy"),
            ]
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

        # Also check if labeled test set exists (e.g., test_X/Y/key.npy)
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
    def _find_npy_file(root: Path, filename: str) -> Path | None:
        """Find ``filename`` in ``root`` or one directory level below."""
        direct = root / filename
        if direct.is_file():
            return direct
        for sub in ("17_data_EEG_1115", "data_1109"):
            candidate = root / sub / filename
            if candidate.is_file():
                return candidate
        return None

    def _cleanup_generated_files(self, target_base: Path) -> None:
        """Remove only ``.pkl`` and ``*_df.csv`` files generated by IIICDataset."""
        meta_path = target_base / self.METADATA_META_FILENAME
        csv_path = target_base / self.METADATA_CSV_FILENAME
        if not meta_path.is_file():
            return
        try:
            saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not saved_meta.get("raw_sources"):
            return

        for split in ("train", "val", "test"):
            old_csv = target_base / f"{split}_df.csv"
            with contextlib.suppress(OSError):
                old_csv.unlink(missing_ok=True)

        if saved_meta.get("save_pkl") and csv_path.is_file():
            try:
                prev_df = pd.read_csv(csv_path, usecols=["signal_file"])
            except (OSError, ValueError, KeyError):
                return
            resolved_base = target_base.resolve()
            for sig_str in prev_df["signal_file"].dropna().astype(str):
                if "::" in sig_str:
                    continue
                sig_path = Path(sig_str)
                with contextlib.suppress(OSError):
                    if sig_path.resolve().is_relative_to(resolved_base):
                        sig_path.unlink(missing_ok=True)

    def _build_metadata_from_raw_npy(
        self,
        root: Path,
        cache_dir: Path,
        raw_sources: list[tuple[Path, Path, Path]],
    ) -> pd.DataFrame:
        """Process raw ``.npy`` files following ``Biosignal_Token`` / SPaRCNet pipeline.

        Steps:
          1. Load ``(X, Y, key)`` from each discovered source.
          2. Filter rows where ``votes.sum(axis=1) + majority > self.vote_threshold``
             (intentionally including ``majority = np.argmax(votes, axis=1)`` to match
             ``Biosignal_Token``'s ``_read_label_with_thres``), and when
             ``self.subset == "10_train"``, also require ``votes.sum(axis=1) >= 10``.
          3. Sort by ``key`` ascending (``kind="stable"``) and drop duplicate
             ``key``s (``keep="last"``) so the kept row's votes, label, and
             signal index come from the same occurrence.
          4. Filter out flat/corrupted signals where ``xi.astype(float).std() == 0``.
          5. Extract ``pid = key.split('_')[0]`` and deterministically split unique
             patients into ``train``, ``val``, ``test`` using
             ``np.random.RandomState(self.seed)`` according to ``self.split_ratios``
             (60% / 20% / 20% by default).
          6. Save per-segment ``{split}/{key}.pkl`` files (if ``self.save_pkl=True``)
             and ``{split}_df.csv`` files.
        """
        df_parts: list[pd.DataFrame] = []
        x_arrays: list[np.ndarray] = []

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

            raw_vote_sum = votes[[0, 1, 2, 3, 4, 5]].values.sum(axis=1)
            votes["key"] = keys.astype(str)
            votes["majority"] = np.argmax(votes[[0, 1, 2, 3, 4, 5]].values, axis=1)
            votes["row_idx"] = np.arange(len(keys), dtype=np.int64)
            # Intentionally include votes["majority"] in the sum to match
            # Biosignal_Token's _read_label_with_thres (which sets
            # votes["majority"] = np.argmax(Y, 1) before votes.sum(1) > thres).
            mask = (raw_vote_sum + votes["majority"].values) > self.vote_threshold
            if self.subset == "10_train":
                mask = mask & (raw_vote_sum >= 10)

            part_df = (
                votes[mask]
                .sort_values(["key"], ascending=True, kind="stable")
                .drop_duplicates(subset=["key"], keep="last")
                .reset_index(drop=True)
                .copy()
            )
            part_df["pid"] = part_df["key"].map(lambda s: str(s).split("_")[0])
            part_df["source_id"] = source_id

            df_parts.append(part_df)
            x_arrays.append(X_mmap)

        if not df_parts:
            return pd.DataFrame()

        combined = pd.concat(df_parts, axis=0, ignore_index=True)
        combined = (
            combined.drop_duplicates(subset=["key"], keep="last")
            .reset_index(drop=True)
            .copy()
        )

        # Filter out flat zero-std segments
        valid_mask: list[bool] = []
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
        target_base = self._select_writable_base(root, cache_dir)
        self._cleanup_generated_files(target_base)

        rows: list[dict[str, object]] = []
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
            except (PermissionError, OSError) as exc:
                logger.debug("Could not write %s_df.csv: %s", split, exc)

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
                        tmp_pkl = pkl_path.with_name(
                            f".{key}.{os.getpid()}.pkl.tmp"
                        )
                        try:
                            with open(tmp_pkl, "wb") as f:
                                pickle.dump({"signal": xi, "label": label}, f)
                            os.replace(tmp_pkl, pkl_path)
                        except BaseException:
                            with contextlib.suppress(OSError):
                                tmp_pkl.unlink()
                            raise
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

    def _save_metadata_csv(
        self, df: pd.DataFrame, shared_csv: Path, cache_csv: Path
    ) -> None:
        """Write ``iiic-processed.csv`` and ``.meta.json`` to ``root`` or cache."""
        params_text = json.dumps(self._build_params(), indent=2, sort_keys=True)
        root = shared_csv.parent
        if root.is_dir() and os.access(root, os.W_OK):
            try:
                df.to_csv(shared_csv, index=False)
                meta_path = shared_csv.with_name(self.METADATA_META_FILENAME)
                meta_path.write_text(params_text, encoding="utf-8")
                logger.info(
                    f"Wrote IIIC metadata ({len(df)} records) to {shared_csv}"
                )
                return
            except (PermissionError, OSError):
                pass

        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_csv, index=False)
        meta_path = cache_csv.with_name(self.METADATA_META_FILENAME)
        meta_path.write_text(params_text, encoding="utf-8")
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

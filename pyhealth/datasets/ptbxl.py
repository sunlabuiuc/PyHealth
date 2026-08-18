"""
PyHealth dataset for PTB-XL (12-lead ECG, PhysioNet).

Dataset link:
    https://physionet.org/content/ptb-xl/1.0.3/

Dataset paper: (please cite if you use this dataset)
    Patrick Wagner, Nils Strodthoff, Ralf-Dieter Bousseljot, Dieter Kreiseler,
    Fatima I. Lunze, Wojciech Samek, and Tobias Schaeffter. "PTB-XL, a large
    publicly available electrocardiography dataset." Scientific Data 7, 154
    (2020).

Dataset paper link:
    https://www.nature.com/articles/s41597-020-0495-6

Author:
    AxelNoun (GitHub: @AxelNoun) — external contributor, no NetID

Description:
    Implements ``PTBXLDataset`` (BaseDataset + YAML) for PTB-XL v1.0.3,
    including metadata preparation, HIPAA age-censor handling, and lazy
    ``wfdb`` signal loading. Task and split helpers live in separate modules.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yaml
from filelock import FileLock

from .base_dataset import BaseDataset
from .configs.config import load_yaml_config
from .utils import MODULE_CACHE_PATH

if TYPE_CHECKING:
    from pyhealth.tasks.ptbxl import PTBXLSuperclassClassification

logger = logging.getLogger(__name__)

# HIPAA: ages ≥ 90 are encoded as this sentinel in ptbxl_database.csv.
AGE_CENSOR_SENTINEL = 300

PTBXL_DATABASE_CSV = "ptbxl_database.csv"
PTBXL_SCP_STATEMENTS_CSV = "scp_statements.csv"

_DEFAULT_METADATA_CACHE = Path(MODULE_CACHE_PATH) / "ptbxl"

def _atomic_replace(tmp_path: Path, dest: Path) -> None:
    """Replace ``dest`` with ``tmp_path`` (same-filesystem atomic on POSIX/NT)."""
    os.replace(tmp_path, dest)


def _write_csv_atomic(df: pd.DataFrame, dest: Path) -> None:
    tmp_path = dest.with_name(dest.name + ".tmp")
    try:
        df.to_csv(tmp_path, index=False)
        _atomic_replace(tmp_path, dest)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _write_yaml_atomic(payload: dict[str, Any], dest: Path) -> None:
    tmp_path = dest.with_name(dest.name + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        _atomic_replace(tmp_path, dest)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def format_patient_id(value: Any, *, ecg_id: Any | None = None) -> str:
    """Cast PTB-XL ``patient_id`` (stored as float, e.g. ``15709.0``) to ``str``.

    Without an explicit int cast, stringification yields ``\"15709.0\"`` and
    silently breaks patient-level splits. Called from
    :meth:`PTBXLDataset.prepare_metadata`.

    Args:
        value (Any): Raw ``patient_id`` cell (float, int, or numeric string).
        ecg_id (Any | None): Optional ``ecg_id`` included in error messages.

    Returns:
        str: Integer patient id as a string (e.g. ``\"15709\"``).

    Examples:
        >>> format_patient_id(15709.0)
        '15709'
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        suffix = f" (ecg_id={ecg_id})" if ecg_id is not None else ""
        raise ValueError(f"patient_id is missing{suffix}")
    try:
        return str(int(float(value)))
    except (TypeError, ValueError) as exc:
        suffix = f" (ecg_id={ecg_id})" if ecg_id is not None else ""
        raise ValueError(
            f"patient_id is not numeric: {value!r}{suffix}"
        ) from exc


def parse_scp_codes(scp_codes: Any) -> dict[str, float]:
    """Parse the stringified ``scp_codes`` dict from ``ptbxl_database.csv``.

    PhysioNet stores entries as ``statement: likelihood`` where likelihood is
    in ``[0, 100]`` and **0 means unknown confidence, not absence**. All keys
    present in the dict are therefore treated as positive statements; only an
    empty dict means no statements. Used by task aggregation helpers.

    Args:
        scp_codes (Any): Raw cell value (dict or stringified dict).

    Returns:
        dict[str, float]: Mapping of SCP statement acronym → likelihood.

    Examples:
        >>> parse_scp_codes("{'IMI': 80.0, 'SR': 0.0}")["SR"]
        0.0
    """
    if scp_codes is None or (isinstance(scp_codes, float) and np.isnan(scp_codes)):
        return {}
    if isinstance(scp_codes, dict):
        return {str(k): float(v) for k, v in scp_codes.items()}
    text = str(scp_codes).strip()
    if not text or text.lower() in {"nan", "none", "{}"}:
        return {}
    parsed = ast.literal_eval(text)
    if not isinstance(parsed, dict):
        raise TypeError(
            f"Expected scp_codes dict, got {type(parsed)!r}: {scp_codes!r}"
        )
    return {str(k): float(v) for k, v in parsed.items()}


def is_age_censored(age: Any, sentinel: int = AGE_CENSOR_SENTINEL) -> bool:
    """Return True when age is the HIPAA ≥90 sentinel (default 300).

    Distinct from missing age (NaN), which is not censored. Used when writing
    ``age_is_censored`` in metadata.

    Args:
        age (Any): Raw age cell from ``ptbxl_database.csv``.
        sentinel (int): Censor value (default ``300``).

    Returns:
        bool: True if ``age`` equals the HIPAA sentinel.

    Examples:
        >>> is_age_censored(300), is_age_censored(float("nan"))
        (True, False)
    """
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return False
    if isinstance(age, str) and not age.strip():
        return False
    try:
        return int(float(age)) == int(sentinel)
    except (TypeError, ValueError):
        return False


def is_age_missing(age: Any) -> bool:
    """Return True when age is genuinely missing (NaN / empty), not censored.

    Args:
        age (Any): Raw age cell from ``ptbxl_database.csv``.

    Returns:
        bool: True if age is missing (not the 300 sentinel).

    Examples:
        >>> is_age_missing(float("nan")), is_age_missing(300)
        (True, False)
    """
    if age is None:
        return True
    if isinstance(age, float) and np.isnan(age):
        return True
    return isinstance(age, str) and not str(age).strip()


def root_cache_key(data_root: str | Path) -> str:
    """Return a short stable hash of the resolved data root path.

    Args:
        data_root (str | Path): PTB-XL version root.

    Returns:
        str: 10-character hex digest used in metadata filenames.

    Examples:
        >>> len(root_cache_key("/data/ptb-xl/1.0.3"))
        10
    """
    resolved = str(Path(data_root).resolve())
    return hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:10]


def metadata_filename(
    sampling_rate: int,
    data_root: str | Path,
    source_key: str | None = None,
) -> str:
    """Return the rate-, root-, and source-specific derived metadata CSV name.

    ``source_key`` is a content hash of ``ptbxl_database.csv``. Including it
    in the filename forces ``prepare_metadata`` to regenerate when the
    official CSV is replaced in-place (e.g. extracting v1.0.3 over v1.0.1).

    Args:
        sampling_rate (int): ``100`` or ``500``.
        data_root (str | Path): Absolute/relative PTB-XL data root.
        source_key (str | None): Optional SHA1[:10] of the source
            ``ptbxl_database.csv`` bytes.

    Returns:
        str: Filename such as ``ptbxl-pyhealth-100hz-<root>[-<source>].csv``.

    Examples:
        >>> name = metadata_filename(100, "/data/ptb-xl/1.0.3")
        >>> name.startswith("ptbxl-pyhealth-100hz-")
        True
        >>> keyed = metadata_filename(100, "/data/ptb-xl/1.0.3", "abc123def0")
        >>> "abc123def0" in keyed
        True
    """
    name = (
        f"ptbxl-pyhealth-{int(sampling_rate)}hz-"
        f"{root_cache_key(data_root)}"
    )
    if source_key:
        name = f"{name}-{source_key}"
    return f"{name}.csv"


def load_ptbxl_record(record_path: str | Path) -> np.ndarray:
    """Load a PTB-XL WFDB record as ``(n_leads, n_samples)``.

    ``wfdb.rdsamp`` returns ``(n_samples, n_channels)`` (e.g. ``(1000, 12)`` at
    100 Hz). This helper **transposes** to ``(n_leads, n_samples)`` so the layout
    matches PyHealth signal tasks such as EEGBCI / SleepEDF, which use
    ``mne.io.BaseRaw.get_data()`` → ``(n_channels, n_times)``. The
    ``\"tensor\"`` processor preserves that shape.

    The path must be a WFDB *record base* without ``.hea`` / ``.dat`` (as in
    ``filename_lr`` / ``filename_hr``). Extensions are stripped only if the
    caller accidentally includes them; they are never appended.

    Args:
        record_path (str | Path): WFDB record base path (no extension).

    Returns:
        np.ndarray: Signal of shape ``(n_leads, n_samples)``, float32,
        physical units.

    Raises:
        ImportError: If the optional ``wfdb`` extra is not installed.
        FileNotFoundError: If the ``.hea`` header cannot be found.

    Examples:
        >>> # doctest: +SKIP
        >>> signal = load_ptbxl_record("/data/ptb-xl/1.0.3/records100/00000/00001_lr")
        >>> signal.shape[0]
        12
    """
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError(
            "Reading PTB-XL waveforms requires the optional 'wfdb' dependency. "
            "Install it with: pip install 'pyhealth[ptbxl]'"
        ) from exc

    path = Path(record_path)
    # Strip accidental extensions; never add .hea/.dat for rdsamp.
    if path.suffix.lower() in {".hea", ".dat"}:
        path = path.with_suffix("")
    header = Path(str(path) + ".hea")
    if not header.is_file():
        raise FileNotFoundError(
            f"PTB-XL WFDB header not found for record base: {record_path}"
        )

    signals, _ = wfdb.rdsamp(str(path))
    # (n_samples, n_leads) → (n_leads, n_samples) to match mne.get_data().
    return np.asarray(signals, dtype=np.float32).T


class PTBXLDataset(BaseDataset):
    """PhysioNet PTB-XL ECG dataset (v1.0.3).

    Dataset: https://physionet.org/content/ptb-xl/1.0.3/

    Expects ``root`` (``data_root``) to point at the extracted **v1.0.3**
    directory containing ``ptbxl_database.csv``, ``scp_statements.csv``,
    ``records100/``, and ``records500/``. Raw data must live outside the git
    repo. Earlier PhysioNet releases (v1.0.1 / v1.0.2) are not supported:
    required columns and record counts differ.

    Derived metadata CSVs are written under PyHealth's dataset cache
    (``~/.cache/pyhealth/datasets/ptbxl/`` by default), **not** into ``root``,
    so read-only / shared data mounts stay untouched. Override with
    ``metadata_cache_dir``. Filenames include the sampling rate, a short
    hash of the resolved data root, and a content hash of
    ``ptbxl_database.csv`` so different roots and source versions never share
    a derived CSV. ``BaseDataset`` cache identity (``global_event_df``) further
    includes those hashes via ``dataset_name`` (same pattern as EEGBCI).

    Args:
        root (str): Version root of the PTB-XL download (signal + official CSVs).
        dataset_name (str | None): Optional name prefix; defaults to
            ``ptbxl_{sampling_rate}hz``. Root and metadata content hashes are
            appended so two data roots cannot share a ``BaseDataset`` cache.
        config_path (str | Path | None): Optional YAML config; defaults to
            ``configs/ptbxl.yaml``.
        sampling_rate (int): ``100`` (default) or ``500``.
        metadata_cache_dir (str | Path | None): Directory for derived
            ``ptbxl-pyhealth-*.csv``. Defaults to
            ``MODULE_CACHE_PATH / \"ptbxl\"``.
        **kwargs: Forwarded to :class:`BaseDataset` (``cache_dir``, ``dev``, …).

    Attributes:
        data_root (Path): User-provided PTB-XL version root (waveforms + CSVs).
        sampling_rate (int): Selected waveform rate (100 or 500).
        metadata_cache_dir (Path): Directory holding derived metadata CSVs.

    Note:
        Age missing (NaN) vs HIPAA-censored (``age == 300`` for ≥90) are
        exposed as ``age_is_missing`` / ``age_is_censored``. Sex is encoded
        as ``0`` = female, ``1`` = male. ``scp_codes`` keeps the original
        stringified dict; likelihood ``0`` means unknown confidence. Label
        aggregation lives in task modules. ``recording_date`` is used as the
        event timestamp.

    Examples:
        >>> dataset = PTBXLDataset(root="/data/ptb-xl/1.0.3")  # doctest: +SKIP
        >>> dataset.stats()  # doctest: +SKIP
        >>> patient = dataset.get_patient(dataset.unique_patient_ids[0])  # doctest: +SKIP
        >>> event = patient.get_events(event_type="records")[0]  # doctest: +SKIP
        >>> signal = load_ptbxl_record(event.signal_file)  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | Path | None = None,
        sampling_rate: int = 100,
        metadata_cache_dir: str | Path | None = None,
        **kwargs,
    ) -> None:
        if sampling_rate not in {100, 500}:
            raise ValueError(
                f"sampling_rate must be 100 or 500, got {sampling_rate}"
            )
        package_config = (
            Path(config_path)
            if config_path is not None
            else Path(__file__).parent / "configs" / "ptbxl.yaml"
        )

        self.data_root = Path(root)
        self.sampling_rate = int(sampling_rate)
        self.metadata_cache_dir = Path(
            metadata_cache_dir
            if metadata_cache_dir is not None
            else _DEFAULT_METADATA_CACHE
        )
        db_path = self.data_root / PTBXL_DATABASE_CSV
        if not db_path.is_file():
            raise FileNotFoundError(
                f"Expected {PTBXL_DATABASE_CSV} under root={self.data_root}. "
                "Download PTB-XL from https://physionet.org/content/ptb-xl/1.0.3/"
            )
        self._source_key = hashlib.sha1(db_path.read_bytes()).hexdigest()[:10]
        self.metadata_file_name = metadata_filename(
            self.sampling_rate, self.data_root, self._source_key
        )
        self.prepare_metadata()

        # BaseDataset only accepts config_path (not an in-memory config). Write
        # a resolved YAML whose file_path matches the derived CSV *before*
        # super().__init__ — loading is lazy via global_event_df, but the
        # config must already be correct when that first happens.
        resolved_config_path = self._write_resolved_config(package_config)

        # BaseDataset._init_cache_dir keys on {root, tables, dataset_name, dev}.
        # root is the shared metadata cache (see comment on super().__init__),
        # so uniqueness of global_event_df must come from dataset_name — same
        # pattern as EEGBCIDataset._metadata_cache_key.
        base_name = dataset_name or f"ptbxl_{self.sampling_rate}hz"
        dataset_name = (
            f"{base_name}_{root_cache_key(self.data_root)}_"
            f"{self._metadata_cache_key()}"
        )

        # BaseDataset.root is the metadata cache (CSV location); waveforms stay
        # under data_root via absolute signal_file paths in the CSV.
        super().__init__(
            root=str(self.metadata_cache_dir),
            tables=["records"],
            dataset_name=dataset_name,
            config_path=str(resolved_config_path),
            **kwargs,
        )

    def _write_resolved_config(self, package_config: Path) -> Path:
        """Write a cache-local YAML with the derived metadata CSV filename.

        Args:
            package_config (Path): Packaged ``configs/ptbxl.yaml`` template.

        Returns:
            Path: Written config path passed to :class:`BaseDataset`.
        """
        config = load_yaml_config(str(package_config))
        config.tables["records"].file_path = self.metadata_file_name
        out_path = self.metadata_cache_dir / (
            f"ptbxl-config-{self.sampling_rate}hz-"
            f"{root_cache_key(self.data_root)}-{self._source_key}.yaml"
        )
        self.metadata_cache_dir.mkdir(parents=True, exist_ok=True)
        lock_path = out_path.with_name(out_path.name + ".lock")
        with FileLock(str(lock_path)):
            _write_yaml_atomic(config.model_dump(), out_path)
        return out_path

    @property
    def scp_statements_path(self) -> Path:
        """Path to official ``scp_statements.csv`` under the data root.

        Returns:
            Path: Absolute path to ``scp_statements.csv``.
        """
        return self.data_root / PTBXL_SCP_STATEMENTS_CSV

    def _metadata_cache_key(self) -> str:
        """SHA1[:10] of the derived metadata CSV bytes (EEGBCI pattern).

        Must be called after :meth:`prepare_metadata` so the file exists.
        Injected into ``dataset_name`` so ``BaseDataset._init_cache_dir``
        does not collapse two data roots onto one ``global_event_df``.
        """
        csv_path = self.metadata_cache_dir / self.metadata_file_name
        return hashlib.sha1(csv_path.read_bytes()).hexdigest()[:10]

    def prepare_metadata(self) -> None:
        """Build rate-/root-specific metadata CSV under ``metadata_cache_dir``.

        Returns:
            None

        Examples:
            >>> # doctest: +SKIP
            >>> ds = PTBXLDataset(root="/data/ptb-xl/1.0.3")
            >>> ds.prepare_metadata()
        """
        csv_path = self.metadata_cache_dir / self.metadata_file_name
        if csv_path.exists() and self._metadata_matches_request(csv_path):
            return

        self.metadata_cache_dir.mkdir(parents=True, exist_ok=True)
        lock_path = csv_path.with_name(csv_path.name + ".lock")
        with FileLock(str(lock_path)):
            # Another process may have finished while we waited.
            if csv_path.exists() and self._metadata_matches_request(csv_path):
                return
            self._write_derived_metadata(csv_path)

    def _write_derived_metadata(self, csv_path: Path) -> None:
        """Generate the derived metadata CSV (caller holds the file lock)."""
        db_path = self.data_root / PTBXL_DATABASE_CSV
        if not db_path.is_file():
            raise FileNotFoundError(
                f"Expected {PTBXL_DATABASE_CSV} under root={self.data_root}. "
                "Download PTB-XL from https://physionet.org/content/ptb-xl/1.0.3/"
            )

        db = pd.read_csv(db_path)
        required = {
            "ecg_id",
            "patient_id",
            "age",
            "sex",
            "site",
            "device",
            "scp_codes",
            "strat_fold",
            "filename_lr",
            "filename_hr",
            "recording_date",
        }
        missing = required - set(db.columns)
        if missing:
            raise ValueError(
                f"ptbxl_database.csv missing columns: {sorted(missing)}. "
                "PTBXLDataset supports PhysioNet PTB-XL v1.0.3 only "
                "(https://physionet.org/content/ptb-xl/1.0.3/)."
            )

        filename_col = "filename_lr" if self.sampling_rate == 100 else "filename_hr"
        rows: list[dict[str, Any]] = []
        for row in db.to_dict(orient="records"):
            ecg_id = row["ecg_id"]
            rel = str(row[filename_col]).strip()
            # Record base path only — no .hea/.dat (matches wfdb.rdsamp).
            if rel.endswith((".hea", ".dat")):
                rel = rel.rsplit(".", 1)[0]
            signal_file = str((self.data_root / rel).resolve())

            age = row["age"]
            missing_age = is_age_missing(age)
            censored = is_age_censored(age)
            if missing_age:
                age_out: Any = pd.NA
            else:
                # Keep numeric age (including sentinel 300).
                age_out = int(float(age))

            rows.append(
                {
                    "patient_id": format_patient_id(
                        row["patient_id"], ecg_id=ecg_id
                    ),
                    "record_id": str(int(float(ecg_id))),
                    "signal_file": signal_file,
                    "sampling_rate": self.sampling_rate,
                    "strat_fold": int(row["strat_fold"]),
                    "age": age_out,
                    "age_is_censored": int(censored),
                    "age_is_missing": int(missing_age),
                    "sex": int(row["sex"]),
                    "site": row["site"],
                    "device": row["device"],
                    "scp_codes": row["scp_codes"],
                    "recording_date": row["recording_date"],
                }
            )

        out = pd.DataFrame(rows)
        out["patient_id"] = out["patient_id"].astype(str)
        out["record_id"] = out["record_id"].astype(str)
        out.sort_values(
            ["patient_id", "record_id"],
            key=lambda col: col.astype(int),
            inplace=True,
        )
        out.reset_index(drop=True, inplace=True)
        _write_csv_atomic(out, csv_path)
        logger.info(
            "Wrote PTB-XL metadata (%d records, %d Hz) to %s",
            len(out),
            self.sampling_rate,
            csv_path,
        )

    def _metadata_matches_request(self, csv_path: Path) -> bool:
        """Reuse cached metadata when the derived CSV has the expected schema.

        Filename already encodes sampling rate, data-root path, and source CSV
        content, so this only guards against a truncated or partial write.

        Args:
            csv_path (Path): Candidate derived metadata CSV.

        Returns:
            bool: True if the cache is safe to reuse.
        """
        try:
            df = pd.read_csv(csv_path, nrows=0)
        except (OSError, ValueError, pd.errors.ParserError):
            return False
        needed = {
            "patient_id",
            "record_id",
            "signal_file",
            "sampling_rate",
            "strat_fold",
            "age",
            "age_is_censored",
            "age_is_missing",
            "sex",
            "site",
            "device",
            "scp_codes",
            "recording_date",
        }
        return needed.issubset(df.columns)

    @property
    def default_task(self) -> PTBXLSuperclassClassification:
        """Return the 5-superclass multi-label task wired to this data root.

        Matches other signal datasets (EEGBCI / TUAB / SleepEDF): BaseDataset
        exposes ``default_task`` as a read-only property and never assigns to
        it in ``__init__``.

        Returns:
            PTBXLSuperclassClassification: Default task instance.
        """
        from pyhealth.tasks.ptbxl import PTBXLSuperclassClassification

        return PTBXLSuperclassClassification(
            scp_statements_path=str(self.scp_statements_path),
        )

"""
PyHealth dataset for the Chapman-Shaoxing / Ningbo 12-lead ECG database.

Dataset link:
    https://physionet.org/content/ecg-arrhythmia/1.0.0/

Dataset paper: (please cite if you use this dataset)
    Jianwei Zheng, Hangyuan Guo, and Huimin Chu. "A large scale 12-lead
    electrocardiogram database for arrhythmia study" (version 1.0.0).
    PhysioNet (2022). https://doi.org/10.13026/wgex-er52

    Jianwei Zheng, Jianming Zhang, Sidy Danioko, Hai Yao, Hangyuan Guo, and
    Cyril Rakovski. "A 12-lead electrocardiogram database for arrhythmia
    research covering more than 10,000 patients." Scientific Data 7, 48
    (2020). https://doi.org/10.1038/s41597-020-0386-x

Description:
    Implements ``ChapmanShaoxingDataset``. It scans the WFDB headers once,
    writes one metadata row per record into a cache directory (never into
    ``root``), and loads waveforms lazily in the tasks.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from filelock import FileLock

from .base_dataset import BaseDataset
from .configs.config import load_yaml_config
from .ptbxl import _write_csv_atomic, _write_yaml_atomic, root_cache_key
from .utils import MODULE_CACHE_PATH

if TYPE_CHECKING:
    from pyhealth.tasks.chapman_shaoxing_classification import (
        ChapmanShaoxingSuperclassClassification,
    )

logger = logging.getLogger(__name__)

_DEFAULT_METADATA_CACHE = Path(MODULE_CACHE_PATH) / "chapman_shaoxing"

#: Highest record number of the PhysioNet/CinC Challenge 2021 subset
#: (``WFDB_ChapmanShaoxing``, JS00001-JS10646) that CLIMB uses.
CLIMB_MAX_RECORD_NUMBER = 10646

METADATA_COLUMNS = (
    "patient_id",
    "record_id",
    "signal_file",
    "sampling_rate",
    "n_samples",
    "age",
    "sex",
    "dx_codes",
)

_RECORD_RE = re.compile(r"^JS(\d+)$")
_COMMENT_RE = re.compile(r"^#\s*(Age|Sex|Dx)\s*:\s*(.*)$", re.IGNORECASE)


def parse_chapman_header(path: str | Path) -> dict[str, Any]:
    """Parse one Chapman-Shaoxing WFDB ``.hea`` header without the waveform.

    Args:
        path (str | Path): Path to a ``JSxxxxx.hea`` file.

    Returns:
        dict[str, Any]: ``record_id``, ``sampling_rate``, ``n_samples``,
        ``age`` (int, or -1 when missing), ``sex`` and ``dx_codes`` (SNOMED-CT
        codes joined with ``;``).

    Examples:
        >>> import tempfile, pathlib
        >>> hea = pathlib.Path(tempfile.mkdtemp()) / "JS00001.hea"
        >>> _ = hea.write_text("JS00001 12 500 5000\\n#Age: 85\\n#Sex: Male\\n#Dx: 164889003,59118001\\n")
        >>> parse_chapman_header(hea)["dx_codes"]
        '164889003;59118001'
    """
    lines = Path(path).read_text(errors="replace").splitlines()
    first = lines[0].split()
    fields: dict[str, str] = {}
    for line in lines[1:]:
        match = _COMMENT_RE.match(line.strip())
        if match:
            fields[match.group(1).lower()] = match.group(2).strip()
    age_text = fields.get("age", "")
    codes = [c.strip() for c in fields.get("dx", "").split(",") if c.strip()]
    return {
        "record_id": first[0],
        "sampling_rate": int(float(first[2])) if len(first) > 2 else 500,
        "n_samples": int(first[3]) if len(first) > 3 else -1,
        "age": int(age_text) if age_text.isdigit() else -1,
        "sex": fields.get("sex", "Unknown") or "Unknown",
        "dx_codes": ";".join(codes),
    }


class ChapmanShaoxingDataset(BaseDataset):
    """Chapman University / Shaoxing and Ningbo hospitals 12-lead ECG database.

    45,152 resting 12-lead ECGs of 10 s at 500 Hz, one per patient. Each is a
    WFDB ``.hea`` / ``.mat`` pair whose header carries ``#Age``, ``#Sex`` and
    ``#Dx`` (comma-separated SNOMED-CT codes). Each record becomes one event
    of table ``records``; waveforms are read lazily by the tasks.

    Getting the data:
        - Source: https://physionet.org/content/ecg-arrhythmia/1.0.0/
          (DOI 10.13026/wgex-er52).
        - Access: open, no account needed. 2.3 GB zip, 5.1 GB uncompressed.
        - Download: ``wget -r -N -c -np https://physionet.org/files/ecg-arrhythmia/1.0.0/``
          or the zip linked on that page.
        - ``root``: the version folder holding ``WFDBRecords/`` (and
          ``ConditionNames_SNOMED-CT.csv``). Headers are found recursively,
          so the flat ``WFDB_ChapmanShaoxing/`` folder of the PhysioNet/CinC
          Challenge 2021 also works.

    Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).
    Cite the dataset papers in the module docstring when you use it.

    Benchmarked in CLIMB (Dai et al., 2025, https://github.com/DDVD233/CLIMB).

    Deviations from CLIMB:
        - CLIMB points at the 45k ``ecg-arrhythmia`` release but its labels
          and split cover only the 10,247-record Challenge subset
          (JS00001-JS10646), and its code table raises ``KeyError`` on codes
          that occur only in the full release. Here ``subset="full"``
          (default) keeps every record, and codes no table covers are ignored
          and counted in a log message. ``subset="climb"`` keeps records up to
          JS10646 to approximate CLIMB's cohort.
        - CLIMB's split is an unseeded stratified 80/20 split and cannot be
          reproduced; there is no official split. Use
          :class:`~pyhealth.datasets.PatientSplit` with a seed.

    Note:
        The source has no patient identifier separate from the record name
        and states one ECG per patient, so ``patient_id`` equals the record
        name (``JS00001``) and patient and record splits coincide.

    Args:
        root (str): The version folder of the download.
        dataset_name (str | None): Optional name prefix. A content hash of
            the derived metadata is appended so different roots never share
            a cache.
        config_path (str | Path | None): Optional YAML config; defaults to
            ``configs/chapman_shaoxing.yaml``.
        subset (str): ``"full"`` (default) or ``"climb"``.
        metadata_cache_dir (str | Path | None): Where the derived metadata
            CSV and resolved YAML are written. Defaults to
            ``~/.cache/pyhealth/datasets/chapman_shaoxing``.
        refresh_metadata (bool): Rescan the headers even if a cached metadata
            CSV exists. Defaults to ``False``.
        **kwargs: Forwarded to :class:`BaseDataset` (``cache_dir``, ``dev``...).

    Examples:
        >>> from pyhealth.datasets import ChapmanShaoxingDataset
        >>> ds = ChapmanShaoxingDataset(root="/data/ecg-arrhythmia/1.0.0")  # doctest: +SKIP
        >>> ds.stats()  # doctest: +SKIP
        >>> samples = ds.set_task()  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | Path | None = None,
        subset: str = "full",
        metadata_cache_dir: str | Path | None = None,
        refresh_metadata: bool = False,
        **kwargs,
    ) -> None:
        subset = (subset or "full").lower()
        if subset not in {"full", "climb"}:
            raise ValueError(f"subset must be 'full' or 'climb', got {subset!r}")
        self.subset = subset
        self.data_root = Path(root)
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"root does not exist: {self.data_root}")
        self.metadata_cache_dir = Path(
            metadata_cache_dir if metadata_cache_dir is not None else _DEFAULT_METADATA_CACHE
        )
        self.refresh_metadata = refresh_metadata
        self.metadata_file_name = (
            f"chapman_shaoxing-pyhealth-{subset}-{root_cache_key(self.data_root)}.csv"
        )
        package_config = (
            Path(config_path)
            if config_path is not None
            else Path(__file__).parent / "configs" / "chapman_shaoxing.yaml"
        )
        self.prepare_metadata()
        resolved_config = self._write_resolved_config(package_config)
        csv_path = self.metadata_cache_dir / self.metadata_file_name
        content_key = hashlib.sha1(csv_path.read_bytes()).hexdigest()[:10]
        super().__init__(
            root=str(self.data_root),
            tables=["records"],
            dataset_name=f"{dataset_name or 'chapman_shaoxing'}_{subset}_{content_key}",
            config_path=str(resolved_config),
            **kwargs,
        )

    def load_table(self, table_name: str):
        """Read the derived metadata CSV from ``metadata_cache_dir``.

        Args:
            table_name (str): Table name from the YAML config.

        Returns:
            The dataframe produced by :meth:`BaseDataset.load_table`.
        """
        original_root = self.root
        self.root = str(self.metadata_cache_dir)
        try:
            return super().load_table(table_name)
        finally:
            self.root = original_root

    def _write_resolved_config(self, package_config: Path) -> Path:
        """Write a YAML whose ``file_path`` names the derived metadata CSV."""
        config = load_yaml_config(str(package_config))
        config.tables["records"].file_path = self.metadata_file_name
        out_path = self.metadata_cache_dir / self.metadata_file_name.replace(
            ".csv", ".yaml"
        )
        with FileLock(str(out_path) + ".lock"):
            _write_yaml_atomic(config.model_dump(), out_path)
        return out_path

    def prepare_metadata(self) -> None:
        """Scan the headers and write the derived metadata CSV (once).

        Uses a file lock so concurrent processes do not scan twice.

        Examples:
            >>> ds = ChapmanShaoxingDataset(root="/data/ecg-arrhythmia/1.0.0")  # doctest: +SKIP
            >>> ds.prepare_metadata()  # doctest: +SKIP
        """
        csv_path = self.metadata_cache_dir / self.metadata_file_name
        self.metadata_cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.refresh_metadata and self._metadata_is_valid(csv_path):
            return
        with FileLock(str(csv_path) + ".lock"):
            if not self.refresh_metadata and self._metadata_is_valid(csv_path):
                return
            self._write_metadata(csv_path)

    @staticmethod
    def _metadata_is_valid(csv_path: Path) -> bool:
        if not csv_path.is_file():
            return False
        try:
            columns = pd.read_csv(csv_path, nrows=0).columns
        except (OSError, ValueError, pd.errors.ParserError):
            return False
        return set(METADATA_COLUMNS).issubset(columns)

    def _write_metadata(self, csv_path: Path) -> None:
        rows = []
        missing_mat = 0
        for hea in sorted(self.data_root.rglob("JS*.hea")):
            match = _RECORD_RE.match(hea.stem)
            if match is None:
                continue
            if (
                self.subset == "climb"
                and int(match.group(1)) > CLIMB_MAX_RECORD_NUMBER
            ):
                continue
            if not hea.with_suffix(".mat").is_file():
                missing_mat += 1
                continue
            row = parse_chapman_header(hea)
            row["patient_id"] = row["record_id"]
            row["signal_file"] = str(hea.with_suffix("").resolve())
            rows.append(row)
        if not rows:
            raise FileNotFoundError(
                f"No JS*.hea / JS*.mat pairs under {self.data_root}. Download "
                "https://physionet.org/content/ecg-arrhythmia/1.0.0/"
            )
        if missing_mat:
            logger.warning("Skipped %d headers with no .mat file", missing_mat)
        df = pd.DataFrame(rows, columns=list(METADATA_COLUMNS))
        df = df.sort_values("record_id").reset_index(drop=True)
        _write_csv_atomic(df, csv_path)
        logger.info("Wrote %d Chapman-Shaoxing records to %s", len(df), csv_path)
        self._log_unmapped_codes(df["dx_codes"])

    @staticmethod
    def _log_unmapped_codes(dx_codes: pd.Series) -> None:
        from pyhealth.tasks.chapman_shaoxing_classification import (
            count_unmapped_codes,
        )

        counts = count_unmapped_codes(dx_codes.tolist())
        if counts:
            top = ", ".join(f"{c} ({n})" for c, n in counts.most_common(10))
            logger.info(
                "%d SNOMED codes are outside the superclass table and are "
                "ignored by ChapmanShaoxingSuperclassClassification; most "
                "frequent (records): %s",
                len(counts),
                top,
            )

    @property
    def condition_names_path(self) -> Path:
        """Path to the release's ``ConditionNames_SNOMED-CT.csv``."""
        return self.data_root / "ConditionNames_SNOMED-CT.csv"

    @property
    def default_task(self) -> ChapmanShaoxingSuperclassClassification:
        """The seven-superclass multi-label task.

        Returns:
            ChapmanShaoxingSuperclassClassification: Default task instance.
        """
        from pyhealth.tasks.chapman_shaoxing_classification import (
            ChapmanShaoxingSuperclassClassification,
        )

        return ChapmanShaoxingSuperclassClassification()

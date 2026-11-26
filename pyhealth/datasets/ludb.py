"""Lobachevsky University Electrocardiography Database (LUDB).

Dataset link:
    https://physionet.org/content/ludb/1.0.1/

Dataset paper: (please cite if you use this dataset)
    Kalyakulina, A.I., Yusipov, I.I., Moskalenko, V.A., Nikolskiy, A.V.,
    Kosonogov, K.A., Osipov, G.V., Zolotykh, N.Yu., Ivanchenko, M.V.: LUDB: A
    New Open-Access Validation Tool for Electrocardiogram Delineation
    Algorithms, IEEE Access, vol. 8, pp. 186181-186190, 2020,
    doi: 10.1109/ACCESS.2020.3029211

Reproduced paper:
    Park, J., Park, T., Kwon, J. & Jo, Y.. (2025). Benchmarking ECG
    Delineation using Deep Neural Network-based Semantic Segmentation
    Models. Proceedings of the sixth Conference on Health, Inference, and
    Learning, in Proceedings of Machine Learning Research 287:63-88
    Available from https://proceedings.mlr.press/v287/park25a.html.

Authors:
    Mateo Ignacio (Mateo.Ignacio@gatech.edu), Minwoo Jo (mjo38@gatech.edu)

Description:
    Implements ``LUDBDataset`` (BaseDataset + YAML). Writes one metadata
    row per record, including whether it has P waves. Signals and wave
    boundaries are loaded on demand with :func:`load_ludb_record`. Also
    provides the train / validation / test split of Park et al.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from pyhealth.tasks import ECGDelineationLUDB

from .base_dataset import BaseDataset
from .metadata_csv import (
    metadata_csv_exists,
    resolve_metadata_root,
    write_metadata_csv,
)
from .splitter import SubjectSplit

logger = logging.getLogger(__name__)

_CSV_NAME = "ludb-pyhealth.csv"
_DATASET_KEY = "ludb"
_SOURCE_CSV = "ludb.csv"

#: Lead names in LUDB order; each is also the annotation file extension.
LUDB_LEADS: tuple[str, ...] = (
    "i", "ii", "iii", "avr", "avl", "avf",  # 6 limb leads
    "v1", "v2", "v3", "v4", "v5", "v6",  # 6 chest (precordial) leads
)  # fmt: skip

#: LUDB sampling frequency in Hz.
LUDB_SAMPLING_RATE = 500

#: Maps a WFDB peak symbol to the wave it marks.
WAVE_SYMBOLS: dict[str, str] = {"p": "P", "N": "QRS", "t": "T"}

#: ``ludb.csv`` encodes subjects aged 90 or older as ``">89"``.
AGE_CENSOR_TEXT = ">89"
AGE_CENSOR_VALUE = 90

#: Columns written by :meth:`LUDBDataset.prepare_metadata`, in order.
METADATA_COLUMNS: tuple[str, ...] = (
    "patient_id",
    "record_id",
    "signal_file",
    "age",
    "age_is_censored",
    "sex",
    "rhythm",
    "electric_axis",
    "has_p_wave",
)

WaveSegment = tuple[str, int, int]

# Record-level 8:1:1 split released by Park et al. (2025) on Hugging Face
# (res/ludb/dataset/{train,valid,test}_for_public.csv), stratified by
# "no p", rhythm and electric axis.
_PARK2025_VAL = tuple(
    str(i)
    for i in (1, 11, 24, 26, 30, 39, 58, 61, 69, 73, 75, 95, 96, 117, 141,
              142, 143, 162, 166, 185)
)  # fmt: skip
_PARK2025_TEST = tuple(
    str(i)
    for i in (12, 15, 25, 27, 29, 36, 41, 45, 57, 79, 80, 85, 93, 97, 136,
              139, 152, 161, 173, 190)
)  # fmt: skip
_PARK2025_TRAIN = tuple(
    str(i) for i in range(1, 201) if str(i) not in _PARK2025_VAL + _PARK2025_TEST
)


def _require_wfdb() -> ModuleType:
    """Import ``wfdb`` or explain how to install it.

    Returns:
        The imported ``wfdb`` module.

    Raises:
        ImportError: If ``wfdb`` is not installed.
    """
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError(
            "Reading LUDB records requires the optional 'wfdb' dependency. "
            "Install it with: pip install wfdb"
        ) from exc
    return wfdb


def parse_wave_annotations(
    symbols: Sequence[str], samples: Sequence[int]
) -> list[WaveSegment]:
    """Turn a WFDB ``( peak )`` annotation stream into wave segments.

    LUDB marks each wave with three annotations: ``(`` at the onset, the
    peak symbol (``p``, ``N`` or ``t``) and ``)`` at the offset. About 110
    waves lack one of the two brackets, mostly QRS complexes whose onset
    coincides with the peak (the database has 58,321 ``(`` but 58,423
    ``)`` markers). For those the peak sample stands in for the missing
    bound. Waves with neither bracket are skipped. This rule reproduces the
    wave boundaries in the public LUDB splits of Park et al. exactly.

    Args:
        symbols: Annotation symbols in sample order.
        samples: Sample index of each symbol.

    Returns:
        ``(wave, onset, offset)`` tuples in sample order, where ``wave`` is
        ``"P"``, ``"QRS"`` or ``"T"`` and ``onset <= offset`` are inclusive
        sample indices.

    Raises:
        ValueError: If ``symbols`` and ``samples`` differ in length.

    Examples:
        >>> parse_wave_annotations(["(", "N", ")", "t", ")"], [10, 12, 15, 30, 40])
        [('QRS', 10, 15), ('T', 30, 40)]
    """
    if len(symbols) != len(samples):
        raise ValueError(
            f"Got {len(symbols)} symbols but {len(samples)} sample positions."
        )
    segments: list[WaveSegment] = []
    for idx, symbol in enumerate(symbols):
        wave = WAVE_SYMBOLS.get(symbol)
        if wave is None:
            continue
        has_onset = idx > 0 and symbols[idx - 1] == "("
        has_offset = idx + 1 < len(symbols) and symbols[idx + 1] == ")"
        if not (has_onset or has_offset):
            continue
        onset = int(samples[idx - 1] if has_onset else samples[idx])
        offset = int(samples[idx + 1] if has_offset else samples[idx])
        if onset > offset:
            continue
        segments.append((wave, onset, offset))
    return segments


def load_ludb_record(
    record_path: str | Path, leads: Sequence[str] | None = None
) -> tuple[np.ndarray, dict[str, list[WaveSegment]]]:
    """Load one LUDB record and its per-lead wave annotations.

    Args:
        record_path: WFDB record base path, without extension (the
            ``signal_file`` column of the metadata CSV).
        leads: Leads to load, as LUDB lead names. Defaults to all 12 leads
            in :data:`LUDB_LEADS` order.

    Returns:
        A tuple ``(signal, annotations)``. ``signal`` is a float32 array of
        shape ``(n_leads, n_samples)`` in millivolts. ``annotations`` maps
        each requested lead to the output of
        :func:`parse_wave_annotations`.

    Raises:
        ImportError: If ``wfdb`` is not installed.
        FileNotFoundError: If the ``.hea`` header or the annotation file
            of a requested lead is missing.
        ValueError: If a requested lead is not in the record.

    Examples:
        >>> signal, ann = load_ludb_record("/data/ludb/data/1")  # doctest: +SKIP
        >>> signal.shape  # doctest: +SKIP
        (12, 5000)
        >>> ann["ii"][:2]  # doctest: +SKIP
        [('QRS', 644, 682), ('T', 776, 878)]
    """
    wfdb = _require_wfdb()
    path = Path(record_path)
    if path.suffix.lower() in {".hea", ".dat"}:
        path = path.with_suffix("")
    if not Path(f"{path}.hea").is_file():
        raise FileNotFoundError(f"LUDB header not found for record: {path}")

    leads = list(leads) if leads is not None else list(LUDB_LEADS)
    record = wfdb.rdrecord(str(path))
    names = [name.lower() for name in record.sig_name]
    missing = [lead for lead in leads if lead not in names]
    if missing:
        raise ValueError(f"Record {path.name} has no lead(s) {missing}.")

    columns = [names.index(lead) for lead in leads]
    signal = np.asarray(record.p_signal[:, columns], dtype=np.float32).T

    annotations: dict[str, list[WaveSegment]] = {}
    for lead in leads:
        ann = wfdb.rdann(str(path), lead)
        annotations[lead] = parse_wave_annotations(ann.symbol, ann.sample)
    return signal, annotations


def _clean_text(value: Any) -> str:
    """Strip whitespace from a ``ludb.csv`` cell; empty for missing values.

    Args:
        value: Raw cell value.

    Returns:
        The stripped string, or ``""`` for NaN / None.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _parse_age(value: Any) -> tuple[int | None, int]:
    """Parse the ``Age`` column of ``ludb.csv``.

    Args:
        value: Raw cell, e.g. ``"51\\n"`` or ``">89"``.

    Returns:
        ``(age, is_censored)``. ``">89"`` becomes ``(90, 1)``; a missing or
        non-numeric age becomes ``(None, 0)``.
    """
    text = _clean_text(value)
    if text == AGE_CENSOR_TEXT:
        return AGE_CENSOR_VALUE, 1
    try:
        return int(float(text)), 0
    except ValueError:
        return None, 0


def _parse_rhythm(value: Any) -> str:
    """Return the primary rhythm (first line of the ``Rhythms`` cell).

    A few records list a second finding such as ``Wandering atrial
    pacemaker`` on a new line; the first line is the rhythm itself, which
    matches the counts in the LUDB README.

    Args:
        value: Raw ``Rhythms`` cell.

    Returns:
        The primary rhythm, or ``""`` if missing.
    """
    lines = [line.strip() for line in _clean_text(value).splitlines()]
    return next((line for line in lines if line), "")


def _parse_electric_axis(value: Any) -> str:
    """Normalise the ``Electric axis of the heart`` cell.

    Args:
        value: Raw cell, e.g. ``"Electric axis of the heart: normal"``.

    Returns:
        The axis category in lower case (``"normal"``, ``"left axis
        deviation"``, ...). Records with a pacemaker have no axis and
        return ``"none"``.
    """
    text = _clean_text(value)
    if not text:
        return "none"
    return text.split(":", 1)[-1].strip().lower()


def _has_p_wave(record_path: Path) -> bool:
    """Check whether any of the 12 leads of a record has an annotated P wave.

    Args:
        record_path: WFDB record base path.

    Returns:
        True if at least one lead has a P wave. Leads whose annotation file
        is missing are skipped with a warning.
    """
    wfdb = _require_wfdb()
    for lead in LUDB_LEADS:
        if not Path(f"{record_path}.{lead}").is_file():
            logger.warning("Missing annotation file %s.%s", record_path, lead)
            continue
        ann = wfdb.rdann(str(record_path), lead)
        segments = parse_wave_annotations(ann.symbol, ann.sample)
        if any(wave == "P" for wave, _, _ in segments):
            return True
    return False


def _metadata_is_current(csv_path: Path) -> bool:
    """Check that an existing ``ludb-pyhealth.csv`` can be reused.

    The CSV stores absolute ``signal_file`` paths, so it goes stale when the
    LUDB folder is moved, copied to another machine or partly deleted. It
    is reused only when it has every column in :data:`METADATA_COLUMNS`,
    at least one row, and a ``.hea`` header for every ``signal_file``.

    Args:
        csv_path: Candidate metadata CSV.

    Returns:
        True if the CSV is complete and all of its records are on disk.
    """
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    if not set(METADATA_COLUMNS).issubset(df.columns) or df.empty:
        return False
    return all(
        isinstance(path, str) and Path(f"{path}.hea").is_file()
        for path in df["signal_file"]
    )


class LUDBDataset(BaseDataset):
    """Lobachevsky University Electrocardiography Database (LUDB v1.0.1).

    LUDB holds 200 ten-second, 12-lead ECGs sampled at 500 Hz, one per
    subject. Cardiologists marked the onset, peak and offset of every
    P wave, QRS complex and T wave on each lead, which makes LUDB the
    standard public benchmark for ECG delineation.

    ``root`` must be the extracted PhysioNet folder that contains
    ``ludb.csv`` and the ``data/`` directory of WFDB files.
    :meth:`prepare_metadata` writes ``ludb-pyhealth.csv`` (one row per
    record) next to the raw files, or under
    ``~/.cache/pyhealth/ludb/<hash of root>`` when ``root`` is read-only.
    Each record is its own patient, so a patient-level split never puts
    leads of the same ECG in two subsets, as required by Park et al.

    Metadata columns:
        - ``record_id`` / ``patient_id``: LUDB record number as text.
        - ``signal_file``: absolute WFDB record base path; pass it to
          :func:`load_ludb_record`.
        - ``age`` (``>89`` is stored as 90), ``age_is_censored``, ``sex``.
        - ``rhythm``: primary rhythm, e.g. ``"Sinus rhythm"``.
        - ``electric_axis``: e.g. ``"normal"``; ``"none"`` for paced ECGs.
        - ``has_p_wave``: 1 if any lead has an annotated P wave. This is the
          ``no p`` label of Park et al. inverted (24 records have none).

    Args:
        root: Path to the LUDB folder holding ``ludb.csv`` and ``data/``.
        dataset_name: Name stored on the dataset. Defaults to ``"ludb"``.
        config_path: YAML config. Defaults to ``configs/ludb.yaml``.
        **kwargs: Forwarded to :class:`~pyhealth.datasets.BaseDataset`
            (``cache_dir``, ``dev``, ``num_workers``).

    Raises:
        FileNotFoundError: If ``root``, ``ludb.csv`` or ``data/`` is missing.
        ValueError: If no record in ``ludb.csv`` has WFDB files on disk.

    Examples:
        >>> from pyhealth.datasets import LUDBDataset
        >>> from pyhealth.datasets.ludb import load_ludb_record
        >>> dataset = LUDBDataset(
        ...     root="/data/lobachevsky-university-electrocardiography-database-1.0.1",
        ... )  # doctest: +SKIP
        >>> dataset.stats()  # doctest: +SKIP
        >>> patient = dataset.get_patient("1")  # doctest: +SKIP
        >>> event = patient.get_events(event_type="recordings")[0]  # doctest: +SKIP
        >>> signal, annotations = load_ludb_record(event.signal_file)  # doctest: +SKIP
    """

    #: Record-level 8:1:1 split released by Park et al. (2025); see
    #: :meth:`default_subject_split`.
    PARK2025_TRAIN_RECORDS: tuple[str, ...] = _PARK2025_TRAIN
    PARK2025_VAL_RECORDS: tuple[str, ...] = _PARK2025_VAL
    PARK2025_TEST_RECORDS: tuple[str, ...] = _PARK2025_TEST

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "ludb.yaml"

        self.root = root
        self.prepare_metadata()

        super().__init__(
            root=resolve_metadata_root(root, _DATASET_KEY, _CSV_NAME),
            tables=["recordings"],
            dataset_name=dataset_name or "ludb",
            config_path=str(config_path),
            **kwargs,
        )

    def prepare_metadata(self) -> None:
        """Write ``ludb-pyhealth.csv`` unless a current copy is on disk.

        Records listed in ``ludb.csv`` whose ``.hea`` file is missing are
        skipped with a warning. An existing CSV, in ``root`` or in the home
        cache, is reused only if it has every expected column and all of
        its ``signal_file`` paths still exist; otherwise it is rebuilt, for
        example after the LUDB folder has been moved.

        Returns:
            None. The CSV is written to ``<root>/ludb-pyhealth.csv`` or to
            ``~/.cache/pyhealth/ludb/<hash of root>/ludb-pyhealth.csv``.

        Raises:
            FileNotFoundError: If ``root``, ``ludb.csv`` or ``data/`` is
                missing.
            ValueError: If ``ludb.csv`` lacks required columns or no record
                has WFDB files on disk.
        """
        root = Path(self.root)
        if not root.is_dir():
            raise FileNotFoundError(f"LUDB root does not exist: {root}")
        if metadata_csv_exists(root, _DATASET_KEY, _CSV_NAME):
            csv_path = (
                Path(resolve_metadata_root(root, _DATASET_KEY, _CSV_NAME)) / _CSV_NAME
            )
            if _metadata_is_current(csv_path):
                return
            logger.warning("Rebuilding stale or incomplete LUDB metadata %s", csv_path)

        source_csv = root / _SOURCE_CSV
        data_dir = root / "data"
        if not source_csv.is_file() or not data_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {_SOURCE_CSV} and data/ under root={root}. "
                "Download LUDB from https://physionet.org/content/ludb/1.0.1/"
            )

        source = pd.read_csv(source_csv)
        required = {"ID", "Sex", "Age", "Rhythms", "Electric axis of the heart"}
        missing = required - set(source.columns)
        if missing:
            raise ValueError(f"{_SOURCE_CSV} is missing columns: {sorted(missing)}")

        rows: list[dict[str, Any]] = []
        for row in source.to_dict(orient="records"):
            record_id = str(int(row["ID"]))
            record_path = data_dir / record_id
            if not Path(f"{record_path}.hea").is_file():
                logger.warning("Skipping LUDB record %s: no .hea file", record_id)
                continue
            age, age_is_censored = _parse_age(row["Age"])
            rows.append(
                {
                    "patient_id": record_id,
                    "record_id": record_id,
                    "signal_file": str(record_path.resolve()),
                    "age": age,
                    "age_is_censored": age_is_censored,
                    "sex": _clean_text(row["Sex"]),
                    "rhythm": _parse_rhythm(row["Rhythms"]),
                    "electric_axis": _parse_electric_axis(
                        row["Electric axis of the heart"]
                    ),
                    "has_p_wave": int(_has_p_wave(record_path)),
                }
            )
        if not rows:
            raise ValueError(f"No LUDB records with WFDB files found under {root}")

        df = pd.DataFrame(rows, columns=list(METADATA_COLUMNS))
        df["age"] = df["age"].astype("Int64")
        df.sort_values("record_id", key=lambda col: col.astype(int), inplace=True)
        df.reset_index(drop=True, inplace=True)
        write_metadata_csv(df, root, _DATASET_KEY, _CSV_NAME)

    @classmethod
    def default_subject_split(cls) -> SubjectSplit:
        """Return the train / validation / test records of Park et al. (2025).

        Using the authors' own split makes test scores directly comparable
        with the paper: 160 training, 20 validation and 20 test records.

        Returns:
            A :class:`~pyhealth.datasets.SubjectSplit` with those three parts.

        Examples:
            >>> from pyhealth.datasets import LUDBDataset
            >>> split = LUDBDataset.default_subject_split()
            >>> [len(part) for part in split.subjects]
            [160, 20, 20]
        """
        return SubjectSplit(
            (
                cls.PARK2025_TRAIN_RECORDS,
                cls.PARK2025_VAL_RECORDS,
                cls.PARK2025_TEST_RECORDS,
            )
        )

    @property
    def default_task(self) -> ECGDelineationLUDB:
        """Return the P / QRS / T delineation task of Park et al.

        Returns:
            An :class:`~pyhealth.tasks.ECGDelineationLUDB` with the paper's
            0.05-150 Hz band-pass filter over all 12 leads.
        """
        return ECGDelineationLUDB()

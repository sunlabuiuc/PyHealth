"""PyHealth dataset for the PhysioNet/Computing in Cardiology Challenge 2012.

Dataset link:
    https://physionet.org/content/challenge-2012/1.0.0/

Dataset paper: (please cite if you use this dataset)
    Ikaro Silva, George Moody, Daniel J. Scott, Leo A. Celi, and Roger G. Mark.
    "Predicting In-Hospital Mortality of ICU Patients: The PhysioNet/Computing
    in Cardiology Challenge 2012." Computing in Cardiology 39 (2012): 245-248.

Description:
    12,000 ICU stays of at least 48 hours (sets a, b and c, 4,000 each). Each
    record holds up to 37 irregularly sampled time series from the first 48
    hours (vital signs and labs) and five general descriptors recorded on
    admission (age, gender, height, ICU type and weight). The data is open access, so it can be downloaded without
    credentials (``download=True``).
"""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import time
import urllib.request
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

PHYSIONET2012_URL = "https://physionet.org/files/challenge-2012/1.0.0/"
PHYSIONET2012_SETS = ("a", "b", "c")

GENERAL_DESCRIPTORS = ("Age", "Gender", "Height", "ICUType", "Weight")
"""Descriptors recorded once on admission, at time 00:00."""


EVENTS_CSV = "physionet2012-pyhealth-events.csv"
PATIENTS_CSV = "physionet2012-pyhealth-patients.csv"

# Records have elapsed times, not dates; they are placed after this origin.
TIME_ORIGIN = datetime.fromisoformat("2012-01-01")  # naive, like PyHealth timestamps
_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

_OUTCOME_COLUMNS = {
    "SAPS-I": "saps_i",
    "SOFA": "sofa",
    "Length_of_stay": "length_of_stay",
    "Survival": "survival",
    "In-hospital_death": "in_hospital_death",
}


def download_physionet2012(
    root: str | Path, sets: Sequence[str] = PHYSIONET2012_SETS, retries: int = 3
) -> None:
    """Downloads and extracts the challenge records and outcomes into ``root``.

    Files already present are kept. Each file is written to a temporary name
    and renamed once complete, and each archive is extracted into a temporary
    folder that is renamed once complete, so an interrupted download or
    extraction is redone rather than used.

    Args:
        root: Directory to download into; ``set-a/`` etc. are created there.
        sets: Which record sets to download.
        retries: Attempts per file.

    Examples:
        >>> download_physionet2012("data/physionet2012")  # doctest: +SKIP
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in sets:
        for filename in (f"set-{name}.tar.gz", f"Outcomes-{name}.txt"):
            _fetch(PHYSIONET2012_URL + filename, root / filename, retries)
        _extract(root / f"set-{name}.tar.gz", root / f"set-{name}")


def _extract(archive_path: Path, dest: Path) -> None:
    """Extracts ``archive_path`` (which holds ``dest.name/``) to ``dest``."""
    if dest.is_dir():
        return
    tmp = dest.with_name(f".{dest.name}.partial")
    shutil.rmtree(tmp, ignore_errors=True)  # left by an interrupted run
    tmp.mkdir()
    try:
        with tarfile.open(archive_path) as archive:
            archive.extractall(tmp, filter="data")
        os.replace(tmp / dest.name, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _fetch(url: str, dest: Path, retries: int) -> None:
    if dest.exists():
        return
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            logger.info("Downloading %s", url)
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, dest)
            return
        except OSError:
            tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


def read_record(path: str | Path) -> tuple[str, dict[str, float], list[tuple]]:
    """Reads one challenge record file.

    Args:
        path: A record file such as ``set-a/132539.txt``.

    Returns:
        ``(record_id, descriptors, events)``: the general descriptors (missing
        ones, coded ``-1``, as NaN) and the time-series observations as
        ``(minutes since admission, parameter, value)``.

    Examples:
        >>> record_id, descriptors, events = read_record("set-a/132539.txt")  # doctest: +SKIP
        >>> events[0]  # doctest: +SKIP
        (7, 'GCS', 15.0)
    """
    record_id = None
    descriptors: dict[str, float] = {}
    events: list[tuple] = []
    with open(path) as handle:
        next(handle)  # header: Time,Parameter,Value
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) != 3 or not parts[1]:
                continue
            stamp, parameter, raw = parts
            hours, minutes = stamp.split(":")
            elapsed = int(hours) * 60 + int(minutes)
            value = float(raw)
            if parameter == "RecordID":
                record_id = str(int(value))
            elif parameter in GENERAL_DESCRIPTORS and elapsed == 0:
                descriptors[parameter] = value if value >= 0 else float("nan")
            else:
                events.append((elapsed, parameter, value))
    if record_id is None:
        raise ValueError(f"{path} has no RecordID")
    return record_id, descriptors, events


def prepare_physionet2012(
    root: str | Path, sets: Iterable[str] = PHYSIONET2012_SETS
) -> None:
    """Writes the event and patient tables that the dataset config reads.

    Reads every ``set-*/*.txt`` record (and ``Outcomes-*.txt`` when present)
    in ``root`` and writes :data:`EVENTS_CSV` (one row per observation) and
    :data:`PATIENTS_CSV` (descriptors and outcomes) next to them.

    Args:
        root: Directory with the extracted ``set-a/``, ``set-b/``, ``set-c/``.
        sets: The record sets to include; missing ones are skipped.

    Examples:
        >>> prepare_physionet2012("data/physionet2012")  # doctest: +SKIP
    """
    root = Path(root)
    events, patients = [], []
    for name in sets:
        folder = root / f"set-{name}"
        if not folder.is_dir():
            logger.warning("%s not found; set %s is skipped", folder, name)
            continue
        outcomes = _read_outcomes(root / f"Outcomes-{name}.txt")
        for path in sorted(folder.glob("*.txt")):
            record_id, descriptors, record_events = read_record(path)
            patients.append({
                "patient_id": record_id,
                "set": name,
                **{k.lower(): descriptors.get(k) for k in GENERAL_DESCRIPTORS},
                **outcomes.get(record_id, {}),
            })
            for elapsed, parameter, value in record_events:
                stamp = TIME_ORIGIN + timedelta(minutes=elapsed)
                events.append(
                    (record_id, stamp.strftime(_TIME_FORMAT), elapsed, parameter, value)
                )
    if not patients:
        raise FileNotFoundError(
            f"No set-*/ record folders in {root}; download them with "
            "download_physionet2012 or PhysioNet2012Dataset(download=True)."
        )
    events_df = pd.DataFrame(
        events, columns=["patient_id", "time", "minutes", "parameter", "value"]
    )
    patients_df = pd.DataFrame(patients)
    for column in _OUTCOME_COLUMNS.values():
        if column not in patients_df:
            patients_df[column] = float("nan")
    _write_csv(events_df, root / EVENTS_CSV)
    _write_csv(patients_df, root / PATIENTS_CSV)


def _read_outcomes(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    table = pd.read_csv(path).rename(columns=_OUTCOME_COLUMNS)
    table["RecordID"] = table["RecordID"].astype(int).astype(str)
    return table.set_index("RecordID").to_dict("index")


def _write_csv(df: pd.DataFrame, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, dest)


class PhysioNet2012Dataset(BaseDataset):
    """The PhysioNet/CinC Challenge 2012 ICU dataset.

    Two tables: ``events`` holds every time-series observation (``parameter``,
    ``value`` and ``minutes`` since ICU admission; the timestamp is
    :data:`TIME_ORIGIN` plus that elapsed time), and ``patients`` holds the
    general descriptors (``age``, ``gender``, ``height``, ``icutype``,
    ``weight``) and the outcomes (``saps_i``, ``sofa``, ``length_of_stay``,
    ``survival``, ``in_hospital_death``). Each record (ICU stay) is one
    patient, keyed by its ``RecordID``.

    The tables are built once from the raw record files and written into
    ``root`` (:data:`EVENTS_CSV`, :data:`PATIENTS_CSV`).

    Getting the data:
        - Source: https://physionet.org/content/challenge-2012/1.0.0/
        - Access: open, no account needed.
        - Download: pass ``download=True``, or fetch it yourself with
          ``wget -r -N -c -np https://physionet.org/files/challenge-2012/1.0.0/``.
        - ``root``: the folder holding the extracted ``set-a/``, ``set-b/`` and
          ``set-c/`` folders.

    Args:
        root: Directory with the extracted ``set-a/``, ``set-b/``, ``set-c/``
            folders (and ``Outcomes-*.txt``), or where to download them.
        tables: Tables to load. Defaults to ``["events", "patients"]``.
        download: Download the open-access files from PhysioNet into
            ``root`` if they are missing.
        sets: Record sets to include. Changing it after the tables were
            written requires deleting them so they are rebuilt.
        dataset_name: Name of the dataset. Defaults to ``"physionet2012"``.
        config_path: Path to a YAML config. Defaults to the bundled one.
        **kwargs: Passed to :class:`~pyhealth.datasets.BaseDataset`
            (e.g. ``cache_dir``, ``dev``).

    Examples:
        >>> from pyhealth.datasets import PhysioNet2012Dataset
        >>> dataset = PhysioNet2012Dataset(
        ...     root="data/physionet2012", download=True
        ... )  # doctest: +SKIP
        >>> dataset.stats()  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str,
        tables: list[str] | None = None,
        download: bool = False,
        sets: Sequence[str] = PHYSIONET2012_SETS,
        dataset_name: str | None = None,
        config_path: str | None = None,
        **kwargs,
    ) -> None:
        root_path = Path(root)
        if download:
            download_physionet2012(root_path, sets)
        if not (root_path / EVENTS_CSV).exists() or not (
            root_path / PATIENTS_CSV
        ).exists():
            logger.info("Building the PhysioNet 2012 tables in %s", root_path)
            prepare_physionet2012(root_path, sets)
        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "physionet2012.yaml")
        super().__init__(
            root=str(root_path),
            tables=tables or ["events", "patients"],
            dataset_name=dataset_name or "physionet2012",
            config_path=config_path,
            **kwargs,
        )

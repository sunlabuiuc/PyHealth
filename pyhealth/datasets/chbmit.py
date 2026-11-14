"""CHB-MIT scalp EEG dataset.

Author: Jathurshan Pradeepkumar (jp65@illinois.edu), UIUC
Description: Loads the CHB-MIT Scalp EEG Database (PhysioNet chbmit 1.0.0).
    One metadata row is written per EDF recording. Seizure onsets come from
    the per-subject summary files. Windowing lives in
    ``SeizureDetectionCHBMIT`` and follows the CBraMod / TFM-Tokenizer
    preprocessing (16 bipolar channels, 10-second windows).

References:
    - Shoeb, A. H. Application of Machine Learning to Epileptic Seizure
      Onset Detection and Treatment. PhD Thesis, Massachusetts Institute
      of Technology, 2009.
      https://physionet.org/content/chbmit/1.0.0/
    - Wang, J. et al. CBraMod: A Criss-Cross Brain Foundation Model for
      EEG Decoding. ICLR 2025. https://github.com/wjq-learning/CBraMod
    - Pradeepkumar, J. et al. Tokenizing Single-Channel EEG with
      Time-Frequency Motif Learning. ICLR 2026.
      https://github.com/Jathurshan0330/TFM-Tokenizer
"""

import logging
from pathlib import Path

import pandas as pd

from pyhealth.tasks import SeizureDetectionCHBMIT

from .base_dataset import BaseDataset
from .metadata_csv import (
    metadata_cache_dir,
    metadata_csv_exists,
    resolve_metadata_root,
    write_metadata_csv,
)

logger = logging.getLogger(__name__)

_CSV_NAME = "chbmit-pyhealth.csv"


def _metadata_cache_dir(root: str | Path) -> Path:
    """Return the home-cache directory used when ``root`` is not writable.

    The directory is keyed by a hash of the resolved root, so two dataset
    roots never share cached metadata or each other's ``signal_file`` paths.

    Args:
        root: Dataset root directory.

    Returns:
        ``~/.cache/pyhealth/chbmit/<hash of root>``.
    """
    return metadata_cache_dir(root, "chbmit")


def _seconds_before_unit(line: str) -> int | None:
    """Read the integer that precedes ``seconds`` on a summary line.

    CHB-MIT writes ``Seizure Start Time: 2996 seconds`` and
    ``Seizure 1 Start Time: 1665 seconds``. Both put the number in the
    second-to-last field.

    Args:
        line: One line from a ``chbXX-summary.txt`` file.

    Returns:
        The integer number of seconds, or None if the line has no integer.
    """
    parts = line.split()
    if len(parts) < 2:
        return None
    try:
        return int(parts[-2])
    except ValueError:
        logger.warning("Skipping summary line with no integer time: %s", line)
        return None


def _parse_summary(summary_path: Path) -> list[dict]:
    """Parse seizure times from one subject summary file.

    A summary lists each EDF, then ``Number of Seizures in File`` and, for
    every seizure, a start line and an end line. Files with several seizures
    number those lines (``Seizure 1 Start Time``). Montage sections
    (``Channels changed``) are ignored.

    Args:
        summary_path: Path to ``chbXX-summary.txt``.

    Returns:
        One dict per ``File Name`` entry, with ``filename``, ``starts`` and
        ``ends`` (seconds). A file that is not on disk is still returned;
        the caller drops it.
    """
    records: list[dict] = []
    current: dict | None = None
    text = summary_path.read_text(errors="replace").splitlines()
    for raw_line in text:
        line = raw_line.strip()
        if line.startswith("File Name:"):
            if current is not None:
                records.append(current)
            current = {
                "filename": line.split(":", 1)[1].strip(),
                "starts": [],
                "ends": [],
            }
            continue
        if current is None or "Seizure" not in line:
            continue
        if "Start" in line:
            seconds = _seconds_before_unit(line)
            if seconds is not None:
                current["starts"].append(seconds)
        elif "End" in line:
            seconds = _seconds_before_unit(line)
            if seconds is not None:
                current["ends"].append(seconds)
    if current is not None:
        records.append(current)
    return records


def _read_subject_info(root: Path) -> dict[str, tuple[str, str]]:
    """Read gender and age from ``SUBJECT-INFO``.

    Args:
        root: Dataset root that may contain ``SUBJECT-INFO``.

    Returns:
        Mapping of subject id (``chb01``) to ``(gender, age)``. Missing file
        gives an empty mapping. Age is kept as text, as printed in the file.
    """
    path = root / "SUBJECT-INFO"
    if not path.is_file():
        return {}
    info: dict[str, tuple[str, str]] = {}
    for line in path.read_text(errors="replace").splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 3 and parts[0].lower().startswith("chb"):
            info[parts[0].lower()] = (parts[1], parts[2])
    return info


def _join_times(
    subject: str, filename: str, starts: list[int], ends: list[int]
) -> tuple[str, str]:
    """Pair seizure start and end times into CSV fields.

    Args:
        subject: Subject id, used only in the warning.
        filename: EDF file name, used only in the warning.
        starts: Seizure start times in seconds.
        ends: Seizure end times in seconds.

    Returns:
        ``(starts_field, ends_field)`` with times joined by ``;``. If the
        two lists differ in length, they are truncated to the shorter one.
    """
    if len(starts) != len(ends):
        logger.warning(
            "%s/%s has %d seizure starts and %d ends; pairing the first %d.",
            subject,
            filename,
            len(starts),
            len(ends),
            min(len(starts), len(ends)),
        )
        n = min(len(starts), len(ends))
        starts, ends = starts[:n], ends[:n]
    return ";".join(str(v) for v in starts), ";".join(str(v) for v in ends)


def _rows_for_subject(
    subject_dir: Path, demographics: dict[str, tuple[str, str]]
) -> list[dict]:
    """Build metadata rows for every EDF of one subject.

    Recordings named in the summary are kept only when the EDF exists.
    An EDF that is on disk but absent from the summary (some ``chb24``
    files) is kept with zero seizures, matching CBraMod, which still
    segments those files.

    Args:
        subject_dir: Directory ``chbXX`` containing the summary and EDFs.
        demographics: Output of :func:`_read_subject_info`.

    Returns:
        One row dict per EDF. ``patient_id`` is the directory name, so
        ``chb17a_03.edf`` stays under subject ``chb17``.
    """
    subject = subject_dir.name.lower()
    gender, age = demographics.get(subject, ("", ""))
    summary_path = subject_dir / f"{subject_dir.name}-summary.txt"
    parsed = _parse_summary(summary_path) if summary_path.is_file() else []

    rows: list[dict] = []
    seen: set[str] = set()
    for record in parsed:
        edf_path = subject_dir / record["filename"]
        if not edf_path.is_file():
            logger.warning("Summary names a missing EDF: %s", edf_path)
            continue
        seen.add(edf_path.name)
        starts, ends = _join_times(
            subject, record["filename"], record["starts"], record["ends"]
        )
        rows.append(
            {
                "patient_id": subject,
                "record_id": edf_path.stem,
                "signal_file": str(edf_path.resolve()),
                "num_seizures": len(starts.split(";")) if starts else 0,
                "seizure_starts": starts,
                "seizure_ends": ends,
                "gender": gender,
                "age": age,
            }
        )

    for edf_path in sorted(subject_dir.glob("*.edf")):
        if edf_path.name in seen:
            continue
        logger.info(
            "EDF not listed in the summary, recorded with 0 seizures: %s", edf_path.name
        )
        rows.append(
            {
                "patient_id": subject,
                "record_id": edf_path.stem,
                "signal_file": str(edf_path.resolve()),
                "num_seizures": 0,
                "seizure_starts": "",
                "seizure_ends": "",
                "gender": gender,
                "age": age,
            }
        )
    return rows


class CHBMITDataset(BaseDataset):
    """CHB-MIT Scalp EEG Database.

    The corpus (PhysioNet ``chbmit``, version 1.0.0) holds scalp EEG from
    23 pediatric subjects at Children's Hospital Boston. Case ``chb21`` was
    recorded from the same person as ``chb01``, about 1.5 years later, and
    ``chb24`` was added later with incomplete summary coverage. Recordings
    are bipolar at 256 Hz. Seizure intervals are read from
    ``chbXX/chbXX-summary.txt``.

    ``prepare_metadata`` writes ``chbmit-pyhealth.csv`` (one row per EDF)
    next to the raw files, or under ``~/.cache/pyhealth/chbmit/<hash of
    root>`` when the dataset directory is not writable. The default task,
    :class:`~pyhealth.tasks.SeizureDetectionCHBMIT`, cuts 10-second windows.

    The CBraMod / TFM-Tokenizer subject split is available as
    :meth:`default_subject_split`: ``chb01``-``chb20`` train, ``chb21`` and
    ``chb22`` validation, ``chb23`` and ``chb24`` test. Because ``chb21``
    is the same person as ``chb01``, that split is not fully
    subject-independent.

    Getting the data:
        - Source: https://physionet.org/content/chbmit/1.0.0/
        - Access: open, no account needed (about 43 GB).
        - Download: ``wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/``
        - ``root``: the folder holding ``chb01/`` ... ``chb24/``.

    Args:
        root: Directory that contains the ``chb01`` ... ``chb24`` folders.
        dataset_name: Name stored on the dataset. Defaults to ``"chbmit"``.
        config_path: YAML config. Defaults to ``configs/chbmit.yaml``.
        **kwargs: Forwarded to :class:`~pyhealth.datasets.BaseDataset`
            (``dev``, ``refresh_cache``, ``num_workers``, ``cache_dir``).

    Examples:
        >>> from pyhealth.datasets import CHBMITDataset
        >>> dataset = CHBMITDataset(
        ...     root="/srv/local/data/physionet.org/files/chbmit/1.0.0/",
        ... )
        >>> dataset.stats()
        >>> train, val, test = dataset.set_task(
        ...     split=dataset.default_subject_split(),
        ... )
    """

    CBRAMOD_TRAIN_SUBJECTS: tuple[str, ...] = tuple(f"chb{i:02d}" for i in range(1, 21))
    CBRAMOD_VAL_SUBJECTS: tuple[str, ...] = ("chb21", "chb22")
    CBRAMOD_TEST_SUBJECTS: tuple[str, ...] = ("chb23", "chb24")

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | None = None,
        **kwargs,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "chbmit.yaml"

        self.root = root
        self.prepare_metadata()

        super().__init__(
            root=resolve_metadata_root(root, "chbmit", _CSV_NAME),
            tables=["recordings"],
            dataset_name=dataset_name or "chbmit",
            config_path=config_path,
            **kwargs,
        )

    def prepare_metadata(self) -> None:
        """Write ``chbmit-pyhealth.csv`` if it is not already on disk.

        The CSV is one row per EDF with ``patient_id``, ``record_id``,
        ``signal_file``, ``num_seizures``, ``seizure_starts``,
        ``seizure_ends`` (seconds, joined by ``;``), ``gender`` and ``age``.
        An existing CSV, in ``root`` or in the cache, is left untouched.

        Returns:
            None. The CSV path is either ``<root>/chbmit-pyhealth.csv`` or
            ``~/.cache/pyhealth/chbmit/<hash of root>/chbmit-pyhealth.csv``.
        """
        root = Path(self.root)
        if metadata_csv_exists(root, "chbmit", _CSV_NAME):
            return

        demographics = _read_subject_info(root)
        rows: list[dict] = []
        subject_dirs = [
            p
            for p in sorted(root.glob("chb*"))
            if p.is_dir() and (p / f"{p.name}-summary.txt").is_file()
        ]
        for subject_dir in subject_dirs:
            rows.extend(_rows_for_subject(subject_dir, demographics))
        if not rows:
            logger.warning("No CHB-MIT recordings found under %s", root)
            return

        df = pd.DataFrame(rows)
        df.sort_values(["patient_id", "record_id"], inplace=True, na_position="last")
        df.reset_index(drop=True, inplace=True)
        write_metadata_csv(df, root, "chbmit", _CSV_NAME)

    @classmethod
    def default_subject_split(cls):
        """Return the CBraMod / TFM-Tokenizer subject split.

        Train is ``chb01``-``chb20``, validation is ``chb21`` and ``chb22``,
        and test is ``chb23`` and ``chb24``.

        Returns:
            A :class:`~pyhealth.datasets.SubjectSplit` with those three parts.

        Examples:
            >>> from pyhealth.datasets import CHBMITDataset
            >>> split = CHBMITDataset.default_subject_split()
            >>> [len(part) for part in split.subjects]
            [20, 2, 2]
        """
        from pyhealth.datasets.splitter import SubjectSplit

        return SubjectSplit(
            (
                cls.CBRAMOD_TRAIN_SUBJECTS,
                cls.CBRAMOD_VAL_SUBJECTS,
                cls.CBRAMOD_TEST_SUBJECTS,
            )
        )

    @property
    def default_task(self) -> SeizureDetectionCHBMIT:
        """Return the default seizure-detection task.

        Returns:
            A :class:`~pyhealth.tasks.SeizureDetectionCHBMIT` with its
            default windowing and normalization.
        """
        return SeizureDetectionCHBMIT()

"""NHANES 2003-2006 accelerometer and linked-mortality dataset.

Paper: Song, Yin, et al., "MotionAge: A Deep Learning Framework for Biological
Age Prediction from Wearable Activity." https://openreview.net/pdf?id=xGw1o9acWa

Turns the raw NHANES 2003-2004 (``_C``) and 2005-2006 (``_D``) files into one
participant table (demographics, body measures, blood pressure and linked
mortality) plus one wear-aware 5-minute epoch file per participant. Non-wear
time is detected from runs of zero counts, so a device left in a drawer is not
mistaken for a participant sitting still: either with the MotionAge authors'
implementation (the default, to reproduce the paper) or with the published
Choi et al. (2011) rule.
"""

import itertools
import json
import logging
import math
import os
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

# NHANES cycle suffix -> years, as used in the mortality file names.
_CYCLES = {"C": "2003_2004", "D": "2005_2006"}

_EXTENSIONS = (".csv", ".xpt", ".zip")  # Preferred first.
_PREPARED_DIR = "nhanes-prepared"
_PARTICIPANTS_FILE = "nhanes-participants.parquet"
_EPOCH_DIR = "epochs"
_INFO_FILE = "nhanes-prepare.json"
_FORMAT_VERSION = 2
_NONWEAR_METHODS = ("authors", "choi")

_EPOCH_MINUTES = 5
_WEAR_THRESHOLD = 0.2  # An epoch counts as worn if >= 20% of its minutes are.
_MIN_AGE = 18
_MAX_INTENSITY = 32767  # PAXINTEN hard edit in the NHANES codebook.
_SAS_ZERO = 1e-10  # pandas reads SAS transport zeros as ~5.4e-79.
_INCOME_MISSING = (77, 99)  # INDHHINC: refused, don't know.

# Choi et al. (2011): >= 90 minutes of zero counts, allowing interruptions
# of up to 2 minutes of nonzero counts that have 30 zero minutes on each side.
_CHOI_MIN_MINUTES = 90
_CHOI_SPIKE_MINUTES = 2
_CHOI_WINDOW_MINUTES = 30
# The MotionAge authors' variant also treats counts of 1-100 as "spikes" and
# allows at most 2 of them, never adjacent, in a non-wear period.
_AUTHORS_SPIKE_MAX = 100
_AUTHORS_SPIKE_TOLERANCE = 2

_PAXRAW_COLUMNS = [
    "SEQN",
    "PAXSTAT",
    "PAXCAL",
    "PAXDAY",
    "PAXN",
    "PAXHOUR",
    "PAXMINUT",
    "PAXINTEN",
]
_BP_COLUMNS = [f"BPX{kind}{k}" for kind in ("SY", "DI") for k in range(1, 5)]
_MORTALITY_COLSPECS = [(0, 14), (14, 15), (15, 16), (42, 45), (45, 48)]
_MORTALITY_NAMES = ["SEQN", "ELIGSTAT", "MORTSTAT", "PERMTH_INT", "PERMTH_EXM"]


class NHANESDataset(BaseDataset):
    """NHANES 2003-2006 physical activity monitor data with linked mortality.

    The National Health and Nutrition Examination Survey (NHANES) asked
    participants to wear a hip ActiGraph accelerometer for 7 days, recording
    an activity count every minute. The NCHS 2019 public-use linked mortality
    files follow the same participants for death through 2019. Both are
    public: https://wwwn.cdc.gov/nchs/nhanes/ and
    https://www.cdc.gov/nchs/data-linkage/mortality-public.htm

    Expected files, anywhere under ``root`` (file names are case-insensitive;
    ``<s>`` is ``C`` for 2003-2004 and ``D`` for 2005-2006; each table can be
    ``.xpt``, a ``.zip`` holding the ``.xpt``, or ``.csv``)::

        PAXRAW_<s>  DEMO_<s>  BMX_<s>  BPX_<s>
        NHANES_<years>_MORT_2019_PUBLIC.dat

    A cycle is used when its ``DEMO_<s>`` file is present. On first use,
    :meth:`prepare_metadata` writes the prepared files into
    ``root/nhanes-prepared/<nonwear>/`` (``<nonwear>-<max_participants>/`` for
    a subset): ``nhanes-participants.parquet``, one ``epochs/<SEQN>.parquet``
    per participant and ``nhanes-prepare.json``. Each setting gets its own
    folder, so the non-wear methods can be compared without overwriting each
    other. Later runs reuse a prepared folder, and need no raw files once it
    exists, so a prepared folder can be shared on its own.

    The participant table has one row per adult (age >= 18) who is eligible
    for mortality linkage and has accelerometer data. Its attributes are
    ``cycle``, ``age``, ``sex`` (1 = male, 2 = female), ``income`` (INDHHINC
    code; refused/don't know are missing), ``bmi``, ``waist``, ``sbp`` and
    ``dbp`` (means of up to four readings; a diastolic of 0 is treated as
    missing), ``eligstat``, ``mortstat``, ``permth_int``, ``permth_exm``,
    ``n_epochs``, ``n_wear_epochs`` and ``epoch_file``.

    Each epoch file has one row per 5-minute epoch: ``epoch``, ``day``
    (0 = Sunday ... 6 = Saturday), ``hour`` (0-23), ``minute`` (0-59, at the
    start of the epoch), ``intensity`` (mean counts per minute over the worn
    minutes, 0 when none were worn), ``wear_minutes`` (0-5) and ``wear_mask``
    (1 when at least 20% of the epoch's minutes were worn). Minutes from an
    unreliable (``PAXSTAT != 1``) or uncalibrated (``PAXCAL != 1``) monitor
    count as not worn.

    Non-wear methods (``nonwear``):

    * ``"authors"``: the MotionAge authors' implementation, adapted from
      https://github.com/jackie-jiaqi-yin/MotionAge (MIT License). Counts of
      1-100 are "spikes"; a non-wear period is >= 90 minutes of zeros and
      spikes with at most 2 non-adjacent spikes, each with 30 zero minutes on
      both sides. It flags less non-wear than the published rule and
      reproduces the paper's wear coverage.
    * ``"choi"``: Choi et al. (2011), Med Sci Sports Exerc 43(2):357-364:
      >= 90 minutes of zero counts, allowing interruptions of up to 2
      consecutive nonzero minutes with 30 zero minutes on both sides.

    Args:
        root: Directory holding the raw NHANES files. Must be writable, since
            the prepared files are written under it.
        dataset_name: Name of the dataset. Defaults to ``"nhanes"``.
        config_path: Path to the YAML config. Defaults to the bundled
            ``configs/nhanes.yaml``.
        nonwear: Non-wear detection method, ``"authors"`` (default) or
            ``"choi"``; see above.
        max_participants: If set, only prepare about this many participants,
            split evenly across the cycles found. Useful for fast development
            runs; ``None`` (the default) prepares everyone.
        cache_dir: Passed to :class:`~pyhealth.datasets.BaseDataset`.
        num_workers: Passed to :class:`~pyhealth.datasets.BaseDataset`.
        dev: Passed to :class:`~pyhealth.datasets.BaseDataset`.

    Raises:
        FileNotFoundError: If ``root`` does not exist, no cycle is found, or a
            cycle is missing one of its files.
        ValueError: If ``nonwear`` is unknown, ``max_participants`` is less
            than 1, a file is missing required columns, or the accelerometer
            file is not sorted.

    Examples:
        >>> import tempfile
        >>> import pandas as pd
        >>> from pyhealth.datasets.nhanes import NHANESDataset
        >>> from pyhealth.datasets.nhanes_synthetic import generate_synthetic_nhanes
        >>> root = tempfile.mkdtemp()
        >>> _ = generate_synthetic_nhanes(  # doctest: +SKIP
        ...     root, n_participants=10, n_days=2
        ... )
        >>> dataset = NHANESDataset(root=root)  # doctest: +SKIP
        >>> dataset.stats()  # doctest: +SKIP
        >>> patient = dataset.get_patient(  # doctest: +SKIP
        ...     dataset.unique_patient_ids[0]
        ... )
        >>> event = patient.get_events(event_type="nhanes")[0]  # doctest: +SKIP
        >>> epochs = pd.read_parquet(event.epoch_file)  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | None = None,
        nonwear: str = "authors",
        max_participants: int | None = None,
        cache_dir: str | None = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        if nonwear not in _NONWEAR_METHODS:
            raise ValueError(
                f"nonwear must be one of {_NONWEAR_METHODS}, got {nonwear!r}"
            )
        if max_participants is not None and max_participants < 1:
            raise ValueError(
                f"max_participants must be at least 1, got {max_participants}"
            )
        if not Path(root).is_dir():
            raise FileNotFoundError(f"NHANES root directory not found: {root}")
        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "nhanes.yaml")
        self.nonwear = nonwear
        self.max_participants = max_participants
        folder = (
            nonwear if max_participants is None else f"{nonwear}-{max_participants}"
        )
        self.prepared_dir = Path(root).resolve() / _PREPARED_DIR / folder

        if self._prepared_info() != self._expected_info():
            self.prepare_metadata(root)
        else:
            _relocate_epoch_files(self.prepared_dir)

        super().__init__(
            root=str(self.prepared_dir),
            tables=["nhanes"],
            dataset_name=dataset_name or "nhanes",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def _expected_info(self) -> dict:
        """Returns the settings that determine the prepared files."""
        return {
            "format_version": _FORMAT_VERSION,
            "nonwear": self.nonwear,
            "max_participants": self.max_participants,
            "epoch_minutes": _EPOCH_MINUTES,
            "wear_threshold": _WEAR_THRESHOLD,
            "min_age": _MIN_AGE,
        }

    def _prepared_info(self) -> dict | None:
        """Reads the settings of an earlier preparation, if one finished."""
        info_path = self.prepared_dir / _INFO_FILE
        if not (
            info_path.is_file() and (self.prepared_dir / _PARTICIPANTS_FILE).is_file()
        ):
            return None
        with open(info_path) as handle:
            return json.load(handle).get("settings")

    def prepare_metadata(self, root: str) -> None:
        """Builds the participant table and per-participant epoch files.

        Steps: find each cycle's files; merge DEMO, BMX, BPX and mortality on
        SEQN; keep adults eligible for mortality linkage; then stream the
        accelerometer file in chunks, detect non-wear for each participant
        and write their 5-minute epochs into :attr:`prepared_dir`.

        Args:
            root: Directory holding the raw NHANES files.

        Raises:
            FileNotFoundError: If no cycle is found or a cycle is incomplete.
            ValueError: If a file lacks required columns or the accelerometer
                file is not sorted by SEQN.
        """
        cycles = _find_cycle_files(Path(root).resolve())
        out_dir = self.prepared_dir
        epoch_dir = out_dir / _EPOCH_DIR
        epoch_dir.mkdir(parents=True, exist_ok=True)
        detector = _authors_nonwear if self.nonwear == "authors" else _choi_nonwear
        per_cycle = (
            None
            if self.max_participants is None
            else math.ceil(self.max_participants / len(cycles))
        )

        tables = []
        for suffix, files in cycles.items():
            participants = _read_participants(suffix, files)
            logger.info(
                f"NHANES {_CYCLES[suffix]}: {len(participants)} eligible adults"
            )
            summaries = _write_epoch_files(
                files["paxraw"],
                set(participants["SEQN"]),
                epoch_dir,
                per_cycle,
                detector,
            )
            summaries = pd.DataFrame(
                summaries, columns=["SEQN", "n_epochs", "n_wear_epochs"]
            )
            tables.append(participants.merge(summaries, on="SEQN", how="inner"))
            logger.info(
                f"NHANES {_CYCLES[suffix]}: {len(summaries)} with accelerometer data"
            )

        table = pd.concat(tables, ignore_index=True).sort_values("SEQN")
        if table.empty:
            raise ValueError("No eligible participants with accelerometer data.")
        table["epoch_file"] = [
            str(epoch_dir / f"{seqn}.parquet") for seqn in table["SEQN"]
        ]
        table = table.rename(columns=_OUTPUT_NAMES)[list(_OUTPUT_NAMES.values())]
        table.to_parquet(out_dir / _PARTICIPANTS_FILE, index=False)
        with open(out_dir / _INFO_FILE, "w") as handle:
            json.dump(
                {"settings": self._expected_info(), "participants": len(table)},
                handle,
                indent=2,
            )
        logger.info(f"Wrote {len(table)} participants to {out_dir}")


# Raw column -> participant table column, in output order.
_OUTPUT_NAMES = {
    "SEQN": "seqn",
    "cycle": "cycle",
    "RIDAGEYR": "age",
    "RIAGENDR": "sex",
    "INDHHINC": "income",
    "BMXBMI": "bmi",
    "BMXWAIST": "waist",
    "sbp": "sbp",
    "dbp": "dbp",
    "ELIGSTAT": "eligstat",
    "MORTSTAT": "mortstat",
    "PERMTH_INT": "permth_int",
    "PERMTH_EXM": "permth_exm",
    "n_epochs": "n_epochs",
    "n_wear_epochs": "n_wear_epochs",
    "epoch_file": "epoch_file",
}


def _relocate_epoch_files(prepared_dir: Path) -> None:
    """Rewrites ``epoch_file`` paths if the prepared folder was moved."""
    table_path = prepared_dir / _PARTICIPANTS_FILE
    table = pd.read_parquet(table_path)
    epoch_dir = prepared_dir / _EPOCH_DIR
    expected = [str(epoch_dir / f"{seqn}.parquet") for seqn in table["seqn"]]
    if table["epoch_file"].tolist() != expected:
        table["epoch_file"] = expected
        table.to_parquet(table_path, index=False)


def _find_cycle_files(root: Path) -> dict[str, dict[str, Path]]:
    """Locates each cycle's five input files under ``root``.

    Returns:
        Maps cycle suffix to ``{"paxraw", "demo", "bmx", "bpx", "mortality"}``
        file paths, for every cycle whose DEMO file is present.
    """
    by_name: dict[str, list[Path]] = {}
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = [d for d in subdirs if d != _PREPARED_DIR]
        for name in files:
            by_name.setdefault(name.lower(), []).append(Path(directory) / name)

    def find(names: list[str]) -> Path | None:
        # Returns the first name found, preferring earlier names in the list.
        # Copies in different folders (e.g. real and synthetic data both under
        # root) are ambiguous, so they raise instead of silently picking one.
        matches = [path for name in names for path in by_name.get(name, [])]
        if len({path.parent for path in matches}) > 1:
            raise ValueError(f"Found {names[0]} in more than one folder: {matches}")
        return matches[0] if matches else None

    cycles = {}
    for suffix, years in _CYCLES.items():
        files = {
            key: find([f"{key}_{suffix.lower()}{ext}" for ext in _EXTENSIONS])
            for key in ("paxraw", "demo", "bmx", "bpx")
        }
        if files["demo"] is None:
            continue
        # Prefer the newest linkage (e.g. MORT_2019 over MORT_2015).
        mortality = sorted(
            name
            for name in by_name
            if name.startswith(f"nhanes_{years}_mort_") and name.endswith(".dat")
        )
        files["mortality"] = find(mortality[-1:])
        missing = [key for key, path in files.items() if path is None]
        if missing:
            raise FileNotFoundError(f"NHANES {years} is missing {missing} under {root}")
        cycles[suffix] = files
    if not cycles:
        raise FileNotFoundError(
            f"No NHANES DEMO_C or DEMO_D file (.xpt, .zip or .csv) under {root}"
        )
    return cycles


def _clean_sas_zeros(frame: pd.DataFrame) -> pd.DataFrame:
    """Replaces the ~5.4e-79 values pandas reads for SAS transport zeros."""
    numeric = frame.select_dtypes("number").columns
    frame[numeric] = frame[numeric].mask(frame[numeric].abs() < _SAS_ZERO, 0.0)
    return frame


def _require_columns(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    """Raises ValueError naming any required columns missing from a file."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")


def _read_table(path: Path, columns: list[str]) -> pd.DataFrame:
    """Reads the given columns of a CSV or SAS transport (.xpt/.zip) file."""
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_sas(path, format="xport")
    _require_columns(frame, columns, path)
    return _clean_sas_zeros(frame[columns].copy())


def _read_mortality(path: Path) -> pd.DataFrame:
    """Parses the NCHS fixed-width public-use linked mortality file."""
    return pd.read_fwf(
        path,
        colspecs=_MORTALITY_COLSPECS,
        names=_MORTALITY_NAMES,
        na_values=["."],
    )


def _read_participants(suffix: str, files: dict[str, Path]) -> pd.DataFrame:
    """Merges one cycle's covariates and mortality into one row per adult."""
    demo = _read_table(files["demo"], ["SEQN", "RIDAGEYR", "RIAGENDR", "INDHHINC"])
    bmx = _read_table(files["bmx"], ["SEQN", "BMXBMI", "BMXWAIST"])
    bpx = _read_table(files["bpx"], ["SEQN", *_BP_COLUMNS])
    mortality = _read_mortality(files["mortality"])

    demo["INDHHINC"] = demo["INDHHINC"].where(~demo["INDHHINC"].isin(_INCOME_MISSING))
    systolic = bpx[[f"BPXSY{k}" for k in range(1, 5)]]
    diastolic = bpx[[f"BPXDI{k}" for k in range(1, 5)]]
    bpx["sbp"] = systolic.mean(axis=1)
    bpx["dbp"] = diastolic.where(diastolic > 0).mean(axis=1)

    frame = demo.dropna(subset=["SEQN"]).astype({"SEQN": "int64"})
    for other in (bmx, bpx[["SEQN", "sbp", "dbp"]], mortality):
        other = other.dropna(subset=["SEQN"]).astype({"SEQN": "int64"})
        frame = frame.merge(other, on="SEQN", how="left", validate="one_to_one")
    keep = (frame["ELIGSTAT"] == 1) & (frame["RIDAGEYR"] >= _MIN_AGE)
    frame = frame[keep].copy()
    frame["cycle"] = _CYCLES[suffix].replace("_", "-")
    return frame


def _activity_blocks(path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    """Yields accelerometer rows in blocks that never split a participant."""
    if path.suffix.lower() == ".csv":
        reader = pd.read_csv(path, chunksize=chunksize)
    else:
        reader = pd.read_sas(path, format="xport", chunksize=chunksize)
    carry = None
    last_seqn = None
    with reader:
        for chunk in reader:
            _require_columns(chunk, _PAXRAW_COLUMNS, path)
            chunk = _clean_sas_zeros(chunk[_PAXRAW_COLUMNS].copy())
            chunk["SEQN"] = chunk["SEQN"].astype("int64")
            seqn = chunk["SEQN"].to_numpy()
            if np.any(np.diff(seqn) < 0) or (
                last_seqn is not None and seqn[0] < last_seqn
            ):
                raise ValueError(f"{path} must be sorted by SEQN")
            last_seqn = seqn[-1]
            if carry is not None:
                chunk = pd.concat([carry, chunk], ignore_index=True)
            is_last = chunk["SEQN"].to_numpy() == last_seqn
            carry = chunk[is_last]
            if not is_last.all():
                yield chunk[~is_last]
    if carry is not None and not carry.empty:
        yield carry


def _write_epoch_files(
    paxraw: Path,
    eligible: set[int],
    epoch_dir: Path,
    limit: int | None,
    detector: Callable[[np.ndarray], np.ndarray],
    chunksize: int = 1_000_000,
) -> list[tuple[int, int, int]]:
    """Writes one epoch file per eligible participant in a PAXRAW file.

    ``detector`` maps a run of valid minute counts to a non-wear flag per
    minute (:func:`_authors_nonwear` or :func:`_choi_nonwear`).

    Returns:
        ``(SEQN, n_epochs, n_wear_epochs)`` for every participant written.
    """
    summaries: list[tuple[int, int, int]] = []
    for block in _activity_blocks(paxraw, chunksize):
        block = block[block["SEQN"].isin(eligible)]
        seqn = block["SEQN"].to_numpy()
        if len(seqn) == 0:
            continue
        starts = np.flatnonzero(np.r_[True, seqn[1:] != seqn[:-1]])
        bounds = [*starts.tolist(), len(seqn)]
        for start, end in itertools.pairwise(bounds):
            minutes = block.iloc[start:end]
            epochs = _participant_epochs(minutes, detector)
            if epochs is None:
                continue  # No valid minutes (unreliable/uncalibrated monitor).
            participant = int(seqn[start])
            epochs.to_parquet(epoch_dir / f"{participant}.parquet", index=False)
            summaries.append((participant, len(epochs), int(epochs["wear_mask"].sum())))
            if len(summaries) % 1000 == 0:
                logger.info(f"Prepared {len(summaries)} participants")
            if limit is not None and len(summaries) >= limit:
                return summaries
    return summaries


def _participant_epochs(
    minutes: pd.DataFrame, detector: Callable[[np.ndarray], np.ndarray]
) -> pd.DataFrame | None:
    """Turns one participant's minute rows into wear-aware 5-minute epochs.

    Minutes are placed on a regular 1-minute grid by PAXN, so gaps stay gaps.
    Non-wear detection runs separately on each run of valid minutes, so an
    invalid stretch can neither create nor extend a non-wear period.

    Returns:
        The epoch table, or ``None`` if the participant has no valid minutes.
    """
    minutes = minutes.sort_values("PAXN")
    paxn = minutes["PAXN"].to_numpy().astype(np.int64)
    if paxn.min() < 1 or np.any(np.diff(paxn) == 0):
        raise ValueError(f"Bad or duplicate PAXN for SEQN {minutes['SEQN'].iat[0]}")
    n_minutes = int(paxn.max())
    position = paxn - 1

    intensity = minutes["PAXINTEN"].to_numpy(dtype=float)
    valid = (
        (minutes["PAXSTAT"].to_numpy() == 1)
        & (minutes["PAXCAL"].to_numpy() == 1)
        & (intensity >= 0)
        & (intensity <= _MAX_INTENSITY)
    )
    if not valid.any():
        return None
    counts = np.zeros(n_minutes)
    is_valid = np.zeros(n_minutes, dtype=bool)
    counts[position] = np.where(valid, intensity, 0.0)
    is_valid[position] = valid

    first = minutes.iloc[0]
    clock = (
        int(first["PAXHOUR"]) * 60
        + int(first["PAXMINUT"])
        + np.arange(n_minutes)
        - position[0]
    )
    day = (int(first["PAXDAY"]) - 1 + clock // 1440) % 7

    wear = np.zeros(n_minutes, dtype=bool)
    for start, end in _true_runs(is_valid):
        wear[start:end] = ~detector(counts[start:end])

    n_epochs = math.ceil(n_minutes / _EPOCH_MINUTES)
    pad = n_epochs * _EPOCH_MINUTES - n_minutes

    def per_epoch(values: np.ndarray) -> np.ndarray:
        padded = np.concatenate([values, np.zeros(pad, dtype=values.dtype)])
        return padded.reshape(n_epochs, _EPOCH_MINUTES).sum(axis=1)

    present = per_epoch(np.ones(n_minutes))
    wear_minutes = per_epoch(wear.astype(float))
    worn_counts = per_epoch(counts * wear)
    mean_intensity = np.divide(
        worn_counts,
        wear_minutes,
        out=np.zeros(n_epochs),
        where=wear_minutes > 0,
    )
    first_minute = np.arange(n_epochs) * _EPOCH_MINUTES
    return pd.DataFrame(
        {
            "epoch": np.arange(n_epochs, dtype=np.int32),
            "day": day[first_minute].astype(np.int8),
            "hour": ((clock[first_minute] // 60) % 24).astype(np.int8),
            "minute": (clock[first_minute] % 60).astype(np.int8),
            "intensity": mean_intensity.astype(np.float32),
            "wear_minutes": wear_minutes.astype(np.int8),
            "wear_mask": (wear_minutes / present >= _WEAR_THRESHOLD).astype(np.int8),
        }
    )


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Returns ``(start, end)`` index pairs of consecutive True values."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def _choi_nonwear(counts: np.ndarray) -> np.ndarray:
    """Flags non-wear minutes with the Choi et al. (2011) algorithm.

    Non-wear is at least 90 consecutive minutes of zero counts. It may
    contain interruptions of at most 2 minutes of nonzero counts, as long as
    the 30 minutes before and after each interruption are all zero. The
    interruption minutes are flagged as non-wear too.

    Args:
        counts: Activity counts for consecutive valid minutes.

    Returns:
        Boolean array, True where the device was not worn.
    """
    n = counts.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    is_zero = counts == 0
    starts = np.flatnonzero(np.r_[True, is_zero[1:] != is_zero[:-1]])
    lengths = np.diff(np.r_[starts, n])
    zero_run = is_zero[starts]
    before = np.r_[0, lengths[:-1]]
    after = np.r_[lengths[1:], 0]
    # Runs alternate zero/nonzero, so a nonzero run's neighbours are zero runs.
    interruption = (
        ~zero_run
        & (lengths <= _CHOI_SPIKE_MINUTES)
        & (before >= _CHOI_WINDOW_MINUTES)
        & (after >= _CHOI_WINDOW_MINUTES)
    )
    candidate = zero_run | interruption
    block = np.cumsum(~candidate)
    block_minutes = np.bincount(
        block[candidate], weights=lengths[candidate], minlength=block.max() + 1
    )
    nonwear_run = candidate & (block_minutes[block] >= _CHOI_MIN_MINUTES)
    return np.repeat(nonwear_run, lengths)


# The "authors" non-wear rule below is adapted from MotionAge
# (https://github.com/jackie-jiaqi-yin/MotionAge, commit 0d52946,
# src/motionage/preprocessing/wear.py: detect_nonwear_choi and its helpers)
# with the parameters of its configs/data/nhanes.yaml. Its license:
#
# MIT License
#
# Copyright (c) 2026 Jackie Jiaqi Yin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def _authors_nonwear(counts: np.ndarray) -> np.ndarray:
    """Flags non-wear minutes the way the MotionAge authors' code does.

    The minutes are split at counts above 100. A piece of at least 90 minutes
    is non-wear if it has no counts of 1-100 ("spikes"), or at most 2
    non-adjacent spikes that each have 30 zero minutes on both sides.
    Otherwise its sub-pieces are searched for non-wear periods meeting the
    same rule. Unlike the published Choi rule, a zero run next to a stretch
    of small counts can be left as worn, so this flags less non-wear.

    Args:
        counts: Activity counts for consecutive valid minutes.

    Returns:
        Boolean array, True where the device was not worn.
    """
    n = counts.size
    nonwear = np.zeros(n, dtype=bool)
    if n == 0:
        return nonwear
    is_zero = counts == 0
    is_spike = (counts > 0) & (counts <= _AUTHORS_SPIKE_MAX)
    high = np.flatnonzero(counts > _AUTHORS_SPIKE_MAX).tolist()
    for seg_start, seg_end in zip([0, *(i + 1 for i in high)], [*high, n]):
        if seg_end - seg_start < _CHOI_MIN_MINUTES:
            continue  # Also skips the empty pieces between adjacent high counts.
        seg_zero = is_zero[seg_start:seg_end]
        spikes = np.flatnonzero(is_spike[seg_start:seg_end])
        if spikes.size == 0 or _authors_valid_period(seg_zero, spikes):
            nonwear[seg_start:seg_end] = True
        else:
            _authors_mark_periods(nonwear, seg_start, seg_zero, spikes)
    return nonwear


def _authors_valid_period(seg_zero: np.ndarray, spikes: np.ndarray) -> bool:
    """Checks the spike rule for a candidate non-wear period."""
    if spikes.size > _AUTHORS_SPIKE_TOLERANCE:
        return False
    if spikes.size > 1 and np.any(np.diff(spikes) == 1):
        return False
    window = _CHOI_WINDOW_MINUTES
    for spike in spikes.tolist():
        left = max(0, spike - window)
        right = min(seg_zero.size, spike + 1 + window)
        if spike - left < window or right - (spike + 1) < window:
            return False
        if not (seg_zero[left:spike].all() and seg_zero[spike + 1 : right].all()):
            return False
    return True


def _authors_mark_periods(
    nonwear: np.ndarray, seg_start: int, seg_zero: np.ndarray, spikes: np.ndarray
) -> None:
    """Marks valid non-wear periods inside a piece that failed as a whole."""
    seg_len = seg_zero.size
    spike_set = set(spikes.tolist())
    i = 0
    while i < seg_len:
        j = i
        local: list[int] = []
        while j < seg_len:
            if j in spike_set:
                if len(local) >= _AUTHORS_SPIKE_TOLERANCE:
                    break
                if local and j == local[-1] + 1:
                    break
                local.append(j)
            j += 1
        if j - i >= _CHOI_MIN_MINUTES and _authors_valid_period(
            seg_zero[i:j], np.asarray(local, dtype=int) - i
        ):
            nonwear[seg_start + i : seg_start + j] = True
            i = j
        else:
            i += 1

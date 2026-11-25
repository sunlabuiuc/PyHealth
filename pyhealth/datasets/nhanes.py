"""NHANES accelerometer and linked mortality dataset."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

CYCLES = {
    "2003-2004": "C",
    "2005-2006": "D",
}

MINUTES_PER_DAY = 1440
MINUTES_PER_WEEK = 10080
EPOCH_MINUTES = 5
EPOCHS_PER_WEEK = 2016

PROCESSED_DIR = "processed/pyhealth"
METADATA_FILE = "nhanes-metadata.parquet"
ACTIVITY_DIR = "activity"


def iter_participants(
    xpt_path: str | Path,
    chunksize: int = 50000,
) -> Iterator[pd.DataFrame]:
    """Yields participant-level accelerometer records from an XPT file.

    Args:
        xpt_path: Path to an NHANES accelerometer XPT file.
        chunksize: Number of rows read per chunk.

    Yields:
        Accelerometer records for one participant.

    Raises:
        FileNotFoundError: If the source file does not exist.
        ValueError: If participant identifiers are not sorted.
    """
    xpt_path = Path(xpt_path)

    if not xpt_path.is_file():
        raise FileNotFoundError(xpt_path)

    if chunksize <= 0:
        raise ValueError("chunksize must be positive")

    remaining = pd.DataFrame()

    with pd.read_sas(
        xpt_path,
        format="xport",
        chunksize=chunksize,
    ) as reader:
        for chunk in reader:
            chunk = chunk[["SEQN", "PAXN", "PAXINTEN"]].copy()

            if not remaining.empty:
                chunk = pd.concat(
                    [remaining, chunk],
                    ignore_index=True,
                )

            if chunk.empty:
                continue

            seqn = chunk["SEQN"].to_numpy()

            if not np.isfinite(seqn).all():
                raise ValueError("Invalid participant identifiers")

            if np.any(np.diff(seqn) < 0):
                raise ValueError("SEQN must be sorted")

            last_id = seqn[-1]

            complete = chunk[chunk["SEQN"] != last_id]

            remaining = chunk[chunk["SEQN"] == last_id].copy()

            for _, person in complete.groupby("SEQN", sort=False):
                yield person

    if not remaining.empty:
        yield remaining


def detect_nonwear_choi(
    counts: np.ndarray,
    min_duration: int = 90,
    spike_tolerance: int = 2,
    flank: int = 30,
) -> np.ndarray:
    """Detects nonwear periods using a Choi-style algorithm.

    Args:
        counts: One-dimensional minute-level activity counts.
        min_duration: Minimum nonwear duration in minutes.
        spike_tolerance: Maximum consecutive nonzero minutes allowed.
        flank: Required zero-count minutes around a tolerated spike.

    Returns:
        Boolean array where True indicates nonwear.

    Raises:
        ValueError: If activity counts or parameters are invalid.
    """
    counts = np.asarray(counts, dtype=float)

    if counts.ndim != 1:
        raise ValueError("counts must be one-dimensional")

    if not np.isfinite(counts).all():
        raise ValueError("counts must be finite")

    if (counts < 0).any():
        raise ValueError("counts must be nonnegative")

    if min_duration <= 0 or spike_tolerance < 0 or flank <= 0:
        raise ValueError("Invalid nonwear parameters")

    counts = np.where(np.abs(counts) < 1e-10, 0, counts)

    zero = counts == 0
    allowed = zero.copy()

    positive = ~zero

    edges = np.diff(np.r_[False, positive, False].astype(int))

    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    for start, end in zip(starts, ends):
        if end - start > spike_tolerance:
            continue

        left = start - flank
        right = end + flank

        if left < 0 or right > len(counts):
            continue

        if zero[left:start].all() and zero[end:right].all():
            allowed[start:end] = True

    edges = np.diff(np.r_[False, allowed, False].astype(int))

    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    nonwear = np.zeros(len(counts), dtype=bool)

    for start, end in zip(starts, ends):
        if end - start >= min_duration:
            nonwear[start:end] = True

    return nonwear


def create_5min_epochs(
    participant: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregates one week of activity into five-minute epochs.

    All epochs are retained without task-specific wear filtering.

    Args:
        participant: Minute-level records for one participant.

    Returns:
        DataFrame containing 2016 five-minute epochs.

    Raises:
        ValueError: If the participant records are incomplete or invalid.
    """
    df = participant.sort_values("PAXN")

    paxn = df["PAXN"].to_numpy(dtype=float)

    if len(df) != MINUTES_PER_WEEK:
        raise ValueError("Expected 10080 minute records")

    if not np.array_equal(paxn, np.arange(1, MINUTES_PER_WEEK + 1)):
        raise ValueError("PAXN must be consecutive from 1 to 10080")

    if df["SEQN"].nunique(dropna=False) != 1:
        raise ValueError("Expected records for one participant")

    counts = df["PAXINTEN"].to_numpy(dtype=float)

    counts = np.where(np.abs(counts) < 1e-10, 0, counts)

    wear = ~detect_nonwear_choi(counts)

    counts_5min = counts.reshape(-1, EPOCH_MINUTES)
    wear_5min = wear.reshape(-1, EPOCH_MINUTES)

    wear_minutes = wear_5min.sum(axis=1)

    raw_intensity = counts_5min.mean(axis=1)

    wear_counts = np.where(wear_5min, counts_5min, 0)

    intensity = np.divide(
        wear_counts.sum(axis=1),
        wear_minutes,
        out=np.zeros(EPOCHS_PER_WEEK, dtype=float),
        where=wear_minutes > 0,
    )

    epoch_index = np.arange(EPOCHS_PER_WEEK)
    minute_start = epoch_index * EPOCH_MINUTES

    return pd.DataFrame(
        {
            "epoch": (epoch_index + 1).astype(np.int16),
            "day": (minute_start // MINUTES_PER_DAY + 1).astype(np.int8),
            "hour": ((minute_start % MINUTES_PER_DAY) // 60).astype(np.int8),
            "raw_intensity": raw_intensity.astype(np.float32),
            "intensity": intensity.astype(np.float32),
            "wear_minutes": wear_minutes.astype(np.int8),
            "wear_fraction": (wear_minutes / EPOCH_MINUTES).astype(np.float32),
        }
    )


def prepare_activity(
    root: str | Path,
    cycle: str,
    max_participants: int | None = None,
) -> pd.DataFrame:
    """Prepares participant-level accelerometer files.

    Reads raw NHANES accelerometer records, detects nonwear,
    and saves all five-minute epochs for each participant.

    Args:
        root: Root directory containing raw NHANES files.
        cycle: NHANES survey cycle.
        max_participants: Optional limit for development testing.

    Returns:
        Metadata containing participant identifiers and
        relative paths to processed activity files.

    Raises:
        ValueError: If the survey cycle or participant limit is invalid.
        FileNotFoundError: If the raw activity file is missing.
    """
    if cycle not in CYCLES:
        raise ValueError(f"Unsupported cycle: {cycle}")

    if max_participants is not None and max_participants <= 0:
        raise ValueError("max_participants must be positive")

    root = Path(root)
    suffix = CYCLES[cycle].lower()

    xpt_path = root / cycle / f"paxraw_{suffix}.xpt"

    output_dir = root / PROCESSED_DIR / ACTIVITY_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    total = 0
    invalid = 0

    reader = iter_participants(xpt_path)

    try:
        for person in reader:
            total += 1
            seqn = int(person["SEQN"].iloc[0])

            try:
                epochs = create_5min_epochs(person)
            except ValueError as exc:
                invalid += 1
                logger.debug("Skipping SEQN %s: %s", seqn, exc)
            else:
                filename = f"{cycle}_{seqn}.parquet"
                output_path = output_dir / filename

                epochs.to_parquet(
                    output_path,
                    index=False,
                    compression="snappy",
                )

                metadata.append(
                    {
                        "patient_id": str(seqn),
                        "cycle": cycle,
                        "activity_file": output_path.relative_to(root).as_posix(),
                        "num_epochs": len(epochs),
                        "total_wear_minutes": int(epochs["wear_minutes"].sum()),
                    }
                )

            if total % 100 == 0:
                logger.info(
                    "%s: %d participants processed",
                    cycle,
                    total,
                )

            if max_participants is not None and total >= max_participants:
                break
    finally:
        reader.close()

    logger.info(
        "%s: total=%d, invalid=%d, retained=%d",
        cycle,
        total,
        invalid,
        len(metadata),
    )

    return pd.DataFrame(
        metadata,
        columns=[
            "patient_id",
            "cycle",
            "activity_file",
            "num_epochs",
            "total_wear_minutes",
        ],
    )


def prepare_clinical(
    root: str | Path,
    cycle: str,
) -> pd.DataFrame:
    """Loads demographic and clinical information from NHANES.

    Args:
        root: Root directory containing raw NHANES files.
        cycle: NHANES survey cycle.

    Returns:
        Participant-level demographic and clinical information.

    Raises:
        ValueError: If the survey cycle is unsupported.
        FileNotFoundError: If a required source file is missing.
    """
    if cycle not in CYCLES:
        raise ValueError(f"Unsupported cycle: {cycle}")

    root = Path(root)
    suffix = CYCLES[cycle]
    cycle_dir = root / cycle

    demo = pd.read_sas(
        cycle_dir / f"DEMO_{suffix}.XPT",
        format="xport",
    )

    bmx = pd.read_sas(
        cycle_dir / f"BMX_{suffix}.XPT",
        format="xport",
    )

    bpx = pd.read_sas(
        cycle_dir / f"BPX_{suffix}.XPT",
        format="xport",
    )

    demo = demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1"]].rename(
        columns={
            "RIDAGEYR": "age",
            "RIAGENDR": "gender",
            "RIDRETH1": "race_ethnicity",
        }
    )

    bmx = bmx[["SEQN", "BMXWT", "BMXHT", "BMXBMI"]].rename(
        columns={
            "BMXWT": "weight",
            "BMXHT": "height",
            "BMXBMI": "bmi",
        }
    )

    systolic_cols = [f"BPXSY{i}" for i in range(1, 5)]

    diastolic_cols = [f"BPXDI{i}" for i in range(1, 5)]

    bpx["systolic_bp"] = bpx[systolic_cols].mean(axis=1)
    bpx["diastolic_bp"] = bpx[diastolic_cols].mean(axis=1)

    bpx = bpx[["SEQN", "systolic_bp", "diastolic_bp"]]

    clinical = demo.merge(
        bmx,
        on="SEQN",
        how="left",
        validate="one_to_one",
    )

    clinical = clinical.merge(
        bpx,
        on="SEQN",
        how="left",
        validate="one_to_one",
    )

    clinical["patient_id"] = clinical["SEQN"].astype(int).astype(str)

    clinical["cycle"] = cycle

    return clinical.drop(columns="SEQN")


def prepare_mortality(
    root: str | Path,
    cycle: str,
) -> pd.DataFrame:
    """Loads NHANES 2019 public-use linked mortality records.

    Args:
        root: Root directory containing raw NHANES files.
        cycle: NHANES survey cycle.

    Returns:
        Participant-level mortality and follow-up information.

    Raises:
        ValueError: If the survey cycle is unsupported.
        FileNotFoundError: If the mortality file is missing.
    """
    if cycle not in CYCLES:
        raise ValueError(f"Unsupported cycle: {cycle}")

    root = Path(root)
    cycle_name = cycle.replace("-", "_")

    path = root / "mortality" / f"NHANES_{cycle_name}_MORT_2019_PUBLIC.dat"

    mortality = pd.read_fwf(
        path,
        colspecs=[
            (0, 6),
            (14, 15),
            (15, 16),
            (42, 45),
            (45, 48),
        ],
        names=[
            "patient_id",
            "eligstat",
            "mortstat",
            "permth_int",
            "permth_exm",
        ],
        header=None,
        na_values=["."],
    )

    mortality["patient_id"] = mortality["patient_id"].astype(int).astype(str)

    mortality["cycle"] = cycle

    return mortality


def merge_metadata(
    activity: pd.DataFrame,
    clinical: pd.DataFrame,
    mortality: pd.DataFrame,
) -> pd.DataFrame:
    """Combines activity, clinical, and mortality metadata.

    Participants with missing clinical or mortality information
    are retained for downstream task-specific filtering.

    Args:
        activity: Participant-level activity metadata.
        clinical: Demographic and clinical metadata.
        mortality: Linked mortality metadata.

    Returns:
        Combined participant-level metadata.
    """
    keys = ["cycle", "patient_id"]

    metadata = activity.merge(
        clinical,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    metadata = metadata.merge(
        mortality,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    if metadata["patient_id"].duplicated().any():
        raise ValueError("Duplicate NHANES patient identifiers")

    return metadata


def prepare_nhanes(
    root: str | Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Prepares NHANES accelerometer and linked mortality data.

    Supports the 2003-2004 and 2005-2006 survey cycles.
    Existing processed activity data are reused when available.

    Args:
        root: Root directory containing raw NHANES files.
        overwrite: Whether to regenerate processed activity data.

    Returns:
        Participant-level NHANES metadata.
    """
    root = Path(root)

    output_dir = root / PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    activity_path = output_dir / "activity-metadata.parquet"
    metadata_path = output_dir / METADATA_FILE

    if metadata_path.exists() and not overwrite:
        return pd.read_parquet(metadata_path)

    if activity_path.exists() and not overwrite:
        activity = pd.read_parquet(activity_path)
    else:
        activity = pd.concat(
            [prepare_activity(root, cycle) for cycle in CYCLES],
            ignore_index=True,
        )

        activity.to_parquet(
            activity_path,
            index=False,
        )

    clinical = pd.concat(
        [prepare_clinical(root, cycle) for cycle in CYCLES],
        ignore_index=True,
    )

    mortality = pd.concat(
        [prepare_mortality(root, cycle) for cycle in CYCLES],
        ignore_index=True,
    )

    metadata = merge_metadata(
        activity,
        clinical,
        mortality,
    )

    tmp_path = metadata_path.with_suffix(".tmp.parquet")

    metadata.to_parquet(tmp_path, index=False)
    tmp_path.replace(metadata_path)

    return metadata


class NHANESDataset(BaseDataset):
    """NHANES accelerometer and linked mortality dataset.

    Supports the 2003-2004 and 2005-2006 NHANES survey cycles.
    Each participant has accelerometer data, demographic and
    clinical information, and linked mortality records.

    Raw data are processed into participant-level activity files
    and a metadata table. Existing processed files are reused.

    Args:
        root: Root directory containing raw NHANES files.
        tables: Tables to load. Defaults to ["participants"].
        dataset_name: Optional dataset name.
        config_path: Optional path to a custom YAML configuration.
        **kwargs: Additional arguments passed to BaseDataset.

    Examples:
        >>> from pyhealth.datasets import NHANESDataset
        >>> dataset = NHANESDataset(root="data/nhanes")
        >>> dataset.stats()
    """

    def __init__(
        self,
        root: str,
        tables: list[str] | None = None,
        dataset_name: str | None = None,
        config_path: str | None = None,
        **kwargs,
    ) -> None:

        root_path = Path(root).resolve()

        if not root_path.is_dir():
            raise FileNotFoundError(root_path)

        metadata_path = root_path / PROCESSED_DIR / METADATA_FILE

        if not metadata_path.exists():
            logger.info("Preparing NHANES data")
            prepare_nhanes(root_path)

        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "nhanes.yaml")

        super().__init__(
            root=str(root_path),
            tables=tables or ["participants"],
            dataset_name=dataset_name or "nhanes",
            config_path=config_path,
            **kwargs,
        )

"""PhysioNet Sleep-Accel heart-rate dataset.

Sleep-Accel contains wrist Apple Watch recordings from 31 subjects during a
night of polysomnography. The dataset ships irregular heart rate in bpm,
three-axis acceleration, step counts and 30 s R&K stage labels. There is no
raw ECG or PPG waveform, so WatchSleepNet IBI features are derived from heart
rate with ``60 / bpm``.

Layout::

    <root>/
        heart_rate/<sid>_heartrate.txt   # "<seconds>,<bpm>"
        labels/<sid>_labeled_sleep.txt   # "<seconds> <stage>"
        motion/<sid>_acceleration.txt    # "<seconds>,<x>,<y>,<z>"
        steps/<sid>_steps.txt            # "<seconds>,<steps>"
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .base_dataset import BaseDataset
from .metadata_csv import (
    metadata_cache_dir,
    metadata_csv_exists,
    resolve_metadata_root,
    write_metadata_csv,
)

logger = logging.getLogger(__name__)

_CSV_NAME = "sleep_accel-pyhealth.csv"
_DATASET_KEY = "sleep_accel"

#: Raw R&K stage codes -> canonical five-class stages.
# Original 4 -> 3: https://github.com/WillKeWang/WatchSleepNet_public/blob/main/dataset_preparation/public_datasets/utils.py#L48
# Original 5 -> 4: https://github.com/WillKeWang/WatchSleepNet_public/blob/main/dataset_preparation/public_datasets/utils.py#L49
SLEEP_ACCEL_STAGE_MAP: dict[int, int] = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 3,
    5: 4,
    -1: -1,
}


def load_sleep_accel_stages(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Reads one Sleep-Accel scoring file.

    Args:
        path: Path to ``<sid>_labeled_sleep.txt``.

    Returns:
        A pair of arrays containing canonical stages and epoch start times
        in seconds relative to polysomnography start.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "1_labeled_sleep.txt"
        ...     _ = path.write_text("0 0\\n30 5\\n60 -1\\n")
        ...     stages, starts = load_sleep_accel_stages(path)
        >>> stages.tolist(), starts.tolist()
        ([0, 4, -1], [0.0, 30.0, 60.0])
    """
    data = np.loadtxt(path, dtype=np.float64, ndmin=2)
    starts = data[:, 0].astype(np.float64)
    codes = data[:, 1].astype(np.int64)
    stages = np.asarray(
        [SLEEP_ACCEL_STAGE_MAP.get(int(code), -1) for code in codes],
        dtype=np.int64,
    )
    return stages, starts


def load_sleep_accel_heart_rate(
    path: str | Path,
    t_start: float | None = None,
    t_end: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reads sorted, de-duplicated heart-rate samples.

    Args:
        path: Path to ``<sid>_heartrate.txt``.
        t_start: Optional inclusive lower time bound in seconds.
        t_end: Optional exclusive upper time bound in seconds.

    Returns:
        A pair of arrays containing times in seconds and heart rate in bpm.
        Non-finite times, non-positive rates and duplicate times are dropped.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "1_heartrate.txt"
        ...     _ = path.write_text("5,60\\n0,50\\n5,60\\n")
        ...     times, bpm = load_sleep_accel_heart_rate(path)
        >>> times.tolist(), bpm.tolist()
        ([0.0, 5.0], [50.0, 60.0])
    """
    data = np.loadtxt(path, dtype=np.float64, delimiter=",", ndmin=2)
    times, bpm = data[:, 0], data[:, 1]
    keep = np.isfinite(times) & np.isfinite(bpm) & (bpm > 0)
    times, bpm = times[keep], bpm[keep]
    order = np.argsort(times, kind="stable")
    times, bpm = times[order], bpm[order]
    times, first = np.unique(times, return_index=True)
    bpm = bpm[first]
    if t_start is not None:
        mask = times >= t_start
        times, bpm = times[mask], bpm[mask]
    if t_end is not None:
        mask = times < t_end
        times, bpm = times[mask], bpm[mask]
    return times, bpm


class SleepAccelDataset(BaseDataset):
    """PyHealth dataset for the PhysioNet Sleep-Accel cohort.

    ``root`` holds one subject directory worth of files in each of the four
    modality folders. The dataset writes ``sleep_accel-pyhealth.csv`` on first
    use and exposes absolute file paths as event attributes so tasks do not
    need the dataset object when reading a recording.

    Args:
        root: Directory containing ``heart_rate``, ``labels``, ``motion`` and
            ``steps`` subdirectories.
        dataset_name: Optional dataset name. Defaults to ``"sleep_accel"``.
        config_path: Optional YAML config. Defaults to the bundled config.
        cache_dir: Optional PyHealth sample cache directory.
        num_workers: Worker count for PyHealth task processing.
        dev: If True, limits the event table with PyHealth's dev cache key.
        refresh_cache: Rebuild the metadata CSV before fitting.

    Raises:
        ValueError: If the root is missing or no usable records are found.

    Examples:
        >>> from pyhealth.datasets import SleepAccelDataset
        >>> SleepAccelDataset.__name__
        'SleepAccelDataset'
    """

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | None = None,
        cache_dir: str | Path | None = None,
        num_workers: int = 1,
        dev: bool = False,
        refresh_cache: bool = False,
    ) -> None:
        root_path = Path(root).expanduser()
        if not root_path.is_dir():
            raise ValueError(f"sleep-accel root is not a directory: {root}")
        self.root = str(root_path.resolve())
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "sleep_accel.yaml"
        if refresh_cache:
            for path in (
                root_path / _CSV_NAME,
                metadata_cache_dir(root_path, _DATASET_KEY) / _CSV_NAME,
            ):
                if path.is_file():
                    path.unlink()
        if not metadata_csv_exists(root_path, _DATASET_KEY, _CSV_NAME):
            self.prepare_metadata(self.root)
        super().__init__(
            root=resolve_metadata_root(root_path, _DATASET_KEY, _CSV_NAME),
            tables=["sleep_accel"],
            dataset_name=dataset_name or "sleep_accel",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def prepare_metadata(self, root: str) -> str:
        """Writes one metadata row per complete Sleep-Accel subject.

        Args:
            root: Directory containing the four modality folders.

        Returns:
            The path of the written CSV.

        Raises:
            ValueError: If no record has all four required files.
        """
        root_path = Path(root).resolve()
        label_dir = root_path / "labels"
        if not label_dir.is_dir():
            raise ValueError(f"missing labels directory under {root}")
        rows: list[dict[str, str]] = []
        for label_file in sorted(label_dir.glob("*_labeled_sleep.txt")):
            subject_id = label_file.name.split("_", 1)[0]
            files = {
                "heart_rate_file": root_path
                / "heart_rate"
                / f"{subject_id}_heartrate.txt",
                "label_file": label_file,
                "motion_file": root_path
                / "motion"
                / f"{subject_id}_acceleration.txt",
                "steps_file": root_path / "steps" / f"{subject_id}_steps.txt",
            }
            missing = [name for name, path in files.items() if not path.is_file()]
            if missing:
                logger.warning(
                    "Skipping Sleep-Accel subject %s: missing %s",
                    subject_id,
                    ", ".join(missing),
                )
                continue
            try:
                stages, starts = load_sleep_accel_stages(files["label_file"])
                heart_times, _ = load_sleep_accel_heart_rate(
                    files["heart_rate_file"]
                )
            except (OSError, ValueError) as err:
                logger.warning(
                    "Skipping Sleep-Accel subject %s: %s", subject_id, err
                )
                continue
            if stages.size == 0 or np.any(np.diff(starts) < 0):
                logger.warning(
                    "Skipping Sleep-Accel subject %s: empty or unsorted labels",
                    subject_id,
                )
                continue
            if heart_times.size == 0:
                logger.warning(
                    "Skipping Sleep-Accel subject %s: no heart-rate samples",
                    subject_id,
                )
                continue
            rows.append(
                {
                    "patient_id": subject_id,
                    "record_name": subject_id,
                    **{key: str(path) for key, path in files.items()},
                }
            )
        if not rows:
            raise ValueError(f"no usable sleep-accel records found under {root}")
        frame = pd.DataFrame(rows)
        path = write_metadata_csv(frame, root_path, _DATASET_KEY, _CSV_NAME)
        return str(path)

    @property
    def default_task(self):
        """Returns a fresh heart-rate-derived IBI staging task."""
        from pyhealth.tasks.sleep_staging_ibi import SleepStagingIBISleepAccel

        return SleepStagingIBISleepAccel()

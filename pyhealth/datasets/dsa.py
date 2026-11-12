"""Daily and Sports Activities (DSA) dataset.

Authors: Joshua Gundersen (jgundersen3), Zachary Gundersen (zgundersen3)
Paper: Zhang et al. (2024), "Daily Physical Activity Monitoring: Adaptive
    Learning from Multi-source Motion Sensor Data", CHIL 2024.
    https://proceedings.mlr.press/v248/zhang24a.html
Description: Loads the DSA recordings as one event per 5-second segment and
    provides helpers that split each segment into its five synchronized
    sensor streams.

Dataset: Barshan, B. and Altun, K. (2010). Daily and Sports Activities.
UCI Machine Learning Repository. https://doi.org/10.24432/C5C59F

Used by Zhang et al. (2024), "Daily Physical Activity Monitoring: Adaptive
Learning from Multi-source Motion Sensor Data" (CHIL 2024).
"""

import logging
import os
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)

#: The five Xsens units, in the column order used by the raw files.
SENSOR_NAMES = ("torso", "right_arm", "left_arm", "right_leg", "left_leg")
#: Number of activities (folders a01 ... a19).
NUM_ACTIVITIES = 19
#: Samples per 5-second segment (25 Hz).
SEGMENT_LENGTH = 125
#: Channels per unit: 3 accelerometer, 3 gyroscope, 3 magnetometer axes.
CHANNELS_PER_SENSOR = 9

_ACTIVITY_DIR = re.compile(r"^a(\d{2})$")
_SUBJECT_DIR = re.compile(r"^p(\d+)$")
_SEGMENT_FILE = re.compile(r"^s(\d{2})\.txt$")


def build_dsa_metadata(root: str) -> pd.DataFrame:
    """Scans a raw DSA directory and lists every segment file.

    The raw layout is ``root/aXX/pY/sZZ.txt`` (activity, subject, segment).

    Args:
        root: Directory that contains the ``a01`` ... ``a19`` folders.

    Returns:
        A DataFrame with columns ``subject`` (e.g. ``"p1"``), ``activity``
        (1-19), ``segment`` (1-60) and ``signal_file`` (absolute path).

    Raises:
        FileNotFoundError: If ``root`` is not a directory.
        ValueError: If no segment files are found under ``root``.

    Examples:
        >>> meta = build_dsa_metadata("/data/dsa")  # doctest: +SKIP
        >>> list(meta.columns)  # doctest: +SKIP
        ['subject', 'activity', 'segment', 'signal_file']
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"DSA root directory not found: {root}")

    rows = []
    for act_name in sorted(os.listdir(root)):
        act_match = _ACTIVITY_DIR.match(act_name)
        act_path = os.path.join(root, act_name)
        if not act_match or not os.path.isdir(act_path):
            continue
        for sub_name in sorted(os.listdir(act_path)):
            sub_match = _SUBJECT_DIR.match(sub_name)
            sub_path = os.path.join(act_path, sub_name)
            if not sub_match or not os.path.isdir(sub_path):
                continue
            for seg_name in sorted(os.listdir(sub_path)):
                seg_match = _SEGMENT_FILE.match(seg_name)
                if not seg_match:
                    continue
                rows.append(
                    {
                        "subject": f"p{int(sub_match.group(1))}",
                        "activity": int(act_match.group(1)),
                        "segment": int(seg_match.group(1)),
                        "signal_file": os.path.join(sub_path, seg_name),
                    }
                )

    if not rows:
        raise ValueError(f"No DSA segment files (aXX/pY/sZZ.txt) found in {root}")
    return pd.DataFrame(rows)


def load_dsa_segment(path: str) -> np.ndarray:
    """Loads one raw 5-second segment.

    Args:
        path: Path to a ``sZZ.txt`` file (125 rows, 45 comma-separated columns).

    Returns:
        A float32 array of shape ``(125, 45)``.

    Raises:
        ValueError: If the file does not have the expected shape.

    Examples:
        >>> x = load_dsa_segment("/data/dsa/a01/p1/s01.txt")  # doctest: +SKIP
        >>> x.shape  # doctest: +SKIP
        (125, 45)
    """
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    expected = (SEGMENT_LENGTH, len(SENSOR_NAMES) * CHANNELS_PER_SENSOR)
    if data.shape != expected:
        raise ValueError(f"{path} has shape {data.shape}, expected {expected}")
    return data


def split_sensors(segment: np.ndarray) -> np.ndarray:
    """Splits a raw segment into its five synchronized sensor streams.

    Row ``i`` of the output is the same time window for every sensor, so
    source/target pairs stay aligned.

    Args:
        segment: Array of shape ``(125, 45)`` from :func:`load_dsa_segment`.

    Returns:
        An array of shape ``(5, 125, 9)`` ordered as :data:`SENSOR_NAMES`.

    Examples:
        >>> import numpy as np
        >>> split_sensors(np.zeros((125, 45), dtype=np.float32)).shape
        (5, 125, 9)
    """
    n_sensors = len(SENSOR_NAMES)
    stacked = segment.reshape(segment.shape[0], n_sensors, CHANNELS_PER_SENSOR)
    return stacked.transpose(1, 0, 2)


def compute_channel_ranges(
    signal_files: Iterable[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Finds the minimum and maximum of every sensor channel.

    The paper rescales each time-series dimension to [-1, 1]; these ranges
    are what that rescaling needs.

    Args:
        signal_files: Paths of the segment files to scan.

    Returns:
        ``(mins, maxs)``, each of shape ``(5, 9)`` (sensor, channel).

    Raises:
        ValueError: If no files are given.

    Examples:
        >>> mins, maxs = compute_channel_ranges(
        ...     ["/data/dsa/a01/p1/s01.txt"]
        ... )  # doctest: +SKIP
        >>> mins.shape  # doctest: +SKIP
        (5, 9)
    """
    mins = np.full((len(SENSOR_NAMES), CHANNELS_PER_SENSOR), np.inf)
    maxs = np.full((len(SENSOR_NAMES), CHANNELS_PER_SENSOR), -np.inf)
    seen = 0
    for path in signal_files:
        sensors = split_sensors(load_dsa_segment(path))
        mins = np.minimum(mins, sensors.min(axis=1))
        maxs = np.maximum(maxs, sensors.max(axis=1))
        seen += 1
    if seen == 0:
        raise ValueError("signal_files is empty")
    return mins, maxs


class DSADataset(BaseDataset):
    """Daily and Sports Activities (DSA) dataset.

    19 daily and sports activities performed by 8 subjects, recorded by five
    Xsens units (torso, right arm, left arm, right leg, left leg), each with
    9 channels at 25 Hz. Each activity is split into 60 five-second segments
    (125 samples). Data is available at
    https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities

    Expected layout under ``root``: ``a01/p1/s01.txt`` ... ``a19/p8/s60.txt``.
    On first use a ``dsa-pyhealth.csv`` metadata file is written to ``root``.
    Each subject is a patient and each segment is one event with the
    attributes ``activity``, ``segment`` and ``signal_file``.

    Args:
        root: Root directory of the raw data.
        dataset_name: Optional name of the dataset. Defaults to ``"dsa"``.
        config_path: Optional path to a YAML config. Defaults to the bundled
            ``configs/dsa.yaml``.
        cache_dir: Optional directory for PyHealth's processed cache. Defaults
            to PyHealth's user cache directory.
        num_workers: Number of worker processes used when processing data.
        dev: Whether to run in dev mode (a small subset of patients).

    Examples:
        >>> from pyhealth.datasets import DSADataset
        >>> dataset = DSADataset(root="/data/dsa")  # doctest: +SKIP
        >>> dataset.stats()  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | None = None,
        cache_dir: str | Path | None = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "dsa.yaml"
            )

        metadata_path = os.path.join(root, "dsa-pyhealth.csv")
        if not os.path.exists(metadata_path):
            self.prepare_metadata(root)

        super().__init__(
            root=root,
            tables=["dsa"],
            dataset_name=dataset_name or "dsa",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    @staticmethod
    def prepare_metadata(root: str) -> None:
        """Writes ``dsa-pyhealth.csv`` listing every segment under ``root``.

        Args:
            root: Root directory of the raw data.

        Examples:
            >>> DSADataset.prepare_metadata("/data/dsa")  # doctest: +SKIP
        """
        metadata = build_dsa_metadata(root)
        metadata.to_csv(os.path.join(root, "dsa-pyhealth.csv"), index=False)

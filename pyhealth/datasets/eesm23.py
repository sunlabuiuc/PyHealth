"""Ear-EEG Sleep Monitoring 2023 (EESM23) dataset.

Author: Jathurshan Pradeepkumar (jp65@illinois.edu), UIUC
Description: Loads OpenNeuro ds005178 (BIDS ear-EEG sleep recordings).
    One metadata row is written per ear-EEG night. Sleep staging lives in
    ``SleepStagingEarEEG`` and follows the TFM-Tokenizer preprocessing for
    EESM23 (30-second epochs, four in-ear channels RB/RT/LB/LT).

References:
    - Mikkelsen, K. B. et al. Ear-EEG sleep monitoring data sets.
      Scientific Data 12, 2025.
      https://www.nature.com/articles/s41597-025-04579-8
    - Rezai Tabar, Y. et al. Ear-EEG Sleep Monitoring 2023 (EESM23).
      OpenNeuro ds005178. https://doi.org/10.18112/openneuro.ds005178.v1.0.0
    - Tabar, Y. R. et al. At-home sleep monitoring using generic ear-EEG.
      Frontiers in Neuroscience 17 (2023).
    - Pradeepkumar, J. et al. Tokenizing Single-Channel EEG with
      Time-Frequency Motif Learning. ICLR 2026.
      https://github.com/Jathurshan0330/TFM-Tokenizer
      (``datasets_processing/EarEEG/process.py``, ``EarEEGloader``)
"""

import hashlib
import logging
from pathlib import Path

import pandas as pd

from pyhealth.tasks import SleepStagingEarEEG

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


def _metadata_cache_dir(root: str | Path) -> Path:
    """Return the home-cache directory used when ``root`` is not writable.

    The directory is keyed by a hash of the resolved root, so two dataset
    roots never share cached metadata or each other's ``signal_file`` paths.

    Args:
        root: Dataset root directory.

    Returns:
        ``~/.cache/pyhealth/eesm23/<hash of root>``.
    """
    resolved = str(Path(root).expanduser().resolve())
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    return Path.home() / ".cache" / "pyhealth" / "eesm23" / digest


def _scoring_path(signal_file: Path) -> Path:
    """Return the manual scoring TSV path for one ear-EEG night.

    Args:
        signal_file: Path to ``*_acq-earEEG_eeg.set``.

    Returns:
        Path to ``*_acq-scoring_events.tsv`` in the same folder.
    """
    name = signal_file.name.replace("_acq-earEEG_eeg.set", "_acq-scoring_events.tsv")
    return signal_file.parent / name


def _failures_path(signal_file: Path) -> Path:
    """Return the recording-failures TSV path for one ear-EEG night.

    Args:
        signal_file: Path to ``*_acq-earEEG_eeg.set``.

    Returns:
        Path to ``*_acq-earEEGRecordingFailures_events.tsv``.
    """
    name = signal_file.name.replace(
        "_acq-earEEG_eeg.set", "_acq-earEEGRecordingFailures_events.tsv"
    )
    return signal_file.parent / name


def _row_for_signal_file(signal_file: Path) -> dict:
    """Build one metadata row from a BIDS ear-EEG ``.set`` path.

    Args:
        signal_file: Resolved path to an ear-EEG recording.

    Returns:
        Row dict with ``patient_id``, ``session_id``, file paths and
        ``has_scoring`` (``"1"`` or ``"0"``).
    """
    parts = signal_file.parts
    patient_id = next(p for p in parts if p.startswith("sub-"))
    session_id = next(p for p in parts if p.startswith("ses-"))
    scoring = _scoring_path(signal_file)
    failures = _failures_path(signal_file)
    has_scoring = scoring.is_file()
    return {
        "patient_id": patient_id,
        "session_id": session_id,
        "signal_file": str(signal_file.resolve()),
        "scoring_file": str(scoring.resolve()) if has_scoring else "",
        "recording_failures_file": str(failures.resolve())
        if failures.is_file()
        else "",
        "has_scoring": "1" if has_scoring else "0",
    }


class EESM23Dataset(BaseDataset):
    """Ear-EEG Sleep Monitoring 2023 (OpenNeuro ds005178).

    The corpus holds BIDS-formatted ear-EEG from ten healthy subjects.
    Each subject has up to twelve home nights; manual sleep scoring (30 s
    epochs) is available only for the first two sessions (partial PSG
    nights). Ear electrodes are RB, RT, LB and LT (average-referenced
    in-ear dry contacts), not a 10-20 scalp montage.

    ``prepare_metadata`` writes ``eesm23-pyhealth.csv`` (one row per
    ear-EEG night) next to the BIDS root, or under
    ``~/.cache/pyhealth/eesm23/<hash of root>`` when the directory is not
    writable. The default task is
    :class:`~pyhealth.tasks.SleepStagingEarEEG`.

    The TFM-Tokenizer subject split is available as
    :meth:`default_subject_split`: ``sub-001``-``sub-006`` train,
    ``sub-007`` and ``sub-008`` validation, ``sub-009`` and ``sub-010``
    test.

    Args:
        root: BIDS root (for example the ``ds005178-download`` folder).
        dataset_name: Name stored on the dataset. Defaults to ``"eesm23"``.
        config_path: YAML config. Defaults to ``configs/eesm23.yaml``.
        **kwargs: Forwarded to :class:`~pyhealth.datasets.BaseDataset`
            (``dev``, ``num_workers``, ``cache_dir``).

    Examples:
        >>> from pyhealth.datasets import EESM23Dataset
        >>> dataset = EESM23Dataset(
        ...     root="/srv/local/data/jp65/earEEG/ds005178-download",
        ... )
        >>> dataset.stats()
        >>> train, val, test = dataset.set_task(
        ...     split=dataset.default_subject_split(),
        ... )
    """

    TFM_TRAIN_SUBJECTS: tuple[str, ...] = tuple(f"sub-{i:03d}" for i in range(1, 7))
    TFM_VAL_SUBJECTS: tuple[str, ...] = ("sub-007", "sub-008")
    TFM_TEST_SUBJECTS: tuple[str, ...] = ("sub-009", "sub-010")

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | None = None,
        **kwargs,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "eesm23.yaml"

        self.root = root
        self.prepare_metadata()

        root_path = Path(root)
        cache_dir = _metadata_cache_dir(root_path)
        shared_csv = root_path / "eesm23-pyhealth.csv"
        cache_csv = cache_dir / "eesm23-pyhealth.csv"
        if not shared_csv.exists() and cache_csv.exists():
            logger.info("Using cached metadata from %s", cache_dir)
            root = str(cache_dir)

        super().__init__(
            root=root,
            tables=["recordings"],
            dataset_name=dataset_name or "eesm23",
            config_path=config_path,
            **kwargs,
        )

    def prepare_metadata(self) -> None:
        """Write ``eesm23-pyhealth.csv`` if it is not already on disk.

        The CSV is one row per ``*_acq-earEEG_eeg.set`` file with paths to
        the recording, optional scoring TSV and optional failures TSV.
        An existing CSV, in ``root`` or in the cache, is left untouched.

        Returns:
            None. The CSV path is either ``<root>/eesm23-pyhealth.csv`` or
            ``~/.cache/pyhealth/eesm23/<hash of root>/eesm23-pyhealth.csv``.
        """
        root = Path(self.root)
        cache_dir = _metadata_cache_dir(root)
        shared_csv = root / "eesm23-pyhealth.csv"
        cache_csv = cache_dir / "eesm23-pyhealth.csv"
        if shared_csv.exists() or cache_csv.exists():
            return

        signal_files = sorted(root.glob("sub-*/ses-*/eeg/*_acq-earEEG_eeg.set"))
        rows = [_row_for_signal_file(path) for path in signal_files]
        if not rows:
            logger.warning("No EESM23 ear-EEG recordings found under %s", root)
            return

        df = pd.DataFrame(rows)
        df.sort_values(["patient_id", "session_id"], inplace=True, na_position="last")
        df.reset_index(drop=True, inplace=True)

        try:
            shared_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(shared_csv, index=False)
            logger.info("Wrote metadata to %s", shared_csv)
        except (PermissionError, OSError):
            cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_csv, index=False)
            logger.info("Wrote metadata to cache: %s", cache_csv)

    @classmethod
    def default_subject_split(cls):
        """Return the TFM-Tokenizer subject split for EESM23.

        Train is ``sub-001``-``sub-006``, validation is ``sub-007`` and
        ``sub-008``, and test is ``sub-009`` and ``sub-010``.

        Returns:
            A :class:`~pyhealth.datasets.SubjectSplit` with those three parts.

        Examples:
            >>> from pyhealth.datasets import EESM23Dataset
            >>> split = EESM23Dataset.default_subject_split()
            >>> [len(part) for part in split.subjects]
            [6, 2, 2]
        """
        from pyhealth.datasets.splitter import SubjectSplit

        return SubjectSplit(
            (
                cls.TFM_TRAIN_SUBJECTS,
                cls.TFM_VAL_SUBJECTS,
                cls.TFM_TEST_SUBJECTS,
            )
        )

    @property
    def default_task(self) -> SleepStagingEarEEG:
        """Return the default ear-EEG sleep staging task.

        Returns:
            A :class:`~pyhealth.tasks.SleepStagingEarEEG` with its default
            epoching and normalization.
        """
        return SleepStagingEarEEG()

"""AHEPA resting-state EEG dataset of dementia (OpenNeuro ds004504).

Author: Axel Coiffard-Donaty

Eyes-closed resting-state EEG of 88 subjects recorded at the AHEPA General
Hospital of Thessaloniki: 36 with Alzheimer's disease, 23 with frontotemporal
dementia and 29 healthy controls. Each subject has one recording of 19 scalp
channels of the 10-20 system at 500 Hz, about 5 to 21 minutes long. The
dataset follows BIDS and ships two versions of every recording: the
unprocessed one in ``sub-0XX/`` and an artifact-cleaned one in
``derivatives/sub-0XX/`` (0.5-45 Hz band-pass, re-referencing to A1-A2,
Artifact Subspace Reconstruction, and removal of the independent components
that ICLabel classifies as eye or jaw artifacts).

References:
    - Miltiadous, A. et al. A Dataset of Scalp EEG Recordings of Alzheimer's
      Disease, Frontotemporal Dementia and Healthy Subjects from Routine
      EEG. Data 8(6), 95, 2023. https://doi.org/10.3390/data8060095
    - Miltiadous, A. et al. The AHEPA EEG benchmark: setting the standard
      for machine learning in dementia diagnosis, a scoping review.
      Cognitive Neurodynamics 20(1), 2026.
      https://doi.org/10.1007/s11571-026-10464-w
    - Dataset: https://doi.org/10.18112/openneuro.ds004504.v1.0.9
"""

import logging
from pathlib import Path

import pandas as pd

from pyhealth.tasks import DementiaClassificationAHEPA

from .base_dataset import BaseDataset
from .metadata_csv import (
    metadata_csv_exists,
    resolve_metadata_root,
    write_metadata_csv,
)

logger = logging.getLogger(__name__)

_NAME = "ahepa_dementia"
_CSV_NAME = "ahepa_dementia-pyhealth.csv"
_RECORDING = "{subject}_task-eyesclosed_eeg.set"

_REQUIRED_COLUMNS: tuple[str, ...] = ("participant_id", "Group")

# Columns of participants.tsv and their names in the metadata.
_PARTICIPANT_COLUMNS: dict[str, str] = {
    "Group": "group",
    "Gender": "gender",
    "Age": "age",
    "MMSE": "mmse",
}


def _read_participants(path: Path) -> pd.DataFrame:
    """Read and check ``participants.tsv``.

    Args:
        path: Path to the tab-separated participants file.

    Returns:
        The participants, one row per id, with ``participant_id`` stripped
        and the ``Group`` codes stripped and in upper case. Rows without an
        id are dropped, and so are the later rows of a repeated id, each
        with a warning.

    Raises:
        ValueError: If the file is empty or lacks the ``participant_id`` or
            ``Group`` column.
    """
    try:
        participants = pd.read_csv(path, sep="\t", dtype=str)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"{path} is empty.") from error
    missing = [c for c in _REQUIRED_COLUMNS if c not in participants.columns]
    if missing:
        raise ValueError(
            f"{path} has no {missing} column(s); its columns are "
            f"{list(participants.columns)}. It has to be the tab-separated "
            "participants.tsv of the dataset."
        )
    participants["participant_id"] = participants["participant_id"].str.strip()
    participants["Group"] = participants["Group"].str.strip().str.upper()
    has_id = participants["participant_id"].notna() & (
        participants["participant_id"] != ""
    )
    if not has_id.all():
        logger.warning(
            "%s has %d row(s) without a participant_id; they are left out.",
            path,
            int((~has_id).sum()),
        )
    participants = participants[has_id]
    repeated = participants["participant_id"].duplicated()
    if repeated.any():
        logger.warning(
            "%s lists %s more than once; the first row of each is used.",
            path,
            sorted(participants.loc[repeated, "participant_id"].unique()),
        )
        participants = participants[~repeated]
    return participants.reset_index(drop=True)


def _recording(folder: Path, subject: str) -> str:
    """Return the path of a subject's recording under ``folder``.

    Args:
        folder: The BIDS root (unprocessed data) or its ``derivatives``
            folder (artifact-cleaned data).
        subject: Participant id, for example ``"sub-001"``.

    Returns:
        The absolute path of the ``.set`` file, or an empty string when it
        is missing.
    """
    path = folder / subject / "eeg" / _RECORDING.format(subject=subject)
    return str(path.absolute()) if path.is_file() else ""


class AHEPADementiaDataset(BaseDataset):
    """AHEPA resting-state EEG dataset of dementia (OpenNeuro ds004504).

    88 subjects with one eyes-closed recording each (19 channels, 500 Hz):
    36 with Alzheimer's disease (group ``A``), 23 with frontotemporal
    dementia (``F``) and 29 healthy controls (``C``). ``patient_id`` is the
    BIDS participant id, for example ``"sub-001"``.

    ``prepare_metadata`` writes ``ahepa_dementia-pyhealth.csv`` (one row per
    subject of ``participants.tsv`` with at least one recording under
    ``root``) in ``root``, or under
    ``~/.cache/pyhealth/ahepa_dementia/<hash of root>`` when ``root`` is not
    writable. Each row holds the artifact-cleaned recording
    (``signal_file``), the unprocessed one (``raw_signal_file``), the group,
    the gender, the age and the Mini-Mental State Examination score
    (``mmse``). The default task is
    :class:`~pyhealth.tasks.DementiaClassificationAHEPA`.

    The MMSE score is a cognitive test used to assess dementia, and every
    control of this dataset scores 30 while every patient scores 27 or
    less: it separates controls from patients by itself and should not be
    a model input.

    Getting the data:
        - Source: https://openneuro.org/datasets/ds004504 (version 1.0.9)
        - Access: open, CC0 license.
        - Download: ``aws s3 sync --no-sign-request
          s3://openneuro.org/ds004504 /path/to/ds004504`` or the other
          options of that page, 5.8 GB in total, about half for each
          version.
        - ``root``: the downloaded folder, the one holding
          ``participants.tsv``.
        - The ``.set`` files are read with MNE; no extra package is needed.
        - Disk for the task cache: with the default
          :class:`~pyhealth.tasks.DementiaClassificationAHEPA`, a window
          takes 152 kB (19 channels by 2000 float32 samples), and the 65
          subjects of Alzheimer's disease against controls give about
          26,500 windows, so about 4 GB of processed samples. ``set_task``
          also caches the task output, which roughly doubles that.

    Args:
        root: BIDS root of the dataset, holding ``participants.tsv``, the
            ``sub-0XX`` folders and ``derivatives``.
        dataset_name: Name stored on the dataset. Defaults to
            ``"ahepa_dementia"``.
        config_path: YAML config. Defaults to
            ``configs/ahepa_dementia.yaml``.
        **kwargs: Forwarded to :class:`~pyhealth.datasets.BaseDataset`
            (``dev``, ``num_workers``, ``cache_dir``).

    Examples:
        >>> from pyhealth.datasets import AHEPADementiaDataset
        >>> dataset = AHEPADementiaDataset(root="/path/to/ds004504")
        >>> dataset.stats()
        >>> samples = dataset.set_task()
        >>> samples[0]["signal"].shape
        torch.Size([19, 2000])
    """

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | None = None,
        **kwargs,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / f"{_NAME}.yaml"

        self.root = root
        self.prepare_metadata()

        super().__init__(
            root=resolve_metadata_root(root, _NAME, _CSV_NAME),
            tables=["recordings"],
            dataset_name=dataset_name or _NAME,
            config_path=config_path,
            **kwargs,
        )

    def prepare_metadata(self) -> None:
        """Write ``ahepa_dementia-pyhealth.csv`` if it is not on disk.

        The CSV has one row per subject of ``participants.tsv``, in that
        order, with ``patient_id``, ``signal_file`` (artifact-cleaned
        recording), ``raw_signal_file`` (unprocessed recording), ``group``,
        ``gender``, ``age`` and ``mmse``. A missing recording is left empty,
        and subjects with neither recording, for example after a partial
        download, are left out with a warning. An existing CSV, in ``root``
        or in the cache, is left untouched: delete it to index recordings
        downloaded later.

        Returns:
            None. The CSV path is ``<root>/ahepa_dementia-pyhealth.csv`` or
            the same file name under
            ``~/.cache/pyhealth/ahepa_dementia/<hash of root>``.

        Raises:
            ValueError: If ``participants.tsv`` is empty or lacks the
                ``participant_id`` or ``Group`` column.
        """
        root = Path(self.root)
        if metadata_csv_exists(root, _NAME, _CSV_NAME):
            return

        participants_file = root / "participants.tsv"
        if not participants_file.is_file():
            logger.warning("No participants.tsv found under %s", root)
            return
        participants = _read_participants(participants_file)

        rows = []
        missing = []
        for _, participant in participants.iterrows():
            subject = participant["participant_id"]
            signal_file = _recording(root / "derivatives", subject)
            raw_signal_file = _recording(root, subject)
            if not signal_file and not raw_signal_file:
                missing.append(subject)
                continue
            row = {
                "patient_id": subject,
                "signal_file": signal_file,
                "raw_signal_file": raw_signal_file,
            }
            for source, column in _PARTICIPANT_COLUMNS.items():
                value = participant.get(source)
                row[column] = "" if pd.isna(value) else str(value).strip()
            rows.append(row)
        if missing:
            logger.warning(
                "%d of %d subjects of participants.tsv have no recording under "
                "%s and are left out. Delete %s to index them once they are "
                "downloaded.",
                len(missing),
                len(participants),
                root,
                _CSV_NAME,
            )
        if not rows:
            logger.warning("No AHEPA recordings found under %s", root)
            return

        write_metadata_csv(pd.DataFrame(rows), root, _NAME, _CSV_NAME)

    @property
    def default_task(self) -> DementiaClassificationAHEPA:
        """Return the default dementia classification task.

        Returns:
            A :class:`~pyhealth.tasks.DementiaClassificationAHEPA` for
            Alzheimer's disease against controls, on 4 second windows of
            the artifact-cleaned recordings.
        """
        return DementiaClassificationAHEPA()

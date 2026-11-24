"""Helsinki neonatal EEG dataset with seizure annotations.

Author: Axel Coiffard-Donaty

Multi-channel EEG of 79 term neonates admitted to the neonatal intensive
care unit of the Helsinki University Hospital, with a median recording
duration of 74 minutes. Three experts annotated seizures independently,
second by second. Each recording ``eeg<N>.edf`` holds 19 referential EEG
channels of the 10-20 system plus ECG and respiration, sampled at 256 Hz.
The annotations of each expert are one CSV file with one column per neonate
and one row per second (1 seizure, 0 no seizure, empty after the end of the
recording).

References:
    - Stevenson, N. J., Tapani, K., Lauronen, L., Vanhatalo, S. A dataset
      of neonatal EEG recordings with seizure annotations. Scientific Data
      6:190039, 2019. https://doi.org/10.1038/sdata.2019.39
    - Dataset, version 4: https://doi.org/10.5281/zenodo.4940267
"""

import logging
import re
from pathlib import Path

import pandas as pd

from pyhealth.tasks import SeizureDetectionHelsinki

from .base_dataset import BaseDataset
from .metadata_csv import (
    metadata_csv_exists,
    resolve_metadata_root,
    write_metadata_csv,
)

logger = logging.getLogger(__name__)

_NAME = "helsinki_neonatal"
_CSV_NAME = "helsinki_neonatal-pyhealth.csv"
_RECORDING = re.compile(r"^eeg(\d+)\.edf$", re.IGNORECASE)

# Version 4 replaced annotations_2017_A.csv by a fixed file: the original had
# values past the end of the recording in its first three columns.
_ANNOTATION_FILES: dict[str, tuple[str, ...]] = {
    "a": ("annotations_2017_A_fixed.csv", "annotations_2017_A.csv"),
    "b": ("annotations_2017_B.csv",),
    "c": ("annotations_2017_C.csv",),
}

# Columns of clinical_information.csv kept in the metadata.
_CLINICAL_COLUMNS: dict[str, str] = {
    "Gender": "gender",
    "GA (weeks)": "gestational_age",
    "Diagnosis": "diagnosis",
    "Number of Reviewers Annotating Seizure": "n_reviewers_seizure",
}


def _annotation_file(root: Path, annotator: str) -> str:
    """Return the path of one expert's annotation file.

    Args:
        root: Dataset folder.
        annotator: ``"a"``, ``"b"`` or ``"c"``.

    Returns:
        The absolute path, or an empty string when the file is missing.
    """
    names = _ANNOTATION_FILES[annotator]
    for name in names:
        path = root / name
        if path.is_file():
            if name != names[0]:
                logger.warning(
                    "Using %s; version 4 of the dataset replaced it by %s.",
                    name,
                    names[0],
                )
            return str(path.absolute())
    logger.warning("No %s found under %s.", names[0], root)
    return ""


def _clinical_information(root: Path) -> pd.DataFrame:
    """Read the clinical columns kept in the metadata, indexed by neonate.

    Args:
        root: Dataset folder.

    Returns:
        A frame indexed by the neonate number as a string, with the columns
        of ``_CLINICAL_COLUMNS``; empty when ``clinical_information.csv`` is
        missing.
    """
    path = root / "clinical_information.csv"
    columns = list(_CLINICAL_COLUMNS.values())
    if not path.is_file():
        logger.warning("No clinical_information.csv found under %s.", root)
        return pd.DataFrame(columns=columns)
    clinical = pd.read_csv(path, dtype=str)
    clinical = clinical.set_index(clinical["ID"].str.strip())
    clinical = clinical.rename(columns=_CLINICAL_COLUMNS)
    return clinical.reindex(columns=columns)


class HelsinkiNeonatalDataset(BaseDataset):
    """Helsinki neonatal EEG dataset with seizure annotations.

    79 term neonates recorded in a neonatal intensive care unit, one
    recording each (``eeg1.edf`` to ``eeg79.edf``, 256 Hz, 19 EEG channels
    plus ECG and respiration). Three experts (A, B and C) annotated seizures
    second by second. According to the dataset description, 39 neonates had
    seizures by consensus of the three experts and 22 were seizure free.
    ``patient_id`` is the recording number, for example ``"1"``.

    ``prepare_metadata`` writes ``helsinki_neonatal-pyhealth.csv`` (one row
    per recording) in ``root``, or under
    ``~/.cache/pyhealth/helsinki_neonatal/<hash of root>`` when ``root`` is
    not writable. Each row holds the recording, the three annotation files
    and, from ``clinical_information.csv`` when present, the gender, the
    gestational age range, the diagnosis and the number of experts who
    annotated at least one seizure. The default task is
    :class:`~pyhealth.tasks.SeizureDetectionHelsinki`.

    Getting the data:
        - Source: https://zenodo.org/records/4940267 (version 4)
        - Access: open, Creative Commons Attribution 4.0 license.
        - Download: the 79 EDF files, the three ``annotations_2017_*.csv``
          files and ``clinical_information.csv`` from that page, 4.3 GB in
          total, into one folder.
        - ``root``: that folder.

    Args:
        root: Folder holding the ``eeg<N>.edf`` files and the annotation
            CSV files.
        dataset_name: Name stored on the dataset. Defaults to
            ``"helsinki_neonatal"``.
        config_path: YAML config. Defaults to
            ``configs/helsinki_neonatal.yaml``.
        **kwargs: Forwarded to :class:`~pyhealth.datasets.BaseDataset`
            (``dev``, ``num_workers``, ``cache_dir``).

    Examples:
        >>> from pyhealth.datasets import HelsinkiNeonatalDataset
        >>> dataset = HelsinkiNeonatalDataset(root="/path/to/helsinki")
        >>> dataset.stats()
        >>> samples = dataset.set_task()
        >>> samples[0]["signal"].shape
        torch.Size([18, 512])
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
        """Write ``helsinki_neonatal-pyhealth.csv`` if it is not on disk.

        The CSV has one row per ``eeg<N>.edf`` file, sorted by ``N``, with
        ``patient_id``, ``signal_file``, the annotation files
        ``annotation_file_a``, ``annotation_file_b`` and
        ``annotation_file_c`` (empty when missing) and the clinical columns
        (empty when ``clinical_information.csv`` is missing). An existing
        CSV, in ``root`` or in the cache, is left untouched.

        Returns:
            None. The CSV path is ``<root>/helsinki_neonatal-pyhealth.csv``
            or the same file name under
            ``~/.cache/pyhealth/helsinki_neonatal/<hash of root>``.
        """
        root = Path(self.root)
        if metadata_csv_exists(root, _NAME, _CSV_NAME):
            return

        recordings = []
        if root.is_dir():
            for path in root.iterdir():
                match = _RECORDING.match(path.name)
                if match and path.is_file():
                    recordings.append((int(match.group(1)), path))
        if not recordings:
            logger.warning("No Helsinki recordings (eeg<N>.edf) found under %s", root)
            return

        annotation_files = {
            f"annotation_file_{annotator}": _annotation_file(root, annotator)
            for annotator in _ANNOTATION_FILES
        }
        clinical = _clinical_information(root)
        rows = []
        for number, path in sorted(recordings):
            patient_id = str(number)
            row = {
                "patient_id": patient_id,
                "signal_file": str(path.absolute()),
                **annotation_files,
            }
            for column in clinical.columns:
                value = (
                    clinical.at[patient_id, column]
                    if patient_id in clinical.index
                    else None
                )
                row[column] = "" if pd.isna(value) else str(value).strip()
            rows.append(row)

        write_metadata_csv(pd.DataFrame(rows), root, _NAME, _CSV_NAME)

    @property
    def default_task(self) -> SeizureDetectionHelsinki:
        """Return the default neonatal seizure detection task.

        Returns:
            A :class:`~pyhealth.tasks.SeizureDetectionHelsinki` with
            consensus labels, the 18-channel bipolar montage and 16 second
            windows at 32 Hz.
        """
        return SeizureDetectionHelsinki()

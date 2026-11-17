"""HUP intracranial EEG epilepsy dataset (OpenNeuro ds004100).

Author: Axel Coiffard-Donaty

Intracranial EEG of 57 patients with drug-resistant epilepsy, recorded at
the Hospital of the University of Pennsylvania before surgical resection or
laser ablation, with subdural grids, strips and depth electrodes (ECoG) or
stereotactic depth electrodes only (SEEG). The dataset holds interictal
recordings and recordings of seizures, in EDF, organized with BIDS. For
every recording, ``channels.tsv`` marks the contacts of the clinically
determined seizure onset zone and those that overlap with the resected or
ablated tissue. ``participants.tsv`` gives the surgical outcome.

References:
    - Bernabei, J. M., Li, A. et al. Quantitative approaches to guide
      epilepsy surgery from intracranial EEG. Brain 146(6), 2248-2258, 2023.
      https://doi.org/10.1093/brain/awad007
    - Dataset: https://doi.org/10.18112/openneuro.ds004100.v1.1.3
"""

import csv
import logging
import re
from pathlib import Path

import pandas as pd

from pyhealth.tasks import SOZLocalizationHUP

from .base_dataset import BaseDataset
from .metadata_csv import (
    metadata_csv_exists,
    resolve_metadata_root,
    write_metadata_csv,
)

logger = logging.getLogger(__name__)

_NAME = "hup_ieeg"
_CSV_NAME = "hup_ieeg-pyhealth.csv"

# sub-HUP060_ses-presurgery_task-ictal_acq-seeg_run-01_ieeg.edf
_RECORDING = re.compile(
    r"^(?P<subject>sub-[^_]+)_(?P<session>ses-[^_]+)_task-(?P<task>[^_]+)"
    r"_acq-(?P<acquisition>[^_]+)_run-(?P<run>\d+)_ieeg\.edf$"
)

# Columns of participants.tsv copied to every recording of the patient.
_PARTICIPANT_COLUMNS: tuple[str, ...] = (
    "age",
    "sex",
    "hand",
    "outcome",
    "engel",
    "therapy",
    "implant",
    "target",
    "lesion_status",
    "age_onset",
)

_ONSET, _OFFSET = "sz onset", "sz offset"


def _read_participants(path: Path) -> pd.DataFrame:
    """Read and check ``participants.tsv``.

    Args:
        path: Path to the tab-separated participants file.

    Returns:
        The participants indexed by ``participant_id``, with the id and
        every value stripped (version 1.1.3 writes ``"sub-HUP151 "`` with a
        trailing space). Rows without an id are dropped, and so are repeated
        ids, with a warning.

    Raises:
        ValueError: If the file is empty or lacks the ``participant_id``
            column.
    """
    try:
        participants = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"{path} is empty.") from error
    if "participant_id" not in participants.columns:
        raise ValueError(
            f"{path} has no participant_id column; its columns are "
            f"{list(participants.columns)}. It has to be the tab-separated "
            "participants.tsv of the dataset."
        )
    participants = participants.apply(lambda column: column.str.strip())
    participants = participants[participants["participant_id"] != ""]
    repeated = participants["participant_id"].duplicated()
    if repeated.any():
        logger.warning(
            "%s lists %s more than once; the first row of each is used.",
            path,
            sorted(participants.loc[repeated, "participant_id"].unique()),
        )
        participants = participants[~repeated]
    return participants.set_index("participant_id")


def _electrodes_file(folder: Path, subject: str, session: str, acquisition: str) -> str:
    """Find the ``electrodes.tsv`` of the recordings of one implant.

    BIDS stores one ``electrodes.tsv`` per session, shared by its runs:
    ``sub-<id>_ses-<id>[_acq-<name>][_space-<name>]_electrodes.tsv``.

    Args:
        folder: The ``ieeg`` folder of the session.
        subject: Participant id, for example ``"sub-HUP060"``.
        session: Session id, for example ``"ses-presurgery"``.
        acquisition: Implant of the recording, ``"seeg"`` or ``"ecog"``.

    Returns:
        The absolute path of the file for this implant, or of the one
        without an ``acq`` entity when there is none; an empty string when
        the folder holds neither. With several candidates, for example one
        per coordinate space, the first in alphabetical order is used, with
        a warning.
    """
    for entities in (f"{subject}_{session}_acq-{acquisition}", f"{subject}_{session}"):
        pattern = re.compile(re.escape(entities) + r"(_space-[^_]+)?_electrodes\.tsv")
        found = sorted(
            (p for p in folder.iterdir() if pattern.fullmatch(p.name)),
            key=lambda p: p.name,
        )
        if len(found) > 1:
            logger.warning(
                "%s holds several electrodes files for %s, %s is used: %s",
                folder,
                entities,
                found[0].name,
                [p.name for p in found],
            )
        if found:
            return str(found[0].absolute())
    return ""


def _seizure_times(events_file: Path) -> tuple[str, str]:
    """Read the seizure onset and offset of an ictal recording.

    Args:
        events_file: Path to the BIDS ``*_events.tsv`` file.

    Returns:
        ``(onset, offset)`` in seconds from the start of the recording, as
        written in the file, or empty strings when the file is missing or
        does not hold exactly one ``sz onset`` and one ``sz offset`` row.
    """
    if not events_file.is_file():
        return "", ""
    with open(events_file, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    times = {}
    for kind in (_ONSET, _OFFSET):
        found = [
            (row.get("onset") or "").strip()
            for row in rows
            if (row.get("trial_type") or "").strip().lower() == kind
        ]
        if len(found) != 1:
            logger.warning(
                "%s holds %d %r row(s) instead of 1; the seizure times of this "
                "recording are left empty.",
                events_file,
                len(found),
                kind,
            )
            return "", ""
        times[kind] = found[0]
    return times[_ONSET], times[_OFFSET]


class HUPiEEGDataset(BaseDataset):
    """HUP intracranial EEG epilepsy dataset (OpenNeuro ds004100).

    57 patients with drug-resistant epilepsy, implanted with intracranial
    electrodes (20 with ECoG, 37 with SEEG) before resection or laser
    ablation; 36 became seizure free. Version 1.1.3 holds 319 EDF
    recordings: 111 interictal ones of 5 minutes (two for 55 patients, one
    for ``sub-HUP064``, none for ``sub-HUP158``) and 208 with one seizure
    each (none for ``sub-HUP165``). The sampling rate is 256, 500, 512 or
    1024 Hz and a recording has 52 to 232 channels, depending on the
    patient. ``patient_id`` is the BIDS
    participant id, for example ``"sub-HUP060"``.

    ``participants.tsv`` lists a 58th patient, ``sub-HUP132``, who has no
    recording in version 1.1.3.

    ``prepare_metadata`` writes ``hup_ieeg-pyhealth.csv`` (one row per EDF
    recording found under ``root``) in ``root``, or under
    ``~/.cache/pyhealth/hup_ieeg/<hash of root>`` when ``root`` is not
    writable. Each row holds the recording (``signal_file``), its
    ``channels.tsv`` (``channels_file``), the ``electrodes.tsv`` of its
    patient and implant (``electrodes_file``), its kind (``recording_type``:
    ``"interictal"`` or ``"ictal"``), the implant of the recording
    (``acquisition``: ``"seeg"`` or ``"ecog"``), its run number, the seizure
    onset and offset in seconds for ictal recordings, and the columns of
    ``participants.tsv`` for the patient (age, sex, handedness, outcome,
    Engel class, therapy, implant, surgical target, lesion status and age at
    onset). The default task is :class:`~pyhealth.tasks.SOZLocalizationHUP`.

    The channel marks are in ``channels.tsv``: ``status`` is ``good`` or
    ``bad`` (the ECG channels and almost all scalp channels are marked bad), and
    ``status_description`` lists ``soz`` for the contacts of the clinically
    determined seizure onset zone and ``resect`` for those that overlap with
    the resected or ablated tissue.
    :func:`~pyhealth.tasks.soz_localization_hup.read_hup_channels` reads
    them.

    The position of the contacts is in ``electrodes.tsv`` (``x``, ``y``,
    ``z``), one file per patient in version 1.1.3, read by
    :func:`~pyhealth.tasks.soz_localization_hup.read_hup_electrodes`. Two
    statements of the dataset about these coordinates disagree with each
    other or with the values. The figures below are for the 6119 good
    contacts of the interictal recordings of version 1.1.3, of which 5993
    have a position (119 of the 126 others belong to ``sub-HUP089``,
    ``sub-HUP116`` and ``sub-HUP126``):

    - Unit: ``coordsystem.json`` declares metres, but the values are
      millimetres. They lie between -103.2 and 90.2, and two contacts of
      one electrode with consecutive numbers (``LA1`` and ``LA2``) are a
      median of 5.5 apart for SEEG and 10.9 for ECoG. These are distances
      between the stored coordinates, not measures of the hardware.
    - Space: ``coordsystem.json`` declares ``fsaverage`` and the README of
      the dataset the ICBM152 MNI space. The files do not tell which one
      holds.
    - Side: ``x`` is negative on the left. Of the 5475 contacts whose
      name, without its ``EEG`` prefix, starts with L or R and whose ``x``
      is more than 5 mm from zero, 5470 are on the side of their name; the
      5 others are named ``LOF`` or ``ROF`` and lie 9 to 14 mm across.

    The outcome and the Engel class are known only after surgery, and the
    ``resect`` marks describe the surgery itself: none of them should be an
    input of a model meant to guide surgery (the marks are labels in
    :class:`~pyhealth.tasks.SOZLocalizationHUP`, never inputs).

    Getting the data:
        - Source: https://openneuro.org/datasets/ds004100 (version 1.1.3)
        - Access: open, CC0 license.
        - Download: with the OpenNeuro command line tool or DataLad, as
          the download page of the dataset explains, 14.2 GB (5.4 GB for
          the interictal recordings). Prefer them to ``aws s3 sync``: when
          this loader was written (November 2026), the bucket
          ``s3://openneuro.org/ds004100`` also served files that are not
          in version 1.1.3 (those of ``sub-HUP132`` and an older version
          of 13 text files of ictal recordings).
        - ``root``: the downloaded folder, the one holding
          ``participants.tsv`` and the ``sub-HUP*`` folders.
        - The EDF files are read with MNE; no extra package is needed.

    Args:
        root: BIDS root of the dataset, holding ``participants.tsv`` and the
            ``sub-HUP*`` folders.
        dataset_name: Name stored on the dataset. Defaults to
            ``"hup_ieeg"``.
        config_path: YAML config. Defaults to ``configs/hup_ieeg.yaml``.
        **kwargs: Forwarded to :class:`~pyhealth.datasets.BaseDataset`
            (``dev``, ``num_workers``, ``cache_dir``).

    Examples:
        >>> from pyhealth.datasets import HUPiEEGDataset
        >>> dataset = HUPiEEGDataset(root="/path/to/ds004100")
        >>> dataset.stats()
        >>> samples = dataset.set_task()
        >>> samples[0]["signal"].shape
        torch.Size([1, 6000])
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
        """Write ``hup_ieeg-pyhealth.csv`` if it is not on disk.

        The CSV has one row per ``*_ieeg.edf`` file under
        ``root/sub-*/ses-*/ieeg``, sorted by path, with ``patient_id``,
        ``signal_file``, ``channels_file``, ``electrodes_file``,
        ``recording_type``, ``acquisition``, ``run``,
        ``seizure_onset_second``, ``seizure_offset_second`` and the columns
        of ``participants.tsv``. ``channels_file`` and ``electrodes_file``
        are empty when the recording has no ``channels.tsv`` or
        ``electrodes.tsv``, the seizure times are empty for interictal
        recordings, and the participant columns are empty for a patient that
        ``participants.tsv`` does not list (with a warning). An existing
        CSV, in ``root`` or in the cache, is left untouched: delete it to
        index recordings downloaded later.

        Returns:
            None. The CSV path is ``<root>/hup_ieeg-pyhealth.csv`` or the
            same file name under ``~/.cache/pyhealth/hup_ieeg/<hash of root>``.

        Raises:
            ValueError: If ``participants.tsv`` is empty or lacks the
                ``participant_id`` column.
        """
        root = Path(self.root)
        if metadata_csv_exists(root, _NAME, _CSV_NAME):
            return

        participants_file = root / "participants.tsv"
        participants = None
        if participants_file.is_file():
            participants = _read_participants(participants_file)
        else:
            logger.warning(
                "No participants.tsv found under %s; the patient columns are "
                "left empty.",
                root,
            )

        rows = []
        unlisted = set()
        for path in sorted(root.glob("sub-*/ses-*/ieeg/*_ieeg.edf")):
            match = _RECORDING.match(path.name)
            if match is None:
                logger.warning(
                    "Skipping %s, its name does not follow "
                    "sub-<id>_ses-<id>_task-<name>_acq-<name>_run-<n>_ieeg.edf.",
                    path,
                )
                continue
            subject = match["subject"]
            stem = path.name[: -len("_ieeg.edf")]
            channels_file = path.with_name(f"{stem}_channels.tsv")
            electrodes_file = _electrodes_file(
                path.parent, subject, match["session"], match["acquisition"]
            )
            onset, offset = "", ""
            if match["task"] == "ictal":
                onset, offset = _seizure_times(path.with_name(f"{stem}_events.tsv"))
            row = {
                "patient_id": subject,
                "signal_file": str(path.absolute()),
                "channels_file": (
                    str(channels_file.absolute()) if channels_file.is_file() else ""
                ),
                "electrodes_file": electrodes_file,
                "recording_type": match["task"],
                "acquisition": match["acquisition"],
                "run": match["run"],
                "seizure_onset_second": onset,
                "seizure_offset_second": offset,
            }
            listed = participants is not None and subject in participants.index
            if participants is not None and not listed:
                unlisted.add(subject)
            for column in _PARTICIPANT_COLUMNS:
                value = ""
                if listed and column in participants.columns:
                    value = participants.at[subject, column]
                row[column] = value
            rows.append(row)
        if unlisted:
            logger.warning(
                "participants.tsv does not list %s; their patient columns are "
                "left empty.",
                sorted(unlisted),
            )
        if not rows:
            logger.warning("No HUP iEEG recordings found under %s", root)
            return

        write_metadata_csv(pd.DataFrame(rows), root, _NAME, _CSV_NAME)

    @property
    def default_task(self) -> SOZLocalizationHUP:
        """Return the default seizure onset zone localization task.

        Returns:
            A :class:`~pyhealth.tasks.SOZLocalizationHUP` on 30 second
            single-channel windows of the interictal recordings, labelled
            with the clinically determined seizure onset zone.
        """
        return SOZLocalizationHUP()

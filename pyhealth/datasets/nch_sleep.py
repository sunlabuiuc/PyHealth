"""NCH Sleep DataBank: pediatric polysomnography from NSRR.

Clinical sleep studies (polysomnography, PSG) recorded at Nationwide
Children's Hospital (Columbus, Ohio): 3,984 studies of 3,673 patients,
mostly children. Each study is an EDF recording of EEG, EOG,
EMG, ECG and breathing channels (nasal pressure, airflow, chest and abdomen
effort, SpO2, end-tidal CO2), plus a tab-separated file of scored
annotations: one 30 s sleep stage per epoch and respiratory events such as
obstructive apneas and hypopneas. Electronic health record tables
(demographics, diagnoses, encounters and others) come with the recordings.

References:
    - Lee, H. et al. "A large collection of real-world pediatric sleep
      studies." Scientific Data 9, 421 (2022).
      https://doi.org/10.1038/s41597-022-01545-6
    - Fayyaz, H. et al. "Multimodal Sleep Apnea Detection with Missing or
      Noisy Modalities" (code: github.com/healthylaife/apnea-missing-modality),
      which uses this dataset; its apnea and hypopnea vocabulary is kept here.
    - Dataset: https://sleepdata.org/datasets/nchsdb
"""

import logging
from pathlib import Path

import pandas as pd

from .base_dataset import BaseDataset
from .metadata_csv import (
    metadata_csv_exists,
    resolve_metadata_root,
    write_metadata_csv,
)
from .psg_metadata import read_edf_header, summarize_events

logger = logging.getLogger(__name__)

_NAME = "nch_sleep"
_CSV_NAME = "nch_sleep-pyhealth.csv"

#: Annotation descriptions counted as apneas (from apnea-missing-modality).
NCH_APNEA_EVENTS: frozenset[str] = frozenset(
    {
        "Obstructive Apnea",
        "Central Apnea",
        "Mixed Apnea",
        "apnea",
        "obstructive apnea",
        "central apnea",
        "Apnea",
    }
)

#: Annotation descriptions counted as hypopneas (from apnea-missing-modality).
NCH_HYPOPNEA_EVENTS: frozenset[str] = frozenset(
    {
        "Obstructive Hypopnea",
        "Hypopnea",
        "hypopnea",
        "Mixed Hypopnea",
        "Central Hypopnea",
    }
)

#: Sleep stage annotations, scored per 30 s epoch.
NCH_SLEEP_STAGES: dict[str, str] = {
    "Sleep stage W": "W",
    "Sleep stage N1": "N1",
    "Sleep stage N2": "N2",
    "Sleep stage N3": "N3",
    "Sleep stage R": "R",
}

_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def _subdir(root: Path, name: str) -> Path:
    """Return ``root/name``, matching the folder name without case.

    The NSRR download uses ``sleep_data`` and ``health_data``; some copies
    use ``Sleep_Data`` and ``Health_Data``.

    Args:
        root: The ``nchsdb`` folder.
        name: Folder name in lower case, for example ``"sleep_data"``.

    Returns:
        The existing folder whose name matches ``name`` without case, or
        ``root/name`` when there is none.
    """
    exact = root / name
    if exact.is_dir():
        return exact
    for child in root.iterdir() if root.is_dir() else ():
        if child.is_dir() and child.name.lower() == name.lower():
            return child
    return exact


def _read_health_table(
    folder: Path, file_name: str, columns: list[str]
) -> pd.DataFrame:
    """Read ``columns`` of a health data CSV, or an empty frame if absent.

    Args:
        folder: The ``health_data`` folder.
        file_name: CSV name, for example ``"SLEEP_STUDY.csv"``.
        columns: Columns to keep. Missing ones are added as empty.

    Returns:
        A frame with exactly ``columns``, all as strings.
    """
    path = folder / file_name
    if not path.is_file():
        logger.warning("%s not found; its columns are left empty.", path)
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path, dtype=str)
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df[columns]


def _sort_key(stem: str) -> tuple:
    """Sort key for ``<patient>_<study>`` file stems.

    Args:
        stem: EDF file name without its extension.

    Returns:
        A tuple that orders numeric ids by value. Numeric parts sort before
        text parts, so odd file names do not break the comparison.
    """
    return tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in stem.split("_"))


def read_nch_annotations(path: str | Path) -> pd.DataFrame:
    """Read an NCH annotation file.

    Args:
        path: The tab-separated ``<patient>_<study>.tsv`` of a study.

    Returns:
        One row per annotation with ``onset`` and ``duration`` in seconds from
        the start of the recording, and ``description``.

    Raises:
        ValueError: If the file lacks one of those columns.

    Examples:
        >>> from pyhealth.datasets.nch_sleep import read_nch_annotations
        >>> read_nch_annotations("sleep_data/10012_21037.tsv")  # doctest: +SKIP
           onset  duration    description
        0    0.0      30.0  Sleep stage W
    """
    df = pd.read_csv(path, sep="\t")
    missing = {"onset", "duration", "description"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} has no {sorted(missing)} column(s).")
    return df[["onset", "duration", "description"]]


def summarize_nch_annotations(path: str | Path) -> dict[str, int | float]:
    """Count respiratory events and sleep time in an NCH annotation file.

    Args:
        path: The tab-separated annotation file of a study, read by
            :func:`read_nch_annotations`.

    Returns:
        ``n_apnea`` and ``n_hypopnea`` (events whose description is in
        :data:`NCH_APNEA_EVENTS` or :data:`NCH_HYPOPNEA_EVENTS`),
        ``sleep_hours`` (total duration of the N1, N2, N3 and R stage
        annotations) and ``ahi``, the apnea-hypopnea index: events per hour
        of sleep, 0 when there is no sleep.

    Examples:
        >>> from pyhealth.datasets.nch_sleep import summarize_nch_annotations
        >>> summarize_nch_annotations("sleep_data/10012_21037.tsv")  # doctest: +SKIP
        {'n_apnea': 1, 'n_hypopnea': 4, 'sleep_hours': 0.2833, 'ahi': 17.65}
    """
    return summarize_events(
        read_nch_annotations(path),
        NCH_APNEA_EVENTS,
        NCH_HYPOPNEA_EVENTS,
        NCH_SLEEP_STAGES,
    )


class NCHSleepDataset(BaseDataset):
    """NCH Sleep DataBank: pediatric sleep studies (NSRR ``nchsdb``).

    3,984 clinical polysomnography studies of 3,673 patients at Nationwide
    Children's Hospital. A patient can have several studies, so split by
    patient. ``patient_id`` is ``STUDY_PAT_ID`` and each study is one
    ``studies`` event, whose timestamp is the start of the study (empty for
    a study missing from ``SLEEP_STUDY.csv``).

    ``prepare_metadata`` writes ``nch_sleep-pyhealth.csv`` in ``root``, or
    under ``~/.cache/pyhealth/nch_sleep/<hash of root>`` when ``root`` is not
    writable. It has one row per ``<STUDY_PAT_ID>_<SLEEP_STUDY_ID>.edf``
    under ``sleep_data/`` that has its annotation ``.tsv``, with:

    - ``study_id`` (``SLEEP_STUDY_ID``), ``record_id``
      (``<patient>_<study>``, the file stem), ``signal_file`` and
      ``annotation_file``;
    - ``study_start``, ``age_days`` (age at the study) from
      ``health_data/SLEEP_STUDY.csv``, and ``gender``, ``race``,
      ``ethnicity`` from ``health_data/DEMOGRAPHIC.csv``;
    - ``sfreq``, ``duration_sec`` and ``channels`` (the EDF channel labels,
      joined with ``|``), read from the EDF header;
    - ``n_apnea``, ``n_hypopnea``, ``sleep_hours`` and ``ahi``, counted
      from the annotations by :func:`summarize_nch_annotations`.

    The ``ahi`` column counts scored events per hour of scored sleep. It is
    meant for selecting studies (apnea-missing-modality keeps studies with
    an AHI of at least 3) and is not the clinical AHI of the sleep report.

    Getting the data:
        - Source: https://sleepdata.org/datasets/nchsdb (NSRR)
        - Access: restricted. Create an NSRR account and submit a data
          access request on the dataset page; approval is manual.
        - Download: once approved, ``gem install nsrr`` and then
          ``nsrr download nchsdb``.
        - ``root``: the ``nchsdb`` folder, holding ``sleep_data/``
          (``<STUDY_PAT_ID>_<SLEEP_STUDY_ID>.edf`` and ``.tsv``) and
          ``health_data/`` (``SLEEP_STUDY.csv``, ``DEMOGRAPHIC.csv``, ...).
        - Indexing reads every EDF header and annotation file once. Delete
          the CSV to index studies downloaded later.

    Args:
        root: The ``nchsdb`` folder.
        dataset_name: Name stored on the dataset. Defaults to
            ``"nch_sleep"``.
        config_path: YAML config. Defaults to ``configs/nch_sleep.yaml``.
        **kwargs: Forwarded to :class:`~pyhealth.datasets.BaseDataset`
            (``dev``, ``num_workers``, ``cache_dir``).

    Examples:
        >>> from pyhealth.datasets import NCHSleepDataset
        >>> dataset = NCHSleepDataset(root="/path/to/nchsdb")
        >>> dataset.stats()
        >>> patient = dataset.get_patient(dataset.unique_patient_ids[0])
        >>> study = patient.get_events("studies")[0]
        >>> study.signal_file, study.ahi  # doctest: +SKIP
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
            tables=["studies"],
            dataset_name=dataset_name or _NAME,
            config_path=config_path,
            **kwargs,
        )

    def prepare_metadata(self) -> None:
        """Write ``nch_sleep-pyhealth.csv`` if it is not on disk.

        Studies are listed in order of patient and study id. An EDF whose
        name is not ``<patient>_<study>.edf``, that has no ``.tsv``, or whose
        header or annotations cannot be read, is left out with a warning. A
        study missing from ``SLEEP_STUDY.csv`` or a patient missing from
        ``DEMOGRAPHIC.csv`` is kept, with those columns empty. An existing
        CSV, in ``root`` or in the cache, is left untouched.

        Returns:
            None. The CSV path is ``<root>/nch_sleep-pyhealth.csv`` or the
            same file name under ``~/.cache/pyhealth/nch_sleep/<hash of root>``.
        """
        root = Path(self.root)
        if metadata_csv_exists(root, _NAME, _CSV_NAME):
            return

        sleep_data = _subdir(root, "sleep_data")
        if not sleep_data.is_dir():
            logger.warning("No sleep_data folder found under %s", root)
            return

        rows = []
        no_annotation = []
        for edf in sorted(sleep_data.glob("*.edf"), key=lambda p: _sort_key(p.stem)):
            parts = edf.stem.split("_")
            if len(parts) != 2:
                logger.warning(
                    "Skipping %s: name is not <patient>_<study>.edf", edf.name
                )
                continue
            tsv = edf.with_suffix(".tsv")
            if not tsv.is_file():
                no_annotation.append(edf.name)
                continue
            try:
                header = read_edf_header(edf)
                summary = summarize_nch_annotations(tsv)
            except Exception as error:  # noqa: BLE001 - any unreadable file
                logger.warning("Skipping %s: cannot read it (%s)", edf.name, error)
                continue
            rows.append(
                {
                    "patient_id": parts[0],
                    "study_id": parts[1],
                    "record_id": edf.stem,
                    "signal_file": str(edf.absolute()),
                    "annotation_file": str(tsv.absolute()),
                    **header,
                    **summary,
                }
            )
        if no_annotation:
            logger.warning(
                "%d EDF file(s) under %s have no .tsv annotation file and are "
                "left out: %s. Delete %s to index them once the annotations "
                "are downloaded.",
                len(no_annotation),
                sleep_data,
                no_annotation[:10],
                _CSV_NAME,
            )
        if not rows:
            logger.warning("No NCH studies found under %s", sleep_data)
            return

        df = self._join_health_data(pd.DataFrame(rows), _subdir(root, "health_data"))
        write_metadata_csv(df, root, _NAME, _CSV_NAME)

    @staticmethod
    def _join_health_data(studies: pd.DataFrame, health_data: Path) -> pd.DataFrame:
        """Add the study start, age and demographics to the study rows.

        Args:
            studies: One row per study, with ``patient_id`` and ``study_id``.
            health_data: The ``health_data`` folder.

        Returns:
            ``studies`` with ``study_start`` (``%Y-%m-%d %H:%M:%S``),
            ``age_days``, ``gender``, ``race`` and ``ethnicity``, empty where
            the health tables have no row.
        """
        sleep_study = _read_health_table(
            health_data,
            "SLEEP_STUDY.csv",
            [
                "STUDY_PAT_ID",
                "SLEEP_STUDY_ID",
                "SLEEP_STUDY_START_DATETIME",
                "AGE_AT_SLEEP_STUDY_DAYS",
            ],
        ).rename(
            columns={
                "STUDY_PAT_ID": "patient_id",
                "SLEEP_STUDY_ID": "study_id",
                "SLEEP_STUDY_START_DATETIME": "study_start",
                "AGE_AT_SLEEP_STUDY_DAYS": "age_days",
            }
        )
        sleep_study = sleep_study.drop_duplicates(["patient_id", "study_id"])
        start = pd.to_datetime(
            sleep_study["study_start"], format="mixed", errors="coerce"
        )
        sleep_study["study_start"] = start.dt.strftime(_TIMESTAMP_FORMAT)

        demographic = _read_health_table(
            health_data,
            "DEMOGRAPHIC.csv",
            ["STUDY_PAT_ID", "GENDER_DESCR", "RACE_DESCR", "ETHNICITY_DESCR"],
        ).rename(
            columns={
                "STUDY_PAT_ID": "patient_id",
                "GENDER_DESCR": "gender",
                "RACE_DESCR": "race",
                "ETHNICITY_DESCR": "ethnicity",
            }
        )
        demographic = demographic.drop_duplicates("patient_id")

        df = studies.merge(sleep_study, on=["patient_id", "study_id"], how="left")
        df = df.merge(demographic, on="patient_id", how="left")
        no_study_row = df["age_days"].isna() & df["study_start"].isna()
        # A missing SLEEP_STUDY.csv was already reported by _read_health_table.
        if no_study_row.any() and len(sleep_study):
            logger.warning(
                "%d study(ies) are not in SLEEP_STUDY.csv; their start and age "
                "are left empty: %s",
                int(no_study_row.sum()),
                list(df.loc[no_study_row, "record_id"])[:10],
            )
        return df

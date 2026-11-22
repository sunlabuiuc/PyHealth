"""CHAT: Childhood Adenotonsillectomy Trial sleep studies from NSRR.

A randomized trial of children aged 5 to 9.9 years with mild to moderate
obstructive sleep apnea: early adenotonsillectomy against watchful waiting
with supportive care. Each child had a baseline polysomnography (PSG) and,
for the randomized children, a follow-up PSG about 7 months later. A
non-randomized group was also recorded. Each study is an EDF recording of
EEG, EOG, EMG, ECG and breathing channels (nasal cannula flow, airflow,
chest and abdomen effort, SaO2, end-tidal CO2), with scored sleep stages and
respiratory events in an NSRR XML annotation file.

References:
    - Marcus, C. L. et al. "A randomized trial of adenotonsillectomy for
      childhood sleep apnea." New England Journal of Medicine 368(25),
      2366-2376 (2013). https://doi.org/10.1056/NEJMoa1215881
    - Fayyaz, H. et al. "Multimodal Sleep Apnea Detection with Missing or
      Noisy Modalities" (code: github.com/healthylaife/apnea-missing-modality),
      which uses the baseline studies; its apnea and hypopnea vocabulary is
      kept here.
    - Dataset: https://sleepdata.org/datasets/chat
"""

import logging
import re
import xml.etree.ElementTree as ET
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

_NAME = "chat"
_CSV_NAME = "chat-pyhealth.csv"
_VISIT_ORDER = {"baseline": 0, "followup": 1, "nonrandomized": 2}

#: Annotation descriptions counted as apneas (from apnea-missing-modality,
#: which does not count mixed apneas for CHAT).
CHAT_APNEA_EVENTS: frozenset[str] = frozenset({"Obstructive apnea", "Central apnea"})

#: Annotation descriptions counted as hypopneas (from apnea-missing-modality).
CHAT_HYPOPNEA_EVENTS: frozenset[str] = frozenset({"Hypopnea"})

#: Sleep stage annotations, scored per 30 s epoch, and their normalized names.
CHAT_SLEEP_STAGES: dict[str, str] = {
    "Wake": "W",
    "Stage 1 sleep": "N1",
    "Stage 2 sleep": "N2",
    "Stage 3 sleep": "N3",
    "Stage 4 sleep": "N3",
    "REM sleep": "R",
}

# Subject table columns and their names in the metadata.
_SUBJECT_COLUMNS: dict[str, str] = {
    "ageyear_at_meas": "age_years",
    "male": "male",
    "race3": "race",
}


def read_chat_annotations(path: str | Path) -> pd.DataFrame:
    """Read a CHAT annotation file into ``onset``, ``duration``, ``description``.

    Two formats are read:

    - NSRR XML (``*-nsrr.xml``): one ``ScoredEvent`` per annotation. The
      description is the part of ``EventConcept`` before ``|``, for example
      ``"Obstructive apnea"`` for ``"Obstructive apnea|Obstructive Apnea"``.
      The ``Recording Start Time`` entry is left out.
    - Tab-separated (``*.tsv``) with ``onset``, ``duration`` and
      ``description`` columns, the layout apnea-missing-modality reads.

    Args:
        path: The annotation file. ``.xml`` is read as XML, anything else as
            tab-separated.

    Returns:
        One row per annotation, onsets and durations in seconds from the
        start of the recording.

    Raises:
        ValueError: If a tab-separated file lacks one of those columns.
        xml.etree.ElementTree.ParseError: If an XML file is malformed.

    Examples:
        >>> from pyhealth.datasets.chat import read_chat_annotations
        >>> read_chat_annotations("chat-baseline-300001-nsrr.xml")  # doctest: +SKIP
           onset  duration   description
        0    0.0      30.0          Wake
    """
    path = Path(path)
    if path.suffix.lower() != ".xml":
        df = pd.read_csv(path, sep="\t")
        missing = {"onset", "duration", "description"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} has no {sorted(missing)} column(s).")
        return df[["onset", "duration", "description"]]

    rows = []
    for event in ET.parse(path).getroot().iter("ScoredEvent"):
        concept = (event.findtext("EventConcept") or "").split("|")[0].strip()
        if not concept or concept == "Recording Start Time":
            continue
        rows.append(
            (
                float(event.findtext("Start") or 0),
                float(event.findtext("Duration") or 0),
                concept,
            )
        )
    return pd.DataFrame(rows, columns=["onset", "duration", "description"])


def summarize_chat_annotations(path: str | Path) -> dict[str, int | float]:
    """Count respiratory events and sleep time in a CHAT annotation file.

    Args:
        path: NSRR XML or tab-separated annotation file, read by
            :func:`read_chat_annotations`.

    Returns:
        ``n_apnea``, ``n_hypopnea``, ``sleep_hours`` and ``ahi`` (events per
        hour of sleep), counted with :data:`CHAT_APNEA_EVENTS`,
        :data:`CHAT_HYPOPNEA_EVENTS` and :data:`CHAT_SLEEP_STAGES`.

    Examples:
        >>> from pyhealth.datasets.chat import summarize_chat_annotations
        >>> path = "chat-baseline-300001-nsrr.xml"
        >>> summarize_chat_annotations(path)  # doctest: +SKIP
        {'n_apnea': 2, 'n_hypopnea': 4, 'sleep_hours': 0.2833, 'ahi': 21.18}
    """
    return summarize_events(
        read_chat_annotations(path),
        CHAT_APNEA_EVENTS,
        CHAT_HYPOPNEA_EVENTS,
        CHAT_SLEEP_STAGES,
    )


def _annotation_file(psg: Path, edf: Path, visit: str) -> Path | None:
    """Find a study's annotations: the NSRR XML first, then a TSV by the EDF.

    Args:
        psg: The ``polysomnography`` folder.
        edf: The study's EDF file.
        visit: The visit folder name, for example ``"baseline"``.

    Returns:
        ``annotations-events-nsrr/<visit>/<stem>-nsrr.xml`` if it exists,
        else ``<stem>-nsrr.tsv`` next to the EDF if it exists, else None.
    """
    xml = psg / "annotations-events-nsrr" / visit / f"{edf.stem}-nsrr.xml"
    if xml.is_file():
        return xml
    tsv = edf.with_name(f"{edf.stem}-nsrr.tsv")
    if tsv.is_file():
        return tsv
    return None


def _version_key(path: Path) -> tuple[int, ...]:
    """Sort key for ``chat-<visit>-dataset-<version>.csv`` by version number.

    Args:
        path: A subject table.

    Returns:
        The numbers in the version part of the name, so ``0.14.0`` sorts
        after ``0.9.0``.
    """
    version = path.stem.rsplit("-", 1)[-1]
    return tuple(int(part) for part in re.findall(r"\d+", version))


def _read_subjects(datasets: Path, visit: str) -> pd.DataFrame:
    """Read the subject table of a visit, ``chat-<visit>-dataset-*.csv``.

    Args:
        datasets: The ``datasets`` folder of the download.
        visit: The visit, for example ``"baseline"``. When several releases
            of its table are present, the highest version is read.

    Returns:
        ``nsrrid`` and the columns of ``_SUBJECT_COLUMNS`` (renamed), as
        strings, one row per subject; empty when the table is absent.
    """
    columns = ["nsrrid", *_SUBJECT_COLUMNS.values()]
    tables = sorted(datasets.glob(f"chat-{visit}-dataset-*.csv"), key=_version_key)
    if not tables:
        logger.warning(
            "No chat-%s-dataset-*.csv under %s; the age, sex and race of the "
            "%s studies are left empty.",
            visit,
            datasets,
            visit,
        )
        return pd.DataFrame(columns=columns)
    if len(tables) > 1:
        logger.warning("Several %s subject tables; using %s", visit, tables[-1].name)
    df = pd.read_csv(tables[-1], dtype=str, encoding_errors="replace")
    df.columns = [c.lower() for c in df.columns]
    for column in ["nsrrid", *_SUBJECT_COLUMNS]:
        if column not in df.columns:
            df[column] = pd.NA
    df = df[["nsrrid", *_SUBJECT_COLUMNS]].rename(columns=_SUBJECT_COLUMNS)
    df["nsrrid"] = df["nsrrid"].str.strip()
    return df.drop_duplicates("nsrrid")


class CHATDataset(BaseDataset):
    """CHAT: Childhood Adenotonsillectomy Trial sleep studies (NSRR ``chat``).

    Polysomnography of children aged 5 to 9.9 years with mild to moderate
    obstructive sleep apnea, at a ``baseline`` visit, a ``followup`` visit
    about 7 months later (randomized children) and for a ``nonrandomized``
    group. ``patient_id`` is the NSRR subject id (``nsrrid``) and each study
    is one ``studies`` event. The data are de-identified without dates, so
    events have no timestamp; ``visit`` says which study is which. A child
    can have a baseline and a follow-up study, so split by patient.

    ``prepare_metadata`` writes ``chat-pyhealth.csv`` in ``root``, or under
    ``~/.cache/pyhealth/chat/<hash of root>`` when ``root`` is not writable.
    It has one row per EDF under ``polysomnography/edfs/<visit>/`` that has
    annotations, with:

    - ``visit``, ``record_id`` (the file stem, ``chat-<visit>-<nsrrid>``),
      ``signal_file``, ``annotation_file`` and ``annotation_format``
      (``"xml"`` for ``annotations-events-nsrr/<visit>/<stem>-nsrr.xml``,
      used when present, or ``"tsv"`` for ``<stem>-nsrr.tsv`` next to the
      EDF, the layout apnea-missing-modality reads);
    - ``sfreq``, ``duration_sec`` and ``channels`` (EDF channel labels,
      joined with ``|``), from the EDF header;
    - ``n_apnea``, ``n_hypopnea``, ``sleep_hours`` and ``ahi``, counted
      from the annotations by :func:`summarize_chat_annotations`;
    - ``age_years``, ``male`` and ``race`` (the ``race3`` code), from
      ``datasets/chat-<visit>-dataset-*.csv``.

    The ``ahi`` column counts scored events per hour of scored sleep with
    the event names of apnea-missing-modality (obstructive and central
    apneas, hypopneas). It is meant for selecting studies and is not the
    ``ahi_*`` variables of the NSRR subject tables.

    Getting the data:
        - Source: https://sleepdata.org/datasets/chat (NSRR)
        - Access: restricted. Create an NSRR account and submit a data
          access request on the dataset page; approval is manual.
        - Download: once approved, ``gem install nsrr`` and then
          ``nsrr download chat`` (or a part of it, for example
          ``nsrr download chat/polysomnography/edfs/baseline``).
        - ``root``: the ``chat`` folder, holding ``polysomnography/``
          (``edfs/`` and ``annotations-events-nsrr/``) and ``datasets/``.
        - Indexing reads every EDF header and annotation file once. Delete
          the CSV to index studies downloaded later.

    Args:
        root: The ``chat`` folder.
        dataset_name: Name stored on the dataset. Defaults to ``"chat"``.
        config_path: YAML config. Defaults to ``configs/chat.yaml``.
        **kwargs: Forwarded to :class:`~pyhealth.datasets.BaseDataset`
            (``dev``, ``num_workers``, ``cache_dir``).

    Examples:
        >>> from pyhealth.datasets import CHATDataset
        >>> dataset = CHATDataset(root="/path/to/chat")
        >>> dataset.stats()
        >>> patient = dataset.get_patient(dataset.unique_patient_ids[0])
        >>> [s.visit for s in patient.get_events("studies")]  # doctest: +SKIP
        ['baseline', 'followup']
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
        """Write ``chat-pyhealth.csv`` if it is not on disk.

        Studies are listed by subject, then visit (baseline, follow-up,
        non-randomized). An EDF without annotations, with a name that is
        not ``chat-<visit>-<nsrrid>.edf``, or that MNE cannot read, is left
        out with a warning. A study whose subject is not in the visit's
        subject table is kept, with those columns empty. An existing CSV, in
        ``root`` or in the cache, is left untouched.

        Returns:
            None. The CSV path is ``<root>/chat-pyhealth.csv`` or the same
            file name under ``~/.cache/pyhealth/chat/<hash of root>``.
        """
        root = Path(self.root)
        if metadata_csv_exists(root, _NAME, _CSV_NAME):
            return

        psg = root / "polysomnography"
        edf_root = psg / "edfs"
        if not edf_root.is_dir():
            logger.warning("No polysomnography/edfs folder found under %s", root)
            return

        rows = []
        no_annotation = []
        for edf in sorted(edf_root.glob("*/*.edf")):
            visit = edf.parent.name
            parts = edf.stem.split("-")
            if len(parts) != 3 or parts[0] != "chat" or parts[1] != visit:
                logger.warning(
                    "Skipping %s: name is not chat-%s-<nsrrid>.edf", edf, visit
                )
                continue
            annotation = _annotation_file(psg, edf, visit)
            if annotation is None:
                no_annotation.append(edf.name)
                continue
            try:
                header = read_edf_header(edf)
                summary = summarize_chat_annotations(annotation)
            except Exception as error:  # noqa: BLE001 - any unreadable file
                logger.warning("Skipping %s: cannot read it (%s)", edf.name, error)
                continue
            rows.append(
                {
                    "patient_id": parts[2],
                    "visit": visit,
                    "record_id": edf.stem,
                    "signal_file": str(edf.absolute()),
                    "annotation_file": str(annotation.absolute()),
                    "annotation_format": annotation.suffix.lower().lstrip("."),
                    **header,
                    **summary,
                }
            )
        if no_annotation:
            logger.warning(
                "%d EDF file(s) under %s have no annotation file and are left "
                "out: %s. Delete %s to index them once the annotations are "
                "downloaded.",
                len(no_annotation),
                edf_root,
                no_annotation[:10],
                _CSV_NAME,
            )
        if not rows:
            logger.warning("No CHAT studies found under %s", edf_root)
            return

        df = self._join_subjects(pd.DataFrame(rows), root / "datasets")
        write_metadata_csv(df, root, _NAME, _CSV_NAME)

    @staticmethod
    def _join_subjects(studies: pd.DataFrame, datasets: Path) -> pd.DataFrame:
        """Add the age, sex and race of each visit's subject table.

        Args:
            studies: One row per study, with ``patient_id`` and ``visit``.
            datasets: The ``datasets`` folder of the download.

        Returns:
            ``studies`` with ``age_years``, ``male`` and ``race`` (empty where
            the subject table has no row), sorted by subject id and then by
            visit: baseline, follow-up, non-randomized, others.
        """
        parts = []
        for visit, group in studies.groupby("visit", sort=False):
            subjects = _read_subjects(datasets, visit)
            merged = group.merge(
                subjects, left_on="patient_id", right_on="nsrrid", how="left"
            ).drop(columns="nsrrid")
            missing = merged["age_years"].isna() & merged["male"].isna()
            if missing.any() and len(subjects):
                logger.warning(
                    "%d %s study(ies) are not in the subject table; their age, "
                    "sex and race are left empty: %s",
                    int(missing.sum()),
                    visit,
                    list(merged.loc[missing, "record_id"])[:10],
                )
            parts.append(merged)
        df = pd.concat(parts, ignore_index=True)
        order = df["visit"].map(_VISIT_ORDER).fillna(len(_VISIT_ORDER))
        numeric_id = pd.to_numeric(df["patient_id"], errors="coerce")
        df = (
            df.assign(_id=numeric_id, _visit=order)
            .sort_values(["_id", "patient_id", "_visit", "visit"], kind="stable")
            .drop(columns=["_id", "_visit"])
        )
        return df.reset_index(drop=True)

"""SHHS (Sleep Heart Health Study) polysomnography dataset.

Reads the NSRR ``polysomnography`` release: one EDF recording per night under
``edfs/`` and one Profusion XML annotation per recording under
``annotations-events-profusion/``.

``prepare_metadata`` writes one metadata CSV per visit
(``shhs1-pyhealth.csv``, ``shhs2-pyhealth.csv``) next to the release, or under
``~/.cache/pyhealth/shhs/<hash of root>`` when that directory is not writable.

The default task is
:class:`~pyhealth.tasks.sleep_staging_shhs.SleepStagingSHHS`.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from .base_dataset import BaseDataset
from .metadata_csv import (
    metadata_csv_exists,
    resolve_metadata_root,
    write_metadata_csv,
)

logger = logging.getLogger(__name__)

# Visits published in the NSRR polysomnography release.
_VALID_VISITS = ("shhs1", "shhs2")

# Metadata CSV name for a single visit. The visit is part of the name, so the
# two tables never write to the same file even though they share a directory.
_CSV_TEMPLATE = "{visit}-pyhealth.csv"

# Candidate parents of the Profusion XML inside
# ``annotations-events-profusion``. The released layout is one directory per
# visit; the legacy 1.x loader looked for shhs2 under ``label/``. Both are
# accepted, in this order, so either release works.
_ANNOTATION_SUBDIRS = ("{visit}", "label")

# Stage codes written by the SHHS reading centre: wake, N1-N4 and REM. Code 9
# marks movement/unscored epochs and is not a class, so it is dropped.
VALID_SLEEP_STAGES = frozenset({0, 1, 2, 3, 4, 5})


def parse_patient_id(file_name: str | Path) -> str:
    """Return the NSRR participant id encoded in a SHHS recording file name.

    SHHS names recordings ``shhs1-200001.edf``. The numeric part is the
    participant, and it is shared by that person's visit 1 and visit 2
    recordings, so grouping on it keeps both nights under one
    :class:`~pyhealth.data.Patient`.

    Args:
        file_name: Recording file name or path, with or without a directory.

    Returns:
        The participant id, for example ``"200001"``.

    Raises:
        ValueError: If the name does not follow the ``<visit>-<pid>`` layout.

    Examples:
        >>> from pyhealth.datasets.shhs import parse_patient_id
        >>> parse_patient_id("shhs1-200001.edf")
        '200001'
        >>> parse_patient_id("/data/edfs/shhs2/shhs2-200001.edf")
        '200001'
    """
    stem = Path(file_name).name.split(".")[0]
    parts = stem.split("-")
    if len(parts) != 2 or not parts[1].isdigit():
        raise ValueError(
            f"Unexpected SHHS recording name {Path(file_name).name!r}; "
            "expected '<visit>-<participant>.edf'"
        )
    return parts[1]


def read_sleep_stages(annotation_file: str | Path) -> list[int]:
    """Read the per-epoch sleep-stage codes from a Profusion XML file.

    Each ``SleepStage`` element is one 30 s epoch, in recording order.

    Args:
        annotation_file: Path to ``<visit>-<participant>-profusion.xml``.

    Returns:
        One integer stage code per scored epoch.

    Raises:
        ValueError: If the document has no ``SleepStages`` element or a stage
            is not an integer.
        xml.etree.ElementTree.ParseError: If the file is not well-formed XML,
            for example a truncated download or an HTML error page. Callers
            that iterate a dataset should catch this alongside ``ValueError``
            to skip one broken night instead of aborting the run.

    Examples:
        >>> from pyhealth.datasets.shhs import read_sleep_stages
        >>> import tempfile, pathlib
        >>> path = pathlib.Path(tempfile.mkdtemp()) / "shhs1-200001-profusion.xml"
        >>> _ = path.write_text(
        ...     "<PSGAnnotation><SleepStages>"
        ...     "<SleepStage>0</SleepStage><SleepStage>2</SleepStage>"
        ...     "</SleepStages></PSGAnnotation>"
        ... )
        >>> read_sleep_stages(path)
        [0, 2]
    """
    root = ET.parse(annotation_file).getroot()
    stages = root.find("SleepStages")
    if stages is None:
        raise ValueError(f"No SleepStages element in {annotation_file}")
    codes = []
    for element in stages.findall("SleepStage"):
        text = (element.text or "").strip()
        try:
            codes.append(int(text))
        except ValueError as error:
            raise ValueError(
                f"Non-integer sleep stage {text!r} in {annotation_file}"
            ) from error
    return codes


def _normalize_visits(visits) -> list[str]:
    """Validate a ``visits`` argument and return the visits in the order given."""
    if visits is None:
        return list(_VALID_VISITS)
    if isinstance(visits, str):
        visits = (visits,)
    normalized: list[str] = []
    for visit in visits:
        key = str(visit).strip().lower()
        if key not in _VALID_VISITS:
            raise ValueError(
                f"visits must be a subset of {_VALID_VISITS}, got {visit!r}"
            )
        if key not in normalized:
            normalized.append(key)
    if not normalized:
        raise ValueError("visits must name at least one visit")
    return normalized


def _annotation_path(root: Path, visit: str, patient_id: str) -> Path | None:
    """Return the Profusion XML for one recording, or None when absent."""
    name = f"{visit}-{patient_id}-profusion.xml"
    base = root / "annotations-events-profusion"
    for subdir in _ANNOTATION_SUBDIRS:
        candidate = base / subdir.format(visit=visit) / name
        if candidate.is_file():
            if subdir == "label":
                logger.warning(
                    "Found %s under the legacy 'label/' directory; the released "
                    "NSRR layout uses '%s/'.",
                    name,
                    visit,
                )
            return candidate
    return None


class SHHSDataset(BaseDataset):
    """Base EEG dataset for Sleep Heart Health Study (SHHS)

    The Sleep Heart Health Study (SHHS) is a multi-center cohort study implemented by the National Heart Lung & Blood Institute to determine the cardiovascular and other consequences of sleep-disordered breathing. It tests whether sleep-related breathing is associated with an increased risk of coronary heart disease, stroke, all cause mortality, and hypertension.  In all, 6,441 men and women aged 40 years and older were enrolled between November 1, 1995 and January 31, 1998 to take part in SHHS Visit 1. During exam cycle 3 (January 2001- June 2003), a second polysomnogram (SHHS Visit 2) was obtained in 3,295 of the participants. CVD Outcomes data were monitored and adjudicated by parent cohorts between baseline and 2011. More than 130 manuscripts have been published investigating predictors and outcomes of sleep disorders.

    Getting the data:
        - Source: https://sleepdata.org/datasets/shhs (NSRR)
        - Access: restricted. Create an NSRR account and submit a data access
          request ("Request Data Access" on the dataset page); approval is manual.
        - Download: once approved, use the NSRR downloader
          (``gem install nsrr``, then ``nsrr download shhs/polysomnography``).
        - ``root``: the ``polysomnography`` folder, which holds ``edfs/`` and
          ``annotations-events-profusion/``.

    Args:
        root: the ``polysomnography`` folder (holds ``edfs/`` and
            ``annotations-events-profusion/``).
        dataset_name: name of the dataset. Defaults to ``"shhs"``.
        config_path: YAML config. Defaults to ``configs/shhs.yaml``.
        visits: which visits to load, any subset of ``("shhs1", "shhs2")``.
            Defaults to both. Each visit becomes its own table.
        **kwargs: forwarded to :class:`~pyhealth.datasets.BaseDataset`
            (``cache_dir``, ``num_workers``, ``dev``).

    The metadata CSV holds one row per recording: ``patient_id`` (the
    participant, shared by both visits, for example ``"200001"``), ``visit``
    (``"shhs1"`` or ``"shhs2"``), ``record_id`` (``"<visit>-<participant>"``),
    ``signal_file`` (absolute path to the EDF) and ``annotation_file``
    (absolute path to the Profusion XML).

    Examples:
        >>> from pyhealth.datasets import SHHSDataset
        >>> dataset = SHHSDataset(  # doctest: +SKIP
        ...     root="/path/to/shhs/polysomnography",
        ... )
        >>> dataset.stats()  # doctest: +SKIP
        >>> samples = dataset.set_task()  # doctest: +SKIP
        >>> samples[0]["signal"].shape  # doctest: +SKIP
        torch.Size([2, 3750])
    """

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | None = None,
        visits=None,
        **kwargs,
    ) -> None:
        if config_path is None:
            logger.debug("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "shhs.yaml"

        self.root = str(root)
        self.visits = _normalize_visits(visits)
        self.prepare_metadata()

        root_path = Path(self.root)
        tables = []
        for visit in self.visits:
            csv_name = _CSV_TEMPLATE.format(visit=visit)
            if metadata_csv_exists(root_path, "shhs", csv_name):
                tables.append(visit)
        if not tables:
            raise FileNotFoundError(
                f"No SHHS metadata for visits {self.visits} under {self.root}. "
                "Expected edfs/shhs1 or edfs/shhs2 next to "
                "annotations-events-profusion."
            )

        metadata_root = resolve_metadata_root(
            root_path, "shhs", _CSV_TEMPLATE.format(visit=tables[0])
        )
        super().__init__(
            root=metadata_root,
            tables=tables,
            dataset_name=dataset_name or "shhs",
            config_path=config_path,
            **kwargs,
        )

    def prepare_metadata(self) -> None:
        """Write one metadata CSV per requested visit, if absent.

        The CSV is skipped when it already exists, either next to the release
        or in ``~/.cache/pyhealth/shhs/<hash of root>``. A recording whose
        Profusion annotation cannot be found is dropped and counted, so a
        partially downloaded release still produces a usable table.

        Returns:
            None. The CSV path is ``<root>/<visit>-pyhealth.csv`` or the cache
            equivalent.
        """
        root = Path(self.root)
        for visit in self.visits:
            csv_name = _CSV_TEMPLATE.format(visit=visit)
            if metadata_csv_exists(root, "shhs", csv_name):
                continue
            rows = self._metadata_rows(root, visit)
            if not rows:
                logger.warning(
                    "No SHHS %s recordings with annotations under %s", visit, root
                )
                continue
            df = pd.DataFrame(rows)
            df.sort_values(
                ["patient_id", "record_id"], inplace=True, na_position="last"
            )
            df.reset_index(drop=True, inplace=True)
            write_metadata_csv(df, root, "shhs", csv_name)

    @staticmethod
    def _metadata_rows(root: Path, visit: str) -> list[dict]:
        """Build one metadata row per paired EDF/XML recording for a visit."""
        edf_dir = root / "edfs" / visit
        if not edf_dir.is_dir():
            logger.debug("SHHS %s recordings not found: %s", visit, edf_dir)
            return []

        rows: list[dict] = []
        missing = 0
        skipped = 0
        for edf_path in sorted(edf_dir.glob("*.edf")):
            try:
                patient_id = parse_patient_id(edf_path.name)
            except ValueError:
                logger.warning("Skipping unparsable SHHS file name: %s", edf_path)
                skipped += 1
                continue
            annotation_file = _annotation_path(root, visit, patient_id)
            if annotation_file is None:
                missing += 1
                continue
            rows.append(
                {
                    "patient_id": patient_id,
                    "visit": visit,
                    "record_id": f"{visit}-{patient_id}",
                    "signal_file": str(edf_path.resolve()),
                    "annotation_file": str(annotation_file.resolve()),
                }
            )

        if missing:
            logger.warning(
                "Skipped %d SHHS %s recordings with no Profusion annotation",
                missing,
                visit,
            )
        if skipped:
            logger.warning(
                "Skipped %d files in %s that do not follow '<visit>-<pid>.edf'",
                skipped,
                edf_dir,
            )
        return rows

    @property
    def default_task(self):
        """Return the default sleep-staging task for SHHS.

        Returns:
            A :class:`~pyhealth.tasks.sleep_staging_shhs.SleepStagingSHHS`
            instance with its default 30 s epochs.
        """
        from pyhealth.tasks.sleep_staging_shhs import SleepStagingSHHS

        return SleepStagingSHHS()


if __name__ == "__main__":
    dataset = SHHSDataset(
        root="/srv/local/data/SHHS/polysomnography",
        visits=("shhs1",),
        dev=True,
    )
    dataset.stats()

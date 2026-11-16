import logging
import os
import re
from typing import Dict, List, Optional

import pandas as pd

from pyhealth.datasets import BaseDataset
from pyhealth.tasks.sleep_staging_nsrr import SleepStagingCardioRespiratory

logger = logging.getLogger(__name__)

# EDF file name pattern per cohort; group "visit" is optional, "pid" is required.
_EDF_PATTERNS: Dict[str, "re.Pattern[str]"] = {
    "shhs": re.compile(r"^(?P<visit>shhs[12])-(?P<pid>\d+)\.edf$", re.IGNORECASE),
    "mesa": re.compile(r"^mesa-sleep-(?P<pid>\d+)\.edf$", re.IGNORECASE),
}


class NSRRSleepDataset(BaseDataset):
    """Overnight polysomnography from the NSRR (SHHS and MESA cohorts).

    The Sleep Heart Health Study (SHHS) and the Multi-Ethnic Study of
    Atherosclerosis (MESA) sleep exam are available from the National Sleep
    Research Resource after a data-use request:
    https://sleepdata.org/datasets/shhs and https://sleepdata.org/datasets/mesa

    The dataset indexes each recording (``.edf``) together with its
    expert-scored sleep stages (``*-profusion.xml``). Files are discovered
    recursively under ``root``, so both the NSRR folder layout
    (``polysomnography/edfs/...`` and
    ``polysomnography/annotations-events-profusion/...``) and a flat folder
    work. A metadata CSV is written to ``root`` on first use; delete it to
    re-index after adding recordings. Signals are read lazily by the task, not
    by the dataset.

    SHHS participants can have two visits (``shhs1``, ``shhs2``); each visit is
    one event of the same patient. SHHS has ECG and thorax/abdomen respiratory
    effort; MESA additionally has a PPG (``Pleth``) channel.

    Args:
        root: Directory containing the downloaded EDF and XML files.
        cohort: ``"shhs"`` or ``"mesa"``. Default ``"shhs"``.
        dataset_name: Optional dataset name. Default ``"nsrr_<cohort>"``.
        config_path: Optional path to a YAML config. Defaults to the bundled
            ``configs/nsrr_sleep.yaml``.
        cache_dir: Optional cache directory (see ``BaseDataset``).
        num_workers: Worker processes for parallel operations.
        dev: If True, limits the dataset to a small number of patients.

    Raises:
        ValueError: If ``cohort`` is unsupported or no recording with a
            matching annotation file is found under ``root``.

    Examples:
        >>> from pyhealth.datasets import NSRRSleepDataset
        >>> dataset = NSRRSleepDataset(root="/data/shhs", cohort="shhs")
        >>> dataset.stats()
        >>> samples = dataset.set_task()  # default: SleepStagingCardioRespiratory
    """

    def __init__(
        self,
        root: str,
        cohort: str = "shhs",
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        cohort = (cohort or "").lower()
        if cohort not in _EDF_PATTERNS:
            raise ValueError(
                f"Unsupported cohort '{cohort}'. Expected one of "
                f"{sorted(_EDF_PATTERNS)}."
            )
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "nsrr_sleep.yaml"
            )
        if not os.path.exists(os.path.join(root, f"nsrr-{cohort}-pyhealth.csv")):
            self.prepare_metadata(root, cohort)

        self.cohort = cohort
        super().__init__(
            root=root,
            tables=[cohort],
            dataset_name=dataset_name or f"nsrr_{cohort}",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    @staticmethod
    def prepare_metadata(root: str, cohort: str) -> None:
        """Indexes recordings and writes ``nsrr-<cohort>-pyhealth.csv``.

        Args:
            root: Directory to search recursively for EDF/XML files.
            cohort: ``"shhs"`` or ``"mesa"``.

        Raises:
            ValueError: If no EDF file with a matching annotation is found.
        """
        pattern = _EDF_PATTERNS[cohort]
        xml_files: Dict[str, str] = {}
        edf_files: List[str] = []
        for folder, _, names in os.walk(root):
            for name in names:
                if name.lower().endswith("-profusion.xml"):
                    xml_files[name.lower()] = os.path.join(folder, name)
                elif pattern.match(name):
                    edf_files.append(os.path.join(folder, name))

        rows = []
        for edf in sorted(edf_files):
            match = pattern.match(os.path.basename(edf))
            stem = os.path.splitext(os.path.basename(edf))[0].lower()
            label_file = xml_files.get(f"{stem}-profusion.xml")
            if label_file is None:
                logger.warning("No sleep stage annotation for %s; skipped", edf)
                continue
            rows.append(
                {
                    "patient_id": match.group("pid"),
                    "visit": match.groupdict().get("visit") or cohort,
                    "signal_file": edf,
                    "label_file": label_file,
                }
            )
        if not rows:
            raise ValueError(
                f"No {cohort.upper()} recordings with annotations found in {root}"
            )
        pd.DataFrame(rows).to_csv(
            os.path.join(root, f"nsrr-{cohort}-pyhealth.csv"), index=False
        )

    @property
    def default_task(self) -> SleepStagingCardioRespiratory:
        """Returns the default task, :class:`SleepStagingCardioRespiratory`."""
        return SleepStagingCardioRespiratory()

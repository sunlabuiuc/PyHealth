"""Haaglanden Medisch Centrum (HMC) sleep staging database.

Author: Axel Coiffard-Donaty

HMC holds 151 whole-night polysomnography (PSG) recordings collected in
2018 at the sleep center of the Haaglanden Medisch Centrum (The
Netherlands), from patients referred for PSG for various sleep disorders.
Sleep technicians scored them in 30 second epochs following the AASM
guidelines, version 2.4. Each
recording ``SNxxx`` comes with three files: ``SNxxx.edf`` (signals at
256 Hz), ``SNxxx_sleepscoring.edf`` and ``SNxxx_sleepscoring.txt`` (the
same hypnogram and lights-off/lights-on markers, as EDF+ annotations and as
comma-separated text).

References:
    - Alvarez-Estevez, D., Rijsman, R. M. Inter-database validation of a
      deep learning approach for automatic sleep scoring. PLoS ONE 16(8):
      e0256111, 2021. https://doi.org/10.1371/journal.pone.0256111
    - Alvarez-Estevez, D., Rijsman, R. Haaglanden Medisch Centrum sleep
      staging database (version 1.1). PhysioNet, 2022.
      https://doi.org/10.13026/t79q-fr32
"""

import logging
from pathlib import Path

import pandas as pd

from pyhealth.tasks import SleepStagingHMC

from .base_dataset import BaseDataset
from .metadata_csv import (
    metadata_csv_exists,
    resolve_metadata_root,
    write_metadata_csv,
)

logger = logging.getLogger(__name__)

_CSV_NAME = "hmc-pyhealth.csv"
_SCORING_SUFFIX = "_sleepscoring"


def _find_recordings(root: Path) -> list[Path]:
    """List the HMC signal files under ``root`` or ``root/recordings``.

    Args:
        root: The ``recordings`` folder of the download, or its parent.

    Returns:
        The ``SNxxx.edf`` signal files, sorted by name. Scoring EDF files
        (``SNxxx_sleepscoring.edf``) are not included.
    """
    folders = [root, root / "recordings"]
    files = {
        path
        for folder in folders
        if folder.is_dir()
        for path in folder.glob("SN*.edf")
        if not path.stem.endswith(_SCORING_SUFFIX)
    }
    return sorted(files, key=lambda path: path.name)


class HMCDataset(BaseDataset):
    """Haaglanden Medisch Centrum (HMC) sleep staging database.

    151 whole-night PSG recordings (85 men, 66 women, mean age 53.9 years,
    according to PhysioNet) from a clinical sleep center, sampled at 256 Hz
    with four EEG derivations (F4-M1, C4-M1, O2-M1, C3-M2), two EOG, chin
    EMG and ECG. Each recording is one subject, so ``patient_id`` is the
    recording id, for example ``"SN001"``. Version 1.1 removed SN014, SN064
    and SN135, so the ids run from SN001 to SN154 with those three gaps
    (see the Release Notes and the ``RECORDS`` file at
    https://physionet.org/content/hmc-sleep-staging/1.1/).

    ``prepare_metadata`` writes ``hmc-pyhealth.csv`` (one row per recording
    that has both a signal file and a ``_sleepscoring.txt`` file) in
    ``root``, or under ``~/.cache/pyhealth/hmc/<hash of root>`` when
    ``root`` is not writable. The CSV is written once: recordings added
    after it, for example when a partial download is completed, are not
    picked up until it is deleted. The default task is
    :class:`~pyhealth.tasks.SleepStagingHMC`.

    Getting the data:
        - Source: https://physionet.org/content/hmc-sleep-staging/1.1/
        - Access: open, Creative Commons Attribution 4.0 license.
        - Download: ``wget -r -N -c -np
          https://physionet.org/files/hmc-sleep-staging/1.1/`` (15.7 GB),
          or the 12.9 GB zip file from the same page.
        - ``root``: the downloaded ``recordings`` folder, or its parent.
        - Disk: the default task cache takes about 34 GB for the 151
          recordings (extrapolated from SN001 to SN004).

    Args:
        root: Folder holding the ``SNxxx.edf`` and
            ``SNxxx_sleepscoring.txt`` files, or its parent.
        dataset_name: Name stored on the dataset. Defaults to ``"hmc"``.
        config_path: YAML config. Defaults to ``configs/hmc.yaml``.
        **kwargs: Forwarded to :class:`~pyhealth.datasets.BaseDataset`
            (``dev``, ``num_workers``, ``cache_dir``).

    Examples:
        >>> from pyhealth.datasets import HMCDataset
        >>> dataset = HMCDataset(root="/path/to/hmc-sleep-staging/1.1")
        >>> dataset.stats()
        >>> samples = dataset.set_task()
        >>> samples[0]["signal"].shape
        torch.Size([4, 7680])
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
            config_path = Path(__file__).parent / "configs" / "hmc.yaml"

        self.root = root
        self.prepare_metadata()

        super().__init__(
            root=resolve_metadata_root(root, "hmc", _CSV_NAME),
            tables=["recordings"],
            dataset_name=dataset_name or "hmc",
            config_path=config_path,
            **kwargs,
        )

    def prepare_metadata(self) -> None:
        """Write ``hmc-pyhealth.csv`` if it is not already on disk.

        The CSV has one row per recording with ``patient_id``,
        ``signal_file`` and ``label_file`` (the ``_sleepscoring.txt``
        file). Recordings without that file, for example after a partial
        download, are left out with a warning. An existing CSV, in ``root``
        or in the cache, is left untouched, so delete it to include
        recordings downloaded after it was written.

        Returns:
            None. The CSV path is either ``<root>/hmc-pyhealth.csv`` or
            ``~/.cache/pyhealth/hmc/<hash of root>/hmc-pyhealth.csv``.
        """
        root = Path(self.root)
        if metadata_csv_exists(root, "hmc", _CSV_NAME):
            return

        rows = []
        missing = []
        for signal_file in _find_recordings(root):
            label_file = signal_file.with_name(
                f"{signal_file.stem}{_SCORING_SUFFIX}.txt"
            )
            if not label_file.is_file():
                missing.append(signal_file.stem)
                continue
            rows.append(
                {
                    "patient_id": signal_file.stem,
                    "signal_file": str(signal_file.absolute()),
                    "label_file": str(label_file.absolute()),
                }
            )
        if missing:
            logger.warning(
                "%d HMC recordings have no _sleepscoring.txt file and are left out: %s",
                len(missing),
                ", ".join(missing),
            )
        if not rows:
            logger.warning("No scored HMC recordings found under %s", root)
            return

        write_metadata_csv(pd.DataFrame(rows), root, "hmc", _CSV_NAME)

    @property
    def default_task(self) -> SleepStagingHMC:
        """Return the default HMC sleep staging task.

        Returns:
            A :class:`~pyhealth.tasks.SleepStagingHMC` on the four EEG
            channels with 30 second epochs between lights off and lights on.
        """
        return SleepStagingHMC()

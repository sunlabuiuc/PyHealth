"""
Author(s): Bronze Frazer
NetID(s):  bfrazer2
Paper:     wav2sleep: A Unified Multi-Modal Approach to Sleep Stage Classification from Physiological Signals
Link:      https://arxiv.org/abs/2411.04644
Desc:      PyHealth Dataset for the collection of 7 datasets used to train wav2sleep
"""

import logging
import os
import pandas as pd
from pathlib import Path
from typing import Optional

from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)


class Wav2SleepDataset(BaseDataset):
    """Unified dataset of PSG recordings (EDF and annotation files)

    Spans 7 datasets hosted on sleepdata.org that are used in wav2sleep: A Unified Multi-Modal Approach to Sleep Stage Classification from Physiological Signals (https://arxiv.org/abs/2411.04644):
    SHHS, MESA, WSC, CHAT, CFS, CCSHS, and MROS.

    Signal availability varies by source dataset:
    - ECG, THX, ABD: available in all datasets
    - PPG: available in MESA, CHAT, CFS, CCSHS only

    Note:
        A Data Use Agreement must be completed via sleepdata.org

    Here are the steps required to download the raw data:
        1. Fill out a Data Use Agreement on sleepdata.org
        2. Receive a Data Access Token (sleepdata.org/token)
        3. Use the nsrr gem tool (https://github.com/nsrr/nsrr-gem)

    Once you have your token...
    Create and enter the directory you want to use as the root:
        mkdir PSG_root
        cd PSG_root

    Then download each dataset using NSRR Ruby Gem (https://github.com/nsrr/nsrr-gem)
    using the following command structures:
        For SHHS, MESA, CHAT, CFS, CCSHS, and MROS:
            nsrr download {dataset}/polysomnography/edfs --fast
            nsrr download {dataset}/polysomnography/annotations-events-profusion --fast
        For WSC:
            nsrr download wsc/polysomnography --fast

    The resulting structure will be:

    PSG_root
    ├── ccshs
    │   └── polysomnography
    │       ├── annotations-events-profusion
    │       └── edfs
    ├── cfs
    │   └── polysomnography
    │       ├── annotations-events-profusion
    │       └── edfs
    ├── chat
    │   └── polysomnography
    │       ├── annotations-events-profusion
    │       │   ├── baseline
    │       │   └── followup
    │       |   └── nonrandomized
    │       └── edfs
    │           ├── baseline
    │           └── followup
    │           └── nonrandomized
    ├── mesa
    │   └── polysomnography
    │       ├── annotations-events-profusion
    │       └── edfs
    ├── mros
    │   └── polysomnography
    │       ├── annotations-events-profusion
    │       │   ├── visit1
    │       │   └── visit2
    │       └── edfs
    │           ├── visit1
    │           └── visit2
    ├── shhs
    │   └── polysomnography
    │       ├── annotations-events-profusion
    │       │   ├── shhs1
    │       │   └── shhs2
    │       └── edfs
    │           ├── shhs1
    │           └── shhs2
    └── wsc
        └── polysomnography

    Args:
        root: root directory containing one subdirectory per source dataset
        config_path: optional path to YAML config, defaults to wav2sleep.yaml

    Examples:
        >>> dataset = Wav2SleepDataset(root = "path/to/root")
        >>> dataset.stats()
    """

    def __init__(self, root: str, config_path: Optional[str] = None) -> None:
        # Validate root
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Root directory not found: {root}")
        
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "wav2sleep.yaml"

        # Prepare metadata file if it does not already exist
        metadata_file = os.path.join(root, "wav2sleep-metadata.csv")
        if not os.path.exists(metadata_file):
            logger.info("Preparing Wav2Sleep metadata...")
            self.prepare_metadata(root)

        super().__init__(
            root=root,
            tables=["wav2sleep"],
            dataset_name="wav2sleep",
            config_path=config_path,
        )

    def prepare_metadata(self, root: str) -> None:
        """Prepares a metadata CSV file that outlines the locations of
        EDF and label files across all datasets found in root.

        Args:
            root: root directory containing one subdirectory per source dataset
        """
        rows = []

        for dataset_dir in Path(root).iterdir():
            if not dataset_dir.is_dir():
                continue

            source_dataset = dataset_dir.name
            logger.info(f"Processing {source_dataset}...")

            for edf_dir, label_dir in self.get_edf_and_label_dirs(dataset_dir):
                for edf_file in edf_dir.glob("*.edf"):
                    patient_id = edf_file.stem

                    label_file_extension = (
                        ".stg.txt" if source_dataset == "wsc" else "-profusion.xml"
                    )

                    label_file = label_dir / f"{patient_id}{label_file_extension}"
                    if not label_file.exists():
                        logger.warning(
                            f"Label file not found for \
                            {patient_id} in {source_dataset}, skipping"
                        )
                        continue

                    rows.append(
                        {
                            "patient_id": patient_id,
                            "source_dataset": source_dataset,
                            "edf_path": str(edf_file),
                            "label_path": str(label_file),
                        }
                    )

        output_path = Path(root) / "wav2sleep-metadata.csv"
        pd.DataFrame(rows).to_csv(output_path, index=False)
        logger.info(f"Metadata saved to {output_path}")

    def get_edf_and_label_dirs(self, dataset_dir: Path) -> list[tuple[Path, Path]]:
        """Retrieves the EDF and label directories for a given dataset.

        Handles datasets that have an extra subdirectory layer (e.g. SHHS).

        Args:
            dataset_dir: path to the dataset directory (e.g. root/shhs)

        Returns:
            list[tuple[Path, Path]]: A list of (edf_dir, label_dir) pairs,
                one per subdirectory if subdirectories exist, otherwise a single pair.
        """

        if dataset_dir.name == "wsc":
            edf_dir = dataset_dir / "polysomnography"
            label_dir = (
                dataset_dir / "polysomnography"
            )  # the annotations for WSC are not in a separate directory
        else:
            edf_dir = dataset_dir / "polysomnography" / "edfs"
            label_dir = dataset_dir / "polysomnography" / "annotations-events-profusion"

        subdirs = [d for d in edf_dir.iterdir() if d.is_dir()]
        if subdirs:
            return [(edf_dir / d.name, label_dir / d.name) for d in sorted(subdirs)]

        return [(edf_dir, label_dir)]


if __name__ == "__main__":
    # ../../../full_sample_PSG/ has one edf and annotation per dataset (currently not per subdir)
    dataset = Wav2SleepDataset(root="../../../full_sample_PSG/")
    dataset.stats()

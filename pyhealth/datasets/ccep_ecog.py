"""
PyHealth dataset for the CCEP ECoG dataset.

Dataset link:
    https://openneuro.org/datasets/ds004080
"""

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import mne_bids

from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)


class CCEPECoGDataset(BaseDataset):
    """Dataset class for the CCEP ECoG dataset.

    Dataset is organized in BIDS format. This class parses and labels subjects who have 
    all electrodes labeled, including at least one electrode in the Seizure Onset Zone (SOZ).
    
    The raw BIDS directory should contain patient folders like `sub-<patient_id>`.

    Attributes:
        root (str): Root directory of the raw data.
        dataset_name (str): Name of the dataset.
        config_path (str): Path to the configuration file.
    """

    def __init__(
        self,
        root: str = ".",
        config_path: Optional[str] = str(Path(__file__).parent / "configs" / "ccep_ecog.yaml"),
        **kwargs,
    ) -> None:
        """Initializes the CCEP ECoG dataset.

        Args:
            root (str): Root directory of the raw data. Defaults to the working directory.
            config_path (Optional[str]): Path to the configuration file. Defaults to "configs/ccep_ecog.yaml".

        Raises:
            FileNotFoundError: If the dataset path does not exist.
            ValueError: If the dataset does not adhere to the expected BIDS structure   .

        Example::
            >>> dataset = CCEPECoGDataset(root="./data/ds004080")
        """
        self._verify_data(root)
        self._index_data(root)

        super().__init__(
            root=root,
            tables=["ecog"],
            dataset_name="ccep_ecog",
            config_path=config_path,
            **kwargs,
        )

    def _verify_data(self, root: str) -> None:
        """Verifies the presence and structure of the dataset directory.

        Ensures the root path exists, verifies the presence of subject directories as well as 
        at least one header file and electrode file.

        Args:
            root (str): Root directory of the raw data.

        Raises:
            FileNotFoundError: If the dataset path does not exist.
            ValueError: If the dataset lacks subjects or core BIDS files.
        """
        if not os.path.exists(root):
            msg = f"Dataset path '{root}' does not exist"
            logger.error(msg)
            raise FileNotFoundError(msg)
        
        # Check for presence of subjects
        subjects = list(Path(root).glob("sub-*"))
        if not subjects:
            msg = f"BIDS root '{root}' contains no 'sub-*' subject folders"
            logger.error(msg)
            raise ValueError(msg)

        # Check for at least one recording
        if not any(Path(root).rglob("*.vhdr")):
            msg = f"BIDS root '{root}' contains no '.vhdr' files"
            logger.error(msg)
            raise ValueError(msg)

        # Check for at least one electrode file
        if not any(Path(root).rglob("*_electrodes.tsv")):
            msg = f"BIDS root '{root}' contains no 'electrodes.tsv' file"
            logger.error(msg)
            raise ValueError(msg)

        # Check for at least one channels file
        if not any(Path(root).rglob("*_channels.tsv")):
            msg = f"BIDS root '{root}' contains no 'channels.tsv' files"
            logger.error(msg)
            raise ValueError(msg)

        # Check for at least one events file
        if not any(Path(root).rglob("*_events.tsv")):
            msg = f"BIDS root '{root}' contains no 'events.tsv' files"
            logger.error(msg)
            raise ValueError(msg)

    def _index_data(self, root: str) -> pd.DataFrame:
        """Parses and indexes metadata for all available patients in the dataset.

        Args:
            root (str): Root directory of the raw data.

        Returns:
            pd.DataFrame: Table of patient ECoG signal metadata.
        """
        try:
            subjects = mne_bids.get_entity_vals(root, "subject")
        except FileNotFoundError:
            subjects = []

        rows = []
        root_path = Path(root)

        for sub in subjects:
            has_soz = False
            patient_dir = root_path / f"sub-{sub}"

            for tsv_file in patient_dir.rglob("*electrodes.tsv"):
                try:
                    df = pd.read_csv(tsv_file, sep="\t")
                    cols = [c.lower() for c in df.columns]
                    if "soz" in cols:
                        col_series = df["soz"].str.lower()
                        # Verify that there is at least one electrode in the SOZ and all electrodes are labeled
                        if (col_series == "yes").any() and col_series.isin(["yes", "no"]).all():
                            has_soz = True
                            break
                except Exception as e:
                    logger.warning(
                        f"Skipping metadata file {tsv_file} due to error: {e}"
                    )
                    continue

            for header_file in patient_dir.rglob("*.vhdr"):
                # Single electrodes.tsv file per session
                elec_match = list(header_file.parent.glob("*electrodes.tsv"))
                electrodes_file = str(elec_match[0]) if elec_match else ""

                # Multiple channels.tsv and events.tsv files per session
                # header_file has the same base name as channels.tsv and events.tsv
                base_name = header_file.name.replace("_ieeg.vhdr", "")
            
                chan_path = header_file.parent / f"{base_name}_channels.tsv"
                channels_file = str(chan_path) if chan_path.exists() else ""

                evt_path = header_file.parent / f"{base_name}_events.tsv"
                events_file = str(evt_path) if evt_path.exists() else ""

                entities = mne_bids.get_entities_from_fname(str(header_file))

                rows.append(
                    {
                        "patient_id": sub,
                        "session_id": entities.get("session", ""),
                        "task_id": entities.get("task", ""),
                        "run_id": entities.get("run", ""),
                        "header_file": str(header_file),
                        "electrodes_file": electrodes_file,
                        "channels_file": channels_file,
                        "events_file": events_file,
                        "has_soz": has_soz,
                    }
                )

        if not rows:
            logger.warning(
                "No valid BIDS ECoG header files (.vhdr) were found for any subjects. "
                "Ensure your root directory matches the BIDS structure (sub-*/ses-*/ieeg/*.vhdr)."
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df.sort_values(["patient_id"], inplace=True)
            df.reset_index(drop=True, inplace=True)

        output_path = os.path.join(root, "ccep_ecog-metadata-pyhealth.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Wrote metadata to {output_path}")

        return df

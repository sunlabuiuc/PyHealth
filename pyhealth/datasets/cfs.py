"""Cleveland Family Study (CFS) Dataset Implementation for PyHealth."""

import logging
import os
from typing import Optional

import pandas as pd

from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)


class CFSDataset(BaseDataset):
    """Cleveland Family Study (CFS) Polysomnography Dataset.

    The Cleveland Family Study (CFS) is a cohort study of sleep-disordered breathing
    in families. This dataset from the National Sleep Research Resource (NSRR) contains
    overnight polysomnography (PSG) recordings with clinical data.

    Dataset is available at https://sleepdata.org/datasets/cfs

    The CFS dataset includes:
    - Raw polysomnography (PSG) data in EDF format (signal files)
    - Sleep stage annotations in XML format (both Profusion and NSRR versions)
    - Clinical and demographic data in CSV format

    Notes:
        - EDF files contain signals such as EEG, EOG, EMG, ECG, respiratory signals
        - Annotations include sleep stages (Wake, N1, N2, N3, REM) and sleep events
        - Clinical variables include AHI, BMI, age, gender, race/ethnicity, smoking status

    Args:
        root (str): Root directory containing the CFS dataset files.
            Should contain 'datasets/' and 'polysomnography/' subdirectories.
        dataset_name (Optional[str]): Name of the dataset. Default is "cfs".
        config_path (Optional[str]): Path to the dataset configuration file.
            If None, uses the default config in configs/cfs.yaml.
        dev (bool): If True, only load a subset of data for development/testing.
            Default is False.

    Attributes:
        root (str): Root directory of the dataset.
        dataset_name (str): Name of the dataset.
        samples (Optional[List[Dict]]): List of patient samples after processing.
        patient_to_index (Optional[Dict[str, List[int]]]): Mapping from patient ID
            to list of sample indices.
        visit_to_index (Optional[Dict[str, List[int]]]): Mapping from visit/record ID
            to list of sample indices.

    Examples:
        >>> from pyhealth.datasets import CFSDataset
        >>> # Load CFS PSG dataset
        >>> dataset = CFSDataset(
        ...     root="/path/to/cfs",
        ... )
        >>> dataset.stat()
        >>> dataset.info()
        >>>
        >>> # Create a sleep staging task
        >>> from pyhealth.tasks import SleepStagingCFS
        >>> task = SleepStagingCFS(dataset, pred_horizon=30)
        >>> samples = task.generate_samples()

    References:
        Dean, D. A., Goldberger, A. L., Mueller, R., ... (2016). "Scaling up scientific
        discovery in sleep medicine: The National Sleep Research Resource." Sleep, 39(5), 1151-1164.
    """

    def __init__(
        self,
        root: str,
        dataset_name: str = "cfs",
        config_path: Optional[str] = None,
        dev: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the CFS dataset.

        Args:
            root: Root directory of the CFS dataset.
            dataset_name: Name of the dataset.
            config_path: Path to the configuration file.
            dev: If True, load only a small subset for development.
            cache_dir: Cache directory for processed data.
            **kwargs: Additional arguments passed to BaseDataset.
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "cfs.yaml"
            )
            logger.info(f"Using default CFS config: {config_path}")

        # Prepare metadata if needed
        self._prepare_metadata(root)

        # Initialize BaseDataset
        super().__init__(
            root=root,
            tables=["demographics", "polysomnography"],
            dataset_name=dataset_name,
            config_path=config_path,
            cache_dir=cache_dir,
            **kwargs,
        )

    def _prepare_metadata(self, root: str) -> None:
        """Prepare polysomnography metadata CSV file.

        Creates a CSV file with metadata about each PSG recording, including
        file paths to EDF files and annotation files.

        Args:
            root: Root directory of the CFS dataset.
        """
        metadata_path = os.path.join(root, "polysomnography-metadata-pyhealth.csv")

        # Only create if it doesn't exist
        if os.path.exists(metadata_path):
            logger.info(f"PSG metadata already exists at {metadata_path}")
            return

        logger.info(f"Creating PSG metadata at {metadata_path}")

        # Find all EDF files
        edf_dir = os.path.join(root, "polysomnography", "edfs")
        if not os.path.exists(edf_dir):
            logger.warning(
                f"EDF directory not found at {edf_dir}. "
                "Skipping PSG metadata creation."
            )
            return

        records = []

        # Iterate through EDF files
        for edf_file in os.listdir(edf_dir):
            if not edf_file.endswith(".edf"):
                continue

            # Extract identifiers from filename
            # Format: nsrrid-nightnum.edf (e.g., 800002-01.edf)
            parts = edf_file.replace(".edf", "").split("-")
            if len(parts) < 2:
                logger.warning(f"Cannot parse filename: {edf_file}")
                continue

            nsrrid = parts[0]
            night_num = parts[1] if len(parts) > 1 else "01"
            study_id = f"{nsrrid}-{night_num}"

            # Build file paths
            signal_file = os.path.join("polysomnography", "edfs", edf_file)

            # Check for annotation files (NSRR version preferred, then Profusion)
            label_file_nsrr = os.path.join(
                "polysomnography", "annotations-events-nsrr",
                f"{study_id}_nsrr.xml"
            )
            label_file = os.path.join(
                "polysomnography", "annotations-events-profusion",
                f"{study_id}_profusion.xml"
            )

            # Check which annotation files actually exist
            label_file_nsrr_full = os.path.join(root, label_file_nsrr)
            label_file_full = os.path.join(root, label_file)

            label_file_nsrr_exists = os.path.exists(label_file_nsrr_full)
            label_file_exists = os.path.exists(label_file_full)

            if not label_file_nsrr_exists and not label_file_exists:
                logger.warning(
                    f"No annotation files found for {study_id}. "
                    f"Checked: {label_file_nsrr}, {label_file}"
                )
                continue

            record = {
                "nsrrid": nsrrid,
                "study_id": study_id,
                "signal_file": signal_file,
                "label_file": label_file if label_file_exists else "",
                "label_file_nsrr": label_file_nsrr if label_file_nsrr_exists else "",
                "record_date": None,  # Date not directly available from filenames
            }
            records.append(record)

        if not records:
            logger.warning(
                "No PSG records found. Check that EDF files exist in "
                f"{edf_dir}"
            )
            return

        # Create DataFrame and save
        metadata_df = pd.DataFrame(records)
        logger.info(f"Created metadata for {len(metadata_df)} PSG records")

        # Sort by nsrrid and study_id for consistency
        metadata_df = metadata_df.sort_values(["nsrrid", "study_id"]).reset_index(
            drop=True
        )

        # Save to CSV
        metadata_df.to_csv(metadata_path, index=False)
        logger.info(f"Saved PSG metadata to {metadata_path}")


__all__ = ["CFSDataset"]

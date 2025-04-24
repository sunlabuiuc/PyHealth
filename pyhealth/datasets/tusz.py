import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .base_dataset import BaseDataset
from ..tasks import COVID19CXRClassification

logger = logging.getLogger(__name__)

class TUSZDataset(BaseDataset):
    def __init__(
    self,
    root: str,
    dataset_name: Optional[str] = None,
    config_path: Optional[str] = None,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = (
                Path(__file__).parent / "configs" / "eeg_tusz.yaml"
            )
        if not os.path.exists(os.path.join(root, "eeg_tusz-metadata-pyhealth.csv")):
            self.prepare_metadata(root)
        default_tables = ["eeg_tusz"]
        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "eeg_tusz",
            config_path=config_path,
        )
        return
    
    def prepare_metadata(self, root: str) -> None:
        """Prepare metadata for the TUSZ dataset.

        Args:
            root: Root directory containing the dataset files.

        This method:
        1. Loops through all of the files in the root directory.
        2. Looks for each instance of an .edf file and its corresponding .csv_bi.
        3. Opens the .csv_bi file as a CSV file and discards the first 5 lines.
        4. Adds to a data frame the file name of the .edf file and columns read from the previous step.
        """
        metadata = []
        root_path = Path(root)

        for edf_file in root_path.rglob("*.edf"):
            
            csv_bi_file = edf_file.with_suffix(".csv_bi")
            if csv_bi_file.exists():
                try:
                    # Read the .csv_bi file, skipping the first 5 lines
                    df = pd.read_csv(csv_bi_file, skiprows=5)
                    # Extract metadata parts from the edf_file name
                    parts = edf_file.name.split("_")
                    if len(parts) >= 3:
                        patient_id, session_id, slug_number = parts[0], parts[1], parts[2]
                    else:
                        logger.warning(f"Unexpected edf_file format: {edf_file.name}")
                        patient_id, session_id, slug_number = None, None, None

                    # Add the .edf file name, extracted metadata, and data from the .csv_bi file to the metadata
                    for _, row in df.iterrows():
                        metadata.append({
                            "edf_file": edf_file.name,
                            "subject_id": patient_id,
                            "session_id": session_id,
                            "slug_number": slug_number,
                            **row.to_dict()
                        })
                except Exception as e:
                    logger.error(f"Error processing file {csv_bi_file}: {e}")
            else:
                logger.warning(f"Missing corresponding .csv_bi file for {edf_file}")


        # Save the metadata to a CSV file
        metadata_df = pd.DataFrame(metadata)

        # Safely drop the "confidence" column if it exists
        if "confidence" in metadata_df.columns:
            metadata_df = metadata_df.drop(columns=["confidence"])

        metadata_csv_path = root_path / "eeg_tusz-metadata-pyhealth.csv"
        metadata_df.to_csv(metadata_csv_path, index=False)
        logger.info(f"Metadata prepared and saved to {metadata_csv_path}")


    @property
    def default_task(self) -> COVID19CXRClassification:
        """Returns the default task for this dataset.

        Returns:
            COVID19CXRClassification: The default classification task.
        """
        return COVID19CXRClassification()
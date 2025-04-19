import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from ..tasks import DREAMTE4SleepingStageClassification
from .base_dataset import BaseDataset
from .dreamt_feature_engineering import *

logger = logging.getLogger(__name__)


class DREAMTE4Dataset(BaseDataset):
    """Base dataset for the DREAMT sleep study dataset.

    The DREAMT dataset contains polysomnography recordings with sleep stage annotations
    and respiratory event information. This dataset is designed for sleep stage 
    classification tasks using physiological signals.

    Dataset is available at:
    https://physionet.org/content/dreamt/1.0.0/

    Data Description:
    ----------------
    - Contains polysomnography recordings from sleep studies
    - Includes sleep stage annotations (0-4 corresponding to W, N1, N2, N3, REM)
    - Provides respiratory event information (apneas, hypopneas)
    - Contains derived physiological features for each 30-second epoch
    - Includes demographic and clinical information (BMI, AHI severity)

    Paper:
    Addressing Wearable Sleep Tracking Inequity: A New Dataset and Novel Methods for a Population with Sleep Disorders
    https://raw.githubusercontent.com/mlresearch/v248/main/assets/wang24a/wang24a.pdf

    References:
    ----------
    [1] https://onlinelibrary.wiley.com/doi/abs/10.1002/0471751723.ch1.
    [2] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3817449/.
    [3] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4400203/.
    [4] https://my.clevelandclinic.org/health/articles/11429-common-sleep-disorders.
    [5] https://www.nature.com/articles/s41746-020-0244-4.
    [6] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5624990/.
    [7] https://github.com/armiro/COVID-CXNet
    [8] https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
    [9] https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

    Args:
        root: Root directory containing the dataset files.
        tables: List of tables to load (default: ["dreams_features"]).
        dataset_name: Optional name of the dataset. Defaults to "dreamt".
        config_path: Optional path to the configuration file. If not provided,
            uses "dreamt.yaml" as default.

    Attributes:
        root: Root directory of the dataset.
        tables: List of loaded tables.
        dataset_name: Name of the dataset.
        config_path: Path to the configuration file.
        patients: Dictionary of processed patient records containing:
            - patient_id: Unique patient identifier
            - records: List of sleep epochs with:
                * record_id: Unique epoch identifier
                * features: Physiological features (numpy array)
                * label: Sleep stage (0-4)

    Examples:
        >>> from pyhealth.datasets import DreamtDataset
        >>> dataset = DreamtDataset(
        ...     root="/path/to/dreamt_data"
        ... )
        >>> dataset.stat()
        >>> samples = dataset.set_task()
        >>> print(samples[0])

    Note:
        The dataset requires pre-processed feature files in CSV format containing:
        - sid: Patient/study identifier
        - Sleep_Stage: Annotated sleep stage (0-4)
        - Various physiological features
        - Respiratory event markers
        - Demographic information
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = (
                Path(__file__).parent / "configs" / "dreamt_e4.yaml"
            )
        if not os.path.exists(os.path.join(root, "all_patients_domain_features.csv")):
            self.prepare_metadata(root)
        default_tables = ["dreamt_features"]
        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "dreamt",
            config_path=config_path,
        )
        # self.patients = self.process()
        return


    def prepare_metadata(self, root: str) -> None:
        """Prepare metadata for the DREAMT dataset by:
        1. Loading raw feature files
        2. Applying quality threshold 0.2
        3. Processing and renaming features
        4. Saving combined metadata to CSV

        Args:
            root: Root directory containing the dataset files.
        """
        # Set paths
        info_dir = os.path.join(root, "participant_info.csv")
        feature_df_dir = os.path.join(root, "features_df/")
        quality_df_dir = os.path.join(root, "quality_scores_per_subject.csv")
        data_folder = os.path.join(root,"data")
        
        # Get subject IDs
        all_sids = pd.read_csv(info_dir).SID.to_list()
        logger.info(f"Found {len(all_sids)} subjects in participant info")

        # Extract and aggregated domain features for E4 data
        for sid in all_sids:
            print(sid)
            try:
                extract_domain_features(
                    sid, data_folder=data_folder, segment_seconds=30, save_folder_dir= feature_df_dir
                )
            except:
                print("ERROR")

        # Verify feature files exist
        for sid in all_sids:
            feature_file = os.path.join(feature_df_dir, f"{sid}_domain_features_df.csv")
            if not os.path.exists(feature_file):
                logger.warning(f"Missing feature file for subject {sid}")

        # Calculate quality score for each participant
        calculate_qaulity_score(feature_df_dir)

        # Run data preparation pipeline
        logger.info("Processing raw data with quality threshold 0.2")
        clean_df, new_features, good_quality_sids = data_preparation(
            threshold=0.2,
            quality_df_dir=quality_df_dir,
            features_dir=feature_df_dir,
            info_dir=info_dir
        )

        # Split and process data
        SW_df, final_features = split_data(clean_df, good_quality_sids, new_features)
        logger.info(f"Processed data with {len(SW_df)} records")

        # Rename columns for consistency
        SW_df = SW_df.rename(columns={
            'HRV_CD': 'HRV_CD_UPPER',
            'HRV_Cd': 'HRV_Cd_MIXED',
            'rolling_var_HRV_CD': 'rolling_var_HRV_CD_UPPER',
            'rolling_var_HRV_Cd': 'rolling_var_HRV_Cd_MIXED',
            'gaussian_HRV_Cd_1st_derivative': 'gaussian_HRV_Cd_1st_derivative_MIXED',
            'gaussian_HRV_CD_1st_derivative': 'gaussian_HRV_CD_1st_derivative_UPPER',
            'raw_HRV_CD_1st_derivative': 'raw_HRV_CD_1st_derivative_UPPER',
            'raw_HRV_Cd_1st_derivative': 'raw_HRV_Cd_1st_derivative_MIXED'
        })

        # Convert and save timestamp
        SW_df['timestamp_start'] = pd.to_datetime(SW_df['timestamp_start'], unit='s')
        
        # Save final metadata
        output_path = os.path.join(root, "all_patients_domain_features.csv")
        SW_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed metadata to {output_path}")
        return
    

    @property
    def default_task(self) -> DREAMTE4SleepingStageClassification:
        """Returns the default task for this dataset.

        Returns:
             DREAMTE4SleepingStageClassification: The default classification task.
        """
        return DREAMTE4SleepingStageClassification()



if __name__ == "__main__":
    # Example test case for the DREAMTE4Dataset.
    # root = "dreamt-dataset-for-real-time-sleep-stage-estimation-using-multisensor-wearable-technology-1.0.0"
    root = "dreamt_e4" # small test sample data subset
    dreamt_dataset = DREAMTE4Dataset(
        root=root,
        tables=["dreams_features"]
    )

    task = DREAMTE4SleepingStageClassification()
    dreamt_samples = dreamt_dataset.set_task(task)
    print(dreamt_samples.input_schema)
    print(dreamt_samples.output_schema)
    print(len(dreamt_samples))
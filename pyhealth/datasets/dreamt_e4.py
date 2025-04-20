import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from pyhealth.tasks.dreamt_sleeping_stage_classification import DREAMTE4SleepingStageClassification
from pyhealth.datasets.base_dataset import BaseDataset
from pyhealth.datasets.dreamt_e4_feature_engineering import *


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
                * label: Sleep stage ("P": 1, "N": 0, "R": 0, "W": 1, "Missing": np.nan)

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
        - Sleep_Stage: Annotated sleep stage ("P": 1, "N": 0, "R": 0, "W": 1, "Missing": np.nan)
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
        calculate_qaulity_score(root, feature_df_dir)

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
    # Running DREAMTE4Dataset.

    #make sure the root is correct
    # root = "dreamt-dataset-for-real-time-sleep-stage-estimation-using-multisensor-wearable-technology-1.0.0"
    # dreamt_dataset = DREAMTE4Dataset(
    #     root=root
    # )

    # task = DREAMTE4SleepingStageClassification()
    # dreamt_samples = dreamt_dataset.set_task(task)
    # print(dreamt_samples.input_schema)
    # print(dreamt_samples.output_schema)
    # print(len(dreamt_samples))

    # here is the test case using dummy data
    import yaml
    import polars as pl
    from pathlib import Path
    from unittest.mock import patch

    cfg_path = Path(__file__).parent / "configs" / "dreamt_e4.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    table_cfg = cfg["tables"]["dreamt_features"]
    attributes = table_cfg["attributes"]

    def generate_dummy_all_patients_domain_features(
        num_sids: int = 80,
        records_per_sid: int = 10,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Same generator as in the unittest example—358 domain + 13 static cols."""
        np.random.seed(seed)
        rows = []
        for pid in range(1, num_sids+1):
            sid = f"S{pid:03d}"
            # per‐patient constants
            bmi = float(np.clip(np.random.normal(25,5), 15, 50))
            obesity = 1.0 if bmi >= 30 else 0.0
            circ_vals = np.random.rand(3)
            for rec in range(records_per_sid):
                row = {}
                for col in attributes:
                    if col == "sid":
                        row[col] = sid
                    elif col == "Sleep_Stage":
                        row[col] = float(np.random.randint(0,5))
                    elif col in {
                        "Central_Apnea","Obstructive_Apnea",
                        "Multiple_Events","Hypopnea","artifact"
                    }:
                        row[col] = float(np.random.binomial(1, 0.1))
                    elif col == "AHI_Severity":
                        row[col] = float(np.random.uniform(0,10))
                    elif col == "Obesity":
                        row[col] = obesity
                    elif col == "BMI":
                        row[col] = bmi
                    elif col in {"circadian_decay","circadian_linear","circadian_cosine"}:
                        # use those same 3 circ_vals
                        row[col] = float(circ_vals[{"circadian_decay":0,"circadian_linear":1,"circadian_cosine":2}[col]])
                    elif col == "timestamp_start":
                        row[col] = f"2025-01-19T00:00:{rec:02d}"
                    else:
                        # everything else is a domain feature → random float
                        row[col] = float(np.random.rand())
                rows.append(row)
        return pd.DataFrame(rows, columns=attributes)


    dummy_df = generate_dummy_all_patients_domain_features()
    dummy_pl_lazy = pl.from_pandas(dummy_df).lazy()
    dummy_pl_eager = pl.from_pandas(dummy_df)

    # 2) Monkey‑patch exists + read_csv/scan_csv
    with patch.object(os.path, "exists", return_value=True), \
         patch("pandas.read_csv", return_value=dummy_df), \
         patch("polars.read_csv", lambda *args, **kwargs: dummy_pl_eager), \
         patch("polars.scan_csv", lambda *args, **kwargs: dummy_pl_lazy):

        # 3) Now run exactly as normal
        root = "does/not/matter"
        ds   = DREAMTE4Dataset(root=root)
        task = DREAMTE4SleepingStageClassification()
        samples = ds.set_task(task)

    print("Loaded samples:", len(samples))
    print("One sample shape:", samples[0]['features'].shape)
    print("Dataset input schema:", samples.input_schema)
    print("Dataset output schema:", samples.output_schema)

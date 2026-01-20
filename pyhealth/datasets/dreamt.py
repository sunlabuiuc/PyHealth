import logging
import os
from pathlib import Path
from typing import  Optional, Union, List, Dict, Any

import pandas as pd
import numpy as np
import warnings
from pyhealth.datasets import BaseDataset
from scipy.signal import convolve, windows

logger = logging.getLogger(__name__)

class DREAMTDataset(BaseDataset):
    """
    Base Dataset for Real-time sleep stage EstimAtion using Multisensor wearable Technology (DREAMT)

    Dataset accepts current versions of DREAMT (1.0.0, 1.0.1, 2.0.0, 2.1.0), available at:
    https://physionet.org/content/dreamt/

    DREAMT includes wrist-based wearable and polysomnography (PSG) sleep data from 100 participants
    recruited from the Duke University Health System (DUHS) Sleep Disorder Lab. This includes 
    wearable signals, PSG signals, sleep labels, and clinical data related to sleep health and disorders. 

    Citations:
    ---------
    When using this dataset, please cite:

    Wang, K., Yang, J., Shetty, A., & Dunn, J. (2025). DREAMT: Dataset for Real-time sleep stage EstimAtion 
    using Multisensor wearable Technology (version 2.1.0). PhysioNet. RRID:SCR_007345. 
    https://doi.org/10.13026/7r9r-7r24

    Will Ke Wang, Jiamu Yang, Leeor Hershkovich, Hayoung Jeong, Bill Chen, Karnika Singh, Ali R Roghanizad, 
    Md Mobashir Hasan Shandhi, Andrew R Spector, Jessilyn Dunn. (2024). Proceedings of the fifth 
    Conference on Health, Inference, and Learning, PMLR 248:380-396.

    Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). 
    PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex 
    physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.

    Note: 
    ---------
    Dataset follows file and folder structure of dataset version, looks for participant_info.csv and data folders, 
    so root path should be version downloaded, example: root = ".../dreamt/1.0.0/" or ".../dreamt/2.0.0/"

    Args:
        root: root directory containing the dataset files
        dataset_name: optional name of dataset, defaults to "dreamt_sleep"
        config_path: optional configuration file, defaults to "dreamt.yaml"
    
    Attributes:
        root: root directory containing the dataset files
        dataset_name: name of dataset
        config_path: path to configuration file

    Examples:
        >>> from pyhealth.datasets import DREAMTDataset
        >>> dataset = DREAMTDataset(root = "/path/to/dreamt/data/version")
        >>> dataset.stats()
        >>>
        >>> # Get all patient ids
        >>> unique_patients = dataset.unique_patient_ids
        >>> print(f"There are {len(unique_patients)} patients")
        >>>
        >>> # Get single patient data
        >>> patient = dataset.get_patient("S002")
        >>> print(f"Patient has {len(patient.data_source)} event")
        >>>
        >>> # Get event
        >>> event = patient.get_events(event_type="dreamt_sleep")
        >>>
        >>> # Get Apnea-Hypopnea Index (AHI)
        >>> ahi = event[0].ahi
        >>> print(f"AHI is {ahi}")
        >>> 
        >>> # Get 64Hz sleep file path
        >>> file_path = event[0].file_64hz
        >>> print(f"64Hz sleep file path: {file_path}") 
    """

    def __init__(
            self,
            root: str,
            dataset_name: Optional[str] = None,
            config_path: Optional[str] = None,
    ) -> None:
        if config_path is None:
            logger.info("No config provided, using default config")
            config_path = Path(__file__).parent / "configs" / "dreamt.yaml"
        
        metadata_file = Path(root) / "dreamt-metadata.csv"

        if not os.path.exists(metadata_file):
            logger.info(f"{metadata_file} does not exist")
            self.prepare_metadata(root)
        
        default_tables = ["dreamt_sleep"]

        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "dreamt_sleep",
            config_path=config_path
        )
    
    def get_patient_file(self, patient_id: str, root: str, file_path: str) -> Union[str | None]:
        """
        Returns file path of 64Hz and 100Hz data for a patient, or None if no file found

        Args:
            patient_id: patient identifier
            root: root directory containing the dataset files
            file_path: path to location of 64Hz or 100Hz file
            
        Returns:
            file: path to file location or None if no file found
        """
        
        if file_path == "data_64Hz" or file_path == "data":
            file = Path(root) / f"{file_path}" / f"{patient_id}_whole_df.csv"
    
        if file_path == "data_100Hz":
            file = Path(root) / f"{file_path}" / f"{patient_id}_PSG_df.csv"

        if not os.path.exists(str(file)):
            logger.info(f"{file} not found")
            file = None
        
        return file


    def prepare_metadata(self, root: str) -> None:
        """
        Prepares metadata csv file for the DREAMT dataset by performing the following:
        1. Obtain clinical data from participant_info.csv file
        2. Process file paths based on patients found in clinical data
        3. Organize all data into a single DataFrame
        4. Save the processed DataFrame to a CSV file

        Args:
            root: root directory containing the dataset files
        """

        output_path = Path(root) / "dreamt-metadata.csv"

        # Obtain patient clinical data
        participant_info_path = Path(root) / "participant_info.csv"
        participant_info = pd.read_csv(participant_info_path)


        # Determine folder structure, assign associated file paths based on folder structure
        all_folders = [item.name for item in Path(root).iterdir() if item.is_dir()]
        file_path_64hz = "data_64Hz" if "data_64Hz" in all_folders else "data"
        file_path_100hz = "data_100Hz"

        # Determine paths for 64Hz and 100Hz files for each patient
        participant_info['file_64hz'] = participant_info['SID'].apply(
            lambda sid: self.get_patient_file(sid, root, file_path_64hz)
        )
        participant_info['file_100hz'] = participant_info['SID'].apply(
            lambda sid: self.get_patient_file(sid, root, file_path_100hz)
        )

        # Remove "%" from mean SaO2 recording
        participant_info['Mean_SaO2'] = participant_info['Mean_SaO2'].str[:-1]

        # Format columns to align with BaseDataset
        participant_info = participant_info.rename(columns = {
            'SID': 'patient_id',
            'AGE': 'age',
            'GENDER': 'gender',
            'BMI': 'bmi',
            'OAHI': 'oahi',
            'AHI': 'ahi',
            'Mean_SaO2': 'mean_sao2',
            'Arousal Index': 'arousal_index',
            "MEDICAL_HISTORY": 'medical_history',
            "Sleep_Disorders": 'sleep_disorders'
        })
        
        # Create csv
        participant_info.to_csv(output_path, index=False)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class DREAMTSleepWakeDataset(BaseDataset):
    """
    PyHealth Dataset wrapper for DREAMT processed through your ML pipeline.

    This dataset uses the `data_preparation` function to process raw CSVs
    into a cleaned dataframe and constructs samples suitable for sleep/wake classification.
    """
    def __init__(self, root: str, config_path: Optional[str] = None,threshold: float = 0.2):
        """Initialize the DREAMT Sleep/Wake Dataset.

        Will run the full data preparation pipeline on the raw CSVs.

        Args:
            root: Path to the dataset root folder.
            config_path: Optional path to a YAML configuration file.
            threshold: Quality score threshold for excluding subjects.
        """
        if config_path is None:
            logger.info("No config provided, using default config")
            config_path = Path(__file__).parent / "configs" / "dreamt.yaml"

        self.root = Path(root)
        self.threshold = threshold

        # Run the full data preparation pipeline
        logger.info("Running data preparation pipeline...")
        features_dir = self.root / "dataset_sample/features_df/"
        info_dir = self.root / "dataset_sample/participant_info.csv"
        quality_df_dir = self.root / "results/quality_scores_per_subject.csv"

        info_df = pd.read_csv(info_dir, index_col="SID")

        # Rename columns to lowercase
        info_df = info_df.rename(columns={"AGE": "age", "GENDER": "gender"})

        self.clean_df, self.new_features, self.good_quality_sids = self.data_preparation(
            threshold=self.threshold,
            quality_df_dir=str(quality_df_dir),
            features_dir=str(features_dir),
            info_dir=str(info_dir)
        )

        if 'sid' in self.clean_df.columns:
            self.clean_df = self.clean_df.rename(columns={'sid': 'patient_id'})
        if 'SID' in self.clean_df.columns:
            self.clean_df = self.clean_df.rename(columns={'sid': 'patient_id'})
        self.clean_df = self.clean_df.merge(
            info_df[['age', 'gender']],
            left_on='patient_id',
            right_index=True,
            how='left'
        )

        self.create_metadata_files()

        super().__init__(
            root=root,
            tables=["dreamt_sleep"],  # just a dummy table name for PyHealth
            dataset_name = "dreamt_sleepwake",
            config_path=config_path,
        )

        logger.info(
            f"Data preparation complete: {len(self.clean_df)} rows, "
            f"{len(self.new_features)} features, "
            f"{len(self.good_quality_sids)} good quality subjects."
        )

    def create_metadata_files(self):
        """Create CSV metadata files for full dataframe, feature-only dataframe, and subject IDs.

        Docs:
            - PyHealth metadata convention: see `pyhealth.datasets` examples

        Returns:
            None
        """
        # Ensure root is a Path object
        self.root = Path(self.root)

        # 1️⃣ Full cleaned dataframe metadata
        if "sid" in self.clean_df.columns:
            self.clean_df = self.clean_df.rename(columns={"sid": "patient_id"})
        clean_df_file = self.root / "dreamt-metadata.csv"
        self.clean_df.to_csv(clean_df_file, index=False)

        # 2️⃣ Only feature columns + patient_id
        features_df = self.clean_df[self.new_features + ["patient_id"]]
        features_file = self.root / "features_metadata.csv"
        features_df.to_csv(features_file, index=False)

        # 3️⃣ Only subject IDs (patient_id)
        subjects_df = pd.DataFrame({"patient_id": self.good_quality_sids})
        subjects_file = self.root / "subjects_metadata.csv"
        subjects_df.to_csv(subjects_file, index=False)

    def build_samples(self) -> List[Dict[str, Any]]:
        """Convert clean_df to PyHealth samples.

        Each sample contains:
            - patient_id
            - record_id
            - features (numpy array)
            - label (sleep stage: 0/1)

        Returns:
            A list of sample dictionaries.
        """
        samples = []

        for idx, row in self.clean_df.iterrows():
            pid = row["sid"]
            features = row[self.new_features].values.astype(np.float32)
            label = int(row["Sleep_Stage"])
            samples.append(
                {
                    "patient_id": pid,
                    "record_id": f"{pid}_{idx}",
                    "features": features,
                    "label": label
                }
            )

        return samples
    

    def data_preparation(self, threshold, quality_df_dir, features_dir, info_dir):
        """
        Prepare the data for modeling by using data preparation functions

        Docs:
            - DREAMT dataset processing pipeline

        Args:
        -------
        threshold: Quality score threshold for subjects.
        quality_df_dir: Path to CSV of subject quality scores.
        features_dir: Path to folder of feature CSVs.
        info_dir: Path to CSV with demographic/clinical info.

        Returns:
        -------
        clean_df : pandas DataFrame
            The cleaned dataframe after applying all preprocessing steps.
        new_features : list
            A list of the names of the features that were retained in the cleaned DataFrame.
        good_quality_sids : list of str
            A list of subject IDs that met the quality score threshold and were included 
            in the analysis.
        """
        nan_feature_names = [
            "HRV_LF",
            "HRV_LFHF",
            "HRV_LFn",
            "HRV_MSEn",
            "HRV_CMSEn",
            "HRV_RCMSEn",
            "LF_frequency_power",
            "LF_normalized_power",
        ]

        circadian_features = [
            "circadian_decay",
            "circadian_linear",
            "circadian_cosine",
            "timestamp_start",
        ]

        label_names = [
            "Sleep_Stage",
            "Obstructive_Apnea",
            "Central_Apnea",
            "Hypopnea",
            "Multiple_Events",
            "artifact",
        ]
        info_df = pd.read_csv(info_dir)
        all_subjects_fe_df, good_quality_sids = self.load_data_to_df(
            threshold, quality_df_dir, info_df, features_dir, nan_feature_names, 
            label_names, circadian_features
    )
        clean_df, new_features = self.clean_features(
            all_subjects_fe_df, info_df, nan_feature_names, label_names
        )
        return clean_df, new_features, good_quality_sids

    def load_data_to_df(
        self, threshold, quality_df_dir, info_df, features_dir, nan_feature_names, label_names, circadian_features
    ):
        """
        Loads and processes feature data from CSV files for subjects meeting a 
        quality score threshold, applying several preprocessing steps including 
        rolling standard deviations, Gaussian filtering, and derivative calculation. 
        The function also classifies subjects based on Apnea-Hypopnea Index (AHI) 
        and Body Mass Index (BMI) into predefined categories.

        Args:
        -----
        threshold: Float threshold for subject quality.
        quality_df_dir: CSV file with quality scores.
        info_df: DataFrame with participant info (indexed by SID).
        features_dir: Folder containing feature CSVs.
        nan_feature_names: List of features considered NaN.
        label_names: List of label columns.
        circadian_features: List of circadian-related features.

        Returns:
        -------
        all_subjects_fe_df : pandas dataFrame
            A DataFrame containing the processed features for all subjects meeting the 
            quality threshold.
        good_quality_sids : list of str
            A list of subject IDs that met the quality score threshold and were included 
            in the analysis.
        """
        # load quality scores
        quality_df = pd.read_csv(quality_df_dir)
        good_quality_sids = quality_df.loc[
            quality_df.percentage_excludes < float(threshold), "sid"
        ].to_list()

        # load demographic info
        info_df.index = info_df.SID

        # Read example from one subject for further processing
        #path = str(features_dir) + str(info_df.SID[0]) + '_domain_features_df.csv'
        path = Path(features_dir) / f"{info_df.SID[0]}_domain_features_df.csv"
        example_df = pd.read_csv(path)

        # select features
        feature_names = [
            f
            for f in example_df.columns.tolist()
            if f not in nan_feature_names + label_names + ["sid"]
        ]

        # select physiological features
        physiological_features = [f for f in feature_names if f not in circadian_features]

        # create dataframe for all the subjects' features
        example_df = self.rolling_stds(example_df, physiological_features, window_size=10)
        example_df = self.gaussian_filtering(
            example_df, physiological_features, kernel_size=20, std_dev=100
        )
        example_df = self.add_derivatives(example_df, physiological_features)

        all_subjects_fe_df = pd.DataFrame(columns=example_df.columns)
        for sid in good_quality_sids:
            path = Path(features_dir) / f"{info_df.SID[0]}_domain_features_df.csv"
            sid_df = pd.read_csv(path)
            sid_df = self.rolling_stds(sid_df, physiological_features, window_size=10)
            sid_df = self.gaussian_filtering(
                sid_df, physiological_features, kernel_size=20, std_dev=100
            )
            sid_df = self.add_derivatives(sid_df, physiological_features)

            # add apnea target
            subject_AHI = int(info_df.loc[sid, "AHI"])
            if subject_AHI < 5:
                sid_df["AHI_Severity"] = 0
            elif 5 <= subject_AHI < 15:
                sid_df["AHI_Severity"] = 1
            elif 15 <= subject_AHI < 30:
                sid_df["AHI_Severity"] = 2
            else:
                sid_df["AHI_Severity"] = 3

            # add BMI target
            subject_BMI = info_df.loc[sid, "BMI"]
            if subject_BMI >= 35:
                sid_df["Obesity"] = 1
            else:
                sid_df["Obesity"] = 0

            sid_df = sid_df.loc[:sid_df[sid_df['Sleep_Stage'].isin(["N1", "N2", "N3", "R", "W"])].last_valid_index(), :]

            all_subjects_fe_df = pd.concat([all_subjects_fe_df, sid_df], ignore_index=True)
        return all_subjects_fe_df, good_quality_sids
   
    def clean_features(self, all_subjects_fe_df, info_df, nan_feature_names, label_names):
        """
        Cleans the feature dataframe by updating feature names, mapping sleep stages,
        replacing infinite values with NaN, deleting features with excessive missing values,
        and merging additional demographic information. It prepares the data for further 
        analysis by filtering out unnecessary columns and rows with missing values, and 
        returns a cleaned dataframe along with a list of the names of the features that 
        were retained.

        Args:
        -----
        all_subjects_fe_df: DataFrame of all processed features.
        info_df: DataFrame with demographic/clinical info.
        nan_feature_names: Features to treat as NaN.
        label_names: Label columns.

        Returns:
        -------
        clean_df : pandas DataFrame
            The cleaned dataframe after applying all preprocessing steps.
        new_features : list
            A list of the names of the features that were retained in the cleaned DataFrame.
        """
        # update features
        updated_feature_names = [
            f
            for f in all_subjects_fe_df.columns.tolist()
            if f not in nan_feature_names + label_names + ["sid"]
        ]

        # get feature dataframe
        df = all_subjects_fe_df.loc[
            :,
            updated_feature_names + label_names + ["sid"],
        ]

        df.Sleep_Stage = df.Sleep_Stage.map(
            {
                "N1": "N",
                "N2": "N",
                "W": "W",
                "N3": "N",
                "P": "P",
                "R": "R",
                "Missing": "Missing",
            }
        )
        # replace inf
        df = df.replace([np.inf, -np.inf], np.nan)

        # delete features if contains too many nan values
        na_count_df = df.isna().sum()
        features_to_delete = na_count_df[na_count_df > 2000].index.to_list()
        cleaned_feature_names = [
            f for f in updated_feature_names if f not in features_to_delete
        ]

        # select feature columns
        df = df.loc[:, cleaned_feature_names + label_names + ["sid"]]
        # drop columns with nan
        df = df.dropna(how="any", axis=0)

        df = pd.merge(df, info_df.loc[:, ["BMI"]], left_on="sid", right_index=True)

        map_stage_to_num = {"P": 1, "N": 0, "R": 0, "W": 1, "Missing": np.nan}
        df["Sleep_Stage"] = df["Sleep_Stage"].map(map_stage_to_num)
        clean_df = df.dropna()

        new_features = clean_df.columns.to_list()
        new_features.remove("sid")
        new_features.remove("Sleep_Stage")
        new_features.remove("Central_Apnea")
        new_features.remove("Obstructive_Apnea")
        new_features.remove("Multiple_Events")
        new_features.remove("Hypopnea")
        new_features.remove("AHI_Severity")
        new_features.remove("Obesity")
        new_features.remove("BMI")
        new_features.remove("circadian_decay")
        new_features.remove("circadian_linear")
        new_features.remove("circadian_cosine")
        new_features.remove("timestamp_start")

        return clean_df, new_features
    
    def rolling_stds(self, df, columns, window_size=20):
        """Calculate rolling standard deviations for the given columns in the DataFrame.

        Parameters
        ----------
        df : pandas DataFrame
            The DataFrame for which to calculate rolling standard deviations.
        columns : list
            The columns for which to calculate rolling standard deviations.
        window_size : int
            The size of the rolling window.

        Returns
        -------
        df : pandas DataFrame
            The DataFrame with the rolling standard deviations added.
        """
        for column in columns:
            df["rolling_var_{}".format(column)] = (
                df[column].rolling(window=window_size, min_periods=1).var()
            )
        return df
    
    def gaussian_filtering(self, df, columns, kernel_size=40, std_dev=100):
        """Perform Gaussian filtering on the given DataFrame.

        Args:
        -----
        df : pandas DataFrame
            The DataFrame to be filtered.
        columns : list
            The columns to be filtered.
        kernel_size : int
            The size of the kernel.
        std_dev : float
            The standard deviation of the kernel.

        Returns:
        -------
        df : pandas DataFrame
            The filtered DataFrame.
        """
        for column in columns:
            interpolated_series = self.missingness_imputation(
                self.apply_gaussian_filter(df[column], kernel_size, std_dev)
            )
            df["gaussian_{}".format(column)] = interpolated_series
        #         df["gaussian_diff_{}".format(column)] = interpolated_series - df[column]
        return df
    def apply_gaussian_filter(self, data, kernel_size, std_dev):
        """Apply a Gaussian filter to the given data.

        Args:
        -----
        data : array-like
            The data to be filtered.
        kernel_size : int
            The size of the kernel.
        std_dev : float
            The standard deviation of the kernel.

        Returns:
        -------
        filtered_data : array
            The filtered data.
        """
        kernel = windows.gaussian(kernel_size, std_dev, sym=True)
        kernel /= np.sum(kernel)
        filtered_data = convolve(data, kernel, mode="same")
        return filtered_data

    def missingness_imputation(self, data):
        """Perform missingness imputation on the given data.

        Args:
        ------
        data : array-like
            The data to be imputed.

        Returns:
        -------
        interpolated_series : pandas Series
            The imputed data.
        """

        indices = np.arange(len(data))
        series = pd.Series(data, index=indices)
        interpolated_series = series.interpolate(method="linear")
        return interpolated_series
    
    def add_derivatives(self, df, features):
        """Add first and second derivatives to the given features in the DataFrame.

        Args:
        -----
        df : pandas DataFrame
            The DataFrame to which to add the derivatives.
        features : list
            The features to which to add the derivatives.
        

        Returns:
        -------
        df : pandas DataFrame
            The DataFrame with the derivatives added.
        """
        for feature in features:
            # First derivative
            first_derivative_column = "gaussian_" + feature + "_1st_derivative"
            df[first_derivative_column] = np.gradient(df["gaussian_" + feature])

            # Second derivative
            raw_derivative_column = "raw_" + feature + "_1st_derivative"
            df[raw_derivative_column] = df[feature].diff()
        return df
    

    def convert_to_samples(self, clean_df, feature_list):
        """Convert clean_df to PyHealth-style samples with detailed labels.

        Args:
            clean_df: Cleaned DataFrame.
            feature_list: List of feature columns.

        Returns:
            List of sample dictionaries with patient_id, record_id, features, and label dict.
        """
        samples = []

        for idx, row in clean_df.iterrows():
            pid = str(row["sid"])
            rid = f"{pid}_{idx}"

            features = {f: row[f] for f in feature_list}

            labels = {
                "Sleep_Stage": row["Sleep_Stage"],
                "Central_Apnea": row["Central_Apnea"],
                "Obstructive_Apnea": row["Obstructive_Apnea"],
                "Hypopnea": row["Hypopnea"],
                "Multiple_Events": row["Multiple_Events"],
                "AHI_Severity": row["AHI_Severity"],
                "Obesity": row["Obesity"],
                "BMI": row["BMI"],
            }

            samples.append({
                "patient_id": pid,
                "record_id": rid,
                "features": features,
                "label": labels
            })

        return samples
    


    # ----------------------------------------------------------
    # PASS-THROUGH PyHealth API
    # ----------------------------------------------------------
    def split(self, *args, **kwargs):
        """Forward to underlying SampleDataset"""
        return self.sample_dataset.split(*args, **kwargs)

    def __len__(self):
        return len(self.sample_dataset)

    def __getitem__(self, idx):
        return self.sample_dataset[idx]
    

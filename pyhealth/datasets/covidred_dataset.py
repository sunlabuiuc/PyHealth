"""
COVID-RED Dataset Loader for PyHealth

This module implements a dataset loader for the COVID-RED (Remote Early Detection 
of SARS-CoV-2 infections) dataset from Utrecht University.

Dataset: https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/FW9PO7
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Literal


class COVIDREDDataset(Dataset):
    """
    COVID-RED Dataset for early detection of COVID-19 from wearable device data.
    
    The COVID-RED dataset contains wearable device measurements (heart rate, steps, sleep)
    from participants during the COVID-19 pandemic, collected to enable early detection
    of SARS-CoV-2 infections before symptom onset.
    
    Parameters
    ----------
    root : str
        Root directory containing the COVID-RED dataset files.
        Expected files (from DataverseNL download):
        - bc_20230515.csv (baseline characteristics)
        - ct_20230515.csv (COVID-19 test results)
        - cv_20230515.csv (COVID-19 vaccination)
        - dm_20230515.csv (daily measurements - heart rate, steps, etc.)
        - field_options.csv (field value mappings)
        - ho_20230515.csv (hospitalization)
        - hu_20230515.csv (healthcare utilization)
        - ie_20230515.csv (illness episodes)
        - mh_20230515.csv (medical history)
        - ov_20230515.csv (overview/participant info)
        - pcr_20230515.csv (PCR test results)
        - sc_20230515.csv (symptom checklist)
        - ser_20230515.csv (serology results)
        - si_20230515.csv (symptom information)
        - variable_descriptions.csv (data dictionary)
        - wd_20230515.csv (wearable device data)
    
    split : Literal["train", "test", "all"], default="train"
        Which split of the data to use.
    
    window_days : int, default=7
        Number of days to include in each sample window.
    
    task : Literal["detection", "prediction"], default="detection"
        Task type:
        - "detection": Classify COVID-19 positive vs negative
        - "prediction": Predict COVID-19 onset before symptom onset
    
    transform : Optional[Callable], default=None
        Optional transform to be applied on a sample.
    
    random_seed : int, default=42
        Random seed for train/test split reproducibility.
    
    Examples
    --------
    >>> from pyhealth.datasets import COVIDREDDataset
    >>> dataset = COVIDREDDataset(
    ...     root="/path/to/covidred",
    ...     split="train",
    ...     window_days=7,
    ...     task="prediction"
    ... )
    >>> print(f"Dataset size: {len(dataset)}")
    >>> sample = dataset[0]
    >>> print(f"Features shape: {sample['features'].shape}")
    
    Notes
    -----
    Download from: https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/FW9PO7
    """
    
    def __init__(
        self,
        root: str,
        split: Literal["train", "test", "all"] = "train",
        window_days: int = 7,
        task: Literal["detection", "prediction"] = "detection",
        transform: Optional[Callable] = None,
        random_seed: int = 42,
    ):
        self.root = root
        self.split = split
        self.window_days = window_days
        self.task = task
        self.transform = transform
        self.random_seed = random_seed
        
        # Feature names
        self.feature_names = [
            "resting_hr_mean",
            "resting_hr_std", 
            "resting_hr_min",
            "resting_hr_max",
            "steps_total",
            "steps_mean_hourly",
            "sleep_duration_hours",
            "sleep_efficiency",
        ]
        
        # Load and process the dataset
        self._load_data()
        self._create_samples()
        
    def _load_data(self):
        """Load CSV files from the COVID-RED dataset directory."""
        # Check if required files exist
        required_files = {
            'daily_measurements': 'dm_20230515.csv',
            'wearable_data': 'wd_20230515.csv',
            'covid_tests': 'ct_20230515.csv',
            'symptom_info': 'si_20230515.csv',
            'illness_episodes': 'ie_20230515.csv',
            'overview': 'ov_20230515.csv',
        }
        
        missing_files = []
        for name, filename in required_files.items():
            file_path = os.path.join(self.root, filename)
            if not os.path.exists(file_path):
                missing_files.append(filename)
        
        if missing_files:
            raise FileNotFoundError(
                f"Required files not found in {self.root}:\n"
                f"{', '.join(missing_files)}\n\n"
                f"Please download the COVID-RED dataset from:\n"
                f"https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/FW9PO7\n\n"
                f"Expected files:\n" + 
                "\n".join(f"  - {f}" for f in required_files.values())
            )
        
        # Load main data files
        print(f"Loading COVID-RED dataset from {self.root}...")
        
        self.daily_measurements = pd.read_csv(os.path.join(self.root, 'dm_20230515.csv'))
        self.wearable_data = pd.read_csv(os.path.join(self.root, 'wd_20230515.csv'))
        self.covid_tests = pd.read_csv(os.path.join(self.root, 'ct_20230515.csv'))
        self.symptom_info = pd.read_csv(os.path.join(self.root, 'si_20230515.csv'))
        self.illness_episodes = pd.read_csv(os.path.join(self.root, 'ie_20230515.csv'))
        self.overview = pd.read_csv(os.path.join(self.root, 'ov_20230515.csv'))
        
        print(f"✓ Loaded {len(self.overview)} participants")
        print(f"✓ Daily measurements: {len(self.daily_measurements)} records")
        print(f"✓ Wearable data: {len(self.wearable_data)} records")
        
        # Convert date columns
        self._convert_dates()
        
    def _convert_dates(self):
        """Convert date columns to datetime format."""
        date_columns_map = {
            'daily_measurements': ['date', 'measurement_date'],
            'wearable_data': ['date', 'wear_date'],
            'covid_tests': ['test_date', 'result_date'],
            'symptom_info': ['symptom_date', 'onset_date'],
            'illness_episodes': ['start_date', 'end_date'],
        }
        
        for df_name, possible_cols in date_columns_map.items():
            df = getattr(self, df_name)
            for col in possible_cols:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        pass
    
    def _create_samples(self):
        """Create samples with sliding windows."""
        self.samples = []
        
        # Get unique participants
        id_col = self._find_id_column(self.overview)
        participants = self.overview[id_col].unique()
        
        # Split participants
        import numpy as np
        np.random.seed(self.random_seed)
        n_train = int(len(participants) * 0.7)
        shuffled = np.random.permutation(participants)
        
        if self.split == "train":
            selected = shuffled[:n_train]
        elif self.split == "test":
            selected = shuffled[n_train:]
        else:
            selected = participants
        
        print(f"\nCreating samples for {len(selected)} participants...")
        
        for participant_id in selected:
            self._create_participant_samples(participant_id)
        
        print(f"✓ Created {len(self.samples)} samples")
    
    def _find_id_column(self, df):
        """Find the participant ID column in a dataframe."""
        for col in ['participant_id', 'subject_id', 'id', 'user_id']:
            if col in df.columns:
                return col
        return df.columns[0]
    
    def _create_participant_samples(self, participant_id):
        """Create samples for a single participant."""
        id_col = self._find_id_column(self.daily_measurements)
        
        # Get participant data
        data = self.daily_measurements[
            self.daily_measurements[id_col] == participant_id
        ].copy()
        
        if len(data) == 0:
            return
        
        # Find date column
        date_col = None
        for col in ['date', 'measurement_date', 'day', 'record_date']:
            if col in data.columns:
                date_col = col
                break
        
        if not date_col:
            return
        
        data = data.sort_values(date_col)
        
        # Get COVID label
        covid_positive, symptom_date = self._get_covid_label(participant_id)
        
        # Create windows
        for i in range(len(data) - self.window_days + 1):
            window = data.iloc[i:i + self.window_days]
            
            window_start = window[date_col].iloc[0]
            window_end = window[date_col].iloc[-1]
            
            # Determine label
            if self.task == "detection":
                label = covid_positive
            else:  # prediction
                label = 0
                if covid_positive == 1 and symptom_date is not None:
                    if pd.notna(symptom_date) and pd.notna(window_end):
                        days_to_onset = (symptom_date - window_end).days
                        label = int(0 < days_to_onset <= 14)
            
            # Extract features
            features = self._extract_features(window)
            
            if features is not None:
                self.samples.append({
                    "participant_id": participant_id,
                    "window_start_date": window_start,
                    "window_end_date": window_end,
                    "features": features,
                    "label": label,
                })
    
    def _get_covid_label(self, participant_id):
        """Get COVID-19 label for a participant."""
        id_col = self._find_id_column(self.covid_tests)
        
        tests = self.covid_tests[self.covid_tests[id_col] == participant_id]
        
        # Check for positive result
        covid_positive = 0
        for col in ['test_result', 'result', 'pcr_result', 'outcome', 'positive']:
            if col in tests.columns and len(tests) > 0:
                results = tests[col].astype(str).str.lower()
                if any(r in ['positive', '1', 'true', 'pos'] for r in results):
                    covid_positive = 1
                    break
        
        # Get symptom onset
        symptom_date = None
        id_col_symptom = self._find_id_column(self.symptom_info)
        symptoms = self.symptom_info[self.symptom_info[id_col_symptom] == participant_id]
        
        if len(symptoms) > 0:
            for col in ['onset_date', 'symptom_date', 'start_date']:
                if col in symptoms.columns:
                    dates = symptoms[col].dropna()
                    if len(dates) > 0:
                        symptom_date = pd.to_datetime(dates.iloc[0])
                        break
        
        return covid_positive, symptom_date
    
    def _extract_features(self, window_data):
        """Extract features from a window."""
        feature_mapping = {
            'resting_hr_mean': ['hr_mean', 'heart_rate_mean', 'resting_hr', 'hr_avg'],
            'resting_hr_std': ['hr_std', 'heart_rate_std', 'hr_sd'],
            'resting_hr_min': ['hr_min', 'heart_rate_min'],
            'resting_hr_max': ['hr_max', 'heart_rate_max'],
            'steps_total': ['steps', 'step_count', 'daily_steps', 'total_steps'],
            'steps_mean_hourly': ['steps_per_hour', 'hourly_steps'],
            'sleep_duration_hours': ['sleep_hours', 'sleep_duration', 'total_sleep'],
            'sleep_efficiency': ['sleep_eff', 'sleep_quality'],
        }
        
        features = []
        
        for _, row in window_data.iterrows():
            day_features = []
            
            for feature_name in self.feature_names:
                value = 0.0
                possible_cols = feature_mapping.get(feature_name, [feature_name])
                
                for col in possible_cols:
                    if col in row.index and pd.notna(row[col]):
                        value = float(row[col])
                        break
                
                # Calculate derived features
                if feature_name == 'steps_mean_hourly' and value == 0.0:
                    for col in feature_mapping['steps_total']:
                        if col in row.index and pd.notna(row[col]):
                            value = float(row[col]) / 24.0
                            break
                
                day_features.append(value)
            
            features.append(day_features)
        
        try:
            tensor = torch.tensor(features, dtype=torch.float32)
            if tensor.shape == (self.window_days, len(self.feature_names)):
                return tensor
        except:
            pass
        
        return None
    
    def __len__(self):
        """Return the number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample."""
        sample = self.samples[idx].copy()
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def get_feature_names(self):
        """Return feature names."""
        return self.feature_names
    
    def get_label_distribution(self):
        """Return label distribution."""
        labels = [s["label"] for s in self.samples]
        return {
            "total_samples": len(labels),
            "positive_samples": sum(labels),
            "negative_samples": len(labels) - sum(labels),
            "positive_ratio": sum(labels) / len(labels) if labels else 0.0,
        }

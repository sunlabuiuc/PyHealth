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
        Expected files: 
        - heart_rate.csv (daily heart rate measurements)
        - steps.csv (daily step counts)
        - sleep.csv (daily sleep duration)
        - labels.csv (COVID-19 test results and symptom dates)
    
    split : Literal["train", "test", "all"], default="train"
        Which split of the data to use.
        - "train": Training set (70% of participants)
        - "test": Test set (30% of participants)
        - "all": All data
    
    window_days : int, default=7
        Number of days to include in each sample window.
    
    task : Literal["detection", "prediction"], default="detection"
        Task type:
        - "detection": Classify COVID-19 positive vs negative during illness period
        - "prediction": Predict COVID-19 onset before symptom onset (early detection)
    
    transform : Optional[Callable], default=None
        Optional transform to be applied on a sample.
    
    random_seed : int, default=42
        Random seed for train/test split reproducibility.
    
    Attributes
    ----------
    samples : list
        List of sample dictionaries containing features and labels.
    
    feature_names : list
        Names of features included in each sample.
    
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
    >>> print(f"Label: {sample['label']}")
    
    Notes
    -----
    The dataset must be manually downloaded from:
    https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/FW9PO7
    
    Citation:
    Olthof, A.W., Schut, A., van Beijnum, B.F. et al. (2021). 
    Remote Early Detection of SARS-CoV-2 infections (COVID-RED).
    DataverseNL. https://doi.org/10.34894/FW9PO7
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
        
        # Feature names for each data type
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
        """Load CSV files from the dataset directory."""
        # Check if required files exist
        required_files = ["heart_rate.csv", "steps.csv", "sleep.csv", "labels.csv"]
        for file in required_files:
            file_path = os.path.join(self.root, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Required file '{file}' not found in {self.root}. "
                    f"Please download the COVID-RED dataset from: "
                    f"https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/FW9PO7"
                )
        
        # Load data files
        self.heart_rate_df = pd.read_csv(os.path.join(self.root, "heart_rate.csv"))
        self.steps_df = pd.read_csv(os.path.join(self.root, "steps.csv"))
        self.sleep_df = pd.read_csv(os.path.join(self.root, "sleep.csv"))
        self.labels_df = pd.read_csv(os.path.join(self.root, "labels.csv"))
        
        # Convert date columns to datetime
        for df in [self.heart_rate_df, self.steps_df, self.sleep_df, self.labels_df]:
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
        
    def _create_samples(self):
        """Create samples with sliding windows."""
        self.samples = []
        
        # Get unique participants
        participants = self.labels_df["participant_id"].unique()
        
        # Split participants into train/test
        import numpy as np
        np.random.seed(self.random_seed)
        n_train = int(len(participants) * 0.7)
        shuffled_participants = np.random.permutation(participants)
        train_participants = shuffled_participants[:n_train]
        test_participants = shuffled_participants[n_train:]
        
        # Select participants based on split
        if self.split == "train":
            selected_participants = train_participants
        elif self.split == "test":
            selected_participants = test_participants
        else:  # "all"
            selected_participants = participants
        
        # Create samples for each participant
        for participant_id in selected_participants:
            self._create_participant_samples(participant_id)
    
    def _create_participant_samples(self, participant_id: int):
        """Create samples for a single participant."""
        # Get participant data
        hr_data = self.heart_rate_df[
            self.heart_rate_df["participant_id"] == participant_id
        ].sort_values("date")
        
        steps_data = self.steps_df[
            self.steps_df["participant_id"] == participant_id
        ].sort_values("date")
        
        sleep_data = self.sleep_df[
            self.sleep_df["participant_id"] == participant_id
        ].sort_values("date")
        
        label_info = self.labels_df[
            self.labels_df["participant_id"] == participant_id
        ].iloc[0]
        
        # Merge data on date
        merged = hr_data.merge(
            steps_data, on=["participant_id", "date"], how="outer"
        ).merge(
            sleep_data, on=["participant_id", "date"], how="outer"
        ).sort_values("date")
        
        # Fill missing values with forward fill then backward fill
        merged = merged.fillna(method="ffill").fillna(method="bfill")
        
        # Create sliding windows
        for i in range(len(merged) - self.window_days + 1):
            window_data = merged.iloc[i:i + self.window_days]
            
            # Determine label based on task type
            if self.task == "detection":
                # COVID-19 positive (1) or negative (0) during illness period
                label = int(label_info["covid_positive"])
            else:  # "prediction"
                # Early detection: predict COVID-19 onset
                # Check if window is before symptom onset
                if pd.notna(label_info.get("symptom_onset_date")):
                    symptom_date = pd.to_datetime(label_info["symptom_onset_date"])
                    window_end = window_data["date"].iloc[-1]
                    # Label as 1 if participant will develop COVID-19
                    # and window is before symptom onset
                    if label_info["covid_positive"] == 1:
                        days_to_onset = (symptom_date - window_end).days
                        # Pre-symptomatic period (1-14 days before onset)
                        label = int(0 < days_to_onset <= 14)
                    else:
                        label = 0
                else:
                    label = 0
            
            # Extract features
            features = self._extract_features(window_data)
            
            # Create sample
            sample = {
                "participant_id": participant_id,
                "window_start_date": window_data["date"].iloc[0],
                "window_end_date": window_data["date"].iloc[-1],
                "features": features,
                "label": label,
            }
            
            self.samples.append(sample)
    
    def _extract_features(self, window_data: pd.DataFrame) -> torch.Tensor:
        """
        Extract features from a window of data.
        
        Parameters
        ----------
        window_data : pd.DataFrame
            DataFrame containing window_days rows of measurements.
        
        Returns
        -------
        torch.Tensor
            Feature tensor of shape (window_days, n_features).
        """
        features = []
        
        for _, row in window_data.iterrows():
            day_features = [
                row.get("resting_hr_mean", 0.0),
                row.get("resting_hr_std", 0.0),
                row.get("resting_hr_min", 0.0),
                row.get("resting_hr_max", 0.0),
                row.get("steps_total", 0.0),
                row.get("steps_mean_hourly", 0.0),
                row.get("sleep_duration_hours", 0.0),
                row.get("sleep_efficiency", 0.0),
            ]
            features.append(day_features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample from the dataset.
        
        Parameters
        ----------
        idx : int
            Index of the sample.
        
        Returns
        -------
        dict
            Sample dictionary containing:
            - participant_id: Participant identifier
            - window_start_date: Start date of the window
            - window_end_date: End date of the window
            - features: Feature tensor of shape (window_days, n_features)
            - label: Binary label (0 or 1)
        """
        sample = self.samples[idx].copy()
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def get_feature_names(self) -> list:
        """Return the list of feature names."""
        return self.feature_names
    
    def get_label_distribution(self) -> dict:
        """Return the distribution of labels in the dataset."""
        labels = [sample["label"] for sample in self.samples]
        return {
            "total_samples": len(labels),
            "positive_samples": sum(labels),
            "negative_samples": len(labels) - sum(labels),
            "positive_ratio": sum(labels) / len(labels) if labels else 0.0,
        }

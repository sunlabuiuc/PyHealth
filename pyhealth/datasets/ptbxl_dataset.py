"""
Author: Damir Temir, Salvador Tranquilino-Ramos
NetID: dtemi2, stran42
Paper title: Data Augmentation for Electrocardiograms
Paper link: https://arxiv.org/pdf/2204.04360

Description:
This module implements a dataset class for PTB-XL electrocardiography data, a large publicly available
ECG dataset. The dataset contains 12-lead ECG recordings from 18,885 patients with various cardiac
abnormalities. It provides functionality to load, process, and split the data for training and
evaluation of ECG classification models.

Data source:
https://physionet.org/content/ptb-xl/1.0.0/
"""

import logging
import os
import numpy as np
import pandas as pd
import wfdb
import ast
from typing import List, Optional, Tuple
from .base_dataset import BaseDataset

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class PTBXL(Dataset):
    """Dataset class for PTB-XL electrocardiography data."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Initialize the PTBXL dataset with ECG signals and labels.

        Args:
            x (np.ndarray): ECG signal data of shape (n_samples, n_timesteps, n_leads)
            y (np.ndarray): Corresponding labels
        """
        super(PTBXL, self).__init__()

        # Downsample to 250 Hz and chop off last 4 samples to get 2496 overall
        if x.shape[1] != 2496 and x.shape[1] == 5000:
            x = x[:, ::2, :]  # Downsample by taking every other point
            x = x[:, :-4]  # Remove last 4 samples to get 2496

        self.x = np.transpose(x, (0, 2, 1)).astype(np.float32)  # Change to (n_samples, n_leads, n_timesteps)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class PTBXLWrapper(BaseDataset):
    """Wrapper class for loading and processing PTB-XL ECG data."""

    def __init__(
            self,
            root: str,
            tables: List[str],
            dataset_name: Optional[str] = None,
            config_path: Optional[str] = None,
            dev: bool = False,
            batch_size: int = 32,
            num_workers: int = 0,
            **kwargs,
    ):
        """
        Initialize the PTBXL dataset wrapper.

        Args:
            root (str): Path to the directory containing PTB-XL data files
            tables (List[str]): List of table names to load (unused in this implementation)
            dataset_name (Optional[str]): Name of the dataset
            config_path (Optional[str]): Path to configuration file
            dev (bool): Whether to run in development mode (limits data size)
            batch_size (int): Batch size for data loaders
            num_workers (int): Number of workers for data loading
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampling_rate = 500  # Original sampling rate in Hz
        self.path = root

        # Diagnostic class mapping
        self.idxd = {'NORM': 0, 'MI': 1, 'STTC': 2, 'CD': 3, 'HYP': 4}

        super().__init__(root, tables, dataset_name, config_path, dev, **kwargs)

    def load_raw_data(self, df: pd.DataFrame, sampling_rate: int) -> np.ndarray:
        """
        Load raw ECG data from WFDB files.

        Args:
            df (pd.DataFrame): DataFrame containing file names
            sampling_rate (int): Desired sampling rate (100 or 500 Hz)

        Returns:
            np.ndarray: Array of ECG signals
        """
        if sampling_rate == 100:
            data = [wfdb.rdsamp(os.path.join(self.path, f)) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(os.path.join(self.path, f)) for f in df.filename_hr]
        return np.array([signal for signal, meta in data])

    def aggregate_diagnostic(self, y_dic: dict) -> np.ndarray:
        """
        Aggregate diagnostic codes into superclasses.

        Args:
            y_dic (dict): Dictionary of diagnostic codes

        Returns:
            np.ndarray: One-hot encoded diagnostic superclass vector
        """
        tmp = np.zeros(5)
        for key in y_dic.keys():
            if key in self.agg_df.index:
                cls = self.agg_df.loc[key].diagnostic_class
                tmp[self.idxd[cls]] = 1
        return tmp

    def get_data_loaders(self, args) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get data loaders for training, validation, and testing.

        Args:
            args: Command line arguments or configuration object

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Load and convert annotation data
        Y = pd.read_csv(os.path.join(self.path, 'ptbxl_database.csv'), index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = self.load_raw_data(Y, self.sampling_rate)

        # Load scp_statements.csv for diagnostic aggregation
        self.agg_df = pd.read_csv(os.path.join(self.path, 'scp_statements.csv'), index_col=0)
        self.agg_df = self.agg_df[self.agg_df.diagnostic == 1]

        # Apply diagnostic superclass
        Y['diagnostic_superclass'] = Y.scp_codes.apply(self.aggregate_diagnostic)

        # Split data into train and test (using fold 10 as test)
        test_fold = 10
        X_train = X[np.where(Y.strat_fold != test_fold)]
        y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
        y_train = np.stack(y_train, axis=0)

        X_test = X[np.where(Y.strat_fold == test_fold)]
        y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
        y_test = np.stack(y_test, axis=0)

        # Normalize data
        meansig = np.mean(X_train.reshape(-1))
        stdsig = np.std(X_train.reshape(-1))
        X_train = (X_train - meansig) / stdsig
        X_test = (X_test - meansig) / stdsig

        # Set random seeds for reproducibility
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        # Create train/validation splits
        rng = np.random.RandomState(args.seed)
        idxs = np.arange(len(y_train))
        rng.shuffle(idxs)

        train_samp = int(0.8 * args.train_samp)
        val_samp = args.train_samp - train_samp
        train_idxs = idxs[:train_samp]
        val_idxs = idxs[train_samp:train_samp + val_samp]

        # Create datasets based on task
        if args.task != 'all':
            task_idx = self.idxd[args.task]
            prevalence = np.mean(y_train[:, task_idx])
            self.weights = []
            for i in y_train[train_idxs][:, task_idx]:
                self.weights.append(1 - prevalence if i == 1 else prevalence)

            ft_train = PTBXL(X_train[train_idxs], y_train[train_idxs][:, task_idx])
            ft_val = PTBXL(X_train[val_idxs], y_train[val_idxs][:, task_idx])
            ft_test = PTBXL(X_test, y_test[:, task_idx])
        else:
            ft_train = PTBXL(X_train[train_idxs], y_train[train_idxs])
            ft_val = PTBXL(X_train[val_idxs], y_train[val_idxs])
            ft_test = PTBXL(X_test, y_test)

        # Create data loaders
        train_loader = DataLoader(
            dataset=ft_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        val_loader = DataLoader(
            dataset=ft_val,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        test_loader = DataLoader(
            dataset=ft_test,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

        return train_loader, val_loader, test_loader

    def stats(self):
        """Print statistics about the dataset."""
        # You can implement this to show dataset statistics
        print(f"PTB-XL Dataset Statistics")
        print(f"Sampling rate: {self.sampling_rate} Hz")
        print(f"Diagnostic classes: {list(self.idxd.keys())}")
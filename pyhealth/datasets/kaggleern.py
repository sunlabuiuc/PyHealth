"""
PyHealth dataset utility for KaggleERN (INRIA BCI Challenge) EEG dataset.

Dataset link:
    https://www.kaggle.com/c/inria-bci-challenge/data

Notes:
    - Kaggle requires authentication; this dataset class does NOT auto-download.
    - This class provides an offline preprocessing utility to convert raw CSV
      into epoch/window pickles for downstream training.

Expected raw folder structure (after manual download/unzip):
    <root>/
      train/
        Data_*.csv
      test/
        Data_*.csv                (optional for preprocessing; no labels)
      TrainLabels.csv
      ChannelsLocation.csv

Pickle output format (kept identical to your current training code):
    {"signal": np.ndarray(C, T), "label": 0/1, "epoch_id": str}
"""

from __future__ import annotations

import os
import glob
import pickle
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pyhealth.datasets import BaseSignalDataset

logger = logging.getLogger(__name__)


@dataclass
class KaggleERNPreprocessConfig:
    """
    Preprocessing configuration.

    Defaults are set to preserve your current EEGPT-style pipeline behavior:
      - resample -> 256 Hz
      - bandpass 0.1~75 Hz
      - notch 50 Hz
      - remove DC offset
      - average reference
      - slice epochs by FeedBackEvent == 1, window = chunk_size_sec
      - stratified split: 80/10/10
      - save pickles as {"signal","label","epoch_id"}
    """
    # raw dataset
    root: str
    train_subdir: str = "train"
    labels_csv: str = "TrainLabels.csv"
    channels_csv: str = "ChannelsLocation.csv"

    # epoching
    chunk_size_sec: float = 3.0
    line_noise_hz: float = 50.0
    random_seed: int = 42
    min_epochs_per_file: int = 60

    # preprocessing parameters (general naming; defaults match your EEGPT pipeline)
    target_sfreq: int = 256
    l_freq: float = 0.1
    h_freq: float = 75.0
    remove_dc_offset: bool = True
    average_reference: bool = True

    # output (will create train/val/test inside)
    output_root: str = "./processed_kaggle_ern"


class KaggleERNDataset(BaseSignalDataset):
    """
    A lightweight dataset utility wrapper for KaggleERN.

    This class is designed to preserve your existing workflow 1:1:
      raw CSV -> offline epoch pickles -> external finetune script reads pickles.
    """

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.train_dir = os.path.join(self.root, "train")
        self.labels_path = os.path.join(self.root, "TrainLabels.csv")
        self.channels_path = os.path.join(self.root, "ChannelsLocation.csv")
        self._verify_raw()

    # -----------------------
    # Verify / metadata
    # -----------------------
    def _verify_raw(self) -> None:
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        if not os.path.isdir(self.train_dir):
            raise FileNotFoundError(f"Missing train/ folder: {self.train_dir}")

        csvs = sorted(glob.glob(os.path.join(self.train_dir, "*.csv")))
        if len(csvs) == 0:
            raise FileNotFoundError(f"No CSV found under: {self.train_dir}")

        if not os.path.isfile(self.labels_path):
            raise FileNotFoundError(f"Missing TrainLabels.csv: {self.labels_path}")

        if not os.path.isfile(self.channels_path):
            raise FileNotFoundError(f"Missing ChannelsLocation.csv: {self.channels_path}")

    def load_labels_map(self) -> Dict[str, int]:
        """
        TrainLabels.csv is expected to contain:
          - IdFeedBack
          - Prediction
        where IdFeedBack matches "<prefix>_FB###" generated from Data_*.csv.
        """
        df = pd.read_csv(self.labels_path)
        required = {"IdFeedBack", "Prediction"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"TrainLabels.csv missing columns: {sorted(missing)}. Found={list(df.columns)}")
        return df.set_index("IdFeedBack")["Prediction"].to_dict()

    def load_channels_location(self) -> pd.DataFrame:
        """Optional utility (not used in preprocessing by default)."""
        return pd.read_csv(self.channels_path)

    # -----------------------
    # Core: preprocessing helpers
    # -----------------------
    @staticmethod
    def _df_to_raw_full(df: pd.DataFrame, sfreq: int = 200):
        """
        Convert EEG CSV DataFrame to an MNE Raw object.

        CSV is expected to include:
          - Time, FeedBackEvent, EOG (may exist)
          - EEG channels in other columns
        """
        import mne  # lazy import
        mne.set_log_level("ERROR")

        if "FeedBackEvent" not in df.columns:
            raise ValueError("CSV missing required column: FeedBackEvent")

        ch_names = [c for c in df.columns if c not in ["Time", "FeedBackEvent", "EOG"]]
        if len(ch_names) == 0:
            raise ValueError("No EEG channels found after excluding Time/FeedBackEvent/EOG columns.")

        eeg_data = df[ch_names].T.values
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(eeg_data, info, verbose=False)
        return raw, ch_names

    @staticmethod
    def _remove_dc_offset(raw):
        """Remove DC offset per channel (identical behavior to your script)."""
        data = raw.get_data()
        data -= np.mean(data, axis=1, keepdims=True)
        raw._data = data
        return raw

    @staticmethod
    def _prepare_and_slice_epochs(
        df: pd.DataFrame,
        raw,
        epoch_duration_sec: float,
        file_path: str,
        labels_map: Dict[str, int],
    ) -> Tuple[List[Tuple[np.ndarray, int, str]], str]:
        """
        Slice continuous EEG into labeled epochs based on feedback events.
        Keeps exact ID generation:
          epoch_id = "<prefix>_FB###"
        where prefix comes from filename:
          Data_<prefix>.csv  -> prefix
        """
        feedback_events = df.index[df["FeedBackEvent"] == 1].tolist()
        epoch_samples = int(epoch_duration_sec * raw.info["sfreq"])
        signals = raw.get_data()

        filename = os.path.basename(file_path)
        prefix = filename.replace("Data_", "").replace(".csv", "")
        epochs: List[Tuple[np.ndarray, int, str]] = []

        for fb_idx, event_idx in enumerate(feedback_events):
            start_idx = event_idx
            end_idx = event_idx + epoch_samples
            if end_idx <= signals.shape[1]:
                epoch_data = signals[:, start_idx:end_idx].copy()
                epoch_id = f"{prefix}_FB{fb_idx + 1:03d}"
                label = labels_map.get(epoch_id, -1)
                if label != -1:
                    epochs.append((epoch_data, int(label), epoch_id))

        return epochs, filename

    @staticmethod
    def _save_epochs(epochs: List[Tuple[np.ndarray, int, str]], folder: str) -> None:
        """Save epoch tuples into .pickle files with your exact dict format."""
        os.makedirs(folder, exist_ok=True)
        for epoch_data, label, epoch_id in epochs:
            sample = {"signal": epoch_data, "label": label, "epoch_id": epoch_id}
            path = os.path.join(folder, f"{epoch_id}.pickle")
            with open(path, "wb") as f:
                pickle.dump(sample, f)

    # -----------------------
    # Single preprocessing path (general naming; behavior matches your EEGPT pipeline)
    # -----------------------
    def _extract_epochs_default(self, file_path: str, labels_map: Dict[str, int], cfg: KaggleERNPreprocessConfig, sfreq: int = 200):
        """
        Default preprocessing pipeline (kept identical to your EEGPT path):
          resample(cfg.target_sfreq)
          filter(cfg.l_freq, cfg.h_freq)
          notch(cfg.line_noise_hz)
          remove DC offset (optional)
          average reference (optional)
          slice epochs by feedback events
        """
        df = pd.read_csv(file_path)
        raw, _ = self._df_to_raw_full(df, sfreq)

        raw.resample(cfg.target_sfreq)
        raw.filter(cfg.l_freq, cfg.h_freq, n_jobs=1)
        raw.notch_filter(cfg.line_noise_hz, n_jobs=1)

        if cfg.remove_dc_offset:
            raw = self._remove_dc_offset(raw)

        if cfg.average_reference:
            raw.set_eeg_reference(ref_channels="average")

        return self._prepare_and_slice_epochs(df, raw, cfg.chunk_size_sec, file_path, labels_map)

    def preprocess_epochs(self, cfg: KaggleERNPreprocessConfig) -> Dict[str, List[str]]:
        """
        Run offline preprocessing:
          raw train/*.csv -> epoch pickles -> stratified train/val/test split (80/10/10).

        Returns:
          dict with:
            - "files_with_few_epochs": list[str]
        """
        # align runtime root with cfg.root (but keep object paths if already initialized)
        if os.path.abspath(cfg.root) != self.root:
            raise ValueError(f"cfg.root ({cfg.root}) must match dataset root used in __init__ ({self.root}).")

        np.random.seed(cfg.random_seed)
        labels_map = self.load_labels_map()

        train_csvs = sorted(glob.glob(os.path.join(self.train_dir, "*.csv")))
        files_with_few_epochs: List[str] = []
        all_epochs: List[Tuple[np.ndarray, int, str]] = []

        for fp in train_csvs:
            epochs, filename = self._extract_epochs_default(fp, labels_map=labels_map, cfg=cfg)
            logger.info(f"[preprocess] {filename}: {len(epochs)} epochs")

            if len(epochs) < cfg.min_epochs_per_file:
                files_with_few_epochs.append(filename)

            all_epochs.extend(epochs)

        if len(all_epochs) == 0:
            raise RuntimeError("No epochs generated. Please check CSV schema and TrainLabels mapping.")

        # Stratified train/val/test split (same as your original script)
        y = [lab for _, lab, _ in all_epochs]
        try:
            train_epochs, temp_epochs = train_test_split(
                all_epochs, test_size=0.2, stratify=y, random_state=cfg.random_seed
            )
            y_temp = [lab for _, lab, _ in temp_epochs]
            val_epochs, test_epochs = train_test_split(
                temp_epochs, test_size=0.5, stratify=y_temp, random_state=cfg.random_seed
            )
        except ValueError as e:
            raise ValueError(
                "Stratified split failed (likely class imbalance / too few samples in one class). "
                f"Original error: {e}"
            ) from e

        out_train = os.path.join(cfg.output_root, "train")
        out_val = os.path.join(cfg.output_root, "val")
        out_test = os.path.join(cfg.output_root, "test")

        self._save_epochs(train_epochs, out_train)
        self._save_epochs(val_epochs, out_val)
        self._save_epochs(test_epochs, out_test)

        logger.info(f"Saved train={len(train_epochs)} val={len(val_epochs)} test={len(test_epochs)}")
        return {"files_with_few_epochs": files_with_few_epochs}

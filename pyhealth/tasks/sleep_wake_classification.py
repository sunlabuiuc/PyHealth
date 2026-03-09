import numpy as np
import pandas as pd
from typing import Any, Dict, List

import neurokit2 as nk
from scipy.signal import butter, cheby2, filtfilt
from scipy.stats import trim_mean
from scipy.stats.mstats import winsorize

from .base_task import BaseTask


class SleepWakeClassification(BaseTask):
    task_name = "SleepWakeClassification"
    input_schema = {"features": "vector"}
    output_schema = {"label": "binary"}

    def __init__(self, epoch_seconds: int = 30, sampling_rate: int = 64):
        self.epoch_seconds = epoch_seconds
        self.sampling_rate = sampling_rate
        super().__init__()

    def _map_sleep_label(self, label):
        if label is None or pd.isna(label):
            return None

        label = str(label).strip().upper()

        if label in {"WAKE", "W"}:
            return 1
        if label in {"REM", "R", "N1", "N2", "N3"}:
            return 0

        return None

    def _safe_numeric(self, series: pd.Series) -> np.ndarray:
        return pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy()

    def _split_into_epochs(self, signal: np.ndarray, fs: float) -> List[np.ndarray]:
        samples_per_epoch = int(fs * self.epoch_seconds)
        num_epochs = len(signal) // samples_per_epoch

        epochs = []
        for i in range(num_epochs):
            start = i * samples_per_epoch
            end = start + samples_per_epoch
            epochs.append(signal[start:end])

        return epochs

    def _butter_bandpass(
        self,
        low_hz: float,
        high_hz: float,
        fs: float,
        order: int,
    ):
        nyq = 0.5 * fs
        low = low_hz / nyq
        high = high_hz / nyq
        return butter(order, [low, high], btype="band")

    def _apply_filter(self, signal: np.ndarray, b, a) -> np.ndarray:
        if signal.ndim != 1:
            raise ValueError("Signal must be 1D.")
        return filtfilt(b, a, signal)

    def _filter_acc(self, signal: np.ndarray, fs: float) -> np.ndarray:
        b, a = self._butter_bandpass(
            low_hz=3.0,
            high_hz=11.0,
            fs=fs,
            order=5,
        )
        return self._apply_filter(signal, b, a)
    
    def _cheby2_bandpass(
        self,
        low_hz: float,
        high_hz: float,
        fs: float,
        order: int,
        rs: float = 40.0,
    ):
        nyq = 0.5 * fs
        low = low_hz / nyq
        high = high_hz / nyq
        return cheby2(order, rs, [low, high], btype="band")

    def _filter_bvp(self, signal: np.ndarray, fs: float) -> np.ndarray:
        b, a = self._cheby2_bandpass(
            low_hz=0.5,
            high_hz=20.0,
            fs=fs,
            order=4,
            rs=40.0,
        )
        return self._apply_filter(signal, b, a)

    def _extract_bvp_features(
        self,
        signal: np.ndarray,
        fs: float,
    ) -> List[Dict[str, float]]:
        filtered = self._filter_bvp(signal, fs)
        epochs = self._split_into_epochs(filtered, fs)

        features = []
        for ep in epochs:
            try:
                _, info = nk.ppg_process(ep, sampling_rate=fs)
                hrv = nk.hrv_time(
                    info["PPG_Peaks"],
                    sampling_rate=fs,
                    show=False,
                )

                features.append(
                    {
                        "rmssd": float(hrv["HRV_RMSSD"].values[0]),
                        "sdnn": float(hrv["HRV_SDNN"].values[0]),
                        "pnn50": float(hrv["HRV_pNN50"].values[0]),
                    }
                )
            except Exception:
                features.append(
                    {
                        "rmssd": np.nan,
                        "sdnn": np.nan,
                        "pnn50": np.nan,
                    }
                )

        return features

    def _iqr(self, x: np.ndarray) -> float:
        return float(np.percentile(x, 75) - np.percentile(x, 25))

    def _extract_acc_axis_features(self, signal: np.ndarray, fs: float) -> List[Dict[str, float]]:
        filtered = self._filter_acc(signal, fs)
        filtered_abs = np.abs(filtered)
        epochs = self._split_into_epochs(filtered_abs, fs)

        features = []
        for ep in epochs:
            features.append(
                {
                    "trimmed_mean": float(trim_mean(ep, proportiontocut=0.10)),
                    "max": float(np.max(ep)),
                    "iqr": self._iqr(ep),
                }
            )
        return features

    def _extract_acc_mad_features(
        self,
        acc_x: np.ndarray,
        acc_y: np.ndarray,
        acc_z: np.ndarray,
        fs: float,
    ) -> List[Dict[str, float]]:
        magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        epochs = self._split_into_epochs(magnitude, fs)

        features = []
        for ep in epochs:
            mad = np.mean(np.abs(ep - np.mean(ep)))
            features.append({"mad": float(mad)})
        return features

    def _extract_temp_features(self, signal: np.ndarray, fs: float) -> List[Dict[str, float]]:
        limits = (0.05, 0.05)
        wins_signal = winsorize(signal, limits=limits)
        wins_signal = np.clip(wins_signal, 31.0, 40.0)
        epochs = self._split_into_epochs(np.asarray(wins_signal), fs)

        features = []
        for ep in epochs:
            features.append(
                {
                    "mean": float(np.mean(ep)),
                    "min": float(np.min(ep)),
                    "max": float(np.max(ep)),
                    "std": float(np.std(ep)),
                }
            )
        return features

    def _build_record_epoch_features(self, df: pd.DataFrame) -> List[List[float]]:
        fs = float(self.sampling_rate)

        required_acc = ["ACC_X", "ACC_Y", "ACC_Z"]
        if not all(col in df.columns for col in required_acc):
            return []

        if "TEMP" not in df.columns:
            return []

        if "BVP" not in df.columns:
            return []

        acc_x = self._safe_numeric(df["ACC_X"])
        acc_y = self._safe_numeric(df["ACC_Y"])
        acc_z = self._safe_numeric(df["ACC_Z"])
        temp = self._safe_numeric(df["TEMP"])
        bvp = self._safe_numeric(df["BVP"])

        acc_x_feats = self._extract_acc_axis_features(acc_x, fs)
        acc_y_feats = self._extract_acc_axis_features(acc_y, fs)
        acc_z_feats = self._extract_acc_axis_features(acc_z, fs)
        acc_mad_feats = self._extract_acc_mad_features(acc_x, acc_y, acc_z, fs)
        temp_feats = self._extract_temp_features(temp, fs)
        bvp_feats = self._extract_bvp_features(bvp, fs)

        num_epochs = min(
            len(acc_x_feats),
            len(acc_y_feats),
            len(acc_z_feats),
            len(acc_mad_feats),
            len(temp_feats),
            len(bvp_feats),
        )

        all_epoch_features = []
        for i in range(num_epochs):
            feats = []

            feats.extend(
                [
                    acc_x_feats[i]["trimmed_mean"],
                    acc_x_feats[i]["max"],
                    acc_x_feats[i]["iqr"],
                ]
            )
            feats.extend(
                [
                    acc_y_feats[i]["trimmed_mean"],
                    acc_y_feats[i]["max"],
                    acc_y_feats[i]["iqr"],
                ]
            )
            feats.extend(
                [
                    acc_z_feats[i]["trimmed_mean"],
                    acc_z_feats[i]["max"],
                    acc_z_feats[i]["iqr"],
                ]
            )
            feats.append(acc_mad_feats[i]["mad"])

            feats.extend(
                [
                    temp_feats[i]["mean"],
                    temp_feats[i]["min"],
                    temp_feats[i]["max"],
                    temp_feats[i]["std"],
                ]
            )

            feats.extend(
                [
                    bvp_feats[i]["rmssd"],
                    bvp_feats[i]["sdnn"],
                    bvp_feats[i]["pnn50"],
                ]
            )

            all_epoch_features.append(feats)

        return all_epoch_features

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        samples = []
        events = patient.get_events(event_type="dreamt_sleep")
        if len(events) == 0:
            return samples

        epoch_size = self.epoch_seconds * self.sampling_rate

        for event_idx, event in enumerate(events):
            file_path = getattr(event, "file_64hz", None)
            if file_path is None:
                continue

            try:
                df = pd.read_csv(file_path)
            except Exception:
                continue

            if "Sleep_Stage" not in df.columns:
                continue

            record_epoch_features = self._build_record_epoch_features(df)
            if len(record_epoch_features) == 0:
                continue

            n_label_epochs = len(df) // epoch_size
            n_epochs = min(len(record_epoch_features), n_label_epochs)

            for epoch_idx in range(n_epochs):
                start = epoch_idx * epoch_size
                end = start + epoch_size
                epoch_df = df.iloc[start:end]

                if len(epoch_df) < epoch_size:
                    continue

                stage_mode = epoch_df["Sleep_Stage"].mode(dropna=True)
                if len(stage_mode) == 0:
                    continue

                raw_label = stage_mode.iloc[0]
                label = self._map_sleep_label(raw_label)
                if label is None:
                    continue

                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "record_id": f"{patient.patient_id}-event{event_idx}-epoch{epoch_idx}",
                        "epoch_index": epoch_idx,
                        "features": record_epoch_features[epoch_idx],
                        "label": label,
                    }
                )

        return samples
from typing import Dict, List

import neurokit2 as nk
import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, cheby2, filtfilt
from scipy.stats import trim_mean
from scipy.stats.mstats import winsorize

from ..data import Patient
from .base_task import BaseTask


class SleepWakeClassification(BaseTask):
    """Binary sleep-wake classification task for DREAMT wearable recordings.

    This task converts each DREAMT wearable recording into fixed-length epochs,
    extracts physiological features from multiple sensor modalities, 
    augments them with temporal context, and assigns a binary sleep/wake 
    label to each epoch.
    """

    task_name = "SleepWakeClassification"
    input_schema = {"features": "tensor"}
    output_schema = {"label": "binary"}

    def __init__(self, epoch_seconds: int = 30, sampling_rate: int = 64):
        """Initializes the sleep-wake classification task.

        Args:
            epoch_seconds: Length of each epoch in seconds.
            sampling_rate: Sampling rate of the wearable recording in Hz.
        """
        self.epoch_seconds = epoch_seconds
        self.sampling_rate = sampling_rate
        super().__init__()

    def _convert_sleep_stage_to_binary_label(self, label):
        """Maps a sleep stage label to the binary sleep-wake target.

        Args:
            label: Raw sleep stage label from the DREAMT recording.

        Returns:
            ``1`` for wake, ``0`` for sleep, or ``None`` if the label is
            missing or unsupported.
        """
        if label is None or pd.isna(label):
            return None

        label = str(label).strip().upper()

        if label in {"WAKE", "W"}:
            return 1
        if label in {"REM", "R", "N1", "N2", "N3"}:
            return 0

        return None

    def _split_signal_into_epochs(
        self,
        signal: np.ndarray,
        sampling_rate_hz: float,
    ) -> List[np.ndarray]:
        """Splits a 1D signal into non-overlapping fixed-length epochs.

        Args:
            signal: One-dimensional signal array.
            sampling_rate_hz: Sampling rate of the signal in Hz.

        Returns:
            A list of signal segments, one per complete epoch.
        """
        samples_per_epoch = int(sampling_rate_hz * self.epoch_seconds)
        num_epochs = len(signal) // samples_per_epoch

        epochs = []
        for i in range(num_epochs):
            start = i * samples_per_epoch
            end = start + samples_per_epoch
            epochs.append(signal[start:end])

        return epochs

    def _design_bandpass_filter_coefficients(
        self,
        filter_family: str,
        low_hz: float,
        high_hz: float,
        sampling_rate_hz: float,
        order: int,
        stopband_attenuation_db: float = 40.0,
    ):
        """Designs band-pass filter coefficients for a supported family.

        Args:
            filter_family: Filter family name, currently ``"butter"`` or
                ``"cheby2"``.
            low_hz: Lower cutoff frequency in Hz.
            high_hz: Upper cutoff frequency in Hz.
            sampling_rate_hz: Sampling rate in Hz.
            order: Filter order.
            stopband_attenuation_db: Stopband attenuation used by Chebyshev-II.

        Returns:
            The numerator and denominator filter coefficients.

        Raises:
            ValueError: If the filter family is unsupported.
        """
        nyq = 0.5 * sampling_rate_hz
        low = low_hz / nyq
        high = high_hz / nyq

        if filter_family == "butter":
            return butter(order, [low, high], btype="band")
        if filter_family == "cheby2":
            return cheby2(
                order,
                stopband_attenuation_db,
                [low, high],
                btype="band",
            )

        raise ValueError(f"Unsupported bandpass filter family: {filter_family}")

    def _apply_zero_phase_filter(self, signal: np.ndarray, b, a) -> np.ndarray:
        """Applies zero-phase filtering to a one-dimensional signal.

        Args:
            signal: One-dimensional signal array.
            b: Numerator filter coefficients.
            a: Denominator filter coefficients.

        Returns:
            The filtered signal.

        Raises:
            ValueError: If ``signal`` is not one-dimensional.
        """
        if signal.ndim != 1:
            raise ValueError("Signal must be 1D.")
        return filtfilt(b, a, signal)

    def _filter_signal_with_lowpass(
        self,
        signal: np.ndarray,
        sampling_rate_hz: float,
        cutoff_hz: float,
        order: int = 4,
    ) -> np.ndarray:
        """Applies a Butterworth low-pass filter to a signal.

        Args:
            signal: One-dimensional signal array.
            sampling_rate_hz: Sampling rate in Hz.
            cutoff_hz: Low-pass cutoff frequency in Hz.
            order: Filter order.

        Returns:
            The filtered signal.
        """
        nyq = 0.5 * sampling_rate_hz
        b, a = butter(order, cutoff_hz / nyq, btype="low")
        return self._apply_zero_phase_filter(signal, b, a)

    def _filter_accelerometer_signal(
        self,
        signal: np.ndarray,
        sampling_rate_hz: float,
    ) -> np.ndarray:
        """Band-pass filters an accelerometer signal for movement features.

        Args:
            signal: Raw accelerometer signal from one axis.
            sampling_rate_hz: Sampling rate in Hz.

        Returns:
            The filtered accelerometer signal.
        """
        b, a = self._design_bandpass_filter_coefficients(
            filter_family="butter",
            low_hz=3.0,
            high_hz=11.0,
            sampling_rate_hz=sampling_rate_hz,
            order=5,
        )
        return self._apply_zero_phase_filter(signal, b, a)

    def _filter_blood_volume_pulse_signal(
        self,
        signal: np.ndarray,
        sampling_rate_hz: float,
    ) -> np.ndarray:
        """Band-pass filters a blood volume pulse signal.

        Args:
            signal: Raw blood volume pulse signal.
            sampling_rate_hz: Sampling rate in Hz.

        Returns:
            The filtered blood volume pulse signal.
        """
        b, a = self._design_bandpass_filter_coefficients(
            filter_family="cheby2",
            low_hz=0.5,
            high_hz=20.0,
            sampling_rate_hz=sampling_rate_hz,
            order=4,
            stopband_attenuation_db=40.0,
        )
        return self._apply_zero_phase_filter(signal, b, a)

    def _detrend_signal_by_segments(
        self,
        signal: np.ndarray,
        sampling_rate_hz: float,
        segment_seconds: int,
    ) -> np.ndarray:
        """Detrends a signal by removing a linear trend per segment.

        Args:
            signal: One-dimensional signal array.
            sampling_rate_hz: Sampling rate in Hz.
            segment_seconds: Segment length used for local detrending.

        Returns:
            A detrended copy of the input signal.
        """
        samples_per_seg = int(sampling_rate_hz * segment_seconds)
        detrended = signal.copy()

        for i in range(0, len(signal), samples_per_seg):
            seg = signal[i : i + samples_per_seg]
            if len(seg) < 2:
                continue

            x = np.arange(len(seg))
            coeffs = np.polyfit(x, seg, deg=1)
            trend = np.polyval(coeffs, x)
            detrended[i : i + len(seg)] = seg - trend

        return detrended

    def _extract_accelerometer_axis_epoch_features(
        self,
        signal: np.ndarray,
        sampling_rate_hz: float,
    ) -> List[Dict[str, float]]:
        """Extracts per-epoch features from one accelerometer axis.

        Args:
            signal: Raw accelerometer signal from one axis.
            sampling_rate_hz: Sampling rate in Hz.

        Returns:
            A list of feature dictionaries, one per epoch.
        """
        filtered = self._filter_accelerometer_signal(signal, sampling_rate_hz)
        filtered_abs = np.abs(filtered)
        epochs = self._split_signal_into_epochs(filtered_abs, sampling_rate_hz)

        return [
            {
                "trimmed_mean": float(trim_mean(epoch, proportiontocut=0.10)),
                "max": float(np.max(epoch)),
                "iqr": float(np.percentile(epoch, 75) - np.percentile(epoch, 25)),
            }
            for epoch in epochs
        ]

    def _extract_accelerometer_magnitude_deviation_epoch_features(
        self,
        accelerometer_x_signal: np.ndarray,
        accelerometer_y_signal: np.ndarray,
        accelerometer_z_signal: np.ndarray,
        sampling_rate_hz: float,
    ) -> List[Dict[str, float]]:
        """Extracts mean absolute deviation features from ACC magnitude.

        Args:
            accelerometer_x_signal: Accelerometer X-axis signal.
            accelerometer_y_signal: Accelerometer Y-axis signal.
            accelerometer_z_signal: Accelerometer Z-axis signal.
            sampling_rate_hz: Sampling rate in Hz.

        Returns:
            A list of feature dictionaries containing magnitude deviation.
        """
        magnitude = np.sqrt(
            accelerometer_x_signal**2
            + accelerometer_y_signal**2
            + accelerometer_z_signal**2
        )
        epochs = self._split_signal_into_epochs(magnitude, sampling_rate_hz)

        return [
            {"mad": float(np.mean(np.abs(epoch - np.mean(epoch))))}
            for epoch in epochs
        ]

    def _extract_temperature_epoch_features(
        self,
        signal: np.ndarray,
        sampling_rate_hz: float,
    ) -> List[Dict[str, float]]:
        """Extracts per-epoch temperature summary statistics.

        Args:
            signal: Raw temperature signal.
            sampling_rate_hz: Sampling rate in Hz.

        Returns:
            A list of feature dictionaries, one per epoch.
        """
        limits = (0.05, 0.05)
        wins_signal = winsorize(signal, limits=limits)
        wins_signal = np.clip(wins_signal, 31.0, 40.0)
        epochs = self._split_signal_into_epochs(
            np.asarray(wins_signal),
            sampling_rate_hz,
        )

        return [
            {
                "mean": float(np.mean(epoch)),
                "min": float(np.min(epoch)),
                "max": float(np.max(epoch)),
                "std": float(np.std(epoch)),
            }
            for epoch in epochs
        ]

    def _extract_blood_volume_pulse_epoch_features(
        self,
        signal: np.ndarray,
        sampling_rate_hz: float,
    ) -> List[Dict[str, float]]:
        """Extracts HRV-based features from blood volume pulse epochs.

        Args:
            signal: Raw blood volume pulse signal.
            sampling_rate_hz: Sampling rate in Hz.

        Returns:
            A list of feature dictionaries, one per epoch.
        """
        filtered = self._filter_blood_volume_pulse_signal(signal, sampling_rate_hz)
        epochs = self._split_signal_into_epochs(filtered, sampling_rate_hz)
        epoch_features = []
        for epoch in epochs:
            try:
                _, info = nk.ppg_process(epoch, sampling_rate=sampling_rate_hz)
                hrv = nk.hrv_time(
                    info["PPG_Peaks"],
                    sampling_rate=sampling_rate_hz,
                    show=False,
                )

                epoch_features.append(
                    {
                        "rmssd": float(hrv["HRV_RMSSD"].values[0]),
                        "sdnn": float(hrv["HRV_SDNN"].values[0]),
                        "pnn50": float(hrv["HRV_pNN50"].values[0]),
                    }
                )
            except Exception:
                epoch_features.append(
                    {"rmssd": np.nan, "sdnn": np.nan, "pnn50": np.nan}
                )

        return epoch_features

    def _extract_electrodermal_activity_epoch_features(
        self,
        signal: np.ndarray,
        sampling_rate_hz: float,
    ) -> List[Dict[str, float]]:
        """Extracts SCR-based features from electrodermal activity epochs.

        Args:
            signal: Raw electrodermal activity signal.
            sampling_rate_hz: Sampling rate in Hz.

        Returns:
            A list of feature dictionaries, one per epoch.
        """
        detrended = self._detrend_signal_by_segments(
            signal,
            sampling_rate_hz,
            segment_seconds=5,
        )
        filtered = self._filter_signal_with_lowpass(
            detrended,
            sampling_rate_hz,
            cutoff_hz=1.0,
        )

        eda_signals, _ = nk.eda_process(filtered, sampling_rate=sampling_rate_hz)
        scr = eda_signals["EDA_Phasic"].values
        epochs = self._split_signal_into_epochs(scr, sampling_rate_hz)
        epoch_features = []
        for epoch in epochs:
            try:
                _, info = nk.eda_peaks(epoch, sampling_rate=sampling_rate_hz)

                amplitudes = info["SCR_Amplitude"]
                rise_times = info["SCR_RiseTime"]
                recovery_times = info["SCR_RecoveryTime"]

                epoch_features.append(
                    {
                        "scr_amp_mean": float(np.mean(amplitudes))
                        if len(amplitudes)
                        else 0.0,
                        "scr_amp_max": float(np.max(amplitudes))
                        if len(amplitudes)
                        else 0.0,
                        "scr_rise_mean": float(np.mean(rise_times))
                        if len(rise_times)
                        else 0.0,
                        "scr_recovery_mean": float(np.mean(recovery_times))
                        if len(recovery_times)
                        else 0.0,
                    }
                )
            except Exception:
                epoch_features.append(
                    {
                        "scr_amp_mean": np.nan,
                        "scr_amp_max": np.nan,
                        "scr_rise_mean": np.nan,
                        "scr_recovery_mean": np.nan,
                    }
                )

        return epoch_features

    def _compute_rolling_variance(
        self,
        values: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """Computes a centered rolling variance over a feature series.

        Args:
            values: One-dimensional array of feature values over epochs.
            window: Rolling window size in number of epochs.

        Returns:
            An array of rolling variance values.
        """
        out = np.zeros_like(values)
        half = window // 2

        for i in range(len(values)):
            start = max(0, i - half)
            end = min(len(values), i + half + 1)
            out[i] = np.var(values[start:end])

        return out

    def _augment_epoch_features_with_temporal_context(
        self,
        epoch_features: List[List[float]],
        gaussian_sigma: float = 2.0,
        variance_window: int = 5,
    ) -> List[List[float]]:
        """Augments each epoch feature vector with temporal context features.

        For each base feature, this method appends a smoothed version, its
        temporal derivative, and a rolling variance estimate.

        Args:
            epoch_features: Base feature matrix as a list of epoch vectors.
            gaussian_sigma: Gaussian smoothing parameter.
            variance_window: Rolling variance window size in epochs.

        Returns:
            The augmented epoch feature matrix.
        """
        if len(epoch_features) == 0:
            return []

        feature_matrix = np.asarray(epoch_features, dtype=float)
        num_epochs, num_features = feature_matrix.shape

        enhanced = feature_matrix.tolist()

        for j in range(num_features):
            values = feature_matrix[:, j]

            smoothed = gaussian_filter1d(values, sigma=gaussian_sigma, mode="nearest")
            deriv = np.diff(smoothed, prepend=smoothed[0])
            var = self._compute_rolling_variance(smoothed, variance_window)

            for i in range(num_epochs):
                enhanced[i].append(float(smoothed[i]))
                enhanced[i].append(float(deriv[i]))
                enhanced[i].append(float(var[i]))

        return enhanced

    def _extract_sensor_signals_from_dataframe(
        self,
        record_dataframe: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """Extracts all sensor modalities from a DREAMT record.

        Args:
            record_dataframe: Wearable recording loaded as a pandas DataFrame.

        Returns:
            A dictionary mapping modality names to numeric NumPy arrays.
        """
        return {
            "accelerometer_x": pd.to_numeric(
                record_dataframe["ACC_X"], errors="coerce"
            ).fillna(0.0).to_numpy(),
            "accelerometer_y": pd.to_numeric(
                record_dataframe["ACC_Y"], errors="coerce"
            ).fillna(0.0).to_numpy(),
            "accelerometer_z": pd.to_numeric(
                record_dataframe["ACC_Z"], errors="coerce"
            ).fillna(0.0).to_numpy(),
            "temperature": pd.to_numeric(
                record_dataframe["TEMP"], errors="coerce"
            ).fillna(0.0).to_numpy(),
            "blood_volume_pulse": pd.to_numeric(
                record_dataframe["BVP"], errors="coerce"
            ).fillna(0.0).to_numpy(),
            "electrodermal_activity": pd.to_numeric(
                record_dataframe["EDA"], errors="coerce"
            ).fillna(0.0).to_numpy(),
        }

    def _extract_feature_sets_for_all_modalities(
        self,
        sensor_signals: Dict[str, np.ndarray],
        sampling_rate_hz: float,
    ) -> Dict[str, List[Dict[str, float]]]:
        """Extracts per-epoch features for every supported sensor modality.

        Args:
            sensor_signals: Dictionary of raw modality arrays.
            sampling_rate_hz: Sampling rate in Hz.

        Returns:
            A dictionary mapping modality names to lists of epoch features.
        """
        return {
            "accelerometer_x": self._extract_accelerometer_axis_epoch_features(
                sensor_signals["accelerometer_x"],
                sampling_rate_hz,
            ),
            "accelerometer_y": self._extract_accelerometer_axis_epoch_features(
                sensor_signals["accelerometer_y"],
                sampling_rate_hz,
            ),
            "accelerometer_z": self._extract_accelerometer_axis_epoch_features(
                sensor_signals["accelerometer_z"],
                sampling_rate_hz,
            ),
            "accelerometer_magnitude_deviation": self._extract_accelerometer_magnitude_deviation_epoch_features(
                sensor_signals["accelerometer_x"],
                sensor_signals["accelerometer_y"],
                sensor_signals["accelerometer_z"],
                sampling_rate_hz,
            ),
            "temperature": self._extract_temperature_epoch_features(
                sensor_signals["temperature"],
                sampling_rate_hz,
            ),
            "blood_volume_pulse": self._extract_blood_volume_pulse_epoch_features(
                sensor_signals["blood_volume_pulse"],
                sampling_rate_hz,
            ),
            "electrodermal_activity": self._extract_electrodermal_activity_epoch_features(
                sensor_signals["electrodermal_activity"],
                sampling_rate_hz,
            ),
        }

    def _count_complete_epochs(
        self,
        feature_sets: Dict[str, List[Dict[str, float]]],
    ) -> int:
        """Counts how many epochs are complete across all modalities.

        Args:
            feature_sets: Per-modality epoch feature dictionaries.

        Returns:
            The minimum number of aligned complete epochs available across all
            modalities.
        """
        return min(len(features) for features in feature_sets.values())

    def _build_epoch_feature_vector(
        self,
        feature_sets: Dict[str, List[Dict[str, float]]],
        epoch_index: int,
    ) -> List[float]:
        """Builds the final feature vector for one epoch.

        Args:
            feature_sets: Per-modality epoch feature dictionaries.
            epoch_index: Epoch index to assemble.

        Returns:
            A flat feature vector for the requested epoch.
        """
        accelerometer_x_features = feature_sets["accelerometer_x"][epoch_index]
        accelerometer_y_features = feature_sets["accelerometer_y"][epoch_index]
        accelerometer_z_features = feature_sets["accelerometer_z"][epoch_index]
        accelerometer_magnitude_deviation_features = feature_sets[
            "accelerometer_magnitude_deviation"
        ][epoch_index]
        temperature_features = feature_sets["temperature"][epoch_index]
        blood_volume_pulse_features = feature_sets["blood_volume_pulse"][epoch_index]
        electrodermal_activity_features = feature_sets["electrodermal_activity"][
            epoch_index
        ]

        features = []
        features.extend(
            accelerometer_x_features[feature_name]
            for feature_name in ["trimmed_mean", "max", "iqr"]
        )
        features.extend(
            accelerometer_y_features[feature_name]
            for feature_name in ["trimmed_mean", "max", "iqr"]
        )
        features.extend(
            accelerometer_z_features[feature_name]
            for feature_name in ["trimmed_mean", "max", "iqr"]
        )
        features.extend(
            accelerometer_magnitude_deviation_features[feature_name]
            for feature_name in ["mad"]
        )
        features.extend(
            temperature_features[feature_name]
            for feature_name in ["mean", "min", "max", "std"]
        )
        features.extend(
            blood_volume_pulse_features[feature_name]
            for feature_name in ["rmssd", "sdnn", "pnn50"]
        )
        features.extend(
            electrodermal_activity_features[feature_name]
            for feature_name in [
                "scr_amp_mean",
                "scr_amp_max",
                "scr_rise_mean",
                "scr_recovery_mean",
            ]
        )
        return features

    def _build_record_epoch_feature_matrix(
        self,
        record_dataframe: pd.DataFrame,
    ) -> List[List[float]]:
        """Builds the full epoch feature matrix for one wearable record.

        Args:
            record_dataframe: Wearable recording loaded as a pandas DataFrame.

        Returns:
            A list of epoch feature vectors. Returns an empty list if required
            sensor columns are missing.
        """
        sampling_rate_hz = float(self.sampling_rate)

        required_columns = {"ACC_X", "ACC_Y", "ACC_Z", "TEMP", "BVP", "EDA"}
        if not required_columns.issubset(record_dataframe.columns):
            return []

        sensor_signals = self._extract_sensor_signals_from_dataframe(record_dataframe)
        feature_sets = self._extract_feature_sets_for_all_modalities(
            sensor_signals,
            sampling_rate_hz,
        )
        num_epochs = self._count_complete_epochs(feature_sets)

        all_epoch_features = []
        for i in range(num_epochs):
            all_epoch_features.append(self._build_epoch_feature_vector(feature_sets, i))

        all_epoch_features = self._augment_epoch_features_with_temporal_context(
            all_epoch_features,
            gaussian_sigma=2.0,
            variance_window=5,
        )

        return all_epoch_features

    def _load_wearable_record_dataframe(self, event) -> pd.DataFrame | None:
        """Loads the wearable CSV file associated with a DREAMT event.

        Args:
            event: DREAMT event containing the ``file_64hz`` attribute.

        Returns:
            A pandas DataFrame if the file can be loaded, otherwise ``None``.
        """
        file_path = getattr(event, "file_64hz", None)
        if file_path is None:
            return None

        try:
            return pd.read_csv(file_path)
        except Exception:
            return None

    def _extract_binary_label_for_epoch(
        self,
        record_dataframe: pd.DataFrame,
        epoch_index: int,
        samples_per_epoch: int,
    ) -> int | None:
        """Extracts the binary sleep-wake label for one epoch.

        Args:
            record_dataframe: Wearable recording loaded as a pandas DataFrame.
            epoch_index: Index of the epoch to label.
            samples_per_epoch: Number of samples contained in one epoch.

        Returns:
            ``1`` for wake, ``0`` for sleep, or ``None`` if the epoch cannot be
            labeled.
        """
        start = epoch_index * samples_per_epoch
        end = start + samples_per_epoch
        epoch_dataframe = record_dataframe.iloc[start:end]

        if len(epoch_dataframe) < samples_per_epoch:
            return None

        stage_mode = epoch_dataframe["Sleep_Stage"].mode(dropna=True)
        if len(stage_mode) == 0:
            return None

        return self._convert_sleep_stage_to_binary_label(stage_mode.iloc[0])

    def _build_samples_for_sleep_event(
        self,
        patient: Patient,
        sleep_event_index: int,
        record_dataframe: pd.DataFrame,
        record_epoch_feature_matrix: List[List[float]],
        samples_per_epoch: int,
    ) -> List[Dict[str, object]]:
        """Builds epoch-level samples for one DREAMT sleep event.

        Args:
            patient: Patient containing the event.
            sleep_event_index: Index of the sleep event within the patient.
            record_dataframe: Wearable recording loaded as a pandas DataFrame.
            record_epoch_feature_matrix: Per-epoch feature matrix for the
                record.
            samples_per_epoch: Number of samples contained in one epoch.

        Returns:
            A list of task samples for the event.
        """
        samples = []
        n_labeled_epochs = len(record_dataframe) // samples_per_epoch
        n_epochs = min(len(record_epoch_feature_matrix), n_labeled_epochs)

        for epoch_idx in range(n_epochs):
            label = self._extract_binary_label_for_epoch(
                record_dataframe,
                epoch_idx,
                samples_per_epoch,
            )
            if label is None:
                continue

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": f"{patient.patient_id}-event{sleep_event_index}-epoch{epoch_idx}",
                    "epoch_index": epoch_idx,
                    "features": record_epoch_feature_matrix[epoch_idx],
                    "label": label,
                }
            )

        return samples

    def __call__(self, patient: Patient) -> List[Dict[str, object]]:
        """Processes one patient into epoch-level sleep-wake samples.

        Args:
            patient: DREAMT patient to process.

        Returns:
            A list of samples containing patient identifier, record identifier,
            epoch index, handcrafted features, and binary sleep-wake label.
        """
        samples = []
        events = patient.get_events(event_type="dreamt_sleep")
        if len(events) == 0:
            return samples

        samples_per_epoch = self.epoch_seconds * self.sampling_rate

        for event_idx, event in enumerate(events):
            record_dataframe = self._load_wearable_record_dataframe(event)
            if record_dataframe is None:
                continue

            if "Sleep_Stage" not in record_dataframe.columns:
                continue

            record_epoch_feature_matrix = self._build_record_epoch_feature_matrix(
                record_dataframe
            )
            if len(record_epoch_feature_matrix) == 0:
                continue

            samples.extend(
                self._build_samples_for_sleep_event(
                    patient=patient,
                    sleep_event_index=event_idx,
                    record_dataframe=record_dataframe,
                    record_epoch_feature_matrix=record_epoch_feature_matrix,
                    samples_per_epoch=samples_per_epoch,
                )
            )

        return samples

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import mne
import numpy as np
import torch

from pyhealth.tasks import BaseTask

EEGBCI_RUN_TYPES = {
    3: "motor_execution_left_right",
    4: "motor_imagery_left_right",
    5: "motor_execution_fists_feet",
    6: "motor_imagery_fists_feet",
    7: "motor_execution_left_right",
    8: "motor_imagery_left_right",
    9: "motor_execution_fists_feet",
    10: "motor_imagery_fists_feet",
    11: "motor_execution_left_right",
    12: "motor_imagery_left_right",
    13: "motor_execution_fists_feet",
    14: "motor_imagery_fists_feet",
}

EEGBCI_LABELS = {
    "rest": 0,
    "execute_left_fist": 1,
    "execute_right_fist": 2,
    "imagine_left_fist": 3,
    "imagine_right_fist": 4,
    "execute_both_fists": 5,
    "execute_both_feet": 6,
    "imagine_both_fists": 7,
    "imagine_both_feet": 8,
}


def run_type_for_run(run: int) -> str:
    try:
        return EEGBCI_RUN_TYPES[int(run)]
    except KeyError as exc:
        raise ValueError(f"Unsupported EEGBCI run: {run}") from exc


def label_family_for_run(run: int) -> str:
    run_type = run_type_for_run(run)
    if "execution" in run_type:
        return "motor_execution"
    if "imagery" in run_type:
        return "motor_imagery"
    return "baseline"


def task_label_for_event(run: int, event_code: str) -> str:
    code = str(event_code).strip()
    if code == "T0":
        return "rest"
    run_type = run_type_for_run(run)
    mapping = {
        "motor_execution_left_right": {
            "T1": "execute_left_fist",
            "T2": "execute_right_fist",
        },
        "motor_imagery_left_right": {
            "T1": "imagine_left_fist",
            "T2": "imagine_right_fist",
        },
        "motor_execution_fists_feet": {
            "T1": "execute_both_fists",
            "T2": "execute_both_feet",
        },
        "motor_imagery_fists_feet": {
            "T1": "imagine_both_fists",
            "T2": "imagine_both_feet",
        },
    }
    try:
        return mapping[run_type][code]
    except KeyError as exc:
        raise ValueError(f"Unsupported EEGBCI event {event_code!r} for run {run}") from exc


def numeric_label_for_task(task_label: str) -> int:
    try:
        return EEGBCI_LABELS[task_label]
    except KeyError as exc:
        raise ValueError(f"Unsupported EEGBCI task label: {task_label}") from exc


EEGBCI_COMPAT_CHANNELS = (
    "FC5",
    "FC3",
    "FC1",
    "FC2",
    "FC4",
    "FC6",
    "C5",
    "C3",
    "C1",
    "C2",
    "C4",
    "C6",
    "CP5",
    "CP3",
    "CP4",
    "CP6",
)


def normalize_eegbci_channel_name(name: str) -> str:
    clean = name.upper().replace(".", "").replace("EEG ", "").replace("-REF", "")
    aliases = {
        "T9": "FT9",
        "T10": "FT10",
    }
    return aliases.get(clean, clean)


def select_eegbci_channels(
    data: np.ndarray,
    ch_names: List[str],
    channel_mode: str = "compat16",
) -> Tuple[np.ndarray, List[str]]:
    if channel_mode == "all":
        return data, list(ch_names)
    if channel_mode != "compat16":
        raise ValueError("channel_mode must be one of {'compat16', 'all'}")

    normalized_to_index = {
        normalize_eegbci_channel_name(name): idx for idx, name in enumerate(ch_names)
    }
    missing = [ch for ch in EEGBCI_COMPAT_CHANNELS if ch not in normalized_to_index]
    if missing:
        raise ValueError(f"Missing EEGBCI channels for compat16 mode: {missing}")
    indices = [normalized_to_index[ch] for ch in EEGBCI_COMPAT_CHANNELS]
    return data[indices], list(EEGBCI_COMPAT_CHANNELS)


def normalize_signal(signal: np.ndarray, mode: str | None) -> np.ndarray:
    if mode is None:
        return signal
    if mode == "95th_percentile":
        scale = np.quantile(
            np.abs(signal), q=0.95, axis=-1, method="linear", keepdims=True
        )
        return signal / (scale + 1e-8)
    if mode == "div_by_100":
        return signal / 100.0
    raise ValueError("normalization must be one of {None, '95th_percentile', 'div_by_100'}")


BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def compute_band_powers(data: np.ndarray, sfreq: float) -> Dict[str, float | str]:
    from scipy.signal import welch

    if data.ndim != 2:
        raise ValueError("data must have shape (channels, time)")
    nperseg = min(data.shape[-1], int(sfreq * 2))
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
    mean_psd = psd.mean(axis=0)

    features: Dict[str, float | str] = {}
    total_power = 0.0
    band_values: Dict[str, float] = {}
    for band, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        value = float(np.trapezoid(mean_psd[mask], freqs[mask])) if np.any(mask) else 0.0
        features[f"{band}_power"] = value
        band_values[band] = value
        total_power += value

    denom = total_power + 1e-12
    for band, value in band_values.items():
        features[f"{band}_relative"] = float(value / denom)

    features["dominant_band"] = max(band_values, key=band_values.get)
    features["alpha_beta_ratio"] = float(
        band_values["alpha"] / (band_values["beta"] + 1e-12)
    )
    features["theta_beta_ratio"] = float(
        band_values["theta"] / (band_values["beta"] + 1e-12)
    )
    return features


def interpret_band_profile(features: Dict[str, float | str]) -> Dict[str, str]:
    dominant = str(features["dominant_band"])
    alpha_rel = float(features.get("alpha_relative", 0.0))
    beta_rel = float(features.get("beta_relative", 0.0))
    theta_rel = float(features.get("theta_relative", 0.0))
    gamma_rel = float(features.get("gamma_relative", 0.0))
    alpha_beta = float(features.get("alpha_beta_ratio", 0.0))
    theta_beta = float(features.get("theta_beta_ratio", 0.0))

    quality_flags: List[str] = []
    hypothesis = "mixed_frequency_profile"
    confidence = "low"

    if dominant == "alpha" and alpha_rel >= 0.45 and alpha_beta >= 2.0:
        hypothesis = "relaxed_or_idle"
        confidence = "medium"
    elif dominant == "beta" and beta_rel >= 0.35:
        hypothesis = "active_sensorimotor_processing"
        confidence = "medium"
    elif dominant == "theta" and theta_rel >= 0.35 and theta_beta >= 1.5:
        hypothesis = "slow_wave_or_drowsy_pattern"
        confidence = "medium"
    elif dominant == "gamma" and gamma_rel >= 0.30:
        hypothesis = "high_frequency_or_artifact_pattern"
        confidence = "low"
        quality_flags.append("possible_muscle_artifact")

    if confidence == "low":
        quality_flags.append("low_confidence")

    return {
        "brain_state_hypothesis": hypothesis,
        "confidence": confidence,
        "quality_flags": ";".join(quality_flags) if quality_flags else "none",
        "interpretation": (
            f"The segment is consistent with {hypothesis} based on a "
            f"{dominant}-dominant frequency profile. This is exploratory signal "
            "metadata, not evidence of cognition or a clinical diagnosis."
        ),
    }


def iter_annotation_windows(
    raw: mne.io.BaseRaw,
    run: int,
    window_size: float = 2.0,
) -> List[Dict[str, Any]]:
    sfreq = float(raw.info["sfreq"])
    window_samples = int(round(window_size * sfreq))
    windows: List[Dict[str, Any]] = []
    for idx, annotation in enumerate(raw.annotations):
        event_code = str(annotation["description"])
        if event_code not in {"T0", "T1", "T2"}:
            continue
        start_sample = int(round(float(annotation["onset"]) * sfreq))
        duration_samples = int(round(float(annotation["duration"]) * sfreq))
        n_full_windows = duration_samples // window_samples
        for window_idx in range(n_full_windows):
            s0 = start_sample + window_idx * window_samples
            s1 = s0 + window_samples
            task_label = task_label_for_event(run, event_code)
            windows.append(
                {
                    "trial_id": f"ann{idx:04d}_win{window_idx:03d}",
                    "event_code": event_code,
                    "task_label": task_label,
                    "label_family": label_family_for_run(run),
                    "label": numeric_label_for_task(task_label),
                    "start_time": s0 / sfreq,
                    "end_time": s1 / sfreq,
                    "start_sample": s0,
                    "end_sample": s1,
                }
            )
    return windows


class EEGMotorImageryEEGBCI(BaseTask):
    task_name: str = "EEGBCI_motor_imagery"
    input_schema: Dict[str, str] = {"signal": "tensor", "stft": "tensor"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __init__(
        self,
        window_size: float = 2.0,
        resample_rate: float | None = 200,
        bandpass_filter: Tuple[float, float] | None = (0.5, 45.0),
        channel_mode: str = "compat16",
        normalization: str | None = "95th_percentile",
        compute_stft: bool = True,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.resample_rate = resample_rate
        self.bandpass_filter = bandpass_filter
        self.channel_mode = channel_mode
        self.normalization = normalization
        self.compute_stft = compute_stft
        if not compute_stft:
            self.input_schema = {"signal": "tensor"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        return self._base_samples_from_patient(patient)

    def read_raw(self, signal_file: str) -> mne.io.BaseRaw:
        raw = mne.io.read_raw_edf(signal_file, preload=True, verbose="error")
        raw.pick_types(eeg=True, stim=False, exclude=[])
        if self.bandpass_filter is not None:
            raw.filter(
                l_freq=self.bandpass_filter[0],
                h_freq=self.bandpass_filter[1],
                verbose="error",
            )
        if self.resample_rate is not None:
            raw.resample(self.resample_rate, n_jobs=1, verbose="error")
        return raw

    def _base_samples_from_patient(self, patient: Any) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for event in patient.get_events("records"):
            raw = self.read_raw(event.signal_file)
            data = raw.get_data(units="uV")
            selected, selected_names = select_eegbci_channels(
                data, raw.ch_names, self.channel_mode
            )
            selected = normalize_signal(selected, self.normalization)
            sfreq = float(raw.info["sfreq"])
            for idx, window in enumerate(
                iter_annotation_windows(raw, int(event.run), self.window_size)
            ):
                signal_np = selected[:, window["start_sample"] : window["end_sample"]]
                if signal_np.shape[-1] != int(round(self.window_size * sfreq)):
                    continue
                signal = torch.FloatTensor(signal_np)
                sample = {
                    "patient_id": patient.patient_id,
                    "record_id": event.record_id,
                    "subject_id": int(event.subject_id),
                    "run": int(event.run),
                    "run_type": event.run_type,
                    "signal_file": event.signal_file,
                    "trial_id": f"{patient.patient_id}_{event.record_id}_{idx:04d}",
                    "event_code": window["event_code"],
                    "task_label": window["task_label"],
                    "label_family": window["label_family"],
                    "label": int(window["label"]),
                    "signal": signal,
                    "channel_names": selected_names,
                    "start_time": window["start_time"],
                    "end_time": window["end_time"],
                    "sample_rate": sfreq,
                }
                if self.compute_stft:
                    from pyhealth.models.tfm_tokenizer import get_stft_torch

                    sample["stft"] = get_stft_torch(signal.unsqueeze(0)).squeeze(0)
                samples.append(sample)
            raw.close()
        return samples


class EEGBCIPatternDiscovery(EEGMotorImageryEEGBCI):
    task_name: str = "EEGBCI_pattern_discovery"

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        samples = self._base_samples_from_patient(patient)
        for sample in samples:
            features = compute_band_powers(
                sample["signal"].detach().cpu().numpy(),
                float(sample["sample_rate"]),
            )
            interpretation = interpret_band_profile(features)
            sample["bandpower"] = features
            sample.update(interpretation)
        return samples

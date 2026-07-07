from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

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

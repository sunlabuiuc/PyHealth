import numpy as np
import torch
import mne
from typing import Any, Dict, List, Tuple
import numpy as np
from pyhealth.tasks import BaseTask


def read_edf_uv(
    file_name: str,
    bandpass_filter: tuple[float, float] | None = None,
    notch_filter: float | None = None,
    resample_rate: float | None = None,
) -> tuple[np.ndarray, list[str], float]:
    """Read an EDF in microvolts, with optional filtering and resampling.

    Args:
        file_name: Path to an EDF recording.
        bandpass_filter: ``(low, high)`` in Hz, or None to skip it.
        notch_filter: Notch frequency in Hz, or None to skip it.
        resample_rate: Target rate in Hz, or None to keep the native rate.

    Returns:
        ``(data, ch_names, sfreq)``: data shaped ``(n_channels, n_samples)``
        in uV, the channel names, and the rate of ``data`` in Hz.

    Examples:
        >>> from pyhealth.tasks.temple_university_EEG_tasks import read_edf_uv
        >>> data, ch_names, sfreq = read_edf_uv(
        ..."""

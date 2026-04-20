"""
PyHealth task for respiratory sound classification on ICBHI 2017.

Consumes the cycle-level event stream produced by
:class:`~pyhealth.datasets.ICBHIDataset` and emits one sample per
respiratory cycle with a 4-class label: Normal, Crackle, Wheeze, or Both.

Dataset link:
    https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge

Task paper: (please cite if you use this task)
    Rocha, Bruno M., et al. "An open access database for the evaluation of
    respiratory sound classification algorithms." Physiological Measurement
    40.3 (2019): 035001.

Paper link:
    https://doi.org/10.1088/1361-6579/ab03ea

Author:
    Andrew Zhao (andrew.zhao@aeroseal.com)
"""

from math import gcd
from typing import Any, Dict, List

import numpy as np
import torch

from pyhealth.tasks import BaseTask


# (crackle, wheeze) → integer class index
_LABEL_MAP: Dict[tuple, int] = {
    (0, 0): 0,  # Normal
    (1, 0): 1,  # Crackle
    (0, 1): 2,  # Wheeze
    (1, 1): 3,  # Both
}

LABEL_NAMES: List[str] = ["Normal", "Crackle", "Wheeze", "Both"]


class ICBHIRespiratoryTask(BaseTask):
    """4-class cycle-level respiratory sound classification on ICBHI 2017.

    For each respiratory cycle event emitted by
    :class:`~pyhealth.datasets.ICBHIDataset`, this task:

    1. Reads the parent WAV at ``event.audio_path``.
    2. Slices the audio to the window ``[cycle_start, cycle_end]``.
    3. Resamples to ``resample_rate`` Hz using integer rational resampling.
    4. Zero-pads or truncates to exactly ``target_length`` seconds so all
       signals share the same shape ``(1, target_length * resample_rate)``.
    5. Returns one sample dict per cycle with the 4-class label.

    **Label mapping:**

    +------------+----------+--------+-------+
    | Class name | Crackle  | Wheeze | Index |
    +============+==========+========+=======+
    | Normal     | 0        | 0      | 0     |
    +------------+----------+--------+-------+
    | Crackle    | 1        | 0      | 1     |
    +------------+----------+--------+-------+
    | Wheeze     | 0        | 1      | 2     |
    +------------+----------+--------+-------+
    | Both       | 1        | 1      | 3     |
    +------------+----------+--------+-------+

    Args:
        resample_rate: Target sample rate in Hz. Default 4000 Hz (low enough
            to cover the respiratory frequency range while keeping tensors
            small).
        target_length: Fixed cycle duration in seconds. Cycles shorter than
            this are zero-padded; longer cycles are truncated. Default 5 s.

    Attributes:
        task_name (str): ``"ICBHIRespiratoryClassification"``
        input_schema (Dict[str, str]): ``{"signal": "tensor"}``
        output_schema (Dict[str, str]): ``{"label": "multiclass"}``

    Examples:
        >>> from pyhealth.datasets import ICBHIDataset
        >>> from pyhealth.tasks import ICBHIRespiratoryTask
        >>> dataset = ICBHIDataset(root="/data/ICBHI_final_database")
        >>> task = ICBHIRespiratoryTask()
        >>> samples = dataset.set_task(task)
        >>> sample = samples[0]
        >>> print(sample["signal"].shape)   # torch.Size([1, 20000])
        >>> print(sample["label"])          # 0, 1, 2, or 3
    """

    task_name: str = "ICBHIRespiratoryClassification"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __init__(
        self,
        resample_rate: int = 4000,
        target_length: float = 5.0,
    ) -> None:
        super().__init__()
        self.resample_rate = resample_rate
        self.target_length = target_length

    @staticmethod
    def _read_wav(path: str):
        """Read a WAV file and return (sample_rate, float32 mono array).

        Args:
            path: Path to the ``.wav`` file.

        Returns:
            Tuple of (sample_rate: int, data: np.ndarray[float32]).
        """
        import scipy.io.wavfile

        rate, data = scipy.io.wavfile.read(path)
        if data.ndim > 1:
            data = data[:, 0]
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0
        else:
            data = data.astype(np.float32)
        return rate, data

    @staticmethod
    def _resample(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """Resample ``data`` from ``src_rate`` to ``dst_rate`` using
        rational resampling (polyphase filter) via
        ``scipy.signal.resample_poly``.

        Args:
            data: 1-D float32 audio array.
            src_rate: Original sample rate in Hz.
            dst_rate: Target sample rate in Hz.

        Returns:
            Resampled float32 array.
        """
        if src_rate == dst_rate:
            return data
        from scipy.signal import resample_poly

        g = gcd(dst_rate, src_rate)
        return resample_poly(data, dst_rate // g, src_rate // g).astype(np.float32)

    def _pad_or_trim(self, data: np.ndarray) -> np.ndarray:
        """Zero-pad or truncate ``data`` to exactly
        ``target_length * resample_rate`` samples."""
        n_target = int(self.target_length * self.resample_rate)
        if len(data) >= n_target:
            return data[:n_target]
        pad = np.zeros(n_target - len(data), dtype=np.float32)
        return np.concatenate([data, pad])

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Generate one sample per respiratory cycle event.

        Args:
            patient: A :class:`~pyhealth.data.Patient` object with
                ``"train"`` and/or ``"test"`` cycle events loaded from
                :class:`~pyhealth.datasets.ICBHIDataset`.

        Returns:
            List of sample dicts, each with:

            - ``patient_id`` (str)
            - ``record_id`` (str) — the recording id
            - ``split`` (str): ``"train"`` or ``"test"``
            - ``begin_time`` (float): cycle start in seconds
            - ``end_time`` (float): cycle end in seconds
            - ``signal`` (torch.FloatTensor): shape
              ``(1, target_length * resample_rate)``
            - ``label`` (int): 0 Normal / 1 Crackle / 2 Wheeze / 3 Both
        """
        pid = patient.patient_id
        samples: List[Dict[str, Any]] = []
        # Cache decoded audio so recordings with many cycles are only
        # read + resampled once per task invocation.
        audio_cache: Dict[str, tuple] = {}

        for split in ("train", "test"):
            events = patient.get_events(split)

            for event in events:
                audio_path: str = event.audio_path
                if audio_path in audio_cache:
                    src_rate, audio = audio_cache[audio_path]
                else:
                    try:
                        src_rate, audio = self._read_wav(audio_path)
                    except Exception:
                        continue
                    audio_cache[audio_path] = (src_rate, audio)

                cycle_start = float(event.cycle_start)
                cycle_end = float(event.cycle_end)
                start_sample = int(cycle_start * src_rate)
                end_sample = int(cycle_end * src_rate)
                cycle = audio[start_sample:end_sample]
                if len(cycle) == 0:
                    continue

                cycle = self._resample(cycle, src_rate, self.resample_rate)
                cycle = self._pad_or_trim(cycle)
                signal = torch.FloatTensor(cycle).unsqueeze(0)  # (1, T)

                label = _LABEL_MAP.get(
                    (int(event.has_crackles), int(event.has_wheezes)), 0
                )

                samples.append(
                    {
                        "patient_id": pid,
                        "record_id": event.recording_id,
                        "split": split,
                        "begin_time": cycle_start,
                        "end_time": cycle_end,
                        "signal": signal,
                        "label": label,
                    }
                )

        return samples

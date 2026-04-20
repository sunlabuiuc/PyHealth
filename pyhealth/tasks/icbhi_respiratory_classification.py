"""
PyHealth task for respiratory-abnormality prediction on ICBHI 2017.

This task follows the screening framing used in the RespLLM paper
("Unifying Audio and Text with Multimodal LLMs for Generalized Respiratory
Health Prediction"): predict whether a respiratory cycle contains an
adventitious sound, using the crackle / wheeze supervision that the
ICBHI 2017 Respiratory Sound Database provides at cycle granularity.

Three ablation modes are supported via ``label_mode``:

- ``"any_abnormal"`` — label is 1 if ``has_crackles`` OR ``has_wheezes``
- ``"crackle_only"`` — label is 1 if ``has_crackles``
- ``"wheeze_only"`` — label is 1 if ``has_wheezes``

All three are binary and derive deterministically from the raw ICBHI
annotations. No labels are invented or smoothed.

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
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from pyhealth.tasks import BaseTask


# Historical 4-class (crackle, wheeze) -> class index mapping. Kept at
# module scope as the canonical description of ICBHI's raw cycle annotation
# space. The current task operates in binary mode via ``label_mode``; this
# constant is retained for reference and backward compatibility.
_LABEL_MAP: Dict[tuple, int] = {
    (0, 0): 0,  # Normal
    (1, 0): 1,  # Crackle
    (0, 1): 2,  # Wheeze
    (1, 1): 3,  # Both
}

LABEL_NAMES: List[str] = ["Normal", "Crackle", "Wheeze", "Both"]


# Supported ablation modes for RespiratoryAbnormalityPredictionICBHI.
_VALID_LABEL_MODES: Tuple[str, ...] = (
    "any_abnormal",
    "crackle_only",
    "wheeze_only",
)

# (negative_name, positive_name) per label mode — used to populate the
# ``label_name`` field in each sample.
_LABEL_NAMES_BY_MODE: Dict[str, Tuple[str, str]] = {
    "any_abnormal": ("normal", "abnormal"),
    "crackle_only": ("no_crackle", "crackle"),
    "wheeze_only": ("no_wheeze", "wheeze"),
}


class RespiratoryAbnormalityPredictionICBHI(BaseTask):
    """Binary respiratory-abnormality prediction on ICBHI 2017.

    Consumes the cycle-level event stream produced by
    :class:`~pyhealth.datasets.ICBHIDataset` and emits one sample per
    respiratory cycle with a binary label computed from the raw
    ``has_crackles`` / ``has_wheezes`` flags. The specific mapping is
    controlled by ``label_mode``:

    +-----------------+------------------------------------+-------------------+
    | ``label_mode``  | positive-class definition          | label names       |
    +=================+====================================+===================+
    | any_abnormal    | ``has_crackles OR has_wheezes``    | normal / abnormal |
    +-----------------+------------------------------------+-------------------+
    | crackle_only    | ``has_crackles``                   | no_crackle /      |
    |                 |                                    | crackle           |
    +-----------------+------------------------------------+-------------------+
    | wheeze_only    | ``has_wheezes``                     | no_wheeze /       |
    |                 |                                    | wheeze            |
    +-----------------+------------------------------------+-------------------+

    **Per-cycle signal pipeline:**

    1. Read the parent WAV at ``event.audio_path`` (cached per call).
    2. Slice to ``[event.cycle_start, event.cycle_end]``.
    3. Resample to ``resample_rate`` Hz via ``scipy.signal.resample_poly``.
    4. Zero-pad or truncate to ``target_length`` seconds so every sample
       has the fixed shape ``(1, target_length * resample_rate)``.

    Args:
        label_mode: Which binary abnormality label to emit. One of
            ``"any_abnormal"``, ``"crackle_only"``, ``"wheeze_only"``.
            Default ``"any_abnormal"``.
        resample_rate: Target sample rate in Hz. Default 4000 — low enough
            to cover respiratory acoustic content while keeping tensors
            small.
        target_length: Fixed cycle duration in seconds. Cycles shorter
            than this are zero-padded; longer cycles are truncated.
            Default 5 s.

    Raises:
        ValueError: If ``label_mode`` is not one of the supported values.

    Attributes:
        task_name (str): ``"RespiratoryAbnormalityPredictionICBHI"``
        input_schema (Dict[str, str]): ``{"signal": "tensor"}``
        output_schema (Dict[str, str]): ``{"label": "binary"}``
        label_mode (str): The selected ablation mode.
        resample_rate (int): Audio resample rate in Hz.
        target_length (float): Fixed cycle length in seconds.

    Examples:
        >>> from pyhealth.datasets import ICBHIDataset
        >>> from pyhealth.tasks import RespiratoryAbnormalityPredictionICBHI
        >>> dataset = ICBHIDataset(root="/data/ICBHI_final_database")
        >>> task = RespiratoryAbnormalityPredictionICBHI(label_mode="any_abnormal")
        >>> samples = dataset.set_task(task)
        >>> sample = samples[0]
        >>> sample["signal"].shape       # torch.Size([1, 20000])
        >>> sample["label"], sample["label_name"]   # (0 or 1, "normal" or "abnormal")
    """

    task_name: str = "RespiratoryAbnormalityPredictionICBHI"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(
        self,
        label_mode: str = "any_abnormal",
        resample_rate: int = 4000,
        target_length: float = 5.0,
    ) -> None:
        if label_mode not in _VALID_LABEL_MODES:
            raise ValueError(
                f"label_mode must be one of {_VALID_LABEL_MODES}, "
                f"got {label_mode!r}"
            )
        super().__init__()
        self.label_mode = label_mode
        self.resample_rate = resample_rate
        self.target_length = target_length

    def _compute_label(self, has_crackles: int, has_wheezes: int) -> int:
        """Map raw (crackle, wheeze) flags to a binary label per ``label_mode``."""
        c = 1 if int(has_crackles) else 0
        w = 1 if int(has_wheezes) else 0
        if self.label_mode == "any_abnormal":
            return int(c or w)
        if self.label_mode == "crackle_only":
            return c
        if self.label_mode == "wheeze_only":
            return w
        # Unreachable: __init__ already validated label_mode.
        raise ValueError(f"Unsupported label_mode: {self.label_mode!r}")

    def _label_name(self, label: int) -> str:
        """Return the human-readable name for ``label`` under the current mode."""
        names = _LABEL_NAMES_BY_MODE[self.label_mode]
        return names[1] if label else names[0]

    @staticmethod
    def _read_wav(path: str) -> Tuple[int, np.ndarray]:
        """Read a WAV file and return ``(sample_rate, float32 mono array)``."""
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
        """Rational resample from ``src_rate`` to ``dst_rate``."""
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
            - ``recording_id`` (str)
            - ``cycle_id`` (int)
            - ``audio_path`` (str)
            - ``segment_start`` (float): cycle start in seconds
            - ``segment_end`` (float): cycle end in seconds
            - ``duration`` (float): cycle length in seconds
            - ``split`` (str): ``"train"`` or ``"test"``
            - ``signal`` (torch.FloatTensor): shape
              ``(1, target_length * resample_rate)``
            - ``label`` (int): 0 / 1 under the selected ``label_mode``
            - ``label_name`` (str): human-readable label string
            - ``metadata_text`` (str, optional): forwarded from the event
              if the dataset provides it
        """
        pid = patient.patient_id
        samples: List[Dict[str, Any]] = []
        # Cache decoded audio so recordings with many cycles are only
        # read + resampled once per task invocation.
        audio_cache: Dict[str, Tuple[int, np.ndarray]] = {}

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

                segment_start = float(event.cycle_start)
                segment_end = float(event.cycle_end)
                duration = segment_end - segment_start

                start_sample = int(segment_start * src_rate)
                end_sample = int(segment_end * src_rate)
                cycle = audio[start_sample:end_sample]
                if len(cycle) == 0:
                    continue

                cycle = self._resample(cycle, src_rate, self.resample_rate)
                cycle = self._pad_or_trim(cycle)
                signal = torch.FloatTensor(cycle).unsqueeze(0)  # (1, T)

                has_crackles = int(event.has_crackles)
                has_wheezes = int(event.has_wheezes)
                label = self._compute_label(has_crackles, has_wheezes)
                label_name = self._label_name(label)

                sample: Dict[str, Any] = {
                    "patient_id": pid,
                    "recording_id": event.recording_id,
                    "cycle_id": int(event.cycle_id),
                    "audio_path": audio_path,
                    "segment_start": segment_start,
                    "segment_end": segment_end,
                    "duration": duration,
                    "split": split,
                    "signal": signal,
                    "label": label,
                    "label_name": label_name,
                }
                metadata_text = getattr(event, "metadata_text", None)
                if metadata_text is not None:
                    sample["metadata_text"] = metadata_text
                samples.append(sample)

        return samples


# Backward-compatible alias. The previous name is still used throughout the
# repo (dataset.default_task, example script, docs, existing tests). The
# refactored task is now binary abnormality prediction with configurable
# label_mode; callers that instantiated with no arguments get the
# ``any_abnormal`` mode by default.
ICBHIRespiratoryTask = RespiratoryAbnormalityPredictionICBHI

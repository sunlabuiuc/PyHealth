"""Cardio-respiratory sleep staging on NSRR cohorts (SHHS, MESA).

Reimplements the data-preparation side of wav2sleep (Carter & Tarassenko,
ML4H 2024, https://arxiv.org/abs/2411.04644): ECG / PPG / thoracic and
abdominal respiratory-effort signals are cut into 30 s epochs, resampled to a
fixed number of points per epoch and paired with 4-class sleep stage labels.
"""

import logging
import os
import xml.etree.ElementTree as ET
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import resample_poly

from pyhealth.tasks.base_task import BaseTask

logger = logging.getLogger(__name__)

EPOCH_SECONDS: int = 30

#: Points per 30 s epoch after resampling (wav2sleep, Sec. 4.1):
#: ~34 Hz for ECG/PPG and ~8.5 Hz for respiratory effort.
POINTS_PER_EPOCH: Dict[str, int] = {"ecg": 1024, "ppg": 1024, "thx": 256, "abd": 256}

#: Accepted (lower-cased) EDF channel labels per modality. SHHS uses e.g.
#: ``ECG`` / ``THOR RES`` / ``ABDO RES``; MESA uses ``EKG`` / ``Pleth`` /
#: ``Thor`` / ``Abdo``.
CHANNEL_ALIASES: Dict[str, Tuple[str, ...]] = {
    "ecg": ("ecg", "ekg"),
    "ppg": ("pleth", "ppg"),
    "thx": ("thor res", "thor", "thx"),
    "abd": ("abdo res", "abdo", "abd"),
}

#: NSRR profusion stage codes -> label for each supported label scheme.
#: Codes not listed (e.g. 9 = unscored) are dropped.
LABEL_SCHEMES: Dict[str, Dict[int, int]] = {
    # Wake / Light (N1+N2) / Deep (N3+N4) / REM -- the wav2sleep classes.
    "four_class": {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3},
    # Wake / N1 / N2 / N3(+N4) / REM -- the AASM classes.
    "five_class": {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4},
    # Wake / Sleep.
    "sleep_wake": {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
}


def _read_edf_channels(
    path: str, labels: Sequence[str]
) -> Dict[str, Tuple[np.ndarray, float]]:
    """Reads selected channels of an EDF file at their native sampling rate.

    Unlike ``mne.io.read_raw_edf``, channels recorded at different rates are
    *not* resampled to a common rate, so low-rate respiratory channels are not
    needlessly upsampled.

    Args:
        path: Path to the EDF file.
        labels: Lower-cased channel labels to read. Labels that are not in the
            file are omitted from the result.

    Returns:
        Mapping from lower-cased label to ``(signal, sampling_rate_hz)`` where
        ``signal`` is a float32 array in physical units.

    Raises:
        ValueError: If the file is not a valid EDF file.
    """
    with open(path, "rb") as f:
        head = f.read(256)
        if len(head) < 256 or not head[:8].startswith(b"0"):
            raise ValueError(f"{path} is not a valid EDF file")
        header_bytes = int(head[184:192])
        n_records = int(head[236:244])
        record_dur = float(head[244:252])
        n_signals = int(head[252:256])
        fields = f.read(n_signals * 256)

    names = ["label", "transducer", "unit", "pmin", "pmax", "dmin", "dmax"]
    widths = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
    names += ["prefilter", "nsamp", "reserved"]
    cols: Dict[str, List[str]] = {}
    offset = 0
    for name, width in zip(names, widths):
        cols[name] = [
            fields[offset + i * width : offset + (i + 1) * width]
            .decode("ascii", "ignore")
            .strip()
            for i in range(n_signals)
        ]
        offset += n_signals * width

    nsamp = [int(x) for x in cols["nsamp"]]
    per_record = sum(nsamp)
    if per_record == 0 or record_dur <= 0:
        raise ValueError(f"{path} has an invalid EDF header")

    n_available = (os.path.getsize(path) - header_bytes) // (2 * per_record)
    if n_records <= 0 or n_records > n_available:
        n_records = n_available  # unknown or truncated record count

    starts = np.concatenate(([0], np.cumsum(nsamp)))
    wanted = {lab.lower() for lab in labels}
    out: Dict[str, Tuple[np.ndarray, float]] = {}
    data = np.memmap(
        path,
        dtype="<i2",
        mode="r",
        offset=header_bytes,
        shape=(n_records, per_record),
    )
    try:
        for i, label in enumerate(cols["label"]):
            key = label.lower()
            if key not in wanted or key in out:
                continue
            raw = np.array(data[:, starts[i] : starts[i + 1]]).reshape(-1)
            dmin, dmax = float(cols["dmin"][i]), float(cols["dmax"][i])
            pmin, pmax = float(cols["pmin"][i]), float(cols["pmax"][i])
            gain = (pmax - pmin) / (dmax - dmin) if dmax != dmin else 1.0
            signal = (raw.astype(np.float32) - dmin) * gain + pmin
            out[key] = (signal.astype(np.float32), nsamp[i] / record_dur)
    finally:
        del data  # release the memory map so the file can be removed on Windows
    return out


def _read_stage_codes(path: str) -> np.ndarray:
    """Reads per-epoch sleep stage codes from an NSRR profusion XML file.

    Args:
        path: Path to ``*-profusion.xml``.

    Returns:
        Integer array with one stage code per 30 s epoch.

    Raises:
        ValueError: If no stages are found or the epoch length is not 30 s.
    """
    root = ET.parse(path).getroot()
    length = root.find(".//EpochLength")
    if length is not None and length.text and int(length.text) != EPOCH_SECONDS:
        raise ValueError(f"{path}: expected {EPOCH_SECONDS} s epochs")
    stages = [int(s.text) for s in root.iter("SleepStage") if s.text]
    if not stages:
        raise ValueError(f"{path} contains no <SleepStage> entries")
    return np.asarray(stages, dtype=np.int64)


def _to_epochs(
    signal: np.ndarray, rate: float, n_epochs: int, points: int, normalize: bool
) -> np.ndarray:
    """Cuts a signal into epochs and resamples each to ``points`` samples."""
    native = rate * EPOCH_SECONDS
    if abs(native - round(native)) > 1e-6:
        raise ValueError(f"{rate} Hz does not give a whole number of samples/epoch")
    native = int(round(native))
    x = signal[: n_epochs * native].astype(np.float64)
    if normalize:
        x = x - x.mean()
        std = x.std()
        if std > 0:
            x = x / std
    ratio = Fraction(points, native)
    y = resample_poly(x, ratio.numerator, ratio.denominator)
    return y.reshape(n_epochs, points).astype(np.float32)


class SleepStagingCardioRespiratory(BaseTask):
    """Sleep staging from cardio-respiratory signals (wav2sleep-style).

    Each 30 s epoch of a night recorded in an NSRR cohort becomes one sample
    holding the requested signal modalities (each resampled to a fixed number
    of points per epoch) and a sleep stage label.

    Attributes:
        task_name (str): ``"SleepStagingCardioRespiratory"``.
        input_schema (Dict[str, str]): One ``"tensor"`` entry per modality.
        output_schema (Dict[str, str]): ``{"label": "multiclass"}``.

    Args:
        modalities: Signals to extract, a subset of ``"ecg"``, ``"ppg"``,
            ``"thx"`` (thorax effort) and ``"abd"`` (abdomen effort).
            Recordings lacking a requested channel are skipped with a warning
            (SHHS has no PPG).
        label_scheme: ``"four_class"`` (Wake/Light/Deep/REM, default),
            ``"five_class"`` (Wake/N1/N2/N3/REM) or ``"sleep_wake"``.
        normalize: If True, z-score each signal over the whole night before
            resampling.

    Examples:
        >>> from pyhealth.datasets import NSRRSleepDataset
        >>> from pyhealth.tasks import SleepStagingCardioRespiratory
        >>> dataset = NSRRSleepDataset(root="/data/shhs", cohort="shhs")
        >>> task = SleepStagingCardioRespiratory(modalities=("ecg", "thx"))
        >>> samples = dataset.set_task(task)
        >>> samples[0]["ecg"].shape  # one epoch, 1024 points
        torch.Size([1024])
    """

    task_name: str = "SleepStagingCardioRespiratory"
    input_schema: Dict[str, str] = {"ecg": "tensor", "thx": "tensor", "abd": "tensor"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __init__(
        self,
        modalities: Sequence[str] = ("ecg", "thx", "abd"),
        label_scheme: str = "four_class",
        normalize: bool = True,
    ) -> None:
        modalities = tuple(modalities)
        unknown = [m for m in modalities if m not in POINTS_PER_EPOCH]
        if not modalities or unknown:
            raise ValueError(
                f"modalities must be a non-empty subset of "
                f"{sorted(POINTS_PER_EPOCH)}; got {modalities}"
            )
        if len(set(modalities)) != len(modalities):
            raise ValueError(f"duplicate modalities in {modalities}")
        if label_scheme not in LABEL_SCHEMES:
            raise ValueError(
                f"label_scheme must be one of {sorted(LABEL_SCHEMES)}; "
                f"got {label_scheme!r}"
            )
        self.modalities = modalities
        self.label_scheme = label_scheme
        self.normalize = normalize
        self.input_schema = {m: "tensor" for m in modalities}
        super().__init__()

    def _process_recording(self, event: Any, pid: str) -> List[Dict[str, Any]]:
        aliases = [a for m in self.modalities for a in CHANNEL_ALIASES[m]]
        channels = _read_edf_channels(event.signal_file, aliases)
        stages = _read_stage_codes(event.label_file)

        found: Dict[str, Tuple[np.ndarray, float]] = {}
        for modality in self.modalities:
            label = next((a for a in CHANNEL_ALIASES[modality] if a in channels), None)
            if label is None:
                raise ValueError(f"no {modality!r} channel in {event.signal_file}")
            found[modality] = channels[label]

        n_epochs = min(
            [len(stages)]
            + [int(len(sig) // (rate * EPOCH_SECONDS)) for sig, rate in found.values()]
        )
        epochs = {
            m: _to_epochs(sig, rate, n_epochs, POINTS_PER_EPOCH[m], self.normalize)
            for m, (sig, rate) in found.items()
        }

        mapping = LABEL_SCHEMES[self.label_scheme]
        visit = getattr(event, "visit", None)
        samples = []
        for i in range(n_epochs):
            label = mapping.get(int(stages[i]))
            if label is None:  # unscored epoch
                continue
            sample: Dict[str, Any] = {
                "patient_id": pid,
                "visit": visit,
                "epoch": i,
                "label": label,
            }
            for m in self.modalities:
                sample[m] = epochs[m][i]
            samples.append(sample)
        return samples

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Builds one sample per scored 30 s epoch for a patient.

        Args:
            patient: A patient from :class:`~pyhealth.datasets.NSRRSleepDataset`.

        Returns:
            A list of dicts with ``patient_id``, ``visit``, ``epoch``, one
            float32 array per modality and an integer ``label``. Recordings
            that are unreadable or lack a requested channel are skipped.
        """
        pid = patient.patient_id
        samples: List[Dict[str, Any]] = []
        for event in patient.get_events():
            try:
                samples.extend(self._process_recording(event, pid))
            except (ValueError, OSError, ET.ParseError) as err:
                logger.warning("Skipping recording of patient %s: %s", pid, err)
        return samples

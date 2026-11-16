"""Neonatal seizure detection on the Helsinki neonatal EEG dataset.

Author: Axel Coiffard-Donaty

Three experts (A, B and C) annotated every second of each recording as
seizure or not. This task builds the 18-channel bipolar montage used by the
dataset authors' seizure detector, cuts the recording into overlapping
windows and labels each window from the per-second annotations, combined
across experts by a documented rule.

References:
    - Stevenson, N. J., Tapani, K., Lauronen, L., Vanhatalo, S. A dataset
      of neonatal EEG recordings with seizure annotations. Scientific Data
      6:190039, 2019. https://doi.org/10.1038/sdata.2019.39
    - Montage: ``read_data_montage.m`` in
      https://github.com/ktapani/Neonatal_Seizure_Detection
    - Default filter, sampling rate, windows and window labels: Borovac, A.
      et al. Ensemble learning using individual neonatal data for seizure
      detection. IEEE Journal of Translational Engineering in Health and
      Medicine 10, 2022. https://doi.org/10.1109/JTEHM.2022.3201167, and
      its code, ``src/data_preprocessing.py`` in
      https://github.com/anaborovac/Distributed-NSDA
"""

import logging
import math
from collections.abc import Sequence
from fractions import Fraction
from typing import Any, ClassVar

import mne
import numpy as np
import pandas as pd
from scipy.signal import cheby2, resample_poly, sosfilt

from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

HELSINKI_BIPOLAR_MONTAGE: tuple[tuple[str, str], ...] = (
    ("Fp2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),
    ("Fp1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),
    ("Fp2", "F8"),
    ("F8", "T4"),
    ("T4", "T6"),
    ("T6", "O2"),
    ("Fp1", "F7"),
    ("F7", "T3"),
    ("T3", "T5"),
    ("T5", "O1"),
    ("Fz", "Cz"),
    ("Cz", "Pz"),
)
"""The 18 bipolar derivations of the dataset authors' code, in their order."""

ANNOTATORS: tuple[str, ...] = ("A", "B", "C")

LABEL_MODES: tuple[str, ...] = ("consensus", "majority", "any", *ANNOTATORS)

_EXCLUDED = -1

# Chebyshev type II band-pass design of Borovac et al.'s code:
# scipy.signal.cheby2(6, 80, ...), with the band edges as stopband edges.
_FILTER_ORDER = 6
_STOPBAND_ATTENUATION_DB = 80


def bandpass_sections(stopband_edges: Sequence[float], sfreq: float) -> np.ndarray:
    """Design the band-pass filter of Borovac et al.'s code.

    A sixth-order Chebyshev type II band-pass with 80 dB of stopband
    attenuation, ``scipy.signal.cheby2(6, 80, ...)`` in their code. It is
    returned as second-order sections: the transfer-function form of this
    design is numerically unstable for narrow bands at 256 Hz.

    Args:
        stopband_edges: ``(low, high)`` frequencies in Hz at which the
            attenuation reaches 80 dB.
        sfreq: Sampling rate of the signal in Hz.

    Returns:
        The filter as an array of second-order sections, for
        ``scipy.signal.sosfilt``.

    Examples:
        >>> from pyhealth.tasks.seizure_detection_helsinki import bandpass_sections
        >>> bandpass_sections((0.5, 32.0), 256.0).shape
        (6, 6)
    """
    return cheby2(
        _FILTER_ORDER,
        _STOPBAND_ATTENUATION_DB,
        list(stopband_edges),
        btype="bandpass",
        fs=sfreq,
        output="sos",
    )


def electrode_name(label: str) -> str:
    """Return the electrode of a Helsinki EDF channel label.

    The recordings name their channels ``EEG <electrode>-Ref`` or
    ``EEG <electrode>-REF``.

    Args:
        label: Channel label as stored in the EDF file.

    Returns:
        The label without the ``EEG`` prefix and the reference suffix.

    Examples:
        >>> from pyhealth.tasks.seizure_detection_helsinki import electrode_name
        >>> [electrode_name(c) for c in ("EEG Fp1-Ref", "EEG Cz-REF", "ECG EKG")]
        ['Fp1', 'Cz', 'ECG EKG']
    """
    name = label.strip()
    if name.upper().startswith("EEG "):
        name = name[4:]
    if name.upper().endswith("-REF"):
        name = name[:-4]
    return name.strip()


def read_helsinki_annotations(annotation_file: str, patient_id: str) -> np.ndarray:
    """Read one expert's per-second annotations of one neonate.

    Args:
        annotation_file: Path to an ``annotations_2017_*.csv`` file, with
            one column per neonate and one row per second.
        patient_id: Column to read, the recording number as a string.

    Returns:
        Integer array with one value per second of the recording (1 seizure,
        0 no seizure). The empty rows after the end of the recording are
        dropped.

    Examples:
        >>> import os, tempfile
        >>> from pyhealth.tasks.seizure_detection_helsinki import (
        ...     read_helsinki_annotations,
        ... )
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = os.path.join(tmp, "annotations_2017_B.csv")
        ...     with open(path, "w") as f:
        ...         _ = f.write("1,2\\n0,1\\n1,1\\n,0\\n")
        ...     read_helsinki_annotations(path, "1").tolist()
        [0, 1]
    """
    column = pd.read_csv(annotation_file, usecols=[patient_id])[patient_id]
    return column.dropna().to_numpy(dtype=np.int64)


def combine_annotations(annotations: np.ndarray, label_mode: str) -> np.ndarray:
    """Combine the per-second annotations of the three experts.

    Args:
        annotations: Array of shape ``(3, n_seconds)`` with the annotations
            of experts A, B and C (1 seizure, 0 no seizure).
        label_mode: ``"consensus"`` (1 when the three experts mark a
            seizure, 0 when none does, excluded otherwise), ``"majority"``
            (at least two experts), ``"any"`` (at least one expert), or
            ``"A"``, ``"B"`` or ``"C"`` for a single expert.

    Returns:
        Integer array of shape ``(n_seconds,)``: 1 seizure, 0 no seizure and
        -1 for the seconds that ``"consensus"`` excludes.

    Raises:
        ValueError: If ``label_mode`` is not one of ``LABEL_MODES``.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.tasks.seizure_detection_helsinki import (
        ...     combine_annotations,
        ... )
        >>> votes = np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]])
        >>> combine_annotations(votes, "consensus").tolist()
        [0, -1, -1, 1]
        >>> combine_annotations(votes, "majority").tolist()
        [0, 0, 1, 1]
        >>> combine_annotations(votes, "any").tolist()
        [0, 1, 1, 1]
    """
    if label_mode not in LABEL_MODES:
        raise ValueError(f"label_mode must be one of {LABEL_MODES}, got {label_mode}.")
    if label_mode in ANNOTATORS:
        return annotations[ANNOTATORS.index(label_mode)].astype(np.int64)
    votes = annotations.sum(axis=0)
    if label_mode == "majority":
        return (votes >= 2).astype(np.int64)
    if label_mode == "any":
        return (votes >= 1).astype(np.int64)
    combined = np.full(votes.shape, _EXCLUDED, dtype=np.int64)
    combined[votes == 0] = 0
    combined[votes == len(ANNOTATORS)] = 1
    return combined


class SeizureDetectionHelsinki(BaseTask):
    """Binary seizure detection on the Helsinki neonatal EEG dataset.

    Runs on :class:`~pyhealth.datasets.HelsinkiNeonatalDataset`. For each
    recording this task:

    1. reads the per-second annotations and combines the experts with
       ``label_mode``
    2. opens the EDF file and builds the bipolar ``montage`` from the
       referential channels, in microvolts
    3. band-pass filters (``bandpass``) and resamples (``resample_rate``)
       the whole recording
    4. cuts windows of ``window_seconds`` every ``stride_seconds`` and
       labels each one

    The defaults follow Borovac et al. (2022) and their published code:

    - Experts: with ``"consensus"`` a second is a seizure when the three
      experts mark it, a non-seizure when none does, and is excluded when
      they disagree; any window that contains an excluded second is
      dropped. ``"majority"`` and ``"any"`` keep every second, and
      ``"A"``, ``"B"`` or ``"C"`` use one expert, for example to study the
      agreement between a model and each expert.
    - Window labels: a window is a seizure (1) when all its seconds are
      seizure seconds and a non-seizure (0) when none is. Windows that are
      partly seizure are dropped: the paper leaves the segments that are
      partly seizure and partly non-seizure out of its training sets.
      Setting ``min_seizure_fraction`` keeps every window instead (apart
      from the excluded seconds of ``"consensus"``) and labels it a
      seizure from that fraction of seizure seconds on; this alternative
      is not from the paper, and is meant for scoring a model on a whole
      recording.
    - Filter: a sixth-order Chebyshev type II band-pass with 80 dB of
      stopband attenuation. ``bandpass`` gives its stopband edges, the
      frequencies where that attenuation is reached, so the band that
      passes is narrower: about 1.25 to 13.4 Hz at -3 dB for the default
      ``(0.5, 32)`` on a 256 Hz recording. The paper states cut-off
      frequencies of 0.5 and 16 Hz; its code uses 0.5 and 32 Hz, followed
      here.
    - Rate and windows: polyphase resampling to 32 Hz, 16 second windows
      shifted by 4 seconds, 18 bipolar channels.

    Differences from that code: the filter runs once over the whole
    recording, as cascaded second-order sections, where the code filters
    each window separately; the signal stays in microvolts, without the
    clipping at 200 microvolts and the 16-bit rescaling of the code. The
    filter is causal (one forward pass), as in the code.

    The recordings are 256 Hz; ``resample_rate=None`` and ``bandpass=None``
    keep the signal as stored.

    A recording missing an electrode of the montage, an annotation file that
    ``label_mode`` needs, or its column in that file, is skipped with a
    warning. When the annotations and the signal differ in length, only the
    seconds present in both are used.

    Each sample contains:

    - ``signal``: float32 array, shape
      ``(len(montage), window_seconds * rate)``, ``(18, 512)`` by default
    - ``label``: 1 seizure, 0 non-seizure
    - ``start_second``: start of the window in the recording
    - ``seizure_fraction``: fraction of seizure seconds in the window

    Args:
        label_mode: How the experts are combined: ``"consensus"``,
            ``"majority"``, ``"any"``, ``"A"``, ``"B"`` or ``"C"``.
        window_seconds: Window length in whole seconds.
        stride_seconds: Shift between windows in whole seconds.
        resample_rate: Sampling rate of the output in Hz, or None to keep
            the rate of the recording.
        bandpass: ``(low, high)`` stopband edges of the band-pass filter in
            Hz, or None for no filtering.
        min_seizure_fraction: None (default) to keep only the windows that
            are entirely seizure or entirely non-seizure. A fraction in
            ``(0, 1]`` keeps the partly seizure windows too and labels a
            window a seizure from that fraction of seizure seconds on.
        montage: Pairs of electrodes ``(a, b)``; each channel is ``a - b``.
            Defaults to ``HELSINKI_BIPOLAR_MONTAGE``.

    Raises:
        ValueError: If ``label_mode`` is unknown, ``window_seconds`` or
            ``stride_seconds`` isn't a positive whole number,
            ``resample_rate`` isn't positive, ``bandpass`` isn't an
            increasing pair of positive frequencies (or, when a file is
            read, its upper edge isn't below half of the recording's rate),
            ``min_seizure_fraction`` is outside ``(0, 1]`` or ``montage`` is
            empty.

    Examples:
        >>> from pyhealth.datasets import HelsinkiNeonatalDataset
        >>> from pyhealth.tasks import SeizureDetectionHelsinki
        >>> dataset = HelsinkiNeonatalDataset(root="/path/to/helsinki")
        >>> task = SeizureDetectionHelsinki(label_mode="consensus")
        >>> samples = dataset.set_task(task)
        >>> samples[0]["signal"].shape
        torch.Size([18, 512])
    """

    task_name: str = "SeizureDetectionHelsinki"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"label": "binary"}

    def __init__(
        self,
        label_mode: str = "consensus",
        window_seconds: int = 16,
        stride_seconds: int = 4,
        resample_rate: float | None = 32.0,
        bandpass: tuple[float, float] | None = (0.5, 32.0),
        min_seizure_fraction: float | None = None,
        montage: Sequence[tuple[str, str]] = HELSINKI_BIPOLAR_MONTAGE,
    ) -> None:
        super().__init__()
        if label_mode not in LABEL_MODES:
            raise ValueError(
                f"label_mode must be one of {LABEL_MODES}, got {label_mode}."
            )
        for name, value in (
            ("window_seconds", window_seconds),
            ("stride_seconds", stride_seconds),
        ):
            if value <= 0 or value != int(value):
                raise ValueError(
                    f"{name} must be a positive whole number of seconds, got {value}."
                )
        if resample_rate is not None and resample_rate <= 0:
            raise ValueError(f"resample_rate must be positive, got {resample_rate}.")
        if bandpass is not None:
            low, high = bandpass
            if not 0 < low < high:
                raise ValueError(
                    f"bandpass must be (low, high) with 0 < low < high, got {bandpass}."
                )
        if min_seizure_fraction is not None and not 0 < min_seizure_fraction <= 1:
            raise ValueError(
                f"min_seizure_fraction must be in (0, 1], got {min_seizure_fraction}."
            )
        if len(montage) == 0:
            raise ValueError("montage must hold at least one electrode pair.")
        self.label_mode = label_mode
        self.window_seconds = int(window_seconds)
        self.stride_seconds = int(stride_seconds)
        self.resample_rate = None if resample_rate is None else float(resample_rate)
        self.bandpass = None if bandpass is None else tuple(map(float, bandpass))
        self.min_seizure_fraction = (
            None if min_seizure_fraction is None else float(min_seizure_fraction)
        )
        self.montage = tuple((str(a), str(b)) for a, b in montage)

    def read_recording(self, signal_file: str) -> tuple[np.ndarray, float] | None:
        """Load one recording as the filtered, resampled bipolar montage.

        Args:
            signal_file: Path to ``eeg<N>.edf``.

        Returns:
            ``(data, sfreq)`` with ``data`` in microvolts, shaped
            ``(len(montage), n_samples)``, and ``sfreq`` the output rate, or
            None if an electrode of the montage is missing.
        """
        raw = mne.io.read_raw_edf(signal_file, preload=False, verbose="error")
        by_electrode: dict[str, str] = {}
        for name in raw.ch_names:
            by_electrode.setdefault(electrode_name(name).upper(), name)
        electrodes = sorted({e.upper() for pair in self.montage for e in pair})
        missing = [e for e in electrodes if e not in by_electrode]
        if missing:
            logger.warning(
                "Skipping %s, it has no %s electrode(s). Channels are %s.",
                signal_file,
                missing,
                raw.ch_names,
            )
            raw.close()
            return None
        referential = raw.get_data(
            picks=[by_electrode[e] for e in electrodes], units="uV"
        )
        sfreq = float(raw.info["sfreq"])
        raw.close()

        row = {e: i for i, e in enumerate(electrodes)}
        data = np.stack(
            [
                referential[row[a.upper()]] - referential[row[b.upper()]]
                for a, b in self.montage
            ]
        )
        if self.bandpass is not None:
            if self.bandpass[1] >= sfreq / 2:
                raise ValueError(
                    f"The upper band edge {self.bandpass[1]} Hz is not below half "
                    f"of the {sfreq} Hz rate of {signal_file}."
                )
            data = sosfilt(bandpass_sections(self.bandpass, sfreq), data, axis=-1)
        if self.resample_rate is not None and self.resample_rate != sfreq:
            ratio = Fraction(self.resample_rate / sfreq).limit_denominator(1000)
            data = resample_poly(data, ratio.numerator, ratio.denominator, axis=-1)
            sfreq = sfreq * ratio.numerator / ratio.denominator
        return data, sfreq

    def _second_labels(self, patient_id: str, event: Any) -> np.ndarray | None:
        """Read and combine the per-second annotations of one recording.

        Args:
            patient_id: Recording number, the column name in the CSV files.
            event: A ``recordings`` event with the annotation file columns.

        Returns:
            Per-second labels (1, 0, or -1 when excluded), or None when an
            annotation file or column that ``label_mode`` needs is missing.
        """
        needed = (self.label_mode,) if self.label_mode in ANNOTATORS else ANNOTATORS
        per_expert: dict[str, np.ndarray] = {}
        for annotator in needed:
            path = getattr(event, f"annotation_file_{annotator.lower()}", "")
            path = "" if path is None else str(path).strip()
            if path in {"", "nan", "None"}:
                logger.warning(
                    "Skipping %s, the annotations of expert %s are missing.",
                    event.signal_file,
                    annotator,
                )
                return None
            try:
                per_expert[annotator] = read_helsinki_annotations(path, patient_id)
            except ValueError:
                logger.warning(
                    "Skipping %s, %s has no column %s.",
                    event.signal_file,
                    path,
                    patient_id,
                )
                return None
        n_seconds = min(len(values) for values in per_expert.values())
        annotations = np.zeros((len(ANNOTATORS), n_seconds), dtype=np.int64)
        for annotator, values in per_expert.items():
            annotations[ANNOTATORS.index(annotator)] = values[:n_seconds]
        return combine_annotations(annotations, self.label_mode)

    def _samples_from_event(self, patient_id: str, event: Any) -> list[dict[str, Any]]:
        """Turn one annotated recording into window samples.

        Args:
            patient_id: Recording number, for example ``"1"``.
            event: A ``recordings`` event with ``signal_file`` and the
                annotation file columns.

        Returns:
            One sample dict per kept window.
        """
        labels = self._second_labels(patient_id, event)
        if labels is None:
            return []
        loaded = self.read_recording(event.signal_file)
        if loaded is None:
            return []
        signal, sfreq = loaded

        signal_seconds = math.floor(signal.shape[1] / sfreq)
        if abs(signal_seconds - len(labels)) > 1:
            logger.warning(
                "%s has %d seconds of signal but %d annotated seconds, using "
                "the first %d.",
                event.signal_file,
                signal_seconds,
                len(labels),
                min(signal_seconds, len(labels)),
            )
        n_seconds = min(signal_seconds, len(labels))
        window_len = round(self.window_seconds * sfreq)
        samples: list[dict[str, Any]] = []
        last_start = n_seconds - self.window_seconds
        for start_second in range(0, last_start + 1, self.stride_seconds):
            window = labels[start_second : start_second + self.window_seconds]
            if (window == _EXCLUDED).any():
                continue
            fraction = float(window.mean())
            if self.min_seizure_fraction is None:
                if 0 < fraction < 1:
                    continue
                label = int(fraction == 1)
            else:
                label = int(fraction >= self.min_seizure_fraction)
            start = round(start_second * sfreq)
            samples.append(
                {
                    "patient_id": patient_id,
                    "start_second": start_second,
                    "seizure_fraction": fraction,
                    "signal": signal[:, start : start + window_len].astype(np.float32),
                    "label": label,
                }
            )
        return samples

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build window samples for one neonate.

        Args:
            patient: A PyHealth patient whose ``recordings`` events are the
                rows of ``helsinki_neonatal-pyhealth.csv``.

        Returns:
            Window samples for every recording of that neonate.
        """
        samples: list[dict[str, Any]] = []
        for event in patient.get_events("recordings"):
            samples.extend(self._samples_from_event(patient.patient_id, event))
        return samples

"""Single-heartbeat myocardial infarction detection on PTB-XL.

Description: Cuts each PTB-XL record into single heartbeats centred on their
    R-peaks and labels them MI or healthy, the setup TarDiff (Deng et al.,
    2025) evaluates on the PTB Diagnostic ECG Database. The heartbeat
    segmentation follows Medformer's PTB preprocessing (Wang et al., NeurIPS
    2024; https://github.com/DL4mHealth/Medformer, MIT licence).

References:
    - Deng, B. et al. TarDiff: Target-Oriented Diffusion Guidance for
      Synthetic Electronic Health Record Time Series Generation.
      arXiv:2504.17613, 2025.
    - Wang, Y. et al. Medformer: A Multi-Granularity Patching Transformer
      for Medical Time-Series Classification. NeurIPS 2024.
    - Wagner, P. et al. PTB-XL, a large publicly available
      electrocardiography dataset. Scientific Data 7, 154 (2020).
"""

from math import gcd
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from scipy.signal import resample_poly

from pyhealth.data import Patient

from .base_task import BaseTask


def detect_r_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
    """Finds the R-peaks of a multi-lead ECG.

    Runs the XQRS detector of ``wfdb.processing`` on every lead. Leads that
    agree on the number of beats (the most common count) are kept, and each
    beat's position is the median over them. Medformer takes the median over
    all leads; dropping the leads that miss or add beats keeps one noisy lead
    from shifting every later beat.

    Args:
        signal: ``(leads, samples)`` ECG.
        fs: Sampling rate in Hz.

    Returns:
        Sorted R-peak sample indices (empty when no lead has a beat).

    Raises:
        ImportError: If the optional ``wfdb`` dependency is not installed.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.tasks.ptbxl_heartbeat import detect_r_peaks
        >>> t = np.arange(2500) / 250
        >>> beats = np.exp(-(((t % 0.8) - 0.4) ** 2) / 2e-4)  # 75 bpm spikes
        >>> peaks = detect_r_peaks(np.stack([beats, 0.5 * beats]), fs=250)
        >>> int(np.median(np.diff(peaks)))
        200
    """
    try:
        from wfdb import processing
    except ImportError as exc:
        raise ImportError(
            "R-peak detection needs the optional 'wfdb' dependency. "
            "Install it with: pip install 'pyhealth[ptbxl]'"
        ) from exc

    per_lead = [
        np.sort(processing.xqrs_detect(sig=lead, fs=fs, verbose=False))
        for lead in np.asarray(signal, dtype=np.float64)
    ]
    counts = [len(p) for p in per_lead if len(p)]
    if not counts:
        return np.zeros(0, dtype=np.int64)
    values, freq = np.unique(counts, return_counts=True)
    n = values[np.argmax(freq)]
    agreeing = np.stack([p for p in per_lead if len(p) == n])
    return np.median(agreeing, axis=0).astype(np.int64)


def segment_heartbeats(
    signal: np.ndarray,
    peaks: np.ndarray,
    length: int,
    skip_partial: bool = False,
) -> np.ndarray:
    """Cuts one window per R-peak, as Medformer does.

    Each window spans the record's median R-R interval (at most ``length``),
    centred on the peak, and is zero-padded on both sides to ``length``.
    Beats at the edges of the record are padded too, unless
    ``skip_partial`` drops them.

    Args:
        signal: ``(leads, samples)`` ECG.
        peaks: R-peak indices from :func:`detect_r_peaks`.
        length: Window length in samples.
        skip_partial: Drop beats whose window is cut by the start or end of
            the record. Default False.

    Returns:
        ``(beats, leads, length)`` float32 array (no beats with fewer than
        two peaks, since the R-R interval is then unknown).

    Examples:
        >>> import numpy as np
        >>> from pyhealth.tasks.ptbxl_heartbeat import segment_heartbeats
        >>> signal = np.ones((2, 1000))
        >>> beats = segment_heartbeats(signal, np.array([100, 300, 500]), length=288)
        >>> beats.shape, int(beats[1, 0].sum())  # R-R = 200 samples kept
        ((3, 2, 288), 200)
    """
    leads, total = signal.shape
    if len(peaks) < 2:
        return np.zeros((0, leads, length), dtype=np.float32)
    half = min(int(np.median(np.diff(peaks))), length) // 2
    if skip_partial:
        peaks = peaks[(peaks >= half) & (peaks + half <= total)]
    beats = np.zeros((len(peaks), leads, length), dtype=np.float32)
    for i, peak in enumerate(peaks):
        window = signal[:, max(0, peak - half) : min(peak + half, total)]
        left = (length - window.shape[1]) // 2
        beats[i, :, left : left + window.shape[1]] = window
    return beats


class PTBXLHeartbeatMI(BaseTask):
    """Myocardial infarction vs healthy, one sample per heartbeat.

    Works with :class:`~pyhealth.datasets.PTBXLDataset` (``sampling_rate=500``
    recommended). A record is MI (label 1) when its diagnostic superclasses
    include ``MI`` and healthy (label 0) when ``NORM`` is its only
    superclass; other records are skipped. Each record is resampled to
    ``target_fs``, z-scored per lead, and cut into heartbeats with
    :func:`detect_r_peaks` and :func:`segment_heartbeats`.

    This follows Medformer's PTB preprocessing, which TarDiff uses, with
    these differences: PTB-XL has 12 leads (PTB has 15) and 10 s records;
    resampling is polyphase (anti-aliased) instead of linear interpolation;
    R-peaks come from ``wfdb``'s XQRS instead of NeuroKit2; records whose
    median R-R interval exceeds ``length`` are kept with the window cropped
    to ``length``, where Medformer drops them. Beats cut by the record's
    edges are padded, as in Medformer, unless ``skip_partial_beats`` drops
    them (with 10 s records and ``max_beats_per_record``, they are a large
    share of the samples). The default ``length`` of 288 is TarDiff's (300
    cropped to a multiple of 8 for its U-Net).

    Each sample has ``patient_id``, ``record_id``, ``beat`` (index among the
    record's kept beats), ``signal`` ``(12, length)``, ``label`` and
    ``strat_fold``, so :func:`~pyhealth.datasets.split_by_strat_fold` gives
    patient-disjoint splits.

    Args:
        scp_statements_path: PTB-XL's ``scp_statements.csv``
            (``PTBXLDataset.scp_statements_path``).
        length: Heartbeat window in samples at ``target_fs``. Default 288.
        target_fs: Sampling rate after resampling. Default 250 (Medformer).
        max_beats_per_record: Keep at most this many beats per record (the
            first ones kept), to bound the dataset size. Default None keeps
            all.
        skip_partial_beats: Drop beats cut by the start or end of the
            record instead of zero-padding them. Default False.

    Examples:
        >>> from pyhealth.datasets import PTBXLDataset
        >>> from pyhealth.tasks import PTBXLHeartbeatMI
        >>> dataset = PTBXLDataset(root="/data/ptb-xl/1.0.3", sampling_rate=500)  # doctest: +SKIP
        >>> task = PTBXLHeartbeatMI(dataset.scp_statements_path)  # doctest: +SKIP
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
        >>> samples[0]["signal"].shape  # doctest: +SKIP
        torch.Size([12, 288])
    """

    task_name: str = "PTBXLHeartbeatMI"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"label": "binary"}

    def __init__(
        self,
        scp_statements_path: str | Path,
        length: int = 288,
        target_fs: int = 250,
        max_beats_per_record: int | None = None,
        skip_partial_beats: bool = False,
    ) -> None:
        if length < 2 or target_fs < 1:
            raise ValueError("length must be at least 2 and target_fs positive.")
        if max_beats_per_record is not None and max_beats_per_record < 1:
            raise ValueError("max_beats_per_record must be positive or None.")
        self.scp_statements_path = Path(scp_statements_path)
        self.length = int(length)
        self.target_fs = int(target_fs)
        self.max_beats_per_record = max_beats_per_record
        self.skip_partial_beats = bool(skip_partial_beats)
        super().__init__()

    @staticmethod
    def label_of(superclasses: list[str]) -> int | None:
        """1 for MI, 0 for NORM only, None for any other record.

        Examples:
            >>> from pyhealth.tasks import PTBXLHeartbeatMI
            >>> [PTBXLHeartbeatMI.label_of(s) for s in (["MI", "STTC"], ["NORM"], ["CD"])]
            [1, 0, None]
        """
        if "MI" in superclasses:
            return 1
        if superclasses == ["NORM"]:
            return 0
        return None

    def heartbeats(self, signal: np.ndarray, fs: int) -> np.ndarray:
        """Resamples, z-scores and segments one ``(leads, samples)`` record."""
        g = gcd(int(fs), self.target_fs)
        x = resample_poly(signal, self.target_fs // g, int(fs) // g, axis=1)
        std = x.std(axis=1, keepdims=True)
        x = (x - x.mean(axis=1, keepdims=True)) / np.where(std > 0, std, 1.0)
        beats = segment_heartbeats(
            x,
            detect_r_peaks(x, self.target_fs),
            self.length,
            skip_partial=self.skip_partial_beats,
        )
        return beats[: self.max_beats_per_record]

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        """Builds one sample per heartbeat of the patient's MI or healthy records."""
        from pyhealth.datasets.ptbxl import load_ptbxl_record
        from pyhealth.tasks.ptbxl import (
            aggregate_diagnostic_superclasses,
            load_diagnostic_class_map,
        )

        class_map = load_diagnostic_class_map(self.scp_statements_path)
        samples = []
        for event in patient.get_events(event_type="records"):
            label = self.label_of(
                aggregate_diagnostic_superclasses(event.scp_codes, class_map)
            )
            if label is None:
                continue
            beats = self.heartbeats(
                load_ptbxl_record(event.signal_file), int(float(event.sampling_rate))
            )
            for i, beat in enumerate(beats):
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "record_id": str(event.record_id),
                        "beat": i,
                        "signal": beat,
                        "label": label,
                        "strat_fold": int(float(event.strat_fold)),
                    }
                )
        return samples

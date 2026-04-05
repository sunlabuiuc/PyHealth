"""PyHealth task: SleepEDF → TF-C / SleepEEG-style fixed-length windows.

Produces samples aligned with the tensor layout used in
`mims-harvard/TFC-pretraining`: a dict with ``samples`` of shape
``[N, n_channels, L]`` and ``labels`` of shape ``[N]``. See the upstream
`dataloader.py` (expects channel in dimension 1, then crops to
``TSlength_aligned``, default 178).

References:
- https://github.com/mims-harvard/TFC-pretraining
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import mne
import numpy as np
import torch

from pyhealth.tasks import BaseTask


def _map_to_MVCL_five_class(pyhealth_stage: int) -> int:
    """Map PyHealth 6-class staging to 5-class AASM-style (N3+N4 → deep)."""
    # PyHealth: W=0, N1=1, N2=2, N3=3, N4=4, R=5
    return (0, 1, 2, 3, 3, 4)[int(pyhealth_stage)]


class MVCLTrainingSleepEEG(BaseTask):
    """Short EEG windows from SleepEDFDataset for time-series pretraining (TF-C style).

    Reads each recording like ``SleepStagingSleepEDF`` (30 s scored epochs), picks a
    single EEG lead, then splits each epoch into non-overlapping windows of
    ``window_size`` samples (default 200 @ 100 Hz, consistent with the SleepEEG
    description in the TF-C paper). Each window inherits the epoch's sleep-stage
    label, remapped to 5 classes (N3 and N4 both map to deep sleep).

    Output samples are dicts with ``signal`` shaped ``(1, L)`` where ``L`` is
    ``window_size`` or ``crop_length`` 
    """

    task_name: str = "MVCLTrainingSleepEEG"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __init__(
        self,
        chunk_duration: float = 30.0,
        window_size: int = 200,
        crop_length: Optional[int] = 178,
        eeg_channel: Optional[str] = "EEG Fpz-Cz",
    ) -> None:
        """
        Args:
            chunk_duration: Hypnogram epoch length in seconds (PyHealth default 30).
            window_size: Non-overlapping window length in samples (TF-C SleepEEG: 200).
            crop_length: If set, each window is truncated to this many samples from
                the start (TF-C code often uses 178 for cross-dataset alignment).
            eeg_channel: MNE channel name to keep (single lead). If None or name
                missing, falls back to the first channel whose name contains ``EEG``,
                then index 0.
        """
        self.chunk_duration = float(chunk_duration)
        self.window_size = int(window_size)
        self.crop_length = int(crop_length) if crop_length is not None else None
        self.eeg_channel = eeg_channel
        super().__init__()

    def _pick_eeg_index(self, ch_names: List[str]) -> int:
        if self.eeg_channel and self.eeg_channel in ch_names:
            return ch_names.index(self.eeg_channel)
        for i, n in enumerate(ch_names):
            if "eeg" in n.lower():
                return i
        return 0

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        pid = patient.patient_id
        events = patient.get_events()
        samples: List[Dict[str, Any]] = []

        event_id = {
            "Sleep stage W": 0,
            "Sleep stage 1": 1,
            "Sleep stage 2": 2,
            "Sleep stage 3": 3,
            "Sleep stage 4": 4,
            "Sleep stage R": 5,
        }

        win = self.window_size
        crop = self.crop_length

        global_epoch = 0
        for event in events:
            if not event.signal_file or not event.label_file:
                continue
            data = mne.io.read_raw_edf(
                event.signal_file,
                stim_channel="Event marker",
                infer_types=True,
                preload=True,
                verbose="error",
            )
            ann = mne.read_annotations(event.label_file)
            data.set_annotations(ann, emit_warning=False)

            ann_events, event_id_used = mne.events_from_annotations(
                data, event_id=event_id, chunk_duration=self.chunk_duration
            )
            if ann_events.size == 0:
                continue

            # Use MNE's filtered event_id (only stages present in this night). Passing
            # the full 6-stage dict causes ValueError for missing N3/N4/etc. on some nights.
            epochs_train = mne.Epochs(
                data,
                ann_events,
                event_id_used,
                tmin=0.0,
                tmax=self.chunk_duration - 1.0 / data.info["sfreq"],
                baseline=None,
                preload=True,
                on_missing="ignore",
                verbose="error",
            )

            ch_i = self._pick_eeg_index(list(epochs_train.ch_names))
            signals = epochs_train.get_data()[:, ch_i, :]
            labels = epochs_train.events[:, 2]

            n_epochs, n_times = signals.shape
            n_full = (n_times // win) * win

            for epi in range(n_epochs):
                lab = _map_to_MVCL_five_class(int(labels[epi]))
                row = signals[epi, :n_full]
                for w in range(n_full // win):
                    seg = row[w * win : (w + 1) * win].astype(np.float32, copy=False)
                    if crop is not None:
                        seg = seg[:crop]
                    vec = seg[np.newaxis, :]
                    samples.append(
                        {
                            "patient_id": pid,
                            "night": event.night,
                            "patient_age": event.age,
                            "patient_sex": event.sex,
                            "epoch_index": global_epoch,
                            "window_in_epoch": w,
                            "signal": vec,
                            "label": lab,
                        }
                    )
                global_epoch += 1

        return samples


def stack_samples_to_mvcl_dict(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Stack PyHealth task outputs into a TF-C-style tensor dict."""
    if not samples:
        raise ValueError("empty sample list")
    xs = np.stack([np.asarray(s["signal"]) for s in samples], axis=0)
    ys = np.array([int(s["label"]) for s in samples], dtype=np.int64)
    x = torch.from_numpy(np.ascontiguousarray(xs)).float()
    y = torch.from_numpy(ys).long()
    if x.ndim == 2:
        x = x.unsqueeze(1)
    return {"samples": x, "labels": y}


def save_mvcl_pt(
    tensor_dict: Dict[str, torch.Tensor],
    path: str,
) -> None:
    """Save ``{"samples", "labels"}`` in PyTorch ``torch.save`` format (``.pt``)."""
    torch.save(tensor_dict, path)

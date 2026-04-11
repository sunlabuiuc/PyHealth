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

from typing import Any, Dict, List, Optional, Tuple
import mne
import numpy as np
import torch
import torch.fft as fft

from pyhealth.tasks import BaseTask



def _map_to_MVCL_five_class(pyhealth_stage: int) -> int:
    """Map PyHealth 6-class staging to 5-class AASM-style (N3+N4 → deep)."""
    return (0, 1, 2, 3, 3, 4)[int(pyhealth_stage)]


class MVCLTrainingSleepEEG(BaseTask):
    """SleepEDF windows with Multi-View contrastive tensor views.

    Applies MV preprocessing per event file (one PSG/Hypnogram pair at a time),
    then appends samples immediately, so each returned sample includes ``xt``,
    ``dx``, and ``xf`` without a patient-level global buffer.

    Tensors are stored as ``numpy.float32`` arrays with shape ``(L, C_view)`` where
    ``C_view`` is 1 by default; with ``time_as_feature=True``, a leading time channel
    in ``[0,1]`` is concatenated so ``C_view`` is 2.
    """

    task_name: str = "MVCLTrainingSleepEEG"
    input_schema = {"signal": "tensor"}
    output_schema = {"label": "multiclass"}

    def __init__(
        self,
        chunk_duration: float = 30.0,
        window_size: int = 200,
        crop_length: Optional[int] = 178,
        eeg_channel: Optional[str] = "EEG Fpz-Cz",
        time_as_feature: bool = False,
        dx_backend: str = "cde",
    ) -> None:
        self.chunk_duration = float(chunk_duration)
        self.window_size = int(window_size)
        self.crop_length = int(crop_length) if crop_length is not None else None
        self.eeg_channel = eeg_channel
        # ``False`` matches ``preprocess_data`` defaults in MV run_pretrain / run_finetune.
        self.time_as_feature = bool(time_as_feature)

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

            event_buffers: List[Dict[str, Any]] = []
            for epi in range(n_epochs):
                lab = _map_to_MVCL_five_class(int(labels[epi]))
                row = signals[epi, :n_full]
                for w in range(n_full // win):
                    seg = row[w * win : (w + 1) * win].astype(np.float32, copy=False)
                    if crop is not None:
                        seg = seg[:crop]
                    event_buffers.append(
                        {
                            "seg_1d": seg.copy(),
                            "label": lab,
                            "night": event.night,
                            "patient_age": event.age,
                            "patient_sex": event.sex,
                            "epoch_index": global_epoch,
                            "window_in_epoch": w,
                        }
                    )
                global_epoch += 1

            if not event_buffers:
                continue

            X = torch.stack(
                [torch.from_numpy(b["seg_1d"]).float() for b in event_buffers], dim=0
            ).unsqueeze(-1)
            xt, dx, xf = preprocess_mvcl_views(
                X,
                time_as_feature=self.time_as_feature
            )

            for i, b in enumerate(event_buffers):
                seg = b["seg_1d"]
                vec = seg[np.newaxis, :]
                samples.append(
                    {
                        "patient_id": pid,
                        "night": b["night"],
                        "patient_age": b["patient_age"],
                        "patient_sex": b["patient_sex"],
                        "epoch_index": b["epoch_index"],
                        "window_in_epoch": b["window_in_epoch"],
                        "signal": vec,
                        "xt": xt[i].detach().cpu().numpy().astype(np.float32),
                        "xd": dx[i].detach().cpu().numpy().astype(np.float32),
                        "xf": xf[i].detach().cpu().numpy().astype(np.float32),
                        "label": b["label"],
                    }
                )

        return samples


def normalize_mvcl(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    epsilon: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = X_train.mean(dim=(0, 1), keepdim=True)
    std = X_train.std(dim=(0, 1), keepdim=True).clamp(min=epsilon)
    return (
        (X_train - mean) / std,
        (X_test - mean) / std,
        mean,
        std,
    )


def add_time_feature(X: torch.Tensor) -> torch.Tensor:
    """X: [num_samples, sequence_length, num_features] -> concat time in last dim."""
    num_samples, seq_length, _ = X.shape
    time_index = torch.linspace(0, 1, steps=seq_length, dtype=X.dtype, device=X.device)
    time_feature = time_index.view(1, seq_length, 1).expand(num_samples, seq_length, 1)
    return torch.cat([time_feature, X], dim=-1)



def get_dx_gradient(X: torch.Tensor) -> torch.Tensor:
    """Time derivative via ``torch.gradient`` along **dim=1** for **X [N, L, D]**.

    This is **not** equivalent to :func:`get_dx` (torchcde spline); see module docstring.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected [N, L, D], got {tuple(X.shape)}")
    return torch.gradient(X, dim=1)[0]



def get_xf(X: torch.Tensor) -> torch.Tensor:
    return torch.abs(fft.fft(X, dim=1))


def preprocess_mvcl_views(
    X: torch.Tensor,
    time_as_feature: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply the same logical pipeline as ``preprocess_data`` when train and test
    are the same tensor (per-domain batch, e.g. all windows of one patient).

    X: float tensor **[N, L, D]** (e.g. ``D=1`` for single-channel EEG; **L** is time length).

    dx_backend: ``"cde"`` (default, torchcde, matches MV) or ``"gradient"`` (no torchcde).

    Returns xt, dx, xf each [N, L, D] or [N, L, D+1] if ``time_as_feature``.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X shape [N, L, D], got {tuple(X.shape)}")
    xt_tr, xt_te, _, _ = normalize_mvcl(X, X)
    xt = xt_tr


    dx_raw = get_dx_gradient(xt)

    dx_tr, dx_te, _, _ = normalize_mvcl(dx_raw, dx_raw)
    dx = dx_tr

    xf_raw = get_xf(xt)
    xf_tr, xf_te, _, _ = normalize_mvcl(xf_raw, xf_raw)
    xf = xf_tr

    if time_as_feature:
        xt = add_time_feature(xt)
        dx = add_time_feature(dx)
        xf = add_time_feature(xf)

    return xt, dx, xf

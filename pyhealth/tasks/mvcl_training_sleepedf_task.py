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

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import mne
import numpy as np
import torch
import torch.fft as fft

from pyhealth.datasets.sample_dataset import create_sample_dataset
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
                preload=False,
                verbose="error",
            )
            ann = mne.read_annotations(event.label_file)
            data.set_annotations(ann, emit_warning=False)

            ann_events, event_id_used = mne.events_from_annotations(
                data, event_id=event_id, chunk_duration=self.chunk_duration
            )
            if ann_events.size == 0:
                continue

            # Pick only the required EEG channel to save memory (7x reduction)
            ch_i = self._pick_eeg_index(list(data.ch_names))
            data.pick([data.ch_names[ch_i]])

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

            # Since we picked exactly 1 channel, it is now at index 0
            signals = epochs_train.get_data()[:, 0, :]
            labels = epochs_train.events[:, 2]

            n_epochs, n_times = signals.shape
            n_full = (n_times // win) * win

            event_buffers: List[Dict[str, Any]] = []
            for epi in range(n_epochs):
                lab = _map_to_MVCL_five_class(int(labels[epi]))
                
                # Take only the first window of the epoch to match TFC-pretraining's sample count
                seg = signals[epi, :win].astype(np.float32, copy=False)
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
                        "window_in_epoch": 0,
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

    This is **not** equivalent to :func:`get_dx` (torchcde spline);.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected [N, L, D], got {tuple(X.shape)}")
    return torch.gradient(X, dim=1)[0]


def get_dx_torchcde_equivalent(X: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch equivalent of torchcde Hermite cubic spline derivative 
    evaluated at the knot points with backward differences.
    """
    N, L, D = X.shape
    dx = torch.zeros_like(X)
    
    if L < 2:
        return dx
        
    dt = 1.0 / (L - 1)
    
    # derivs[i] = X[i+1] - X[i]
    derivs = X[:, 1:, :] - X[:, :-1, :]
    
    # derivs_prev[i] = derivs[i-1] for i > 0, and derivs[0] for i = 0
    derivs_prev = torch.cat([derivs[:, :1, :], derivs[:, :-1, :]], dim=1)
    
    # For i = 0
    dx[:, 0, :] = derivs[:, 0, :]
    
    # For i > 0
    factor = (4 - 3 * dt) * dt
    b = derivs_prev
    D = derivs - b
    
    dx[:, 1:, :] = b + D * factor
    
    return dx



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


    # dx_raw = get_dx_gradient(xt) # this is approxi to paper's torchcde spline
    dx_raw = get_dx_torchcde_equivalent(xt)

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


def _to_tensor(data: Any, key_name: str) -> torch.Tensor:
    """Convert input payloads to a detached CPU tensor for validation."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    try:
        return torch.as_tensor(data).detach().cpu()
    except Exception as err:  # pragma: no cover - defensive branch
        raise TypeError(f"Could not convert `{key_name}` to tensor: {err}") from err


def _normalize_signal_array(samples_obj: Any) -> np.ndarray:
    """Normalize `.pt` sample tensor to [N, L] float32."""
    signal_tensor = _to_tensor(samples_obj, "samples").float().contiguous()
    if signal_tensor.ndim == 2:
        return signal_tensor.numpy().astype(np.float32, copy=False)
    if signal_tensor.ndim != 3:
        raise ValueError(
            "Expected `samples` shape [N, L], [N, 1, L], or [N, L, 1], "
            f"got {tuple(signal_tensor.shape)}"
        )

    if signal_tensor.shape[1] == 1:
        signal_tensor = signal_tensor[:, 0, :]
    elif signal_tensor.shape[2] == 1:
        signal_tensor = signal_tensor[:, :, 0]
    else:
        raise ValueError(
            "Expected a single-channel tensor for `samples` when rank is 3; "
            f"got {tuple(signal_tensor.shape)}"
        )
    return signal_tensor.numpy().astype(np.float32, copy=False)


def _normalize_label_array(labels_obj: Any) -> np.ndarray:
    """Normalize labels to [N] int64."""
    label_tensor = _to_tensor(labels_obj, "labels").long().contiguous()
    if label_tensor.ndim == 1:
        return label_tensor.numpy().astype(np.int64, copy=False)
    if label_tensor.ndim == 2 and 1 in label_tensor.shape:
        return label_tensor.reshape(-1).numpy().astype(np.int64, copy=False)
    raise ValueError(
        "Expected `labels` shape [N] or [N, 1], "
        f"got {tuple(label_tensor.shape)}"
    )


def pt_dict_to_pyhealth_samples(
    tensor_dict: Mapping[str, Any],
    *,
    patient_id_prefix: str = "epilepsy_patient",
    record_id_prefix: str = "epilepsy_record",
    time_as_feature: bool = False,
) -> List[Dict[str, Any]]:
    """Convert `{samples, labels}` tensors into PyHealth raw sample dicts."""
    if "samples" not in tensor_dict or "labels" not in tensor_dict:
        keys = sorted(tensor_dict.keys())
        raise KeyError(
            "Expected keys `samples` and `labels` in tensor dict, "
            f"but found keys: {keys}"
        )

    signal_array = _normalize_signal_array(tensor_dict["samples"])
    label_array = _normalize_label_array(tensor_dict["labels"])

    if signal_array.shape[0] != label_array.shape[0]:
        raise ValueError(
            "`samples` and `labels` length mismatch: "
            f"{signal_array.shape[0]} vs {label_array.shape[0]}"
        )

    # preprocess_mvcl_views expects [N, L, D], use D=1 for single-channel EEG.
    signal_tensor = torch.from_numpy(np.ascontiguousarray(signal_array)).float().unsqueeze(-1)
    xt, dx, xf = preprocess_mvcl_views(signal_tensor, time_as_feature=time_as_feature)

    samples: List[Dict[str, Any]] = []
    for i in range(signal_array.shape[0]):
        samples.append(
            {
                "patient_id": f"{patient_id_prefix}_{i}",
                "record_id": f"{record_id_prefix}_{i}",
                "signal": signal_array[i][np.newaxis, :],
                "xt": xt[i].detach().cpu().numpy().astype(np.float32),
                "xd": dx[i].detach().cpu().numpy().astype(np.float32),
                "xf": xf[i].detach().cpu().numpy().astype(np.float32),
                "label": int(label_array[i]),
            }
        )
    return samples


def pt_file_to_sample_dataset(
    pt_path: Union[str, Path],
    *,
    dataset_name: str = "epilepsy_pt",
    task_name: str = "MVCLTrainingEpilepsyPT",
    in_memory: bool = True,
    patient_id_prefix: str = "epilepsy_patient",
    record_id_prefix: str = "epilepsy_record",
    time_as_feature: bool = False,
):
    """Load one `.pt` file and return a PyHealth SampleDataset."""
    try:
        tensor_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        tensor_dict = torch.load(pt_path, map_location="cpu")

    if not isinstance(tensor_dict, Mapping):
        raise TypeError(
            f"Expected `{pt_path}` to load as a mapping, got {type(tensor_dict)}"
        )

    samples = pt_dict_to_pyhealth_samples(
        tensor_dict,
        patient_id_prefix=patient_id_prefix,
        record_id_prefix=record_id_prefix,
        time_as_feature=time_as_feature,
    )
    return create_sample_dataset(
        samples=samples,
        input_schema={
            "signal": "tensor",
            "xt": "tensor",
            "xd": "tensor",
            "xf": "tensor",
        },
        output_schema={"label": "multiclass"},
        dataset_name=dataset_name,
        task_name=task_name,
        in_memory=in_memory,
    )

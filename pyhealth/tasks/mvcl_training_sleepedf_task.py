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

import os
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
    """
    SleepEDF windows with Multi-View contrastive tensor views.

    This dataset contains 153 whole-night sleep electroencephalography
    (EEG) recordings collected from 82 healthy subjects. Each recording is sampled at 100 Hz using a 1-lead
    EEG signal. The EEG signals are segmented into non-overlapping windows of size 200, each forming
    one sample. Each sample is labeled with one of five sleep stages: Wake (W), Non-rapid Eye Movement
    (N1, N2, N3), and Rapid Eye Movement (REM). This segmentation results in 371,055 samples.

    Applies MV preprocessing per event file (one PSG/Hypnogram pair at a time),
    then appends samples immediately, so each returned sample includes ``xt``,
    ``xd``, and ``xf`` without a patient-level global buffer.

    Tensors are stored as ``numpy.float32`` arrays with shape ``(L, C_view)`` where
    ``C_view`` is 1 by default; with ``time_as_feature=True``, a leading time channel
    in ``[0,1]`` is concatenated so ``C_view`` is 2.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for the task input.
        output_schema (Dict[str, str]): The schema for the task output.

    Examples:
        >>> from pyhealth.datasets import SleepEDFDataset
        >>> from pyhealth.tasks import MVCLTrainingSleepEEG
        >>> import os
        >>> os.chdir("/path/to/sleep-edf")
        >>> dataset = SleepEDFDataset(
        ...     root="/path/to/sleep-edf",
        ... )
        >>> task = MVCLTrainingSleepEEG()
        >>> samples = dataset.set_task(task)
        >>> print(samples[0])
    """

    task_name: str = "MVCLTrainingSleepEEG"
    input_schema = {"xt": "tensor", "xd": "tensor", "xf": "tensor"}
    output_schema = {"label": "multiclass"}

    def __init__(
        self,
        chunk_duration: float = 30.0,
        window_size: int = 200,
        crop_length: Optional[int] = 178,
        eeg_channel: Optional[str] = "EEG Fpz-Cz",
        time_as_feature: bool = False,
        dx_backend: str = "cde",
        root_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initializes the task object.

        Args:
            chunk_duration: How long each chunk of EEG signal is (in seconds). Defaults to 30.0.
            window_size: Number of samples per window. Defaults to 200.
            crop_length: Optional length to crop the windows to. Defaults to 178.
            eeg_channel: Which EEG channel to pick. Defaults to "EEG Fpz-Cz".
            time_as_feature: Whether to add a time feature channel. Defaults to False.
            dx_backend: Backend to use for computing the derivative view. Defaults to "cde".
            root_path: Optional path to the root directory of the dataset. Defaults to None.
        """
        self.chunk_duration = float(chunk_duration)
        self.window_size = int(window_size)
        self.crop_length = int(crop_length) if crop_length is not None else None
        self.eeg_channel = eeg_channel
        # ``False`` matches ``preprocess_data`` defaults in MV run_pretrain / run_finetune.
        self.time_as_feature = bool(time_as_feature)
        self.root_path = root_path

        super().__init__()

    def _pick_eeg_index(self, ch_names: List[str]) -> int:
        if self.eeg_channel and self.eeg_channel in ch_names:
            return ch_names.index(self.eeg_channel)
        for i, n in enumerate(ch_names):
            if "eeg" in n.lower():
                return i
        return 0

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """
        Generates classification data samples for a single patient.

        Args:
            patient (Any): A PyHealth patient object containing sleep events.

        Returns:
            List[Dict[str, Any]]: A list containing a dictionary for each sleep window sample with:
                - 'patient_id': Patient identifier.
                - 'night': The night number of the recording.
                - 'patient_age': Age of the patient.
                - 'patient_sex': Sex of the patient.
                - 'epoch_index': Global index of the 30s epoch.
                - 'window_in_epoch': Index of the window within the epoch.
                - 'signal': Original raw signal slice.
                - 'xt': Time-domain view tensor.
                - 'xd': Derivative view tensor.
                - 'xf': Frequency-domain view tensor.
                - 'label': Mapped 5-class sleep stage label.
        """
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
            signal_file = os.path.join(self.root_path, event.signal_file) if self.root_path else event.signal_file
            label_file = os.path.join(self.root_path, event.label_file) if self.root_path else event.label_file
            data = mne.io.read_raw_edf(
                signal_file,
                stim_channel="Event marker",
                infer_types=True,
                preload=False,
                verbose="error",
            )
            ann = mne.read_annotations(label_file)
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
            n_windows = n_times // win
            n_full = n_windows * win

            if n_epochs == 0 or n_windows == 0:
                continue

            # Vectorized window extraction
            segs = signals[:, :n_full].reshape(n_epochs, n_windows, win)
            if crop is not None:
                segs = segs[:, :, :crop]
            segs = segs.reshape(-1, segs.shape[-1]).astype(np.float32)

            # Vectorized metadata mapping
            mapping = np.array([0, 1, 2, 3, 3, 4], dtype=np.int64)
            mapped_labels = mapping[labels.astype(np.int64)]
            labels_rep = np.repeat(mapped_labels, n_windows)
            epoch_indices = np.repeat(np.arange(global_epoch, global_epoch + n_epochs), n_windows)
            window_indices = np.tile(np.arange(n_windows), n_epochs)

            global_epoch += n_epochs

            # Create numpy array directly
            X_np = segs[..., np.newaxis]
            xt_np, dx_np, xf_np = preprocess_mvcl_views_numpy(
                X_np,
                time_as_feature=self.time_as_feature
            )

            # Construct samples in a single loop
            for i in range(len(segs)):
                samples.append(
                    {
                        "patient_id": pid,
                        "night": event.night,
                        "patient_age": event.age,
                        "patient_sex": event.sex,
                        "epoch_index": int(epoch_indices[i]),
                        "window_in_epoch": int(window_indices[i]),
                        "signal": segs[i][np.newaxis, :].copy(),
                        "xt": torch.from_numpy(xt_np[i]),
                        "xd": torch.from_numpy(dx_np[i]),
                        "xf": torch.from_numpy(xf_np[i]),
                        "label": int(labels_rep[i]),
                    }
                )

        return samples


def normalize_mvcl_numpy(
    X_train: np.ndarray,
    X_test: np.ndarray,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = np.maximum(X_train.std(axis=(0, 1), keepdims=True), epsilon)
    return (
        (X_train - mean) / std,
        (X_test - mean) / std,
        mean,
        std,
    )


def add_time_feature_numpy(X: np.ndarray) -> np.ndarray:
    """X: [num_samples, sequence_length, num_features] -> concat time in last dim."""
    num_samples, seq_length, _ = X.shape
    time_index = np.linspace(0, 1, num=seq_length, dtype=X.dtype)
    time_feature = np.broadcast_to(
        time_index.reshape(1, seq_length, 1),
        (num_samples, seq_length, 1)
    )
    return np.concatenate([time_feature, X], axis=-1)


def get_dx_torchcde_equivalent_numpy(X: np.ndarray) -> np.ndarray:
    """
    Pure NumPy equivalent of torchcde Hermite cubic spline derivative 
    evaluated at the knot points with backward differences.
    """
    N, L, D = X.shape
    dx = np.zeros_like(X)
    
    if L < 2:
        return dx
        
    dt = 1.0 / (L - 1)
    
    # derivs[i] = X[i+1] - X[i]
    derivs = X[:, 1:, :] - X[:, :-1, :]
    
    # derivs_prev[i] = derivs[i-1] for i > 0, and derivs[0] for i = 0
    derivs_prev = np.concatenate([derivs[:, :1, :], derivs[:, :-1, :]], axis=1)
    
    # For i = 0
    dx[:, 0, :] = derivs[:, 0, :]
    
    # For i > 0
    factor = (4 - 3 * dt) * dt
    b = derivs_prev
    D_diff = derivs - b
    
    dx[:, 1:, :] = b + D_diff * factor
    
    return dx


def get_xf_numpy(X: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.fft(X, axis=1)).astype(np.float32)


def preprocess_mvcl_views_numpy(
    X: np.ndarray,
    time_as_feature: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the same logical pipeline as ``preprocess_data`` when train and test
    are the same tensor (per-domain batch, e.g. all windows of one patient).

    X: float numpy array **[N, L, D]** (e.g. ``D=1`` for single-channel EEG; **L** is time length).

    Returns xt, dx, xf each [N, L, D] or [N, L, D+1] if ``time_as_feature``.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X shape [N, L, D], got {tuple(X.shape)}")
    xt_tr, xt_te, _, _ = normalize_mvcl_numpy(X, X)
    xt = xt_tr

    dx_raw = get_dx_torchcde_equivalent_numpy(xt)

    dx_tr, dx_te, _, _ = normalize_mvcl_numpy(dx_raw, dx_raw)
    dx = dx_tr

    xf_raw = get_xf_numpy(xt)
    xf_tr, xf_te, _, _ = normalize_mvcl_numpy(xf_raw, xf_raw)
    xf = xf_tr

    if time_as_feature:
        xt = add_time_feature_numpy(xt)
        dx = add_time_feature_numpy(dx)
        xf = add_time_feature_numpy(xf)

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

    # preprocess_mvcl_views_numpy expects [N, L, D], use D=1 for single-channel EEG.
    signal_np = np.ascontiguousarray(signal_array)[..., np.newaxis]
    xt_np, dx_np, xf_np = preprocess_mvcl_views_numpy(signal_np, time_as_feature=time_as_feature)

    samples: List[Dict[str, Any]] = []
    for i in range(signal_array.shape[0]):
        samples.append(
            {
                "patient_id": f"{patient_id_prefix}_{i}",
                "record_id": f"{record_id_prefix}_{i}",
                "signal": signal_array[i][np.newaxis, :].copy(),
                "xt": torch.from_numpy(xt_np[i]),
                "xd": torch.from_numpy(dx_np[i]),
                "xf": torch.from_numpy(xf_np[i]),
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
            "xt": "tensor",
            "xd": "tensor",
            "xf": "tensor",
        },
        output_schema={"label": "multiclass"},
        dataset_name=dataset_name,
        task_name=task_name,
        in_memory=in_memory,
    )

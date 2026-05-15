from __future__ import annotations
import os
import pickle
from types import SimpleNamespace
from typing import Dict, List, Optional, Callable
import numpy as np
from pyhealth.datasets import BaseSignalDataset


def apply_nonstationarity(
    signal: np.ndarray,
    mode: str = "none",
    change_type: str = "mean",
    magnitude: float = 0.2,
    duration_ratio: float = 0.2,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Applies simple nonstationarity to a 1D signal.

    This is a lightweight, paper-inspired augmentation. It introduces one
    changepoint and modifies a contiguous segment by shifting the mean,
    scaling the standard deviation, or both.

    Args:
        signal: 1D signal array.
        mode: one of {"none", "random", "learned"}.
        change_type: one of {"mean", "std", "both"}.
        magnitude: base change magnitude.
        duration_ratio: fraction of the signal length to modify.
        random_state: optional RNG seed.

    Returns:
        Augmented 1D signal with the same shape as the input.
    """
    x = np.asarray(signal, dtype=float).copy()
    if x.ndim != 1:
        raise ValueError("signal must be a 1D array")
    if len(x) == 0 or mode == "none":
        return x

    if change_type not in {"mean", "std", "both"}:
        raise ValueError("change_type must be one of {'mean', 'std', 'both'}")
    if mode not in {"none", "random", "learned"}:
        raise ValueError("mode must be one of {'none', 'random', 'learned'}")

    rng = np.random.default_rng(random_state)
    n = len(x)

    seg_len = max(1, int(round(duration_ratio * n)))
    seg_len = min(seg_len, n)

    start_low = max(0, n // 4)
    start_high = max(start_low + 1, n - seg_len + 1)
    start = int(rng.integers(start_low, start_high))
    end = start + seg_len

    if mode == "random":
        mean_delta = rng.normal(0.0, magnitude)
        std_scale = max(0.1, 1.0 + rng.normal(0.0, magnitude))
    else:
        # "learned" mode: deterministic-ish paper-inspired defaults
        mean_delta = magnitude
        std_scale = 1.0 + magnitude

    segment = x[start:end]
    seg_mean = float(np.mean(segment))
    seg_std = float(np.std(segment))

    if change_type in {"std", "both"} and seg_std > 1e-8:
        segment = (segment - seg_mean) * std_scale + seg_mean

    if change_type in {"mean", "both"}:
        ramp = np.linspace(0.0, 1.0, num=len(segment), endpoint=True)
        segment = segment + mean_delta * ramp

    x[start:end] = segment
    return x


class WESADNonstationaryDataset(BaseSignalDataset):
    """WESAD-based signal dataset with optional nonstationarity augmentation.

    Expected raw file format for each subject:
        <root>/<subject_id>.pkl

    Each subject pickle should contain:
        {
            "eda": np.ndarray,   # shape (T,)
            "label": np.ndarray, # shape (T,)
            "fs": int,           # usually 4 for wrist EDA
        }

    This class keeps the base dataset minimal and delegates window creation
    and label generation to the task function.
    """

    def __init__(
        self,
        root: str,
        augmentation_mode: str = "none",
        change_type: str = "mean",
        magnitude: float = 0.2,
        duration_ratio: float = 0.2,
        dataset_name: str = "WESADNonstationary",
        dev: bool = False,
        refresh_cache: bool = False,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        self.root = root
        self.dataset_name = dataset_name
        self.dev = dev
        self.refresh_cache = refresh_cache
        self.augmentation_mode = augmentation_mode
        self.change_type = change_type
        self.magnitude = magnitude
        self.duration_ratio = duration_ratio
        self.random_state = random_state

        super().__init__(
            root=root,
            dataset_name=dataset_name,
            dev=dev,
            refresh_cache=refresh_cache,
            **kwargs,
        )

        # Fallback cache/output directory for processed subject files.
        # If the base class defines filepath, use it. Otherwise create one.
        if not hasattr(self, "filepath") or self.filepath is None:
            self.filepath = os.path.join(self.root, "_processed")
        os.makedirs(self.filepath, exist_ok=True)

        self.patients = self.process_data()

    def process_EEG_data(self):
        raise NotImplementedError(
            "WESADNonstationaryDataset focuses on EDA, not EEG."
        )
    
    def set_task(self, task_fn: Callable, **task_kwargs):
        """Applies a task function to the dataset and returns a sample wrapper.

        This is a lightweight compatibility method for environments where the
        inherited base class does not provide set_task().

        Args:
            task_fn: callable that takes a single patient record list and returns
                a list of sample dicts.
            **task_kwargs: keyword arguments forwarded to task_fn.

        Returns:
            A lightweight object with a `.samples` attribute containing all
            generated samples.
        """
        all_samples = []
        patient_to_index = {}
        visit_to_index = {}

        for patient_id, record in self.patients.items():
            samples = task_fn(record, **task_kwargs)
            start_idx = len(all_samples)
            all_samples.extend(samples)
            end_idx = len(all_samples)

            patient_to_index[patient_id] = list(range(start_idx, end_idx))
            for i in range(start_idx, end_idx):
                visit_id = all_samples[i].get("visit_id", str(i))
                visit_to_index.setdefault(visit_id, []).append(i)

        return SimpleNamespace(
            samples=all_samples,
            patient_to_index=patient_to_index,
            visit_to_index=visit_to_index,
            task_fn=task_fn.__name__,
            dataset_name=self.dataset_name,
        )

    def process_data(self) -> Dict[str, List[Dict]]:
        """Processes subject-level WESAD files into a patient dictionary.

        Returns:
            Dict mapping patient_id -> list of one record dict.
        """
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        subject_files = sorted(
            f for f in os.listdir(self.root) if f.endswith(".pkl")
        )
        if self.dev:
            subject_files = subject_files[:2]

        patients: Dict[str, List[Dict]] = {}

        for subject_file in subject_files:
            patient_id = os.path.splitext(subject_file)[0]
            subject_path = os.path.join(self.root, subject_file)

            with open(subject_path, "rb") as f:
                subject_data = pickle.load(f)

            if "eda" not in subject_data or "label" not in subject_data:
                raise ValueError(
                    f"Subject file {subject_file} must contain 'eda' and 'label'"
                )
            if "fs" not in subject_data:
                raise ValueError(f"Subject file {subject_file} must contain 'fs'")

            eda = np.asarray(subject_data["eda"], dtype=float)
            label = np.asarray(subject_data["label"], dtype=int)
            fs = int(subject_data["fs"])

            if eda.ndim != 1:
                raise ValueError(f"EDA signal for {subject_file} must be 1D")
            if label.ndim != 1:
                raise ValueError(f"Label signal for {subject_file} must be 1D")
            if len(eda) != len(label):
                raise ValueError(
                    f"EDA and label length mismatch in {subject_file}: "
                    f"{len(eda)} vs {len(label)}"
                )
            if fs <= 0:
                raise ValueError(f"Sampling rate must be positive in {subject_file}")

            if self.augmentation_mode != "none":
                eda = apply_nonstationarity(
                    signal=eda,
                    mode=self.augmentation_mode,
                    change_type=self.change_type,
                    magnitude=self.magnitude,
                    duration_ratio=self.duration_ratio,
                    random_state=self.random_state,
                )

            save_dir = os.path.join(self.filepath, patient_id)
            os.makedirs(save_dir, exist_ok=True)

            processed_subject_path = os.path.join(save_dir, f"{patient_id}.pkl")
            with open(processed_subject_path, "wb") as f:
                pickle.dump({"eda": eda, "label": label, "fs": fs}, f)

            patients[patient_id] = [
                {
                    "load_from_path": save_dir,
                    "patient_id": patient_id,
                    "signal_file": f"{patient_id}.pkl",
                    "save_to_path": save_dir,
                }
            ]

        return patients
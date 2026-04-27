from __future__ import annotations
import os
import pickle
from typing import Dict, List
import numpy as np


def wesad_stress_detection_fn(
    record: List[Dict],
    window_sec: int = 10,
    shift_sec: int = 5,
    stress_label: int = 2,
    baseline_label: int = 1,
    keep_baseline_only: bool = True,
) -> List[Dict]:
    """Processes one WESAD subject into binary stress-detection samples.

    Args:
        record: singleton list containing one subject record from
            WESADNonstationaryDataset.
        window_sec: window length in seconds.
        shift_sec: sliding step size in seconds.
        stress_label: raw label value representing stress.
        baseline_label: raw label value representing baseline.
        keep_baseline_only: if True, only keep windows whose labels are entirely
            baseline or contain stress. This avoids mixing unrelated states.

    Returns:
        A list of sample dicts. Each sample contains:
            - patient_id
            - visit_id
            - record_id
            - epoch_path
            - label
    """
    samples: List[Dict] = []

    for visit in record:
        root = visit["load_from_path"]
        patient_id = visit["patient_id"]
        signal_file = visit["signal_file"]
        save_path = visit["save_to_path"]

        subject_path = os.path.join(root, signal_file)
        with open(subject_path, "rb") as f:
            subject_data = pickle.load(f)

        eda = np.asarray(subject_data["eda"], dtype=float)
        labels = np.asarray(subject_data["label"], dtype=int)
        fs = int(subject_data["fs"])

        if window_sec <= 0 or shift_sec <= 0:
            raise ValueError("window_sec and shift_sec must be positive")
        if fs <= 0:
            raise ValueError("Sampling rate must be positive")

        window_size = fs * window_sec
        shift_size = fs * shift_sec

        if len(eda) < window_size:
            continue

        for index, start in enumerate(range(0, len(eda) - window_size + 1, shift_size)):
            end = start + window_size
            eda_window = eda[start:end]
            label_window = labels[start:end]

            has_stress = np.any(label_window == stress_label)
            is_all_baseline = np.all(label_window == baseline_label)

            if keep_baseline_only:
                if has_stress:
                    y = 1
                elif is_all_baseline:
                    y = 0
                else:
                    continue
            else:
                y = 1 if has_stress else 0

            save_file_path = os.path.join(
                save_path, f"{patient_id}-stress-{index}.pkl"
            )
            with open(save_file_path, "wb") as f:
                pickle.dump(
                    {
                        "signal": eda_window,
                        "label": y,
                        "fs": fs,
                        "start": start,
                        "end": end,
                    },
                    f,
                )

            samples.append(
                {
                    "patient_id": patient_id,
                    "visit_id": patient_id,
                    "record_id": len(samples) + 1,
                    "epoch_path": save_file_path,
                    "label": y,
                }
            )

    return samples
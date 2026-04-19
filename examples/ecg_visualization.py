"""
Reusable ECG visualization example using task outputs with pulse/non-pulse comparison.

This script is designed to be extensible across ECG datasets. It currently includes
LUDB integration and compares:

1) Non-pulse mode (full window per lead)
2) Pulse mode (all pulse-centered windows per lead)

The plotting utilities are dataset-agnostic as long as task samples provide:
    - "signal": 1D or (C, T) array/tensor
    - "mask":   1D segmentation labels
    - optional metadata keys like "patient_id", "record_id", "lead"

Usage:
    python examples/ecg_visualization.py \
        --dataset ludb \
        --root /path/to/physionet.org/files/ludb/1.0.1/data \
        --patient-id 1 \
        --lead i \
        --pulse-window 250 \
        --filter-incomplete-pulses \
        --dev \
        --save-path ./ecg_compare_all_pulses.png

Requirements:
    pip install matplotlib
    (and dataset/task dependencies, e.g. wfdb for LUDB delineation)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from pyhealth.datasets import LUDBDataset
from pyhealth.tasks import ECGDelineationLUDB

# ---------------------------------------------------------------------
# Generic utilities (dataset/task-agnostic)
# ---------------------------------------------------------------------

CLASS_NAMES = {
    0: "background",
    1: "P",
    2: "QRS",
    3: "T",
}

CLASS_COLORS = {
    0: "#9E9E9E",
    1: "#4CAF50",
    2: "#F44336",
    3: "#2196F3",
}


def _to_numpy(x: Any) -> np.ndarray:
    """Convert tensor/array/list-like object to np.ndarray safely."""
    if x is None:
        raise ValueError("Expected array-like input, got None.")

    # Torch-like tensor support without hard dependency
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        x = x.detach().cpu().numpy()

    return np.asarray(x)


def _signal_1d(signal: Any) -> np.ndarray:
    """Normalize signal to shape (T,) for visualization."""
    arr = _to_numpy(signal)

    if arr.ndim == 1:
        return arr.astype(np.float32)

    if arr.ndim == 2:
        # expected ECG delineation shape is (1, T), but handle generic multi-channel
        if arr.shape[0] == 1:
            return arr[0].astype(np.float32)
        return arr[0].astype(np.float32)

    raise ValueError(f"Unsupported signal shape: {arr.shape}")


def _mask_1d(mask: Any) -> np.ndarray:
    """Normalize mask to shape (T,) with integer labels."""
    arr = _to_numpy(mask)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(np.int64)


def _collect_samples(
    sample_dataset: Any,
    patient_id: str,
    lead: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Collect all samples for a patient (optionally filtered by lead).

    Uses `patient_to_index` if present (SampleDataset), otherwise scans dataset.
    """
    pid = str(patient_id)
    candidates: List[Dict[str, Any]] = []

    if hasattr(sample_dataset, "patient_to_index"):
        indices = sample_dataset.patient_to_index.get(pid, [])
        for idx in indices:
            s = sample_dataset[idx]
            if lead is not None and str(s.get("lead", "")).lower() != lead.lower():
                continue
            if "signal" in s and "mask" in s:
                candidates.append(s)
    else:
        # fallback scan
        for idx in range(len(sample_dataset)):
            s = sample_dataset[idx]
            if str(s.get("patient_id", "")) != pid:
                continue
            if lead is not None and str(s.get("lead", "")).lower() != lead.lower():
                continue
            if "signal" in s and "mask" in s:
                candidates.append(s)

    return candidates


def _plot_signal_with_mask(
    ax: plt.Axes,
    signal: np.ndarray,
    mask: np.ndarray,
    title: str,
    alpha: float = 0.25,
) -> None:
    """Plot ECG signal and color-overlay segmentation mask classes."""
    t = np.arange(len(signal))
    y_min, y_max = float(signal.min()), float(signal.max())
    if np.isclose(y_min, y_max):
        y_min -= 1e-3
        y_max += 1e-3

    ax.plot(t, signal, linewidth=1.0, color="black")

    # overlay class regions
    unique_classes = sorted(np.unique(mask).tolist())
    for cls in unique_classes:
        cls_int = int(cls)
        cls_mask = mask == cls_int
        if not np.any(cls_mask):
            continue
        color = CLASS_COLORS.get(cls_int, "#BDBDBD")
        ax.fill_between(
            t,
            y_min,
            y_max,
            where=cls_mask,
            color=color,
            alpha=alpha,
            step="mid",
            label=f"{cls_int}:{CLASS_NAMES.get(cls_int, 'unknown')}",
        )

    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.2)

    # deduplicate legend labels
    handles, labels = ax.get_legend_handles_labels()
    dedup = {}
    for h, l in zip(handles, labels):
        dedup[l] = h
    if dedup:
        ax.legend(dedup.values(), dedup.keys(), fontsize=8, loc="upper right")


def visualize_comparison_all_pulses(
    raw_sample: Dict[str, Any],
    pulse_samples: List[Dict[str, Any]],
    save_path: Optional[Path] = None,
    max_pulses: int = 0,
) -> None:
    """
    Create stacked comparison:
      - first row: non-pulse (full record) sample
      - next rows: all pulse samples (or first `max_pulses` if > 0)
    """
    if not pulse_samples:
        raise ValueError("No pulse samples to visualize.")

    if max_pulses > 0:
        pulse_samples = pulse_samples[:max_pulses]

    nrows = 1 + len(pulse_samples)
    fig, axes = plt.subplots(
        nrows, 1, figsize=(14, max(4, 2.8 * nrows)), constrained_layout=True
    )
    if nrows == 1:
        axes = [axes]

    # Raw sample panel
    raw_signal = _signal_1d(raw_sample["signal"])
    raw_mask = _mask_1d(raw_sample["mask"])
    raw_title = (
        f"Non-pulse mode | patient={raw_sample.get('patient_id')} "
        f"| lead={raw_sample.get('lead')} | record_id={raw_sample.get('record_id')} "
        f"| T={len(raw_signal)}"
    )
    _plot_signal_with_mask(axes[0], raw_signal, raw_mask, raw_title)

    # Pulse panels
    for i, pulse_sample in enumerate(pulse_samples, start=1):
        pulse_signal = _signal_1d(pulse_sample["signal"])
        pulse_mask = _mask_1d(pulse_sample["mask"])
        pulse_title = (
            f"Pulse {i}/{len(pulse_samples)} | patient={pulse_sample.get('patient_id')} "
            f"| lead={pulse_sample.get('lead')} | record_id={pulse_sample.get('record_id')} "
            f"| T={len(pulse_signal)}"
        )
        _plot_signal_with_mask(axes[i], pulse_signal, pulse_mask, pulse_title)

    fig.suptitle("ECG Delineation: Non-pulse vs All Pulse-aligned Splits", fontsize=14)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)
        print(f"Saved figure to: {save_path}")

    plt.show()


# ---------------------------------------------------------------------
# Dataset/task adapters
# ---------------------------------------------------------------------


def build_ludb_sample_datasets(
    root: str,
    dev: bool,
    num_workers: int,
    pulse_window: int,
    filter_incomplete_pulses: bool,
):
    """Build non-pulse and pulse-mode SampleDatasets for LUDB."""
    dataset = LUDBDataset(root=root, dev=dev, num_workers=num_workers)

    raw_task = ECGDelineationLUDB(
        split_by_pulse=False,
        pulse_window=pulse_window,
        filter_incomplete_pulses=False,
    )
    pulse_task = ECGDelineationLUDB(
        split_by_pulse=True,
        pulse_window=pulse_window,
        filter_incomplete_pulses=filter_incomplete_pulses,
    )

    raw_ds = dataset.set_task(raw_task, num_workers=num_workers)
    pulse_ds = dataset.set_task(pulse_task, num_workers=num_workers)
    return raw_ds, pulse_ds


# Future extension point:
# Add adapters for other ECG datasets (e.g., QTDB) with the same return interface.
DATASET_BUILDERS = {
    "ludb": build_ludb_sample_datasets,
}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reusable ECG visualization from task outputs (non-pulse vs all pulse splits)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ludb",
        choices=sorted(DATASET_BUILDERS.keys()),
        help="ECG dataset adapter to use.",
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default="1",
        help="Patient ID to visualize (record ID for LUDB).",
    )
    parser.add_argument(
        "--lead",
        type=str,
        default="i",
        help="Lead name to visualize (e.g., i, ii, v1).",
    )
    parser.add_argument(
        "--pulse-window",
        type=int,
        default=250,
        help="Pulse half-window in samples for pulse mode.",
    )
    parser.add_argument(
        "--filter-incomplete-pulses",
        action="store_true",
        help="If set, keep only pulse windows that contain P, QRS, and T labels.",
    )
    parser.add_argument(
        "--max-pulses",
        type=int,
        default=0,
        help="Optional cap on number of pulse windows to plot (0 = show all).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Workers for set_task processing.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use dataset dev mode.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="",
        help="Optional output image path. If empty, no file is saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    builder = DATASET_BUILDERS[args.dataset]
    raw_ds, pulse_ds = builder(
        root=args.root,
        dev=args.dev,
        num_workers=args.num_workers,
        pulse_window=args.pulse_window,
        filter_incomplete_pulses=args.filter_incomplete_pulses,
    )

    raw_samples = _collect_samples(raw_ds, patient_id=args.patient_id, lead=args.lead)
    pulse_samples = _collect_samples(
        pulse_ds, patient_id=args.patient_id, lead=args.lead
    )

    if not raw_samples:
        raise ValueError(
            f"No non-pulse sample found for patient_id='{args.patient_id}', lead='{args.lead}'."
        )
    if not pulse_samples:
        raise ValueError(
            "No pulse samples found. Try disabling --filter-incomplete-pulses "
            "or selecting a different patient/lead."
        )

    raw_sample = raw_samples[0]
    save_path = Path(args.save_path) if args.save_path else None
    visualize_comparison_all_pulses(
        raw_sample=raw_sample,
        pulse_samples=pulse_samples,
        save_path=save_path,
        max_pulses=args.max_pulses,
    )


if __name__ == "__main__":
    main()

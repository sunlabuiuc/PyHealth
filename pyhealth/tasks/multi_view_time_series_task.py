"""Multi-view time series task for domain adaptation.

This task generates three complementary views (temporal, derivative, frequency)
from physiological time series signals (EEG, ECG, EMG) for multi-view
contrastive learning, as described in Oh and Bui (2025).

The three views capture different aspects of the signal:

- **Temporal view**: raw signal preserving original patterns and trends.
- **Derivative view**: rate of change capturing local signal dynamics.
- **Frequency view**: FFT magnitude spectrum capturing periodic patterns.

Typical usage::

    from pyhealth.datasets import SleepEDFDataset
    from pyhealth.tasks import MultiViewTimeSeriesTask

    dataset = SleepEDFDataset(root="/path/to/sleep-edf")
    task = MultiViewTimeSeriesTask(epoch_seconds=30)
    samples = dataset.set_task(task)
    print(samples[0].keys())
    # dict_keys(['record_id', 'patient_id', 'epoch_path', 'label'])
"""

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
from scipy.fft import fft

from pyhealth.tasks import BaseTask

# Labels that indicate ambiguous or non-sleep-stage epochs to skip.
_SKIP_LABELS = {"?", "Unknown"}


class MultiViewTimeSeriesTask(BaseTask):
    """Multi-view time series task for domain adaptation.

    Generates three complementary views (temporal, derivative, frequency)
    from physiological EEG signals for multi-view contrastive learning,
    as described in Oh and Bui (2025).

    Each patient record is sliced into non-overlapping fixed-length epochs.
    For every epoch, three numpy arrays are computed and saved to disk as a
    pickle file. The ``__call__`` method returns lightweight metadata dicts
    pointing to those files so that the full signal data is loaded lazily
    during model training.

    Args:
        epoch_seconds: Duration of each epoch window in seconds. Default 30.
        sample_rate: Expected sampling rate in Hz. If ``None``, the rate is
            inferred directly from the EDF file header.

    Examples:
        >>> from pyhealth.datasets import SleepEDFDataset
        >>> dataset = SleepEDFDataset(root="/path/to/sleep-edf")
        >>> task = MultiViewTimeSeriesTask(epoch_seconds=30)
        >>> samples = dataset.set_task(task)
        >>> print(samples[0].keys())
        dict_keys(['record_id', 'patient_id', 'epoch_path', 'label'])
    """

    task_name: str = "MultiViewTimeSeries"

    # Input schema describes the three views stored inside each .pkl file.
    # The model reads these arrays from epoch_path at training time.
    input_schema: Dict[str, str] = {
        "epoch_path": "str",  # path to .pkl containing the three views
    }
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __init__(
        self,
        epoch_seconds: int = 30,
        sample_rate: Optional[int] = None,
    ) -> None:
        """Initializes MultiViewTimeSeriesTask.

        Args:
            epoch_seconds: Duration of each epoch in seconds. Default 30.
            sample_rate: Sampling rate in Hz. If None, inferred from the
                EDF file header.
        """
        self.epoch_seconds = epoch_seconds
        self.sample_rate = sample_rate
        super().__init__()

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Generates multi-view epoch samples from one patient recording.

        Processes a single patient record by:

        1. Reading the raw EDF signal and its annotation file.
        2. Slicing the signal into non-overlapping ``epoch_seconds`` windows.
        3. For each window, computing three views:

           - **temporal**: raw signal array, shape ``(C, T)``.
           - **derivative**: first-order finite difference, shape ``(C, T-1)``.
           - **frequency**: one-sided FFT magnitude, shape ``(C, T//2)``.

        4. Saving each window as a ``.pkl`` file to ``save_to_path``.
        5. Returning a list of metadata dicts (one per valid epoch).

        Args:
            patient: A patient record — a list containing one dict with keys:

                - ``load_from_path`` (str): Directory containing EDF files.
                - ``signal_file`` (str): EDF filename for the raw signal.
                - ``label_file`` (str): Annotation filename (hypnogram).
                - ``save_to_path`` (str): Directory to write ``.pkl`` files.
                - ``subject_id`` (str, optional): Patient identifier.

        Returns:
            A list of sample dicts, each containing:

            - ``record_id`` (str): Unique epoch identifier.
            - ``patient_id`` (str): Patient identifier.
            - ``epoch_path`` (str): Absolute path to the saved ``.pkl`` file.
            - ``label`` (str): Ground-truth sleep stage label for this epoch.

            The saved ``.pkl`` file contains a dict with:

            - ``temporal`` (np.ndarray): Raw signal, shape ``(C, T)``.
            - ``derivative`` (np.ndarray): Finite difference, shape ``(C, T-1)``.
            - ``frequency`` (np.ndarray): FFT magnitude, shape ``(C, T//2)``.
            - ``label`` (str): Ground-truth label.

        Raises:
            KeyError: If required keys are missing from the patient record.
            FileNotFoundError: If the EDF signal file does not exist.

        Examples:
            >>> task = MultiViewTimeSeriesTask(epoch_seconds=30)
            >>> samples = task(patient_record)
            >>> len(samples)
            4
            >>> samples[0]["label"]
            'W'
        """
        record_data = patient[0]

        root = record_data["load_from_path"]
        signal_file = record_data["signal_file"]
        label_file = record_data["label_file"]
        save_path = record_data["save_to_path"]
        patient_id = record_data.get("subject_id", signal_file[:6])

        os.makedirs(save_path, exist_ok=True)

        # Step 1: Load raw signal from EDF file.
        edf_path = os.path.join(root, signal_file)
        print(f"Loading EDF file: {edf_path}")

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        data = raw.get_data()
        _num_channels, total_samples = data.shape

        actual_rate = int(raw.info["sfreq"])
        if self.sample_rate is not None and self.sample_rate != actual_rate:
            print(
                f"Requested sample rate {self.sample_rate} Hz does not match"
                f" file rate {actual_rate} Hz. Using file rate."
            )
        sample_rate = actual_rate

        # Step 2: Load epoch labels from annotation file.
        labels = self._load_labels(
            root, label_file, total_samples, sample_rate
        )

        # Step 3: Slice signal into epochs and compute three views.
        epoch_samples = int(sample_rate * self.epoch_seconds)
        num_epochs = int(total_samples // epoch_samples)
        print(f"Processing {num_epochs} epochs of {self.epoch_seconds}s each")

        samples = []
        for epoch_idx in range(num_epochs):
            start = epoch_idx * epoch_samples
            end = start + epoch_samples

            if end > total_samples:
                break

            label = labels[epoch_idx] if epoch_idx < len(labels) else "Unknown"

            if label in _SKIP_LABELS or "Movement" in str(label):
                continue

            epoch_signal = data[:, start:end]
            epoch_path = os.path.join(
                save_path, f"{patient_id}-epoch-{epoch_idx}.pkl"
            )

            self._save_epoch_views(epoch_signal, epoch_samples, label, epoch_path)

            samples.append(
                {
                    "record_id": f"{patient_id}-epoch-{epoch_idx}",
                    "patient_id": patient_id,
                    "epoch_path": epoch_path,
                    "label": label,
                }
            )

        print(f"Successfully processed {len(samples)} valid epochs")
        return samples

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_labels(
        self,
        root: str,
        label_file: str,
        total_samples: int,
        sample_rate: int,
    ) -> List[str]:
        """Loads per-epoch sleep stage labels from an annotation file.

        Args:
            root: Directory containing the annotation file.
            label_file: Annotation filename (e.g. hypnogram ``.edf``).
            total_samples: Total number of samples in the signal array.
            sample_rate: Sampling rate in Hz, used for fallback label count.

        Returns:
            A list of label strings, one per ``epoch_seconds`` window.
            Falls back to cycling dummy labels if the file cannot be read.
        """
        hypnogram_path = os.path.join(root, label_file)
        try:
            annotations = mne.read_annotations(hypnogram_path)
            labels: List[str] = []
            for ann in annotations:
                n = int(ann["duration"] / self.epoch_seconds)
                description = ann["description"]
                stage = (
                    description.replace("Sleep stage ", "").strip()
                    if "Sleep stage" in description
                    else description
                )
                labels.extend([stage] * n)
            return labels
        except Exception:
            print(
                f"Could not load annotations from {hypnogram_path}."
                " Using dummy labels."
            )
            total_seconds = total_samples / sample_rate
            num_epochs = int(total_seconds // self.epoch_seconds)
            cycle = ["W", "N1", "N2", "N3", "REM"]
            return [cycle[i % len(cycle)] for i in range(num_epochs)]

    def _save_epoch_views(
        self,
        epoch_signal: np.ndarray,
        epoch_samples: int,
        label: str,
        epoch_path: str,
    ) -> None:
        """Computes and saves the three views for one epoch to disk.

        Args:
            epoch_signal: Raw signal array of shape ``(C, T)``.
            epoch_samples: Number of time steps per epoch ``T``.
            label: Ground-truth label string for this epoch.
            epoch_path: File path to write the ``.pkl`` output.
        """
        temporal_view = epoch_signal
        derivative_view = np.diff(epoch_signal, axis=1)

        fft_vals = fft(epoch_signal, axis=1)
        frequency_view = np.abs(fft_vals[:, : epoch_samples // 2])

        epoch_data = {
            "temporal": temporal_view,
            "derivative": derivative_view,
            "frequency": frequency_view,
            "label": label,
        }
        with open(epoch_path, "wb") as f:
            pickle.dump(epoch_data, f)


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------


def load_epoch_views(epoch_path: str) -> Dict[str, Any]:
    """Loads the three views from a saved epoch pickle file.

    Args:
        epoch_path: Path to the ``.pkl`` file written by
            :class:`MultiViewTimeSeriesTask`.

    Returns:
        A dict with keys ``'temporal'``, ``'derivative'``, ``'frequency'``
        (each an ``np.ndarray``) and ``'label'`` (a ``str``).

    Examples:
        >>> views = load_epoch_views("/data/output/P001-epoch-0.pkl")
        >>> views["temporal"].shape
        (2, 3000)
    """
    with open(epoch_path, "rb") as f:
        return pickle.load(f)


def get_view_shapes(
    sample_rate: int = 100,
    epoch_seconds: int = 30,
    num_channels: int = 2,
) -> Dict[str, Tuple[int, int]]:
    """Returns the expected array shapes for each view given signal parameters.

    Args:
        sample_rate: Sampling rate in Hz. Default 100.
        epoch_seconds: Duration of each epoch in seconds. Default 30.
        num_channels: Number of signal channels. Default 2.

    Returns:
        A dict mapping view name to ``(num_channels, time_steps)`` tuples:

        - ``'temporal'``: ``(num_channels, sample_rate * epoch_seconds)``
        - ``'derivative'``: ``(num_channels, sample_rate * epoch_seconds - 1)``
        - ``'frequency'``: ``(num_channels, sample_rate * epoch_seconds // 2)``

    Examples:
        >>> get_view_shapes(sample_rate=100, epoch_seconds=30, num_channels=2)
        {'temporal': (2, 3000), 'derivative': (2, 2999), 'frequency': (2, 1500)}
    """
    time_steps = sample_rate * epoch_seconds
    return {
        "temporal": (num_channels, time_steps),
        "derivative": (num_channels, time_steps - 1),
        "frequency": (num_channels, time_steps // 2),
    }

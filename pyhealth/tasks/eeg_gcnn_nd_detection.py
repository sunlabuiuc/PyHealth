"""EEG-GCNN neurological disease detection task.

Implements the preprocessing and feature-extraction pipeline from:

    Wagh, N. & Varatharajah, Y. (2020). EEG-GCNN: Augmenting
    Electroencephalogram-based Neurological Disease Diagnosis using a
    Domain-guided Graph Convolutional Neural Network. ML4H @ NeurIPS 2020.
    https://proceedings.mlr.press/v136/wagh20a.html
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
import torch

from pyhealth.tasks.base_task import BaseTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants from the EEG-GCNN paper (Section 4)
# ---------------------------------------------------------------------------

# 8 bipolar channels used in the paper.
# Each entry is (anode_ref_name, cathode_ref_name, canonical_name).
BIPOLAR_CHANNELS: List[Tuple[str, str, str]] = [
    ("EEG F7-REF", "EEG F3-REF", "F7-F3"),
    ("EEG F8-REF", "EEG F4-REF", "F8-F4"),
    ("EEG T3-REF", "EEG C3-REF", "T7-C3"),
    ("EEG T4-REF", "EEG C4-REF", "T8-C4"),
    ("EEG T5-REF", "EEG P3-REF", "P7-P3"),
    ("EEG T6-REF", "EEG P4-REF", "P8-P4"),
    ("EEG O1-REF", "EEG P3-REF", "O1-P3"),
    ("EEG O2-REF", "EEG P4-REF", "O2-P4"),
]

# 3D MNI coordinates for the 8 bipolar channel mid-points (approximate).
# Used to build the spatial adjacency matrix.
CHANNEL_POSITIONS_MNI: np.ndarray = np.array(
    [
        [-0.054, 0.044, 0.038],   # F7-F3
        [0.054, 0.044, 0.038],    # F8-F4
        [-0.069, -0.014, 0.034],  # T7-C3
        [0.069, -0.014, 0.034],   # T8-C4
        [-0.059, -0.067, 0.034],  # P7-P3
        [0.059, -0.067, 0.034],   # P8-P4
        [-0.037, -0.094, 0.020],  # O1-P3
        [0.037, -0.094, 0.020],   # O2-P4
    ],
    dtype=np.float64,
)

# Frequency bands (Hz) from the paper.
DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "lower_beta": (12.0, 20.0),
    "higher_beta": (20.0, 30.0),
    "gamma": (30.0, 50.0),
}

NUM_CHANNELS = 8


class EEGGCNNDiseaseDetection(BaseTask):
    """Binary classification: patient-normal (0) vs healthy-control (1).

    For each EEG recording the task:

    1. Reads the EDF / BrainVision file via MNE.
    2. Resamples to ``resample_rate`` Hz, applies a 1 Hz high-pass and a
       ``notch_freq`` Hz notch filter.
    3. Computes the 8 bipolar channels defined in the paper.
    4. Segments the signal into non-overlapping ``window_sec``-second windows.
    5. For each window, extracts PSD band-power features (8 channels x
       ``len(bands)`` bands) via Welch's method.
    6. Computes an 8x8 graph adjacency matrix (spatial, functional, combined,
       or identity) to accompany each sample.

    Args:
        resample_rate: Target sampling rate in Hz. Default ``250``.
        highpass_freq: High-pass filter cutoff in Hz. Default ``1.0``.
        notch_freq: Notch filter frequency in Hz. Default ``50.0``.
        window_sec: Window length in seconds. Default ``10``.
        bands: Frequency bands to extract. Defaults to all 6 paper bands.
            Pass a subset dict to run a band-ablation experiment.
        adjacency_type: One of ``"combined"`` (default), ``"spatial"``,
            ``"functional"``, or ``"none"`` (identity matrix).
        connectivity_measure: ``"coherence"`` (default) or ``"wpli"``.

    Examples:
        >>> from pyhealth.datasets import EEGGCNNDataset
        >>> from pyhealth.tasks import EEGGCNNDiseaseDetection
        >>> dataset = EEGGCNNDataset(root="/data/eeg-gcnn/")
        >>> task = EEGGCNNDiseaseDetection(adjacency_type="spatial")
        >>> sample_dataset = dataset.set_task(task)
        >>> sample = sample_dataset[0]
        >>> print(sample["psd_features"].shape)  # (8, 6)
        >>> print(sample["adjacency"].shape)     # (8, 8)
    """

    task_name: str = "eeg_gcnn_nd_detection"
    input_schema: Dict[str, str] = {
        "psd_features": "tensor",
        "adjacency": "tensor",
    }
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(
        self,
        resample_rate: int = 250,
        highpass_freq: float = 1.0,
        notch_freq: float = 50.0,
        window_sec: int = 10,
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
        adjacency_type: str = "combined",
        connectivity_measure: str = "coherence",
    ) -> None:
        super().__init__()
        self.resample_rate = resample_rate
        self.highpass_freq = highpass_freq
        self.notch_freq = notch_freq
        self.window_sec = window_sec
        self.bands = bands if bands is not None else dict(DEFAULT_BANDS)
        if adjacency_type not in ("combined", "spatial", "functional", "none"):
            raise ValueError(
                f"adjacency_type must be 'combined', 'spatial', "
                f"'functional', or 'none', got '{adjacency_type}'"
            )
        self.adjacency_type = adjacency_type
        if connectivity_measure not in ("coherence", "wpli"):
            raise ValueError(
                f"connectivity_measure must be 'coherence' or 'wpli', "
                f"got '{connectivity_measure}'"
            )
        self.connectivity_measure = connectivity_measure

        # Pre-compute spatial adjacency (constant across recordings).
        self._spatial_adj = self._build_spatial_adjacency()

    # ------------------------------------------------------------------
    # Signal I/O and preprocessing
    # ------------------------------------------------------------------

    def _read_eeg(self, filepath: str) -> mne.io.BaseRaw:
        """Read an EEG file (EDF or BrainVision) into MNE Raw."""
        filepath_lower = filepath.lower()
        if filepath_lower.endswith(".edf"):
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose="error")
        elif filepath_lower.endswith(".vhdr"):
            raw = mne.io.read_raw_brainvision(
                filepath, preload=True, verbose="error"
            )
        else:
            raise ValueError(f"Unsupported EEG format: {filepath}")
        return raw

    def _preprocess(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Resample, high-pass, and notch-filter."""
        raw.filter(
            l_freq=self.highpass_freq,
            h_freq=None,
            verbose="error",
        )
        raw.notch_filter(self.notch_freq, verbose="error")
        raw.resample(self.resample_rate, verbose="error")
        return raw

    @staticmethod
    def _compute_bipolar(raw: mne.io.BaseRaw) -> np.ndarray:
        """Compute the 8 bipolar channels from reference montage.

        Returns:
            np.ndarray of shape ``(8, n_samples)``.
        """
        data = raw.get_data()
        ch_map = {
            name: idx for idx, name in enumerate(raw.ch_names)
        }
        bipolar = np.zeros((NUM_CHANNELS, data.shape[1]))
        for i, (anode, cathode, _) in enumerate(BIPOLAR_CHANNELS):
            bipolar[i] = data[ch_map[anode]] - data[ch_map[cathode]]
        return bipolar

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_psd_features(
        self, window: np.ndarray
    ) -> np.ndarray:
        """Extract PSD band-power features for one window.

        Args:
            window: shape ``(8, n_samples)``.

        Returns:
            np.ndarray of shape ``(8, n_bands)`` — log10 average PSD per
            channel per band via Welch's method.
        """
        from scipy.signal import welch

        fs = self.resample_rate
        band_list = list(self.bands.values())
        n_bands = len(band_list)
        features = np.zeros((NUM_CHANNELS, n_bands))

        freqs, pxx = welch(window, fs=fs, nperseg=min(fs * 2, window.shape[1]))

        for b_idx, (fmin, fmax) in enumerate(band_list):
            band_mask = (freqs >= fmin) & (freqs < fmax)
            if band_mask.any():
                features[:, b_idx] = np.log10(
                    pxx[:, band_mask].mean(axis=1) + 1e-10
                )

        return features

    # ------------------------------------------------------------------
    # Adjacency matrices
    # ------------------------------------------------------------------

    @staticmethod
    def _build_spatial_adjacency() -> np.ndarray:
        """Build spatial adjacency from inverse Euclidean distance.

        Returns:
            np.ndarray of shape ``(8, 8)`` with self-loops set to 1.
        """
        pos = CHANNEL_POSITIONS_MNI
        n = pos.shape[0]
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist[i, j] = np.linalg.norm(pos[i] - pos[j])
        # Inverse distance (avoid div-by-zero on diagonal)
        with np.errstate(divide="ignore"):
            adj = np.where(dist > 0, 1.0 / dist, 0.0)
        # Row-normalise
        row_sum = adj.sum(axis=1, keepdims=True)
        adj = adj / (row_sum + 1e-10)
        # Self-loops
        np.fill_diagonal(adj, 1.0)
        return adj

    def _build_functional_adjacency(
        self, window: np.ndarray
    ) -> np.ndarray:
        """Build functional adjacency from inter-channel connectivity.

        Args:
            window: shape ``(8, n_samples)``.

        Returns:
            np.ndarray of shape ``(8, 8)``.
        """
        from mne_connectivity import spectral_connectivity_epochs

        fs = self.resample_rate
        # spectral_connectivity_epochs expects (n_epochs, n_channels, n_times)
        data_3d = window[np.newaxis, :, :]

        conn = spectral_connectivity_epochs(
            data_3d,
            method=self.connectivity_measure,
            sfreq=fs,
            fmin=0.5,
            fmax=50.0,
            verbose="error",
        )
        conn_data = conn.get_data(output="dense")
        # Average across frequencies → (8, 8)
        adj = conn_data.mean(axis=-1)
        adj = np.abs(adj)
        # Symmetrise and set diagonal to 1
        adj = (adj + adj.T) / 2.0
        np.fill_diagonal(adj, 1.0)
        return adj

    def _build_adjacency(self, window: np.ndarray) -> np.ndarray:
        """Build the adjacency matrix according to ``adjacency_type``.

        Args:
            window: shape ``(8, n_samples)``.

        Returns:
            np.ndarray of shape ``(8, 8)``.
        """
        if self.adjacency_type == "none":
            return np.eye(NUM_CHANNELS)
        elif self.adjacency_type == "spatial":
            return self._spatial_adj.copy()
        elif self.adjacency_type == "functional":
            return self._build_functional_adjacency(window)
        else:  # combined
            spatial = self._spatial_adj
            functional = self._build_functional_adjacency(window)
            combined = (spatial + functional) / 2.0
            np.fill_diagonal(combined, 1.0)
            return combined

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process one patient and return a list of sample dicts.

        Each sample contains:
            - ``patient_id``: str
            - ``psd_features``: torch.FloatTensor of shape ``(8, n_bands)``
            - ``adjacency``: torch.FloatTensor of shape ``(8, 8)``
            - ``label``: int (0 = patient-normal, 1 = healthy-control)
        """
        pid = patient.patient_id
        samples: List[Dict[str, Any]] = []
        fs = self.resample_rate
        win_samples = int(self.window_sec * fs)

        for table in ("tuab", "lemon"):
            events = patient.get_events(table)
            for event in events:
                filepath = event.signal_file
                label = int(event.label)

                try:
                    raw = self._read_eeg(filepath)
                    raw = self._preprocess(raw)
                    bipolar = self._compute_bipolar(raw)
                except (ValueError, KeyError, RuntimeError) as exc:
                    logger.warning(
                        "Skipping %s for patient %s: %s",
                        filepath, pid, exc,
                    )
                    continue

                n_windows = bipolar.shape[1] // win_samples
                for w in range(n_windows):
                    start = w * win_samples
                    end = start + win_samples
                    window = bipolar[:, start:end]

                    psd_feat = self._extract_psd_features(window)
                    adj = self._build_adjacency(window)

                    samples.append(
                        {
                            "patient_id": pid,
                            "signal_file": filepath,
                            "psd_features": torch.FloatTensor(psd_feat),
                            "adjacency": torch.FloatTensor(adj),
                            "label": label,
                        }
                    )

        return samples

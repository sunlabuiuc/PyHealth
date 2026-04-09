import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .base_task import BaseTask

logger = logging.getLogger(__name__)

# Band name → column index in the (8, 6) node feature matrix.
BAND_NAMES: List[str] = ["delta", "theta", "alpha", "beta", "low_gamma", "high_gamma"]


class EEGGCNNClassification(BaseTask):
    """Per-window binary classification of neurological disease from EEG graphs.

    Classifies each EEG window as diseased (0) or healthy (1) using
    pre-computed PSD node features and a spectral-coherence + geodesic-distance
    adjacency matrix.

    Input schema keys:
        node_features : float32 tensor, shape (8, 6) — PSD band power per node
        adj_matrix    : float32 tensor, shape (8, 8) — combined edge weights

    Output schema keys:
        label : binary — 0 = diseased, 1 = healthy

    Args:
        excluded_bands: Frequency band names to zero out in node features at
            load time, for ablation studies. Valid values are elements of
            ``BAND_NAMES``. Default ``[]`` (all bands active).

    Examples:
        >>> from pyhealth.datasets import EEGGCNNDataset
        >>> from pyhealth.tasks import EEGGCNNClassification
        >>> dataset = EEGGCNNDataset(root="/path/to/eeg-gcnn")
        >>> # baseline
        >>> samples = dataset.set_task(EEGGCNNClassification())
        >>> # ablation: remove delta band
        >>> samples = dataset.set_task(EEGGCNNClassification(excluded_bands=["delta"]))
    """

    task_name: str = "EEGGCNNClassification"
    input_schema: Dict[str, str] = {
        "node_features": "tensor",
        "adj_matrix": "tensor",
    }
    output_schema: Dict[str, str] = {
        "label": "binary",
    }

    def __init__(self, excluded_bands: Optional[List[str]] = None) -> None:
        super().__init__()
        excluded_bands = excluded_bands or []
        invalid = [b for b in excluded_bands if b not in BAND_NAMES]
        if invalid:
            raise ValueError(
                f"Unknown band(s) {invalid}. Valid options: {BAND_NAMES}"
            )
        self.excluded_band_indices: List[int] = [
            BAND_NAMES.index(b) for b in excluded_bands
        ]

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Convert one patient's EEG windows into classification samples.

        Args:
            patient: A Patient object whose ``eeg_windows`` events each carry
                ``node_features_path`` and ``adj_matrix_path`` attributes
                pointing to pre-saved ``.npy`` files.

        Returns:
            List of sample dicts, one per EEG window, each containing:
                - ``node_features``: np.ndarray shape (8, 6), float32
                - ``adj_matrix``:    np.ndarray shape (8, 8), float32
                - ``label``:         int (0 or 1)
                - ``patient_id``:    str
                - ``window_idx``:    int
        """
        events = patient.get_events(event_type="eeg_windows")
        if len(events) == 0:
            logger.warning(
                "Patient %s has no eeg_windows events — returning empty sample list.",
                patient.patient_id,
            )
            return []

        samples = []
        for event in events:
            node_features_path = event.node_features_path
            adj_matrix_path = event.adj_matrix_path

            node_features = np.load(node_features_path).astype(np.float32)
            if self.excluded_band_indices:
                node_features[:, self.excluded_band_indices] = 0.0
            adj_matrix = np.load(adj_matrix_path).astype(np.float32)

            if node_features.shape != (8, 6):
                raise ValueError(
                    f"Expected node_features shape (8, 6), got {node_features.shape} "
                    f"for patient {patient.patient_id} window {event.window_idx}"
                )
            if adj_matrix.shape != (8, 8):
                raise ValueError(
                    f"Expected adj_matrix shape (8, 8), got {adj_matrix.shape} "
                    f"for patient {patient.patient_id} window {event.window_idx}"
                )

            samples.append({
                "patient_id":    patient.patient_id,
                "window_idx":    int(event.window_idx),
                "node_features": node_features,
                "adj_matrix":    adj_matrix,
                "label":         int(event.label),
            })

        return samples

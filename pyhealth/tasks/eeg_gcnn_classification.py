import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .base_task import BaseTask

logger = logging.getLogger(__name__)

# Band name → column index in the (8, 6) node feature matrix.
BAND_NAMES: List[str] = ["delta", "theta", "alpha", "beta", "low_gamma", "high_gamma"]


class EEGGCNNClassification(BaseTask):
    """Per-window neurological disease classification from EEG graphs.

    Based on the EEG-GCNN pipeline introduced in:
        Wagh, N., & Varatharajah, Y. (2020). EEG-GCNN: Augmenting
        Electroencephalogram-based Neurological Disease Diagnosis using a
        Domain-guided Graph Convolutional Neural Network. *Machine Learning
        for Health (ML4H) workshop, NeurIPS 2020*.
        https://arxiv.org/abs/2011.10432

    Classifies each EEG window as diseased or healthy using pre-computed PSD
    node features and a combined spectral-coherence / geodesic-distance
    adjacency matrix produced by :class:`~pyhealth.datasets.EEGGCNNDataset`.

    At load time this task reads per-window ``.npy`` files written by
    :meth:`~pyhealth.datasets.EEGGCNNDataset.prepare_metadata` and returns
    one sample dict per window. Frequency-band ablations can be performed
    without rebuilding the dataset cache by passing ``excluded_bands``, which
    zeros out the specified columns of the node-feature matrix on the fly.

    Attributes:
        task_name (str): Unique identifier for this task, used by PyHealth
            internals. Set to ``"EEGGCNNClassification"``.
        input_schema (Dict[str, str]): Maps input keys to their PyHealth type.
            ``node_features`` and ``adj_matrix`` are both ``"tensor"``.
        output_schema (Dict[str, str]): Maps output keys to their PyHealth
            type. ``label`` is ``"binary"``.
        excluded_band_indices (List[int]): Column indices of bands to zero out,
            derived from ``excluded_bands`` at construction time.

    Args:
        excluded_bands (List[str], optional): Frequency band names to zero out
            in the node-feature matrix at load time, for ablation studies.
            Valid values are elements of ``BAND_NAMES``
            (``["delta", "theta", "alpha", "beta", "low_gamma", "high_gamma"]``).
            Defaults to ``[]`` (all bands active).

    Raises:
        ValueError: If any entry in ``excluded_bands`` is not in ``BAND_NAMES``.

    Examples:
        Baseline — all frequency bands active:

        >>> from pyhealth.datasets import EEGGCNNDataset
        >>> from pyhealth.tasks import EEGGCNNClassification
        >>> dataset = EEGGCNNDataset(root="/path/to/eeg-gcnn")
        >>> samples = dataset.set_task(EEGGCNNClassification())

        Leave-one-out frequency-band ablation:

        >>> from pyhealth.tasks.eeg_gcnn_classification import BAND_NAMES
        >>> for band in BAND_NAMES:
        ...     samples = dataset.set_task(
        ...         EEGGCNNClassification(excluded_bands=[band])
        ...     )
        ...     # train and evaluate — performance drop vs. baseline
        ...     # indicates importance of that band

        Keep-one-in ablation (all bands except one zeroed):

        >>> for band in BAND_NAMES:
        ...     others = [b for b in BAND_NAMES if b != band]
        ...     samples = dataset.set_task(
        ...         EEGGCNNClassification(excluded_bands=others)
        ...     )
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
        """Initialise the task, optionally configuring band exclusions.

        Args:
            excluded_bands (List[str], optional): Names of frequency bands to
                zero out in node features during sample loading. Must be a
                subset of ``BAND_NAMES``. Defaults to ``[]``.

        Raises:
            ValueError: If any entry in ``excluded_bands`` is not found in
                ``BAND_NAMES``.
        """
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

        Iterates over all ``eeg_windows`` events for the given patient, loads
        the pre-saved node-feature and adjacency-matrix arrays from disk,
        optionally zeros out excluded frequency bands, validates array shapes,
        and returns a list of sample dicts ready for model consumption.

        Args:
            patient (Any): A PyHealth ``Patient`` object whose ``eeg_windows``
                events each carry ``node_features_path``, ``adj_matrix_path``,
                ``window_idx``, and ``label`` attributes, as written by
                :meth:`~pyhealth.datasets.EEGGCNNDataset.prepare_metadata`.

        Returns:
            List[Dict[str, Any]]: One dict per EEG window, each containing:

                - ``patient_id`` (str): Patient identifier.
                - ``window_idx`` (int): Zero-based index of this window within
                  the patient's recording.
                - ``node_features`` (np.ndarray): Shape ``(8, 6)``, float32.
                  PSD band-power values per electrode node, with any excluded
                  bands zeroed out.
                - ``adj_matrix`` (np.ndarray): Shape ``(8, 8)``, float32.
                  Combined geodesic / coherence edge weights.
                - ``label`` (int): Binary class label — ``0`` = diseased,
                  ``1`` = healthy.

            Returns an empty list if the patient has no ``eeg_windows`` events.

        Raises:
            ValueError: If a loaded ``node_features`` array does not have shape
                ``(8, 6)`` or a loaded ``adj_matrix`` does not have shape
                ``(8, 8)``.
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

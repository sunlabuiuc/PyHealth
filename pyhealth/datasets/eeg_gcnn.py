import logging
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import load
from sklearn import preprocessing

from ..tasks.eeg_gcnn_classification import EEGGCNNClassification
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


# Bipolar montage channel names — node order matches the pre-computed arrays.
_CH_NAMES: List[str] = [
    "F7-F3", "F8-F4", "T7-C3", "T8-C4", "P7-P3", "P8-P4", "O1-P3", "O2-P4"
]

# Corresponding 10-10 reference electrodes used for geodesic distance calc.
_REF_NAMES: List[str] = ["F5", "F6", "C5", "C6", "P5", "P6", "O1", "O2"]

_NUM_NODES: int = len(_CH_NAMES)        # 8 nodes = 8 bipolar channels in the montage
_NUM_NODE_FEATURES: int = 6             # PSD bands per channel
_NUM_EDGES: int = _NUM_NODES ** 2       # 64 fully connected edges

# Required pre-computed raw files in the dataset root for metadata preparation.
# Downloaded from the Figshare repository linked in the paper.
_REQUIRED_FILES: List[str] = [
    "psd_features_data_X",
    "labels_y",
    "master_metadata_index.csv",
    "spec_coh_values.npy",
    "standard_1010.tsv.txt",
]

_METADATA_CSV_TEMPLATE: str = "eeg_gcnn_windows_alpha{alpha:.2f}.csv"
_NPY_CACHE_DIR_TEMPLATE: str = "npy_cache_alpha{alpha:.2f}"


class EEGGCNNDataset(BaseDataset):
    """EEG neurological disease dataset built on EEG-GCNN pre-computed features.

    Based on the EEG-GCNN pipeline introduced in:
        Wagh, N., & Varatharajah, Y. (2020). EEG-GCNN: Augmenting
        Electroencephalogram-based Neurological Disease Diagnosis using a
        Domain-guided Graph Convolutional Neural Network. *Machine Learning
        for Health (ML4H) workshop, NeurIPS 2020*.
        https://arxiv.org/abs/2011.10432

    This dataset wraps the pre-processed outputs of the EEG-GCNN pipeline
    and exposes them as a PyHealth-compatible graph dataset suitable for
    GNN-based neurological disease classification.

    Each EEG recording is segmented into fixed-length windows. Every window is
    represented as a fully-connected graph whose **nodes** are bipolar EEG
    electrode pairs and whose **edges** encode both spatial and functional brain
    connectivity.

    Graph construction:
        Node features (shape ``(8, 6)``):
            Six PSD band-power values (delta, theta, alpha, beta, low-gamma,
            high-gamma) computed per bipolar channel and L2-normalised across
            the full 48-dimensional feature vector for each window.

        Edge weights (shape ``(8, 8)``):
            A weighted average of two complementary connectivity measures, both
            normalised to ``[0, 1]``::

                edge_weight = alpha * geodesic_distance
                            + (1 - alpha) * spectral_coherence

            - **Geodesic distance**: the arc length between electrode pairs when
              the standard 10-20 layout is projected onto a unit sphere —
              a proxy for *spatial* brain connectivity.
            - **Spectral coherence**: the average cross-spectral coherence
              between the two electrodes' time-series — a proxy for *functional*
              brain connectivity.
            - **alpha** (default ``0.5``) controls the balance between the two
              components and can be varied for ablation studies.

    On first instantiation :meth:`prepare_metadata` is called automatically.
    It reads the raw arrays, builds per-window ``.npy`` files under
    ``<root>/npy_cache_alpha<alpha>/``, and writes a flat metadata CSV that
    BaseDataset ingests via the bundled YAML config.

    Required files in ``root``:
        - ``psd_features_data_X``: joblib array, shape ``(N, 48)``
        - ``labels_y``: joblib array, shape ``(N,)``
        - ``master_metadata_index.csv``: CSV with a ``patient_ID`` column
        - ``spec_coh_values.npy``: numpy array, shape ``(N, 64)``
        - ``standard_1010.tsv.txt``: TSV with columns ``label``, ``x``, ``y``, ``z``

    Args:
        root (str): Root directory containing all required dataset files.
        dataset_name (str, optional): Name override for this dataset instance.
            Defaults to ``"eeg_gcnn"``.
        config_path (str, optional): Path to a custom YAML config. Defaults to
            the bundled ``configs/eeg_gcnn.yaml``.
        cache_dir (str, optional): Directory for BaseDataset's parquet cache.
        num_workers (int): Number of parallel workers for BaseDataset
            processing. Defaults to ``1``.
        dev (bool): When ``True`` only the first 1000 patients are loaded
            (BaseDataset dev mode). Defaults to ``False``.
        alpha (float): Mixing weight ``[0, 1]`` for the edge-weight formula
            ``alpha * geodesic + (1 - alpha) * coherence``. Use ``1.0`` for
            geodesic-only, ``0.0`` for coherence-only, ``0.5`` (default) for
            the equal average from the original paper.

    Raises:
        ValueError: If ``alpha`` is outside ``[0, 1]``.

    Examples:
        Basic usage with default settings (equal geodesic/coherence blend):

        >>> from pyhealth.datasets import EEGGCNNDataset
        >>> dataset = EEGGCNNDataset(root="/path/to/eeg-gcnn")
        >>> dataset.stats()
        >>> samples = dataset.set_task()
        >>> print(samples[0].keys())
        dict_keys(['patient_id', 'window_idx', 'node_features', 'adj_matrix', 'label'])

        Ablation study varying the edge-weight composition:

        >>> for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ...     dataset = EEGGCNNDataset(root="/path/to/eeg-gcnn", alpha=alpha)
        ...     samples = dataset.set_task()
        ...     # train and evaluate model, record metrics per alpha

        Frequency-band ablation using the task's ``excluded_bands`` parameter
        (no dataset rebuild required — zeroing happens at load time):

        >>> from pyhealth.tasks.eeg_gcnn_classification import (
        ...     EEGGCNNClassification, BAND_NAMES
        ... )
        >>> dataset = EEGGCNNDataset(root="/path/to/eeg-gcnn")
        >>> for band in BAND_NAMES:
        ...     samples = dataset.set_task(
        ...         EEGGCNNClassification(excluded_bands=[band])
        ...     )
        ...     # train and evaluate — drop vs. baseline = importance of that band

    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        num_workers: int = 1,
        dev: bool = False,
        alpha: float = 0.5,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        if config_path is None:
            logger.info("No config_path provided, using default eeg_gcnn config.")
            config_path = str(Path(__file__).parent / "configs" / "eeg_gcnn.yaml")

        metadata_csv = os.path.join(root, _METADATA_CSV_TEMPLATE.format(alpha=alpha))
        if not os.path.exists(metadata_csv):
            logger.info(
                "Metadata CSV not found for alpha=%.2f. Running prepare_metadata().",
                alpha,
            )
            self.prepare_metadata(root, alpha=alpha)

        # Copy the alpha-specific CSV to the name BaseDataset expects.
        base_csv = os.path.join(root, "eeg_gcnn_windows.csv")
        shutil.copy2(metadata_csv, base_csv)
        logger.info("Copied %s -> %s", metadata_csv, base_csv)

        super().__init__(
            root=root,
            tables=["eeg_windows"],
            dataset_name=dataset_name or "eeg_gcnn",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    # ------------------------------------------------------------------
    # Metadata preparation (run once per alpha, result cached as CSV)
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_metadata(root: str, alpha: float = 0.5) -> None:
        """Convert raw EEG-GCNN binary files into a per-window metadata CSV.

        This method is called automatically by :meth:`__init__` the first time
        a given ``alpha`` value is used. It performs the following steps:

        1. Loads and validates ``psd_features_data_X`` and ``labels_y``.
        2. L2-normalises each window's 48-dimensional PSD feature vector.
        3. Maps string class labels to integer indices (alphabetical order).
        4. Computes pairwise geodesic distances for the 8 electrode nodes and
           normalises them to ``[0, 1]``.
        5. For each window, blends geodesic distances and spectral coherence
           values into a single ``(8, 8)`` adjacency matrix.
        6. Saves per-window ``node_features`` and ``adj_matrix`` as ``.npy``
           files under ``<root>/npy_cache_alpha<alpha>/``.
        7. Writes ``<root>/eeg_gcnn_windows_alpha<alpha>.csv`` with one row per
           window containing paths to those ``.npy`` files.

        Args:
            root (str): Root directory containing the raw dataset files.
                Must contain all entries listed in ``_REQUIRED_FILES``.
            alpha (float): Mixing weight for the edge-weight formula
                ``alpha * geodesic + (1 - alpha) * coherence``. Defaults to
                ``0.5`` (equal average). Results are cached under a filename
                that encodes the ``alpha`` value, so different values can
                coexist on disk without conflict.

        Raises:
            FileNotFoundError: If any file listed in ``_REQUIRED_FILES`` is
                absent from ``root``.
            ValueError: If ``psd_features_data_X`` does not have shape
                ``(N, 48)``, if ``labels_y`` length does not match, if
                ``master_metadata_index.csv`` row count does not match, or if
                ``spec_coh_values.npy`` does not have shape ``(N, 64)``.
        """
        root = os.path.abspath(root)
        EEGGCNNDataset._validate_root(root)

        # --- load raw feature arrays --------------------------------------
        logger.info("Loading psd_features_data_X...")
        X_raw: np.ndarray = load(os.path.join(root, "psd_features_data_X"))
        logger.info("Loading labels_y...")
        y_raw: np.ndarray = load(os.path.join(root, "labels_y"))

        expected_features = _NUM_NODES * _NUM_NODE_FEATURES
        if X_raw.ndim != 2 or X_raw.shape[1] != expected_features:
            raise ValueError(
                f"psd_features_data_X must have shape (N, {expected_features}), "
                f"got {X_raw.shape}"
            )
        if y_raw.shape[0] != X_raw.shape[0]:
            raise ValueError(
                f"labels_y length {y_raw.shape[0]} does not match "
                f"psd_features_data_X length {X_raw.shape[0]}"
            )
        n_windows: int = len(y_raw)

        # L2-normalise each window feature vector
        logger.info("Normalising feature vectors...")
        X: np.ndarray = np.vstack([
            preprocessing.normalize(X_raw[i].reshape(1, -1))
            for i in range(n_windows)
        ]).reshape(n_windows, _NUM_NODES * _NUM_NODE_FEATURES)

        # Map string labels to integer indices (alphabetical order)
        label_classes: np.ndarray
        y: np.ndarray
        label_classes, y = np.unique(y_raw, return_inverse=True)
        logger.info("Label mapping: %s", dict(enumerate(label_classes)))

        # --- load metadata index ------------------------------------------
        metadata_path = os.path.join(root, "master_metadata_index.csv")
        metadata: pd.DataFrame = pd.read_csv(
            metadata_path, dtype={"patient_ID": str}
        )
        if "patient_ID" not in metadata.columns:
            raise ValueError(
                "master_metadata_index.csv must contain a 'patient_ID' column."
            )
        if len(metadata) != n_windows:
            raise ValueError(
                f"master_metadata_index.csv has {len(metadata)} rows but "
                f"labels_y has {n_windows} entries."
            )

        # --- pre-compute geodesic electrode distances (shared across windows)
        logger.info("Computing geodesic electrode distances...")
        coords_path = os.path.join(root, "standard_1010.tsv.txt")
        coords: pd.DataFrame = pd.read_csv(coords_path, sep="\t")
        distances_flat: np.ndarray = EEGGCNNDataset._compute_distances(coords)
        distances_flat = EEGGCNNDataset._normalise_01(distances_flat)  # shape (64,)

        # --- load spectral coherence values (per-window) ------------------
        logger.info("Loading spec_coh_values.npy...")
        spec_coh: np.ndarray = np.load(
            os.path.join(root, "spec_coh_values.npy"), allow_pickle=True
        )
        if spec_coh.shape != (n_windows, _NUM_EDGES):
            raise ValueError(
                f"spec_coh_values.npy must have shape ({n_windows}, {_NUM_EDGES}), "
                f"got {spec_coh.shape}"
            )

        # --- save per-window npy files ------------------------------------
        npy_dir = os.path.join(root, _NPY_CACHE_DIR_TEMPLATE.format(alpha=alpha))
        os.makedirs(npy_dir, exist_ok=True)
        logger.info(
            "Saving per-window .npy files to %s (alpha=%.2f)...", npy_dir, alpha
        )

        rows: List[Dict] = []
        for idx in range(n_windows):
            patient_id: str = metadata["patient_ID"].iloc[idx]

            node_features: np.ndarray = (
                X[idx].reshape(_NUM_NODES, _NUM_NODE_FEATURES).astype(np.float64)
            )
            edge_weights: np.ndarray = (
                alpha * distances_flat + (1 - alpha) * spec_coh[idx]  # shape (64,)
            )
            adj_matrix: np.ndarray = (
                EEGGCNNDataset._build_adj_matrix(edge_weights).astype(np.float64)
            )

            nf_path = os.path.join(npy_dir, f"{idx}_node_features.npy")
            am_path = os.path.join(npy_dir, f"{idx}_adj_matrix.npy")
            np.save(nf_path, node_features)
            np.save(am_path, adj_matrix)

            rows.append({
                "patient_id":         patient_id,
                "window_idx":         idx,
                "label":              int(y[idx]),
                "node_features_path": nf_path,
                "adj_matrix_path":    am_path,
            })

        # --- write metadata CSV -------------------------------------------
        out_csv = os.path.join(root, _METADATA_CSV_TEMPLATE.format(alpha=alpha))
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        logger.info(
            "Wrote %d window rows to %s (alpha=%.2f)",
            len(df),
            out_csv,
            alpha,
        )

    # ------------------------------------------------------------------
    # Default task
    # ------------------------------------------------------------------

    @property
    def default_task(self) -> EEGGCNNClassification:
        """Returns the default classification task for this dataset.

        Returns:
            EEGGCNNClassification: Task instance with all frequency bands active
                and no custom configuration. Pass an explicit task instance to
                :meth:`set_task` to use ablation settings.
        """
        return EEGGCNNClassification()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_root(root: str) -> None:
        """Assert that all required source files are present in ``root``.

        Args:
            root (str): Absolute path to the dataset root directory.

        Raises:
            FileNotFoundError: If one or more entries from ``_REQUIRED_FILES``
                are absent from ``root``.
        """
        missing: List[str] = [
            f for f in _REQUIRED_FILES
            if not os.path.exists(os.path.join(root, f))
        ]
        if missing:
            raise FileNotFoundError(
                f"The following required files are missing from root='{root}':\n"
                + "\n".join(f"  - {f}" for f in missing)
            )

    @staticmethod
    def _compute_distances(coords: pd.DataFrame) -> np.ndarray:
        """Compute pairwise geodesic distances for all electrode node pairs.

        Iterates over every ordered pair ``(src, dst)`` in the 8-node set and
        delegates to :meth:`_geodesic_distance`. The result is a flat array
        whose layout matches the row-major unrolling of an ``(8, 8)`` matrix.

        Args:
            coords (pd.DataFrame): Electrode coordinate table loaded from
                ``standard_1010.tsv.txt``, with columns ``label``, ``x``,
                ``y``, ``z``.

        Returns:
            np.ndarray: Float64 array of shape ``(64,)`` containing the raw
                (un-normalised) geodesic distance for each ordered node pair.
        """
        distances: List[float] = []
        for src in range(_NUM_NODES):
            for dst in range(_NUM_NODES):
                distances.append(
                    EEGGCNNDataset._geodesic_distance(src, dst, coords)
                )
        return np.array(distances, dtype=np.float64)

    @staticmethod
    def _geodesic_distance(
        src_idx: int, dst_idx: int, coords: pd.DataFrame
    ) -> float:
        """Compute the geodesic (great-circle) distance between two electrodes.

        Looks up the Cartesian coordinates of both electrodes in the standard
        10-10 layout, computes their dot product on the unit sphere, and
        returns the arc-cosine (i.e. the central angle in radians), which
        equals the geodesic distance on a unit sphere.

        Args:
            src_idx (int): Index into ``_REF_NAMES`` for the source electrode.
            dst_idx (int): Index into ``_REF_NAMES`` for the destination
                electrode.
            coords (pd.DataFrame): Electrode coordinate table with columns
                ``label``, ``x``, ``y``, ``z``.

        Returns:
            float: Geodesic distance in radians, in the range ``[0, π]``.

        Raises:
            ValueError: If either electrode label is not found in ``coords``.
        """
        def _xyz(name: str) -> Tuple[float, float, float]:
            row = coords[coords.label == name]
            if row.empty:
                raise ValueError(
                    f"Electrode '{name}' not found in standard_1010.tsv.txt"
                )
            return (
                float(row["x"].iloc[0]),
                float(row["y"].iloc[0]),
                float(row["z"].iloc[0]),
            )

        x1, y1, z1 = _xyz(_REF_NAMES[src_idx])
        x2, y2, z2 = _xyz(_REF_NAMES[dst_idx])
        dot = round(x1 * x2 + y1 * y2 + z1 * z2, 2)
        # Clamp to [-1, 1] for numerical safety before acos.
        dot = max(-1.0, min(1.0, dot))
        return math.acos(dot)

    @staticmethod
    def _normalise_01(arr: np.ndarray) -> np.ndarray:
        """Apply min-max normalisation to scale an array into ``[0, 1]``.

        Args:
            arr (np.ndarray): Input array of any shape and dtype.

        Returns:
            np.ndarray: Array of the same shape with values in ``[0, 1]``.
                Returns an all-zeros array of the same shape when all values
                in ``arr`` are identical (zero range), avoiding division by
                zero.
        """
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    @staticmethod
    def _build_adj_matrix(edge_weights: np.ndarray) -> np.ndarray:
        """Reshape a flat edge-weight vector into a square adjacency matrix.

        Args:
            edge_weights (np.ndarray): Float array of shape ``(64,)``
                containing one weight per ordered node pair in row-major order.

        Returns:
            np.ndarray: Float32 array of shape ``(8, 8)`` where entry
                ``[i, j]`` is the edge weight from node ``i`` to node ``j``.
        """
        return edge_weights.reshape(_NUM_NODES, _NUM_NODES).astype(np.float32)

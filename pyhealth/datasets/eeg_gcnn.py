import logging
import math
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from joblib import load
from sklearn import preprocessing

from ..tasks.eeg_gcnn_classification import EEGGCNNClassification
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (must stay consistent with the pre-computed feature arrays)
# ---------------------------------------------------------------------------

# Bipolar montage channel names — node order matches the pre-computed arrays.
_CH_NAMES = ["F7-F3", "F8-F4", "T7-C3", "T8-C4", "P7-P3", "P8-P4", "O1-P3", "O2-P4"]

# Corresponding 10-10 reference electrodes used for geodesic distance calc.
_REF_NAMES = ["F5", "F6", "C5", "C6", "P5", "P6", "O1", "O2"]

_NUM_NODES = len(_CH_NAMES)       # 8
_NUM_NODE_FEATURES = 6            # PSD bands per channel
_NUM_EDGES = _NUM_NODES ** 2      # 64 (fully connected, including self-loops)

# PSD frequency band names — index matches column position within each node's feature vector.
_BAND_NAMES: List[str] = ["delta", "theta", "alpha", "beta", "low_gamma", "high_gamma"]

_REQUIRED_FILES = [
    "psd_features_data_X",
    "labels_y",
    "master_metadata_index.csv",
    "spec_coh_values.npy",
    "standard_1010.tsv.txt",
]

_METADATA_CSV_TEMPLATE = "eeg_gcnn_windows_alpha{alpha:.2f}_excl{bands}.csv"
_NPY_CACHE_DIR_TEMPLATE = "npy_cache_alpha{alpha:.2f}_excl{bands}"


class EEGGCNNDataset(BaseDataset):
    """EEG neurological disease dataset wrapping EEG-GCNN pre-computed features.

    Converts joblib / NumPy binary arrays from the EEG-GCNN pipeline into a
    flat per-window CSV that BaseDataset can ingest, following the same pattern
    as :class:`~pyhealth.datasets.COVID19CXRDataset`.

    On first instantiation :meth:`prepare_metadata` is called automatically.
    It reads the raw arrays, computes geodesic electrode distances, builds
    per-window adjacency matrices, saves each window's node-feature and
    adjacency arrays as individual ``.npy`` files under ``<root>/npy_cache/``,
    and writes a single ``<root>/eeg_gcnn_windows.csv`` that BaseDataset reads
    through the YAML config.

    Dataset files expected in ``root``:
        - ``psd_features_data_X``         joblib array, shape (N, 48)
        - ``labels_y``                     joblib array, shape (N,)
        - ``master_metadata_index.csv``    CSV with a ``patient_ID`` column
        - ``spec_coh_values.npy``          numpy array, shape (N, 64)
        - ``standard_1010.tsv.txt``        TSV with columns label, x, y, z

    Args:
        root: Root directory containing all dataset files.
        dataset_name: Optional name override. Defaults to ``"eeg_gcnn"``.
        config_path: Optional path to a custom YAML config. Defaults to the
            bundled ``configs/eeg_gcnn.yaml``.
        cache_dir: Optional cache directory for BaseDataset's parquet cache.
        num_workers: Number of parallel workers for BaseDataset processing.
        dev: When ``True`` only the first 1000 patients are loaded (BaseDataset
            dev mode).
        alpha: Mixing weight for geodesic distance vs. coherence in edge weights.
            Default ``0.5``.
        excluded_bands: Frequency bands to zero out for ablation studies. Valid
            values are elements of ``_BAND_NAMES``. Default ``[]`` (all active).

    Examples:
        >>> from pyhealth.datasets import EEGGCNNDataset
        >>> dataset = EEGGCNNDataset(root="/path/to/eeg-gcnn")
        >>> dataset.stats()
        >>> samples = dataset.set_task()
        >>> print(samples[0])
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
        excluded_bands: Optional[List[str]] = None,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        excluded_bands = excluded_bands or []
        invalid = [b for b in excluded_bands if b not in _BAND_NAMES]
        if invalid:
            raise ValueError(
                f"Unknown band(s) {invalid}. Valid options: {_BAND_NAMES}"
            )

        if config_path is None:
            logger.info("No config_path provided, using default eeg_gcnn config.")
            config_path = str(Path(__file__).parent / "configs" / "eeg_gcnn.yaml")

        bands_tag = "_".join(sorted(excluded_bands)) if excluded_bands else "none"
        metadata_csv = os.path.join(
            root, _METADATA_CSV_TEMPLATE.format(alpha=alpha, bands=bands_tag)
        )
        if not os.path.exists(metadata_csv):
            logger.info(
                "Metadata CSV not found for alpha=%.2f, excluded_bands=%s. "
                "Running prepare_metadata().",
                alpha,
                excluded_bands,
            )
            self.prepare_metadata(root, alpha=alpha, excluded_bands=excluded_bands)

        # Copy the ablation-specific CSV to the name BaseDataset expects.
        import shutil as _shutil
        base_csv = os.path.join(root, "eeg_gcnn_windows.csv")
        _shutil.copy2(metadata_csv, base_csv)
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
    # Metadata preparation (run once, result cached as CSV)
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_metadata(
        root: str,
        alpha: float = 0.5,
        excluded_bands: Optional[List[str]] = None,
    ) -> None:
        """Convert raw EEG-GCNN binary files into a per-window CSV.

        Reads joblib / NumPy arrays, computes the graph structure for every
        window, saves per-window node-feature and adjacency matrices as ``.npy``
        files, and writes a metadata CSV named after the ``alpha`` and
        ``excluded_bands`` values.

        Args:
            root: Root directory containing the raw dataset files.
            alpha: Mixing weight for geodesic distance vs. coherence.
                Edge weights are computed as
                ``alpha * geodesic + (1 - alpha) * coherence``.
                ``alpha=0.5`` (default) reproduces the original equal average;
                ``alpha=1.0`` uses geodesic distance only;
                ``alpha=0.0`` uses coherence only.
            excluded_bands: Frequency bands to zero out in node features. Valid
                values are elements of ``_BAND_NAMES``. Default ``[]``.

        Raises:
            FileNotFoundError: If any required source file is missing.
            ValueError: If array shapes are inconsistent.
        """
        excluded_bands = excluded_bands or []
        root = os.path.abspath(root)
        EEGGCNNDataset._validate_root(root)

        # --- load raw feature arrays --------------------------------------
        logger.info("Loading psd_features_data_X...")
        X_raw = load(os.path.join(root, "psd_features_data_X"))
        logger.info("Loading labels_y...")
        y_raw = load(os.path.join(root, "labels_y"))

        if X_raw.ndim != 2 or X_raw.shape[1] != _NUM_NODES * _NUM_NODE_FEATURES:
            raise ValueError(
                f"psd_features_data_X must have shape (N, {_NUM_NODES * _NUM_NODE_FEATURES}), "
                f"got {X_raw.shape}"
            )
        if y_raw.shape[0] != X_raw.shape[0]:
            raise ValueError(
                f"labels_y length {y_raw.shape[0]} does not match "
                f"psd_features_data_X length {X_raw.shape[0]}"
            )
        n_windows = len(y_raw)

        # L2-normalise each window feature vector
        logger.info("Normalising feature vectors...")
        X = np.vstack([
            preprocessing.normalize(X_raw[i].reshape(1, -1))
            for i in range(n_windows)
        ]).reshape(n_windows, _NUM_NODES * _NUM_NODE_FEATURES)

        # Map string labels to integer indices (alphabetical order)
        label_classes, y = np.unique(y_raw, return_inverse=True)
        logger.info("Label mapping: %s", dict(enumerate(label_classes)))

        # --- load metadata index ------------------------------------------
        metadata_path = os.path.join(root, "master_metadata_index.csv")
        metadata = pd.read_csv(metadata_path, dtype={"patient_ID": str})
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
        coords = pd.read_csv(coords_path, sep="\t")
        distances_flat = EEGGCNNDataset._compute_distances(coords)
        distances_flat = EEGGCNNDataset._normalise_01(distances_flat)  # shape (64,)

        # --- load spectral coherence values (per-window) ------------------
        logger.info("Loading spec_coh_values.npy...")
        spec_coh = np.load(os.path.join(root, "spec_coh_values.npy"), allow_pickle=True)
        if spec_coh.shape != (n_windows, _NUM_EDGES):
            raise ValueError(
                f"spec_coh_values.npy must have shape ({n_windows}, {_NUM_EDGES}), "
                f"got {spec_coh.shape}"
            )

        # --- save per-window npy files ------------------------------------
        bands_tag = "_".join(sorted(excluded_bands)) if excluded_bands else "none"
        npy_dir = os.path.join(
            root, _NPY_CACHE_DIR_TEMPLATE.format(alpha=alpha, bands=bands_tag)
        )
        os.makedirs(npy_dir, exist_ok=True)
        logger.info(
            "Saving per-window .npy files to %s (alpha=%.2f, excluded_bands=%s)...",
            npy_dir, alpha, excluded_bands,
        )

        # Pre-compute which band column indices to zero out (shared across windows).
        excluded_indices = [_BAND_NAMES.index(b) for b in excluded_bands]

        rows = []
        for idx in range(n_windows):
            patient_id = metadata["patient_ID"].iloc[idx]

            node_features = X[idx].reshape(_NUM_NODES, _NUM_NODE_FEATURES).astype(np.float64)
            if excluded_indices:
                node_features[:, excluded_indices] = 0.0
            edge_weights = alpha * distances_flat + (1 - alpha) * spec_coh[idx]  # (64,)
            adj_matrix = EEGGCNNDataset._build_adj_matrix(edge_weights).astype(np.float64)

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
        out_csv = os.path.join(
            root, _METADATA_CSV_TEMPLATE.format(alpha=alpha, bands=bands_tag)
        )
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        logger.info(
            "Wrote %d window rows to %s (alpha=%.2f, excluded_bands=%s)",
            len(df),
            out_csv,
            alpha,
            excluded_bands,
        )

    # ------------------------------------------------------------------
    # Default task
    # ------------------------------------------------------------------

    @property
    def default_task(self) -> EEGGCNNClassification:
        """Returns the default classification task for this dataset."""
        return EEGGCNNClassification()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_root(root: str) -> None:
        missing = [
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
        """Return flat (64,) array of geodesic distances for all node pairs."""
        distances = []
        for src in range(_NUM_NODES):
            for dst in range(_NUM_NODES):
                distances.append(
                    EEGGCNNDataset._geodesic_distance(src, dst, coords)
                )
        return np.array(distances, dtype=np.float64)

    @staticmethod
    def _geodesic_distance(src_idx: int, dst_idx: int, coords: pd.DataFrame) -> float:
        """Geodesic distance between two electrodes on the unit sphere."""
        def _xyz(name: str):
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
        # clamp to [-1, 1] for numerical safety before acos
        dot = max(-1.0, min(1.0, dot))
        return math.acos(dot)

    @staticmethod
    def _normalise_01(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    @staticmethod
    def _build_adj_matrix(edge_weights: np.ndarray) -> np.ndarray:
        """Convert flat (64,) edge weight array to dense (8, 8) adjacency matrix."""
        return edge_weights.reshape(_NUM_NODES, _NUM_NODES).astype(np.float32)

"""MIMIC-III TLS Dataset adapter for PyHealth.

Contributors:
    Akshad Pai (NetID: avpai2), Matthew Ruth (NetID: mrruth2)

Paper:
    On the Importance of Step-wise Embeddings for Heterogeneous Clinical
    Time-Series (Kuznetsova et al., JMLR 2023)

Paper link:
    https://jmlr.org/papers/v24/22-0850.html

Description:
    Dataset for MIMIC-III ICU series preprocessed by the TLS pipeline: loads
    gridded CSV/HDF5 exports into PyHealth events and exposes organ/type
    feature groups for step-wise embedding models.

This module provides a dataset class for loading MIMIC-III data that has been
preprocessed by the TLS (Time-series Learning from Scratch) pipeline. The TLS
pipeline converts raw MIMIC-III data into dense, regularly-gridded ICU
time-series stored in HDF5 format.

Reference:
    Kuznetsova et al., "On the Importance of Step-wise Embeddings for
    Heterogeneous Clinical Time-Series", JMLR 2023.
"""

import logging
import os
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

import numpy as np

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

# Number of features after TLS preprocessing (17 raw -> 42 after one-hot)
NUM_FEATURES: int = 42


class MIMIC3TLSDataset(BaseDataset):
    """Dataset for MIMIC-III data preprocessed by the TLS pipeline.

    The TLS preprocessing pipeline converts raw MIMIC-III data into dense,
    regularly-gridded time-series (1-hour resolution) with 42 features per
    timestep. Features include vitals, lab values, and one-hot encoded
    categorical variables (e.g., Glasgow Coma Scale).

    This dataset adapter loads the TLS output (converted to flat CSV format)
    into PyHealth's patient-event model, where each row represents one
    timestep for one patient.

    The class also provides feature grouping metadata (``ORGAN_GROUPS`` and
    ``TYPE_GROUPS``) derived from the paper's clinical domain knowledge,
    which can be passed to the :class:`~pyhealth.models.StepwiseEmbedding`
    model for grouped feature embedding.

    Args:
        root: Path to the directory containing TLS-preprocessed CSV files.
        tables: List of table names to load. Defaults to ``["timeseries"]``.
        dataset_name: Name of the dataset. Defaults to ``"MIMIC3TLS"``.
        config_path: Path to the YAML config file. Defaults to the
            built-in ``configs/mimic3_tls.yaml``.
        **kwargs: Additional keyword arguments passed to
            :class:`~pyhealth.datasets.BaseDataset`.

    Examples:
        >>> from pyhealth.datasets import MIMIC3TLSDataset
        >>> from pyhealth.tasks import InHospitalMortalityTLS
        >>> dataset = MIMIC3TLSDataset(
        ...     root="/path/to/tls_output/",
        ...     dev=True,
        ... )
        >>> task = InHospitalMortalityTLS(observation_hours=48)
        >>> samples = dataset.set_task(task)

    Note:
        The ``root`` directory should contain a ``timeseries.csv`` file
        produced by :meth:`export_h5_to_csv`. If starting from the raw
        TLS HDF5 output, call ``export_h5_to_csv`` first to generate
        the CSV.
    """

    FEATURE_NAMES: ClassVar[List[str]] = [
        "Height",
        "Weight",
        "Diastolic blood pressure",
        "Heart Rate",
        "Glucose",
        "Mean blood pressure",
        "Systolic blood pressure",
        "Temperature",
        "Fraction inspired oxygen",
        "Oxygen saturation",
        "Respiratory rate",
        "Capillary refill rate",
        "pH",
        "Glascow coma scale eye opening_cat_0",
        "Glascow coma scale eye opening_cat_1",
        "Glascow coma scale eye opening_cat_2",
        "Glascow coma scale eye opening_cat_3",
        "Glascow coma scale motor response_cat_0",
        "Glascow coma scale motor response_cat_1",
        "Glascow coma scale motor response_cat_2",
        "Glascow coma scale motor response_cat_3",
        "Glascow coma scale motor response_cat_4",
        "Glascow coma scale motor response_cat_5",
        "Glascow coma scale total_cat_0",
        "Glascow coma scale total_cat_1",
        "Glascow coma scale total_cat_2",
        "Glascow coma scale total_cat_3",
        "Glascow coma scale total_cat_4",
        "Glascow coma scale total_cat_5",
        "Glascow coma scale total_cat_6",
        "Glascow coma scale total_cat_7",
        "Glascow coma scale total_cat_8",
        "Glascow coma scale total_cat_9",
        "Glascow coma scale total_cat_10",
        "Glascow coma scale total_cat_11",
        "Glascow coma scale total_cat_12",
        "Glascow coma scale verbal response_cat_0",
        "Glascow coma scale verbal response_cat_1",
        "Glascow coma scale verbal response_cat_2",
        "Glascow coma scale verbal response_cat_3",
        "Glascow coma scale verbal response_cat_4",
        "Capillary refill rate_cat",
    ]
    """Ordered list of the 42 feature names after TLS preprocessing."""

    ORGAN_GROUPS: ClassVar[Dict[str, List[int]]] = {
        "CNS": list(range(14, 42)),
        "circulation": [2, 5, 6, 9, 10, 13],
        "hematology": [4],
        "pulmonary": [3, 7, 8],
        "renal": [12],
        "other": [0, 1, 11],
    }
    """Feature indices grouped by organ system."""

    ORGAN_GROUPS_INDICES: ClassVar[List[List[int]]] = [
        list(range(14, 42)),
        [2, 5, 6, 9, 10, 13],
        [4],
        [3, 7, 8],
        [12],
        [0, 1, 11],
    ]
    """Feature index lists for organ-based grouping (list-of-lists)."""

    TYPE_GROUPS: ClassVar[Dict[str, List[int]]] = {
        "lab": [4, 12],
        "monitored": [2, 3, 5, 6, 7, 8, 9, 10],
        "observed": list(range(13, 42)),
        "other": [0, 1, 11],
    }
    """Feature indices grouped by variable type."""

    TYPE_GROUPS_INDICES: ClassVar[List[List[int]]] = [
        [4, 12],
        [2, 3, 5, 6, 7, 8, 9, 10],
        list(range(13, 42)),
        [0, 1, 11],
    ]
    """Feature index lists for type-based grouping (list-of-lists)."""

    def __init__(
        self,
        root: str,
        tables: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ):
        if config_path is None:
            config_path = str(
                Path(__file__).parent / "configs" / "mimic3_tls.yaml"
            )
        if tables is None:
            tables = ["timeseries"]
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "MIMIC3TLS",
            config_path=config_path,
            **kwargs,
        )

    @staticmethod
    def export_h5_to_csv(
        h5_path: str,
        output_dir: str,
        task: str = "ihm",
        splits: Optional[List[str]] = None,
    ) -> None:
        """Convert TLS HDF5 output to flat CSV for PyHealth consumption.

        Reads the TLS-preprocessed HDF5 file and writes a flat CSV file
        (``timeseries.csv``) where each row is one timestep for one patient.

        Args:
            h5_path: Path to the TLS HDF5 file (e.g.,
                ``Standard_scaled.h5``).
            output_dir: Directory to write ``timeseries.csv`` into.
            task: Task name to extract labels for. One of ``"ihm"``,
                ``"decomp_24Hours"``, or ``"los"``. Default ``"ihm"``.
            splits: List of splits to export (e.g., ``["train", "val",
                "test"]``). If ``None``, exports all available splits.

        Raises:
            ImportError: If the ``tables`` package is not installed.
            FileNotFoundError: If ``h5_path`` does not exist.
        """
        try:
            import tables as tb
        except ImportError:
            raise ImportError(
                "The 'tables' (PyTables) package is required to read "
                "HDF5 files. Install it with: pip install tables"
            )

        import pandas as pd
        from datetime import datetime, timedelta

        os.makedirs(output_dir, exist_ok=True)
        h5 = tb.open_file(h5_path, "r")

        try:
            available_splits = [
                node._v_name
                for node in h5.root.data._f_iter_nodes()
            ]
            if splits is None:
                splits = available_splits

            # Resolve task index from task name
            if hasattr(h5.root, "tasks"):
                task_names = [
                    t.decode() if isinstance(t, bytes) else t
                    for t in h5.root.tasks[:]
                ]
                task_idx = task_names.index(task)
            else:
                task_idx = 0

            feature_names = MIMIC3TLSDataset.FEATURE_NAMES
            rows = []

            for split in splits:
                data = h5.root.data._f_get_child(split)[:]
                labels = h5.root.labels._f_get_child(split)[:]
                windows = h5.root.patient_windows._f_get_child(split)[:]

                for start, stop, pid in windows:
                    start, stop, pid = int(start), int(stop), int(pid)
                    patient_data = data[start:stop]
                    patient_labels = labels[start:stop]

                    # For IHM, the label is the same for all timesteps;
                    # take the first non-NaN label
                    label_col = patient_labels[:, task_idx]
                    valid = ~np.isnan(label_col)
                    if not np.any(valid):
                        continue
                    ihm_label = int(label_col[valid][0])

                    # Base timestamp (arbitrary epoch, 1h resolution)
                    base_time = datetime(2000, 1, 1)
                    for t in range(patient_data.shape[0]):
                        row = {
                            "patient_id": str(pid),
                            "stay_id": str(pid),
                            "timestamp": (
                                base_time + timedelta(hours=t)
                            ).strftime("%Y-%m-%d %H:%M:%S"),
                            "ihm_label": ihm_label,
                        }
                        for f_idx, f_name in enumerate(feature_names):
                            row[f_name] = float(patient_data[t, f_idx])
                        rows.append(row)

            df = pd.DataFrame(rows)
            csv_path = os.path.join(output_dir, "timeseries.csv")
            df.to_csv(csv_path, index=False)
            logger.info(
                f"Exported {len(df)} rows ({len(rows)} timesteps) "
                f"to {csv_path}"
            )
        finally:
            h5.close()

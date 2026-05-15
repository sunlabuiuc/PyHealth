"""
PyHealth dataset for the PTB-XL electrocardiography dataset.

Dataset link:
    https://physionet.org/content/ptb-xl/1.0.3/

Dataset paper (please cite if you use this dataset):
    Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T.
    "PTB-XL, a large publicly available electrocardiography dataset."
    Scientific Data, 7(1), 154. https://doi.org/10.1038/s41597-020-0495-6

Dataset paper link:
    https://doi.org/10.1038/s41597-020-0495-6

Reference paper reproduced:
    Nonaka, K., & Seita, D. (2021). In-depth Benchmarking of Deep Neural
    Network Architectures for ECG Diagnosis. Proceedings of Machine Learning
    Research, 149, 414-424.
    https://proceedings.mlr.press/v149/nonaka21a.html

Author:
    Ankita Jain (ankitaj3@illinois.edu), Manish Singh (manishs4@illinois.edu)
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class PTBXLDataset(BaseDataset):
    """Dataset class for the PTB-XL electrocardiography dataset.

    PTB-XL contains 21,837 clinical 12-lead ECG recordings of 10 seconds
    duration from 18,885 patients. Each recording is annotated with SCP-ECG
    statements covering diagnostic, form, and rhythm labels.

    The dataset is publicly available on PhysioNet and does not require
    credentialed access.

    Expected directory layout after download::

        root/
        ├── ptbxl_database.csv          # Main metadata (ships with dataset)
        ├── scp_statements.csv          # SCP code definitions
        ├── records100/                 # 100 Hz recordings (.dat / .hea pairs)
        │   ├── 00000/
        │   │   ├── 00001_lr.dat
        │   │   ├── 00001_lr.hea
        │   │   └── ...
        │   └── ...
        └── records500/                 # 500 Hz recordings
            └── ...

    Attributes:
        root (str): Root directory of the raw data.
        dataset_name (str): Name of the dataset.
        sampling_rate (int): Sampling rate used (100 or 500 Hz).
        SUPERCLASSES (List[str]): The 5 superdiagnostic classes used in the
            Nonaka & Seita (2021) benchmark.

    Note:
        On first use, this class automatically generates a flattened metadata
        CSV (``ptbxl-metadata-pyhealth.csv``) from the raw ``ptbxl_database.csv``.
        Subsequent runs reuse the cached metadata file.
    """

    # Superdiagnostic classes used in the Nonaka & Seita (2021) benchmark.
    SUPERCLASSES: List[str] = ["NORM", "MI", "STTC", "CD", "HYP"]

    def __init__(
        self,
        root: str,
        sampling_rate: int = 100,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        """Initializes the PTB-XL dataset.

        Parses the raw ``ptbxl_database.csv`` shipped with the PTB-XL download,
        builds a flattened metadata CSV for PyHealth's BaseDataset, and
        initializes the dataset pipeline.

        Args:
            root (str): Root directory of the raw PTB-XL data. Must contain
                ``ptbxl_database.csv`` and the ``records100/`` or ``records500/``
                signal directories.
            sampling_rate (int): Sampling rate to use for ECG signals. Must be
                either 100 (low-resolution) or 500 (high-resolution) Hz.
                Defaults to 100.
            dataset_name (Optional[str]): Name override for the dataset.
                Defaults to ``"ptbxl"``.
            config_path (Optional[str]): Path to a custom YAML config file.
                If ``None``, uses the built-in ``configs/ptbxl.yaml``.
            cache_dir (Optional[str]): Directory for caching processed data.
                If ``None``, uses PyHealth's default cache directory.
            num_workers (int): Number of parallel workers for data processing.
                Defaults to 1.
            dev (bool): If ``True``, loads only a small subset for development
                and debugging. Defaults to ``False``.

        Raises:
            ValueError: If ``sampling_rate`` is not 100 or 500.
            FileNotFoundError: If ``ptbxl_database.csv`` is not found in
                ``root`` (raised during metadata preparation).

        Example::

            >>> from pyhealth.datasets import PTBXLDataset
            >>> from pyhealth.tasks import PTBXLDiagnosis
            >>> dataset = PTBXLDataset(root="/path/to/ptb-xl")
            >>> dataset.stats()
            >>> samples = dataset.set_task(PTBXLDiagnosis())
            >>> print(samples[0])
        """
        if sampling_rate not in (100, 500):
            raise ValueError(
                f"sampling_rate must be 100 or 500, got {sampling_rate}."
            )
        self.sampling_rate = sampling_rate

        # Use built-in config if none provided.
        if config_path is None:
            logger.info("No config path provided, using default PTB-XL config.")
            config_path = Path(__file__).parent / "configs" / "ptbxl.yaml"

        # Build metadata CSV on first run.
        metadata_csv = os.path.join(root, "ptbxl-metadata-pyhealth.csv")
        if not os.path.exists(metadata_csv):
            self.prepare_metadata(root)

        super().__init__(
            root=root,
            tables=["ptbxl"],
            dataset_name=dataset_name or "ptbxl",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def prepare_metadata(self, root: Optional[str] = None) -> None:
        """Build ``ptbxl-metadata-pyhealth.csv`` from the raw PTB-XL database.

        Reads ``ptbxl_database.csv`` (shipped with the PTB-XL download) and
        writes a flattened CSV that :class:`BaseDataset` can consume directly.
        The output CSV contains one row per ECG recording with the following
        columns:

        - ``patient_id`` (str): Unique patient identifier.
        - ``record_id`` (int): Unique ECG record identifier (``ecg_id``).
        - ``signal_file`` (str): Relative path to the WFDB signal file
          (e.g., ``records100/00000/00001_lr``).
        - ``scp_codes`` (str): String representation of the SCP diagnostic
          code dictionary (e.g., ``"{'NORM': 100.0}"``).
        - ``sampling_rate`` (int): Sampling rate in Hz (100 or 500).
        - ``num_leads`` (int): Number of ECG leads (always 12).

        Args:
            root (Optional[str]): Root directory of the raw PTB-XL data.
                Uses ``self.root`` when called after ``__init__``.

        Raises:
            FileNotFoundError: If ``ptbxl_database.csv`` is not found under
                ``root``. This typically means the PTB-XL dataset has not been
                downloaded yet.
        """
        root = root or self.root
        db_path = os.path.join(root, "ptbxl_database.csv")
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"ptbxl_database.csv not found in {root}. "
                "Please download PTB-XL from "
                "https://physionet.org/content/ptb-xl/1.0.3/"
            )

        df = pd.read_csv(db_path, index_col="ecg_id")

        # Choose the correct filename column based on sampling rate.
        # 100 Hz → filename_lr (low resolution)
        # 500 Hz → filename_hr (high resolution)
        rate_col = "filename_lr" if self.sampling_rate == 100 else "filename_hr"

        records = []
        for ecg_id, row in df.iterrows():
            patient_id = str(int(row["patient_id"]))
            signal_file = str(row[rate_col])
            scp_codes = str(row.get("scp_codes", "{}"))
            records.append(
                {
                    "patient_id": patient_id,
                    "record_id": int(ecg_id),
                    "signal_file": signal_file,
                    "scp_codes": scp_codes,
                    "sampling_rate": self.sampling_rate,
                    "num_leads": 12,
                }
            )

        out_df = pd.DataFrame(records)
        out_path = os.path.join(root, "ptbxl-metadata-pyhealth.csv")
        out_df.to_csv(out_path, index=False)
        logger.info(
            f"Wrote PTB-XL metadata to {out_path} ({len(out_df)} records)."
        )

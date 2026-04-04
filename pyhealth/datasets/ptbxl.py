"""
PTB-XL ECG Dataset for PyHealth.

Dataset paper (please cite if you use this dataset):
    Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T.
    "PTB-XL, a large publicly available electrocardiography dataset."
    Scientific Data, 7(1), 154. https://doi.org/10.1038/s41597-020-0495-6

Dataset link:
    https://physionet.org/content/ptb-xl/1.0.3/

Reference paper reproduced:
    Nonaka, K., & Seita, D. (2021). In-depth Benchmarking of Deep Neural
    Network Architectures for ECG Diagnosis. Proceedings of Machine Learning
    Research, 149, 414-424.

Author:
    Ankita Jain (ankitaj3@illinois.edu), Manish Singh (manishs4@illinois.edu)
"""

import ast
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class PTBXLDataset(BaseDataset):
    """PTB-XL: A large publicly available 12-lead ECG dataset.

    PTB-XL contains 21,837 clinical 12-lead ECG recordings of 10 seconds
    duration from 18,885 patients. Each recording is annotated with SCP-ECG
    statements covering diagnostic, form, and rhythm labels.

    Dataset is available at:
        https://physionet.org/content/ptb-xl/1.0.3/

    Expected directory layout after download::

        root/
        ├── ptbxl_database.csv
        ├── scp_statements.csv
        ├── records100/          # 100 Hz recordings (.dat / .hea pairs)
        │   ├── 00000/
        │   │   ├── 00001_lr.dat
        │   │   ├── 00001_lr.hea
        │   │   └── ...
        │   └── ...
        └── records500/          # 500 Hz recordings
            └── ...

    Args:
        root: Root directory of the raw PTB-XL data.
        sampling_rate: Sampling rate to use, either 100 or 500 Hz.
            Defaults to 100.
        dataset_name: Optional name override. Defaults to "ptbxl".
        config_path: Optional path to a custom YAML config file.
        cache_dir: Optional directory for caching processed data.
        num_workers: Number of parallel workers. Defaults to 1.
        dev: If True, loads only a small subset for development. Defaults to False.

    Attributes:
        root: Root directory of the raw data.
        dataset_name: Name of the dataset.
        sampling_rate: Sampling rate used (100 or 500 Hz).

    Examples:
        >>> from pyhealth.datasets import PTBXLDataset
        >>> from pyhealth.tasks import PTBXLDiagnosis
        >>> dataset = PTBXLDataset(root="/path/to/ptb-xl")
        >>> dataset.stats()
        >>> samples = dataset.set_task(PTBXLDiagnosis())
        >>> print(samples[0])
    """

    # Superdiagnostic classes used in the Nonaka & Seita (2021) benchmark.
    SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

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
        if sampling_rate not in (100, 500):
            raise ValueError("sampling_rate must be 100 or 500.")
        self.sampling_rate = sampling_rate

        if config_path is None:
            logger.info("No config path provided, using default PTB-XL config.")
            config_path = Path(__file__).parent / "configs" / "ptbxl.yaml"

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

        Reads ``ptbxl_database.csv`` (shipped with the dataset) and writes a
        flattened CSV that BaseDataset can consume directly.

        Args:
            root: Root directory of the raw PTB-XL data. Uses ``self.root``
                when called after ``__init__``.

        Raises:
            FileNotFoundError: If ``ptbxl_database.csv`` is not found under
                ``root``.
        """
        root = root or self.root
        db_path = os.path.join(root, "ptbxl_database.csv")
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"ptbxl_database.csv not found in {root}. "
                "Please download PTB-XL from https://physionet.org/content/ptb-xl/1.0.3/"
            )

        df = pd.read_csv(db_path, index_col="ecg_id")

        # Choose the correct filename column based on sampling rate.
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
        logger.info(f"Wrote PTB-XL metadata to {out_path} ({len(out_df)} records).")

"""IBISleepDataset: PyHealth dataset for IBI-based sleep staging.

Loads preprocessed NPZ files produced by the preprocess_dreamt_to_ibi.py,
preprocess_shhs_to_ibi.py, or preprocess_mesa_to_ibi.py scripts in examples/.
Pass the dst_dir from those scripts as the root argument here.
"""

import logging
import os
from typing import Literal, Optional

import numpy as np
import pandas as pd

from pyhealth.datasets import BaseDataset
from pyhealth.tasks.sleep_staging_ibi import SleepStagingIBI

logger = logging.getLogger(__name__)


class IBISleepDataset(BaseDataset):
    """Dataset for IBI-based sleep staging from DREAMT, SHHS, or MESA.

    Loads preprocessed NPZ files where each file contains the IBI time series,
    per-sample sleep stage labels, sampling rate, and AHI for one subject.
    Use one of the preprocessing scripts in examples/ to produce these files
    before constructing this dataset.

    Args:
        root: Directory containing ``*.npz`` files and where
            ``ibi_sleep-metadata.csv`` will be written.
        source: Dataset origin — one of ``"dreamt"``, ``"shhs"``, or
            ``"mesa"``. Affects documentation context only; loading
            behavior is identical for all three.
        dataset_name: Optional name override. Defaults to the class name.
        config_path: Path to YAML schema config. Defaults to
            ``pyhealth/datasets/configs/ibi_sleep.yaml``.
        dev: If ``True``, limits to the first 1000 patients (inherited from
            ``BaseDataset``).

    Raises:
        FileNotFoundError: If ``root`` does not exist or contains no
            readable ``.npz`` files.

    Examples:
        >>> from pyhealth.datasets import IBISleepDataset
        >>> dataset = IBISleepDataset(
        ...     root="/path/to/dreamt_npz",
        ...     source="dreamt",
        ... )
        >>> sample_ds = dataset.set_task()
        >>> sample_ds[0]
        {
            'patient_id': 'S002',
            'signal': array([...],
            dtype=float32),
            'label': 1,
            'ahi': 5.2,
        }
    """

    def __init__(
        self,
        root: str,
        source: Literal["dreamt", "shhs", "mesa"],
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        dev: bool = False,
        **kwargs,
    ) -> None:
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "ibi_sleep.yaml"
            )

        metadata_path = os.path.join(root, "ibi_sleep-metadata.csv")
        if not os.path.exists(metadata_path):
            self.prepare_metadata(root)

        self.source = source
        super().__init__(
            root=root,
            tables=["ibi_sleep"],
            dataset_name=dataset_name or "IBISleepDataset",
            config_path=config_path,
            dev=dev,
            **kwargs,
        )

    def prepare_metadata(self, root: str) -> None:
        """Scan root for NPZ files and write ibi_sleep-metadata.csv.

        Args:
            root: Directory to scan for ``*.npz`` files.

        Raises:
            FileNotFoundError: If no readable ``.npz`` files are found.
        """
        npz_paths = sorted(
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.endswith(".npz")
        )

        rows = []
        for npz_path in npz_paths:
            try:
                data = np.load(npz_path, allow_pickle=False)
            except Exception as exc:
                logger.warning("Skipping unreadable NPZ file %s: %s", npz_path, exc)
                continue

            ahi = float(data["ahi"]) if "ahi" in data else float("nan")
            patient_id = os.path.splitext(os.path.basename(npz_path))[0]
            rows.append(
                {
                    "patient_id": patient_id,
                    "npz_path": os.path.abspath(npz_path),
                    "ahi": ahi,
                }
            )

        if not rows:
            raise FileNotFoundError(
                f"No readable .npz files found in '{root}'. "
                "Run one of the preprocess_*.py scripts in examples/ first."
            )

        df = pd.DataFrame(rows, columns=["patient_id", "npz_path", "ahi"])
        df.to_csv(os.path.join(root, "ibi_sleep-metadata.csv"), index=False)
        logger.info("Wrote ibi_sleep-metadata.csv with %d subjects.", len(rows))

    @property
    def default_task(self) -> SleepStagingIBI:
        """Returns the default task for this dataset.

        Returns:
            SleepStagingIBI: Default task instance with ``num_classes=3``.
        """
        return SleepStagingIBI()

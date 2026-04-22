"""WESAD (Wearable Stress and Affect Detection) dataset for PyHealth.

This module implements the WESAD dataset for stress detection from
electrodermal activity (EDA) signals. The dataset was collected from
15 subjects wearing chest- and wrist-mounted devices during baseline,
stress, and amusement conditions.

Paper:
    Schmidt et al., "Introducing WESAD, a Multimodal Dataset for
    Wearable Stress and Affect Detection", ICMI 2018.

Dataset URL:
    https://www.eti.uni-siegen.de/ubicomp/home/datasets/icmi18/index.html.en

    Mirror (UCI ML Repository):
    https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection
"""

import logging
import os
from typing import Optional

import pandas as pd

from pyhealth.datasets import BaseDataset
from pyhealth.tasks.stress_classification_wesad import (
    StressClassificationWESAD,
)

logger = logging.getLogger(__name__)

WESAD_SUBJECT_IDS = [
    "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10",
    "S11", "S13", "S14", "S15", "S16", "S17",
]


class WESADDataset(BaseDataset):
    """Wearable Stress and Affect Detection (WESAD) dataset.

    The WESAD dataset contains physiological and motion data from
    15 subjects collected during a lab study. Data was recorded using
    both a chest-worn device (RespiBAN) and a wrist-worn device
    (Empatica E4).

    This implementation focuses on the **wrist EDA signal** sampled
    at 4 Hz, which is used for stress detection tasks.

    Each subject's data is stored as a Python pickle file containing:
        - Wrist signals: EDA, BVP, ACC, TEMP (from Empatica E4)
        - Chest signals: ECG, EDA, EMG, TEMP, ACC, Resp (from RespiBAN)
        - Labels: 0=not defined, 1=baseline, 2=stress, 3=amusement

    Dataset is available at:
        https://www.eti.uni-siegen.de/ubicomp/home/datasets/icmi18/index.html.en

        UCI mirror:
        https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection

    Args:
        root: Root directory of the extracted WESAD dataset. Should
            contain subject folders (S2/, S3/, ..., S17/).
        dataset_name: Name of the dataset. Default is ``"wesad"``.
        config_path: Path to the YAML config file. If ``None``, the
            built-in config is used.

    Attributes:
        task: Name of the task. Default is ``None``.
        samples: List of sample dicts after calling ``set_task()``.

    Examples:
        >>> from pyhealth.datasets import WESADDataset
        >>> dataset = WESADDataset(
        ...     root="/path/to/WESAD/",
        ... )
        >>> dataset.stats()
        >>> sample_dataset = dataset.set_task()
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "wesad.yaml"
            )

        metadata_path = os.path.join(root, "wesad-pyhealth.csv")
        if not os.path.exists(metadata_path):
            self.prepare_metadata(root)

        super().__init__(
            root=root,
            tables=["wrist"],
            dataset_name=dataset_name or "wesad",
            config_path=config_path,
            **kwargs,
        )

    def prepare_metadata(self, root: str) -> None:
        """Prepare metadata CSV for the WESAD dataset.

        Scans the root directory for subject folders and verifies that
        each contains a valid pickle file with wrist EDA data.

        Args:
            root: Root directory containing subject folders (S2/, ...).

        Raises:
            FileNotFoundError: If no valid subject directories are found.
        """
        records = []
        for sid in WESAD_SUBJECT_IDS:
            pkl_path = os.path.join(root, sid, f"{sid}.pkl")
            if not os.path.exists(pkl_path):
                logger.warning("Subject file not found: %s", pkl_path)
                continue
            records.append({
                "subject_id": sid,
                "signal_file": pkl_path,
                "sampling_rate": 4,
            })

        if not records:
            raise FileNotFoundError(
                f"No valid WESAD subject files found in {root}. "
                "Expected directories S2/, S3/, ..., S17/ each containing "
                "a .pkl file."
            )

        metadata = pd.DataFrame(records)
        out_path = os.path.join(root, "wesad-pyhealth.csv")
        metadata.to_csv(out_path, index=False)
        logger.info(
            "Prepared WESAD metadata with %d subjects at %s",
            len(records),
            out_path,
        )

    @property
    def default_task(self) -> StressClassificationWESAD:
        """Returns the default task for this dataset.

        Returns:
            StressClassificationWESAD: Binary stress vs baseline
                classification task.
        """
        return StressClassificationWESAD()

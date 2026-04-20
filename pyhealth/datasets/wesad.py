"""
PyHealth dataset for the WESAD (Wearable Stress and Affect Detection) dataset.

Dataset link:
    https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection

Dataset paper: (please cite if you use this dataset)
    Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K.
    "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect
    Detection." Proceedings of the 20th ACM International Conference on
    Multimodal Interaction, 2018, pp. 400-408.

Dataset paper link:
    https://dl.acm.org/doi/10.1145/3242969.3242985

Authors:
    Megan Saunders, Jennifer Miranda, Jesus Torres
    {meganas4, jm123, jesusst2}@illinois.edu
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)

# Affective state labels as defined in the WESAD protocol
WESAD_LABELS = {
    0: "not_defined",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
}

# Subjects in the dataset (S1 was discarded by original authors)
WESAD_SUBJECTS = [
    "S2", "S3", "S4", "S5", "S6",
    "S7", "S8", "S9", "S10", "S11",
    "S13", "S14", "S15", "S16", "S17",
]

# EDA sampling rate from wrist-worn Empatica E4 device (Hz)
EDA_SAMPLE_RATE = 4


class WESADDataset(BaseDataset):
    """Dataset class for the WESAD wearable stress and affect detection dataset.

    WESAD contains physiological and motion data from 15 subjects recorded
    during a lab study with three affective states: baseline, stress, and
    amusement. This class extracts the electrodermal activity (EDA) signal
    from the wrist-worn Empatica E4 device and windows it into fixed-length
    segments for downstream classification.

    Attributes:
        root (str): Root directory of the raw data.
        dataset_name (str): Name of the dataset.
        config_path (str): Path to the configuration file.
        window_size (int): Number of samples per window.
        step_size (int): Step size between consecutive windows in samples.
        label_map (Dict[int, int]): Mapping from WESAD label codes to
            task-specific integer labels.
        subjects (List[str]): List of subject IDs included in the dataset.
    """

    def __init__(
        self,
        root: str,
        window_size: int = 60,
        step_size: int = 10,
        label_map: Optional[Dict[int, int]] = None,
        subjects: Optional[List[str]] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initializes the WESAD dataset.

        Args:
            root (str): Root directory containing per-subject subdirectories,
                each with a '<subject_id>.pkl' file.
            window_size (int): Number of EDA samples per window. At 4 Hz,
                60 samples = 15 seconds. Defaults to 60.
            step_size (int): Number of samples to advance between windows.
                Defaults to 10.
            label_map (Optional[Dict[int, int]]): Mapping from raw WESAD label
                codes to output class integers. Raw codes are: 1=baseline,
                2=stress, 3=amusement. Defaults to binary stress detection:
                {1: 0, 2: 1} (baseline=0, stress=1, amusement excluded).
            subjects (Optional[List[str]]): List of subject IDs to load.
                Defaults to all 15 subjects.
            config_path (Optional[str]): Path to PyHealth config YAML.

        Raises:
            FileNotFoundError: If root directory does not exist.
            FileNotFoundError: If no subject pickle files are found in root.

        Example::
            >>> dataset = WESADDataset(root="./WESAD")
            >>> print(len(dataset.samples))
        """
        self.window_size = window_size
        self.step_size = step_size
        self.label_map = label_map or {1: 0, 2: 1}  # binary: baseline vs stress
        self.subjects = subjects or WESAD_SUBJECTS

        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "wesad.yaml")

        super().__init__(
            root=root,
            tables=["wesad"],
            dataset_name="WESAD",
            config_path=config_path,
            **kwargs,
        )

        self._verify_data(root)
        self.samples = self._load_and_window(root)

    def _verify_data(self, root: str) -> None:
        """Verifies the dataset directory structure.

        Args:
            root (str): Root directory of the raw data.

        Raises:
            FileNotFoundError: If root does not exist.
            FileNotFoundError: If no subject pickle files are found.
        """
        if not os.path.exists(root):
            msg = f"Dataset root does not exist: {root}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        pkl_files = list(Path(root).rglob("*.pkl"))
        if not pkl_files:
            msg = (
                f"No .pkl files found under {root}. "
                "Ensure WESAD is downloaded and extracted."
            )
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info(f"Found {len(pkl_files)} subject pickle files.")

    def _load_subject(
        self, root: str, subject_id: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Loads EDA signal and labels for a single subject.

        Args:
            root (str): Root directory of the raw data.
            subject_id (str): Subject identifier (e.g. 'S2').

        Returns:
            Tuple[np.ndarray, np.ndarray]: EDA signal array of shape (N,)
                and label array of shape (N,) at EDA sampling rate.

        Raises:
            FileNotFoundError: If the subject pickle file does not exist.
        """
        pkl_path = os.path.join(root, subject_id, f"{subject_id}.pkl")
        if not os.path.exists(pkl_path):
            msg = f"Subject file not found: {pkl_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        # EDA from wrist device (Empatica E4), shape (N,)
        eda = data["signal"]["wrist"]["EDA"].flatten()

        # Labels are at chest device rate (700 Hz), downsample to EDA rate (4 Hz)
        labels_chest = data["label"].flatten()
        downsample_factor = len(labels_chest) // len(eda)
        labels = labels_chest[::downsample_factor][: len(eda)]

        logger.info(f"Loaded subject {subject_id}: {len(eda)} EDA samples.")
        return eda, labels

    def _load_and_window(self, root: str) -> List[Dict]:
        """Loads all subjects and segments EDA into labeled windows.

        Windows with labels not present in self.label_map are discarded.
        A window's label is assigned by majority vote over its samples.

        Args:
            root (str): Root directory of the raw data.

        Returns:
            List[Dict]: List of sample dicts with keys:
                - 'subject_id' (str): Subject identifier.
                - 'eda' (np.ndarray): EDA window of shape (window_size,).
                - 'label' (int): Integer class label.
        """
        samples = []

        for subject_id in self.subjects:
            try:
                eda, labels = self._load_subject(root, subject_id)
            except FileNotFoundError:
                logger.warning(f"Skipping missing subject: {subject_id}")
                continue

            # Slide window over signal
            for start in range(0, len(eda) - self.window_size + 1, self.step_size):
                end = start + self.window_size
                window_eda = eda[start:end]
                window_labels = labels[start:end]

                # Majority vote for window label
                values, counts = np.unique(window_labels, return_counts=True)
                majority_label = int(values[np.argmax(counts)])

                # Skip windows with labels not in the label map
                if majority_label not in self.label_map:
                    continue

                samples.append(
                    {
                        "subject_id": subject_id,
                        "eda": window_eda.astype(np.float32),
                        "label": self.label_map[majority_label],
                    }
                )

        logger.info(f"Total windows: {len(samples)}")
        return samples

    def set_task(self, task_fn=None, **kwargs):
        """Wraps BaseDataset.set_task(); see BaseDataset for full signature.

        WESAD samples are pre-windowed in ``__init__`` and stored in
        ``self.samples``. The standard patient/visit/event pipeline from
        BaseDataset does not apply. Use ``StressDetectionDataset`` directly::

            from pyhealth.tasks.stress_detection import StressDetectionDataset
            task = StressDetectionDataset(dataset.samples)

        Args:
            task_fn: Task function applied per sample. If provided, delegates
                to BaseDataset.set_task for compatibility.
            **kwargs: Additional arguments forwarded to BaseDataset.set_task.

        Returns:
            Result of BaseDataset.set_task if task_fn is provided, else self.
        """
        if task_fn is None:
            return self
        return super().set_task(task_fn, **kwargs)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]
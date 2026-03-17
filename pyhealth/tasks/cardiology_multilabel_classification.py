"""
PyHealth task for multi-label ECG classification using the CardiologyDataset.

Dataset link:
    https://physionet.org/content/challenge-2020/1.0.2/

Dataset paper: (please cite if you use this dataset)
    Erick A Perez Alday, Annie Gu, Amit J Shah, et al. "Classification of
    12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020."
    Physiological Measurement, 41(12):124003, 2020.

Task paper: (please cite if you use this task)
    Naoki Nonaka and Jun Seita. "In-depth Benchmarking of Deep Neural Network
    Architectures for ECG Diagnosis." Proceedings of Machine Learning Research,
    vol. 149, pp. 1–17, 2021.

Task paper link:
    https://proceedings.mlr.press/v149/nonaka21a/nonaka21a.pdf

Author:
    John Ma (jm119@illinois.edu)
"""

import logging
import os
from typing import Dict, List

import numpy as np
from scipy.io import loadmat

from pyhealth.data import Event, Patient
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)


class CardiologyMultilabelClassification(BaseTask):
    """Multi-label ECG classification task for the CardiologyDataset.

    Each 2.5-second ECG window is labeled with all SNOMED-CT diagnosis codes
    present in the recording header, spanning five symptom categories:
    Arrhythmias (AR), Bundle Branch & Fascicular Blocks (BBBFB), Axis
    Deviations (AD), Conduction Delays (CD), and Wave Abnormalities (WA).
    The label space is defined by :data:`Cardiology2Dataset.classes` (24 codes).

    The task follows the multi-label benchmark protocol of Nonaka & Seita
    (2021), evaluated with macro-averaged ROC-AUC over all label classes.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for the task input.
        output_schema (Dict[str, str]): The schema for the task output.
        epoch_sec (float): Window length in seconds. Default 2.5.
        shift (float): Sliding window step in seconds. Default 1.25.

    Examples:
        >>> from pyhealth.datasets import Cardiology2Dataset
        >>> from pyhealth.tasks import CardiologyMultilabelClassification
        >>> dataset = Cardiology2Dataset(
        ...     root="/data/physionet.org/files/challenge-2020/1.0.2/training",
        ... )
        >>> task = CardiologyMultilabelClassification()
        >>> sample_dataset = dataset.set_task(task)
    """

    task_name: str = "CardiologyMultilabelClassification"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"labels": "multilabel"}

    def __init__(
        self,
        epoch_sec: float = 2.5,
        shift: float = 1.25,
    ) -> None:
        """Initializes the task.

        Args:
            epoch_sec (float): Length of each sliding window in seconds. Default 2.5.
            shift (float): Step size of the sliding window in seconds. Default 1.25.
        """
        self.epoch_sec = epoch_sec
        self.shift = shift

    def __call__(self, patient: Patient) -> List[Dict]:
        """Generates multi-label classification samples for a single patient.

        For each ECG recording event, the raw signal is loaded from the
        '.mat' file and sliced into overlapping windows of 'epoch_sec'
        seconds with a step of 'shift' seconds.

        Args:
            patient (Patient): A Patient object produced by
                :class:`~pyhealth.datasets.Cardiology2Dataset`. Each event of
                type 'cardiology' must contain 'signal_path' and 'dx'
                attributes.

        Returns:
            List[Dict]: One dict per epoch window, each containing:
                - 'patient_id' (str): patient identifier.
                - 'visit_id' (str): stem of the signal filename.
                - 'signal' (np.ndarray): epoch array of shape
                  '(12, epoch_sec * 500)'.
                - 'labels' (List[str]): SNOMED-CT codes present in this
                  recording (filtered to 'Cardiology2Dataset.classes').
        """
        from pyhealth.datasets import Cardiology2Dataset

        events: List[Event] = patient.get_events(event_type="cardiology")
        samples = []

        known_codes = set(Cardiology2Dataset.classes)
        fs = 500
        epoch_samples = int(fs * self.epoch_sec)
        shift_samples = int(fs * self.shift)

        for event in events:
            signal_path = event["signal_path"]
            dx_raw = event["dx"]

            labels: List[str] = [
                code for code in dx_raw.split(",") if code.strip() in known_codes
            ]

            try:
                X: np.ndarray = loadmat(signal_path)["val"]
            except Exception as exc:
                logger.warning(f"Failed to load {signal_path}: {exc}")
                continue

            if X.shape[1] < epoch_samples: # if the signal is too short, skip
                continue

            visit_id = os.path.splitext(os.path.basename(signal_path))[0]
            n_windows = (X.shape[1] - epoch_samples) // shift_samples + 1

            for i in range(n_windows):
                epoch = X[:, shift_samples * i : shift_samples * i + epoch_samples]

                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "visit_id": visit_id,
                        "signal": epoch,
                        "labels": labels,
                    }
                )

        return samples

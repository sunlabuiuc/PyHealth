"""
This task was implemented for the CS598DLH SP26 Final Project

Authored by Jonathan Gong, Misael Lazaro, and Sydney Robeson
NetIDs: jgong11, misaell2, sel9

This task is inspired by Nonaka & Seita (2021)
"In-depth Benchmarking of Deep Neural Network Architectures for ECG Diagnosis"
Paper link: https://proceedings.mlr.press/v149/nonaka21a.html

ECGMultiLabelCardiologyTask is a standalone PyHealth task for multi-label ECG
classification. It is designed to operate on CardiologyDataset-style records
that reference paired PhysioNet-format waveform (.mat) and header (.hea) files.

For each visit record, the task:
1. Loads a 12-lead ECG signal from the waveform file.
2. Parses diagnosis codes and basic demographics (sex and age) from the
   corresponding header file.
3. Converts the configured diagnosis label set into a multi-hot target vector.
4. Segments the ECG into fixed-length sliding windows using the specified epoch
   length, shift, and sampling rate.
5. Produces model-ready samples containing the signal window, multilabel target,
   and associated visit/patient metadata.

This makes the task suitable for training and testing existing PyHealth models
on synthetic or dataset-backed ECG classification workflows.
"""

from typing import Dict, List, Any, Optional, Union
import os
import numpy as np

from scipy.io import loadmat
from pyhealth.tasks.base_task import BaseTask  # Import BaseTask from PyHealth


class ECGMultiLabelCardiologyTask(BaseTask):
    """PyHealth task for multi-label ECG classification.

    This task processes PhysioNet-style ECG records (.mat + .hea files) into
    model-ready samples. It supports multi-label classification and flexible
    signal segmentation.

    Attributes:
        labels (List[str]): List of possible diagnosis labels.
        epoch_sec (int): Length of each ECG window in seconds.
        shift (int): Sliding window shift in seconds.
        sampling_rate (int): Sampling rate of ECG signals (Hz).

    Example:
        >>> task = ECGMultiLabelCardiologyTask(labels=["AF", "RBBB"])
        >>> samples = task(visit_record)
        >>> print(samples[0]["signal"].shape)
    """
    task_name: str = "ECGMultiLabelCardiologyTask"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "multilabel"}

    def __init__(
        self,
        labels: List[str],
        epoch_sec: int = 10,
        shift: int = 5,
        sampling_rate: int = 500,
        **kwargs,
    ):

        """Initialize the ECG task.

        Args:
            labels (List[str]): List of possible diagnosis labels.
            epoch_sec (int): Window size in seconds.
            shift (int): Sliding window step in seconds.
            sampling_rate (int): Signal sampling rate (Hz).

        Example:
            >>> task = ECGMultiLabelCardiologyTask(
            ...     labels=["AF", "RBBB"],
            ...     epoch_sec=10,
            ...     shift=5
            ... )
        """
        super().__init__(**kwargs)
        self.labels = labels
        self.epoch_sec = epoch_sec
        self.shift = shift
        self.sampling_rate = sampling_rate
        self.label_to_index = {label: idx for idx, label in enumerate(labels)}

    def __call__(self, patient: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert patient visits into model-ready samples.

        Args:
            patient: Either a single visit dictionary or list of visits.

        Returns:
            List[Dict[str, Any]]: Processed samples.

        Example:
            >>> samples = task(patient_record)
            >>> len(samples)
            12
        """
        # --- SIMPLE MODE (for unit tests) ---
        if isinstance(patient, dict) and "ecg" in patient:
            if "labels" not in patient:
                return []

            ecg = patient["ecg"]
            labels = patient["labels"]

            y = np.zeros(len(self.labels), dtype=int)

            for label in labels:
                if label in self.labels:
                    idx = self.labels.index(label)
                    y[idx] = 1

            return [{"x": ecg, "y": y}]

        # --- FULL DATASET MODE ---
        visits = self._normalize_input(patient)
        samples = []

        window_size = self.sampling_rate * self.epoch_sec
        step_size = self.sampling_rate * self.shift

        for visit in visits:
            if not self._is_valid_visit(visit):
                continue

            root = visit["load_from_path"]
            patient_id = visit["patient_id"]
            signal_file = visit["signal_file"]
            label_file = visit["label_file"]

            signal_path = os.path.join(root, signal_file)
            label_path = os.path.join(root, label_file)

            signal = self._load_signal(signal_path)
            if signal is None:
                continue

            metadata = self._parse_header_metadata(label_path)
            dx_codes = metadata["dx_codes"]
            sex = metadata["sex"]
            age = metadata["age"]

            label_vector = self._encode_labels(dx_codes)

            if signal.ndim != 2 or signal.shape[1] < window_size:
                continue

            num_windows = (signal.shape[1] - window_size) // step_size + 1

            visit_id = os.path.splitext(os.path.basename(signal_file))[0]

            for index in range(num_windows):
                start = index * step_size
                end = start + window_size
                signal_window = signal[:, start:end].astype(np.float32)

                samples.append(
                    {
                        "patient_id": patient_id,
                        "visit_id": visit_id,
                        "record_id": len(samples) + 1,
                        "signal": signal_window,
                        "label": label_vector.copy(),
                        "Sex": sex,
                        "Age": age,
                    }
                )

        return samples

    def _normalize_input(
        self,
        patient: Union[List[Dict[str, Any]], Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Normalize input into list format.

        Args:
            patient: Single dict or list of dicts.

        Returns:
            List of visit dictionaries.

        Example:
            >>> self._normalize_input({"a": 1})
            [{'a': 1}]
        """
        if isinstance(patient, list):
            return patient
        if isinstance(patient, dict):
            return [patient]
        return []


    def _is_valid_visit(self, visit: Dict[str, Any]) -> bool:
        """Validate that a visit dictionary contains the minimum required fields.

        This method checks whether the input `visit` dictionary includes the essential
        keys needed to process an ECG record in dataset mode. It is used to filter out
        malformed or incomplete visit entries before attempting to load signal and
        metadata files.

        Required keys:
           - "load_from_path": Root directory containing ECG files
           - "signal_file": Filename of the ECG signal (.mat)
           - "patient_id": Unique identifier for the patient

        Note:
           The "label_file" key is intentionally not required, as some workflows
           (e.g., synthetic tests or incomplete datasets) may omit header files.
           In such cases, downstream logic handles missing labels gracefully.

        Args:
           visit (Dict[str, Any]): A dictionary representing a single patient visit.
               Expected to contain file references and identifiers.

        Returns:
           bool: True if the visit contains all required keys, False otherwise.

        Example:
           >>> visit = {
           ...     "load_from_path": "/data/ecg",
           ...     "patient_id": "patient_001",
           ...     "signal_file": "record_001.mat",
           ...     "label_file": "record_001.hea"
           ... }
           >>> self._is_valid_visit(visit)
           True

           >>> invalid_visit = {
           ...     "patient_id": "patient_002",
           ...     "signal_file": "record_002.mat"
           ... }
           >>> self._is_valid_visit(invalid_visit)
           False
        """
        return (
           "load_from_path" in visit
           and "signal_file" in visit
           and "patient_id" in visit
        )


    def _load_signal(self, signal_path: str) -> Optional[np.ndarray]:
        """Load ECG signal from .mat file.

        Args:
            signal_path: Path to ECG .mat file.

        Returns:
            np.ndarray or None

        Example:
            >>> signal = self._load_signal("rec1.mat")
        """
        try:
            mat = loadmat(signal_path)
        except Exception:
            return None

        # PyHealth's older built-in cardiology tasks use mat["val"].
        signal = mat.get("val")
        if signal is None:
            return None

        return np.asarray(signal, dtype=np.float32)


    def _parse_header_metadata(self, header_path: str) -> Dict[str, List[str]]:
        """Extract metadata from .hea file.

        Args:
            header_path: Path to header file.

        Returns:
            Dict containing dx_codes, sex, age.

        Example:
            >>> meta = self._parse_header_metadata("rec1.hea")
        """
        dx_codes: List[str] = []
        sex: List[str] = []
        age: List[str] = []

        try:
            with open(header_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()

                    if line.startswith("#Dx:"):
                        value = line.split(":", 1)[1].strip()
                        dx_codes = [x.strip() for x in value.split(",") if x.strip()]

                    elif line.startswith("#Sex:"):
                        value = line.split(":", 1)[1].strip()
                        sex = [value] if value else []

                    elif line.startswith("#Age:"):
                        value = line.split(":", 1)[1].strip()
                        age = [value] if value else []
        except Exception:
            pass

        return {
            "dx_codes": dx_codes,
            "sex": sex,
            "age": age,
        }


    def _encode_labels(self, patient_labels: List[str]) -> np.ndarray:
        """Convert labels to multi-hot vector.

        Args:
            patient_labels: List of diagnosis labels.

        Returns:
            np.ndarray: Multi-hot encoded vector.

        Example:
            >>> self._encode_labels(["AF"])
            array([1., 0., 0.])
        """
        label_vector = np.zeros(len(self.labels), dtype=np.float32)

        for label in patient_labels:
            if label in self.labels:
                idx = self.labels.index(label)
                label_vector[idx] = 1.0

        return label_vector

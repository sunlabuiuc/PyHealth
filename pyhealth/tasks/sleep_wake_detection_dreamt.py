"""Sleep-wake detection task for the DREAMT dataset.

This task processes overnight Empatica E4 wearable recordings from the
DREAMT dataset into per-epoch feature vectors with binary sleep/wake labels.
This is suitable for classical and deep learning models.

The feature extraction follows the methodology described in:
    Wang et al. "Addressing wearable sleep tracking inequity: a new
    dataset and novel methods for a population with sleep disorders."
    CHIL 2024, PMLR 248:380-396.

Per-epoch feature extraction (30-second windows):
    - **ACC (ACC_X, ACC_Y, ACC_Z)**: Trimmed mean, max, IQR, and MAD
      extracted from each axis after Butterworth bandpass filtering (3-11 Hz).
    - **TEMP**: Mean, min, max, and standard deviation after winsorization
      to [31, 40] degrees C.
    - **BVP**: HRV metrics (rMSSD, SDNN, pNN50, LF power, HF power) after
      Chebyshev Type II bandpass filtering (0.5-20 Hz).
    - **EDA**: Mean tonic and phasic components after decomposition.
    - **HR**: Mean and standard deviation per epoch.
    - **ACC_INDEX**: Activity index derived from accelerometry.
    - **HRV_HFD**: Higuchi Fractal Dimension derived from BVP — one of the
      two strongest predictors identified in the paper.

Labels are binary: wake (PSG Sleep_Stage == 0) mapped to 1, all sleep
stages (REM, N1, N2, N3) mapped to 0.

"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

# Columns produced by the DREAMT feature pipeline (features_df/ folder).
# Each row is one 30-second epoch; Sleep_Stage is the PSG ground-truth label.
FEATURE_COLUMNS = [
    "ACC_X_mean", "ACC_X_max", "ACC_X_iqr", "ACC_X_mad",
    "ACC_Y_mean", "ACC_Y_max", "ACC_Y_iqr", "ACC_Y_mad",
    "ACC_Z_mean", "ACC_Z_max", "ACC_Z_iqr", "ACC_Z_mad",
    "TEMP_mean", "TEMP_min", "TEMP_max", "TEMP_std",
    "BVP_rmssd", "BVP_sdnn", "BVP_pnn50", "BVP_lf", "BVP_hf",
    "EDA_tonic", "EDA_phasic",
    "HR_mean", "HR_std",
    "ACC_INDEX", "HRV_HFD",
]

# PSG sleep stage labels → binary wake (1) vs. sleep (0) mapping
WAKE_LABEL = "W"
EXCLUDE_LABELS = {"P", "Missing"}  # drop prep and missing epochs


class SleepWakeDetectionDREAMT(BaseTask):
    """Binary sleep-wake classification task for the DREAMT dataset.

    Each sample corresponds to one 30-second epoch from a single participant.
    The task reads the pre-extracted feature CSV (from the ``features_df/``
    folder) for each patient and converts the multi-class PSG sleep stage
    annotation into a binary wake (1) vs. sleep (0) label.

    The feature set follows Wang et al. (2024): statistical summaries of ACC,
    TEMP, BVP, EDA, and HR signals, plus the ACC_INDEX activity metric and
    HRV_HFD Higuchi Fractal Dimension — the two strongest predictors identified
    in the paper.

    Attributes:
        task_name (str): ``"SleepWakeDetectionDREAMT"``
        input_schema (Dict[str, str]): ``{"features": "tensor"}``
        output_schema (Dict[str, str]): ``{"label": "binary"}``
        feature_columns (List[str]): Names of the feature columns to extract.
            Defaults to all columns listed in ``FEATURE_COLUMNS``.

    Examples:
        >>> from pyhealth.datasets import DREAMTDataset
        >>> from pyhealth.tasks import SleepWakeDetectionDREAMT
        >>> dataset = DREAMTDataset(root="/path/to/dreamt/2.1.0/")
        >>> task = SleepWakeDetectionDREAMT()
        >>> samples = dataset.set_task(task)
        >>> samples[0]
        {
            'patient_id': 'S002',
            'epoch_index': 0,
            'ahi': 15.3,
            'bmi': 32.1,
            'features': array([...]),
            'label': 0
        }
    """

    task_name: str = "SleepWakeDetectionDREAMT"
    input_schema: Dict[str, str] = {"features": "tensor"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
    ) -> None:
        """Initializes the SleepWakeDetectionDREAMT task.

        Args:
            feature_columns: List of feature column names to include in each
                sample. If None, all columns in ``FEATURE_COLUMNS`` are used.
        """
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        super().__init__()

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes a single patient into a list of per-epoch samples.

        Reads the patient's pre-extracted feature CSV, drops rows with missing
        values, converts the multi-class sleep stage annotation to a binary
        wake/sleep label, and returns one sample dict per epoch.

        Args:
            patient: A patient object from ``DREAMTDataset``. Must have a
                ``dreamt_sleep`` event with ``file_64hz``, ``ahi``, and
                ``bmi`` attributes.

        Returns:
            List[Dict[str, Any]]: One dict per epoch, each containing:
                - ``patient_id`` (str): Participant identifier.
                - ``epoch_index`` (int): Zero-based epoch position in the night.
                - ``ahi`` (float): Apnea-Hypopnea Index (clinical metadata).
                - ``bmi`` (float): Body Mass Index (clinical metadata).
                - ``features`` (List[float]): Feature vector for the epoch.
                - ``label`` (int): 1 if wake, 0 if sleep.
        """
        events = patient.get_events(event_type="dreamt_sleep")
        if not events:
            logger.warning(
                "Patient %s has no dreamt_sleep events; skipping.",
                patient.patient_id,
            )
            return []

        event = events[0]
        feature_file = event.file_64hz

        if feature_file is None:
            logger.warning(
                "Patient %s has no feature file; skipping.",
                patient.patient_id,
            )
            return []

        try:
            df = pd.read_csv(feature_file)
        except Exception as exc:
            logger.warning(
                "Could not read feature file for patient %s: %s",
                patient.patient_id,
                exc,
            )
            return []

        # Validate required columns exist
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            logger.warning(
                "Patient %s feature file missing columns %s; skipping.",
                patient.patient_id,
                missing,
            )
            return []

        if "Sleep_Stage" not in df.columns:
            logger.warning(
                "Patient %s feature file has no Sleep_Stage column; skipping.",
                patient.patient_id,
            )
            return []

        # Drop incomplete epochs
        df = df[~df["Sleep_Stage"].isin(EXCLUDE_LABELS)].reset_index(drop=True)

        samples = []
        for epoch_index, row in df.iterrows():
            # Binary label: wake (PSG label == 0) → 1, all sleep stages → 0
            label = int(row["Sleep_Stage"] == WAKE_LABEL)
            features = row[self.feature_columns].tolist()

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "epoch_index": int(epoch_index),
                    "ahi": float(getattr(event, "ahi", float("nan"))),
                    "bmi": float(getattr(event, "bmi", float("nan"))),
                    "features": features,
                    "label": label,
                }
            )

        return samples
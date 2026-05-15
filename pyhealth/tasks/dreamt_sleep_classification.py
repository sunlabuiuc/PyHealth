import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

# 3-class: Wake / NREM (N1+N2+N3 merged) / REM
_LABEL_MAP_3CLASS: Dict[str, int] = {
    "W": 0,
    "N1": 1,
    "N2": 1,
    "N3": 1,
    "R": 2,
}

# 4-class: Wake / N1 / N2 / N3 / REM  (5 distinct stages, labelled 0-4)
_LABEL_MAP_4CLASS: Dict[str, int] = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "R": 4,
}


class DREAMTSleepClassification(BaseTask):
    """IBI-based sleep staging task for the DREAMT wearable dataset.

    Implements the preprocessing pipeline from WatchSleepNet (Wang et al., 2025),
    which performs sequence-to-sequence sleep stage classification using
    Inter-Beat Interval (IBI) signals derived from wrist-worn PPG sensors.

    Each patient's full-night 64 Hz recording is segmented into non-overlapping
    ``epoch_seconds``-second windows. For each window, the non-zero IBI values
    are extracted as the input signal and a majority-vote sleep stage label is
    assigned.

    Two label configurations are supported:

    - **3-class** (default, matches the paper): Wake (0), NREM (1), REM (2).
      N1, N2, and N3 are merged into a single NREM class.
    - **4-class** (ablation extension): Wake (0), N1 (1), N2 (2), N3 (3),
      REM (4). This provides finer clinical granularity beyond the paper.

    An optional accelerometer ablation (``use_accelerometer=True``) appends the
    raw ACC_X/ACC_Y/ACC_Z time series to each sample for wake-detection studies.

    Attributes:
        task_name (str): ``"DREAMTSleepClassification"``.
        input_schema (Dict[str, str]): ``{"ibi_sequence": "tensor"}`` by default;
            ``{"ibi_sequence": "tensor", "accelerometer": "tensor"}`` when
            ``use_accelerometer=True``.
        output_schema (Dict[str, str]): ``{"label": "multiclass"}``.

    References:
        Wang et al., "WatchSleepNet: A Novel Model and Pretraining Approach for
        Advancing Sleep Staging with Smartwatches", 2025.
        https://doi.org/10.48550/arXiv.2501.17268
    """

    task_name: str = "DREAMTSleepClassification"
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __init__(
        self,
        epoch_seconds: int = 30,
        num_classes: int = 3,
        use_accelerometer: bool = False,
        sample_rate: int = 64,
    ) -> None:
        """Initializes the DREAMTSleepClassification task.

        Args:
            epoch_seconds: Duration of each epoch window in seconds. Default 30.
            num_classes: Number of sleep stage classes. Use 3 for Wake/NREM/REM
                (paper default) or 4 for Wake/N1/N2/N3/REM (ablation). Default 3.
            use_accelerometer: If True, include raw ACC_X/ACC_Y/ACC_Z signals
                alongside IBI in each sample (ablation for wake detection).
                Default False.
            sample_rate: Sampling rate (Hz) of the DREAMT data file used for
                epoch windowing. Default 64.

        Raises:
            ValueError: If ``num_classes`` is not 3 or 4.
        """
        if num_classes not in (3, 4):
            raise ValueError(
                f"num_classes must be 3 (Wake/NREM/REM) or 4 (Wake/N1/N2/N3/REM),"
                f" got {num_classes}."
            )
        self.epoch_seconds = epoch_seconds
        self.num_classes = num_classes
        self.use_accelerometer = use_accelerometer
        self.sample_rate = sample_rate
        self.label_map: Dict[str, int] = (
            _LABEL_MAP_3CLASS if num_classes == 3 else _LABEL_MAP_4CLASS
        )
        self.input_schema: Dict[str, str] = (
            {"ibi_sequence": "tensor", "accelerometer": "tensor"}
            if use_accelerometer
            else {"ibi_sequence": "tensor"}
        )
        super().__init__()

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Segments a patient's full-night recording into labeled IBI epochs.

        Reads the patient's 64 Hz DREAMT CSV file, creates non-overlapping
        ``epoch_seconds``-second windows, and for each window extracts the
        IBI sequence and majority-vote sleep stage label.

        Args:
            patient: A patient object returned by ``DREAMTDataset``.  Must
                expose a ``get_events(event_type="dreamt_sleep")`` method whose
                event objects carry a ``file_64hz`` attribute.

        Returns:
            A list of sample dicts.  Each dict contains:

            - ``patient_id`` (str): Patient identifier (e.g., ``"S001"``).
            - ``epoch_idx`` (int): Zero-based index of the epoch within the
              recording.
            - ``ibi_sequence`` (np.ndarray, float32): IBI values (in seconds)
              that fell within the epoch.  Length varies with heart rate.
            - ``label`` (int): Sleep stage class index.
            - ``accelerometer`` (np.ndarray, float32, shape
              ``(samples_per_epoch, 3)``): Raw ACC_X/Y/Z signals. **Only
              present when** ``use_accelerometer=True``.

        Examples:
            >>> from pyhealth.datasets import DREAMTDataset
            >>> from pyhealth.tasks import DREAMTSleepClassification
            >>> dreamt = DREAMTDataset(root="/path/to/dreamt/2.1.0")
            >>> task = DREAMTSleepClassification(num_classes=3)
            >>> dataset = dreamt.set_task(task)
            >>> dataset[0]
            {
                'patient_id': 'S001',
                'epoch_idx': 0,
                'ibi_sequence': array([0.84, 0.81, ...], dtype=float32),
                'label': 1
            }
        """
        samples_per_epoch = self.sample_rate * self.epoch_seconds
        samples: List[Dict[str, Any]] = []

        events = patient.get_events(event_type="dreamt_sleep")
        for event in events:
            if event.file_64hz is None:
                logger.warning(
                    "Patient %s has no 64 Hz file; skipping.", patient.patient_id
                )
                continue

            try:
                df = pd.read_csv(str(event.file_64hz))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not read %s for patient %s: %s",
                    event.file_64hz,
                    patient.patient_id,
                    exc,
                )
                continue

            if not {"IBI", "Sleep_Stage"}.issubset(df.columns):
                logger.warning(
                    "Patient %s: required columns missing from %s; skipping.",
                    patient.patient_id,
                    event.file_64hz,
                )
                continue

            n_epochs = len(df) // samples_per_epoch
            for epoch_idx in range(n_epochs):
                start = epoch_idx * samples_per_epoch
                end = start + samples_per_epoch
                epoch_df = df.iloc[start:end]

                # Determine label by majority vote across the epoch's rows.
                stage_counts = epoch_df["Sleep_Stage"].value_counts()
                if stage_counts.empty:
                    continue
                majority_stage = stage_counts.index[0]
                if majority_stage not in self.label_map:
                    continue
                label = self.label_map[majority_stage]

                # Extract IBI: only rows where a heartbeat was detected.
                ibi_mask = epoch_df["IBI"].notna() & (epoch_df["IBI"] > 0)
                ibi_values = epoch_df.loc[ibi_mask, "IBI"].to_numpy(
                    dtype=np.float32
                )
                if ibi_values.size == 0:
                    continue

                sample: Dict[str, Any] = {
                    "patient_id": patient.patient_id,
                    "epoch_idx": epoch_idx,
                    "ibi_sequence": ibi_values,
                    "label": label,
                }

                if self.use_accelerometer:
                    acc_cols = ["ACC_X", "ACC_Y", "ACC_Z"]
                    if not set(acc_cols).issubset(df.columns):
                        logger.warning(
                            "Patient %s: accelerometer columns missing at "
                            "epoch %d; skipping.",
                            patient.patient_id,
                            epoch_idx,
                        )
                        continue
                    sample["accelerometer"] = epoch_df[acc_cols].to_numpy(
                        dtype=np.float32
                    )

                samples.append(sample)

        return samples

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)


class SleepStagingDREAMT(BaseTask):
    """Create sleep-staging samples from DREAMT wearable recordings.

    The task supports two input modes:

    - ``BVP`` (default): Converts each BVP epoch into a sample-level
      inter-beat interval (IBI) sequence using NeuroKit2.
    - ``IBI``: Uses precomputed IBI values directly from the CSV file.

    Signals are divided into non-overlapping epochs. By default, the
    sampling rate is 64 Hz and each epoch is 30 seconds long, producing
    1,920 samples per epoch. The output signal contains IBI values in
    seconds and has shape ``(1, sampling_rate * chunk_duration)``.

    In precomputed IBI mode, non-finite, non-positive, and IBI values
    greater than or equal to ``max_ibi_seconds`` are replaced with zero.
    In BVP mode, epochs containing non-finite BVP values are skipped.

    Each output sample contains:
        - ``signal``: IBI array with shape
          ``(1, sampling_rate * chunk_duration)``.
        - ``label``: Three-class sleep-stage label.

    Label mapping:
        - ``0``: Wake (W).
        - ``1``: NREM (N1, N2, N3).
        - ``2``: REM (R).

    Epochs with missing, unknown, or inconsistent stage labels are skipped.

    Example:
        >>> task = SleepStagingDREAMT()
        >>> task.signal_type
        'BVP'
        >>> task.sampling_rate
        64
        >>> task.chunk_duration
        30
        >>> ibi_task = SleepStagingDREAMT(signal_type="IBI")
    """

    task_name = "SleepStagingDREAMT"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __init__(
        self,
        sampling_rate: int = 64,
        chunk_duration: int = 30,
        max_ibi_seconds: float = 2.0,
        bvp_column: str = "BVP",
        ibi_column: str = "IBI",
        signal_type: str = "BVP",
        stage_column: str = "Sleep_Stage",
        sequence_first: bool = False,
    ) -> None:
        """Initialize the DREAMT sleep-staging task.

        Args:
            sampling_rate: BVP sampling frequency in Hz.
            chunk_duration: Epoch duration in seconds.
            max_ibi_seconds: Maximum accepted inter-beat interval in seconds.
            bvp_column: CSV column containing the BVP signal.
            ibi_column: CSV column containing precomputed IBI values.
            signal_type: Input signal type, either BVP or IBI.
            stage_column: CSV column containing sleep-stage labels.
            sequence_first: Return signals as (sequence, features) when True.

        Raises:
            ValueError: If a numeric parameter is invalid or a column
                name is empty.
        """
        if (
            isinstance(sampling_rate, bool)
            or not isinstance(sampling_rate, (int, np.integer))
            or sampling_rate <= 0
        ):
            raise ValueError("sampling_rate must be a positive integer.")

        if (
            isinstance(chunk_duration, bool)
            or not isinstance(chunk_duration, (int, np.integer))
            or chunk_duration <= 0
        ):
            raise ValueError("chunk_duration must be a positive integer.")

        if (
            isinstance(max_ibi_seconds, bool)
            or not isinstance(max_ibi_seconds, (int, float, np.integer, np.floating))
            or not np.isfinite(max_ibi_seconds)
            or max_ibi_seconds <= 0
        ):
            raise ValueError("max_ibi_seconds must be a positive finite number.")

        if not isinstance(bvp_column, str) or not bvp_column.strip():
            raise ValueError("bvp_column must be a non-empty string.")

        if not isinstance(ibi_column, str) or not ibi_column.strip():
            raise ValueError("ibi_column must be a non-empty string.")

        if not isinstance(signal_type, str) or signal_type.upper() not in {
            "BVP",
            "IBI",
        }:
            raise ValueError("signal_type must be either 'BVP' or 'IBI'.")

        if not isinstance(stage_column, str) or not stage_column.strip():
            raise ValueError("stage_column must be a non-empty string.")
        if not isinstance(sequence_first, bool):
            raise ValueError("sequence_first must be a boolean.")

        self.sampling_rate = sampling_rate
        self.chunk_duration = chunk_duration
        self.max_ibi_seconds = max_ibi_seconds
        self.bvp_column = bvp_column
        self.ibi_column = ibi_column
        self.signal_type = signal_type.upper()
        self.stage_column = stage_column
        self.sequence_first = sequence_first
        super().__init__()

    @staticmethod
    def _map_stage(stage: Any) -> int | None:
        """Map a DREAMT sleep-stage label to one of three classes.

        Args:
            stage: Original DREAMT sleep-stage value.

        Returns:
            Integer class label (0=Wake, 1=NREM, 2=REM), or None
            when the stage is missing or unsupported.
        """
        if pd.isna(stage):
            return None

        stage = str(stage).strip().upper()

        if stage == "W":
            return 0
        if stage in {"N1", "N2", "N3"}:
            return 1
        if stage == "R":
            return 2

        return None

    def _extract_ibi(self, bvp_epoch: np.ndarray) -> np.ndarray:
        """Convert one BVP epoch into a sample-level IBI sequence.

        Args:
            bvp_epoch: One-dimensional BVP signal for a single epoch.
                Its length is ``sampling_rate * chunk_duration`` samples.

        Returns:
            A float32 IBI array with the same length as the input.
            Intervals exceeding ``max_ibi_seconds`` are represented by zero.
        """
        if (
            not isinstance(bvp_epoch, np.ndarray)
            or bvp_epoch.ndim != 1
            or bvp_epoch.size == 0
        ):
            raise ValueError(
                "bvp_epoch must be a non-empty one-dimensional NumPy array."
            )

        if not np.all(np.isfinite(bvp_epoch)):
            raise ValueError("bvp_epoch must contain only finite values.")

        if np.ptp(bvp_epoch) == 0:
            return np.zeros(len(bvp_epoch), dtype=np.float32)

        try:
            import neurokit2 as nk
        except ImportError as exc:
            raise ImportError(
                "BVP processing requires NeuroKit2. "
                "Install it with `pip install pyhealth[dreamt]`."
            ) from exc

        cleaned = nk.ppg_clean(
            bvp_epoch,
            sampling_rate=self.sampling_rate,
        )

        peak_info = nk.ppg_findpeaks(
            cleaned,
            sampling_rate=self.sampling_rate,
        )
        peaks = np.asarray(peak_info["PPG_Peaks"], dtype=int)

        ibi = np.zeros(len(bvp_epoch), dtype=np.float32)

        if len(peaks) < 2:
            return ibi

        intervals = np.diff(peaks) / self.sampling_rate

        for index, interval in enumerate(intervals):
            if interval < self.max_ibi_seconds:
                ibi[peaks[index] : peaks[index + 1]] = interval

        return ibi

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Process a DREAMT participant and generate epoch-level samples.

        Each valid recording is divided into non-overlapping epochs.
        BVP signals are converted into IBI sequences, and each epoch
        receives a three-class sleep-stage label.

        Args:
            patient: PyHealth patient object containing DREAMT recording
                events. Each event is expected to provide a ``file_64hz``
                path to its wearable recording.

        Returns:
            A list of sample dictionaries. Each dictionary contains:
                - ``patient_id``: Participant identifier.
                - ``signal``: IBI array with shape ``(1, samples_per_epoch)``.
                - ``label``: Integer sleep-stage label (0=Wake, 1=NREM, 2=REM).
                - ``age``, ``gender``, ``ahi``: Available participant metadata.
                - ``epoch_index``: Zero-based epoch index within the recording.

            Returns an empty list if no valid epochs are found.
        """
        samples = []
        events = patient.get_events()

        samples_per_epoch = self.sampling_rate * self.chunk_duration

        for event in events:
            file_path = getattr(event, "file_64hz", None)

            if not file_path or not Path(file_path).is_file():
                continue

            try:
                signal_column = (
                    self.bvp_column if self.signal_type == "BVP" else self.ibi_column
                )
                data = pd.read_csv(
                    file_path,
                    usecols=[signal_column, self.stage_column],
                )

                input_signal = pd.to_numeric(
                    data[signal_column], errors="coerce"
                ).to_numpy(dtype=np.float64)
                stages = data[self.stage_column].to_numpy()

            except (
                OSError,
                ValueError,
                pd.errors.ParserError,
                pd.errors.EmptyDataError,
            ) as exc:
                logger.warning(
                    "Skipping unreadable DREAMT file for patient %s: %s",
                    patient.patient_id,
                    exc,
                )
                continue

            num_epochs = len(input_signal) // samples_per_epoch

            for epoch_index in range(num_epochs):
                start = epoch_index * samples_per_epoch
                end = start + samples_per_epoch

                epoch_signal = input_signal[start:end]
                epoch_stages = stages[start:end]

                mapped_stages = [self._map_stage(stage) for stage in epoch_stages]

                if not mapped_stages or mapped_stages[0] is None:
                    continue

                label = mapped_stages[0]

                if any(stage != label for stage in mapped_stages):
                    continue

                if self.signal_type == "BVP":
                    if not np.all(np.isfinite(epoch_signal)):
                        continue
                    try:
                        signal = self._extract_ibi(epoch_signal)
                    except Exception as exc:
                        logger.warning(
                            "Failed to extract IBI for patient %s, epoch %s: %s",
                            patient.patient_id,
                            epoch_index,
                            exc,
                        )
                        continue
                else:
                    signal = np.where(
                        np.isfinite(epoch_signal)
                        & (epoch_signal > 0)
                        & (epoch_signal < self.max_ibi_seconds),
                        epoch_signal,
                        0.0,
                    ).astype(np.float32)

                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "signal": (
                            signal.reshape(-1, 1)
                            if self.sequence_first
                            else signal.reshape(1, -1)
                        ),
                        "label": int(label),
                        "age": getattr(event, "age", None),
                        "gender": getattr(event, "gender", None),
                        "ahi": getattr(event, "ahi", None),
                        "epoch_index": epoch_index,
                    }
                )

        return samples

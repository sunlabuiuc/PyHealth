"""Sequence-level DREAMT preprocessing for WatchSleepNet."""

import logging
import warnings
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)


class SleepStagingDREAMTSequence(BaseTask):
    """Create participant-level IBI sequences for sleep staging.

    Reads 100 Hz DREAMT BVP recordings, extracts IBI values from
    consistent 30-second epochs, and downsamples each epoch to 25 Hz.

    Each participant produces one sample:
        - signal: float array shaped (number_of_epochs, 750)
        - label: time-series labels with ``time`` and ``values`` fields

    Labels are Wake=0, NREM=1, and REM=2.

    Args:
        context_seconds: Seconds of surrounding BVP signal
            used on each side of an epoch for optional
            context-aware IBI extraction. The default of zero
            preserves independent epochwise extraction.
            Five seconds reproduces the evaluated extension.
            Positive context requires valid recording timestamps;
            otherwise, epochwise extraction is retained.
            Future context makes this a retrospective method.

    Examples:
        >>> task = SleepStagingDREAMTSequence()
        >>> task.context_seconds
        0
        >>> contextual_task = SleepStagingDREAMTSequence(
        ...     context_seconds=5
        ... )
        >>> contextual_task.context_seconds
        5
    """

    task_name = "SleepStagingDREAMTSequence"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, Any]] = {
        "label": (
            "timeseries_multiclass",
            {"labels": 3, "timeline": ["signal"]},
        )
    }

    def __init__(
        self,
        sampling_rate: int = 100,
        chunk_duration: int = 30,
        target_sampling_rate: int = 25,
        max_ibi_seconds: float = 2.0,
        bvp_column: str = "BVP",
        stage_column: str = "Sleep_Stage",
        context_seconds: int = 0,
    ) -> None:
        if sampling_rate <= 0 or target_sampling_rate <= 0:
            raise ValueError("Sampling rates must be positive.")
        if chunk_duration <= 0:
            raise ValueError("chunk_duration must be positive.")
        if sampling_rate % target_sampling_rate != 0:
            raise ValueError("sampling_rate must be divisible by target_sampling_rate.")
        if max_ibi_seconds <= 0:
            raise ValueError("max_ibi_seconds must be positive.")
        if not bvp_column or not stage_column:
            raise ValueError("Column names must be non-empty.")

        if (
            isinstance(context_seconds, bool)
            or not isinstance(context_seconds, int)
            or context_seconds < 0
        ):
            raise ValueError(
                "context_seconds must be a non-negative integer."
            )

        self.context_seconds = context_seconds
        self.sampling_rate = sampling_rate
        self.chunk_duration = chunk_duration
        self.target_sampling_rate = target_sampling_rate
        self.max_ibi_seconds = max_ibi_seconds
        self.bvp_column = bvp_column
        self.stage_column = stage_column
        super().__init__()

    @staticmethod
    def _map_stage(stage: Any) -> int | None:
        """Map DREAMT stages to Wake=0, NREM=1, and REM=2."""
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
        """Extract sample-level IBI values using NeuroKit2."""
        if bvp_epoch.ndim != 1 or not np.all(np.isfinite(bvp_epoch)):
            raise ValueError("BVP epoch must be finite and one-dimensional.")

        if np.ptp(bvp_epoch) == 0:
            return np.zeros(len(bvp_epoch), dtype=np.float32)

        try:
            import neurokit2 as nk
        except ImportError as exc:
            raise ImportError(
                "Install NeuroKit2 with `pip install pyhealth[dreamt]`."
            ) from exc

        ibi = np.zeros(len(bvp_epoch), dtype=np.float32)
        signals, info = nk.ppg_process(
            bvp_epoch,
            sampling_rate=self.sampling_rate,
        )
        del signals

        peaks = np.asarray(info["PPG_Peaks"], dtype=int)
        if len(peaks) < 2:
            return ibi

        intervals = np.diff(peaks) / self.sampling_rate
        for index, interval in enumerate(intervals):
            if interval < self.max_ibi_seconds:
                ibi[peaks[index] : peaks[index + 1]] = interval

        return ibi

    def _apply_ibi_context(
        self,
        bvp: np.ndarray,
        timestamps: np.ndarray | None,
        start: int,
        end: int,
        epochwise_ibi: np.ndarray,
    ) -> np.ndarray:
        """Refine an existing epoch using surrounding BVP.

        Preserve the epochwise result if the context is invalid
        or extraction fails. Context never crosses file boundaries.
        """
        if self.context_seconds == 0 or timestamps is None:
            return epochwise_ibi

        context_samples = (
            self.context_seconds * self.sampling_rate
        )
        left = max(0, start - context_samples)
        right = min(len(bvp), end + context_samples)

        segment = bvp[left:right]
        segment_times = timestamps[left:right]

        if not np.isfinite(segment).all():
            return epochwise_ibi

        if not np.isfinite(segment_times).all():
            return epochwise_ibi

        if not np.allclose(
            np.diff(segment_times),
            1.0 / self.sampling_rate,
            atol=0.0001,
            rtol=0,
        ):
            return epochwise_ibi

        factor = self.sampling_rate // self.target_sampling_rate
        target_samples = (
            self.chunk_duration * self.target_sampling_rate
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                expanded = self._extract_ibi(segment)

            candidate = expanded[
                start - left : end - left
            ][::factor].astype(np.float32)

        except ImportError:
            raise
        except Exception:  # noqa: BLE001
            return epochwise_ibi

        if not (
            candidate.shape == (target_samples,)
            and np.isfinite(candidate).all()
            and (candidate >= 0).all()
            and (candidate < self.max_ibi_seconds).all()
        ):
            return epochwise_ibi

        return candidate

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Generate one chronological sequence sample per participant."""
        all_signals = []
        all_labels = []
        all_times = []
        metadata = None

        samples_per_epoch = self.sampling_rate * self.chunk_duration
        factor = self.sampling_rate // self.target_sampling_rate
        target_samples = self.chunk_duration * self.target_sampling_rate
        elapsed_samples = 0

        for event in patient.get_events():
            file_path = getattr(event, "file_100hz", None)
            if not file_path or not Path(file_path).is_file():
                continue

            try:
                columns = [
                    self.bvp_column,
                    self.stage_column,
                ]

                timestamps = None

                if self.context_seconds > 0:
                    requested = set(columns) | {"TIMESTAMP"}

                    data = pd.read_csv(
                        file_path,
                        usecols=lambda name, requested=requested: name in requested,
                    )

                    if not set(columns).issubset(data.columns):
                        raise ValueError(
                            "Required DREAMT columns are missing."
                        )

                    if "TIMESTAMP" in data.columns:
                        timestamps = pd.to_numeric(
                            data["TIMESTAMP"],
                            errors="coerce",
                        ).to_numpy(dtype=np.float64)
                else:
                    data = pd.read_csv(
                        file_path,
                        usecols=columns,
                    )
                bvp = pd.to_numeric(data[self.bvp_column], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                stages = data[self.stage_column].to_numpy()
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                logger.warning("Could not read %s: %s", file_path, exc)
                continue

            if len(bvp) != len(stages):
                continue

            if metadata is None:
                metadata = {
                    "age": getattr(event, "age", None),
                    "gender": getattr(event, "gender", None),
                    "ahi": getattr(event, "ahi", None),
                }

            # Map stages first, then identify contiguous runs. This reproduces
            # the original sliding-alignment behavior without checking every
            # overlapping 30-second window individually.
            mapped = np.fromiter(
                (
                    stage if (stage := self._map_stage(value)) is not None else -1
                    for value in stages
                ),
                dtype=np.int8,
                count=len(stages),
            )

            change_points = np.flatnonzero(mapped[1:] != mapped[:-1]) + 1
            run_starts = np.concatenate(([0], change_points))
            run_ends = np.concatenate((change_points, [len(mapped)]))

            for run_start, run_end in zip(run_starts, run_ends):
                label = int(mapped[run_start])
                if label < 0:
                    continue

                # A valid epoch begins at the start of a consistent stage run.
                # Any remainder shorter than one full epoch is not used.
                for start in range(
                    run_start,
                    run_end - samples_per_epoch + 1,
                    samples_per_epoch,
                ):
                    end = start + samples_per_epoch
                    epoch_bvp = bvp[start:end]

                    if not np.all(np.isfinite(epoch_bvp)):
                        continue

                    try:
                        ibi = self._extract_ibi(epoch_bvp)
                    except ImportError:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "IBI extraction failed for patient %s, "
                            "recording sample %s: %s",
                            patient.patient_id,
                            start,
                            exc,
                        )
                        continue

                    ibi = ibi[::factor].astype(np.float32)
                    if ibi.shape != (target_samples,):
                        continue

                    if self.context_seconds > 0:
                        ibi = self._apply_ibi_context(
                            bvp,
                            timestamps,
                            start,
                            end,
                            ibi,
                        )

                    all_signals.append(ibi)
                    all_labels.append(label)
                    all_times.append(
                        float((elapsed_samples + start) / self.sampling_rate)
                    )

            elapsed_samples += len(bvp)

        if not all_signals:
            return []

        return [
            {
                "patient_id": patient.patient_id,
                "signal": np.stack(all_signals).astype(np.float32),
                "label": {"time": all_times, "values": all_labels},
                **(metadata or {}),
            }
        ]

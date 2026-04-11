"""Task for classifying Daily and Sports Activities (DSA) sensor segments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .base_task import BaseTask


class DSAActivityClassification(BaseTask):
    """Multi-class activity-recognition task for the DSA dataset.

    The DSA paper models each wearable placement as a domain. This task keeps
    that structure explicit by letting the caller choose one or more body-site
    units and returning a fixed-length multivariate time series for each
    activity segment.

    Each output sample corresponds to one ``sXX.txt`` segment file and contains:

    - ``signal``: a ``(125, 9 * num_selected_units)`` float tensor-like array
    - ``label``: an integer in ``[0, 18]`` representing one of the 19 DSA
      activities

    The default ``normalization="minmax"`` follows the paper's preprocessing by
    rescaling each channel independently to ``[-1, 1]``. Constant-valued
    channels are mapped to zeros.

    Args:
        dataset_root: Root directory of the DSA dataset. The current DSA
            manifest stores segment paths relative to this directory, so the
            task needs the root to load segment files.
        selected_units: Optional subset of DSA body sites to use. Valid values
            are ``"T"``, ``"RA"``, ``"LA"``, ``"RL"``, and ``"LL"``. ``None``
            uses all five units in canonical DSA order.
        normalization: Feature normalization strategy. ``"minmax"`` rescales
            each selected channel to ``[-1, 1]``; ``"none"`` preserves raw
            values.

    Examples:
        >>> from pyhealth.datasets import DSADataset
        >>> from pyhealth.tasks import DSAActivityClassification
        >>> dataset = DSADataset(root="/path/to/dsa")
        >>> task = DSAActivityClassification(
        ...     dataset_root="/path/to/dsa",
        ...     selected_units=("LA",),
        ... )
        >>> sample_dataset = dataset.set_task(task)
        >>> sample = sample_dataset[0]
        >>> tuple(sample["signal"].shape)
        (125, 9)
    """

    task_name: str = "DSAActivityClassification"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    VALID_UNITS: ClassVar[Tuple[str, ...]] = ("T", "RA", "LA", "RL", "LL")
    SENSOR_KEYS: ClassVar[Tuple[str, ...]] = (
        "xacc",
        "yacc",
        "zacc",
        "xgyro",
        "ygyro",
        "zgyro",
        "xmag",
        "ymag",
        "zmag",
    )
    CHANNELS_PER_UNIT: ClassVar[int] = 9
    SEGMENT_LENGTH: ClassVar[int] = 125
    TOTAL_CHANNELS: ClassVar[int] = 45
    NUM_CLASSES: ClassVar[int] = 19

    def __init__(
        self,
        dataset_root: Union[str, Path],
        selected_units: Optional[Union[str, Sequence[str]]] = None,
        normalization: str = "minmax",
    ) -> None:
        """Initialize the DSA activity-classification task."""
        if normalization not in {"minmax", "none"}:
            raise ValueError(
                "Unsupported normalization. Expected 'minmax' or 'none'."
            )

        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.selected_units = self._normalize_units(selected_units)
        self.normalization = normalization
        self.channel_names = self._build_channel_names(self.selected_units)

    @classmethod
    def _normalize_units(
        cls,
        selected_units: Optional[Union[str, Sequence[str]]],
    ) -> Tuple[str, ...]:
        """Validate and normalize the requested DSA body-site units."""
        if selected_units is None:
            return cls.VALID_UNITS

        if isinstance(selected_units, str):
            requested_units = [selected_units]
        else:
            requested_units = list(selected_units)

        normalized_units: List[str] = []
        seen_units = set()
        for unit in requested_units:
            normalized = unit.upper()
            if normalized not in cls.VALID_UNITS:
                raise ValueError(
                    f"Unsupported DSA unit '{unit}'. "
                    f"Expected one of {cls.VALID_UNITS}."
                )
            if normalized not in seen_units:
                normalized_units.append(normalized)
                seen_units.add(normalized)

        if not normalized_units:
            raise ValueError("selected_units must contain at least one DSA unit.")

        return tuple(normalized_units)

    @classmethod
    def _build_channel_names(cls, selected_units: Sequence[str]) -> List[str]:
        """Build human-readable channel names for the selected units."""
        return [
            f"{unit}_{sensor_key}"
            for unit in selected_units
            for sensor_key in cls.SENSOR_KEYS
        ]

    @classmethod
    def _channel_slice_for_unit(cls, unit: str) -> slice:
        """Return the column slice for a single DSA body-site unit."""
        unit_index = cls.VALID_UNITS.index(unit)
        start = unit_index * cls.CHANNELS_PER_UNIT
        stop = start + cls.CHANNELS_PER_UNIT
        return slice(start, stop)

    @classmethod
    def _extract_units(
        cls,
        full_signal: np.ndarray,
        selected_units: Sequence[str],
    ) -> np.ndarray:
        """Extract and concatenate columns for the chosen DSA units."""
        unit_signals = [
            full_signal[:, cls._channel_slice_for_unit(unit)]
            for unit in selected_units
        ]
        return np.concatenate(unit_signals, axis=1)

    @staticmethod
    def _minmax_normalize(signal: np.ndarray) -> np.ndarray:
        """Scale each channel independently to ``[-1, 1]``."""
        signal_min = signal.min(axis=0, keepdims=True)
        signal_max = signal.max(axis=0, keepdims=True)
        signal_range = signal_max - signal_min

        safe_range = np.where(signal_range > 0.0, signal_range, 1.0)
        normalized = 2.0 * (signal - signal_min) / safe_range - 1.0
        normalized[:, signal_range[0] == 0.0] = 0.0
        return normalized.astype(np.float32, copy=False)

    @classmethod
    def _activity_label(cls, activity_code: str) -> int:
        """Convert a DSA activity code like ``A01`` into a zero-based label."""
        if not isinstance(activity_code, str) or not activity_code.startswith("A"):
            raise ValueError(f"Invalid DSA activity code: {activity_code!r}")

        try:
            label = int(activity_code[1:]) - 1
        except ValueError as exc:
            raise ValueError(
                f"Invalid DSA activity code: {activity_code!r}"
            ) from exc

        if label < 0 or label >= cls.NUM_CLASSES:
            raise ValueError(f"Invalid DSA activity code: {activity_code!r}")
        return label

    def _resolve_segment_path(self, segment_path: str) -> Path:
        """Resolve a stored segment path against the configured dataset root."""
        path = Path(segment_path)
        if path.is_absolute():
            return path
        return self.dataset_root / path

    @classmethod
    def _load_segment(cls, file_path: Path) -> np.ndarray:
        """Load a DSA segment file and validate its expected shape."""
        try:
            signal = np.loadtxt(file_path, delimiter=",", dtype=np.float64)
        except Exception as exc:
            raise ValueError(
                f"Failed to parse DSA segment {file_path}."
            ) from exc

        if signal.ndim == 1:
            signal = signal.reshape(1, -1)

        expected_shape = (cls.SEGMENT_LENGTH, cls.TOTAL_CHANNELS)
        if signal.shape != expected_shape:
            raise ValueError(
                f"{file_path} has shape {signal.shape}, expected {expected_shape}."
            )
        if not np.isfinite(signal).all():
            raise ValueError(f"{file_path} contains non-finite values.")
        return signal

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Create one activity-classification sample per DSA segment event.

        Args:
            patient: A PyHealth patient object created from ``DSADataset``.

        Returns:
            A list of task samples. Each sample includes metadata for the
            subject and segment alongside the extracted multivariate signal and
            multiclass label.
        """
        samples: List[Dict[str, Any]] = []

        for event in patient.get_events(event_type="segments"):
            segment_file = self._resolve_segment_path(event.segment_path)
            full_signal = self._load_segment(segment_file)
            signal = self._extract_units(full_signal, self.selected_units)

            if self.normalization == "minmax":
                signal = self._minmax_normalize(signal)
            else:
                signal = signal.astype(np.float32, copy=False)

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "subject_id": patient.patient_id,
                    "sample_id": f"{patient.patient_id}:{event.segment_path}",
                    "segment_path": event.segment_path,
                    "activity_name": event.activity_name,
                    "activity_code": event.activity_code,
                    "unit_combo": "+".join(self.selected_units),
                    "num_channels": signal.shape[1],
                    "channel_names": self.channel_names,
                    "signal": signal,
                    "label": self._activity_label(event.activity_code),
                }
            )

        return samples

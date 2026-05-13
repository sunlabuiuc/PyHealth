"""SleepStagingIBI: PyHealth task for IBI-based sleep staging.

Converts IBISleepDataset patient records into per-epoch sample dicts
for 3-class (Wake / NREM / REM) or 5-class (W / N1 / N2 / N3 / REM)
sleep staging using 30-second IBI signal epochs at 25 Hz.
"""

from typing import Any, Dict, List, Literal

import numpy as np

from pyhealth.tasks import BaseTask

_SAMPLES_PER_EPOCH: int = 750
_MAX_EPOCHS: int = 1_100
_LABEL_MAP_3CLASS: Dict[int, int] = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}
_LABEL_MAP_5CLASS: Dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}


class SleepStagingIBI(BaseTask):
    """Multi-class sleep staging task for IBI signals from IBISleepDataset.

    Each 30-second epoch of the IBI time series (750 samples at 25 Hz) is
    mapped to a single sleep stage label. Supports 3-class and 5-class label
    spaces.

    Attributes:
        task_name (str): ``"SleepStagingIBI"``
        input_schema (Dict[str, str]): ``{"signal": "tensor"}``
        output_schema (Dict[str, str]): ``{"label": "multiclass"}``

    Args:
        num_classes: Label granularity. ``3`` → Wake/NREM/REM;
            ``5`` → W/N1/N2/N3/REM. Default ``3``.
        max_epochs: Maximum epochs to return per subject. Default ``1100``.

    Examples:
        >>> from pyhealth.tasks import SleepStagingIBI
        >>> task = SleepStagingIBI(num_classes=3)
        >>> task.task_name
        'SleepStagingIBI'
    """

    task_name: str = "SleepStagingIBI"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __init__(
        self,
        num_classes: Literal[3, 5] = 3,
        max_epochs: int = _MAX_EPOCHS,
    ) -> None:
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self._label_map = _LABEL_MAP_3CLASS if num_classes == 3 else _LABEL_MAP_5CLASS
        super().__init__()

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Convert a patient's IBI record into per-epoch sample dicts.

        Args:
            patient: A ``Patient`` object from ``IBISleepDataset``.
                Each event exposes ``event.npz_path`` and ``event.ahi``.

        Returns:
            List of sample dicts, one per valid epoch::

                {
                    "patient_id": str,
                    "signal":     np.ndarray,  # float32, shape (750,)
                    "label":      int,          # mapped sleep stage
                    "ahi":        float,        # may be NaN
                }

            Returns ``[]`` if the NPZ contains fewer than 750 samples.

        Raises:
            ValueError: If ``fs != 25`` in the NPZ file.
        """
        pid = patient.patient_id
        samples: List[Dict[str, Any]] = []

        for event in patient.get_events():
            npz = np.load(event.npz_path, allow_pickle=False)
            signal_data = npz["data"].astype(np.float32)
            stages = npz["stages"].astype(np.int32)
            fs = int(npz["fs"])
            ahi_val = getattr(event, "ahi", None)
            ahi = float(ahi_val) if ahi_val is not None else float("nan")

            if fs != 25:
                raise ValueError(
                    f"Expected fs=25, got fs={fs} in {event.npz_path}"
                )

            n_samples = len(signal_data)
            if n_samples < _SAMPLES_PER_EPOCH:
                return []

            n_epochs = n_samples // _SAMPLES_PER_EPOCH
            epochs = signal_data[: n_epochs * _SAMPLES_PER_EPOCH].reshape(
                n_epochs, _SAMPLES_PER_EPOCH
            )

            for i in range(n_epochs):
                if len(samples) >= self.max_epochs:
                    break
                raw_label = int(stages[i * _SAMPLES_PER_EPOCH])
                mapped = self._label_map.get(raw_label)
                if mapped is None:
                    continue
                samples.append(
                    {
                        "patient_id": pid,
                        "signal": epochs[i],
                        "label": mapped,
                        "ahi": ahi,
                    }
                )

        return samples

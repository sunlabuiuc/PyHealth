"""Binary detection of one PTB-XL diagnostic superclass.

Authors:
    Bjorn Elvar (bthor5@illinois.edu), Rohan Vanjani

Paper:
    Aniruddh Raghu, Divya Shanmugam, Eugene Pomerantsev, John Guttag and
    Collin M. Stultz. "Data Augmentation for Electrocardiograms." CHIL 2022.

Paper link:
    https://proceedings.mlr.press/v174/raghu22a.html

Description:
    The paper's PTB-XL setup: one binary target per ECG (does it carry a
    statement of the chosen diagnostic superclass?) on a 250 Hz, 2496-sample
    signal. It is the task :class:`~pyhealth.models.TaskAug` trains on, since
    its augmentation strengths are indexed by a binary label.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from pyhealth.tasks.base_task import BaseTask


class PTBXLBinaryClassification(BaseTask):
    """Binary detection of one diagnostic superclass on PTB-XL ECGs.

    The label is 1 when any diagnostic SCP statement of the ECG maps to
    ``target`` in ``scp_statements.csv`` and 0 otherwise. Every ECG gives a
    sample: one with no diagnostic superclass at all (mainly pacemaker ECGs)
    is a negative, as in the reference loader of Raghu et al. (2022).

    Unlike :class:`~pyhealth.tasks.PTBXLSuperclassClassification` and
    :class:`~pyhealth.tasks.PTBXLMultilabelClassification`, which are
    multi-label at the dataset's own rate, this task has a ``"binary"`` label
    and resamples the signal. Build the dataset with
    ``PTBXLDataset(..., sampling_rate=500)`` to get the paper's 250 Hz input.

    The signal is not normalized: fit normalization statistics on your
    training split and apply them to all splits.

    Args:
        scp_statements_path: Path to ``scp_statements.csv`` from the PTB-XL
            download. Use ``dataset.scp_statements_path``.
        target: The diagnostic superclass to detect: ``"MI"``, ``"HYP"``,
            ``"STTC"``, ``"CD"`` or ``"NORM"``. Defaults to ``"MI"``.
        sampling_rate: Output sampling rate in Hz. It must divide the
            dataset's rate. The signal is decimated by keeping every n-th
            sample, without a low-pass filter, as in the reference loader.
            Defaults to 250.
        length: Number of samples kept from the start of the resampled signal;
            shorter signals are zero-padded at the end. ``None`` keeps the
            signal as is. Defaults to 2496, the reference loader's length (10 s
            at 250 Hz minus 4 samples).

    Attributes:
        task_name (str): ``"PTBXLBinaryClassification"``.
        input_schema (dict): ``{"signal": "tensor"}``.
        output_schema (dict): ``{"label": "binary"}``.

    Raises:
        ValueError: If ``target`` is not a diagnostic superclass, or if
            ``sampling_rate`` or ``length`` is not positive.

    Examples:
        >>> from pyhealth.datasets import PTBXLDataset
        >>> from pyhealth.tasks import PTBXLBinaryClassification
        >>> dataset = PTBXLDataset(  # doctest: +SKIP
        ...     root="/path/to/ptb-xl/1.0.3", sampling_rate=500
        ... )
        >>> task = PTBXLBinaryClassification(  # doctest: +SKIP
        ...     scp_statements_path=dataset.scp_statements_path, target="MI"
        ... )
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
        >>> samples[0]["signal"].shape  # doctest: +SKIP
        torch.Size([12, 2496])
    """

    task_name: str = "PTBXLBinaryClassification"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"label": "binary"}

    def __init__(
        self,
        scp_statements_path: str | os.PathLike,
        target: str = "MI",
        sampling_rate: int = 250,
        length: int | None = 2496,
    ) -> None:
        from pyhealth.tasks.ptbxl import PTBXL_DIAGNOSTIC_SUPERCLASSES

        if target not in PTBXL_DIAGNOSTIC_SUPERCLASSES:
            raise ValueError(
                f"target must be one of {PTBXL_DIAGNOSTIC_SUPERCLASSES}, got {target!r}"
            )
        if sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")
        if length is not None and length <= 0:
            raise ValueError(f"length must be positive or None, got {length}")
        # A canonical string keeps the set_task cache key stable.
        self.scp_statements_path = str(Path(scp_statements_path).resolve())
        self.target = target
        self.sampling_rate = sampling_rate
        self.length = length
        super().__init__()

    def _prepare_signal(self, signal_file: str, source_rate: int) -> np.ndarray:
        """Loads one ECG, decimates it and crops or pads it to ``length``.

        Args:
            signal_file: WFDB record path without extension.
            source_rate: Sampling rate of the stored record in Hz.

        Returns:
            Float32 array of shape ``(12, length)``.

        Raises:
            ValueError: If ``sampling_rate`` does not divide ``source_rate``.
        """
        from pyhealth.datasets.ptbxl import load_ptbxl_record

        step, remainder = divmod(source_rate, self.sampling_rate)
        if step < 1 or remainder:
            raise ValueError(
                f"Cannot decimate {source_rate} Hz records to "
                f"{self.sampling_rate} Hz: the output rate must divide the "
                "dataset's. For 250 Hz use PTBXLDataset(..., sampling_rate=500)."
            )
        signal = load_ptbxl_record(signal_file)[:, ::step]
        if self.length is not None:
            signal = signal[:, : self.length]
            missing = self.length - signal.shape[1]
            if missing > 0:
                signal = np.pad(signal, ((0, 0), (0, missing)))
        return np.ascontiguousarray(signal, dtype=np.float32)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds one sample per ECG of a patient.

        Args:
            patient: A patient with ``records`` events from
                :class:`~pyhealth.datasets.PTBXLDataset`.

        Returns:
            A list of dicts with ``patient_id``, ``record_id``, ``strat_fold``,
            ``signal`` (float32 array ``(12, length)``) and ``label`` (0 or 1).
        """
        from pyhealth.tasks.ptbxl import (
            aggregate_diagnostic_superclasses,
            load_diagnostic_class_map,
        )

        class_map = load_diagnostic_class_map(self.scp_statements_path)
        samples = []
        for event in patient.get_events(event_type="records"):
            superclasses = aggregate_diagnostic_superclasses(event.scp_codes, class_map)
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": str(event.record_id),
                    "strat_fold": int(float(event.strat_fold)),
                    "signal": self._prepare_signal(
                        event.signal_file, int(event.sampling_rate)
                    ),
                    "label": int(self.target in superclasses),
                }
            )
        return samples

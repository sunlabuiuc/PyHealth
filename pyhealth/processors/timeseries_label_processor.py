"""Labels given at time points: restricted :class:`SequenceProcessorV2` fields."""

from typing import ClassVar

import numpy as np
import torch

from . import register_processor
from .base_processor import FeatureProcessor
from .label_processor import (
    BinaryLabelProcessor,
    LabelProcessor,
    RegressionLabelProcessor,
)
from .sequence_processor_v2 import Records, SequenceProcessorV2


class TimeseriesLabelProcessor(SequenceProcessorV2, LabelProcessor):
    """Base processor for labels given at time points (a restricted V2 field).

    A :class:`SequenceProcessorV2` with one fixed ``series`` level of scalar
    values, so a label is a :class:`Records`: ``{"time": [...], "values":
    [[v], ...]}`` (a flat ``[v, ...]`` also works). Times are in the unit of
    the inputs' top level and must be strictly increasing: two labels at one
    time raise. A missing label is left out rather than given as NaN, which
    raises. There is no grid, resampling, imputation or mask.

    A model predicts one value per label, at its time, from the input records
    at or before it (see :class:`~pyhealth.models.RNN_V2`).

    Output:
        - ``(values, time)``: ``[N, 1]`` values and ``[N]`` times. A batch
          (:meth:`collate`) is packed: ``[ΣN, 1]`` values, ``[ΣN]`` times
          and the batch offsets ``[B + 1]``.

    The registered names ``"timeseries_binary"`` and
    ``"timeseries_regression"`` are model modes of their own, so modules that
    assume one prediction per sample (calibration, prediction sets,
    interpretability) do not treat these labels as ``"binary"`` or
    ``"regression"``.

    The activation and loss are those of the static counterpart
    (:attr:`base_cls`), applied to each label record: a model gives one
    logit per record, ``[ΣN, 1]``.

    Examples:
        >>> from pyhealth.processors import TimeseriesRegressionLabelProcessor
        >>> TimeseriesRegressionLabelProcessor().process({"time": [0.0, 3.0], "values": [2.5, 4.0]})
        (tensor([[2.5000],
                [4.0000]]), tensor([0., 3.]))
    """

    is_timeseries = True
    base_cls: ClassVar[type[LabelProcessor]]
    """The static label processor whose activation and loss apply per record."""

    def __init__(self) -> None:
        super().__init__(levels=[{"ordering": "series", "values": [1]}])
        # No tokens and no statistics: there is nothing to fit.
        self._fitted = True

    # FeatureProcessor's no-op, not SequenceProcessorV2's pass over every
    # sample, which would find nothing to learn.
    fit = FeatureProcessor.fit

    @property
    def base(self) -> LabelProcessor:
        """An instance of :attr:`base_cls`. Built on each use rather than
        stored, so it adds no state (pickles, cache keys) to the processor."""
        return self.base_cls()

    def activation(self, logits: torch.Tensor) -> torch.Tensor:
        return self.base.activation(logits)

    def loss(self, logits, y_true, pos_weight=None, **kwargs):
        """Averages the base loss over the label records of a batch.

        Args:
            logits: One prediction per label record, ``[ΣN, 1]``.
            y_true: The packed batch: ``(values [ΣN, 1], time [ΣN], batch
                offsets [B + 1])``.
        """
        self._check_pos_weight(pos_weight)
        values = y_true[0]
        if logits.shape != values.shape:
            raise ValueError(
                f"Expected one prediction per label record, {tuple(values.shape)}, "
                f"got {tuple(logits.shape)}. Per-step labels need a model that "
                "predicts at each label's time, such as RNN_V2."
            )
        if values.numel() == 0:
            # A batch without label records: a zero loss that keeps the graph,
            # instead of the NaN mean of an empty tensor.
            return logits.sum() * 0.0
        return self.base.loss(logits, values, **kwargs)

    def process(self, value: Records) -> tuple[torch.Tensor, ...]:
        if isinstance(value, dict) and "time" in value:
            time = np.asarray(value["time"], dtype=np.float64)
            if (np.diff(time) <= 0).any():
                raise ValueError(
                    "Label times must be strictly increasing: give at most one "
                    "label per time."
                )
        output = super().process(value)
        self._check(output[0])
        return output

    def _check(self, values: torch.Tensor) -> None:
        pass

    def size(self) -> int:
        """One output per label."""
        return 1

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


@register_processor("timeseries_binary")
class TimeseriesBinaryLabelProcessor(TimeseriesLabelProcessor):
    """Binary (0/1) labels at time points, see :class:`TimeseriesLabelProcessor`.

    Examples:
        >>> from pyhealth.processors import TimeseriesBinaryLabelProcessor
        >>> TimeseriesBinaryLabelProcessor().process({"time": [0.0, 1.0], "values": [0, 1]})
        (tensor([[0.],
                [1.]]), tensor([0., 1.]))
    """

    mode = "timeseries_binary"
    base_cls = BinaryLabelProcessor

    def _check(self, values: torch.Tensor) -> None:
        if not torch.isin(values, torch.tensor([0.0, 1.0])).all():
            raise ValueError("Binary labels must be 0 or 1.")


@register_processor("timeseries_regression")
class TimeseriesRegressionLabelProcessor(TimeseriesLabelProcessor):
    """Real-valued labels at time points, see :class:`TimeseriesLabelProcessor`.

    Examples:
        >>> from pyhealth.processors import TimeseriesRegressionLabelProcessor
        >>> TimeseriesRegressionLabelProcessor().process({"time": [0.0, 2.0], "values": [2.5, 0.5]})
        (tensor([[2.5000],
                [0.5000]]), tensor([0., 2.]))
    """

    mode = "timeseries_regression"
    base_cls = RegressionLabelProcessor

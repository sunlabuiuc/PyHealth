"""Labels given at time points: one label per (sample, time)."""

from collections.abc import Hashable, Iterable, Sequence
from itertools import pairwise
from typing import Any, ClassVar

import numpy as np
import torch

from . import register_processor
from .label_processor import (
    BinaryLabelProcessor,
    LabelProcessor,
    MultiClassLabelProcessor,
    MultiLabelProcessor,
    PosWeight,
    RegressionLabelProcessor,
)
from .utils import Normalization


class TimeseriesLabelProcessor[B: LabelProcessor](LabelProcessor):
    """Base class of labels given at time points.

    A label is ``{"time": [t, ...], "values": [v, ...]}``: one value per time.
    Times are in the unit of the inputs' top level and must be strictly
    increasing, so there is at most one label per time. A missing label is
    left out rather than given as NaN. Each value is a label of the static
    counterpart, :attr:`base_cls` (a scalar may also be given as ``[v]``),
    which also gives the activation and loss of each label record, and is
    fitted on the values of all records.

    A model predicts one output per label record, at its time, from the inputs
    at or before it (see :class:`~pyhealth.models.RNN_V2`).

    Output:
        - ``(values, time)``: one row of ``values`` per label record (the
          static counterpart's output) and the ``[N]`` times. A batch
          (:meth:`collate`) is packed: ``(values, time [ΣN], offsets
          [B + 1])``, where sample ``b`` has records ``offsets[b]`` to
          ``offsets[b + 1]``.

    The registered names (``"timeseries_binary"``, ...) are modes of their
    own, so modules that assume one prediction per sample (calibration,
    prediction sets, interpretability) do not treat these labels as their
    static counterparts.

    Examples:
        >>> from pyhealth.processors import TimeseriesRegressionLabelProcessor
        >>> TimeseriesRegressionLabelProcessor().process({"time": [0.0, 3.0], "values": [2.5, 4.0]})
        (tensor([[2.5000],
                [4.0000]]), tensor([0., 3.]))
    """

    base_cls: ClassVar[type[LabelProcessor]]
    """The class of :attr:`base`, the static label processor of each record."""
    _scalar_values: ClassVar[bool] = True
    """Whether each value is one label (as opposed to a set of labels)."""
    _index_values: ClassVar[bool] = False
    """Whether :attr:`base` outputs a class index (a ``long`` scalar) per value,
    rather than a float vector of :meth:`size`."""

    def __init__(self, base: B) -> None:
        super().__init__()
        self.base: B = base
        """The static label processor of each record, fitted on all records."""
        self.learns_statistics = base.learns_statistics
        self.need_fit_test = base.need_fit_test

    # ------------------------------------------------------------------
    # FeatureProcessor
    # ------------------------------------------------------------------
    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        """Fits :attr:`base` on the values of every label record."""
        self.base.fit(self._records(samples, field), field)

    def fit_test(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        """Adds the labels of every label record to :attr:`base`, see
        :meth:`LabelProcessor.fit_test`."""
        if self.need_fit_test:
            self.base.fit_test(self._records(samples, field), field)

    def _records(self, samples: Iterable[dict[str, Any]], field: str) -> list[dict]:
        """One sample per label record, holding its value under ``field``."""
        return [
            {field: self._value(value)}
            for sample in samples
            for value in sample[field]["values"]
        ]

    def process(self, value: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(value, dict) or "time" not in value or "values" not in value:
            raise ValueError(
                'A time-series label is {"time": [...], "values": [...]}, '
                f"got {value!r}."
            )
        time = torch.as_tensor(np.asarray(value["time"], dtype=np.float64)).float()
        raw = list(value["values"])
        if time.dim() != 1 or len(time) != len(raw):
            raise ValueError(
                f"Expected one label value per time, got {len(raw)} values for "
                f"times of shape {tuple(time.shape)}."
            )
        if bool((time.diff() <= 0).any()):
            raise ValueError(
                "Label times must be strictly increasing: give at most one "
                "label per time."
            )
        if raw:
            values = torch.stack([self.base.process(self._value(v)) for v in raw])
        else:
            values = torch.zeros((0, *self._value_shape()), dtype=self._dtype())
        if values.is_floating_point() and bool(values.isnan().any()):
            raise ValueError("Leave a missing label out instead of giving NaN.")
        return values, time

    def collate(
        self, items: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Packs a batch: values and times of all samples, and the offsets."""
        if not items:
            raise ValueError("collate needs at least one sample.")
        counts = torch.tensor([len(time) for _, time in items])
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
        values = torch.cat([values for values, _ in items])
        time = torch.cat([time for _, time in items])
        return values, time, offsets

    def size(self) -> int:
        return self.base.size()

    def __repr__(self) -> str:
        inner = repr(self.base)
        return type(self).__name__ + inner[inner.index("(") :]

    # ------------------------------------------------------------------
    # LabelProcessor
    # ------------------------------------------------------------------
    def activation(self, logits: torch.Tensor) -> torch.Tensor:
        return self.base.activation(logits)

    def loss(self, logits: torch.Tensor, y_true: Any, **kwargs: Any) -> torch.Tensor:
        """The static loss over the label records of a batch.

        Args:
            logits: One prediction per label record, ``[ΣN, output_size()]``.
            y_true: The packed batch, ``(values, time, offsets)``.
        """
        values = y_true[0]
        expected = (values.shape[0], self.output_size())
        if tuple(logits.shape) != expected:
            raise ValueError(
                f"Expected one prediction per label record, {expected}, got "
                f"{tuple(logits.shape)}. Per-step labels need a model that "
                "predicts at each label's time, such as RNN_V2."
            )
        if values.shape[0] == 0:
            # A batch without label records: a zero loss that keeps the graph,
            # instead of the NaN mean of an empty tensor.
            return logits.sum() * 0.0
        return self.base.loss(logits, values, **kwargs)

    def to_numpy(
        self, y_true: Any, y_prob: torch.Tensor
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """One array of labels and one of predictions per sample."""
        (values,), (prob,) = self.base.to_numpy(y_true[0], y_prob)
        offsets = y_true[2].detach().cpu().numpy()
        if self._scalar_values and values.ndim == 2:
            values, prob = values[:, 0], prob[:, 0]
        true_parts, prob_parts = [], []
        for lo, hi in pairwise(offsets):
            true_parts.append(values[lo:hi])
            prob_parts.append(prob[lo:hi])
        return true_parts, prob_parts

    def gather(
        self, y_true: list[np.ndarray], y_prob: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Keeps one array per sample, over its label records."""
        return y_true, y_prob

    @classmethod
    def metrics_fn(cls, y_true, y_prob, metrics=None):
        """Pools the label records of all samples, then scores them as
        :attr:`base_cls` does, as in benchmarks that evaluate every hour of
        every ICU stay (e.g. YAIB)."""
        return cls.base_cls.metrics_fn(
            np.concatenate(list(y_true), axis=0),
            np.concatenate(list(y_prob), axis=0),
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _value(self, value: Any) -> Any:
        """A scalar label given as ``[v]`` is ``v``."""
        if not self._scalar_values or not isinstance(value, (list, tuple, np.ndarray)):
            return value
        if len(value) != 1:
            raise ValueError(f"Expected one label value per time, got {value!r}.")
        return value[0]

    def _value_shape(self) -> tuple[int, ...]:
        return () if self._index_values else (self.base.size(),)

    def _dtype(self) -> torch.dtype:
        return torch.long if self._index_values else torch.float32


@register_processor("timeseries_binary")
class TimeseriesBinaryLabelProcessor(TimeseriesLabelProcessor[BinaryLabelProcessor]):
    """Binary (0/1) labels at time points, see :class:`TimeseriesLabelProcessor`.

    Args:
        pos_weight: As in :class:`BinaryLabelProcessor`.

    Examples:
        >>> from pyhealth.processors import TimeseriesBinaryLabelProcessor
        >>> TimeseriesBinaryLabelProcessor().process({"time": [0.0, 1.0], "values": [0, 1]})
        (tensor([[0.],
                [1.]]), tensor([0., 1.]))
    """

    mode = "timeseries_binary"
    base_cls = BinaryLabelProcessor

    def __init__(self, pos_weight: float | str | None = None) -> None:
        super().__init__(BinaryLabelProcessor(pos_weight=pos_weight))
        # The labels are 0 and 1: there is nothing to learn but the weight.
        self.base.label_vocab = {0: 0, 1: 1}

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        if self.base.pos_weight == "balanced":
            self.base._fit_pos_weight(
                [
                    self.base.process(self._value(value))
                    for sample in samples
                    for value in sample[field]["values"]
                ]
            )


@register_processor("timeseries_multiclass")
class TimeseriesMultiClassLabelProcessor(
    TimeseriesLabelProcessor[MultiClassLabelProcessor]
):
    """One class per time point, see :class:`TimeseriesLabelProcessor`.

    Args:
        labels: As in :class:`MultiClassLabelProcessor`.

    Examples:
        >>> from pyhealth.processors import TimeseriesMultiClassLabelProcessor
        >>> p = TimeseriesMultiClassLabelProcessor(labels=["awake", "light", "deep"])
        >>> p.process({"time": [0.0, 30.0], "values": ["light", "deep"]})
        (tensor([1, 2]), tensor([ 0., 30.]))
    """

    mode = "timeseries_multiclass"
    base_cls = MultiClassLabelProcessor
    _index_values = True

    def __init__(self, labels: int | Sequence[Hashable] | None = None) -> None:
        super().__init__(MultiClassLabelProcessor(labels=labels))


@register_processor("timeseries_multilabel")
class TimeseriesMultiLabelProcessor(TimeseriesLabelProcessor[MultiLabelProcessor]):
    """A set of labels per time point, see :class:`TimeseriesLabelProcessor`.

    Args:
        labels: As in :class:`MultiLabelProcessor`.
        pos_weight: As in :class:`MultiLabelProcessor`.

    Examples:
        >>> from pyhealth.processors import TimeseriesMultiLabelProcessor
        >>> p = TimeseriesMultiLabelProcessor(labels=["a", "b"])
        >>> p.process({"time": [0.0, 1.0], "values": [["a"], ["a", "b"]]})[0]
        tensor([[1., 0.],
                [1., 1.]])
    """

    mode = "timeseries_multilabel"
    base_cls = MultiLabelProcessor
    _scalar_values = False

    def __init__(
        self,
        labels: int | Sequence[Hashable] | None = None,
        pos_weight: PosWeight = None,
    ) -> None:
        super().__init__(MultiLabelProcessor(labels=labels, pos_weight=pos_weight))


@register_processor("timeseries_regression")
class TimeseriesRegressionLabelProcessor(
    TimeseriesLabelProcessor[RegressionLabelProcessor]
):
    """Real-valued labels at time points, see :class:`TimeseriesLabelProcessor`.

    Args:
        normalization: As in :class:`RegressionLabelProcessor`; a learned
            one is fitted on the label records of all fitted samples.

    Examples:
        >>> from pyhealth.processors import TimeseriesRegressionLabelProcessor
        >>> TimeseriesRegressionLabelProcessor().process({"time": [0.0, 2.0], "values": [2.5, 0.5]})
        (tensor([[2.5000],
                [0.5000]]), tensor([0., 2.]))
        >>> p = TimeseriesRegressionLabelProcessor(normalization=(0, 10))
        >>> p.base.center, p.base.scale
        (0.0, 10.0)
    """

    mode = "timeseries_regression"
    base_cls = RegressionLabelProcessor

    def __init__(self, normalization: Normalization | Sequence[float] = None) -> None:
        super().__init__(RegressionLabelProcessor(normalization=normalization))

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        if self.base.learns_statistics:  # otherwise, no need to read the samples
            super().fit(samples, field)

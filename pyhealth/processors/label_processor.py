"""Output processors: what a label is, and how to train on and score it."""

import logging
import math
from numbers import Real
from abc import abstractmethod
from collections.abc import Hashable, Iterable, Sequence
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F

from . import register_processor
from .base_processor import FeatureProcessor
from .utils import NORMALIZATIONS, Normalization, fit_normalization

logger = logging.getLogger(__name__)

PosWeight = float | Sequence[float] | str | None
"""A positive-class weight: a number, one number per label, ``"balanced"``
(negatives / positives, learned in ``fit``) or None."""


class LabelProcessor(FeatureProcessor):
    """Base class of output processors: a label and how to train on it.

    A label processor turns a sample's raw label into a tensor
    (:meth:`process`), like any :class:`FeatureProcessor`, and also defines
    everything that depends on the kind of label:

    - :attr:`mode`: the label kind, e.g. ``"binary"``.
    - :meth:`output_size`: the number of model outputs.
    - :meth:`activation`: logits to predictions (``y_prob``).
    - :meth:`loss`: the training loss of logits against processed labels.
    - :meth:`to_numpy` and :meth:`gather`: model outputs to the arrays that
      :meth:`metrics_fn` takes, used by :class:`~pyhealth.trainer.Trainer`.
    - :meth:`metrics_fn`: the metrics of this label kind.

    Models read it from :attr:`~pyhealth.models.BaseModel.label_processor`.

    Attributes:
        mode: The label kind. It is also the name the processor is
            registered under, so ``output_schema={"y": "binary"}`` selects
            :class:`BinaryLabelProcessor`.

    Examples:
        >>> import torch
        >>> from pyhealth.processors import BinaryLabelProcessor
        >>> p = BinaryLabelProcessor()
        >>> p.mode
        'binary'
        >>> p.activation(torch.tensor([[0.0]]))
        tensor([[0.5000]])
        >>> p.loss(torch.tensor([[0.0]]), torch.tensor([[1.0]]))
        tensor(0.6931)
    """

    mode: ClassVar[str]
    stores_tensor = True
    need_fit_test: bool = False
    """Whether :meth:`fit_test` has anything to learn: True for a label
    vocabulary learned from the data (``labels`` not given)."""

    def fit_test(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        """Adds the labels that only occur outside the training samples.

        This is the only fitting step that sees validation and test samples.
        It exists because a label vocabulary (the classes, and so
        :meth:`size` and the model's output layer) has to cover the test
        labels too: a class seen only in validation or test, common for rare
        drugs in drug recommendation, would otherwise have no output and fail
        in :meth:`process`.

        Call it after :meth:`fit`: :meth:`fit` learns from the training
        samples, then this adds the labels of all samples to the vocabulary.
        :class:`~pyhealth.datasets.sample_dataset.SampleBuilder` does this for
        every processor with :attr:`need_fit_test` when it is given a split;
        without one, ``fit`` already saw every label. Calling it again, or on
        the samples ``fit`` saw, changes nothing.

        .. warning::

           **Learn only which label values exist here, nothing else.**
           Anything computed from the label values (label frequencies, class
           weights such as ``pos_weight="balanced"``, means, scales, ...)
           must be learned in :meth:`fit`, from the training samples only.
           Learning it here leaks test labels into training.

        The default learns nothing: labels without a vocabulary to learn
        (binary, regression, given ``labels``) need none.

        Args:
            samples: Samples of every split, test included.
            field: The label field.

        Examples:
            >>> from pyhealth.processors import MultiClassLabelProcessor
            >>> p = MultiClassLabelProcessor()
            >>> p.fit([{"y": "a"}, {"y": "b"}], "y")  # training split
            >>> p.fit_test([{"y": "a"}, {"y": "b"}, {"y": "c"}], "y")  # all splits
            >>> p.label_vocab
            {'a': 0, 'b': 1, 'c': 2}
        """

    @abstractmethod
    def size(self) -> int:
        """The width of the processed label: 1 for binary and regression, the
        number of classes or labels otherwise."""

    def output_size(self) -> int:
        """The number of model outputs (logits) for this label.

        Models size their output layer with it. It is :meth:`size` for every
        label processor so far; a label whose logits differ from its width
        overrides it.
        """
        return self.size()

    @abstractmethod
    def activation(self, logits: torch.Tensor) -> torch.Tensor:
        """Turns logits into predictions (``y_prob``)."""

    @abstractmethod
    def loss(self, logits: torch.Tensor, y_true: Any, **kwargs: Any) -> torch.Tensor:
        """The training loss of ``logits`` against a batch of processed labels.

        Args:
            logits: The model's logits.
            y_true: A batch of this processor's outputs.
            **kwargs: Passed to the underlying torch loss.
        """

    def to_numpy(
        self, y_true: Any, y_prob: torch.Tensor
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """One batch of labels and predictions, as one array per part.

        :meth:`gather` joins the parts of all batches. A label given once per
        sample has one part per batch; a label given at time points has one
        part per sample.
        """
        return [_numpy(y_true)], [_numpy(y_prob)]

    def gather(
        self, y_true: list[np.ndarray], y_prob: list[np.ndarray]
    ) -> tuple[Any, Any]:
        """Joins the parts of :meth:`to_numpy` into what :meth:`metrics_fn` takes."""
        return np.concatenate(y_true, axis=0), np.concatenate(y_prob, axis=0)

    @classmethod
    @abstractmethod
    def metrics_fn(
        cls, y_true: Any, y_prob: Any, metrics: list[str] | None = None
    ) -> dict[str, float]:
        """Scores the gathered labels and predictions of this label kind."""

    def __repr__(self) -> str:
        vocab = getattr(self, "label_vocab", None)
        if vocab is None:
            return f"{type(self).__name__}()"
        return f"{type(self).__name__}(label_vocab_size={len(vocab)})"


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _fixed_vocab(labels: int | Sequence[Hashable]) -> dict[Any, int]:
    """Index of each given label, in order; ``K`` means ``range(K)``."""
    labels = list(range(labels)) if isinstance(labels, int) else list(labels)
    if not labels:
        raise ValueError("labels must not be empty.")
    if len(set(labels)) != len(labels):
        raise ValueError(f"labels must be distinct, got {labels}.")
    return {label: i for i, label in enumerate(labels)}


def _learned_vocab(seen: set) -> dict[Any, int]:
    """``range(K)`` labels keep their values; any other set is sorted."""
    if seen == set(range(len(seen))):
        return {i: i for i in range(len(seen))}
    return {label: i for i, label in enumerate(sorted(seen))}


def _add_labels(processor: Any, seen: set, field: str) -> dict[Any, int] | None:
    """Rebuilds ``label_vocab`` over its labels and ``seen``, in the order of
    :func:`_learned_vocab`, so the indices do not depend on the split.

    Returns the old vocabulary when labels were added, else None.
    """
    old = processor.label_vocab
    new = seen - old.keys()
    if not new:
        return None
    processor.label_vocab = _learned_vocab(set(old) | seen)
    logger.info(
        f"Label {field}: added {len(new)} labels that occur only outside the "
        f"fitted samples; vocab: {processor.label_vocab}"
    )
    return old


def _index(vocab: dict[Any, int], label: Any) -> int:
    try:
        return vocab[label]
    except (KeyError, TypeError):
        raise ValueError(f"Unknown label {label!r}; expected one of {list(vocab)}.")


class _WeightedBCE:
    """Shared by the processors whose loss is BCE: a ``pos_weight`` option.

    ``pos_weight`` is part of the task: a number (binary), one number per
    label (multilabel), or ``"balanced"`` for negatives / positives over the
    samples the processor is fitted on. ``"balanced"`` learns from the labels,
    so PyHealth warns unless the processors are fitted on a training split.
    """

    pos_weight: PosWeight = None
    _weight: torch.Tensor | None = None

    def _init_pos_weight(self, pos_weight: PosWeight) -> None:
        if isinstance(pos_weight, str) and pos_weight != "balanced":
            raise ValueError(
                "pos_weight must be a number, a sequence, 'balanced' or None, "
                f"not {pos_weight!r}."
            )
        self.pos_weight = pos_weight
        self.learns_statistics = pos_weight == "balanced"
        self._weight = (
            None
            if pos_weight is None or isinstance(pos_weight, str)
            else torch.as_tensor(pos_weight, dtype=torch.float32).reshape(-1)
        )

    def _fit_pos_weight(self, targets: list[torch.Tensor]) -> None:
        """Sets the ``"balanced"`` weight from processed labels."""
        if not targets:
            raise ValueError("pos_weight='balanced' got no samples to fit on.")
        stacked = torch.stack([t.reshape(-1) for t in targets])
        positives = stacked.sum(0)
        if bool((positives == 0).any()):
            logger.warning(
                "pos_weight='balanced': some labels have no positive examples; "
                "their weight is set to 1."
            )
        self._weight = torch.where(
            positives > 0, (len(stacked) - positives) / positives.clamp(min=1), 1.0
        )

    def _remap_pos_weight(self, old_vocab: dict[Any, int]) -> None:
        """Moves a ``"balanced"`` weight to the indices of a grown vocabulary.

        Labels new to the vocabulary had no positives in the fitted samples,
        so their weight is 1, as for any label without positives.
        """
        if self.pos_weight != "balanced" or self._weight is None:
            return
        vocab = self.label_vocab  # type: ignore[attr-defined]
        weight = torch.ones(len(vocab))
        for label, i in old_vocab.items():
            weight[vocab[label]] = self._weight[i]
        self._weight = weight
        logger.warning(
            "pos_weight='balanced': %d labels occur only outside the fitted "
            "(training) samples; their weight is set to 1.",
            len(vocab) - len(old_vocab),
        )

    @property
    def weight(self) -> torch.Tensor | None:
        """The positive-class weight the loss uses, or None."""
        if self.pos_weight == "balanced" and self._weight is None:
            raise ValueError("pos_weight='balanced' is learned in fit(); fit first.")
        return self._weight

    def _bce(self, logits: torch.Tensor, y_true: torch.Tensor, **kwargs: Any):
        weight = self.weight
        if weight is not None:
            kwargs.setdefault("pos_weight", weight.to(logits.device))
        return F.binary_cross_entropy_with_logits(logits, y_true, **kwargs)


@register_processor("binary")
class BinaryLabelProcessor(_WeightedBCE, LabelProcessor):
    """Binary classification labels.

    Two label values, mapped to 0 and 1: ``{0, 1}`` and ``{False, True}`` as
    is, any other pair in sorted order. The model has one output; the loss is
    binary cross-entropy with logits.

    Args:
        pos_weight: Weight of positive examples in the loss, for rare
            outcomes: a number, ``"balanced"`` (negatives / positives in the
            fitted samples) or None. Weighting changes the scale of the
            predicted probabilities, so check calibration when you use it.

    Examples:
        >>> from pyhealth.processors import BinaryLabelProcessor
        >>> p = BinaryLabelProcessor()
        >>> p.fit([{"y": 0}, {"y": 1}], "y")
        >>> p.process(1)
        tensor([1.])
        >>> p = BinaryLabelProcessor(pos_weight="balanced")
        >>> p.fit([{"y": 0}, {"y": 0}, {"y": 0}, {"y": 1}], "y")
        >>> p.weight
        tensor([3.])
    """

    mode = "binary"

    def __init__(self, pos_weight: float | str | None = None):
        super().__init__()
        self.label_vocab: dict[Any, int] = {}
        self._init_pos_weight(pos_weight)

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        values = [sample[field] for sample in samples]
        all_labels = set(values)
        if len(all_labels) != 2:
            raise ValueError(f"Expected 2 unique labels, got {len(all_labels)}")
        if all_labels == {0, 1}:
            self.label_vocab = {0: 0, 1: 1}
        elif all_labels == {False, True}:
            self.label_vocab = {False: 0, True: 1}
        else:
            self.label_vocab = {label: i for i, label in enumerate(sorted(all_labels))}
        logger.info(f"Label {field} vocab: {self.label_vocab}")
        if self.pos_weight == "balanced":
            self._fit_pos_weight([self.process(value) for value in values])

    def process(self, value: Any) -> torch.Tensor:
        return torch.tensor([_index(self.label_vocab, value)], dtype=torch.float32)

    def size(self) -> int:
        return 1

    def activation(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    def loss(self, logits: torch.Tensor, y_true: Any, **kwargs: Any) -> torch.Tensor:
        return self._bce(logits, y_true, **kwargs)

    @classmethod
    def metrics_fn(cls, y_true, y_prob, metrics=None):
        from pyhealth.metrics.binary import binary_metrics_fn

        return binary_metrics_fn(y_true, y_prob, metrics=metrics)


@register_processor("multiclass")
class MultiClassLabelProcessor(LabelProcessor):
    """Multi-class classification labels: one class per sample.

    The model has one output per class; the loss is cross-entropy.

    Args:
        labels: The classes in index order, or their number ``K`` for classes
            already ``0..K-1``. When given, the classes (and :meth:`size`) are
            fixed, so a class missing from the fitted samples still gets an
            output, and ``fit`` only checks that every label is one of them.
            When omitted, the classes are learned in ``fit``: ``0..K-1`` as is,
            any other set in sorted order.

    Examples:
        >>> from pyhealth.processors import MultiClassLabelProcessor
        >>> p = MultiClassLabelProcessor()
        >>> p.fit([{"y": "a"}, {"y": "b"}, {"y": "c"}], "y")
        >>> p.process("b")
        tensor(1)
        >>> MultiClassLabelProcessor(labels=["low", "mid", "high"]).size()
        3
    """

    mode = "multiclass"
    labels: int | Sequence[Hashable] | None = None

    def __init__(self, labels: int | Sequence[Hashable] | None = None):
        super().__init__()
        self.labels = labels
        self.label_vocab: dict[Any, int] = {} if labels is None else _fixed_vocab(labels)
        self.need_fit_test = labels is None

    def fit_test(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        """Adds the classes of all samples, see :meth:`LabelProcessor.fit_test`."""
        if self.need_fit_test:
            _add_labels(self, {sample[field] for sample in samples}, field)

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        seen = {sample[field] for sample in samples}
        if self.labels is None:
            self.label_vocab = _learned_vocab(seen)
        else:
            for label in seen:
                _index(self.label_vocab, label)
        logger.info(f"Label {field} vocab: {self.label_vocab}")

    def process(self, value: Any) -> torch.Tensor:
        return torch.tensor(_index(self.label_vocab, value), dtype=torch.long)

    def size(self) -> int:
        return len(self.label_vocab)

    def activation(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits, dim=-1)

    def loss(self, logits: torch.Tensor, y_true: Any, **kwargs: Any) -> torch.Tensor:
        return F.cross_entropy(logits, y_true, **kwargs)

    @classmethod
    def metrics_fn(cls, y_true, y_prob, metrics=None):
        from pyhealth.metrics.multiclass import multiclass_metrics_fn

        return multiclass_metrics_fn(y_true, y_prob, metrics=metrics)


@register_processor("multilabel")
class MultiLabelProcessor(_WeightedBCE, LabelProcessor):
    """Multi-label classification labels: a set of labels per sample.

    The model has one output per label; the loss is binary cross-entropy with
    logits, per label.

    Args:
        labels: The labels in index order, or their number ``K`` for labels
            already ``0..K-1``. When given, the labels (and :meth:`size`) are
            fixed, and ``fit`` only checks that every label is one of them.
            When omitted, they are learned in ``fit``: ``0..K-1`` as is, any
            other set in sorted order.
        pos_weight: Weight of positive examples in the loss: a number for all
            labels, one number per label (in the order of ``labels``, which
            must then be given), ``"balanced"`` (negatives / positives of
            each label in the fitted samples) or None.

    Examples:
        >>> from pyhealth.processors import MultiLabelProcessor
        >>> p = MultiLabelProcessor()
        >>> p.fit([{"y": ["a"]}, {"y": ["a", "b"]}], "y")
        >>> p.process(["b"])
        tensor([0., 1.])
    """

    mode = "multilabel"
    labels: int | Sequence[Hashable] | None = None

    def __init__(
        self,
        labels: int | Sequence[Hashable] | None = None,
        pos_weight: PosWeight = None,
    ):
        super().__init__()
        self.labels = labels
        self.label_vocab: dict[Any, int] = {} if labels is None else _fixed_vocab(labels)
        self.need_fit_test = labels is None
        self._init_pos_weight(pos_weight)
        self._check_per_label_weight()

    def _check_per_label_weight(self) -> None:
        """A per-label ``pos_weight`` needs given ``labels``, one weight each.

        Weights are matched to labels by position, and a learned vocabulary's
        order is only known after fitting, so without ``labels`` a user
        cannot know which weight goes to which label.
        """
        pos_weight = self.pos_weight
        if pos_weight is None or isinstance(pos_weight, str):
            return
        if isinstance(pos_weight, Real) or (
            isinstance(pos_weight, torch.Tensor) and pos_weight.dim() == 0
        ):
            return  # one number for every label
        if self.labels is None:
            raise ValueError(
                "A per-label pos_weight needs labels=[...], which fixes the "
                "label order the weights are matched to; without labels, the "
                "order is learned from the data. Pass labels, or use one "
                "number or 'balanced' for every label."
            )
        weight = self._weight
        if weight is not None and len(weight) != len(self.label_vocab):
            raise ValueError(
                f"pos_weight has {len(weight)} weights for "
                f"{len(self.label_vocab)} labels; give one weight per label."
            )

    def fit_test(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        """Adds the labels of all samples, see :meth:`LabelProcessor.fit_test`.

        A ``"balanced"`` weight keeps its training values; added labels get 1.
        """
        if not self.need_fit_test:
            return
        seen = {label for sample in samples for label in sample[field]}
        old_vocab = _add_labels(self, seen, field)
        if old_vocab is not None:
            self._remap_pos_weight(old_vocab)

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        values = [sample[field] for sample in samples]
        seen = {label for value in values for label in value}
        if self.labels is None:
            self.label_vocab = _learned_vocab(seen)
        else:
            for label in seen:
                _index(self.label_vocab, label)
        logger.info(f"Label {field} vocab: {self.label_vocab}")
        if self.pos_weight == "balanced":
            self._fit_pos_weight([self.process(value) for value in values])

    def process(self, value: Any) -> torch.Tensor:
        if not isinstance(value, list):
            raise ValueError("Expected a list of labels for multilabel task.")
        target = torch.zeros(len(self.label_vocab), dtype=torch.float32)
        for label in value:
            target[_index(self.label_vocab, label)] = 1.0
        return target

    def size(self) -> int:
        return len(self.label_vocab)

    def activation(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    def loss(self, logits: torch.Tensor, y_true: Any, **kwargs: Any) -> torch.Tensor:
        return self._bce(logits, y_true, **kwargs)

    @classmethod
    def metrics_fn(cls, y_true, y_prob, metrics=None, **kwargs):
        """Scores with :func:`~pyhealth.metrics.multilabel_metrics_fn`.

        Extra keyword arguments (e.g. ``ddi_adj`` for the ``"ddi"`` metric)
        are passed through.

        Examples:
            >>> import numpy as np
            >>> MultiLabelProcessor.metrics_fn(
            ...     np.array([[1, 0]]), np.array([[0.9, 0.2]]), metrics=["f1_micro"]
            ... )
            {'f1_micro': 1.0}
        """
        from pyhealth.metrics.multilabel import multilabel_metrics_fn

        return multilabel_metrics_fn(y_true, y_prob, metrics=metrics, **kwargs)


@register_processor("regression")
class RegressionLabelProcessor(LabelProcessor):
    """Real-valued labels. The model has one output; the loss is MSE.

    Labels in large units (e.g. hours) can stall training, so
    ``normalization`` rescales them to ``(label - center) / scale``, as
    :class:`~pyhealth.processors.TensorProcessor` does for inputs.
    :meth:`process` is the only place that does: like the index of a class,
    the processed label is what the model sees, so ``y_true``, logits,
    ``y_prob`` and the loss, including a model's own loss, are all in
    normalized units. :meth:`to_numpy`, which turns model outputs into metric
    inputs, maps ``y_true`` and ``y_prob`` back together, so
    :meth:`~pyhealth.trainer.Trainer.inference` and metrics are in the
    original units.

    Args:
        normalization: ``"minmax"`` (to 0..1 on the fitted range) or
            ``"zscore"`` (zero mean, unit variance), learned in ``fit`` (so
            fit on a training split); a fixed ``(low, high)`` range, mapped to
            0..1; or None. Labels are never clipped.

    Attributes:
        center: The label that maps to 0, or None without normalization.
        scale: The label distance that maps to 1, or None without
            normalization.

    Examples:
        >>> from pyhealth.processors import RegressionLabelProcessor
        >>> RegressionLabelProcessor().process(2.5)
        tensor([2.5000])
        >>> p = RegressionLabelProcessor(normalization="minmax")
        >>> p.fit([{"y": 10.0}, {"y": 30.0}], "y")
        >>> p.center, p.scale
        (10.0, 20.0)
        >>> y_true = p.process(20.0)
        >>> y_true
        tensor([0.5000])
        >>> p.to_numpy(y_true, y_true + 0.25)
        ([array([20.], dtype=float32)], [array([25.], dtype=float32)])
    """

    mode = "regression"
    normalization: Normalization | Sequence[float] = None
    center: float | None = None
    scale: float | None = None

    def __init__(self, normalization: Normalization | Sequence[float] = None):
        super().__init__()
        if normalization is None or isinstance(normalization, str):
            if normalization not in NORMALIZATIONS:
                raise ValueError(
                    f"normalization must be one of {NORMALIZATIONS} or "
                    f"(low, high), not {normalization!r}."
                )
        else:
            low, high = (float(end) for end in normalization)
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError(
                    f"A fixed range needs finite low < high, got {normalization!r}."
                )
            self.center, self.scale = low, high - low
        self.normalization = normalization
        self.learns_statistics = isinstance(normalization, str)

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        if not isinstance(self.normalization, str):
            return
        values = np.array([float(sample[field]) for sample in samples]).reshape(-1, 1)
        if values.size == 0:
            raise ValueError(
                f"normalization={self.normalization!r} got no samples to fit on."
            )
        (self.center,), (self.scale,) = fit_normalization([values], self.normalization)
        if np.nanmin(values) == np.nanmax(values):
            logger.warning("Label %s is constant; its scale is set to 1.", field)
        logger.info(f"Label {field} center {self.center}, scale {self.scale}")

    def _center_scale(self) -> tuple[float, float] | None:
        if self.normalization is None:
            return None
        if self.center is None or self.scale is None:
            raise ValueError(
                f"normalization={self.normalization!r} is learned in fit(); "
                "fit first."
            )
        return self.center, self.scale

    def process(self, value: Any) -> torch.Tensor:
        value = float(value)
        if (center_scale := self._center_scale()) is not None:
            center, scale = center_scale
            value = (value - center) / scale
        return torch.tensor([value], dtype=torch.float32)

    def size(self) -> int:
        return 1

    def activation(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    def loss(self, logits: torch.Tensor, y_true: Any, **kwargs: Any) -> torch.Tensor:
        return F.mse_loss(logits, y_true, **kwargs)

    def to_numpy(
        self, y_true: Any, y_prob: torch.Tensor
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Labels and predictions in the original units."""
        (true,), (prob,) = super().to_numpy(y_true, y_prob)
        if (center_scale := self._center_scale()) is None:
            return [true], [prob]
        center, scale = center_scale
        return [true * scale + center], [prob * scale + center]

    @classmethod
    def metrics_fn(cls, y_true, y_prob, metrics=None):
        from pyhealth.metrics.regression import regression_metrics_fn

        return regression_metrics_fn(y_true, y_prob, metrics=metrics)

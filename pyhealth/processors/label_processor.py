"""Output processors: what a label is, and how to train on and score it."""

import logging
from abc import abstractmethod
from collections.abc import Hashable, Iterable, Sequence
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F

from . import register_processor
from .base_processor import FeatureProcessor

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

    def schema(self) -> tuple[str, ...]:
        return ("value",)

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

    def is_token(self) -> bool:
        """Binary labels are continuous float targets for BCE loss."""
        return False

    def dim(self) -> tuple[int, ...]:
        """Output shape is (1,), so 1 dimension."""
        return (1,)

    def spatial(self) -> tuple[bool, ...]:
        return (False,)

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

    def is_token(self) -> bool:
        """Multi-class labels are discrete token indices."""
        return True

    def dim(self) -> tuple[int, ...]:
        """Output is a scalar tensor (dim 0)."""
        return (0,)

    def spatial(self) -> tuple[bool, ...]:
        return ()

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
            labels, one number per label, ``"balanced"`` (negatives /
            positives of each label in the fitted samples) or None.

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
        self._init_pos_weight(pos_weight)

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

    def is_token(self) -> bool:
        """Multi-label indicators are continuous float targets for BCE loss."""
        return False

    def dim(self) -> tuple[int, ...]:
        """Output shape is (num_classes,), so 1 dimension."""
        return (1,)

    def spatial(self) -> tuple[bool, ...]:
        return (False,)

    def activation(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    def loss(self, logits: torch.Tensor, y_true: Any, **kwargs: Any) -> torch.Tensor:
        return self._bce(logits, y_true, **kwargs)

    @classmethod
    def metrics_fn(cls, y_true, y_prob, metrics=None):
        from pyhealth.metrics.multilabel import multilabel_metrics_fn

        return multilabel_metrics_fn(y_true, y_prob, metrics=metrics)


@register_processor("regression")
class RegressionLabelProcessor(LabelProcessor):
    """Real-valued labels. The model has one output; the loss is MSE.

    Examples:
        >>> from pyhealth.processors import RegressionLabelProcessor
        >>> RegressionLabelProcessor().process(2.5)
        tensor([2.5000])
    """

    mode = "regression"

    def process(self, value: Any) -> torch.Tensor:
        return torch.tensor([float(value)], dtype=torch.float32)

    def size(self) -> int:
        return 1

    def is_token(self) -> bool:
        """Regression labels are continuous, not discrete tokens."""
        return False

    def dim(self) -> tuple[int, ...]:
        """Output shape is (1,), so 1 dimension."""
        return (1,)

    def spatial(self) -> tuple[bool, ...]:
        return (False,)

    def activation(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    def loss(self, logits: torch.Tensor, y_true: Any, **kwargs: Any) -> torch.Tensor:
        return F.mse_loss(logits, y_true, **kwargs)

    @classmethod
    def metrics_fn(cls, y_true, y_prob, metrics=None):
        from pyhealth.metrics.regression import regression_metrics_fn

        return regression_metrics_fn(y_true, y_prob, metrics=metrics)

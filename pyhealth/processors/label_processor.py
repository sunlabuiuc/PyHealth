import logging
from collections.abc import Hashable, Iterable, Sequence
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from . import register_processor
from .base_processor import FeatureProcessor, FieldKey

logger = logging.getLogger(__name__)


class LabelProcessor:
    """Mixin for output processors: what the label is and how to train on it.

    A label processor names its label kind (:attr:`mode`), turns a model's
    logits into predictions (:meth:`activation`) and scores them against
    the processed labels (:meth:`loss`). :class:`~pyhealth.models.BaseModel`
    uses them for its default loss and ``y_prob``.

    Attributes:
        mode: The label kind, e.g. ``"binary"``; one of
            ``pyhealth.models.base_model.VALID_MODES``.
        is_timeseries: Whether the label is given at time points (one prediction
            per label record) rather than once per sample.
        supports_pos_weight: Whether :meth:`loss` takes a ``pos_weight``.

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
    is_timeseries: ClassVar[bool] = False
    supports_pos_weight: ClassVar[bool] = False

    def activation(self, logits: torch.Tensor) -> torch.Tensor:
        """Turns logits into predictions (``y_prob``)."""
        raise NotImplementedError

    def loss(
        self,
        logits: torch.Tensor,
        y_true: Any,
        pos_weight: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """The training loss of ``logits`` against processed labels.

        Args:
            logits: The model's logits.
            y_true: A batch of this processor's outputs.
            pos_weight: Weight of positive examples, if
                :attr:`supports_pos_weight`.
            **kwargs: Passed to the underlying torch loss.
        """
        raise NotImplementedError

    def _check_pos_weight(self, pos_weight: torch.Tensor | None) -> None:
        if pos_weight is not None and not self.supports_pos_weight:
            raise ValueError(
                f"{type(self).__name__} ({self.mode!r} labels) does not take "
                "a pos_weight."
            )


def _bce_loss(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    pos_weight: torch.Tensor | None,
    **kwargs: Any,
) -> torch.Tensor:
    if pos_weight is not None:
        kwargs["pos_weight"] = pos_weight.to(logits.device)
    return F.binary_cross_entropy_with_logits(logits, y_true, **kwargs)


def _fixed_vocab(labels: int | Sequence[Hashable]) -> dict[Any, int]:
    """Index of each given label, in order; ``K`` means ``range(K)``."""
    labels = list(range(labels)) if isinstance(labels, int) else list(labels)
    if not labels:
        raise ValueError("labels must not be empty.")
    if len(set(labels)) != len(labels):
        raise ValueError(f"labels must be distinct, got {labels}.")
    return {label: i for i, label in enumerate(labels)}


def _check_known(field: str, seen: set, vocab: dict[Any, int]) -> None:
    unknown = [label for label in seen if label not in vocab]
    if unknown:
        raise ValueError(
            f"Label {field}: {sorted(map(repr, unknown))} not in the given "
            f"labels {list(vocab)}."
        )


@register_processor("binary")
class BinaryLabelProcessor(FeatureProcessor, LabelProcessor):
    """
    Processor for binary classification labels.

    Examples:
        >>> from pyhealth.processors import BinaryLabelProcessor
        >>> p = BinaryLabelProcessor()
        >>> p.fit([{"y": 0}, {"y": 1}], "y")
        >>> p.process(1)
        tensor([1.])
    """

    stores_tensor = True
    mode = "binary"
    supports_pos_weight = True

    def __init__(self):
        super().__init__()
        self.label_vocab: dict[Any, int] = {}

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        all_labels = set([sample[field] for sample in samples])
        if len(all_labels) != 2:
            raise ValueError(f"Expected 2 unique labels, got {len(all_labels)}")
        if all_labels == {0, 1}:
            self.label_vocab = {0: 0, 1: 1}
        elif all_labels == {False, True}:
            self.label_vocab = {False: 0, True: 1}
        else:
            all_labels = list(all_labels)
            all_labels.sort()
            self.label_vocab = {label: i for i, label in enumerate(all_labels)}
        logger.info(f"Label {field} vocab: {self.label_vocab}")

    def process(self, value: Any) -> torch.Tensor:
        index = self.label_vocab[value]
        return torch.tensor([index], dtype=torch.float32)

    def size(self):
        return 1

    def is_token(self) -> bool:
        """Binary labels are continuous float targets for BCE loss."""
        return False

    def schema(self) -> tuple[FieldKey, ...]:
        return ((0, "values"),)

    def dim(self) -> tuple[int, ...]:
        """Output shape is (1,), so 1 dimension."""
        return (1,)

    def spatial(self) -> dict[int, tuple[bool, ...]]:
        return {0: (False,)}

    def activation(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    def loss(self, logits, y_true, pos_weight=None, **kwargs):
        return _bce_loss(logits, y_true, pos_weight, **kwargs)

    def __repr__(self):
        return f"BinaryLabelProcessor(label_vocab_size={len(self.label_vocab)})"


@register_processor("multiclass")
class MultiClassLabelProcessor(FeatureProcessor, LabelProcessor):
    """
    Processor for multi-class classification labels.

    Args:
        labels: The classes, in index order, or their number ``K`` for
            classes that are already ``0..K-1``. When given, ``fit`` only
            checks that every label is one of them; by default the classes
            are learned in ``fit`` (sorted, or kept when already
            ``0..K-1``).

    Examples:
        >>> from pyhealth.processors import MultiClassLabelProcessor
        >>> p = MultiClassLabelProcessor()
        >>> p.fit([{"y": "a"}, {"y": "b"}, {"y": "c"}], "y")
        >>> p.process("b")
        tensor(1)
        >>> p = MultiClassLabelProcessor(labels=["low", "mid", "high"])
        >>> p.fit([{"y": "high"}, {"y": "low"}], "y")
        >>> p.process("high"), p.size()
        (tensor(2), 3)
    """

    stores_tensor = True
    mode = "multiclass"
    _labels_given = False

    def __init__(self, labels: int | Sequence[Hashable] | None = None):
        super().__init__()
        self.label_vocab: dict[Any, int] = {}
        if labels is not None:
            self.label_vocab = _fixed_vocab(labels)
            self._labels_given = True

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        all_labels = set([sample[field] for sample in samples])
        if self._labels_given:
            _check_known(field, all_labels, self.label_vocab)
            return
        num_classes = len(all_labels)
        if all_labels == set(range(num_classes)):
            self.label_vocab = {i: i for i in range(num_classes)}
        else:
            all_labels = list(all_labels)
            all_labels.sort()
            self.label_vocab = {label: i for i, label in enumerate(all_labels)}
        logger.info(f"Label {field} vocab: {self.label_vocab}")

    def process(self, value: Any) -> torch.Tensor:
        index = self.label_vocab[value]
        return torch.tensor(index, dtype=torch.long)

    def size(self):
        return len(self.label_vocab)

    def is_token(self) -> bool:
        """Multi-class labels are discrete token indices."""
        return True

    def schema(self) -> tuple[FieldKey, ...]:
        return ((0, "tokens"),)

    def dim(self) -> tuple[int, ...]:
        """Output is a scalar tensor (dim 0)."""
        return (0,)

    def spatial(self) -> dict[int, tuple[bool, ...]]:
        return {0: ()}

    def activation(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits, dim=-1)

    def loss(self, logits, y_true, pos_weight=None, **kwargs):
        self._check_pos_weight(pos_weight)
        return F.cross_entropy(logits, y_true, **kwargs)

    def __repr__(self):
        return f"MultiClassLabelProcessor(label_vocab_size={len(self.label_vocab)})"


@register_processor("multilabel")
class MultiLabelProcessor(FeatureProcessor, LabelProcessor):
    """
    Processor for multi-label classification labels.

    Args:
        labels: The labels, in index order, or their number ``K`` for labels
            that are already ``0..K-1``. When given, ``fit`` only checks that
            every label is one of them; by default the labels are learned in
            ``fit`` (sorted, or kept when already ``0..K-1``).

    Examples:
        >>> from pyhealth.processors import MultiLabelProcessor
        >>> p = MultiLabelProcessor()
        >>> p.fit([{"y": ["a"]}, {"y": ["a", "b"]}], "y")
        >>> p.process(["b"])
        tensor([0., 1.])
        >>> p = MultiLabelProcessor(labels=3)
        >>> p.fit([{"y": [0]}, {"y": [0, 2]}], "y")
        >>> p.process([2])
        tensor([0., 0., 1.])
    """

    stores_tensor = True
    mode = "multilabel"
    supports_pos_weight = True
    _labels_given = False

    def __init__(self, labels: int | Sequence[Hashable] | None = None):
        super().__init__()
        self.label_vocab: dict[Any, int] = {}
        if labels is not None:
            self.label_vocab = _fixed_vocab(labels)
            self._labels_given = True

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        all_labels = set()
        for sample in samples:
            for label in sample[field]:
                all_labels.add(label)
        if self._labels_given:
            _check_known(field, all_labels, self.label_vocab)
            return
        num_classes = len(all_labels)
        if all_labels == set(range(num_classes)):
            self.label_vocab = {i: i for i in range(num_classes)}
        else:
            all_labels = list(all_labels)
            all_labels.sort()
            self.label_vocab = {label: i for i, label in enumerate(all_labels)}
        logger.info(f"Label {field} vocab: {self.label_vocab}")

    def process(self, value: Any) -> torch.Tensor:
        if not isinstance(value, list):
            raise ValueError("Expected a list of labels for multilabel task.")
        target = torch.zeros(len(self.label_vocab), dtype=torch.float32)
        for label in value:
            index = self.label_vocab[label]
            target[index] = 1.0
        return target

    def size(self):
        return len(self.label_vocab)

    def is_token(self) -> bool:
        """Multi-label indicators are continuous float targets for BCE loss."""
        return False

    def schema(self) -> tuple[FieldKey, ...]:
        return ((0, "values"),)

    def dim(self) -> tuple[int, ...]:
        """Output shape is (num_classes,), so 1 dimension."""
        return (1,)

    def spatial(self) -> dict[int, tuple[bool, ...]]:
        return {0: (False,)}

    def activation(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    def loss(self, logits, y_true, pos_weight=None, **kwargs):
        return _bce_loss(logits, y_true, pos_weight, **kwargs)

    def __repr__(self):
        return f"MultiLabelProcessor(label_vocab_size={len(self.label_vocab)})"


@register_processor("regression")
class RegressionLabelProcessor(FeatureProcessor, LabelProcessor):
    """
    Processor for regression labels.

    Examples:
        >>> from pyhealth.processors import RegressionLabelProcessor
        >>> RegressionLabelProcessor().process(2.5)
        tensor([2.5000])
    """

    stores_tensor = True
    mode = "regression"

    def process(self, value: Any) -> torch.Tensor:
        return torch.tensor([float(value)], dtype=torch.float32)
    
    def size(self):
        return 1

    def is_token(self) -> bool:
        """Regression labels are continuous, not discrete tokens."""
        return False

    def schema(self) -> tuple[FieldKey, ...]:
        return ((0, "values"),)

    def dim(self) -> tuple[int, ...]:
        """Output shape is (1,), so 1 dimension."""
        return (1,)

    def spatial(self) -> dict[int, tuple[bool, ...]]:
        return {0: (False,)}

    def activation(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    def loss(self, logits, y_true, pos_weight=None, **kwargs):
        self._check_pos_weight(pos_weight)
        return F.mse_loss(logits, y_true, **kwargs)

    def __repr__(self):
        return "RegressionLabelProcessor()"

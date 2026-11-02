import logging
from collections.abc import Iterable
from typing import Any

import torch

from . import register_processor
from .base_processor import FeatureProcessor, FieldKey

logger = logging.getLogger(__name__)


@register_processor("multi_hot")
class MultiHotProcessor(FeatureProcessor):
    """Processor for converting categorical variables into multi-hot encoded vectors.

    Use it for input features like patient demographics (e.g., ethnicity,
    race, comorbidities). For multi-label targets use
    :class:`~pyhealth.processors.MultiLabelProcessor`, which encodes labels
    the same way; the two are separate classes so an input is never treated
    as a label.

    The processor builds a vocabulary during the fit() phase and converts each sample's
    categorical values into a fixed-size binary vector.

    Input:
        - List of categorical tokens (e.g., ["asian", "non_hispanic"])
        - Each token is a hashable value (typically strings or integers)

    Processing:
        1. During fit(): builds vocabulary by collecting all unique categorical values
           across the dataset and mapping each to a unique index
        2. During process(): converts a list of tokens into a 1D binary tensor where
           positions corresponding to present categories are set to 1.0, others to 0.0

    Output:
        - torch.Tensor of shape (num_categories,) with dtype float32
        - Binary encoding: 1.0 at indices where categories are present, 0.0 elsewhere
        - The size() method returns num_categories (vocabulary size)

    Example:
        Given samples with ethnicity field:
            - Sample 1: ["asian", "non_hispanic"]
            - Sample 2: ["white", "hispanic"]
            - Sample 3: ["black"]

        After fit():
            vocabulary = {"asian": 0, "black": 1, "hispanic": 2, "non_hispanic": 3, "white": 4}
            size() returns 5

        Processing Sample 1 produces:
            tensor([1.0, 0.0, 0.0, 1.0, 0.0])  # asian and non_hispanic are present

    Note:
        The processor sorts categories alphabetically during vocabulary construction
        to ensure consistent index assignments across runs. Categories that
        are already 0..K-1 keep their values as indices.

    Examples:
        >>> from pyhealth.processors import MultiHotProcessor
        >>> p = MultiHotProcessor()
        >>> p.fit([{"x": ["asian", "white"]}, {"x": ["black"]}], "x")
        >>> p.process(["white"])
        tensor([0., 0., 1.])
    """

    stores_tensor = True

    def __init__(self):
        super().__init__()
        self.label_vocab: dict[Any, int] = {}

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        all_labels = set()
        for sample in samples:
            for label in sample[field]:
                all_labels.add(label)
        num_classes = len(all_labels)
        if all_labels == set(range(num_classes)):
            self.label_vocab = {i: i for i in range(num_classes)}
        else:
            all_labels = list(all_labels)
            all_labels.sort()
            self.label_vocab = {label: i for i, label in enumerate(all_labels)}
        logger.info(f"Multi-hot {field} vocab: {self.label_vocab}")

    def process(self, value: Any) -> torch.Tensor:
        if not isinstance(value, list):
            raise TypeError("Expected a list of categories for a multi-hot feature.")
        target = torch.zeros(len(self.label_vocab), dtype=torch.float32)
        for label in value:
            index = self.label_vocab[label]
            target[index] = 1.0
        return target

    def size(self):
        return len(self.label_vocab)

    def is_token(self) -> bool:
        """Multi-hot indicators are continuous float inputs, not token indices."""
        return False

    def schema(self) -> tuple[FieldKey, ...]:
        return ((0, "values"),)

    def dim(self) -> tuple[int, ...]:
        """Output shape is (num_categories,), so 1 dimension."""
        return (1,)

    def spatial(self) -> dict[int, tuple[bool, ...]]:
        return {0: (False,)}

    def __repr__(self):
        return f"MultiHotProcessor(label_vocab_size={len(self.label_vocab)})"

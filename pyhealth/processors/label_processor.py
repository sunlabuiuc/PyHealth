import logging
from typing import Any, Dict, List

import torch

from . import register_processor
from .base_processor import FeatureProcessor

logger = logging.getLogger(__name__)


@register_processor("binary")
class BinaryLabelProcessor(FeatureProcessor):
    """
    Processor for binary classification labels.
    """

    def __init__(self):
        super().__init__()
        self.label_vocab: Dict[Any, int] = {}

    def fit(self, samples: List[Dict[str, Any]], field: str) -> None:
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

    def __repr__(self):
        return f"BinaryLabelProcessor(label_vocab_size={len(self.label_vocab)})"


@register_processor("multiclass")
class MultiClassLabelProcessor(FeatureProcessor):
    """
    Processor for multi-class classification labels.
    """

    def __init__(self):
        super().__init__()
        self.label_vocab: Dict[Any, int] = {}

    def fit(self, samples: List[Dict[str, Any]], field: str) -> None:
        all_labels = set([sample[field] for sample in samples])
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

    def __repr__(self):
        return f"MultiClassLabelProcessor(label_vocab_size={len(self.label_vocab)})"


@register_processor("multilabel")
class MultiLabelProcessor(FeatureProcessor):
    """
    Processor for multi-label classification labels.

    Args:
        num_classes (int): Number of classes.
    """

    def __init__(self):
        super().__init__()
        self.label_vocab: Dict[Any, int] = {}

    def fit(self, samples: List[Dict[str, Any]], field: str) -> None:
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

    def __repr__(self):
        return f"MultiLabelProcessor(label_vocab_size={len(self.label_vocab)})"


@register_processor("regression")
class RegressionLabelProcessor(FeatureProcessor):
    """
    Processor for regression labels.
    """

    def process(self, value: Any) -> torch.Tensor:
        return torch.tensor([float(value)], dtype=torch.float32)
    
    def size(self):
        return 1

    def __repr__(self):
        return "RegressionLabelProcessor()"

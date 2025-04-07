from typing import Any, Dict, Optional

import torch

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("binary")
class BinaryLabelProcessor(FeatureProcessor):
    """
    Processor for binary classification labels.
    """

    def __init__(self):
        super().__init__()
        self.label_vocab: Dict[Any, int] = {}
        self._next_index = 0

    def process(self, value: Any) -> torch.Tensor:
        if value not in self.label_vocab:
            self.label_vocab[value] = self._next_index
            self._next_index += 1
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
        self._next_index = 0

    def process(self, value: Any) -> torch.Tensor:
        if value not in self.label_vocab:
            self.label_vocab[value] = self._next_index
            self._next_index += 1
        index = self.label_vocab[value]
        return torch.tensor([index], dtype=torch.long)
    
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

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.label_vocab: Dict[Any, int] = {}
        self._next_index = 0

        if self.num_classes is None:
            raise ValueError("Multi-label task requires specifying num_classes.")

    def process(self, value: Any) -> torch.Tensor:
        if not isinstance(value, list):
            raise ValueError("Expected a list of labels for multilabel task.")
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in value:
            if label not in self.label_vocab:
                self.label_vocab[label] = self._next_index
                self._next_index += 1
            index = self.label_vocab[label]
            target[index] = 1.0
        return target

    def size(self):
        return self.num_classes

    def __repr__(self):
        return (
            f"MultiLabelProcessor(label_vocab_size={len(self.label_vocab)}, "
            f"num_classes={self.num_classes})"
        )


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

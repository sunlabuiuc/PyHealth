from typing import Any, Dict, List, Iterable

import torch

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("sequence")
class SequenceProcessor(FeatureProcessor):
    """
    Feature processor for encoding categorical sequences (e.g., medical codes) into numerical indices.

    Supports single or multiple tokens (e.g., single diagnosis or list of procedures).
    Can build vocabulary on the fly if not provided.
    """

    def __init__(self):
        # <unk> will be set to len(vocab) after fit
        self.code_vocab: Dict[Any, int] = {"<pad>": 0}
        self._next_index = 1

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
        for sample in samples:
            for token in sample[field]:
                if token is None:
                    continue  # skip missing values
                elif token not in self.code_vocab:
                    self.code_vocab[token] = self._next_index
                    self._next_index += 1

        self.code_vocab["<unk>"] = len(self.code_vocab)

    def process(self, value: Any) -> torch.Tensor:
        """Process token value(s) into tensor of indices.

        Args:
            value: Raw token string or list of token strings.

        Returns:
            Tensor of indices.
        """
        indices = []
        for token in value:
            if token in self.code_vocab:
                indices.append(self.code_vocab[token])
            else:
                indices.append(self.code_vocab["<unk>"])

        return torch.tensor(indices, dtype=torch.long)

    def size(self):
        return len(self.code_vocab)

    def __repr__(self):
        return f"SequenceProcessor(code_vocab_size={len(self.code_vocab)})"

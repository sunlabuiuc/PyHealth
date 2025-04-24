from typing import Any, Dict, List

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
        # -1 for <unk> for ease of boolean arithmetic > 0, > -1, etc.
        # TODO: this can be a problem if we pass -1 into nn.Embedding
        self.code_vocab: Dict[Any, int] = {"<unk>": -1, "<pad>": 0}
        self._next_index = 1

    def process(self, value: Any) -> torch.Tensor:
        """Process token value(s) into tensor of indices.

        Args:
            value: Raw token string or list of token strings.

        Returns:
            Tensor of indices.
        """
        indices = []
        for token in value:
            if token is None: # missing values
                indices.append(self.code_vocab["<unk>"])
            else:
                if token not in self.code_vocab:
                    self.code_vocab[token] = self._next_index
                    self._next_index += 1
                indices.append(self.code_vocab[token])
        return torch.tensor(indices, dtype=torch.long)
    
    def size(self):
        return len(self.code_vocab)
    
    def __repr__(self):
        return (
            f"SequenceProcessor(code_vocab_size={len(self.code_vocab)})"
        )

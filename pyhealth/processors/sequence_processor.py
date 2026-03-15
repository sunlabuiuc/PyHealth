from typing import Any, Dict, List, Iterable

import torch

from . import register_processor
from .base_processor import FeatureProcessor, TokenProcessorInterface


@register_processor("sequence")
class SequenceProcessor(FeatureProcessor, TokenProcessorInterface):
    """
    Feature processor for encoding categorical sequences (e.g., medical codes) into numerical indices.

    Supports single or multiple tokens (e.g., single diagnosis or list of procedures).
    Can build vocabulary on the fly if not provided.
    """

    def __init__(self):
        self.code_vocab: Dict[Any, int] = {"<pad>": self.PAD, "<unk>": self.UNK}
        self._next_index = 2

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
        for sample in samples:
            for token in sample[field]:
                if token is None:
                    continue  # skip missing values
                elif token not in self.code_vocab:
                    self.code_vocab[token] = self._next_index
                    self._next_index += 1

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
    
    def remove(self, tokens: set[str]):
        """Remove specified vocabularies from the processor."""
        keep = set(self.code_vocab.keys()) - tokens | {"<pad>", "<unk>"}
        order = [k for k, v in sorted(self.code_vocab.items(), key=lambda x: x[1]) if k in keep]
        
        self.code_vocab = { k : i for i, k in enumerate(order) }

    def retain(self, tokens: set[str]):
        """Retain only the specified vocabularies in the processor."""
        keep = set(self.code_vocab.keys()) & tokens | {"<pad>", "<unk>"}
        order = [k for k, v in sorted(self.code_vocab.items(), key=lambda x: x[1]) if k in keep]
        
        self.code_vocab = { k : i for i, k in enumerate(order) }

    def add(self, tokens: set[str]):
        """Add specified vocabularies to the processor."""
        i = len(self.code_vocab)
        for token in tokens:
            if token not in self.code_vocab:
                self.code_vocab[token] = i
                i += 1

    def tokens(self) -> set[str]:
        """Return the set of tokens in the processor's vocabulary."""
        return set(self.code_vocab.keys())

    def vocab_size(self) -> int:
        """Return the size of the processor's vocabulary."""
        return len(self.code_vocab)

    def size(self):
        return len(self.code_vocab)

    def is_token(self) -> bool:
        """Sequence codes are discrete token indices."""
        return True

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        """Output is a 1D tensor of code indices."""
        return (1,)

    def spatial(self) -> tuple[bool, ...]:
        return (True,)

    def __repr__(self):
        return f"SequenceProcessor(code_vocab_size={len(self.code_vocab)})"

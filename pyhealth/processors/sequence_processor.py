from typing import Any, Dict, List, Iterable, Optional, Tuple

import torch

from . import register_processor
from .base_processor import FeatureProcessor, TokenProcessorInterface


@register_processor("sequence")
class SequenceProcessor(FeatureProcessor, TokenProcessorInterface):
    """Feature processor for encoding categorical sequences.

    Encodes medical codes (e.g., diagnoses, procedures) into numerical
    indices. Supports single or multiple tokens and can build vocabulary
    on the fly if not provided.

    Args:
        code_mapping: optional tuple of (source_vocabulary, target_vocabulary)
            to map raw codes to a grouped vocabulary before tokenizing.
            Uses ``pyhealth.medcode.CrossMap`` internally. For example,
            ``("ICD9CM", "CCSCM")`` maps ~128K ICD-9 diagnosis codes to
            ~280 CCS categories, and ``("NDC", "ATC")`` maps ~940K drug
            codes to ~5K ATC categories. When None (default), codes are
            used as-is with no change to existing behavior.

    Examples:
        >>> proc = SequenceProcessor()  # no mapping, same as before
        >>> proc = SequenceProcessor(code_mapping=("ICD9CM", "CCSCM"))
    """

    def __init__(self, code_mapping: Optional[Tuple[str, str]] = None):
        self.code_vocab: Dict[Any, int] = {"<pad>": self.PAD, "<unk>": self.UNK}
        self._next_index = 2
        self._mapper = None
        if code_mapping is not None:
            from pyhealth.medcode import CrossMap
            self._mapper = CrossMap.load(code_mapping[0], code_mapping[1])

    def _map(self, token: str) -> List[str]:
        """Map a single token through the code mapping, if configured.

        Returns the token unchanged (as a single-element list) when no
        mapping is configured or when the token has no mapping.
        """
        if self._mapper is None:
            return [token]
        mapped = self._mapper.map(token)
        return mapped if mapped else [token]

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
        """Build vocabulary from samples, applying code mapping if set.

        Args:
            samples: iterable of sample dicts.
            field: key whose values are token lists.
        """
        for sample in samples:
            for token in sample[field]:
                if token is None:
                    continue  # skip missing values
                for mapped in self._map(token):
                    if mapped not in self.code_vocab:
                        self.code_vocab[mapped] = self._next_index
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
            if token is None:
                continue  # skip missing values, consistent with fit()
            for mapped in self._map(token):
                if mapped in self.code_vocab:
                    indices.append(self.code_vocab[mapped])
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

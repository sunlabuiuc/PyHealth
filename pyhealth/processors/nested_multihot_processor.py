from collections.abc import Iterable
from typing import Any

import torch

from . import register_processor
from .base_processor import FeatureProcessor, TokenProcessorInterface


@register_processor("nested_multihot")
class NestedMultiHotProcessor(FeatureProcessor, TokenProcessorInterface):
    """Nested categorical sequences as per-visit multi-hot vectors.

    Same input as :class:`NestedSequenceProcessor` -- a list of visits, each a
    list of codes -- but it emits what set-membership models actually consume:
    one row per visit, one column per vocabulary entry, 1 where the code is
    present.

    Why this exists rather than indices
    -----------------------------------
    ``NestedSequenceProcessor`` pads every visit to the longest visit seen
    during ``fit``. That width is set by a single outlier: on eICU one visit
    holds 3,951 code entries (the same diagnosis re-charted through a stay)
    while a typical visit holds about five. Every downstream consumer then pays
    for it twice -- once in memory, once again if it loops over the padding to
    find the handful of real codes.

    For a 921-code vocabulary on eICU the multi-hot form is **8.6x smaller**
    than the padded index form (4.5 KB vs 38.7 KB per patient), because it is
    sized by the vocabulary rather than by the worst-case visit.

    Repeats collapse
    ----------------
    A code charted five times in one visit sets the same column to 1 once. That
    matches what set-membership models already do with the index form, so
    swapping this in changes nothing about what a model sees. If you need
    counts, this is the wrong processor.

    Special tokens
    --------------
    ``<pad>`` (0) and ``<unk>`` (1) keep their indices, so the vocabulary is
    interchangeable with ``NestedSequenceProcessor`` -- a cached vocabulary from
    one restores into the other. Column 0 is therefore always zero: ``<pad>``
    means "nothing here", and marking it would make padding indistinguishable
    from a real code.

    Examples:
        >>> processor = NestedMultiHotProcessor()
        >>> samples = [{"codes": [["A", "B"], ["C"]]}]
        >>> processor.fit(samples, "codes")
        >>> out = processor.process([["A", "B"], ["A"]])
        >>> out.shape           # (2 visits, vocab_size)
        torch.Size([2, 5])
        >>> out[0].nonzero().flatten().tolist()   # A and B present in visit 0
        [2, 3]
    """

    def __init__(self, padding: int = 0):
        # `padding` is accepted and ignored so this is a drop-in swap for
        # NestedSequenceProcessor in a schema. There is no inner axis to pad --
        # that is the entire point -- so honouring it would be misleading.
        self.code_vocab: dict[Any, int] = {"<pad>": self.PAD, "<unk>": self.UNK}
        self._next_index = 2
        self._padding = padding

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        """Build the vocabulary. Inner length is irrelevant here, so unlike
        ``NestedSequenceProcessor`` nothing is measured about visit width.

        Args:
            samples: Sample dictionaries.
            field: Field holding the nested sequence.
        """
        for sample in samples:
            if field not in sample or sample[field] is None:
                continue
            nested_seq = sample[field]
            if not isinstance(nested_seq, list):
                continue
            for inner_seq in nested_seq:
                if not isinstance(inner_seq, list):
                    continue
                for code in inner_seq:
                    if code is not None and code not in self.code_vocab:
                        self.code_vocab[code] = self._next_index
                        self._next_index += 1

    def remove(self, tokens: set[str]):
        """Remove specified vocabularies from the processor."""
        keep = set(self.code_vocab.keys()) - tokens | {"<pad>", "<unk>"}
        order = [k for k, v in sorted(self.code_vocab.items(), key=lambda x: x[1])
                 if k in keep]
        self.code_vocab = {k: i for i, k in enumerate(order)}
        self._next_index = len(self.code_vocab)

    def retain(self, tokens: set[str]):
        """Retain only the specified vocabularies in the processor."""
        keep = set(self.code_vocab.keys()) & tokens | {"<pad>", "<unk>"}
        order = [k for k, v in sorted(self.code_vocab.items(), key=lambda x: x[1])
                 if k in keep]
        self.code_vocab = {k: i for i, k in enumerate(order)}
        self._next_index = len(self.code_vocab)

    def add(self, tokens: set[str]):
        """Add specified vocabularies to the processor."""
        i = len(self.code_vocab)
        for token in tokens:
            if token not in self.code_vocab:
                self.code_vocab[token] = i
                i += 1
        self._next_index = len(self.code_vocab)

    def tokens(self) -> set[str]:
        """Return the set of tokens in the processor's vocabulary."""
        return set(self.code_vocab.keys())

    def process(self, value: list[list[Any]]) -> torch.Tensor:
        """Nested sequence -> ``(num_visits, vocab_size)`` float multi-hot.

        Built with one ``scatter_`` per visit rather than per-code assignment,
        so cost tracks the number of real codes, never the vocabulary size or a
        padded width.

        An empty or ``None`` sample yields a single all-zero visit, mirroring
        ``NestedSequenceProcessor`` returning one all-``<pad>`` row: both say
        "one visit, nothing in it".

        Args:
            value: Nested list of codes ``[[code1, code2], [code3], ...]``.

        Returns:
            2D float tensor ``(num_visits, vocab_size)``, 1.0 where present.
        """
        vocab_size = len(self.code_vocab)
        unk = self.code_vocab["<unk>"]

        if not value or len(value) == 0:
            return torch.zeros(1, vocab_size, dtype=torch.float)

        out = torch.zeros(len(value), vocab_size, dtype=torch.float)
        for row, inner_seq in enumerate(value):
            if inner_seq is None or len(inner_seq) == 0:
                continue                      # empty visit stays all-zero
            idx = [self.code_vocab.get(code, unk) if code is not None else unk
                   for code in inner_seq]
            # scatter_ over the visit's own codes: duplicates write 1.0 twice,
            # which is still 1.0 -- repeats collapse by construction.
            out[row].scatter_(0, torch.tensor(idx, dtype=torch.long),
                              torch.ones(len(idx)))
        return out

    def size(self) -> int:
        """Feature width: the vocabulary, since that is the row length."""
        return len(self.code_vocab)

    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.code_vocab)

    def __repr__(self):
        return f"NestedMultiHotProcessor(vocab_size={len(self.code_vocab)})"

    def is_token(self) -> bool:
        """Output is a dense indicator vector, not token indices."""
        return False

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        """Output is a 2D tensor (visits, vocab)."""
        return (2,)

    def spatial(self) -> tuple[bool, ...]:
        # Visits (time) are ordered; the vocabulary axis is an unordered set.
        return (True, False)

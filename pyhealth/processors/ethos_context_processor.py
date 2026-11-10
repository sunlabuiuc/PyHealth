"""Static patient context for ETHOS timelines.

ETHOS opens every timeline with tokens for the patient's demographics and age.
:class:`~pyhealth.processors.EthosProcessor` keeps those codes out of the
timeline itself; this processor encodes them.

Paper:
    Renc et al. Foundation Model of Electronic Medical Records for Adaptive
    Risk Estimation. https://arxiv.org/abs/2502.06124

Adapted from ethos-ares (commit 2d54383), ``src/ethos/datasets/base.py`` and
``src/ethos/tokenize/common/basic.py``:
https://github.com/ipolharvard/ethos-ares

MIT License. Copyright (c) 2024 Paweł Renc.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import orjson
import torch

from . import register_processor
from .base_processor import FeatureProcessor
from .ethos_processor import STATIC_CODE_PREFIXES, EthosVocab

BIRTH_CODE = "MEDS_BIRTH"
UNKNOWN_SUFFIX = "//UNKNOWN"
DAYS_PER_YEAR = 365.25


def _value_at_start(
    rows: list[tuple[datetime | None, str]],
    attribute: str,
    start: datetime | None,
) -> str:
    """Returns the code of one attribute at the start of a timeline.

    An attribute with a single value keeps it whenever it was recorded. With
    several values, the latest one recorded by ``start`` applies, as ETHOS does.

    Args:
        rows: ``(time, code)`` rows of this attribute in time order. A row
            without a time counts as known from the beginning.
        attribute: The attribute's prefix.
        start: Start of the timeline.

    Returns:
        The code, or ``<attribute>//UNKNOWN`` if nothing applies yet.
    """
    if len(rows) == 1:
        return rows[0][1]
    known = [
        code for time, code in rows if time is None or start is None or time <= start
    ]
    return known[-1] if known else attribute + UNKNOWN_SUFFIX


@register_processor("ethos_context")
class EthosContextProcessor(FeatureProcessor):
    """Encodes the static context that opens an ETHOS timeline.

    The input is one patient's static rows:

    * ``time``: a timestamp or ``None`` per row.
    * ``code``: a string per row, such as ``GENDER//F`` or ``MEDS_BIRTH``.
    * ``start``: when the timeline starts.

    The output has a fixed length. Attributes are sorted by name, so with the
    default prefixes it reads ``GENDER``, ``MARITAL``, the two age tokens,
    ``RACE``. A patient with no value gets ``<attribute>//UNKNOWN``, and a value
    unseen in :meth:`fit` gets ``<unk>``. The age tokens hold ``<pad>`` here:
    their quantile tokens belong to the timeline vocabulary and depend on where
    a window of the timeline starts, so :class:`~pyhealth.models.ETHOS` fills
    them in from the returned age.

    Args:
        prefixes: Static code prefixes, including ``MEDS_BIRTH``. Use the same
            ones as ``exclude_prefixes`` of
            :class:`~pyhealth.processors.EthosProcessor`.

    Raises:
        ValueError: If ``prefixes`` leaves out ``MEDS_BIRTH``.

    Examples:
        >>> from datetime import datetime
        >>> from pyhealth.processors import EthosContextProcessor
        >>> static = {
        ...     "time": [datetime(1980, 1, 1), None],
        ...     "code": ["MEDS_BIRTH", "GENDER//F"],
        ...     "start": datetime(2024, 1, 1),
        ... }
        >>> processor = EthosContextProcessor()
        >>> processor.fit([{"context": static}], "context")
        >>> age, tokens = processor.process(static)
        >>> round(age.item(), 1)
        44.0
        >>> processor.decode(tokens)
        ['GENDER//F', 'MARITAL//UNKNOWN', '<pad>', '<pad>', 'RACE//UNKNOWN']
    """

    stores_tensor = True

    def __init__(self, prefixes: Sequence[str] = STATIC_CODE_PREFIXES) -> None:
        if BIRTH_CODE not in prefixes:
            raise ValueError(f"prefixes must include {BIRTH_CODE}, got {prefixes}.")
        self.prefixes = tuple(sorted(prefixes))
        self.vocab: EthosVocab | None = None

    @property
    def age_position(self) -> int:
        """Index of the first of the two age tokens in the output."""
        return self.prefixes.index(BIRTH_CODE)

    @property
    def context_size(self) -> int:
        """Length of the output: one per attribute, and two for age."""
        return len(self.prefixes) + 1

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        """Learns the vocabulary of attribute values.

        Args:
            samples: Sample dictionaries.
            field: Key of the static rows in each sample.
        """
        self.vocab = EthosVocab()
        for prefix in self.prefixes:
            if prefix != BIRTH_CODE:
                self.vocab.add_token(prefix + UNKNOWN_SUFFIX)
        for sample in samples:
            for code in sorted(sample[field]["code"]):
                if code.startswith(self.prefixes) and code != BIRTH_CODE:
                    self.vocab.add_token(code)

    def process(self, value: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes one patient's static rows.

        Args:
            value: Static rows, as in the class docstring.

        Returns:
            ``(age, tokens)``. ``age`` has one ``float64`` element, the age in
            years at ``start``, or ``nan`` without a usable birth date.
            ``tokens`` is ``long`` and has :attr:`context_size` elements.

        Raises:
            RuntimeError: If called before :meth:`fit` or :meth:`load`.
        """
        if self.vocab is None:
            raise RuntimeError("EthosContextProcessor must be fitted or loaded first.")

        start = value["start"]
        pairs = list(zip(value["time"], value["code"]))
        rows = [row for row in pairs if row[0] is None]
        rows += sorted((row for row in pairs if row[0] is not None), key=lambda r: r[0])
        tokens: list[int] = []
        age = float("nan")
        for prefix in self.prefixes:
            matching = [row for row in rows if row[1].startswith(prefix)]
            if prefix == BIRTH_CODE:
                if matching and matching[0][0] is not None and start is not None:
                    age = (start - matching[0][0]).days / DAYS_PER_YEAR
                tokens += [self.vocab.pad_id, self.vocab.pad_id]
            elif matching:
                tokens.append(self.vocab[_value_at_start(matching, prefix, start)])
            else:
                tokens.append(self.vocab[prefix + UNKNOWN_SUFFIX])
        return (
            torch.tensor([age], dtype=torch.float64),
            torch.tensor(tokens, dtype=torch.long),
        )

    def decode(self, tokens: torch.Tensor | Iterable[int]) -> list[str]:
        """Maps token ids back to token strings.

        Args:
            tokens: Token ids, such as the output of :meth:`process`.

        Returns:
            The token strings.

        Examples:
            >>> from pyhealth.processors import EthosContextProcessor
            >>> processor = EthosContextProcessor()
            >>> processor.fit([], "context")
            >>> processor.decode([0, 1])
            ['<pad>', '<unk>']
        """
        by_id = {index: token for token, index in self.vocab.token_to_id.items()}
        return [by_id[int(token)] for token in tokens]

    def save(self, path: str) -> None:
        """Saves the configuration and fitted state as JSON.

        Args:
            path: File to write.
        """
        state = {
            "prefixes": self.prefixes,
            "vocab": None if self.vocab is None else self.vocab.to_json(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(orjson.dumps(state, option=orjson.OPT_SORT_KEYS))

    def load(self, path: str) -> None:
        """Restores what :meth:`save` wrote, replacing the current state.

        Args:
            path: File to read.
        """
        state = orjson.loads(Path(path).read_bytes())
        self.prefixes = tuple(state["prefixes"])
        vocab = state["vocab"]
        self.vocab = None if vocab is None else EthosVocab.from_json(vocab)

    def size(self) -> int:
        """Returns the vocabulary size."""
        return 0 if self.vocab is None else self.vocab.vocab_size

    def is_token(self) -> bool:
        """Output tokens are discrete indices."""
        return True

    def schema(self) -> tuple[str, ...]:
        return ("age", "value")

    def dim(self) -> tuple[int, ...]:
        return (1, 1)

    def spatial(self) -> tuple[bool, ...]:
        return (False,)

    def __repr__(self) -> str:
        return (
            f"EthosContextProcessor(prefixes={self.prefixes}, "
            f"vocab_size={self.size()})"
        )

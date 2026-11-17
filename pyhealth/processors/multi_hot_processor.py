import logging
from collections.abc import Iterable
from typing import Any

import torch

from . import register_processor
from .base_processor import LegacyFeatureProcessor

logger = logging.getLogger(__name__)


@register_processor("multi_hot")
class MultiHotProcessor(LegacyFeatureProcessor):
    """Processor for converting categorical variables into multi-hot encoded vectors.

    Use ``MultiHotProcessor`` for input features like patient demographics (e.g.,
    ethnicity, race, comorbidities). For output labels in multi-label classification
    tasks, use :class:`MultiLabelProcessor` instead. The two classes are independent:
    neither subclasses the other.

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

    Examples:
        >>> from pyhealth.processors import MultiHotProcessor
        >>> p = MultiHotProcessor()
        >>> p.fit(
        ...     [
        ...         {"ethnicity": ["asian", "non_hispanic"]},
        ...         {"ethnicity": ["white", "hispanic"]},
        ...         {"ethnicity": ["black"]},
        ...     ],
        ...     "ethnicity",
        ... )
        >>> p.label_vocab
        {'asian': 0, 'black': 1, 'hispanic': 2, 'non_hispanic': 3, 'white': 4}
        >>> p.size()
        5
        >>> p.process(["asian", "non_hispanic"])
        tensor([1., 0., 0., 1., 0.])

    Note:
        The processor sorts categories alphabetically during vocabulary construction
        to ensure consistent index assignments across runs.

    See Also:
        :class:`MultiLabelProcessor`: Use for output labels in multi-label
            classification tasks.
    """

    stores_tensor = True

    def __init__(self):
        super().__init__()
        self.label_vocab: dict[Any, int] = {}

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        all_tokens = set()
        for sample in samples:
            for token in sample[field]:
                all_tokens.add(token)
        num_categories = len(all_tokens)
        if all_tokens == set(range(num_categories)):
            self.label_vocab = {i: i for i in range(num_categories)}
        else:
            all_tokens = sorted(all_tokens)
            self.label_vocab = {token: i for i, token in enumerate(all_tokens)}
        logger.info(f"Multi-hot {field} vocab: {self.label_vocab}")

    def process(self, value: Any) -> torch.Tensor:
        if not isinstance(value, list):
            # ValueError (not TypeError) to match the behaviour before the split.
            raise ValueError(  # noqa: TRY004
                "Expected a list of categorical values for multi-hot feature."
            )
        encoded = torch.zeros(len(self.label_vocab), dtype=torch.float32)
        for token in value:
            encoded[self.label_vocab[token]] = 1.0
        return encoded

    def size(self):
        return len(self.label_vocab)

    def is_token(self) -> bool:
        """Multi-hot indicators are continuous float values, not token indices."""
        return False

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        """Output shape is (num_categories,), so 1 dimension."""
        return (1,)

    def spatial(self) -> tuple[bool, ...]:
        return (False,)

    def __repr__(self):
        return f"MultiHotProcessor(label_vocab_size={len(self.label_vocab)})"



from typing import Any, Dict, List

import torch

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("nested_sequence")
class NestedSequenceProcessor(FeatureProcessor):
    """
    Feature processor for nested categorical sequences with vocabulary.

    Handles nested sequences like drug recommendation history where each sample
    contains a list of visits, and each visit contains a list of codes:
    [["code1", "code2"], ["code3"], ["code4", "code5", "code6"]]

    The processor:
    1. Builds a vocabulary from all codes across all samples
    2. Encodes codes to indices
    3. Pads inner sequences to the maximum sequence length found during fit
    4. Returns a 2D tensor of shape (num_visits, max_codes_per_visit)

    Special tokens:
        - <unk>: -1 for unknown codes
        - <pad>: 0 for padding

    Examples:
        >>> processor = NestedSequenceProcessor()
        >>> # During fit, determines max inner sequence length
        >>> samples = [
        ...     {"codes": [["A", "B"], ["C", "D", "E"]]},
        ...     {"codes": [["F"]]}
        ... ]
        >>> processor.fit(samples, "codes")
        >>> # Process nested sequence
        >>> result = processor.process([["A", "B"], ["C"]])
        >>> result.shape  # (2, 3) - 2 visits, padded to max length 3
    """

    def __init__(self):
        # -1 for <unk> for ease of boolean arithmetic > 0, > -1, etc.
        self.code_vocab: Dict[Any, int] = {"<unk>": -1, "<pad>": 0}
        self._next_index = 1
        self._max_inner_len = 1  # Maximum length of inner sequences

    def fit(self, samples: List[Dict[str, Any]], field: str) -> None:
        """Build vocabulary and determine maximum inner sequence length.

        Args:
            samples: List of sample dictionaries.
            field: The field name containing nested sequences.
        """
        max_inner_len = 0

        for sample in samples:
            if field in sample and sample[field] is not None:
                nested_seq = sample[field]

                # Nested sequences: [["A", "B"], ["C"], ...]
                if isinstance(nested_seq, list):
                    for inner_seq in nested_seq:
                        if isinstance(inner_seq, list):
                            # Track max inner length
                            max_inner_len = max(max_inner_len, len(inner_seq))

                            # Build vocabulary
                            for code in inner_seq:
                                if code is not None and code not in self.code_vocab:
                                    self.code_vocab[code] = self._next_index
                                    self._next_index += 1

        # Store max inner length (at least 1 for empty sequences)
        self._max_inner_len = max(1, max_inner_len)

    def process(self, value: List[List[Any]]) -> torch.Tensor:
        """Process nested sequence into padded 2D tensor.

        Empty or None visits are filled with padding tokens.

        Args:
            value: Nested list of codes [[code1, code2], [code3], ...]

        Returns:
            2D tensor of shape (num_visits, max_inner_len) with code indices
        """
        # Handle empty nested sequence
        if not value or len(value) == 0:
            pad_token = self.code_vocab["<pad>"]
            padded_row = [pad_token] * self._max_inner_len
            return torch.tensor([padded_row], dtype=torch.long)

        encoded_sequences = []
        pad_token = self.code_vocab["<pad>"]

        for inner_seq in value:
            # Check if this visit is empty/null - use padding tokens
            if inner_seq is None or len(inner_seq) == 0:
                encoded_sequences.append([pad_token] * self._max_inner_len)
                continue

            indices = []

            # Encode each code in the inner sequence
            for code in inner_seq:
                if code is None or code not in self.code_vocab:
                    indices.append(self.code_vocab["<unk>"])
                else:
                    indices.append(self.code_vocab[code])

            # Pad to maximum inner length
            while len(indices) < self._max_inner_len:
                indices.append(pad_token)

            encoded_sequences.append(indices)

        return torch.tensor(encoded_sequences, dtype=torch.long)

    def size(self) -> int:
        """Return max inner length (embedding dimension) for unified API."""
        return self._max_inner_len

    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.code_vocab)

    def __repr__(self):
        return (
            f"NestedSequenceProcessor("
            f"vocab_size={len(self.code_vocab)}, "
            f"max_inner_len={self._max_inner_len})"
        )


@register_processor("nested_sequence_floats")
class NestedFloatsProcessor(FeatureProcessor):
    """
    Feature processor for nested numerical sequences without vocabulary.

    Handles nested sequences of floats/numerical values where each sample
    contains a list of visits, and each visit contains a list of values:
    [[1.5, 2.3], [4.1], [0.9, 1.2, 3.4]]

    The processor:
    1. Determines the maximum inner sequence length during fit
    2. Optionally applies forward-fill for missing values
    3. Returns a 2D tensor of shape (num_visits, max_values_per_visit)

    Args:
        forward_fill: If True, applies forward fill for NaN values across
            time steps and empty visits. If False, sets null values to 0.
            Default is True.

    Examples:
        >>> processor = NestedFloatsProcessor()
        >>> # During fit, determines max inner sequence length
        >>> samples = [
        ...     {"values": [[1.0, 2.0], [3.0, 4.0, 5.0]]},
        ...     {"values": [[6.0]]}
        ... ]
        >>> processor.fit(samples, "values")
        >>> # Process nested sequence
        >>> result = processor.process([[1.0, 2.0], [3.0]])
        >>> result.shape  # (2, 3) - 2 visits, padded to max length 3
    """

    def __init__(self, forward_fill: bool = True):
        self._max_inner_len = 1  # Maximum length of inner sequences
        self.forward_fill = forward_fill

    def fit(self, samples: List[Dict[str, Any]], field: str) -> None:
        """Determine maximum inner sequence length.

        Args:
            samples: List of sample dictionaries.
            field: The field name containing nested sequences.
        """
        max_inner_len = 0

        for sample in samples:
            if field in sample and sample[field] is not None:
                nested_seq = sample[field]

                # Nested sequences: [[1.0, 2.0], [3.0], ...]
                if isinstance(nested_seq, list):
                    for inner_seq in nested_seq:
                        if isinstance(inner_seq, list):
                            # Track max inner length
                            max_inner_len = max(max_inner_len, len(inner_seq))

        # Store max inner length (at least 1 for empty sequences)
        self._max_inner_len = max(1, max_inner_len)

    def process(self, value: List[List[float]]) -> torch.Tensor:
        """Process nested numerical sequence with optional forward fill.

        For missing values (None or empty visits):
        - If forward_fill=True: uses forward fill from last valid visit
        - If forward_fill=False: sets null values to 0.0 (for masking)

        Args:
            value: Nested list of floats [[1.0, 2.0], [3.0], ...]

        Returns:
            2D tensor of shape (num_visits, max_inner_len) with float values
        """
        import numpy as np

        # Handle empty nested sequence
        if not value or len(value) == 0:
            if self.forward_fill:
                return torch.full(
                    (1, self._max_inner_len), float("nan"), dtype=torch.float
                )
            else:
                return torch.zeros((1, self._max_inner_len), dtype=torch.float)

        encoded_sequences = []
        last_valid_values = None

        for inner_seq in value:
            # Check if this visit is empty/null
            if inner_seq is None or len(inner_seq) == 0:
                if self.forward_fill and last_valid_values is not None:
                    # Forward fill: use last valid visit's values
                    encoded_sequences.append(last_valid_values.copy())
                else:
                    # No forward fill or no prior visit, use zeros
                    encoded_sequences.append([0.0] * self._max_inner_len)
                continue

            values = []

            # Convert each value to float
            for val in inner_seq:
                if val is None:
                    if self.forward_fill:
                        values.append(float("nan"))
                    else:
                        values.append(0.0)
                else:
                    try:
                        values.append(float(val))
                    except (ValueError, TypeError):
                        if self.forward_fill:
                            values.append(float("nan"))
                        else:
                            values.append(0.0)

            # Pad to maximum inner length
            while len(values) < self._max_inner_len:
                if self.forward_fill:
                    values.append(float("nan"))
                else:
                    values.append(0.0)

            # Store as last valid values for forward fill
            last_valid_values = values.copy()
            encoded_sequences.append(values)

        # Convert to numpy array
        values_array = np.array(encoded_sequences, dtype=float)

        # Apply forward fill for NaN values if enabled
        # Forward fill happens in two passes:
        # 1. Across visits (column-wise): missing values get previous visit
        # 2. Within each visit (row-wise): pad positions get last valid value
        if self.forward_fill:
            # First: forward fill across visits (column-wise)
            # For each feature dimension, fill NaN with previous visit's value
            for feature_idx in range(values_array.shape[1]):
                last_value = None
                for visit_idx in range(values_array.shape[0]):
                    if not np.isnan(values_array[visit_idx, feature_idx]):
                        last_value = values_array[visit_idx, feature_idx]
                    elif last_value is not None:
                        values_array[visit_idx, feature_idx] = last_value

            # Second: forward fill within each visit (row-wise)
            # For padding positions, fill with last valid value in that visit
            for visit_idx in range(values_array.shape[0]):
                last_value = None
                for feature_idx in range(values_array.shape[1]):
                    if not np.isnan(values_array[visit_idx, feature_idx]):
                        last_value = values_array[visit_idx, feature_idx]
                    elif last_value is not None:
                        values_array[visit_idx, feature_idx] = last_value

            # Third: any remaining NaN values (first visit with no prior)
            # are set to 0.0
            values_array = np.nan_to_num(values_array, nan=0.0)

        return torch.tensor(values_array, dtype=torch.float)

    def size(self) -> int:
        """Return max inner length (embedding dimension) for unified API."""
        return self._max_inner_len

    def __repr__(self):
        return (
            f"NestedFloatsProcessor("
            f"max_inner_len={self._max_inner_len}, "
            f"forward_fill={self.forward_fill})"
        )

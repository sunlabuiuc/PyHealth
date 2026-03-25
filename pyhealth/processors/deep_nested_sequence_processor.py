from typing import Any, Dict, List, Iterable

import torch

from . import register_processor
from .base_processor import FeatureProcessor, TokenProcessorInterface


@register_processor("deep_nested_sequence")
class DeepNestedSequenceProcessor(FeatureProcessor, TokenProcessorInterface):
    """
    Feature processor for deeply nested categorical sequences with vocabulary.

    Handles 3-level nested sequences where each sample contains a list of
    groups, each group contains a list of visits, and each visit contains a
    list of codes:

        [
            [["code1", "code2"], ["code3"]],
            [["code4"], ["code5", "code6"]]
        ]

    The processor:
    1. Builds a vocabulary from all codes across all samples
    2. Encodes codes to indices
    3. Pads:
        - The inner-most sequences (codes) to max_codes_per_visit
        - The middle sequences (visits) to max_visits_per_group
    4. Returns a 3D tensor of shape:
        (num_groups, max_visits_per_group, max_codes_per_visit)

    Special tokens:
        - <pad>: 0 for padding
        - <unk>: 1 for unknown codes

    Examples:
        >>> processor = DeepNestedSequenceProcessor()
        >>> samples = [
        ...     {"codes": [[["A", "B"], ["C", "D", "E"]]]},
        ...     {"codes": [[["F"]]]},
        ... ]
        >>> processor.fit(samples, "codes")
        >>> result = processor.process([[["A", "B"], ["C"]]])
        >>> result.shape  # (1, max_visits_per_group, max_codes_per_visit)
    """

    def __init__(self):
        self.code_vocab: Dict[Any, int] = {"<pad>": self.PAD, "<unk>": self.UNK}
        self._next_index = 2
        self._max_middle_len = 1  # Maximum length of middle sequences (e.g. visits)
        self._max_inner_len = 1   # Maximum length of inner sequences (e.g. codes per visit)

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
        """Build vocabulary and determine maximum sequence lengths.

        Args:
            samples: List of sample dictionaries.
            field: The field name containing deep nested sequences.
        """
        max_middle_len = 0
        max_inner_len = 0

        for sample in samples:
            if field in sample and sample[field] is not None:
                deep_nested = sample[field]

                # Deep nested sequences: [[[...], [...]], ...]
                if isinstance(deep_nested, list):
                    for middle_seq in deep_nested:  # e.g. per group
                        if isinstance(middle_seq, list):
                            # Track max # of visits per group
                            max_middle_len = max(max_middle_len, len(middle_seq))

                            for inner_seq in middle_seq:  # per visit
                                if isinstance(inner_seq, list):
                                    # Track max # of codes per visit
                                    max_inner_len = max(max_inner_len, len(inner_seq))

                                    # Build vocabulary
                                    for code in inner_seq:
                                        if code is not None and code not in self.code_vocab:
                                            self.code_vocab[code] = self._next_index
                                            self._next_index += 1

        self._max_middle_len = max(1, max_middle_len)
        self._max_inner_len = max(1, max_inner_len)

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

    def process(self, value: List[List[List[Any]]]) -> torch.Tensor:
        """Process deep nested sequence into padded 3D tensor.

        Empty or None groups/visits are filled with padding tokens.

        Args:
            value: Deep nested list of codes
                   [[[code1, code2], [code3]], [[code4], ...], ...]

        Returns:
            3D tensor of shape
            (num_groups, max_middle_len, max_inner_len) with code indices
        """
        pad_token = self.code_vocab["<pad>"]

        # Handle completely empty deep nested sequence
        if not value or len(value) == 0:
            padded_row = [pad_token] * self._max_inner_len
            # One group, one (padded) visit
            return torch.tensor([[padded_row]], dtype=torch.long)

        encoded_groups: List[List[List[int]]] = []

        for middle_seq in value:
            # middle_seq is a list of visits (or None)
            group_encoded: List[List[int]] = []

            if middle_seq is None or len(middle_seq) == 0:
                # Entire group is empty -> fill with padded visits
                for _ in range(self._max_middle_len):
                    group_encoded.append([pad_token] * self._max_inner_len)
                encoded_groups.append(group_encoded)
                continue

            for inner_seq in middle_seq:
                # inner_seq is a list of codes (or None)
                if inner_seq is None or len(inner_seq) == 0:
                    group_encoded.append([pad_token] * self._max_inner_len)
                    continue

                indices: List[int] = []

                for code in inner_seq:
                    if code is None or code not in self.code_vocab:
                        indices.append(self.code_vocab["<unk>"])
                    else:
                        indices.append(self.code_vocab[code])

                # Pad codes dimension to max_inner_len
                while len(indices) < self._max_inner_len:
                    indices.append(pad_token)

                group_encoded.append(indices)

            # Pad visits dimension to max_middle_len
            while len(group_encoded) < self._max_middle_len:
                group_encoded.append([pad_token] * self._max_inner_len)

            encoded_groups.append(group_encoded)

        return torch.tensor(encoded_groups, dtype=torch.long)

    def vocab_size(self) -> int:
        """Return the size of the processor's vocabulary."""
        return len(self.code_vocab)

    def size(self) -> int:
        """Return max inner length (embedding dimension) for unified API."""
        return self._max_inner_len

    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.code_vocab)

    def __repr__(self):
        return (
            f"DeepNestedSequenceProcessor("
            f"vocab_size={len(self.code_vocab)}, "
            f"max_middle_len={self._max_middle_len}, "
            f"max_inner_len={self._max_inner_len})"
        )

    def is_token(self) -> bool:
        """Deep nested sequence codes are discrete token indices."""
        return True

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        """Output is a 3D tensor (groups, visits, codes)."""
        return (3,)

    def spatial(self) -> tuple[bool, ...]:
        # Groups are not sequential; visits are temporal/spatial; codes-per-visit is an unordered set
        return (False, True, False)


@register_processor("deep_nested_sequence_floats")
class DeepNestedFloatsProcessor(FeatureProcessor):
    """
    Feature processor for 3-level nested numerical sequences without vocabulary.

    Handles deep nested sequences of floats/numerical values where each sample
    contains a list of groups, each group contains a list of visits, and each
    visit contains a list of values:

        [
            [[1.5, 2.3], [4.1]],
            [[0.9, 1.2, 3.4]]
        ]

    The processor:
    1. Determines:
        - max_visits_per_group (middle dimension)
        - max_values_per_visit (inner dimension)
       during fit.
    2. Optionally applies forward-fill for missing values.
    3. Returns a 3D tensor of shape:
        (num_groups, max_visits_per_group, max_values_per_visit)

    Args:
        forward_fill: If True, applies forward fill for NaN values across
            time steps (visits) and within visits. If False, sets null values
            to 0. Default is True.

    Examples:
        >>> processor = DeepNestedFloatsProcessor()
        >>> samples = [
        ...     {"values": [[[1.0, 2.0], [3.0, 4.0, 5.0]]]},
        ...     {"values": [[[6.0]]]},
        ... ]
        >>> processor.fit(samples, "values")
        >>> result = processor.process([[[1.0, 2.0], [3.0]]])
        >>> result.shape  # (1, max_visits_per_group, max_values_per_visit)
    """

    def __init__(self, forward_fill: bool = True):
        self._max_middle_len = 1  # Maximum length of middle sequences (visits)
        self._max_inner_len = 1   # Maximum length of inner sequences (values per visit)
        self.forward_fill = forward_fill

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
        """Determine maximum sequence lengths.

        Args:
            samples: List of sample dictionaries.
            field: The field name containing deep nested sequences.
        """
        max_middle_len = 0
        max_inner_len = 0

        for sample in samples:
            if field in sample and sample[field] is not None:
                deep_nested = sample[field]

                # Deep nested sequences: [[[1.0, 2.0], [3.0]], ...]
                if isinstance(deep_nested, list):
                    for middle_seq in deep_nested:  # per group
                        if isinstance(middle_seq, list):
                            max_middle_len = max(max_middle_len, len(middle_seq))

                            for inner_seq in middle_seq:  # per visit
                                if isinstance(inner_seq, list):
                                    max_inner_len = max(max_inner_len, len(inner_seq))

        self._max_middle_len = max(1, max_middle_len)
        self._max_inner_len = max(1, max_inner_len)

    def process(self, value: List[List[List[float]]]) -> torch.Tensor:
        """Process deep nested numerical sequence with optional forward fill.

        For missing values (None or empty visits/groups):
        - If forward_fill=True: uses forward fill within each group
        - If forward_fill=False: sets null values to 0.0 (for masking)

        Args:
            value: Deep nested list of floats [[[1.0, 2.0], [3.0]], ...]

        Returns:
            3D tensor of shape
            (num_groups, max_middle_len, max_inner_len) with float values
        """
        import numpy as np

        # Handle completely empty deep nested sequence
        if not value or len(value) == 0:
            if self.forward_fill:
                return torch.full(
                    (1, 1, self._max_inner_len), float("nan"), dtype=torch.float
                )
            else:
                return torch.zeros((1, 1, self._max_inner_len), dtype=torch.float)

        encoded_groups: List[List[List[float]]] = []

        for middle_seq in value:
            group_encoded: List[List[float]] = []
            last_valid_values = None  # within this group

            # If middle_seq is not a list, treat as empty group
            visits = middle_seq if isinstance(middle_seq, list) else []

            for inner_seq in visits:
                # inner_seq is a list of values (or None)
                if inner_seq is None or len(inner_seq) == 0:
                    # Empty visit
                    if self.forward_fill and last_valid_values is not None:
                        group_encoded.append(last_valid_values.copy())
                    else:
                        group_encoded.append([0.0] * self._max_inner_len)
                    continue

                values: List[float] = []

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

                # Pad inner dimension
                while len(values) < self._max_inner_len:
                    if self.forward_fill:
                        values.append(float("nan"))
                    else:
                        values.append(0.0)

                last_valid_values = values.copy()
                group_encoded.append(values)

            # Pad visits dimension to max_middle_len
            while len(group_encoded) < self._max_middle_len:
                if self.forward_fill and last_valid_values is not None:
                    group_encoded.append(last_valid_values.copy())
                else:
                    group_encoded.append([0.0] * self._max_inner_len)

            encoded_groups.append(group_encoded)

        # Convert to numpy array: (num_groups, max_middle_len, max_inner_len)
        values_array = np.array(encoded_groups, dtype=float)

        if self.forward_fill:
            num_groups, max_middle, max_inner = values_array.shape

            # 1) Forward fill across visits within each group (column-wise)
            for g_idx in range(num_groups):
                for feature_idx in range(max_inner):
                    last_value = None
                    for visit_idx in range(max_middle):
                        v = values_array[g_idx, visit_idx, feature_idx]
                        if not np.isnan(v):
                            last_value = v
                        elif last_value is not None:
                            values_array[g_idx, visit_idx, feature_idx] = last_value

                # 2) Forward fill within each visit (row-wise)
                for visit_idx in range(max_middle):
                    last_value = None
                    for feature_idx in range(max_inner):
                        v = values_array[g_idx, visit_idx, feature_idx]
                        if not np.isnan(v):
                            last_value = v
                        elif last_value is not None:
                            values_array[g_idx, visit_idx, feature_idx] = last_value

            # 3) Any remaining NaN values are set to 0.0
            values_array = np.nan_to_num(values_array, nan=0.0)

        return torch.tensor(values_array, dtype=torch.float)

    def size(self) -> int:
        """Return max inner length (last dimension) for unified API."""
        return self._max_inner_len

    def __repr__(self):
        return (
            f"DeepNestedFloatsProcessor("
            f"max_middle_len={self._max_middle_len}, "
            f"max_inner_len={self._max_inner_len}, "
            f"forward_fill={self.forward_fill})"
        )

    def is_token(self) -> bool:
        """Deep nested float values are continuous, not discrete tokens."""
        return False

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        """Output is a 3D tensor (groups, visits, features)."""
        return (3,)

    def spatial(self) -> tuple[bool, ...]:
        # Groups are not sequential; visits are temporal/spatial; features dimension is not
        return (False, True, False)
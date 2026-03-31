from typing import Any, Dict, List, Optional, Tuple, Iterable

import torch

from . import register_processor
from .base_processor import FeatureProcessor, ModalityType, TemporalFeatureProcessor, TokenProcessorInterface


@register_processor("stagenet")
class StageNetProcessor(TemporalFeatureProcessor, TokenProcessorInterface):
    """
    Feature processor for StageNet CODE inputs with coupled value/time data.

    This processor handles categorical code sequences (flat or nested).
    For numeric features, use StageNetTensorProcessor instead.

    Input Format (tuple):
        (time, values) where:
        - time: List of scalars [0.0, 2.0, 1.3] or None
        - values: ["code1", "code2"] or [["A", "B"], ["C"]]

    The processor automatically detects:
    - List of strings -> flat code sequences
    - List of lists of strings -> nested code sequences

    Args:
        padding: Additional padding to add on top of the observed maximum nested
            sequence length. The actual padding length will be observed_max + padding.
            This ensures the processor can handle sequences longer than those in the
            training data. Default: 0 (no extra padding). Only applies to nested sequences.

    Returns:
        Tuple of (time_tensor, value_tensor) where time_tensor can be None

    Examples:
        >>> # Case 1: Code sequence with time
        >>> processor = StageNetProcessor()
        >>> data = ([0.0, 1.5, 2.3], ["code1", "code2", "code3"])
        >>> time, values = processor.process(data)
        >>> values.shape  # (3,) - sequence of code indices
        >>> time.shape    # (3,) - time intervals

        >>> # Case 2: Nested codes with time (with custom padding for extra capacity)
        >>> processor = StageNetProcessor(padding=20)
        >>> data = ([0.0, 1.5], [["A", "B"], ["C"]])
        >>> time, values = processor.process(data)
        >>> values.shape  # (2, observed_max + 20) - padded nested sequences
        >>> time.shape    # (2,)

        >>> # Case 3: Codes without time
        >>> data = (None, ["code1", "code2"])
        >>> time, values = processor.process(data)
        >>> values.shape  # (2,)
        >>> time          # None
    """

    def __init__(self, padding: int = 0):
        self.code_vocab: Dict[Any, int] = {"<pad>": self.PAD, "<unk>": self.UNK}
        self._next_index = 2
        self._is_nested = None  # Will be determined during fit
        # Max inner sequence length for nested codes
        self._max_nested_len = None
        self._padding = padding  # Additional padding beyond observed max

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
        """Build vocabulary and determine input structure.

        Args:
            samples: List of sample dictionaries
            key: The key in samples that contains tuple (time, values)
        """
        # Examine first non-None sample to determine structure
        for sample in samples:
            if field in sample and sample[field] is not None:
                # Unpack tuple: (time, values)
                time_data, value_data = sample[field]

                # Determine nesting level for codes
                if isinstance(value_data, list) and len(value_data) > 0:
                    first_elem = value_data[0]

                    if isinstance(first_elem, str):
                        # Case 1: ["code1", "code2", ...]
                        self._is_nested = False
                    elif isinstance(first_elem, list):
                        if len(first_elem) > 0 and isinstance(first_elem[0], str):
                            # Case 2: [["A", "B"], ["C"], ...]
                            self._is_nested = True
                break

        # Build vocabulary for codes and find max nested length
        max_inner_len = 0
        for sample in samples:
            if field in sample and sample[field] is not None:
                # Unpack tuple: (time, values)
                time_data, value_data = sample[field]

                if self._is_nested:
                    # Nested codes
                    for inner_list in value_data:
                        # Track max inner length
                        max_inner_len = max(max_inner_len, len(inner_list))
                        for code in inner_list:
                            if code is not None and code not in self.code_vocab:
                                self.code_vocab[code] = self._next_index
                                self._next_index += 1
                else:
                    # Flat codes
                    for code in value_data:
                        if code is not None and code not in self.code_vocab:
                            self.code_vocab[code] = self._next_index
                            self._next_index += 1

        # Store max nested length: add user-specified padding to observed maximum
        # This ensures the processor can handle sequences longer than those in training data
        if self._is_nested:
            observed_max = max(1, max_inner_len)
            self._max_nested_len = observed_max + self._padding

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

    def process(
        self, value: Tuple[Optional[List], List]
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Process tuple format data into tensors.

        Args:
            value: Tuple of (time, values) where values are codes

        Returns:
            Tuple of (time_tensor, value_tensor), time can be None
        """
        # Unpack tuple: (time, values)
        time_data, value_data = value

        # Encode codes to indices
        if self._is_nested:
            # Nested codes: [["A", "B"], ["C"]]
            value_tensor = self._encode_nested_codes(value_data)
        else:
            # Flat codes: ["code1", "code2"]
            value_tensor = self._encode_codes(value_data)

        # Process time if present
        time_tensor = None
        if time_data is not None and len(time_data) > 0:
            # Handle both [0.0, 1.5] and [[0.0], [1.5]] formats
            if isinstance(time_data[0], list):
                # Flatten [[0.0], [1.5]] -> [0.0, 1.5]
                time_data = [t[0] if isinstance(t, list) else t for t in time_data]
            time_tensor = torch.tensor(time_data, dtype=torch.float)

        return (time_tensor, value_tensor)

    def _encode_codes(self, codes: List[str]) -> torch.Tensor:
        """Encode flat code list to indices."""
        # Handle empty code list - return single padding token
        if len(codes) == 0:
            return torch.tensor([self.code_vocab["<pad>"]], dtype=torch.long)

        indices = []
        for code in codes:
            if code is None or code not in self.code_vocab:
                indices.append(self.code_vocab["<unk>"])
            else:
                indices.append(self.code_vocab[code])
        return torch.tensor(indices, dtype=torch.long)

    def _encode_nested_codes(self, nested_codes: List[List[str]]) -> torch.Tensor:
        """Encode nested code lists to padded 2D tensor.

        Pads all inner sequences to self._max_nested_len (global max).
        """
        assert self._max_nested_len is not None, "Max nested length must be set during fit()"
        
        # Handle empty nested codes (no visits/events)
        # Return single padding token with shape (1, max_len)
        if len(nested_codes) == 0:
            pad_token = self.code_vocab["<pad>"]
            return torch.tensor([[pad_token] * self._max_nested_len], dtype=torch.long)

        encoded_sequences = []
        # Use global max length determined during fit
        max_len = self._max_nested_len

        for inner_codes in nested_codes:
            indices = []
            for code in inner_codes:
                if code is None or code not in self.code_vocab:
                    indices.append(self.code_vocab["<unk>"])
                else:
                    indices.append(self.code_vocab[code])
            # Pad to GLOBAL max_len
            while len(indices) < max_len:
                indices.append(self.code_vocab["<pad>"])
            encoded_sequences.append(indices)

        return torch.tensor(encoded_sequences, dtype=torch.long)

    def vocab_size(self) -> int:
        """Return the size of the processor's vocabulary."""
        return len(self.code_vocab)

    def size(self) -> int:
        """Return vocabulary size."""
        return len(self.code_vocab)

    def is_token(self) -> bool:
        """Code indices are discrete token indices."""
        return True

    def schema(self) -> tuple[str, ...]:
        """Output is a tuple of (time_tensor, value_tensor)."""
        return ("time", "value")

    def dim(self) -> tuple[int, ...]:
        """Number of dimensions for each output tensor.

        Time tensor is 1D. Value tensor is 1D (flat) or 2D (nested).
        Must be called after fit().

        Returns:
            (1, 1) for flat codes or (1, 2) for nested codes.
        """
        if self._is_nested is None:
            raise NotImplementedError(
                "StageNetProcessor.dim() requires fit() to be called first "
                "to determine whether codes are flat or nested."
            )
        if self._is_nested:
            return (1, 2)
        return (1, 1)

    def spatial(self) -> tuple[bool, ...]:
        """Whether each dimension of the value tensor is spatial."""
        if self._is_nested is None:
            raise NotImplementedError(
                "StageNetProcessor.spatial() requires fit() to be called first."
            )
        if self._is_nested:
            # (visits, codes_per_visit) - visits are sequential/spatial,
            # codes_per_visit is an unordered set and not spatial
            return (True, False)
        # Flat codes: single sequence dimension is spatial
        return (True,)

    def modality(self) -> ModalityType:
        """Discrete EHR codes → CODE modality."""
        return ModalityType.CODE

    def value_dim(self) -> int:
        """Vocabulary size (used with nn.Embedding in UnifiedMultimodalEmbeddingModel).
        Must be called after fit()."""
        return len(self.code_vocab)

    def process_temporal(self, value) -> dict:
        """Return dict output for UnifiedMultimodalEmbeddingModel.

        Calls the existing process() (backward-compatible tuple) and wraps
        the result as a dict with 'value' and 'time' keys.

        Returns:
            {"value": LongTensor (S,), "time": FloatTensor (S,) or None}
        """
        time_tensor, value_tensor = self.process(value)
        return {"value": value_tensor, "time": time_tensor}

    def __repr__(self):
        if self._is_nested:
            return (
                f"StageNetProcessor(is_nested={self._is_nested}, "
                f"vocab_size={len(self.code_vocab)}, "
                f"max_nested_len={self._max_nested_len}, "
                f"padding={self._padding})"
            )
        else:
            return (
                f"StageNetProcessor(is_nested={self._is_nested}, "
                f"vocab_size={len(self.code_vocab)}, "
                f"padding={self._padding})"
            )


@register_processor("stagenet_tensor")
class StageNetTensorProcessor(TemporalFeatureProcessor):
    """
    Feature processor for StageNet NUMERIC inputs with coupled value/time data.

    This processor handles numeric feature sequences (flat or nested) and applies
    forward-fill imputation to handle missing values (NaN/None).
    For categorical codes, use StageNetProcessor instead.

    Format:
    {
        "value": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # nested numerics
        "time": [0.0, 1.5] or None
    }

    The processor automatically detects:
    - List of numbers -> flat numeric sequences
    - List of lists of numbers -> nested numeric sequences (feature vectors)

    Imputation Strategy:
    - Forward-fill: Missing values (NaN/None) are filled with the last observed
      value for that feature dimension. If no prior value exists, 0.0 is used.
    - Applied per feature dimension independently

    Returns:
        Tuple of (time_tensor, value_tensor) where time_tensor can be None

    Examples:
        >>> # Case 1: Feature vectors with missing values
        >>> processor = StageNetTensorProcessor()
        >>> data = {
        ...     "value": [[1.0, None, 3.0], [None, 5.0, 6.0], [7.0, 8.0, None]],
        ...     "time": [0.0, 1.5, 3.0]
        ... }
        >>> time, values = processor.process(data)
        >>> values  # [[1.0, 0.0, 3.0], [1.0, 5.0, 6.0], [7.0, 8.0, 6.0]]
        >>> values.dtype  # torch.float32
        >>> time.shape    # (3,)
    """

    def __init__(self):
        self._size = None  # Feature dimension (set during fit)
        self._is_nested = None

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
        """Determine input structure.

        Args:
            samples: List of sample dictionaries
            key: The key in samples that contains tuple (time, values)
        """
        # Examine first non-None sample to determine structure
        for sample in samples:
            if field in sample and sample[field] is not None:
                # Unpack tuple: (time, values)
                time_data, value_data = sample[field]

                # Determine nesting level for numerics
                if isinstance(value_data, list) and len(value_data) > 0:
                    first_elem = value_data[0]

                    if isinstance(first_elem, (int, float)):
                        # Flat numeric: [1.5, 2.0, ...]
                        self._is_nested = False
                        self._size = 1
                    elif isinstance(first_elem, list):
                        if len(first_elem) > 0:
                            if isinstance(first_elem[0], (int, float)):
                                # Nested numerics: [[1.0, 2.0], [3.0, 4.0]]
                                self._is_nested = True
                                self._size = len(first_elem)
                break

    def process(
        self, value: Tuple[Optional[List], List]
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Process tuple format numeric data into tensors.

        Applies forward-fill imputation to handle NaN/None values.
        For each feature dimension, missing values are filled with the
        last observed value (or 0.0 if no prior value exists).

        Args:
            value: Tuple of (time, values) where values are numerics

        Returns:
            Tuple of (time_tensor, value_tensor), time can be None
        """
        # Unpack tuple: (time, values)
        time_data, value_data = value

        # Convert to numpy for easier imputation handling
        import numpy as np

        value_array = np.array(value_data, dtype=float)

        # Apply forward-fill imputation
        if value_array.ndim == 1:
            # Flat numeric: [1.5, 2.0, nan, 3.0, ...]
            last_value = 0.0
            for i in range(len(value_array)):
                if not np.isnan(value_array[i]):
                    last_value = value_array[i]
                else:
                    value_array[i] = last_value
        elif value_array.ndim == 2:
            # Feature vectors: [[1.0, nan, 3.0], [nan, 5.0, 6.0]]
            num_features = value_array.shape[1]
            for f in range(num_features):
                last_value = 0.0
                for t in range(value_array.shape[0]):
                    if not np.isnan(value_array[t, f]):
                        last_value = value_array[t, f]
                    else:
                        value_array[t, f] = last_value

        # Convert to float tensor
        value_tensor = torch.tensor(value_array, dtype=torch.float)

        # Process time if present
        time_tensor = None
        if time_data is not None and len(time_data) > 0:
            # Handle both [0.0, 1.5] and [[0.0], [1.5]] formats
            if isinstance(time_data[0], list):
                # Flatten [[0.0], [1.5]] -> [0.0, 1.5]
                time_data = [t[0] if isinstance(t, list) else t for t in time_data]
            time_tensor = torch.tensor(time_data, dtype=torch.float)

        return (time_tensor, value_tensor)

    def size(self):
        """Return feature dimension."""
        return self._size

    def is_token(self) -> bool:
        """Numeric values are continuous, not discrete tokens."""
        return False

    def schema(self) -> tuple[str, ...]:
        """Output is a tuple of (time_tensor, value_tensor)."""
        return ("time", "value")

    def dim(self) -> tuple[int, ...]:
        """Number of dimensions for each output tensor.

        Time tensor is 1D. Value tensor is 1D (flat) or 2D (nested).
        Must be called after fit().

        Returns:
            (1, 1) for flat values or (1, 2) for nested values.
        """
        if self._is_nested is None:
            raise NotImplementedError(
                "StageNetTensorProcessor.dim() requires fit() to be called first "
                "to determine whether values are flat or nested."
            )
        if self._is_nested:
            return (1, 2)
        return (1, 1)

    def spatial(self) -> tuple[bool, ...]:
        """Whether each dimension of the value tensor is spatial."""
        if self._is_nested is None:
            raise NotImplementedError(
                "StageNetTensorProcessor.spatial() requires fit() to be called first."
            )
        if self._is_nested:
            # (time_steps, features) - time is spatial, features are not
            return (True, False)
        # Flat: single sequence dimension is spatial
        return (True,)

    def modality(self) -> ModalityType:
        """Continuous lab/vital measurements → NUMERIC modality."""
        return ModalityType.NUMERIC

    def value_dim(self) -> int:
        """Number of numeric features per time-step (used with nn.Linear).
        Must be called after fit()."""
        return self._size if self._size is not None else 1

    def process_temporal(self, value) -> dict:
        """Return dict output for UnifiedMultimodalEmbeddingModel.

        Returns:
            {"value": FloatTensor (T, F), "time": FloatTensor (T,) or None}
        """
        time_tensor, value_tensor = self.process(value)
        return {"value": value_tensor, "time": time_tensor}

    def __repr__(self):
        return (
            f"StageNetTensorProcessor(is_nested={self._is_nested}, "
            f"feature_dim={self._size})"
        )

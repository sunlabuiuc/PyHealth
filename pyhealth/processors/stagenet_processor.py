from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch

from . import register_processor
from .base_processor import FeatureProcessor


@dataclass
class StageNetFeature:
    """Container for StageNet feature with values and optional time intervals.

    Attributes:
        value: The feature tensor (1D for sequences, 2D for nested sequences, 3D for feature vectors)
        time: Optional time interval tensor (1D, matching the sequence length of value)
    """

    value: torch.Tensor
    time: Optional[torch.Tensor] = None


@register_processor("stagenet")
class StageNetProcessor(FeatureProcessor):
    """
    Feature processor for StageNet CODE inputs with coupled value/time data.

    This processor handles categorical code sequences (flat or nested).
    For numeric features, use StageNetTensorProcessor instead.

    Format:
    {
        "value": ["code1", "code2"] or [["A", "B"], ["C"]],
        "time": [0.0, 2.0, 1.3] or None
    }

    The processor automatically detects:
    - List of strings -> flat code sequences
    - List of lists of strings -> nested code sequences

    Time intervals should be simple lists of scalars, one per sequence position.

    Examples:
        >>> # Case 1: Code sequence with time
        >>> processor = StageNetProcessor()
        >>> data = {"value": ["code1", "code2", "code3"], "time": [0.0, 1.5, 2.3]}
        >>> result = processor.process(data)
        >>> result.value.shape  # (3,) - sequence of code indices
        >>> result.time.shape   # (3,) - time intervals

        >>> # Case 2: Nested codes with time
        >>> data = {"value": [["A", "B"], ["C"]], "time": [0.0, 1.5]}
        >>> result = processor.process(data)
        >>> result.value.shape  # (2, max_inner_len) - padded nested sequences
        >>> result.time.shape   # (2,)

        >>> # Case 3: Feature vectors without time
        >>> data = {"value": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "time": None}
        >>> result = processor.process(data)
        >>> result.value.shape  # (2, 3)
        >>> result.time         # None
    """

    def __init__(self):
        self.code_vocab: Dict[Any, int] = {"<unk>": -1, "<pad>": 0}
        self._next_index = 1
        self._is_nested = None  # Will be determined during fit

    def fit(self, samples: List[Dict], key: str) -> None:
        """Build vocabulary and determine input structure.

        Args:
            samples: List of sample dictionaries
            key: The key in samples that contains StageNet format data
        """
        # Examine first non-None sample to determine structure
        for sample in samples:
            if key in sample and sample[key] is not None:
                value_data = sample[key]["value"]

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

        # Build vocabulary for codes
        for sample in samples:
            if key in sample and sample[key] is not None:
                value_data = sample[key]["value"]

                if self._is_nested:
                    # Nested codes
                    for inner_list in value_data:
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

    def process(self, value: Dict[str, Any]) -> StageNetFeature:
        """Process StageNet format data into tensors.

        Args:
            value: Dictionary with "value" and optional "time" keys

        Returns:
            StageNetFeature with value and time tensors
        """
        value_data = value["value"]
        time_data = value.get("time", None)

        # Encode codes to indices
        if self._is_nested:
            # Nested codes: [["A", "B"], ["C"]]
            value_tensor = self._encode_nested_codes(value_data)
        else:
            # Flat codes: ["code1", "code2"]
            value_tensor = self._encode_codes(value_data)

        # Process time if present
        time_tensor = None
        if time_data is not None:
            # Handle both [0.0, 1.5] and [[0.0], [1.5]] formats
            if isinstance(time_data[0], list):
                # Flatten [[0.0], [1.5]] -> [0.0, 1.5]
                time_data = [t[0] if isinstance(t, list) else t for t in time_data]
            time_tensor = torch.tensor(time_data, dtype=torch.float)

        return StageNetFeature(value=value_tensor, time=time_tensor)

    def _encode_codes(self, codes: List[str]) -> torch.Tensor:
        """Encode flat code list to indices."""
        indices = []
        for code in codes:
            if code is None or code not in self.code_vocab:
                indices.append(self.code_vocab["<unk>"])
            else:
                indices.append(self.code_vocab[code])
        return torch.tensor(indices, dtype=torch.long)

    def _encode_nested_codes(self, nested_codes: List[List[str]]) -> torch.Tensor:
        """Encode nested code lists to padded 2D tensor."""
        encoded_sequences = []
        max_len = max(len(inner) for inner in nested_codes)

        for inner_codes in nested_codes:
            indices = []
            for code in inner_codes:
                if code is None or code not in self.code_vocab:
                    indices.append(self.code_vocab["<unk>"])
                else:
                    indices.append(self.code_vocab[code])
            # Pad to max_len
            while len(indices) < max_len:
                indices.append(self.code_vocab["<pad>"])
            encoded_sequences.append(indices)

        return torch.tensor(encoded_sequences, dtype=torch.long)

    def size(self) -> int:
        """Return vocabulary size."""
        return len(self.code_vocab)

    def __repr__(self):
        return (
            f"StageNetProcessor(is_nested={self._is_nested}, "
            f"vocab_size={len(self.code_vocab)})"
        )


@register_processor("stagenet_tensor")
class StageNetTensorProcessor(FeatureProcessor):
    """
    Feature processor for StageNet NUMERIC inputs with coupled value/time data.

    This processor handles numeric feature sequences (flat or nested).
    For categorical codes, use StageNetProcessor instead.

    Format:
    {
        "value": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # nested numerics
        "time": [0.0, 1.5] or None
    }

    The processor automatically detects:
    - List of numbers -> flat numeric sequences
    - List of lists of numbers -> nested numeric sequences (feature vectors)

    Examples:
        >>> # Case 1: Feature vectors without time
        >>> processor = StageNetTensorProcessor()
        >>> data = {"value": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "time": None}
        >>> result = processor.process(data)
        >>> result.value.shape  # (2, 3)
        >>> result.value.dtype  # torch.float32
        >>> result.time         # None
    """

    def __init__(self):
        self._size = None  # Feature dimension (set during fit)
        self._is_nested = None

    def fit(self, samples: List[Dict], key: str) -> None:
        """Determine input structure.

        Args:
            samples: List of sample dictionaries
            key: The key in samples that contains StageNet format data
        """
        # Examine first non-None sample to determine structure
        for sample in samples:
            if key in sample and sample[key] is not None:
                value_data = sample[key]["value"]

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

    def process(self, value: Dict[str, Any]) -> StageNetFeature:
        """Process StageNet format numeric data into tensors.

        Args:
            value: Dictionary with "value" and optional "time" keys

        Returns:
            StageNetFeature with value and time tensors
        """
        value_data = value["value"]
        time_data = value.get("time", None)

        # Convert to float tensor
        value_tensor = torch.tensor(value_data, dtype=torch.float)

        # Process time if present
        time_tensor = None
        if time_data is not None:
            # Handle both [0.0, 1.5] and [[0.0], [1.5]] formats
            if isinstance(time_data[0], list):
                # Flatten [[0.0], [1.5]] -> [0.0, 1.5]
                time_data = [t[0] if isinstance(t, list) else t for t in time_data]
            time_tensor = torch.tensor(time_data, dtype=torch.float)

        return StageNetFeature(value=value_tensor, time=time_tensor)

    @property
    def size(self):
        """Return feature dimension."""
        return self._size

    def __repr__(self):
        return (
            f"StageNetTensorProcessor(is_nested={self._is_nested}, "
            f"feature_dim={self._size})"
        )

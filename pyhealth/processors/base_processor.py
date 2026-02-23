from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Iterable

import torch


class ModalityType(str, Enum):
    """Standard modality identifiers for routing in UnifiedMultimodalEmbeddingModel.

    Using ``str, Enum`` so values serialise cleanly (e.g. JSON / pickle).
    """
    CODE    = "code"     # Discrete ICD / medication / procedure codes
    TEXT    = "text"     # Clinical notes, reports (tokenised to int tensors)
    IMAGE   = "image"    # Medical images (X-ray, CT slice, etc.)
    NUMERIC = "numeric"  # Lab values, vitals, continuous measurements
    AUDIO   = "audio"    # Heart/lung sounds, speech waveforms
    SIGNAL  = "signal"   # ECG, EEG time-series waveforms


class Processor(ABC):
    """
    Abstract base processor class.

    Defines optional hooks for saving/loading state to/from disk.
    """

    def save(self, path: str) -> None:
        """Optional: Save processor state to disk.

        Args:
            path: File path to save processor state.
        """
        pass

    def load(self, path: str) -> None:
        """Optional: Load processor state from disk.

        Args:
            path: File path to load processor state from.
        """
        pass


class FeatureProcessor(Processor):
    """
    Processor for individual fields (features).

    Example: Tokenization, image loading, normalization.
    """

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
        """Fit the processor to the samples.

        Args:
            samples: List of sample dictionaries.
        """
        pass

    @abstractmethod
    def process(self, value: Any) -> Any:
        """Process an individual field value.

        Args:
            value: Raw field value.

        Returns:
            Processed value.
        """
        pass
    
    def is_token(self) -> bool:
        """Returns whether the output (in particular, the value tensor) of the processor 
        represents discrete token indices (True) or continuous values (False). This is used to 
        determine whether to apply token-based transformations (e.g. `nn.Embedding`) or 
        value-based augmentations (e.g. `nn.Linear`). 

        Returns:
            True if the output of the processor represents discrete token indices, False otherwise.
        """
        raise NotImplementedError("is_token method is not implemented for this processor.")
    
    def schema(self) -> tuple[str, ...]:
        """Returns the schema of the processed feature. For a processor that emits a single tensor,
        this should just return `["value"]`. For a processor that emits a tuple of tensors, 
        this should return a tuple of the same length as the tuple, with the semantic name of each tensor,
        such as `["time", "value"]`, `["value", "mask"]`, etc.
        
        Typical semantic names include:
            - "value": the main processed tensor output of the processor
            - "time": the time tensor output of the processor (mostly for StageNet)
            - "mask": the mask tensor output of the processor (if applicable)
        
        Returns:
            Tuple of semantic names corresponding to the output of the processor.
        """
        raise NotImplementedError("Schema method is not implemented for this processor.")
    
    def dim(self) -> tuple[int, ...]:
        """Number of dimensions (`Tensor.dim()`) for each output
        tensor, in the same order as the output tuple.

        Returns:
            Tuple of integers corresponding to the number of dimensions of each output tensor.
        """
        raise NotImplementedError("dim method is not implemented for this processor.")
    
    def spatial(self) -> tuple[bool, ...]:
        """Whether each dimension (axis) of the value tensor is spatial (i.e. corresponds to a spatial 
        axis like time, height, width, etc.) or not. This is used to determine how to apply 
        augmentations and other transformations that should only be applied to spatial dimensions.
        
        E.g. for CNN or RNN features, this would help determine which dimensions to apply spatial augmentations to, 
        and which dimensions to treat as channels or features.
        
        Returns:
            Tuple of booleans corresponding to whether each axis of the value tensor is spatial or not.
        """
        raise NotImplementedError("spatial method is not implemented for this processor.")
    
    


class SampleProcessor(Processor):
    """
    Processor for individual samples (dict of fields).

    Example: Imputation, sample-level augmentation, label smoothing.
    """

    @abstractmethod
    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample dictionary.

        Args:
            sample: Sample dictionary.

        Returns:
            Processed sample dictionary.
        """
        pass


class DatasetProcessor(Processor):
    """
    Processor for the full dataset.

    Example: Global normalization, train/val/test splitting, dataset caching.
    """

    @abstractmethod
    def process(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process the entire dataset.

        Args:
            samples: List of sample dictionaries.

        Returns:
            List of processed sample dictionaries.
        """
        pass

class TokenProcessorInterface(ABC):
    """
    Base class for feature processors that build a vocabulary.

    Provides a common interface for accessing vocabulary-related information.
    """
    
    PAD = 0
    UNK = 1

    @abstractmethod
    def remove(self, tokens: set[str]):
        """Remove specified vocabularies from the processor."""
        pass
    
    @abstractmethod
    def retain(self, tokens: set[str]):
        """Retain only the specified vocabularies in the processor."""
        pass
    
    @abstractmethod
    def add(self, tokens: set[str]):
        """Add specified vocabularies to the processor."""
        pass
    
    @abstractmethod
    def tokens(self) -> set[str]:
        """Return the set of tokens in the processor's vocabulary."""
        pass
    
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the size of the processor's vocabulary."""
        pass


class TemporalFeatureProcessor(FeatureProcessor):
    """Abstract base class for processors whose features are paired with timestamps.

    **Contract** — every subclass must implement:

    - ``modality() -> ModalityType``  — what kind of data this processor handles.
    - ``value_dim() -> int``          — size of the raw value vector *before* any
      learned embedding (e.g. vocab_size for codes, n_features for numerics).
    - ``process(value) -> dict[str, torch.Tensor]``  — must return a dict with at
      least the keys ``"value"`` and ``"time"``, and optionally ``"mask"``.

    **Backward compatibility** — the existing ``FeatureProcessor`` API
    (``is_token``, ``schema``, ``dim``, ``spatial``) is *kept* on the parent class
    and continues to work for all non-temporal processors.  Subclasses of
    ``TemporalFeatureProcessor`` should still implement those methods if they want
    to remain compatible with the existing ``EmbeddingModel`` / ``MultimodalRNN``
    pipeline.  The new ``modality()`` / ``value_dim()`` API is *additive* — used
    exclusively by ``UnifiedMultimodalEmbeddingModel``.

    **Why dict output?**

    ================ ======================== ==================================
    Concern          Tuple (current)          Dict (this class)
    ================ ======================== ==================================
    Collation        Custom per arity         Generic: stack/pad per key
    litdata          List[str] breaks         All values tensors/scalars ✓
    Schema           Positional, fragile      Named keys, self-documenting
    Extensibility    Adding field = new arity Adding key = backward-compat
    ================ ======================== ==================================
    """

    # ── New API (required by UnifiedMultimodalEmbeddingModel) ─────────────────

    @abstractmethod
    def modality(self) -> ModalityType:
        """Return the modality type of the data this processor handles."""
        ...

    @abstractmethod
    def value_dim(self) -> int:
        """Dimensionality of the raw value vector *before* learned embedding.

        For codes:   ``vocab_size``  (used with ``nn.Embedding``)
        For images:  ``C * H * W``   (used with CNN encoder)
        For numerics: ``n_features`` (used with ``nn.Linear``)
        For text:    ``vocab_size``  (used with transformer encoder)
        """
        ...

    @abstractmethod
    def process(self, value) -> dict[str, torch.Tensor]:
        """Process raw input and return a dict of tensors.

        Required keys:
            ``"value"``  — main feature tensor.
            ``"time"``   — 1-D float32 tensor, one timestamp per event.

        Optional keys:
            ``"mask"``   — validity / attention mask for ``"value"``.
        """
        ...

    def schema(self) -> tuple[str, ...]:
        """Standardised schema: at minimum ``('value', 'time')``."""
        return ("value", "time")

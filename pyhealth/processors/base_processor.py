from abc import ABC, abstractmethod
from enum import Enum
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Iterable

import torch

logger = logging.getLogger(__name__)


def _warn_truncated(processor: Any, what: str, got: int, limit: int) -> None:
    """Logs once per processor that an input was cut to the fitted width."""
    if getattr(processor, "_warned_truncation", False):
        return
    processor._warned_truncation = True
    logger.warning(
        "%s: an input has %d %s, more than the %d seen in fit(); keeping the "
        "first %d. Inputs longer than the fitted data are truncated so every "
        "sample has the same shape. Pass a larger `padding` to keep more.",
        type(processor).__name__, got, what, limit, limit,
    )


FieldKey = tuple[int, str]
"""A field of a processor's output: ``(level, role)``, see :meth:`FeatureProcessor.schema`."""


def _field_matches(entry: FieldKey | str, role: str, level: int) -> bool:
    # A bare role is the not-yet-migrated form of (0, role).
    if isinstance(entry, str):
        return level == 0 and entry == role
    return tuple(entry) == (level, role)


def has_field(
    schema: Sequence[FieldKey] | Sequence[str], role: str, level: int = 0
) -> bool:
    """Whether a processor's :meth:`~FeatureProcessor.schema` has a field.

    Accepts ``(level, role)`` keys and the bare roles of processors not
    migrated yet.

    Args:
        schema: The output of ``schema()``.
        role: The role to look for, e.g. ``"mask"``.
        level: The level of the field.

    Examples:
        >>> from pyhealth.processors import has_field
        >>> has_field(((0, "value"), (0, "mask")), "mask")
        True
        >>> has_field(("time", "value"), "mask")
        False
    """
    return any(_field_matches(entry, role, level) for entry in schema)


def field_index(
    schema: Sequence[FieldKey] | Sequence[str], role: str, level: int = 0
) -> int:
    """Position of a field in a processor's :meth:`~FeatureProcessor.schema`.

    Accepts ``(level, role)`` keys and the bare roles of processors not
    migrated yet.

    Args:
        schema: The output of ``schema()``.
        role: The role to look for, e.g. ``"value"``.
        level: The level of the field.

    Returns:
        The index of the field's tensor in the processor's output tuple.

    Raises:
        ValueError: If the schema has no such field.

    Examples:
        >>> from pyhealth.processors import field_index
        >>> field_index(((0, "time"), (0, "value")), "value")
        1
        >>> field_index(("time", "value"), "value")
        1
    """
    for i, entry in enumerate(schema):
        if _field_matches(entry, role, level):
            return i
    raise ValueError(f"No field {(level, role)} in schema {tuple(schema)}.")


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

    Set ``learns_statistics = True`` on processors whose ``fit()`` learns values
    from the data (means, standard deviations, medians, bins). Fitting such a
    processor on all samples leaks validation/test information, so PyHealth
    warns unless it is fitted on a training split
    (``set_task(task, split=PatientSplit(...))``).

    Set ``stores_tensor = True`` only if ``process()`` always returns a
    ``torch.Tensor``, or a tuple with one tensor per :meth:`schema` entry
    (each tensor may change shape between samples). Such fields are written
    to the disk cache as tensors; every other output (lists, other tuples,
    dicts, strings, None, objects) is stored as one pickled value per sample,
    so it may vary freely in length and shape between samples.

    A processor whose batches need more than padding can define
    ``collate(items)``, taking the ``process()`` outputs of one batch;
    :func:`~pyhealth.datasets.get_dataloader` then uses it for that field.

    Examples:
        >>> import torch
        >>> from pyhealth.processors.base_processor import FeatureProcessor
        >>> class MeanCenter(FeatureProcessor):
        ...     learns_statistics = True
        ...     def fit(self, samples, field):
        ...         values = [s[field] for s in samples]
        ...         self.mean = sum(values) / len(values)
        ...     def process(self, value):
        ...         return torch.tensor(value - self.mean)
        >>> p = MeanCenter()
        >>> p.fit([{"x": 1.0}, {"x": 3.0}], "x")
        >>> p.process(5.0)
        tensor(3.)
    """

    learns_statistics: bool = False
    stores_tensor: bool = False

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
    
    def schema(self) -> tuple[FieldKey, ...] | tuple[str, ...]:
        """Returns the fields of the processed feature, one per output tensor.

        Each field is a ``(level, role)`` key, in the order of the output
        tuple. A processor that emits a single tensor returns
        ``((0, "value"),)``; one that emits several returns e.g.
        ``((0, "time"), (0, "value"))``. Packed processors
        (:class:`~pyhealth.processors.SequenceProcessorV2`) use the level of
        each record; every other processor puts all fields at level 0.

        Typical roles include:
            - "value": the main processed tensor output of the processor
            - "time": the time tensor output of the processor (mostly for StageNet)
            - "mask": the mask tensor output of the processor (if applicable)

        Processors not migrated yet return the roles alone (e.g.
        ``("time", "value")``), meaning level 0. Read a schema with
        :func:`field_index` and :func:`has_field`, which accept both forms.

        Returns:
            Tuple of ``(level, role)`` keys, one per output tensor.
        """
        raise NotImplementedError("Schema method is not implemented for this processor.")
    
    def dim(self) -> tuple[int, ...]:
        """Number of dimensions (`Tensor.dim()`) for each output
        tensor, in the same order as the output tuple.

        Returns:
            Tuple of integers corresponding to the number of dimensions of each output tensor.
        """
        raise NotImplementedError("dim method is not implemented for this processor.")
    
    def spatial(self) -> Mapping[int, tuple[bool, ...] | None] | tuple[bool, ...]:
        """Whether each dimension (axis) of the value tensor is spatial (i.e. corresponds to a spatial
        axis like time, height, width, etc.) or not. This is used to determine how to apply
        augmentations and other transformations that should only be applied to spatial dimensions.

        E.g. for CNN or RNN features, this would help determine which dimensions to apply spatial augmentations to,
        and which dimensions to treat as channels or features.

        Keyed by level, like :meth:`schema`. At level 0 of a processor that
        is not packed, the flags cover every axis of the value tensor; for
        :class:`~pyhealth.processors.SequenceProcessorV2` they cover the
        trailing value shape of each level, or are None at a level without
        values. Processors not migrated yet return the level-0 tuple alone.

        Returns:
            Dict from level to a tuple of booleans, one per axis.
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

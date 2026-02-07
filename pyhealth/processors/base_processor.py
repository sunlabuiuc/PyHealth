from abc import ABC, abstractmethod
from typing import Any, Dict, List, Iterable


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
    
    def is_continuous(self) -> bool:
        """Returns whether the output (in particular, the value tensor) of the processor is 
        continuous (float) or discrete (int).

        Returns:
            True if the output is continuous, False if it is discrete.
        """
        raise NotImplementedError("is_continuous method is not implemented for this processor.")
    
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
    
    def spatial(self, i: int) -> tuple[bool, ...]:
        """Whether each dimension (axis) of the i-th output tensor is spatial (i.e. corresponds to a spatial 
        axis like time, height, width, etc.) or not. This is used to determine how to apply 
        augmentations and other transformations that should only be applied to spatial dimensions.
        
        E.g. for CNN or RNN features, this would help determine which dimensions to apply spatial augmentations to, 
        and which dimensions to treat as channels or features.
        
        Args:
            i: Index of the output tensor to check.
        
        Returns:
            Tuple of booleans corresponding to whether each axis of the i-th output tensor is spatial or not.
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
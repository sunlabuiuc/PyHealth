from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


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

    def fit(self, samples: List[Dict[str, Any]], field: str) -> None:
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

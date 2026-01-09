from typing import Any, Dict, Iterable
from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("ignore")
class IgnoreProcessor(FeatureProcessor):
    """A special feature processor that marks a feature to be ignored during processing.
    """

    def __init__(self) -> None:
        pass

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
        """This method is intentionally not implemented.

        Args:
            samples: An iterable of raw data samples.
            field: The field name to be ignored.
        
        Raises:
            NotImplementedError: Always raised to indicate this processor ignores the field.
        """
        raise NotImplementedError("IgnoreProcessor does not implement fit method.")

    def process(self, value: Any) -> Any:
        """This method is intentionally not implemented.

        Args:
            value: Any raw field value.

        Raises:
            NotImplementedError: Always raised to indicate this processor ignores the field.
        """
        raise NotImplementedError("IgnoreProcessor does not implement process method.")

    def __repr__(self) -> str:
        return (f"IgnoreProcessor()")

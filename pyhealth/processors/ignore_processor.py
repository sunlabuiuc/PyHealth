from typing import Any, Dict, Iterable
from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("ignore")
class IgnoreProcessor(FeatureProcessor):
    """A special feature processor that marks a feature to be ignored during processing.

    This processor is useful when you want to remove a specific feature from the dataset
    after the task function processing, but without modifying the task function itself.

    Example:
        >>> from pyhealth.processors import IgnoreProcessor
        >>> # Assume we have a task that outputs "feature1" and "feature2"
        >>> # We want to remove "feature2" from the final dataset
        >>> dataset.set_task(task, input_processors={
        ...     "feature1": SequenceProcessor(code_to_index),
        ...     "feature2": IgnoreProcessor()
        ... })
        >>> # Now samples in dataset will only contain "feature1"
    """

    def __init__(self) -> None:
        pass

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

from typing import Any

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("raw")
class RawProcessor(FeatureProcessor):
    """
    Processor that returns the raw value.
    """

    def process(self, value: Any) -> str:
        return value

    def size(self):
        return None

    def __repr__(self):
        return "RawProcessor()"

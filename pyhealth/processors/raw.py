from typing import Any

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("raw")
class RawFeatures(FeatureProcessor):
    """
    Processor that let's you define any custom return in the SampleDataset
    """

    def process(self, value: Any) -> str:
        return value
    
    def size(self):
        return None

    def __repr__(self):
        return "RawFeatures()"

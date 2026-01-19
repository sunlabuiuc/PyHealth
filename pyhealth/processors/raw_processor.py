import json
from typing import Any

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("raw")
class RawProcessor(FeatureProcessor):
    """
    Processor that JSON-serializes raw values for litdata compatibility.

    Converts floats and complex structures to JSON strings.
    Use json.loads() to deserialize when accessing the data.
    """

    def process(self, value: Any) -> str:
        # JSON serialize - handles strings, lists, tuples, floats, etc.
        return json.dumps(value)

    def size(self):
        return None

    def __repr__(self):
        return "RawProcessor()"

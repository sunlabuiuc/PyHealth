import warnings
from typing import Any

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("raw")
class RawProcessor(FeatureProcessor):
    """
    Processor that returns values unchanged (pass-through).

    This processor is intended for simple data types that don't need
    special encoding (strings, integers, etc.).

    Warning:
        Float values may have serialization issues with litdata. If you're
        working with float sequences, consider using 'nested_sequence_floats'
        or 'tensor' processor instead for better compatibility.
    """

    _float_warning_shown = False

    def process(self, value: Any) -> Any:
        # Warn if float is detected (may have litdata serialization issues)
        if not RawProcessor._float_warning_shown:
            if self._contains_float(value):
                warnings.warn(
                    "RawProcessor received float values. Float sequences may "
                    "have serialization issues with litdata. Consider using "
                    "'nested_sequence_floats' or 'tensor' processor instead.",
                    UserWarning,
                    stacklevel=2,
                )
                RawProcessor._float_warning_shown = True

        return value

    def _contains_float(self, value: Any) -> bool:
        """Check if value contains floats by sampling first elements."""
        if isinstance(value, float):
            return True
        if isinstance(value, (list, tuple)) and len(value) > 0:
            first = value[0]
            if isinstance(first, float):
                return True
            # Check nested list/tuple (e.g., [[1.0, 2.0], ...])
            if isinstance(first, (list, tuple)) and len(first) > 0:
                return isinstance(first[0], float)
        return False

    def size(self):
        return None

    def __repr__(self):
        return "RawProcessor()"

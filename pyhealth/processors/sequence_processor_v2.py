"""Former home of :class:`~pyhealth.processors.UnifiedFeatureProcessor`.

``SequenceProcessorV2`` is the processor's previous name. It and the types
below are re-exported from :mod:`pyhealth.processors.unified_processor` so
existing imports, and caches pickled under this module, keep working.
"""

from .unified_processor import (
    CodeMapping,
    FieldKey,
    Impute,
    ImputeMethod,
    LevelConfig,
    Normalize,
    NormalizeMethod,
    OrderType,
    Records,
    Resample,
    ResampleAlign,
    ResampleMethod,
    Role,
)
from .unified_processor import UnifiedFeatureProcessor as SequenceProcessorV2

__all__ = [
    "CodeMapping",
    "FieldKey",
    "Impute",
    "ImputeMethod",
    "LevelConfig",
    "Normalize",
    "NormalizeMethod",
    "OrderType",
    "Records",
    "Resample",
    "ResampleAlign",
    "ResampleMethod",
    "Role",
    "SequenceProcessorV2",
]

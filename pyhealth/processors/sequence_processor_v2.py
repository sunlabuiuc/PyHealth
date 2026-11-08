"""Former home of :class:`~pyhealth.processors.UnifiedFeatureProcessor`.

.. deprecated::
    :class:`SequenceProcessorV2` and the ``"sequence_v2"`` processor name are
    deprecated; use :class:`~pyhealth.processors.UnifiedFeatureProcessor` and
    ``"unified"``, which behave identically. The types below are re-exported
    from :mod:`pyhealth.processors.unified_processor` so existing imports, and
    caches pickled under this module, keep working.
"""

import warnings
from collections.abc import Sequence

from . import register_processor
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
    UnifiedFeatureProcessor,
)

_DEPRECATION = (
    "SequenceProcessorV2 and the 'sequence_v2' processor name are deprecated and "
    "will be removed in a future release; use UnifiedFeatureProcessor and "
    "'unified', which behave identically."
)


@register_processor("sequence_v2")
class SequenceProcessorV2(UnifiedFeatureProcessor):
    """Deprecated former name of :class:`~pyhealth.processors.UnifiedFeatureProcessor`.

    .. deprecated::
        Use :class:`~pyhealth.processors.UnifiedFeatureProcessor` (registered
        as ``"unified"``), which behaves identically. This subclass adds
        nothing but a ``FutureWarning`` when it is constructed, so that
        schemas naming ``"sequence_v2"`` and caches pickled with this class
        keep working. It must stay empty otherwise.

    Examples:
        >>> import warnings
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     proc = SequenceProcessorV2(levels=[{"tokens": 1}])
        >>> isinstance(proc, UnifiedFeatureProcessor), caught[0].category.__name__
        (True, 'FutureWarning')
    """

    def __init__(self, levels: Sequence[LevelConfig]):
        warnings.warn(_DEPRECATION, FutureWarning, stacklevel=2)
        super().__init__(levels)


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

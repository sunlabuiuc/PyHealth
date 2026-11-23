"""Former home of :class:`~pyhealth.processors.JaggedFeatureProcessor`.

.. deprecated::
    :class:`SequenceProcessorV2` and the ``"sequence_v2"`` processor name are
    deprecated; use :class:`~pyhealth.processors.JaggedFeatureProcessor` and
    ``"jagged"``, which behave identically. The types below are re-exported
    from :mod:`pyhealth.processors.jagged_processor` so existing imports, and
    caches pickled under this module, keep working.
"""

import warnings
from collections.abc import Sequence

from . import register_processor
from .jagged_processor import (
    CodeMapping,
    FieldKey,
    Impute,
    ImputeMethod,
    JaggedFeatureProcessor,
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

_DEPRECATION = (
    "SequenceProcessorV2 and the 'sequence_v2' processor name are deprecated and "
    "will be removed in a future release; use JaggedFeatureProcessor and "
    "'jagged', which behave identically."
)


@register_processor("sequence_v2")
class SequenceProcessorV2(JaggedFeatureProcessor):
    """Deprecated former name of :class:`~pyhealth.processors.JaggedFeatureProcessor`.

    .. deprecated::
        Use :class:`~pyhealth.processors.JaggedFeatureProcessor` (registered
        as ``"jagged"``), which behaves identically. This subclass adds
        nothing but a ``FutureWarning`` when it is constructed, so that
        schemas naming ``"sequence_v2"`` and caches pickled with this class
        keep working. It must stay empty otherwise.

    Examples:
        >>> import warnings
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     proc = SequenceProcessorV2(levels=[{"tokens": 1}])
        >>> isinstance(proc, JaggedFeatureProcessor), caught[0].category.__name__
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

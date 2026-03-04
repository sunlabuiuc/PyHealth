PROCESSOR_REGISTRY = {}


def register_processor(name: str):
    def decorator(cls):
        if name in PROCESSOR_REGISTRY:
            raise ValueError(f"Processor '{name}' already registered.")
        PROCESSOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_processor(name: str):
    if name not in PROCESSOR_REGISTRY:
        raise ValueError(f"Unknown processor: {name}")
    return PROCESSOR_REGISTRY[name]


# Import all processors so they register themselves
from .base_processor import FeatureProcessor
try:
    from .image_processor import ImageProcessor
    _has_image_processor = True
except (ImportError, RuntimeError):
    _has_image_processor = False  # PIL/torchvision unavailable or broken
from .label_processor import (
    BinaryLabelProcessor,
    MultiClassLabelProcessor,
    MultiLabelProcessor,
    RegressionLabelProcessor,
)
from .multi_hot_processor import MultiHotProcessor
from .nested_sequence_processor import (
    NestedFloatsProcessor,
    NestedSequenceProcessor,
)
from .deep_nested_sequence_processor import (
    DeepNestedFloatsProcessor,
    DeepNestedSequenceProcessor,
)
from .raw_processor import RawProcessor
from .sequence_processor import SequenceProcessor
from .signal_processor import SignalProcessor
from .stagenet_processor import (
    StageNetProcessor,
    StageNetTensorProcessor,
)
from .tensor_processor import TensorProcessor
from .text_processor import TextProcessor
from .timeseries_processor import TimeseriesProcessor
try:
    from .time_image_processor import TimeImageProcessor
    _has_time_image_processor = True
except (ImportError, RuntimeError):
    _has_time_image_processor = False  # PIL/torchvision unavailable or broken
from .audio_processor import AudioProcessor
from .ignore_processor import IgnoreProcessor
from .tuple_time_text_processor import TupleTimeTextProcessor

# Expose public API — optional processors only listed if successfully imported
__all__ = [
    "FeatureProcessor",
    "BinaryLabelProcessor",
    "MultiClassLabelProcessor",
    "MultiLabelProcessor",
    "RegressionLabelProcessor",
    "MultiHotProcessor",
    "NestedFloatsProcessor",
    "NestedSequenceProcessor",
    "RawProcessor",
    "SequenceProcessor",
    "SignalProcessor",
    "StageNetProcessor",
    "StageNetTensorProcessor",
    "TensorProcessor",
    "TextProcessor",
    "TimeseriesProcessor",
    "AudioProcessor",
    "TupleTimeTextProcessor",
]
if _has_image_processor:
    __all__.append("ImageProcessor")
if _has_time_image_processor:
    __all__.append("TimeImageProcessor")

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
from .image_processor import ImageProcessor
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
from .time_image_processor import TimeImageProcessor
from .graph_processor import GraphProcessor
from .audio_processor import AudioProcessor
from .ignore_processor import IgnoreProcessor
from .temporal_timeseries_processor import TemporalTimeseriesProcessor
from .tuple_time_text_processor import TupleTimeTextProcessor

# Expose public API
from .base_processor import (
    FeatureProcessor,
    ModalityType,
    TemporalFeatureProcessor,
)
__all__ = [
    "FeatureProcessor",
    "ModalityType",
    "TemporalFeatureProcessor",
    "ImageProcessor",
    "LabelProcessor",
    "MultiHotProcessor",
    "NestedFloatsProcessor",
    "NestedSequenceProcessor",
    "RawProcessor",
    "SequenceProcessor",
    "SignalProcessor",
    "StageNetProcessor",
    "StageNetTensorProcessor",
    "TensorProcessor",
    "TemporalTimeseriesProcessor",
    "TextProcessor",
    "TimeseriesProcessor",
    "TimeImageProcessor",
    "GraphProcessor",
    "AudioProcessor",
    "TupleTimeTextProcessor",
]

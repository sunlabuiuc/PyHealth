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
from .raw_processor import RawProcessor
from .sequence_processor import SequenceProcessor
from .signal_processor import SignalProcessor
from .tensor_processor import TensorProcessor
from .text_processor import TextProcessor
from .timeseries_processor import TimeseriesProcessor

# Expose public API
__all__ = [
    "get_processor",
    "ImageProcessor",
    "SequenceProcessor",
    "TensorProcessor",
    "TimeseriesProcessor",
    "SignalProcessor",
    "BinaryLabelProcessor",
    "MultiClassLabelProcessor",
    "MultiLabelProcessor",
    "RegressionLabelProcessor",
    "RawProcessor",
    "TextProcessor",
]

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
from .dicom_processor import DicomProcessor
from .dicom_series_processor import DicomSeriesProcessor
from .nifti_processor import NiftiProcessor
from .image_mask_processor import ImageMaskDirectoryProcessor
from .image_io.metadata import read_image_metadata
from .label_processor import (
    BinaryLabelProcessor,
    LabelProcessor,
    MultiClassLabelProcessor,
    MultiLabelProcessor,
    RegressionLabelProcessor,
)
from .segmentation_label_processors import (
    SegmentationBinaryLabelProcessor,
    SegmentationMultiClassLabelProcessor,
)
from .multi_hot_processor import MultiHotProcessor
from .nested_multihot_processor import NestedMultiHotProcessor
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
from .jagged_processor import (
    Impute,
    JaggedFeatureProcessor,
    LevelConfig,
    Normalize,
    Records,
    Resample,
)
from .sequence_processor_v2 import SequenceProcessorV2
from .signal_processor import SignalProcessor
from .stagenet_processor import (
    StageNetProcessor,
    StageNetTensorProcessor,
)
from .tensor_processor import TensorProcessor
from .text_processor import TextProcessor
from .timeseries_processor import TimeseriesProcessor
from .timeseries_label_processor import (
    TimeseriesBinaryLabelProcessor,
    TimeseriesLabelProcessor,
    TimeseriesMultiClassLabelProcessor,
    TimeseriesMultiLabelProcessor,
    TimeseriesRegressionLabelProcessor,
)
from .time_image_processor import TimeImageProcessor
from .time_region_image_processor import TimeRegionImageProcessor
from .graph_processor import GraphProcessor
from .audio_processor import AudioProcessor
from .ignore_processor import IgnoreProcessor
from .temporal_timeseries_processor import TemporalTimeseriesProcessor
from .tuple_time_text_processor import TupleTimeTextProcessor
from .cehr_processor import CehrProcessor, ConceptVocab
from .ethos_processor import EthosProcessor
from .ethos_context_processor import EthosContextProcessor
from .unified_bridge import (
    BridgeDataset,
    UnifiedBridge,
    transform_batch,
    transform_processor,
)

# Expose public API
from .base_processor import (
    FeatureProcessor,
    LegacyFeatureProcessor,
    ModalityType,
    TemporalFeatureProcessor,
)
__all__ = [
    "FeatureProcessor",
    "LegacyFeatureProcessor",
    "ModalityType",
    "TemporalFeatureProcessor",
    "ImageProcessor",
    "DicomProcessor",
    "DicomSeriesProcessor",
    "NiftiProcessor",
    "ImageMaskDirectoryProcessor",
    "read_image_metadata",
    "LabelProcessor",
    "MultiHotProcessor",
    "SegmentationBinaryLabelProcessor",
    "SegmentationMultiClassLabelProcessor",
    "NestedFloatsProcessor",
    "DeepNestedFloatsProcessor",
    "DeepNestedSequenceProcessor",
    "NestedMultiHotProcessor",
    "NestedSequenceProcessor",
    "RawProcessor",
    "SequenceProcessor",
    "SequenceProcessorV2",
    "JaggedFeatureProcessor",
    "UnifiedBridge",
    "BridgeDataset",
    "transform_batch",
    "transform_processor",
    "Records",
    "LevelConfig",
    "Impute",
    "Normalize",
    "Resample",
    "SignalProcessor",
    "StageNetProcessor",
    "StageNetTensorProcessor",
    "TensorProcessor",
    "TemporalTimeseriesProcessor",
    "TextProcessor",
    "TimeseriesBinaryLabelProcessor",
    "TimeseriesLabelProcessor",
    "TimeseriesMultiClassLabelProcessor",
    "TimeseriesMultiLabelProcessor",
    "TimeseriesProcessor",
    "TimeseriesRegressionLabelProcessor",
    "TimeImageProcessor",
    "TimeRegionImageProcessor",
    "GraphProcessor",
    "AudioProcessor",
    "IgnoreProcessor",
    "TupleTimeTextProcessor",
    "CehrProcessor",
    "ConceptVocab",
    "EthosProcessor",
    "EthosContextProcessor",
]

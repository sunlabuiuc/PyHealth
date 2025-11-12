Processors
===============

Processors in PyHealth handle data preprocessing and transformation for healthcare predictive tasks. They convert raw data into tensors suitable for machine learning models.

Overview
--------

Processors are automatically applied based on the ``input_schema`` and ``output_schema`` defined in tasks. PyHealth supports three ways to customize processors:

1. **String Dictionary Key Notation** (Recommended): Use string keys in your task schema (see Example 1)
2. **Processor Class Notation**: Pass processor classes directly in the schema (see Example 2)
3. **Kwargs Tuple Notation**: Use tuples with processor keys and kwargs for custom configurations (see Example 5)

Available Processors
--------------------

**Sequence Processors:**

- ``SequenceProcessor``: For categorical sequences (e.g., medical codes like diagnoses, procedures)
- ``NestedSequenceProcessor``: For nested categorical sequences (e.g., drug recommendation with visit history)
- ``NestedFloatsProcessor``: For nested numerical sequences with optional forward-fill

**Label Processors:**

- ``BinaryLabelProcessor``: Binary classification labels (e.g., mortality prediction)
- ``MultiClassLabelProcessor``: Multi-class classification labels
- ``MultiLabelProcessor``: Multi-label classification (e.g., drug recommendation)
- ``RegressionLabelProcessor``: Continuous regression targets

**Data Type Processors:**

- ``ImageProcessor``: For image data (e.g., chest X-rays)
- ``TextProcessor``: For text/clinical notes data
- ``AudioProcessor``: For audio signal data
- ``SignalProcessor``: For general signal data (e.g., EEG, ECG)
- ``TimeseriesProcessor``: For time-series data
- ``TensorProcessor``: For pre-processed tensor data
- ``RawProcessor``: Pass-through processor for raw data

**Specialized Processors:**

- ``StageNetProcessor``: For StageNet model with lab measurements
- ``StageNetTensorProcessor``: Tensor processing for StageNet
- ``MultiHotProcessor``: For multi-hot encoding

Usage Examples
--------------

**Example 1: Using String Dictionary Keys in Task Schema**

This is the recommended approach. Define your schema using string keys, and PyHealth will automatically apply the appropriate processors:

.. code-block:: python

    from pyhealth.tasks import BaseTask
    from typing import Dict, List, Any

    class MortalityPredictionMIMIC3(BaseTask):
        """Mortality prediction task with automatic processor selection."""
        
        task_name: str = "MortalityPredictionMIMIC3"
        
        # String keys automatically map to processors
        input_schema: Dict[str, str] = {
            "conditions": "sequence",        # -> SequenceProcessor
            "procedures": "sequence",        # -> SequenceProcessor
            "drugs": "sequence",             # -> SequenceProcessor
        }
        output_schema: Dict[str, str] = {
            "mortality": "binary"            # -> BinaryLabelProcessor
        }
        
        def __call__(self, patient: Any) -> List[Dict[str, Any]]:
            # Task implementation
            samples = []
            # ... process patient data ...
            return samples

**Example 2: Using Processor Classes Directly in Schema**

For direct control over processor instantiation, you can pass processor classes directly in the schema. This allows for explicit class references without relying on string mappings:

.. code-block:: python

    from pyhealth.tasks import BaseTask
    from pyhealth.processors import SequenceProcessor, BinaryLabelProcessor
    from typing import Dict, List, Any

    class MortalityPredictionDirect(BaseTask):
        """Mortality prediction task using direct processor class references."""
        
        task_name: str = "MortalityPredictionDirect"
        
        # Use processor classes directly instead of strings
        input_schema: Dict[str, Any] = {
            "conditions": SequenceProcessor,        # Direct class reference
            "procedures": SequenceProcessor,        # Direct class reference
            "drugs": SequenceProcessor,             # Direct class reference
        }
        output_schema: Dict[str, Any] = {
            "mortality": BinaryLabelProcessor        # Direct class reference
        }
        
        def __call__(self, patient: Any) -> List[Dict[str, Any]]:
            # Task implementation
            samples = []
            # ... process patient data ...
            return samples

**Example 3: Nested Sequences for Drug Recommendation**

For tasks requiring cumulative history (like drug recommendation), use nested sequences:

.. code-block:: python

    class DrugRecommendationMIMIC3(BaseTask):
        """Drug recommendation with visit history."""
        
        task_name: str = "DrugRecommendationMIMIC3"
        
        input_schema: Dict[str, str] = {
            "conditions": "nested_sequence",   # -> NestedSequenceProcessor
            "procedures": "nested_sequence",   # -> NestedSequenceProcessor
            "drugs_hist": "nested_sequence",   # -> NestedSequenceProcessor
        }
        output_schema: Dict[str, str] = {
            "drugs": "multilabel"              # -> MultiLabelProcessor
        }
        
        def __call__(self, patient: Any) -> List[Dict[str, Any]]:
            # Returns samples with nested lists like:
            # {
            #     "conditions": [["code1", "code2"], ["code3"], ...],
            #     "procedures": [["proc1"], ["proc2", "proc3"], ...],
            #     "drugs_hist": [[], ["drug1"], ...],  # Empty for current visit
            #     "drugs": ["drug1", "drug2", ...]      # Target drugs
            # }
            ...

**Example 4: Multimodal Data**

Combine different data types using appropriate processor keys:

.. code-block:: python

    class MultimodalMortalityPredictionMIMIC4(BaseTask):
        """Multimodal mortality prediction with clinical notes and images."""
        
        task_name: str = "MultimodalMortalityPredictionMIMIC4"
        
        input_schema: Dict[str, str] = {
            "conditions": "sequence",           # -> SequenceProcessor
            "procedures": "sequence",           # -> SequenceProcessor
            "drugs": "sequence",                # -> SequenceProcessor
            "discharge": "text",                # -> TextProcessor
            "radiology": "text",                # -> TextProcessor
            "xrays_negbio": "sequence",         # -> SequenceProcessor
            "image_paths": "text",              # -> TextProcessor
        }
        output_schema: Dict[str, str] = {
            "mortality": "binary"               # -> BinaryLabelProcessor
        }

**Example 5: Custom Processor Configuration with Kwargs Tuples**

For advanced customization with parameters, use the kwargs tuple format ``(processor_key, kwargs_dict)``:

.. code-block:: python

    from pyhealth.processors import TimeseriesProcessor, ImageProcessor
    from pyhealth.tasks import BaseTask
    from datetime import timedelta
    from typing import Dict, List, Any, Tuple, Union

    class CustomMultimodalTask(BaseTask):
        """Task with custom processor parameters using kwargs tuples."""
        
        task_name: str = "CustomMultimodalTask"
        
        # Use kwargs tuples for processors with custom parameters
        input_schema: Dict[str, Union[str, Tuple[str, Dict]]] = {
            "conditions": "sequence",                              # Simple string key
            "vitals": (                                            # Kwargs tuple
                "timeseries",
                {
                    "sampling_rate": timedelta(minutes=30),       # Custom sampling rate
                    "impute_strategy": "forward_fill",             # Custom imputation
                },
            ),
            "chest_xray": (                                        # Kwargs tuple for images
                "image",
                {
                    "image_size": 256,                             # Custom image size
                    "normalize": True,                             # Enable normalization
                    "mean": [0.485, 0.456, 0.406],                # ImageNet means
                    "std": [0.229, 0.224, 0.225],                 # ImageNet stds
                    "mode": "RGB",                                 # Convert to RGB
                },
            ),
        }
        output_schema: Dict[str, str] = {
            "outcome": "binary"
        }
        
        def __call__(self, patient: Any) -> List[Dict[str, Any]]:
            # Task implementation that returns samples with:
            # - conditions: list of diagnosis codes
            # - vitals: tuple of (timestamps, values_array) for time series
            # - chest_xray: path to chest X-ray image file
            # - outcome: binary label
            samples = []
            # ... process patient data ...
            return samples


Processor String Keys
---------------------

Common string keys for automatic processor selection:

- ``"sequence"``: For categorical sequences (medical codes)
- ``"nested_sequence"``: For nested categorical sequences (visit history)
- ``"nested_sequence_floats"``: For nested numerical sequences
- ``"binary"``: For binary labels
- ``"multiclass"``: For multi-class labels  
- ``"multilabel"``: For multi-label classification
- ``"regression"``: For regression targets
- ``"text"``: For text data
- ``"image"``: For image data
- ``"audio"``: For audio data
- ``"signal"``: For signal data
- ``"timeseries"``: For time-series data
- ``"tensor"``: For pre-processed tensors
- ``"raw"``: For raw/unprocessed data

API Reference
-------------

.. toctree::
    :maxdepth: 3

    processors/pyhealth.processors.Processor
    processors/pyhealth.processors.FeatureProcessor
    processors/pyhealth.processors.SampleProcessor
    processors/pyhealth.processors.DatasetProcessor
    processors/pyhealth.processors.SequenceProcessor
    processors/pyhealth.processors.NestedSequenceProcessor
    processors/pyhealth.processors.NestedFloatsProcessor
    processors/pyhealth.processors.BinaryLabelProcessor
    processors/pyhealth.processors.MultiClassLabelProcessor
    processors/pyhealth.processors.MultiLabelProcessor
    processors/pyhealth.processors.RegressionLabelProcessor
    processors/pyhealth.processors.ImageProcessor
    processors/pyhealth.processors.TextProcessor
    processors/pyhealth.processors.AudioProcessor
    processors/pyhealth.processors.SignalProcessor
    processors/pyhealth.processors.TimeseriesProcessor
    processors/pyhealth.processors.TensorProcessor
    processors/pyhealth.processors.RawProcessor
    processors/pyhealth.processors.MultiHotProcessor
    processors/pyhealth.processors.StageNetProcessor
    processors/pyhealth.processors.StageNetTensorProcessor
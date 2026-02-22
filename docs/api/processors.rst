Processors
===============

Processors in PyHealth handle data preprocessing and transformation for healthcare predictive tasks. They convert raw data into tensors suitable for machine learning models.

Overview
--------

Processors are automatically applied based on the ``input_schema`` and ``output_schema`` defined in tasks. PyHealth supports four ways to customize processors:

1. **String Dictionary Key Notation** (Recommended): Use string keys in your task schema (see Example 1)
2. **Processor Class Notation**: Pass processor classes directly in the schema (see Example 2)
3. **Processor Instance Notation**: Pass pre-configured processor instances directly in the schema (see Example 6)
4. **Kwargs Tuple Notation**: Use tuples with processor keys and kwargs for custom configurations (see Example 5)

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
- ``TupleTimeTextProcessor``: For text paired with temporal information (time-aware text)
- ``AudioProcessor``: For audio signal data
- ``SignalProcessor``: For general signal data (e.g., EEG, ECG)
- ``TimeseriesProcessor``: For time-series data
- ``TimeImageProcessor``: For time-stamped image sequences (e.g., serial X-rays)
- ``TensorProcessor``: For pre-processed tensor data
- ``RawProcessor``: Pass-through processor for raw data

**Specialized Processors:**

- ``StageNetProcessor``: For StageNet model with lab measurements
- ``StageNetTensorProcessor``: Tensor processing for StageNet
- ``MultiHotProcessor``: For multi-hot encoding
- ``IgnoreProcessor``: A special feature processor that marks a feature to be ignored.
- ``GraphProcessor``: For knowledge graph subgraph extraction (e.g., GraphCare, G-BERT)

**Temporal Multimodal Processors (** :class:`~pyhealth.processors.TemporalFeatureProcessor` **subclasses):**

- ``TemporalTimeseriesProcessor``: Drop-in replacement for ``TimeseriesProcessor`` that preserves timestamps in ``{"value", "time"}`` dict output — enables temporal alignment in ``UnifiedMultimodalEmbeddingModel``
- ``StageNetProcessor``: Also a ``TemporalFeatureProcessor`` — adds ``modality()`` / ``value_dim()`` / ``process_temporal()``
- ``StageNetTensorProcessor``: Also a ``TemporalFeatureProcessor`` — numeric vitals with dict output
- ``TupleTimeTextProcessor``: Also a ``TemporalFeatureProcessor`` — tokenised clinical text with time
- ``TimeImageProcessor``: Also a ``TemporalFeatureProcessor`` — serial image sequences with timestamps

**Supporting Types:**

- ``ModalityType``: Enum of modality identifiers (``CODE``, ``TEXT``, ``IMAGE``, ``NUMERIC``, ``AUDIO``, ``SIGNAL``) used for routing in ``UnifiedMultimodalEmbeddingModel``
- ``TemporalFeatureProcessor``: Abstract base class for all temporal processors; requires ``modality()``, ``value_dim()``, and ``process()`` returning ``dict``

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

**Example 6: Using Processor Instances Directly in Schema**

For maximum control, you can pass pre-configured processor instances directly in the schema. This allows reusing fitted processors or applying specific configurations:

.. code-block:: python

    from pyhealth.tasks import BaseTask
    from pyhealth.processors import TimeseriesProcessor, BinaryLabelProcessor
    from datetime import timedelta
    from typing import Dict, List, Any

    class CustomTimeseriesTask(BaseTask):
        """Task using pre-configured processor instances."""
        
        task_name: str = "CustomTimeseriesTask"
        
        # Create processor instances with specific parameters
        timeseries_processor = TimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            impute_strategy="forward_fill"
        )
        
        # Use instances directly in schema
        input_schema: Dict[str, Any] = {
            "vitals": timeseries_processor,        # Pre-configured instance
            "conditions": "sequence",             # String key for comparison
        }
        output_schema: Dict[str, Any] = {
            "outcome": BinaryLabelProcessor()      # Instance without custom params
        }
        
        def __call__(self, patient: Any) -> List[Dict[str, Any]]:
            # Task implementation
            samples = []
            # ... process patient data ...
            return samples


Processor String Keys
---------------------

Common string keys for automatic processor selection:

- ``"temporal_timeseries"``: For time-series data with preserved timestamps (use in place of ``"timeseries"`` when building ``UnifiedMultimodalEmbeddingModel``)
- ``"sequence"``: For categorical sequences (medical codes)
- ``"nested_sequence"``: For nested categorical sequences (visit history)
- ``"nested_sequence_floats"``: For nested numerical sequences
- ``"binary"``: For binary labels
- ``"multiclass"``: For multi-class labels  
- ``"multilabel"``: For multi-label classification
- ``"regression"``: For regression targets
- ``"text"``: For text data
- ``"tuple_time_text"``: For text paired with temporal information
- ``"image"``: For image data
- ``"audio"``: For audio data
- ``"signal"``: For signal data
- ``"timeseries"``: For time-series data
- ``"time_image"``: For time-stamped image sequences
- ``"tensor"``: For pre-processed tensors
- ``"raw"``: For raw/unprocessed data
- ``"graph"``: For knowledge graph subgraphs

Writing Custom FeatureProcessors
---------------------------------

PyHealth's processor framework is highly flexible and allows you to create custom processors for your specific data transformation needs. This section explains how to write your own ``FeatureProcessor``.

Core Concepts
~~~~~~~~~~~~~

When creating a custom ``FeatureProcessor``, you need to understand two key methods:

1. **process()** - Required method that transforms individual feature values
2. **fit()** - Optional method for learning global data statistics (e.g., vocabularies, normalization parameters)

The ``process()`` Method
~~~~~~~~~~~~~~~~~~~~~~~~

The ``process()`` method is called **once per sample during the caching phase** (in ``BaseDataset.set_task()``). It transforms a single raw feature value into the format that will be stored in the cache. This method can return:

- Raw strings or primitives
- PyTorch tensors
- NumPy arrays
- Any other data structure your model expects

**Important:** ``process()`` should be **stateless**. Any mutations made to the processor during ``process()`` are not saved—the processor state is fixed after ``fit()`` completes. This is by design to ensure reproducibility and consistency across distributed workers.

The ``fit()`` Method
~~~~~~~~~~~~~~~~~~~~

The ``fit()`` method is called **once during the caching phase** (in ``BaseDataset.set_task()``), before any ``process()`` calls. Use ``fit()`` when you need to:

- Build vocabularies from the entire dataset
- Calculate normalization statistics (mean, std, min, max)
- Learn any global parameters from the data distribution
- Store variables that need to be shared across all samples

**Important:** Variables set in ``fit()`` are saved with the processor and reused when loading cached datasets.

Example: SequenceProcessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's how the built-in ``SequenceProcessor`` implements both methods:

.. code-block:: python

    from typing import Any, Dict, List, Iterable
    import torch
    from pyhealth.processors import register_processor
    from pyhealth.processors.base_processor import FeatureProcessor

    @register_processor("sequence")
    class SequenceProcessor(FeatureProcessor):
        """
        Processor for encoding categorical sequences (e.g., medical codes)
        into numerical indices.
        """

        def __init__(self):
            # Initialize vocabulary with padding token
            self.code_vocab: Dict[Any, int] = {"<pad>": 0}
            self._next_index = 1

        def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
            """Build vocabulary from all samples (called once during caching)."""
            for sample in samples:
                for token in sample[field]:
                    if token is None:
                        continue  # Skip missing values
                    elif token not in self.code_vocab:
                        # Add new token to vocabulary
                        self.code_vocab[token] = self._next_index
                        self._next_index += 1
            
            # Add unknown token at the end
            self.code_vocab["<unk>"] = len(self.code_vocab)

        def process(self, value: Any) -> torch.Tensor:
            """Convert tokens to indices (called during data loading)."""
            indices = []
            for token in value:
                if token in self.code_vocab:
                    indices.append(self.code_vocab[token])
                else:
                    indices.append(self.code_vocab["<unk>"])
            
            return torch.tensor(indices, dtype=torch.long)

        def size(self):
            """Return vocabulary size."""
            return len(self.code_vocab)

Key Design Decisions
~~~~~~~~~~~~~~~~~~~~

**When to Use fit():**

- Learning vocabularies (like ``SequenceProcessor``)
- Computing normalization statistics (like ``TimeseriesProcessor``)
- Determining feature dimensions (e.g., ``n_features`` in time series)
- Any stateful transformation that depends on the entire dataset

**When fit() is Optional:**

- Stateless transformations (e.g., converting strings to lowercase)
- Fixed transformations (e.g., resizing images to a fixed size)
- Pass-through operations (e.g., ``RawProcessor``)

Example: Simple Custom Processor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's a minimal example of a custom processor that normalizes numerical values:

.. code-block:: python

    from pyhealth.processors import register_processor
    from pyhealth.processors.base_processor import FeatureProcessor
    import torch
    import numpy as np

    @register_processor("z_score")
    class ZScoreProcessor(FeatureProcessor):
        """Normalize numerical features using z-score normalization."""

        def __init__(self):
            self.mean = 0.0
            self.std = 1.0

        def fit(self, samples, field):
            """Calculate mean and std from all samples."""
            values = []
            for sample in samples:
                if field in sample and sample[field] is not None:
                    values.extend(sample[field])
            
            self.mean = np.mean(values)
            self.std = np.std(values)

        def process(self, value):
            """Apply z-score normalization."""
            normalized = [(x - self.mean) / self.std for x in value]
            return torch.tensor(normalized, dtype=torch.float32)

Registering Your Processor
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``@register_processor`` decorator to make your processor available via string keys:

.. code-block:: python

    @register_processor("my_custom_processor")
    class MyCustomProcessor(FeatureProcessor):
        # ... implementation ...
        pass

Then use it in your task schema:

.. code-block:: python

    input_schema = {
        "my_field": "my_custom_processor",  # String key
        # Or with kwargs:
        "my_field": ("my_custom_processor", {"param1": value1}),
    }


API Reference
-------------

.. toctree::
    :maxdepth: 3

    processors/pyhealth.processors.Processor
    processors/pyhealth.processors.FeatureProcessor
    processors/pyhealth.processors.TemporalFeatureProcessor
    processors/pyhealth.processors.ModalityType
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
    processors/pyhealth.processors.TupleTimeTextProcessor
    processors/pyhealth.processors.AudioProcessor
    processors/pyhealth.processors.SignalProcessor
    processors/pyhealth.processors.TimeseriesProcessor
    processors/pyhealth.processors.TemporalTimeseriesProcessor
    processors/pyhealth.processors.TimeImageProcessor
    processors/pyhealth.processors.TensorProcessor
    processors/pyhealth.processors.RawProcessor
    processors/pyhealth.processors.IgnoreProcessor
    processors/pyhealth.processors.MultiHotProcessor
    processors/pyhealth.processors.StageNetProcessor
    processors/pyhealth.processors.StageNetTensorProcessor
    processors/pyhealth.processors.GraphProcessor
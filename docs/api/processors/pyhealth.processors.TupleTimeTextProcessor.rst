pyhealth.processors.TupleTimeTextProcessor
============================================

Processor for tuple time-based text data with temporal information.

.. autoclass:: pyhealth.processors.TupleTimeTextProcessor
    :members:
    :undoc-members:
    :show-inheritance:

**Overview**

``TupleTimeTextProcessor`` handles clinical text paired with temporal information (time differences), enabling automatic modality routing in multimodal fusion pipelines.

**Input/Output**

- **Input:** ``Tuple[List[str], List[float]]`` (texts, time differences)
- **Output:** ``Tuple[List[str], torch.Tensor, str]`` (texts, 1D time tensor, modality tag)

**Use Case**

The ``type_tag`` parameter enables automatic modality routing without hardcoding feature names in multimodal pipelines:

- ``type_tag="note"`` routes to text encoder
- ``type_tag="image"`` routes to vision encoder  
- ``type_tag="ehr"`` routes to EHR encoder

**Example Usage**

.. code-block:: python

    from pyhealth.processors import TupleTimeTextProcessor

    # Initialize processor with modality tag
    processor = TupleTimeTextProcessor(type_tag="clinical_note")

    # Patient notes with time differences (hours since admission)
    texts = [
        "Patient admitted with chest pain.",
        "Follow-up: symptoms improved.",
        "Discharge: stable condition."
    ]
    time_diffs = [0.0, 24.0, 72.0]

    # Process tuple
    processed_texts, time_tensor, modality_tag = processor.process((texts, time_diffs))
    
    print(time_tensor)      # tensor([0., 24., 72.])
    print(modality_tag)     # "clinical_note"

**Multimodal Fusion**

Use different type tags for automatic routing in multimodal models:

.. code-block:: python

    # Different modalities with different type tags
    note_processor = TupleTimeTextProcessor(type_tag="note")
    ehr_processor = TupleTimeTextProcessor(type_tag="ehr")

    # Process different data types
    note_texts, note_times, note_tag = note_processor.process((notes, note_time_diffs))
    ehr_texts, ehr_times, ehr_tag = ehr_processor.process((events, event_time_diffs))

    # Tags enable automatic routing to appropriate encoders
    # note_tag="note" -> TextEmbedding encoder
    # ehr_tag="ehr" -> EHR encoder

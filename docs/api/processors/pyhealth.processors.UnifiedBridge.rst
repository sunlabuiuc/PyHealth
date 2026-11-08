pyhealth.processors.UnifiedBridge
=================================

Bridges describe a fitted legacy processor in the unified packed-record
format of :class:`~pyhealth.processors.JaggedFeatureProcessor`, so models
built for that format can read legacy fields. The legacy processor still
processes and collates the data; the bridge converts each collated batch.

.. code-block:: python

    from pyhealth.processors import transform_batch, transform_processor

    processors = transform_processor(dataset.input_processors)
    for batch in loader:
        batch = transform_batch(processors, batch)

Every legacy processor has a bridge except
:class:`~pyhealth.processors.GraphProcessor` and
:class:`~pyhealth.processors.TupleTimeTextProcessor`, for which
:func:`~pyhealth.processors.transform_processor` raises ``TypeError``.

.. autoclass:: pyhealth.processors.UnifiedBridge
    :members: transform_batch
    :show-inheritance:

.. autofunction:: pyhealth.processors.transform_processor

.. autofunction:: pyhealth.processors.transform_batch

Bridges
-------

.. automodule:: pyhealth.processors.unified_bridge
    :members: SequenceToUnifiedBridge, CehrToUnifiedBridge,
        NestedSequenceToUnifiedBridge, DeepNestedSequenceToUnifiedBridge,
        MultiHotToUnifiedBridge, NestedMultiHotToUnifiedBridge,
        StageNetToUnifiedBridge, NestedFloatsToUnifiedBridge,
        DeepNestedFloatsToUnifiedBridge, StageNetTensorToUnifiedBridge,
        TimeseriesToUnifiedBridge, TemporalTimeseriesToUnifiedBridge,
        TimeImageToUnifiedBridge, TensorToUnifiedBridge, ImageToUnifiedBridge,
        AudioToUnifiedBridge
    :no-index:

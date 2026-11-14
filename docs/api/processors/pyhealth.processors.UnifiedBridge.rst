pyhealth.processors.UnifiedBridge
=================================

Bridges describe a fitted legacy processor in the unified packed-record
format of :class:`~pyhealth.processors.JaggedFeatureProcessor`, so models
built for that format can read legacy fields. The legacy processor still
processes and collates the data; the bridge converts each collated batch.

:class:`~pyhealth.models.RNN_V2` and
:class:`~pyhealth.models.SequenceEmbeddingModel` read legacy fields this way
on their own: each wraps the dataset it is given in a
:class:`~pyhealth.processors.BridgeDataset` and converts every batch, so data
loaders are built from the dataset as usual.

.. code-block:: python

    model = RNN_V2(dataset, embedding_dim={"codes": [{"tokens": (64,)}]})
    loader = get_dataloader(dataset, batch_size=32)

A bridge made from a legacy processor converts; one made from another bridge
(for a model nested in one that already converts, such as the embedding inside
RNN_V2) passes batches through, so each batch is converted once. Other code can
do the same by hand:

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
    :members: transform_batch, converts
    :show-inheritance:

.. autoclass:: pyhealth.processors.BridgeDataset
    :members: dataset

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

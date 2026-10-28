pyhealth.processors.SequenceProcessorV2
=======================================

Processor for hierarchical packed records: codes, values and timestamps at any
depth of nesting, stored without padding or truncation.

The task returns each field as a ``Records``, a plain dict of columns.
``children`` holds the records one level down, and the innermost records
(level 0) have no ``children``. The processor emits a tuple of tensors in
``schema()`` order: by level, then by role (``tokens``, ``values``, ``time``,
``offsets``). ``offsets`` at level ``l`` partitions the level-``l`` records into
level-``l + 1`` records, in CSR layout.

Each level's configuration declares its roles: ``tokens`` gives the number of
token namespaces ``K`` and ``values`` gives the value shape ``S``. ``fit()``
only builds the vocabularies, and every record is checked against the
configuration.

.. code-block:: python

    proc = SequenceProcessorV2(levels=[{"ordering": "set", "tokens": 1}, {"ordering": "sequence"}])
    samples = [{"visits": {"children": [{"tokens": ["A", "B"]}, {"tokens": ["C"]}]}}]
    proc.fit(samples, "visits")
    proc.schema()  # ((0, 'tokens'), (0, 'offsets'))
    proc.process({"children": [{"tokens": ["A", "B"]}, {"tokens": []}, {"tokens": ["C"]}]})
    # (tensor([2, 3, 4]), tensor([0, 2, 2, 3]))

In a task schema the configuration is plain data:

.. code-block:: python

    input_schema = {
        "visits": ("sequence_v2", {"levels": [{"ordering": "set", "tokens": 1}, {"ordering": "sequence"}]}),
        "vitals": ("sequence_v2", {"levels": [{"ordering": "series", "values": [2], "resample": {"step": 1.0, "method": "last"}}]}),
    }

Batches are packed with ``collate()``, which follows ``batch_schema()``: the
fields of ``schema()`` plus the batch offsets ``[B + 1]``. After resampling,
``impute`` fills NaN values (see ``Impute``) and ``normalize`` scales them (see
``Normalize``); ``missing_indicator`` is reserved for future work and only
accepts ``False``. Values still containing NaN or inf raise.

.. autoclass:: pyhealth.processors.SequenceProcessorV2
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.processors.Records
    :members:

.. autoclass:: pyhealth.processors.LevelConfig
    :members:

.. autoclass:: pyhealth.processors.Resample
    :members:

.. autoclass:: pyhealth.processors.Impute
    :members:

.. autoclass:: pyhealth.processors.Normalize
    :members:

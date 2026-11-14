pyhealth.processors.SequenceProcessor
===================================

Processor for sequence data.

Editing the vocabulary
----------------------

After ``fit``, the vocabulary can be edited with ``remove`` (drop the given
codes), ``retain`` (keep only the given codes) and ``add`` (append new codes).
``<pad>`` and ``<unk>`` are always kept. ``remove`` and ``retain`` renumber the
remaining codes so the indices stay contiguous, and ``add`` appends after the
last index. Calling ``fit`` again afterwards gives new codes the next free
index, so every index stays below ``vocab_size()``. ``NestedSequenceProcessor``,
``DeepNestedSequenceProcessor``, ``StageNetProcessor`` and
``NestedMultiHotProcessor`` have the same methods.

.. code-block:: python

    proc = SequenceProcessor()
    proc.fit([{"codes": ["A", "B", "C"]}], "codes")
    proc.retain({"A", "C"})  # B now maps to <unk>
    proc.fit([{"codes": ["D"]}], "codes")
    proc.code_vocab  # {'<pad>': 0, '<unk>': 1, 'A': 2, 'C': 3, 'D': 4}

.. autoclass:: pyhealth.processors.SequenceProcessor
    :members:
    :undoc-members:
    :show-inheritance:

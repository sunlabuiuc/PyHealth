pyhealth.processors.NestedSequenceProcessor
===================================

Processor for nested categorical sequence data with vocabulary.

Handles nested sequences like drug recommendation history where each sample
contains a list of visits, and each visit contains a list of codes.
For example: [["code1", "code2"], ["code3"], ["code4", "code5", "code6"]]

Output width
------------

Each visit becomes a row of the same width: the longest visit seen in ``fit()``
plus ``padding``. Shorter visits are padded with ``<pad>``. Longer visits, which
can occur when processors are fitted on the training split and applied to
validation or test data, are truncated to that width, keeping the first codes,
with a one-time warning. Set ``padding`` to leave room for longer visits.
``NestedFloatsProcessor``, ``DeepNestedSequenceProcessor`` and
``DeepNestedFloatsProcessor`` truncate codes or values per visit the same way.

In PyHealth 2.0.2 and earlier, longer visits were not truncated. Samples then had
different widths and batching them failed with a tensor shape mismatch (the deep
processors failed already while processing the sample).

See ``examples/nested_sequence_fit_on_train.py``.

.. autoclass:: pyhealth.processors.NestedSequenceProcessor
    :members:
    :undoc-members:
    :show-inheritance:

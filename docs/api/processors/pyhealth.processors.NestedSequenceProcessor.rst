pyhealth.processors.NestedSequenceProcessor
===================================

Processor for nested categorical sequence data with vocabulary.

Handles nested sequences like drug recommendation history where each sample
contains a list of visits, and each visit contains a list of codes.
For example: [["code1", "code2"], ["code3"], ["code4", "code5", "code6"]]

.. autoclass:: pyhealth.processors.NestedSequenceProcessor
    :members:
    :undoc-members:
    :show-inheritance:

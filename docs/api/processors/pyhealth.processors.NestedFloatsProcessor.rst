pyhealth.processors.NestedFloatsProcessor
===================================

Processor for nested numerical sequence data without vocabulary.

Handles nested sequences of floats/numerical values where each sample
contains a list of visits, and each visit contains a list of values.
For example: [[1.5, 2.3], [4.1], [0.9, 1.2, 3.4]]

Supports forward-fill for missing values across time steps.

.. autoclass:: pyhealth.processors.NestedFloatsProcessor
    :members:
    :undoc-members:
    :show-inheritance:

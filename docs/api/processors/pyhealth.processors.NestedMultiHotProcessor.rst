pyhealth.processors.NestedMultiHotProcessor
===========================================

Processor for nested categorical sequences, emitted as per-visit multi-hot
vectors.

Takes the same input as ``NestedSequenceProcessor`` -- a list of visits, each a
list of codes -- but emits one row per visit with one column per vocabulary
entry, set to 1 where the code is present. Repeats within a visit collapse, so
this records presence rather than order or count.

Prefer it over ``NestedSequenceProcessor`` for set-membership models such as
generative EHR models: the index form pads every visit to the longest visit
seen during ``fit``, so a single outlier visit sets the width for the whole
dataset. Sizing by the vocabulary instead is both smaller and cheaper to
consume.

.. autoclass:: pyhealth.processors.NestedMultiHotProcessor
    :members:
    :undoc-members:
    :show-inheritance:

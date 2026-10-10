pyhealth.processors.KGTripleProcessor
=======================================

Overview
--------
Processor for a knowledge-graph triple ``(head, relation, tail)`` of integer
ids, registered as ``"kg_triple"`` in the processor registry. It returns the
triple as a ``LongTensor`` and, from the triples it is fitted on, keeps plain
dicts used to train KG embedding models: the known heads of each
``(relation, tail)`` pair, the known tails of each ``(head, relation)`` pair,
and the pair frequencies behind subsampling weights.

Fit it on the training triples only, with
``set_task(task, split=PatientSplit(...))``: the training negatives and
weights then never depend on validation or test triples. ``size()`` returns
the number of entities, which is given (ids are global), not fitted.

API Reference
-------------
.. automodule:: pyhealth.processors.kg_triple_processor
    :members:
    :undoc-members:
    :show-inheritance:

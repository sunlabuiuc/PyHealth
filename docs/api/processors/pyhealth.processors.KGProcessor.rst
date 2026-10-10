pyhealth.processors.KGProcessor
=================================

Overview
--------
Processor that pads a variable-length list of knowledge-graph entity ids,
such as the ground-truth sets used for filtered link-prediction ranking,
and returns it with a mask. Registered as ``"kg_entity_list"`` in the
processor registry. A list longer than the fitted length is kept whole.

API Reference
-------------
.. automodule:: pyhealth.processors.kg_processor
    :members:
    :undoc-members:
    :show-inheritance:

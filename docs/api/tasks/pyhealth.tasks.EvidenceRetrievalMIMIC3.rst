pyhealth.tasks.EvidenceRetrievalMIMIC3
======================================

Overview
--------

Task for LLM-based evidence retrieval from EHR clinical notes, inspired
by Ahsan et al. 2024 (`PMLR <https://proceedings.mlr.press/v248/ahsan24a.html>`_).
Each sample carries a raw note, a diagnosis condition query, and a
weak ground-truth ``is_positive`` label used by standard PyHealth
binary metrics for the note-level classification sub-task.

API Reference
-------------

.. autoclass:: pyhealth.tasks.EvidenceRetrievalMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:

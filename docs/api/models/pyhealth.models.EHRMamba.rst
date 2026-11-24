EHRMamba
========

EHRMamba is a compact, CPU-compatible, selective state-space architecture for
supervised prediction from longitudinal EHR code sequences.

It provides code embeddings, optional age/time/segment metadata, stacked
causal Mamba-style blocks, and a task-specific PyHealth prediction head.
The implementation is a supervised architectural adaptation rather than a
reproduction of the paper's large-scale pretraining or forecasting tasks.

Reference
---------

Fallahpour et al. *EHRMAMBA: Towards Generalizable and Scalable Foundation
Models for Electronic Health Records*. PMLR 259 (2025).

https://proceedings.mlr.press/v259/fallahpour25a.html

API
---

.. autoclass:: pyhealth.models.ehrmamba.EHRMamba
   :members:
   :show-inheritance:

.. autoclass:: pyhealth.models.ehrmamba.CEHREmbeddings
   :members:

.. autoclass:: pyhealth.models.ehrmamba.SelectiveSSMBlock
   :members:

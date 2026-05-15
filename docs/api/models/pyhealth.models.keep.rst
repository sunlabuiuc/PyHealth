pyhealth.models.keep
====================

Overview
--------

This module implements **KEEP** (Knowledge-Preserving and Empirically
Refined Embedding Process), a method for learning robust medical code
embeddings by integrating structured medical ontologies with empirical
co-occurrence patterns from electronic health records (EHR).

KEEP addresses the trade-off between:

- Knowledge-graph-based embeddings (which preserve ontology structure)
- Data-driven embeddings (which capture empirical associations)

Our implementation provides:

- Lightweight co-occurrence-based embedding pretraining
- Optional frequency-aware ontology regularization
- Supervised readmission prediction via mean pooling
- Compatibility with the PyHealth Trainer API

This implementation is adapted for coursework-scale experiments
using MIMIC-IV.

Paper Reference
---------------

Ahmed Elhussein, Paul Meddeb, Abigail Newbury, Jeanne Mirone,
Martin Stoll, and Gamze Gursoy.

**"KEEP: Integrating Medical Ontologies with Clinical Data for Robust Code Embeddings."**

Proceedings of Machine Learning Research (PMLR), vol. 287,
pp. 1–19, 2025.

arXiv: https://arxiv.org/abs/2510.05049  
DOI: https://doi.org/10.48550/arXiv.2510.05049

API Reference
-------------

.. autoclass:: pyhealth.models.KEEP
   :members:
   :undoc-members:
   :show-inheritance:
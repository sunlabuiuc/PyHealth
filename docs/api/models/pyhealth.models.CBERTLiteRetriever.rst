pyhealth.models.CBERTLiteRetriever
==================================

Overview
--------

A lightweight neural IR baseline that reproduces the *CBERT* retrieval
style from Ahsan et al. 2024
(`PMLR <https://proceedings.mlr.press/v248/ahsan24a.html>`_). Given a
fixed risk-factor sentence per condition, the baseline encodes both
the query and each note sentence, then returns the top-K candidates
ranked by cosine similarity.

The default encoder is :class:`HashingEncoder`, which keeps the
baseline fully offline. Plug in a clinical or biomedical sentence
encoder (e.g. Bio_ClinicalBERT) by passing it as the ``encoder``
argument.

API Reference
-------------

.. autoclass:: pyhealth.models.CBERTLiteRetriever
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.HashingEncoder
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.RankedSentence
    :members:
    :undoc-members:
    :show-inheritance:

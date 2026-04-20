pyhealth.models.LLMEvidenceRetriever
====================================

Overview
--------

Zero-shot LLM-based evidence retriever for EHR clinical notes,
implementing the sequential two-pass prompting strategy from Ahsan et
al. 2024 (`PMLR <https://proceedings.mlr.press/v248/ahsan24a.html>`_)
together with a single-prompt ablation variant.

The retriever ships with a deterministic :class:`StubLLMBackend` so
unit tests run without any network calls. Swap in a hosted-API client
(or a local generation pipeline) by passing it as the ``backend``
argument.

API Reference
-------------

.. autoclass:: pyhealth.models.LLMEvidenceRetriever
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.LLMRetrieverConfig
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.EvidenceSnippet
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.StubLLMBackend
    :members:
    :undoc-members:
    :show-inheritance:

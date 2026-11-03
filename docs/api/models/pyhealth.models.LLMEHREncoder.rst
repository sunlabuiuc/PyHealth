pyhealth.models.LLMEHREncoder
=============================

A frozen LLM embedding model (e.g. Qwen3-Embedding) as an EHR encoder with a
linear head, from Hegselmann et al., "Large language models are powerful
electronic health record encoders" (npj Digital Medicine, 2025). Pair it with
:class:`~pyhealth.tasks.EHRTextSerializationTask`, which turns any task's
samples into Markdown text.

.. autoclass:: pyhealth.models.LLMEHREncoder
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.models.llm_ehr_encoder.qwen3_instruction

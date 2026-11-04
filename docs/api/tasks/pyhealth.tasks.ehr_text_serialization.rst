pyhealth.tasks.ehr_text_serialization
=====================================

Serializes a task's coded samples into Markdown text for an LLM encoder
(:class:`~pyhealth.models.LLMEHREncoder`), following Hegselmann et al.,
"Large language models are powerful electronic health record encoders"
(npj Digital Medicine, 2025).

.. autoclass:: pyhealth.tasks.ehr_text_serialization.EHRTextSerializationTask
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyhealth.tasks.ehr_text_serialization.CodeDescriber
   :members:

.. autofunction:: pyhealth.tasks.ehr_text_serialization.serialize_ehr

.. autofunction:: pyhealth.tasks.ehr_text_serialization.mimic_demographics


pyhealth.tasks.QueryFocusedEHRTask
==================================

Query-focused longitudinal EHR sample construction using MIMIC-III,
inspired by McInerney et al. (2020), *Query-Focused EHR Summarization
to Aid Imaging Diagnosis*.

Constructs diagnosis-conditioned examples from historical clinical
notes and future ICD-9 diagnoses. Supports individual diagnosis
codes, grouped ICD-9 queries, configurable negative-query selection,
temporal filtering, and sentence extraction.

The ``negative_sampling`` parameter supports two strategies:

- ``"never_observed"`` (default): Selects negative queries only
  when the diagnosis group is absent from the patient's entire
  observed record.
- ``"target_absent"``: Selects negative queries when the diagnosis
  group is absent from the target admission.

Labels provide distant supervision for diagnosis prediction,
not ground-truth sentence relevance annotations.

``examples/mimic3_query_focused_ehr_attention.py`` demonstrates
this task with :class:`~pyhealth.models.QueryFocusedEHRModel`,
including training, evaluation, and history-length ablations.

.. autoclass:: pyhealth.tasks.QueryFocusedEHRTask
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.query_focused_ehr.split_clinical_sentences

.. autofunction:: pyhealth.tasks.query_focused_ehr.normalize_icd9

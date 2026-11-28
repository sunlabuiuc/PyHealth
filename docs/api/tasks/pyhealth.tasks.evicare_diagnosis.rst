pyhealth.tasks.evicare_diagnosis
================================

Next-visit CCS diagnosis prediction for :class:`~pyhealth.models.EviCare`
(Zhang et al., `KDD 2026 <https://arxiv.org/abs/2604.10455>`_) on MIMIC-III
and MIMIC-IV. ICD diagnosis codes are mapped to CCS categories, and each
admission after the first is predicted from all earlier admissions of the
patient (patients need at least two admissions with mapped diagnoses).

Two settings, as in the paper:

- ``setting="overall"``: the label is every CCS code of the target admission.
- ``setting="novel"``: the label keeps only the CCS codes absent from all
  earlier admissions; a target with no such code gives no sample.

The guide model reads ``conditions`` (CCS codes per history visit) and the
label ``ccs_targets``. The evidence fields ``history_icd``,
``history_icd_names``, ``history_ccs`` and ``days_ago`` are outside the schema
and pass through processing unchanged for the prompt.

For EviCare's co-occurrence graph, ``task.patient_code_sets(dataset,
patient_ids=None)`` returns, for every eligible patient, all CCS codes over
all of their admissions. It reads the patients rather than the samples, so it
ignores ``setting`` and ``sample_mode`` and includes patients that give no
sample; pass its training-patient entries to
:meth:`~pyhealth.models.EviCare.fit_cooccurrence`.

The ICD to CCS mapping is a :class:`~pyhealth.tasks.CCSMapping`. By default
it is built from PyHealth's single-level CCS tables
(:class:`~pyhealth.medcode.CrossMap` ``ICD9CM``/``ICD10CM -> CCSCM``, 285
categories), the same ICD-9 table as the official EviCare code. Pass
``ccs_mapping="path/to.csv"`` for any other grouping, such as CCSR built from
the `HCUP CCSR tables
<https://hcup-us.ahrq.gov/toolssoftware/ccsr/ccs_refined.jsp>`_; see
:meth:`~pyhealth.tasks.CCSMapping.from_csv` for the columns.

.. note::

   Differences from the paper and the official code:

   - Sampling: ``sample_mode="all_prefixes"`` (the default) gives one sample
     per admission after the first. This follows the TRANS preprocessing that
     EviCare's data pipeline builds on and the paper's visit-level P@k,
     averaged over all test visits. It differs from the default of
     :class:`~pyhealth.tasks.NextVisitDiagnosisPredictionMIMIC3`
     (``"last"``, one sample per patient), which is available here too.
   - The TRANS preprocessing also emits a sample for each patient's first
     admission, with an empty history. It is skipped here: it is not a
     next-visit prediction and leaves no evidence for the prompt.
   - TRANS keeps only admissions that also have procedures and
     prescriptions; here an admission needs only a mapped diagnosis.
   - Ontology: single-level CCS through CrossMap by default; CCSR through
     ``ccs_mapping``. The paper reports 285 CCS codes on MIMIC-III (matching
     single-level CCS) and 842 on MIMIC-IV, which matches neither mapping
     cleanly; its Appendix C and the official README mention CCSR.
   - MIMIC-IV keeps both ICD-9 and ICD-10 diagnoses, each mapped with its own
     table.
   - The paper's MIMIC-III/MIMIC-IV versions, 5-fold protocol and 5%
     training subsample are not part of the task; the example script
     implements the 7:1:2 patient split and ``--train-fraction``.

.. autoclass:: pyhealth.tasks.EviCareDiagnosisPredictionMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.EviCareDiagnosisPredictionMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:

Both tasks also provide (inherited from their shared base class):

.. automethod:: pyhealth.tasks.EviCareDiagnosisPredictionMIMIC3.patient_code_sets

.. automethod:: pyhealth.tasks.EviCareDiagnosisPredictionMIMIC3.patient_codes

.. autoclass:: pyhealth.tasks.CCSMapping
    :members:
    :undoc-members:

.. autofunction:: pyhealth.tasks.evicare_diagnosis.normalize_icd

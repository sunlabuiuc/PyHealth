pyhealth.tasks.next_visit_diagnosis_prediction
===============================================

Next-visit diagnosis prediction: from the ICD-9 diagnosis codes of a
patient's past admissions, predict the codes of the next admission
(multi-label). One sample per patient by default, with the last admission as
the label, as in the SHy paper (https://arxiv.org/abs/2502.10689). Options
cover history length (``max_visits``), 3-digit codes, one sample per
admission (``sample_mode="all_prefixes"``) and visit time gaps.

.. autoclass:: pyhealth.tasks.next_visit_diagnosis_prediction.NextVisitDiagnosisPredictionMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.next_visit_diagnosis_prediction.NextVisitDiagnosisPredictionMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.next_visit_diagnosis_prediction.truncate_icd9

pyhealth.tasks.mortality\_prediction\_with\_fairness
=====================================================

``MortalityPredictionWithFairnessMIMIC3`` — a drop-in replacement for
:class:`~pyhealth.tasks.MortalityPredictionMIMIC3` that additionally attaches
seven demographic cohort attributes (``sex``, ``age_group``, ``ethnicity_4``,
``ethnicity_W``, ``insurance_type``, ``surgical_status``, ``admission_type``)
to each sample. The filter logic and labels are identical to the standard
mortality task, so models trained on the two tasks are directly comparable.

These cohort attributes are **not** part of the model's ``input_schema`` —
they are pass-through fields consumed by
:func:`pyhealth.tasks.fairness_utils.audit_predictions` after inference, not
by the model during training.

Task schema
-----------

- ``input_schema``: ``{"conditions": "sequence", "procedures": "sequence",
  "drugs": "sequence"}``
- ``output_schema``: ``{"mortality": "binary"}``

.. automodule:: pyhealth.tasks.mortality_prediction_with_fairness
    :members:
    :undoc-members:
    :show-inheritance:

References
----------

Hoche, M., Mineeva, O., Burger, M., Blasimme, A., & Rätsch, G. (2024).
*FAMEWS: A Fairness Auditing Tool for Medical Early-Warning Systems.*
Proceedings of the 5th Conference on Health, Inference, and Learning,
PMLR 248:297-311. https://proceedings.mlr.press/v248/hoche24a.html

Upstream FAMEWS code: https://github.com/ratschlab/famews

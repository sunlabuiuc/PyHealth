pyhealth.metrics.fairness
===================================

``fairness_metrics_fn`` is also importable directly from the top-level
``pyhealth.metrics`` package (``from pyhealth.metrics import
fairness_metrics_fn``), not just from this submodule.

.. currentmodule:: pyhealth.metrics.fairness

.. autofunction:: fairness_metrics_fn

.. currentmodule:: pyhealth.metrics.fairness_utils

Both ``disparate_impact`` and ``statistical_parity_difference`` raise
``ValueError`` if either the protected or unprotected group has zero
instances -- the favorable-outcome rate is undefined for an empty group,
so this is always an error rather than a value (e.g. 0 or NaN) that could
silently poison a downstream average across folds/seeds.

.. autofunction:: disparate_impact

.. autofunction:: statistical_parity_difference

.. autofunction:: sensitive_attributes_from_patient_ids
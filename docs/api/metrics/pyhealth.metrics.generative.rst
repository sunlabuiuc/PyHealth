pyhealth.metrics.generative
===================================

Evaluation metrics for synthetic (generative) EHR data, covering privacy,
utility, and statistical fidelity.

.. currentmodule:: pyhealth.metrics.generative

.. autofunction:: evaluate_synthetic_ehr

Privacy metrics
-------------------------------------

.. autofunction:: calc_nnaar

.. autofunction:: calc_membership_inference

.. autofunction:: compute_discriminator_privacy

Utility and fidelity metrics
-------------------------------------

.. autofunction:: compute_mle

.. autofunction:: compute_prevalence_metrics

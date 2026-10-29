Survival-curve metrics
======================

These metrics evaluate predicted survival probabilities against right-censored
outcomes. They complement metrics that accept one risk score per patient.
Inputs are CPU arrays or callbacks returning patient-ordered NumPy arrays.

Concordance and integrated binomial log-likelihood are higher-is-better; the
integrated Brier score is lower-is-better. Curve probabilities must be finite,
within [0, 1], and non-increasing over an increasing time grid. D-calibration
uses exact survival probabilities at observed event/censoring times.

The general API aligns prediction and scoring grids. The benchmark's separately
named reference-compatible report documents the paper evaluator's differing
widest-horizon convention. It does not alter the general metric functions.

.. automodule:: pyhealth.metrics.survival_curve
    :members:

See :doc:`../../survival_pipeline` for the shared benchmark and saved outputs.

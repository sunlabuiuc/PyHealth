Survival-curve metrics
======================

These metrics evaluate predicted survival probabilities against right-censored
outcomes. They complement :mod:`pyhealth.metrics.survival`, which provides
Harrell's concordance index and Brier/AUC scores at individual horizons.
This module instead provides IPCW time-dependent concordance, integrated
Brier/log-likelihood scores and D-calibration, following the Survival MDN
evaluator's curve comparisons and linearly interpolated censoring estimate.
The two APIs use different weighting and tie conventions and are not
interchangeable for paper comparisons.
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

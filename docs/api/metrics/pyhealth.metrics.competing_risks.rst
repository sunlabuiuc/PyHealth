Competing-risk metrics
======================

.. automodule:: pyhealth.metrics.competing_risks
    :members:

Fit censoring weights on training patients. The primary Brier score treats an
observed competing first event as a known negative for the target first event,
not ordinary censoring. Concordance is horizon-specific and includes earlier
competing events, with its pair weights documented in the function.
For a target event at ``t``, strictly later comparators require censoring
survival through ``t`` and use ``1 / (G(t-) G(t))``. An earlier competing
event at ``s`` instead uses ``1 / (G(t-) G(s-))``.

IPCW requires independent censoring and supported follow-up. Metrics with
exhausted support or no eligible pairs raise ``MetricSupportError``. Malformed
inputs instead raise ``ValueError``. Report unsupported Brier/concordance
independently; do not substitute zero or remove their rows.

The optional example's binary-cause scikit-survival lane is a historical
diagnostic with a different estimand, not a replacement for these metrics.

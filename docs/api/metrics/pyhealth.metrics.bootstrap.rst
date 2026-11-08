pyhealth.metrics.bootstrap
===================================

Confidence intervals for evaluation metrics, as clinical reporting guidelines
such as TRIPOD+AI expect. Samples from one patient are correlated, so pass
``groups=patient_ids`` to resample whole patients (a cluster bootstrap) rather
than individual samples. To compare two models, use ``paired_bootstrap_diff``:
both are scored on identical resamples, so its interval is for their difference.

.. code-block:: python

    from pyhealth.metrics import bootstrap_ci, paired_bootstrap_diff

    ci = bootstrap_ci(y_true, y_prob, "pr_auc", groups=patient_ids, n_boot=1000)
    # {'estimate': 0.41, 'lower': 0.35, 'upper': 0.47, 'n_boot': 1000, 'n_skipped': 0}

    diff = paired_bootstrap_diff(y_true, y_prob_new, y_prob_baseline, "pr_auc",
                                 groups=patient_ids)
    # an interval that excludes 0 indicates a difference at the 5% level

``metric`` is any name accepted by ``binary_metrics_fn``, including the
calibration metrics ``brier``, ``oe_ratio``, ``calibration_slope`` and
``calibration_intercept``, or a callable ``metric(y_true, y_prob) -> float``.
Resamples that contain a single class are skipped and counted in ``n_skipped``.
See ``examples/calibration_and_bootstrap_ci.py``.

.. currentmodule:: pyhealth.metrics.bootstrap

.. autofunction:: bootstrap_ci

.. autofunction:: paired_bootstrap_diff

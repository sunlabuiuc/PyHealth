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

``bootstrap_metrics_ci`` covers multiclass and multilabel metrics: it
reproduces the bootstrap protocol of the ECG benchmark's released code
(Nonaka & Seita, 2021) — 100 resamples, bounds at the 5% / 95%
percentiles, and rejection of resamples that do not define every class (a
multilabel class without a positive, or a multiclass class without a
member). That is not how the paper decides whether one setting beats
another: for that, compare per-split scores from repeated runs with
:func:`pyhealth.metrics.significance.welch_relative_improvement_test`.
See
``examples/ecg/ptbxl/ptbxl_multilabel_classification_resnet1d.py``.

.. currentmodule:: pyhealth.metrics.bootstrap

.. autofunction:: bootstrap_ci

.. autofunction:: bootstrap_metrics_ci

.. autofunction:: paired_bootstrap_diff

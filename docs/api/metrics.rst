Metrics
===============

We provide easy to use metrics (the same style and args as `sklearn.metrics`) for binary classification,
multiclass classification, multilabel classification, and segmentation (Dice/IoU).
For applicable tasks, we provide the relevant metrics for model calibration, as well as those for prediction set evaluation.
Among these we also provide metrics related to uncertainty quantification, for model calibration, as well as metrics that measure the quality of prediction sets
We also provide other metrics specically for healthcare
tasks, such as drug drug interaction (DDI) rate.
For synthetic (generative) EHR data, we provide privacy, utility, and statistical
fidelity metrics.

Expected Calibration Error (ECE) is available for binary and multiclass
predictions through :func:`pyhealth.metrics.calibration.ece_confidence_binary`
and :func:`pyhealth.metrics.calibration.ece_confidence_multiclass`. The API
reference includes the ECE definition and formula used by these helpers.

Confidence intervals for one run's test set come from
:mod:`pyhealth.metrics.bootstrap`. To compare settings that were each trained
once per data split, use
:func:`pyhealth.metrics.significance.welch_relative_improvement_test`:
Welch's t-test against a baseline with Benjamini-Hochberg correction, the
protocol of Nonaka & Seita (2021).


.. toctree::
    :maxdepth: 3

    metrics/pyhealth.metrics.multiclass
    metrics/pyhealth.metrics.multilabel
    metrics/pyhealth.metrics.ranking
    metrics/pyhealth.metrics.binary
    metrics/pyhealth.metrics.segmentation
    metrics/pyhealth.metrics.calibration
    metrics/pyhealth.metrics.bootstrap
    metrics/pyhealth.metrics.significance
    metrics/pyhealth.metrics.clinical
    metrics/pyhealth.metrics.competing_risks
    metrics/pyhealth.metrics.prediction_set
    metrics/pyhealth.metrics.fairness
    metrics/pyhealth.metrics.interpretability
    metrics/pyhealth.metrics.generative
    metrics/pyhealth.metrics.survival
    metrics/pyhealth.metrics.survival_curve
